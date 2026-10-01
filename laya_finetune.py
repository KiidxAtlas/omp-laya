"""Fine-tune a Laya decision checkpoint on decisions exported by the omp-laya extension.

Single-device port of the upstream Laya fine-tuning notebook: the same item encoding, RLCD
loss (noisy-logit policy gradient on a proper scoring reward plus soft cross-entropy),
hyperparameters, per-type temperature fit and checkpoint layout, without DDP.

    python laya_finetune.py --data TRAIN.jsonl --out DIR [--base convaiinnovations/laya-typed-decisions]
        [--epochs 4] [--device auto|mps|cuda|cpu] [--min-examples 50] [--force] [--max-items N]

Each JSONL row is one state with its questions and gold answers:

    {"state": ..., "questions": {qid: {"type", "instructions", "criteria"?}},
     "gold": {qid: {"probabilities": {...}}}}

noul gold is keyed "true"/"false", choice gold by option label, score gold by level index
("0", "1", ...). Questions without gold are ignored. The output directory loads with
`laya.load(DIR)` and can be served by setting LAYA_CHECKPOINT=DIR for laya_server.py.
Progress goes to stderr; a JSON summary goes to stdout.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Any

import torch
from laya.agent import Agent, _fix_tokenizer_config
from laya.common import QTYPE_NAMES, QTYPES, build_sequence, proper_reward, render_options

DEFAULT_BASE = "convaiinnovations/laya-typed-decisions"
# What laya.load itself downloads; a full snapshot would also pull unrelated repo files.
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")

# Notebook hyperparameters. Micro-batch and accumulation are per device; the notebook ran
# two GPUs, so one device sees half its effective batch.
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START = 0.4
SIGMA_END = 0.1
EVAL_BATCH = 16
CALIB_SEED = 20260922
CALIB_MAX = 400
# Temperature for a question type with no held-out items to fit on (the notebook's prior).
UNFITTED_TEMPERATURE = 1.2

EXIT_REFUSED = 2


class FinetuneError(Exception):
    """The run cannot proceed as requested; the message says why and what to change."""


def gold_target(q: dict, gold_q: Any) -> list[float]:
    """Gold distribution in the model's option order for internal question `q`."""
    probs = gold_q.get("probabilities") if isinstance(gold_q, dict) else None
    if not isinstance(probs, dict):
        raise ValueError("gold needs a 'probabilities' object")
    t, crit = q["t"], q["crit"]
    if t == "choice":
        keys = [str(k) for k in crit]
    elif t == "noul":
        keys = ["false", "true"]
    else:
        keys = [str(i) for i in range(len(crit))]
    unknown = sorted(set(probs) - set(keys))
    if unknown:
        raise ValueError("gold probabilities name %s, which are not options of this question %s" % (unknown, keys))
    for key, value in probs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("gold probability for %r must be a non-negative number, got %r" % (key, value))
    if t == "noul" and len(probs) == 1:
        # One side of a yes/no answer determines the other.
        (key, value), = probs.items()
        if value > 1:
            raise ValueError("gold probability for %r must be at most 1, got %r" % (key, value))
        probs = {key: value, ("true" if key == "false" else "false"): 1.0 - value}
    target = [float(probs.get(k, 0.0)) for k in keys]
    total = sum(target)
    if total <= 0:
        raise ValueError("gold probabilities carry no mass")
    return [v / total for v in target]


def parse_row(row: Any) -> list[dict]:
    """Decision examples of one JSONL row: one per question that has gold."""
    if not isinstance(row, dict):
        raise ValueError("row must be a JSON object")
    missing = [key for key in ("state", "questions", "gold") if key not in row]
    if missing:
        raise ValueError("row is missing %s" % missing)
    questions, gold = row["questions"], row["gold"]
    if not isinstance(questions, dict) or not isinstance(gold, dict):
        raise ValueError("'questions' and 'gold' must be objects keyed by question id")
    examples = []
    for qid, qdef in questions.items():
        if qid not in gold:
            continue
        # The same validation and normalization predict() applies, so training sees the
        # questions exactly as inference will.
        Agent._check_question(qid, qdef)
        q = Agent._to_internal(qdef)
        try:
            target = gold_target(q, gold[qid])
        except ValueError as error:
            raise ValueError("question %r: %s" % (qid, error)) from error
        examples.append({"state": row["state"], "q": q, "target": target, "label": target.index(max(target))})
    return examples


def load_examples(path: str) -> list[dict]:
    """Read and validate a training JSONL file. Any malformed row fails the whole file."""
    examples = []
    try:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    examples.extend(parse_row(json.loads(line)))
                except json.JSONDecodeError as error:
                    raise FinetuneError("%s:%d: not valid JSON (%s)" % (path, line_no, error.msg)) from error
                except ValueError as error:
                    raise FinetuneError("%s:%d: %s" % (path, line_no, error)) from error
    except OSError as error:
        raise FinetuneError("cannot read training data %s: %s" % (path, error.strerror or error)) from error
    return examples


def check_enough(n_items: int, min_examples: int, force: bool) -> None:
    if n_items == 0:
        raise FinetuneError("no decision items with gold answers to train on")
    if n_items < min_examples and not force:
        raise FinetuneError(
            "refusing to fine-tune on %d decision items (fewer than --min-examples %d): a model "
            "fine-tuned on this little data usually gets worse. Collect more decisions, or pass "
            "--force to train anyway." % (n_items, min_examples)
        )


def build_items(examples: list[dict], tok, max_len: int, head_max_len: int) -> tuple[list[dict], int]:
    """Tokenize examples into training items; returns (items, skipped).

    Items whose options do not all fit the head budget are skipped, as in the notebook.
    Conversation lists keep their newest turns when truncated, matching inference.
    """
    items, skipped = [], 0
    for ex in examples:
        q = ex["q"]
        seq, markers = build_sequence(tok, ex["state"], q, max_len, head_max_len,
                                      truncate_left=isinstance(ex["state"], list))
        if len(markers) != len(render_options(q)):
            skipped += 1
            continue
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                      "target": ex["target"], "label": ex["label"]})
    return items, skipped


def split_calibration(items: list, seed: int = CALIB_SEED) -> tuple[list, list]:
    """(train, calibration): a fixed-seed ~10% slice withheld from training.

    Temperatures fitted on items the run trained on measure the fit, not the calibration.
    """
    n = len(items)
    n_calib = min(CALIB_MAX, max(1, n // 10)) if n >= 2 else 0
    order = list(range(n))
    random.Random(seed).shuffle(order)
    held = set(order[:n_calib])
    return [it for i, it in enumerate(items) if i not in held], [it for i, it in enumerate(items) if i in held]


def collate_train_batch(items: list[dict], pad_id: int) -> dict:
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel: list) -> float:
    """Temperature minimizing held-out cross-entropy; 1.0 when there is too little to fit."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    z_all = torch.full((len(sel), kmax), -1e4)
    t_all = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        z_all[i, : len(z)] = torch.tensor(z)
        t_all[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(t_all * torch.log_softmax(z_all / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def fit_temperatures(preds: list[tuple]) -> list[float]:
    """Per-type temperatures [choice, score, noul] from (qtype, logits, target, label) rows."""
    temps = [UNFITTED_TEMPERATURE] * 3
    try:
        for qt in range(3):
            sel = [(z, t) for q_type, z, t, _ in preds if q_type == qt]
            if sel:
                temps[qt] = fit_one_temp(sel)
    except Exception as error:
        log("temperature fitting fell back to defaults: %s" % error)
    return temps


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_device(name: str) -> torch.device:
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if mps else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise FinetuneError("--device cuda requested but CUDA is not available")
    if name == "mps" and not mps:
        raise FinetuneError("--device mps requested but MPS is not available")
    return torch.device(name)


def resolve_base(base: str) -> str:
    """Local checkpoint directory for `base`, downloading a Hugging Face id if needed."""
    if os.path.isdir(base):
        model_dir = base
    elif base.startswith(("/", "./", "../", "~")):
        raise FinetuneError("base checkpoint directory not found: %s" % base)
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(base, allow_patterns=list(CHECKPOINT_FILES),
                                      token=os.environ.get("HF_TOKEN") or None)
    for name in ("rl_agent_config.json", "model.safetensors", "tokenizer"):
        if not os.path.exists(os.path.join(model_dir, name)):
            raise FinetuneError("%s is not a Laya checkpoint: it has no %s" % (base, name))
    _fix_tokenizer_config(model_dir)
    return model_dir


def load_model(model_dir: str, cfg: dict, device: torch.device):
    from laya.common import build_model
    from safetensors.torch import load_file

    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights

    enc_dir = os.path.join(model_dir, "encoder")
    # The checkpoint supplies every parameter; skip random initialization.
    with no_init_weights():
        model = build_model(cfg, encoder_dir=enc_dir if os.path.isdir(enc_dir) else None, pretrained=False)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    # Train in fp32 everywhere; CUDA adds fp16 autocast on top.
    model.float()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    return model.to(device)


def autocast_for(device: torch.device):
    if device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    return nullcontext


def predict_logits(model, items: list[dict], pad_id: int, device: torch.device) -> list[tuple]:
    """(qtype, logits over the item's options, target, gold label) per item."""
    autocast = autocast_for(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(items), EVAL_BATCH):
            chunk = items[start:start + EVAL_BATCH]
            b = collate_train_batch(chunk, pad_id)
            with autocast():
                logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                  b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float().cpu().numpy()
            for row, it in enumerate(chunk):
                preds.append((it["qtype"], logits[row, : len(it["markers"])], it["target"], it["label"]))
    return preds


def argmax_accuracy(preds: list[tuple]) -> float | None:
    if not preds:
        return None
    return sum(int(z.argmax()) == label for _, z, _, label in preds) / len(preds)


def train(model, items: list[dict], pad_id: int, device: torch.device, epochs: int) -> dict:
    """The notebook's RLCD training loop on one device."""
    items = list(items)
    autocast = autocast_for(device)
    use_scaler = device.type == "cuda"
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}], weight_decay=0.01
    )
    micro_batches = math.ceil(len(items) / MICRO_BATCH)
    total_updates = math.ceil(micro_batches / GRAD_ACCUM) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    model.train()
    log("training on %d items for %d epochs (%d optimizer updates) on %s" % (len(items), epochs, total_updates, device))
    t0, avg_loss, updates = time.time(), float("nan"), 0
    for epoch in range(epochs):
        random.Random(42 + epoch).shuffle(items)
        epoch_loss, n_batches, accum_step = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
        for b_idx in range(0, len(items), MICRO_BATCH):
            batch = collate_train_batch(items[b_idx:b_idx + MICRO_BATCH], pad_id)
            mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            with autocast():
                logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                  batch["marker_pos"].to(device), mask, qtype)
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions with zero-mean projection.
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Proper scoring reward, standardized into group advantages.
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient plus soft cross-entropy guidance.
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or b_idx + MICRO_BATCH >= len(items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % 50 == 0:
                log("  epoch %d/%d | step %d | loss %.4f | reward %.3f | lr %.2e"
                    % (epoch + 1, epochs, n_batches, loss.item() * GRAD_ACCUM, r.mean().item(),
                       scheduler.get_last_lr()[0]))
        avg_loss = epoch_loss / max(1, n_batches)
        log("epoch %d/%d done in %.1fs | avg loss %.4f" % (epoch + 1, epochs, time.time() - t0, avg_loss))
    return {"updates": updates, "avg_loss": avg_loss, "seconds": round(time.time() - t0, 1)}


def save_checkpoint(model, tok, cfg: dict, out_dir: str, temperatures: list[float], training: dict) -> None:
    """Write the layout laya.load reads: weights, encoder config, tokenizer, agent config."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg = dict(cfg, fine_tuned=True, temperature=temperatures, training=training)
    # This fit is per type; inherited bucket overrides would hide the new values.
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Laya checkpoint on exported omp-laya decisions.")
    parser.add_argument("--data", required=True, help="training JSONL exported by the omp-laya extension")
    parser.add_argument("--out", required=True, help="directory to write the fine-tuned checkpoint to")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Hugging Face id or local checkpoint directory")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--min-examples", type=int, default=50,
                        help="refuse to train on fewer decision items than this unless --force")
    parser.add_argument("--force", action="store_true", help="train even below --min-examples")
    parser.add_argument("--max-items", type=int, default=None, help="use only the first N decision items")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        examples = load_examples(args.data)[: args.max_items]
        # Fail before downloading or loading anything when the data cannot support a run.
        check_enough(len(examples), args.min_examples, args.force)
        device = resolve_device(args.device)
        model_dir = resolve_base(args.base)
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            cfg = json.load(f)
        cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096, max_len=1024, head_max_len=256)

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        items, skipped = build_items(examples, tok, cfg["max_len"], cfg["head_max_len"])
        if skipped:
            log("skipped %d items whose options exceed head_max_len=%d" % (skipped, cfg["head_max_len"]))
        check_enough(len(items), args.min_examples, args.force)
    except FinetuneError as error:
        log("laya_finetune: %s" % error)
        return EXIT_REFUSED

    train_items, calib_items = split_calibration(items)
    log("%d decision items: %d train, %d held out for calibration" % (len(items), len(train_items), len(calib_items)))
    model = load_model(model_dir, cfg, device)
    before = argmax_accuracy(predict_logits(model, calib_items, tok.pad_token_id, device))
    stats = train(model, train_items, tok.pad_token_id, device, args.epochs)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    calib_preds = predict_logits(model, calib_items, tok.pad_token_id, device)
    after = argmax_accuracy(calib_preds)
    temperatures = fit_temperatures(calib_preds)
    save_checkpoint(model, tok, cfg, args.out, temperatures, {
        "updates": stats["updates"],
        "epochs_completed": args.epochs,
        "hours": round(stats["seconds"] / 3600, 3),
        "world_size": 1,
        "fine_tuned_from_checkpoint": True,
        "base": args.base,
        "n_train": len(train_items),
    })
    log("saved fine-tuned checkpoint to %s" % args.out)
    print(json.dumps({
        "out": args.out,
        "base": args.base,
        "n_items": len(items),
        "n_skipped": skipped,
        "n_train": len(train_items),
        "n_calib": len(calib_items),
        "epochs": args.epochs,
        "device": device.type,
        "train_seconds": stats["seconds"],
        "final_avg_loss": stats["avg_loss"],
        "temperatures": {QTYPE_NAMES[qt]: round(t, 4) for qt, t in enumerate(temperatures)},
        "heldout_accuracy": {"before": before, "after": after},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
