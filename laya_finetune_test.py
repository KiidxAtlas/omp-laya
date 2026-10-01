import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import laya_finetune
from laya_finetune import FinetuneError

NOUL = {"type": "noul", "instructions": "Does the command delete data?"}
CHOICE = {
    "type": "choice",
    "instructions": "What does the command do to the filesystem?",
    "criteria": {"read": "only reads", "write": "creates or edits files", "delete": "removes files"},
}


def row(state="rm -rf build", questions=None, gold=None):
    return {
        "state": state,
        "questions": questions if questions is not None else {"destructive": NOUL},
        "gold": gold if gold is not None else {"destructive": {"probabilities": {"true": 0.9, "false": 0.1}}},
    }


class FakeTokenizer:
    """Whitespace tokenizer with the special tokens build_sequence uses."""

    mask_token = "[MASK]"
    pad_token_id, mask_token_id, cls_token_id, sep_token_id = 0, 1, 2, 3

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + len(word) for word in text.split()]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write_jsonl(self, lines):
        path = os.path.join(self.dir.name, "train.jsonl")
        with open(path, "w") as f:
            f.write("\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n")
        return path

    def test_gold_targets_follow_the_model_option_order(self):
        path = self.write_jsonl([
            row(gold={"destructive": {"probabilities": {"true": 0.8, "false": 0.2}}}),
            row(questions={"effect": CHOICE}, gold={"effect": {"probabilities": {"delete": 0.7, "read": 0.3}}}),
            row(questions={"effect": dict(CHOICE, criteria=["read", "write", "delete"])},
                gold={"effect": {"probabilities": {"write": 1.0}}}),
            row(questions={"risk": {"type": "score", "instructions": "How risky?", "criteria": ["none", "some", "high"]}},
                gold={"risk": {"probabilities": {"2": 3, "1": 1}}}),
            row(gold={"destructive": {"probabilities": {"true": 0.25}}}),
        ])

        examples = laya_finetune.load_examples(path)

        self.assertEqual([ex["target"] for ex in examples], [
            [0.2, 0.8],
            [0.3, 0.0, 0.7],
            [0.0, 1.0, 0.0],
            [0.0, 0.25, 0.75],
            [0.75, 0.25],
        ])
        self.assertEqual([ex["label"] for ex in examples], [1, 2, 1, 2, 0])

    def test_questions_without_gold_are_not_training_items(self):
        path = self.write_jsonl([row(questions={"destructive": NOUL, "effect": CHOICE})])

        examples = laya_finetune.load_examples(path)

        self.assertEqual([ex["q"]["t"] for ex in examples], ["noul"])

    def test_malformed_rows_fail_with_their_line_number(self):
        cases = {
            "invalid json": "{not json",
            "missing gold": {"state": "ls", "questions": {"destructive": NOUL}},
            "unknown option": row(questions={"effect": CHOICE}, gold={"effect": {"probabilities": {"format": 1.0}}}),
            "choice without criteria": row(questions={"effect": {"type": "choice", "instructions": "?"}},
                                           gold={"effect": {"probabilities": {"a": 1.0}}}),
            "negative probability": row(gold={"destructive": {"probabilities": {"true": 1.2, "false": -0.2}}}),
            "no mass": row(gold={"destructive": {"probabilities": {"true": 0, "false": 0}}}),
        }
        for name, bad in cases.items():
            with self.subTest(name):
                path = self.write_jsonl([row(), bad])
                with self.assertRaises(FinetuneError) as caught:
                    laya_finetune.load_examples(path)
                self.assertIn("train.jsonl:2:", str(caught.exception))

    def test_items_whose_options_do_not_fit_are_skipped(self):
        examples = laya_finetune.load_examples(self.write_jsonl([row(), row(questions={"effect": CHOICE},
                                                                           gold={"effect": {"probabilities": {"read": 1}}})]))

        items, skipped = laya_finetune.build_items(examples, FakeTokenizer(), max_len=1024, head_max_len=256)
        self.assertEqual(skipped, 0)
        self.assertEqual([len(it["markers"]) for it in items], [2, 3])
        self.assertEqual([it["label"] for it in items], [1, 0])

        items, skipped = laya_finetune.build_items(examples, FakeTokenizer(), max_len=8, head_max_len=256)
        self.assertEqual((items, skipped), ([], 2))


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.data = os.path.join(self.dir.name, "train.jsonl")
        with open(self.data, "w") as f:
            f.writelines(json.dumps(row()) + "\n" for _ in range(3))
        self.out = os.path.join(self.dir.name, "out")

    def run_main(self, *extra):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = laya_finetune.main(["--data", self.data, "--out", self.out, "--device", "cpu", *extra])
        return code, stderr.getvalue()

    def test_refuses_too_few_examples_before_loading_anything(self):
        with patch.object(laya_finetune, "resolve_base") as resolve_base:
            code, stderr = self.run_main()

        self.assertEqual(code, 2)
        self.assertIn("3 decision items", stderr)
        self.assertIn("--force", stderr)
        resolve_base.assert_not_called()
        self.assertFalse(os.path.exists(self.out))

    def test_force_and_lower_threshold_pass_the_gate(self):
        for extra in (["--force"], ["--min-examples", "3"]):
            with self.subTest(extra=extra), \
                    patch.object(laya_finetune, "resolve_base", side_effect=FinetuneError("stop here")) as resolve_base:
                code, stderr = self.run_main(*extra)
                resolve_base.assert_called_once_with(laya_finetune.DEFAULT_BASE)
                self.assertIn("stop here", stderr)
                self.assertEqual(code, 2)

    def test_max_items_counts_toward_the_threshold(self):
        with patch.object(laya_finetune, "resolve_base") as resolve_base:
            code, stderr = self.run_main("--min-examples", "3", "--max-items", "2")

        self.assertEqual(code, 2)
        self.assertIn("2 decision items", stderr)
        resolve_base.assert_not_called()

    def test_missing_data_file_is_a_clear_refusal(self):
        os.remove(self.data)

        code, stderr = self.run_main("--force")

        self.assertEqual(code, 2)
        self.assertIn("cannot read training data", stderr)


class CalibrationTests(unittest.TestCase):
    def test_calibration_slice_is_held_out_and_fixed(self):
        items = list(range(25))

        train, calib = laya_finetune.split_calibration(items)

        self.assertEqual(len(calib), 2)
        self.assertEqual(sorted(train + calib), items)
        self.assertEqual(laya_finetune.split_calibration(items), (train, calib))
        self.assertEqual([len(laya_finetune.split_calibration(list(range(n)))[1]) for n in (1, 2, 9)], [0, 1, 1])

    def test_temperature_fit_recovers_overconfident_logits(self):
        generator = torch.Generator().manual_seed(0)
        true_logits = torch.randn(200, 3, generator=generator)
        targets = torch.softmax(true_logits, -1)
        sel = [(z.tolist(), t.tolist()) for z, t in zip(true_logits * 2.5, targets)]

        self.assertAlmostEqual(laya_finetune.fit_one_temp(sel), 2.5, places=2)
        self.assertEqual(laya_finetune.fit_one_temp(sel[:9]), 1.0)


if __name__ == "__main__":
    unittest.main()
