# OMP Laya

Local decision-policy extension for [Oh My Pi](https://omp.sh). It uses a local Laya model to constrain unnecessary discovery, protect risky operations, prune superseded evidence, and require post-mutation verification.

## Install

```bash
git clone https://github.com/KiidxAtlas/omp-laya ~/.omp/agent/plugins/omp-laya
cd ~/.omp/agent/plugins/omp-laya
bun install
python3 -m venv ~/.omp/agent/laya-venv
uv pip install --python ~/.omp/agent/laya-venv/bin/python --upgrade -r ~/.omp/agent/plugins/omp-laya/requirements.txt
omp plugin link ~/.omp/agent/plugins/omp-laya
```

`omp plugin link` symlinks the clone into `~/.omp/plugins/node_modules`, where OMP loads the `omp.extensions` entry from `package.json`. `~/.omp/agent/plugins` itself is not an extension discovery root. Alternatively, `omp plugin install github:KiidxAtlas/omp-laya` installs the published copy; use one method, not both.

## Local service

The extension launches the model server (`laya_server.py`, uvicorn on `127.0.0.1:8001`) on first use and logs to `laya-server.log`. The server outlives OMP sessions, so concurrent sessions share one process of about 3 GB, mostly GPU memory on Apple Silicon.

- After 15 minutes without an API request the server exits to free that memory; `/health` checks don't count. The next Laya request relaunches it, which takes a few seconds. Set `LAYA_IDLE_TIMEOUT_SECONDS` in OMP's environment to change the timeout; `0` keeps the server running.
- The server claims its port before loading the checkpoint and answers `/health` with `"starting"` meanwhile, so a session that finds it loading waits instead of launching a second copy.
- A request that finds the server gone relaunches it and retries. If the request's deadline is too short to wait, advisories resume on a later turn.

## Policy

```yaml
laya:
  enabled: true
  policyMode: enforce
```

### Routing, budgets, and metrics

`laya.profile` defaults to `balanced`. Model routing produces **advisory recommendations**, not model switches: the checked OMP 18.3.0 extension runner does not consume model or effort overrides from `before_agent_start`. Laya preserves the active model and effort and never writes model-role settings.

- `savings` may recommend `@smol` for trivial/easy non-sensitive work and `@slow` for hard work.
- `balanced` keeps the current model rather than recommending a downgrade.
- `safety-first` may recommend `@slow` for hard or sensitive work, never `@smol`.

Effort recommendations respect advertised model support. For an unchanged model they never recommend lowering the current level; an unknown current level is left alone.

`laya.contextBudgetEnabled` defaults to `true`. At 70% observed context pressure, discovery / same-selector reread / total synthesis-read thresholds are `1 / 1 / 12` for Savings, `2 / 2 / 20` for Balanced, and `3 / 3 / 32` for Safety-first. Different ranges of the same file are new evidence, not rereads. Synthesis holds are one-shot: a needed read can be reissued. Explicitly named user paths remain exempt. Without observed pressure, the profile thresholds do not activate.

Pattern searches (`grep`) are not treated as broad discovery. A targeted retrieval plan does not block file discovery needed to locate its target.

Existing switches such as `laya.economyEnabled`, `laya.synthesisGuardEnabled`, and `laya.contextPruningEnabled` remain independent controls. `/settings` exposes the profile, model-routing, and context-budget controls.

Context pruning considers only exact duplicate all-text results from the same tool, at least 2,000 characters long. It retains a pointer in place of an older copy, preserving tool-call/result pairing. Nonidentical reads are retained: neither a broader requested range nor a similar-looking body proves the earlier result is redundant. UI-only `Laya` transcript cards are excluded from model context even when tool-output pruning is disabled; verification continuation messages remain. Skill and path hints are appended as hidden context messages rather than replacing the system prompt.

Session entries record provider-reported input/output and cache tokens separately, reported context usage, inference latency, and blocked/pruned operation counts. Pruned-token counts are estimates, not measured counterfactual savings. Routing metrics describe recommendations, not applied switches. Use `/laya stats` for observed metrics. `/laya keep-model` bypasses the next routing recommendation without disabling other safeguards.

### Completion policy

- `advisory` records and displays verification still required after a behavior-affecting mutation.
- `enforce` schedules one follow-up agent turn to run the narrowest relevant verification command.

The extension records its decisions as `laya-activity`, `laya-verification`, and `laya-failure` session entries. Laya service failures fail open: normal OMP work continues, while the session trace retains the error and stack.

Clean source edits require verification without an additional classifier request. Classifier-only security suspicions do not block ordinary changes without a deterministic security prefilter hit. Repeated edits still advance the verification ledger, but add a model-facing verification reminder only when an obligation first becomes pending.

Automatic verification recognition is deliberately limited to successful direct commands. Quoted commands, pipelines, shell composition, redirects, substitutions, and unrecognized smoke commands do not automatically satisfy the ledger: a shell exit code alone can mask failed checks. Recognized completion is not certification that checks passed; inspect the output. Verification evidence for the current mutation survives later model iterations. Missing recognition is reported as incomplete evidence, not proof that no verification happened.

## OMP judge integration

The extension registers `laya-systemone/laya` as a custom OMP model. With `modelRoles.judge: laya-systemone/laya`, OMP's `@judge` requests are mapped to typed `choice` and `noul` questions and sent to `POST /v1/systemone`, the native Jev wire contract. `laya/laya` remains the separate OpenAI-compatible chat adapter. The typed path returns JSON score content to OMP; it does not generate prose or tool calls.

The extension requires `laya>=0.3.20,<0.4` and loads `convaiinnovations/laya-typed-decisions` by its Hugging Face model ID. A new server start resolves the checkpoint's current default Hub revision. On the public typed-decisions test split, this installation scored **0.767 accuracy over 2,000 decisions**, versus the published **0.727** for TypeSafe Jev 1.13.0. That measures native typed predictions, not the chat adapter or your private judge prompts.

To list the separate OpenAI chat adapter as a regular model:

```yaml
providers:
  laya:
    baseUrl: http://127.0.0.1:8001/v1
    api: openai-completions
    auth: none
    models:
      - id: laya
        name: Laya (local predictor)
        input: [text]
```


## Development

```bash
bun test
~/.omp/agent/laya-venv/bin/python -m unittest laya_server_test
```

`LAYA_PYTHON` overrides the default `~/.omp/agent/laya-venv/bin/python`; `PI_CODING_AGENT_DIR` overrides the default agent directory.
