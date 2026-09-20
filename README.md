# OMP Laya

Local decision-policy extension for [Oh My Pi](https://omp.sh). It uses a local Laya model to constrain unnecessary discovery, protect risky operations, prune superseded evidence, and require post-mutation verification.

## Install

```bash
git clone https://github.com/KiidxAtlas/omp-laya ~/.omp/agent/extensions/omp-laya
python3 -m venv ~/.omp/agent/laya-venv
~/.omp/agent/laya-venv/bin/pip install -r ~/.omp/agent/extensions/omp-laya/requirements.txt
```

OMP discovers the plugin directory through its `package.json` manifest. To configure it explicitly instead, add this to `~/.omp/agent/config.yml`:

```yaml
extensions:
  - ~/.omp/agent/extensions/omp-laya
```

## Policy

```yaml
laya:
  enabled: true
  policyMode: enforce
```

- `advisory` records and displays verification still required after a behavior-affecting mutation.
- `enforce` schedules one follow-up agent turn to run the narrowest relevant verification command.

The extension records its decisions as `laya-activity`, `laya-verification`, and `laya-failure` session entries. Laya service failures fail open: normal OMP work continues, while the session trace retains the error and stack.

## Development

```bash
bun test
```

`LAYA_PYTHON` overrides the default `~/.omp/agent/laya-venv/bin/python`; `PI_CODING_AGENT_DIR` overrides the default agent directory.
