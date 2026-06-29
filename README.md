# RWKV Synthetic Data Pipeline

This repo is split into two implementations:

- `python/`: experimental logic for fast iteration, real generation trials, and
  validation design.
- `rust/`: formal high-concurrency pipeline for larger synthesis/evaluation
  runs after the logic is stable.

The pipeline generates high-quality variants from wrong or failed evaluation
samples.

## Data Authority

The database is the authoritative source for failed samples, completions,
evaluation status, and synthesis progress. JSONL files are snapshots extracted
from the database for reproducible experiments, local debugging, and handoff
between the Python prototype and the Rust pipeline.

Recommended contract:

- DB stores every evaluated completion and its status.
- DB-derived failed samples are exported as JSONL snapshots when needed.
- Generated variants and rejection reasons are written as JSONL during Python
  experiments, then imported back into DB or consumed by the Rust runner.
- Codex automation should inspect DB progress first, not infer progress from
  local JSON files alone.

## Five Domains

Only these top-level domains are routed by config:

- `coding`
- `knowledge`
- `math`
- `instruction_following`
- `function_calling`

Subtypes such as GitHub issue repair, translation, single choice, multiple
choice, and JSON tool calls belong inside the corresponding domain prompt and
metadata. They should not become separate task kinds.

## Simplified Logic

The shared synthesis logic is intentionally small:

1. Route each failed sample to one of the five domain TOML profiles.
2. Build a subspace plan from that profile's axes, such as concept, edge case,
   API contract, or output format.
3. Generate variants with structured metadata:
   `semantic_plan`, `validation_contract`, `changed_factor`, and
   `diversity_signature`.
4. Validate and select non-duplicate variants before writing JSONL output.

This keeps paper-inspired ideas in implementation and config, not as research
notes inside prompts.

## Python Prototype

Run from the repo root:

```bash
export DEEPSEEK_V4_PRO_API_KEY=...
rtk env PYTHONPATH=python python3 -m synth --config python/mode.example.toml
```

The Python prototype writes:

```text
data/python_synth/generate/tasks.jsonl
data/python_synth/generate/rejected.jsonl
```

It is meant for quick logic tests and can make real OpenAI-compatible API calls.
It is intentionally not the high-concurrency implementation.

Input is DB-first. Configure a snapshot path and, optionally, a command that
extracts failed samples from DB:

```toml
[input.db_snapshot]
path = "../data/db_failed_samples.snapshot.jsonl"
# command = ["psql", "-X", "-A", "-t", "-f", "sql/export_failed_samples.sql"]
```

`rejected.jsonl` is intentionally kept during experiments. It records structural
validation failures, duplicate signatures, and model-validator rejections so the
next prompt/config edit is based on evidence.

## Rust Pipeline

Run from the repo root:

```bash
export DEEPSEEK_V4_PRO_API_KEY=...
rtk cargo run --manifest-path rust/Cargo.toml --release -- synthesize --config rust/mode.example.toml
```

Rust outputs are written under `[output].run_dir`, for example:

```text
rust/data/rwkv_train/generate/tasks.jsonl
rust/data/rwkv_train/done/success.jsonl
rust/data/rwkv_train/done/failed.jsonl
```

## Prompt Routing

Both implementations use domain TOML profiles:

```toml
[prompt]
profile_path = "prompts/knowledge.toml"
selector_keys = ["profile", "task_kind", "domain", "subject", "dataset", "source"]

[prompt.profiles]
coding = "prompts/coding.toml"
knowledge = "prompts/knowledge.toml"
math = "prompts/math.toml"
instruction_following = "prompts/instruction_following.toml"
function_calling = "prompts/function_calling.toml"
```

Input metadata should use one of the five domain names. If a sample needs finer
typing, put it in metadata such as `subject`, `dataset`, `source`, `skill_node`,
or domain-specific axes.

## Export Training Data

Export helpers live under `python/scripts/`:

```bash
rtk python3 python/scripts/export_rwkv_train.py --run-dir rust/data/rwkv_train
```

Use `--assistant-style no-think` if the target training set should omit
`<think>...</think>` reasoning blocks.
