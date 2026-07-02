# DAPO-Math-17k SFT Synthesis

This folder contains the dedicated Python workflow for DAPO-Math-17k numeric
variant synthesis.

Default pipeline:

1. Normalize DAPO-Math-17k into local JSONL input.
2. Use `gpt-5.5` to generate 20 numeric variants per source problem.
3. Randomly route each variant through visible-CoT rollout attempts.
4. Immediately judge answer-matching rollouts with `gpt-5.5`.
5. Stop each variant after the first judged-valid rollout.
6. Export one judged-valid rollout per variant to chat SFT JSONL.

One-command full run:

```bash
uv run python/math_17k/dapo_math_17k_sft.py run-all \
  --config config.dapo_math_17k.toml
```

The command is resumable by default. Re-running the same command skips completed
generation batches, rollout rows, and judged rows.

The default rollout setting is 8 attempts per generated variant. With 20
variants per source problem, the target is 20 SFT rows; rejected or failed
attempts are retained, and later rollout indexes are added by re-running with a
larger `--rollouts-per-variant` value.

Rollout config currently uses OpenRouter models for stability. Keep `gpt-5.5`
generation and judge on next-token, and route rollout through OpenRouter.

If the normalized DAPO input JSONL is missing, the script downloads the dataset
and builds the 17k input snapshot automatically.

Output fields are intentionally small:

- `variants.jsonl`: `id`, `question`, `ref_answer`, `language`
- `rollouts.passed.jsonl`: variant fields plus `rollout_index`, `rollout_model`, `assistant`, `predicted_answer`, `answer_passed`
- `judged.jsonl` / `accepted.jsonl`: passed-rollout fields plus `judge_valid`
- `sft.jsonl`: `id`, `messages`; ids are sequential from `dapo_math_17k_0`, and the user message is the bare math problem only

The local `config.dapo_math_17k.toml` is ignored by Git and stores model/API
settings. Generated data under `data/` is also ignored by Git.
