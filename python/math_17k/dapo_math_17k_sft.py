#!/usr/bin/env python3
"""DAPO-Math-17k numeric-variant SFT pipeline.

This is a small, stdlib-only CLI for the DAPO-Math-17k workflow:

1. Download/normalize source problems.
2. Ask GPT-5.5 to create number-changed variants and reference answers.
3. Run a visible-CoT rollout model on the new questions.
4. Keep rollouts whose final answer matches the generated reference answer.
5. Ask GPT-5.5 to judge whether the reasoning chain is valid.
6. Export accepted rows as chat SFT JSONL.

Model/API settings are read from a TOML config by default. CLI args and
environment variables remain as optional overrides.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 17_000
DEFAULT_OUT_DIR = Path("data/experiments/dapo_math_17k")
DEFAULT_INPUT_JSONL = Path("data/inputs/dapo_math_17k/dapo_math_17k.input.jsonl")
DEFAULT_RAW_DIR = Path("data/inputs/dapo_math_17k/raw")
DEFAULT_LANGUAGES = ["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "ar"]

ANSWER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
WS_RE = re.compile(r"\s+")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        command_download(args)
    elif args.command == "inspect":
        command_inspect(args)
    elif args.command == "estimate":
        command_estimate(args)
    elif args.command == "generate":
        command_generate(args)
    elif args.command == "rollout":
        command_rollout(args)
    elif args.command == "judge":
        command_judge(args)
    elif args.command == "export-sft":
        command_export_sft(args)
    elif args.command == "run-all":
        command_run_all(args)
    elif args.command == "usage-summary":
        command_usage_summary(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download parquet and export normalized JSONL input.")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="0 means all rows reported by datasets-server.")
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_INPUT_JSONL)
    p.add_argument("--skip-parquet", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="Rewrite output instead of resuming from the current line count.")

    p = sub.add_parser("inspect", help="Inspect normalized input structure.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSONL)
    p.add_argument("--examples", type=int, default=3)

    p = sub.add_parser("estimate", help="Estimate tokens for one 20-variant generation request.")
    p.add_argument("--config", type=Path, default=Path("config.dapo_math_17k.toml"))
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSONL)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--variants-per-problem", type=int, default=20)
    p.add_argument("--expected-output-tokens-per-variant", type=int, default=260)

    p = sub.add_parser("generate", help="Generate numeric variants with GPT-5.5.")
    p.add_argument("--config", type=Path, default=Path("config.dapo_math_17k.toml"))
    add_model_args(p, "generator", default_model="gpt-5.5")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSONL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "variants.jsonl")
    p.add_argument("--rejected-output", type=Path, default=DEFAULT_OUT_DIR / "variants.rejected.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--variants-per-problem", type=int, default=20)
    p.add_argument("--generation-batch-size", type=int, default=2)
    p.add_argument("--languages", default="", help="Comma-separated target languages for generated variants.")
    p.add_argument("--json-response-format", action="store_true", help="Ask provider for JSON mode during generation.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=12_000)
    p.add_argument("--sleep", type=float, default=0.0)

    p = sub.add_parser("rollout", help="Run visible-CoT rollout model and filter by final answer.")
    p.add_argument("--config", type=Path, default=Path("config.dapo_math_17k.toml"))
    add_model_args(p, "rollout", default_model="")
    p.add_argument("--input", type=Path, default=DEFAULT_OUT_DIR / "variants.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "rollouts.passed.jsonl")
    p.add_argument("--failed-output", type=Path, default=DEFAULT_OUT_DIR / "rollouts.failed.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    p.add_argument("--rollouts-per-variant", type=int, default=8)
    p.add_argument("--rollout-policy", choices=["random", "all"], default="random")
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=4_096)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--no-resume", action="store_true", help="Do not skip completed rollout rows.")

    p = sub.add_parser("judge", help="Judge passed rollouts with GPT-5.5.")
    p.add_argument("--config", type=Path, default=Path("config.dapo_math_17k.toml"))
    add_model_args(p, "judge", default_model="gpt-5.5")
    p.add_argument("--input", type=Path, default=DEFAULT_OUT_DIR / "rollouts.passed.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "judged.jsonl")
    p.add_argument("--accepted-output", type=Path, default=DEFAULT_OUT_DIR / "accepted.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--no-resume", action="store_true", help="Do not skip completed judged rows.")

    p = sub.add_parser("export-sft", help="Export accepted rows as chat SFT JSONL.")
    p.add_argument("--input", type=Path, default=DEFAULT_OUT_DIR / "accepted.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "sft.jsonl")
    p.add_argument("--answer-prefix", default="")
    p.add_argument("--id-prefix", default="dapo_math_17k_")

    p = sub.add_parser("usage-summary", help="Summarize token usage from a JSONL output.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--usage-key", default="usage")
    p.add_argument("--request-id-key", default="id")

    p = sub.add_parser("run-all", help="One-command resumable pipeline: download, generate, rollout, judge, export SFT.")
    p.add_argument("--config", type=Path, default=Path("config.dapo_math_17k.toml"))
    add_model_args(p, "generator", default_model="gpt-5.5")
    add_model_args(p, "judge", default_model="gpt-5.5")
    add_model_args(p, "rollout", default_model="")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--source-limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Number of source problems to process; 0 means all rows.")
    p.add_argument("--variants-per-problem", type=int, default=20)
    p.add_argument("--generation-batch-size", type=int, default=2)
    p.add_argument("--languages", default="", help="Comma-separated target languages for generated variants.")
    p.add_argument("--rollouts-per-variant", type=int, default=8)
    p.add_argument("--rollout-policy", choices=["random", "all"], default="random")
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSONL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-parquet", action="store_true")

    return parser


def add_model_args(parser: argparse.ArgumentParser, prefix: str, default_model: str) -> None:
    env_prefix = prefix.upper()
    parser.add_argument(f"--{prefix}-url", default=os.getenv(f"{env_prefix}_URL") or os.getenv("OPENAI_CHAT_URL") or "")
    parser.add_argument(f"--{prefix}-base-url", default=os.getenv(f"{env_prefix}_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "")
    parser.add_argument(f"--{prefix}-api-key", default=os.getenv(f"{env_prefix}_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    parser.add_argument(f"--{prefix}-model", default=os.getenv(f"{env_prefix}_MODEL") or "")
    parser.set_defaults(**{f"{prefix}_default_model": default_model})


@dataclass
class ModelConfig:
    name: str
    url: str
    api_key: str
    model: str


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig, timeout: int = 180) -> None:
        if not cfg.api_key:
            raise SystemExit("missing API key")
        if not cfg.model:
            raise SystemExit("missing model name")
        if not cfg.url:
            raise SystemExit("missing chat completions URL or base URL")
        self.cfg = cfg
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_json: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        data = self._post_with_retry(body)
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning and not content:
            content = f"<think>\n{reasoning.strip()}\n</think>"
        elif reasoning and reasoning not in content:
            content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
        data["_content"] = content
        return data

    def _post_with_retry(self, body: bytes, retries: int = 4) -> dict[str, Any]:
        last_error: BaseException | None = None
        for attempt in range(retries):
            request = urllib.request.Request(
                self.cfg.url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "rwkv-dapo-math-sft",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not data.get("choices"):
                    raise RuntimeError(f"chat response missing choices: {json.dumps(data, ensure_ascii=False)[:1000]}")
                return data
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
                last_error = exc
            except (http.client.RemoteDisconnected, TimeoutError, urllib.error.URLError, OSError, RuntimeError) as exc:
                last_error = exc
            sleep_s = min(2**attempt, 15)
            print(
                f"retrying {self.cfg.name} chat after {type(last_error).__name__}: sleep={sleep_s}s",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
        assert last_error is not None
        raise last_error


def model_config_from_args(args: argparse.Namespace, prefix: str) -> ModelConfig:
    config = load_toml_config(getattr(args, "config", None))
    table = config.get(prefix, {}) if isinstance(config.get(prefix), dict) else {}
    url = getattr(args, f"{prefix}_url") or str(table.get("url") or "")
    base_url = getattr(args, f"{prefix}_base_url") or str(table.get("base_url") or "")
    if not url and base_url:
        url = base_url
    url = normalize_chat_url(url)
    api_key = getattr(args, f"{prefix}_api_key") or str(table.get("api_key") or "")
    model = (
        getattr(args, f"{prefix}_model")
        or str(table.get("model") or "")
        or getattr(args, f"{prefix}_default_model", "")
    )
    return ModelConfig(
        name=str(table.get("name") or model or prefix),
        url=url,
        api_key=api_key,
        model=model,
    )


def rollout_model_configs(args: argparse.Namespace) -> list[ModelConfig]:
    cli_cfg = model_config_from_args(args, "rollout")
    if args.rollout_model:
        return [cli_cfg]

    config = load_toml_config(getattr(args, "config", None))
    raw_models = config.get("rollout_models", [])
    if not isinstance(raw_models, list) or not raw_models:
        return [cli_cfg]

    out = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        url = normalize_chat_url(str(item.get("url") or item.get("base_url") or ""))
        out.append(
            ModelConfig(
                name=str(item.get("name") or item.get("model") or "rollout"),
                url=url,
                api_key=str(item.get("api_key") or ""),
                model=str(item.get("model") or ""),
            )
        )
    return out


def generation_languages(args: argparse.Namespace) -> list[str]:
    if getattr(args, "languages", ""):
        raw = str(args.languages)
        langs = [item.strip() for item in raw.split(",") if item.strip()]
        if langs:
            return langs
    config = load_toml_config(getattr(args, "config", None))
    generation = config.get("generation", {})
    if isinstance(generation, dict):
        raw_langs = generation.get("languages")
        if isinstance(raw_langs, list):
            langs = [str(item).strip() for item in raw_langs if str(item).strip()]
            if langs:
                return langs
    return DEFAULT_LANGUAGES


def choose_language(languages: list[str], source_id: str, variant_index: int) -> str:
    if not languages:
        return "en"
    return languages[stable_int(f"{source_id}:{variant_index}") % len(languages)]


def stable_int(text: str) -> int:
    value = 0
    for byte in text.encode("utf-8"):
        value = ((value * 131) + byte) & 0xFFFFFFFF
    return value


def load_toml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def normalize_chat_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def command_download(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    info = fetch_json(f"https://huggingface.co/api/datasets/{urllib.parse.quote(args.dataset_id, safe='/')}")
    siblings = [item.get("rfilename", "") for item in info.get("siblings", [])]
    parquet_files = [name for name in siblings if name.endswith(".parquet")]
    if parquet_files and not args.skip_parquet:
        for filename in parquet_files:
            target = args.raw_dir / Path(filename).name
            if not target.exists():
                url = f"https://huggingface.co/datasets/{args.dataset_id}/resolve/main/{filename}"
                download_file(url, target)

    total = fetch_total_rows(args.dataset_id, args.config, args.split)
    wanted = total if args.limit == 0 else min(args.limit, total)
    page_size = min(max(args.page_size, 1), 100)
    if page_size != args.page_size:
        print(
            f"clamped page-size from {args.page_size} to {page_size} for datasets-server",
            file=sys.stderr,
        )
    start_offset = 0
    mode = "w"
    if not args.no_resume and args.output.exists():
        start_offset = min(count_jsonl_lines(args.output), wanted)
        mode = "a"
        print(f"resuming from row offset {start_offset}", file=sys.stderr)

    count = start_offset
    with args.output.open(mode, encoding="utf-8") as fh:
        for offset in range(start_offset, wanted, page_size):
            length = min(page_size, wanted - offset)
            payload = fetch_rows(args.dataset_id, args.config, args.split, offset, length)
            for item in payload.get("rows", []):
                row = normalize_dapo_row(item["row"], item.get("row_idx", offset + count), args.dataset_id)
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            print(f"downloaded normalized rows: {count}/{wanted}", file=sys.stderr)

    print_json(
        {
            "dataset_id": args.dataset_id,
            "total_rows_reported": total,
            "normalized_rows": count,
            "output": str(args.output),
            "raw_dir": str(args.raw_dir),
            "parquet_files": parquet_files,
        }
    )


def ensure_dapo_input(input_path: Path) -> None:
    if input_path.exists():
        return
    print(
        f"missing DAPO input {input_path}; downloading {DEFAULT_DATASET_ID}",
        file=sys.stderr,
    )
    command_download(
        argparse.Namespace(
            dataset_id=DEFAULT_DATASET_ID,
            config=DEFAULT_CONFIG,
            split=DEFAULT_SPLIT,
            limit=DEFAULT_LIMIT,
            page_size=100,
            raw_dir=DEFAULT_RAW_DIR,
            output=input_path,
            skip_parquet=False,
            no_resume=False,
        )
    )


def command_inspect(args: argparse.Namespace) -> None:
    ensure_dapo_input(args.input)
    rows = list(read_jsonl(args.input))
    key_shapes = Counter(tuple(row.keys()) for row in rows)
    lengths = [len(row.get("question", "")) for row in rows]
    out = {
        "path": str(args.input),
        "rows": len(rows),
        "key_shapes": {str(key): value for key, value in key_shapes.items()},
        "question_length": quantiles(lengths),
        "examples": rows[: args.examples],
    }
    print_json(out)


def command_estimate(args: argparse.Namespace) -> None:
    ensure_dapo_input(args.input)
    sample = read_jsonl_row(args.input, args.index)
    messages = build_generation_messages(sample, args.variants_per_problem)
    prompt_text = json.dumps(messages, ensure_ascii=False)
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = args.variants_per_problem * args.expected_output_tokens_per_variant
    print_json(
        {
            "input": str(args.input),
            "index": args.index,
            "variants_per_problem": args.variants_per_problem,
            "estimated_prompt_tokens": input_tokens,
            "estimated_completion_tokens": output_tokens,
            "estimated_total_tokens": input_tokens + output_tokens,
            "method": "tiktoken if installed, otherwise ceil(chars/4)",
            "question_chars": len(sample["question"]),
            "prompt_chars": len(prompt_text),
        }
    )


def command_generate(args: argparse.Namespace) -> None:
    ensure_dapo_input(args.input)
    client = OpenAICompatClient(model_config_from_args(args, "generator"))
    languages = generation_languages(args)
    rows = slice_rows(read_jsonl(args.input), args.start, args.limit)
    done_ids = existing_ids(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for source in rows:
        source_id = source["id"]
        existing_for_source = {
            row_id
            for row_id in done_ids
            if row_id.startswith(f"{source_id}__v")
        }
        if len(existing_for_source) >= args.variants_per_problem:
            print(f"skip {source_id}: already has {len(existing_for_source)} variants", file=sys.stderr)
            continue
        all_new_rows = []
        try:
            next_index = 0
            while len(existing_for_source) + len(all_new_rows) < args.variants_per_problem:
                while f"{source_id}__v{next_index:02d}" in existing_for_source:
                    next_index += 1
                remaining = args.variants_per_problem - len(existing_for_source) - len(all_new_rows)
                batch_size = min(max(args.generation_batch_size, 1), remaining)
                accepted_questions = [row["question"] for row in all_new_rows]
                batch_languages = [
                    choose_language(languages, source_id, next_index + offset)
                    for offset in range(batch_size)
                ]
                messages = build_generation_messages(
                    source, batch_size, accepted_questions, batch_languages
                )
                response = client.chat(
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    response_json=args.json_response_format,
                )
                parsed = parse_json_object(response["_content"])
                variants = validate_variants(parsed, batch_size)
                batch_rows = []
                for item_offset, item in enumerate(variants):
                    while f"{source_id}__v{next_index:02d}" in existing_for_source:
                        next_index += 1
                    language = str(item.get("language") or batch_languages[item_offset]).strip()
                    row = {
                        "id": f"{source_id}__v{next_index:02d}",
                        "question": item["question"].strip(),
                        "ref_answer": item["answer"].strip(),
                        "language": language,
                    }
                    batch_rows.append(row)
                    all_new_rows.append(row)
                    next_index += 1
                append_jsonl(args.output, batch_rows)
                print(
                    f"generated batch {len(batch_rows)} variants for {source_id} "
                    f"({len(existing_for_source) + len(all_new_rows)}/{args.variants_per_problem})",
                    file=sys.stderr,
                )
                if args.sleep:
                    time.sleep(args.sleep)
        except Exception as exc:
            append_jsonl(
                args.rejected_output,
                [{"id": source_id, "error": type(exc).__name__}],
            )
            print(f"rejected {source_id}: {exc}", file=sys.stderr)
        if args.sleep:
            time.sleep(args.sleep)


def command_rollout(args: argparse.Namespace) -> None:
    model_cfgs = rollout_model_configs(args)
    client_cache: dict[str, OpenAICompatClient] = {}
    rows = slice_rows(read_jsonl(args.input), args.start, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.no_resume else completed_rollout_keys(args.output, args.failed_output)
    for variant in rows:
        for rollout_index in range(args.rollouts_per_variant):
            selected_cfgs = select_rollout_models(
                model_cfgs,
                variant["id"],
                rollout_index,
                args.rollout_policy,
                args.seed,
            )
            for cfg in selected_cfgs:
                key = progress_key(variant["id"], cfg.model, rollout_index)
                if key in completed:
                    print(f"skip rollout {variant['id']}#{rollout_index}@{cfg.name}: completed", file=sys.stderr)
                    continue
                messages = build_rollout_messages(variant["question"])
                try:
                    client = client_cache.setdefault(cfg.name, OpenAICompatClient(cfg))
                    response = client.chat(
                        messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    assistant = response["_content"].strip()
                    predicted = extract_final_answer(assistant)
                    passed = answers_match(predicted, variant["ref_answer"])
                    row = {
                        **variant,
                        "rollout_index": rollout_index,
                        "rollout_model": cfg.model,
                        "assistant": assistant,
                        "predicted_answer": predicted,
                        "answer_passed": passed,
                    }
                    append_jsonl(args.output if passed else args.failed_output, [row])
                    completed.add(key)
                    print(
                        f"rollout {variant['id']}#{rollout_index}@{cfg.name}: {'pass' if passed else 'fail'}",
                        file=sys.stderr,
                    )
                except Exception as exc:
                    append_jsonl(
                        args.failed_output,
                        [
                            {
                                **variant,
                                "rollout_index": rollout_index,
                                "rollout_model": cfg.model,
                                "error": type(exc).__name__,
                                "answer_passed": False,
                            }
                        ],
                    )
                    completed.add(key)
                    print(
                        f"rollout {variant['id']}#{rollout_index}@{cfg.name}: api_error {type(exc).__name__}",
                        file=sys.stderr,
                    )
        if args.sleep:
            time.sleep(args.sleep)


def command_judge(args: argparse.Namespace) -> None:
    client = OpenAICompatClient(model_config_from_args(args, "judge"))
    rows = slice_rows(read_jsonl(args.input), args.start, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.no_resume else completed_judge_keys(args.output, args.accepted_output)
    for row in rows:
        key = progress_key(row["id"], str(row.get("rollout_model") or ""), int(row.get("rollout_index", 0)))
        if key in completed:
            print(f"skip judge {row['id']}#{row.get('rollout_index', 0)}@{row.get('rollout_model')}: completed", file=sys.stderr)
            continue
        try:
            response = client.chat(
                build_judge_messages(row),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_json=True,
            )
            verdict = parse_json_object(response["_content"])
            valid = bool(verdict.get("valid"))
            judged = {
                **row,
                "judge_valid": valid,
            }
            append_jsonl(args.output, [judged])
            if valid:
                append_jsonl(args.accepted_output, [judged])
            completed.add(key)
            print(f"judge {row['id']}#{row.get('rollout_index', 0)}: {'accept' if valid else 'reject'}", file=sys.stderr)
        except Exception as exc:
            append_jsonl(
                args.output,
                [{**row, "judge_valid": False, "error": type(exc).__name__}],
            )
        if args.sleep:
            time.sleep(args.sleep)


def command_accept_rollouts(args: argparse.Namespace) -> None:
    rollout_cfgs = rollout_model_configs(args)
    rollout_clients: dict[str, OpenAICompatClient] = {}
    judge_client = OpenAICompatClient(model_config_from_args(args, "judge"))
    rows = slice_rows(read_jsonl(args.input), args.start, args.limit)

    for path in (
        args.rollout_output,
        args.rollout_failed_output,
        args.judged_output,
        args.accepted_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    passed_by_key = keyed_rows(args.rollout_output)
    failed_keys = set(keyed_rows(args.rollout_failed_output))
    judged_by_key = keyed_rows(args.judged_output)
    accepted_ids = accepted_variant_ids(args.accepted_output)

    for variant in rows:
        variant_id = str(variant["id"])
        if variant_id in accepted_ids:
            print(f"skip accept {variant_id}: already accepted", file=sys.stderr)
            continue

        accepted = False
        for rollout_index in range(args.rollouts_per_variant):
            selected_cfgs = select_rollout_models(
                rollout_cfgs,
                variant_id,
                rollout_index,
                args.rollout_policy,
                args.seed,
            )
            for cfg in selected_cfgs:
                key = progress_key(variant_id, cfg.model, rollout_index)
                judged = judged_by_key.get(key)
                if judged is not None:
                    if judged.get("judge_valid"):
                        append_jsonl(args.accepted_output, [judged])
                        accepted_ids.add(variant_id)
                        accepted = True
                        print(
                            f"accept {variant_id}#{rollout_index}@{cfg.name}: reused judged",
                            file=sys.stderr,
                        )
                        break
                    continue

                row = passed_by_key.get(key)
                if row is None:
                    if key in failed_keys:
                        continue
                    try:
                        rollout_client = rollout_clients.setdefault(cfg.name, OpenAICompatClient(cfg))
                        response = rollout_client.chat(
                            build_rollout_messages(variant["question"]),
                            temperature=args.rollout_temperature,
                            max_tokens=args.rollout_max_tokens,
                        )
                        assistant = response["_content"].strip()
                        predicted = extract_final_answer(assistant)
                        passed = answers_match(predicted, variant["ref_answer"])
                        row = {
                            **variant,
                            "rollout_index": rollout_index,
                            "rollout_model": cfg.model,
                            "assistant": assistant,
                            "predicted_answer": predicted,
                            "answer_passed": passed,
                        }
                        append_jsonl(args.rollout_output if passed else args.rollout_failed_output, [row])
                        if passed:
                            passed_by_key[key] = row
                        else:
                            failed_keys.add(key)
                            print(
                                f"rollout {variant_id}#{rollout_index}@{cfg.name}: fail",
                                file=sys.stderr,
                            )
                            continue
                    except Exception as exc:
                        failed = {
                            **variant,
                            "rollout_index": rollout_index,
                            "rollout_model": cfg.model,
                            "error": type(exc).__name__,
                            "answer_passed": False,
                        }
                        append_jsonl(args.rollout_failed_output, [failed])
                        failed_keys.add(key)
                        print(
                            f"rollout {variant_id}#{rollout_index}@{cfg.name}: api_error {type(exc).__name__}",
                            file=sys.stderr,
                        )
                        continue

                try:
                    response = judge_client.chat(
                        build_judge_messages(row),
                        temperature=args.judge_temperature,
                        max_tokens=args.judge_max_tokens,
                        response_json=True,
                    )
                    verdict = parse_json_object(response["_content"])
                    valid = bool(verdict.get("valid"))
                    judged = {**row, "judge_valid": valid}
                    append_jsonl(args.judged_output, [judged])
                    judged_by_key[key] = judged
                    if valid:
                        append_jsonl(args.accepted_output, [judged])
                        accepted_ids.add(variant_id)
                        accepted = True
                        print(
                            f"accept {variant_id}#{rollout_index}@{cfg.name}: accept",
                            file=sys.stderr,
                        )
                        break
                    print(
                        f"accept {variant_id}#{rollout_index}@{cfg.name}: judge_reject",
                        file=sys.stderr,
                    )
                except Exception as exc:
                    judged = {**row, "judge_valid": False, "error": type(exc).__name__}
                    append_jsonl(args.judged_output, [judged])
                    judged_by_key[key] = judged
                    print(
                        f"accept {variant_id}#{rollout_index}@{cfg.name}: judge_error {type(exc).__name__}",
                        file=sys.stderr,
                    )
            if accepted:
                break
        if not accepted:
            print(
                f"accept {variant_id}: no judged-valid rollout after {args.rollouts_per_variant} attempts",
                file=sys.stderr,
            )
        if args.sleep:
            time.sleep(args.sleep)


def command_export_sft(args: argparse.Namespace) -> None:
    out = []
    seen_variant_ids = set()
    duplicate_accepted = 0
    for row in read_jsonl(args.input):
        if not row.get("judge_valid", False):
            continue
        variant_id = str(row["id"])
        if variant_id in seen_variant_ids:
            duplicate_accepted += 1
            continue
        user = args.answer_prefix + row["question"].strip()
        assistant = normalize_sft_assistant(row)
        if not assistant:
            continue
        seen_variant_ids.add(variant_id)
        out.append(
            {
                "id": f"{args.id_prefix}{len(out)}",
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    write_jsonl(args.output, out)
    print_json(
        {
            "input": str(args.input),
            "output": str(args.output),
            "rows": len(out),
            "unique_variants": len(seen_variant_ids),
            "duplicate_accepted_skipped": duplicate_accepted,
        }
    )


def command_usage_summary(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    requests: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        request_id = str(row.get(args.request_id_key) or row.get("id") or index)
        usage = row.get(args.usage_key) or {}
        if usage and request_id not in requests:
            requests[request_id] = usage

    def values(key: str) -> list[int]:
        return [int((usage.get(key) or 0)) for usage in requests.values()]

    def nested_values(parent: str, key: str) -> list[int]:
        out = []
        for usage in requests.values():
            details = usage.get(parent) or {}
            out.append(int(details.get(key) or 0))
        return out

    print_json(
        {
            "input": str(args.input),
            "rows": len(rows),
            "unique_requests": len(requests),
            "prompt_tokens": summarize_numbers(values("prompt_tokens")),
            "completion_tokens": summarize_numbers(values("completion_tokens")),
            "total_tokens": summarize_numbers(values("total_tokens")),
            "reasoning_tokens": summarize_numbers(
                nested_values("completion_tokens_details", "reasoning_tokens")
            ),
        }
    )


def command_run_all(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = args.out_dir / "variants.jsonl"
    rejected = args.out_dir / "variants.rejected.jsonl"
    rollouts_passed = args.out_dir / "rollouts.passed.jsonl"
    rollouts_failed = args.out_dir / "rollouts.failed.jsonl"
    judged = args.out_dir / "judged.jsonl"
    accepted = args.out_dir / "accepted.jsonl"
    sft = args.out_dir / "sft.jsonl"

    if not args.skip_download and not args.input.exists():
        download_args = argparse.Namespace(
            dataset_id=args.dataset_id,
            config=args.dataset_config,
            split=args.split,
            limit=args.source_limit,
            page_size=100,
            raw_dir=DEFAULT_RAW_DIR,
            output=args.input,
            skip_parquet=args.skip_parquet,
            no_resume=False,
        )
        command_download(download_args)

    generate_args = argparse.Namespace(
        config=args.config,
        generator_url=args.generator_url,
        generator_base_url=args.generator_base_url,
        generator_api_key=args.generator_api_key,
        generator_model=args.generator_model,
        generator_default_model=args.generator_default_model,
        input=args.input,
        output=variants,
        rejected_output=rejected,
        start=args.start,
        limit=args.limit,
        variants_per_problem=args.variants_per_problem,
        generation_batch_size=args.generation_batch_size,
        languages=args.languages,
        json_response_format=False,
        temperature=0.7,
        max_tokens=2_500,
        sleep=0.0,
    )
    command_generate(generate_args)

    accept_args = argparse.Namespace(
        config=args.config,
        rollout_url=args.rollout_url,
        rollout_base_url=args.rollout_base_url,
        rollout_api_key=args.rollout_api_key,
        rollout_model=args.rollout_model,
        rollout_default_model=args.rollout_default_model,
        judge_url=args.judge_url,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        judge_default_model=args.judge_default_model,
        input=variants,
        rollout_output=rollouts_passed,
        rollout_failed_output=rollouts_failed,
        judged_output=judged,
        accepted_output=accepted,
        start=0,
        limit=0,
        rollouts_per_variant=args.rollouts_per_variant,
        rollout_policy=args.rollout_policy,
        seed=args.seed,
        rollout_temperature=0.6,
        rollout_max_tokens=4_096,
        judge_temperature=0.0,
        judge_max_tokens=700,
        sleep=0.0,
    )
    command_accept_rollouts(accept_args)

    export_args = argparse.Namespace(
        input=accepted,
        output=sft,
        answer_prefix="",
        id_prefix="dapo_math_17k_",
    )
    command_export_sft(export_args)


def normalize_dapo_row(row: dict[str, Any], row_idx: int, dataset_id: str) -> dict[str, Any]:
    prompt = row.get("prompt") or []
    prompt_content = ""
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                prompt_content = str(message.get("content", ""))
                break
        if not prompt_content and prompt:
            prompt_content = str(prompt[0].get("content", ""))
    else:
        prompt_content = str(prompt)

    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}
    source_index = str(extra_info.get("index") or row_idx)
    return {
        "id": f"dapo_{source_index}",
        "dataset_id": dataset_id,
        "source_row": row_idx,
        "data_source": row.get("data_source"),
        "ability": row.get("ability"),
        "question": strip_dapo_prompt_wrapper(prompt_content),
        "original_prompt": prompt_content,
        "ground_truth": str(reward_model.get("ground_truth", "")).strip(),
        "reward_style": reward_model.get("style"),
        "extra_info": extra_info,
    }


def strip_dapo_prompt_wrapper(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    marker = "\n\n"
    if marker in text and text.lower().startswith("solve the following math problem"):
        text = text.split(marker, 1)[1]
    text = re.sub(
        r'\n*Remember to put your answer on its own line after "Answer:"\.\s*$',
        "",
        text,
        flags=re.I,
    ).strip()
    return text


def build_generation_messages(
    source: dict[str, Any],
    variants_per_problem: int,
    accepted_questions: list[str] | None = None,
    target_languages: list[str] | None = None,
) -> list[dict[str, str]]:
    language_block = ""
    if target_languages:
        language_block = (
            "\nTarget languages in order; write each corresponding question in that language "
            "while keeping math notation in LaTeX:\n"
            + json.dumps(target_languages, ensure_ascii=False)
            + "\nEach item must include its language code.\n"
        )
    user = f"""Do not think step by step. Return valid JSON only.
Generate exactly {variants_per_problem} variants.

Rules:
- Same math structure and difficulty as the original.
- Change only numbers/parameters; the final answer must change.
- New question must be self-contained.
- answer is final answer only, no explanation, under 40 characters.
- Prefer exact integer/rational/radical answers.
- Return JSON exactly like:
{{"items":[{{"question":"...","answer":"...","language":"en"}}]}}
{language_block}

Original:
{source["question"]}

Original answer:
{source["ground_truth"]}
"""
    if accepted_questions:
        user += "\nAlready generated variants; do not duplicate these:\n"
        user += json.dumps(accepted_questions, ensure_ascii=False, indent=2)
    return [{"role": "user", "content": user}]


def build_rollout_messages(question: str) -> list[dict[str, str]]:
    user = (
        "Solve the following math problem step by step. "
        "End with a final line exactly in the form `Answer: <answer>`.\n\n"
        f"{question.strip()}"
    )
    return [{"role": "user", "content": user}]


def build_judge_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a math proof judge. Check whether the assistant reasoning is "
        "mathematically valid and whether the final answer equals the reference answer."
    )
    user = f"""Return pure JSON:
{{"valid":true|false}}

Question:
{row["question"]}

Reference final answer:
{row["ref_answer"]}

Assistant solution:
{row["assistant"]}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_sft_assistant(row: dict[str, Any]) -> str:
    assistant = str(row.get("assistant") or "").strip()
    if not assistant:
        return ""
    final = str(row.get("predicted_answer") or row.get("ref_answer") or extract_final_answer(assistant)).strip()
    if "<think" not in assistant.lower():
        assistant = f"<think>\n{assistant}\n</think>"
    if not has_strict_final_answer_line(assistant):
        assistant = assistant.rstrip() + f"\n\nAnswer: {final}"
    return assistant


def has_strict_final_answer_line(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and re.match(r"^Answer\s*:\s*.+$", lines[-1]))


def validate_variants(payload: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("generator output missing items list")
    if len(items) != expected_count:
        raise ValueError(f"expected {expected_count} items, got {len(items)}")
    out = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} is not an object")
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if len(question) < 20:
            raise ValueError(f"item {index} question is too short")
        if not answer:
            raise ValueError(f"item {index} answer is empty")
        if len(answer) > 120:
            raise ValueError(f"item {index} answer is too long; expected final answer only")
        out.append(item)
    return out


def extract_final_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def answers_match(predicted: str, reference: str) -> bool:
    return canonical_answer(predicted) == canonical_answer(reference)


def canonical_answer(value: str) -> str:
    text = value.strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.strip().strip("$").strip()
    text = re.sub(r"^\\\(|\\\)$", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace(",", "")
    text = text.rstrip(".")
    text = WS_RE.sub("", text)
    return text.lower()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def fetch_total_rows(dataset_id: str, config: str, split: str) -> int:
    payload = fetch_rows(dataset_id, config, split, 0, 1)
    return int(payload.get("num_rows_total") or 0)


def fetch_rows(dataset_id: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset_id,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")


def fetch_json(url: str, retries: int = 5) -> dict[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise
            last_error = exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
        sleep_s = min(2**attempt, 20)
        print(f"retrying HTTP fetch after {type(last_error).__name__}: sleep={sleep_s}s", file=sys.stderr)
        time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "rwkv-dapo-math-sft/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, target.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL line {line_no} in {path}: {exc}") from exc


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def read_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    for row_index, row in enumerate(read_jsonl(path)):
        if row_index == index:
            return row
    raise SystemExit(f"index {index} out of range for {path}")


def slice_rows(rows: Iterable[dict[str, Any]], start: int, limit: int) -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows):
        if index < start:
            continue
        if limit and len(out) >= limit:
            break
        out.append(row)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        row_id = row.get("id")
        if row_id:
            ids.add(str(row_id))
    return ids


def accepted_variant_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in read_jsonl(path) if row.get("judge_valid", False)}


def progress_key(variant_id: str, model_name: str, rollout_index: int) -> str:
    return f"{variant_id}\t{model_name}\t{rollout_index}"


def keyed_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for row in read_jsonl(path):
        key = progress_key(
            str(row.get("id", "")),
            str(row.get("rollout_model") or ""),
            int(row.get("rollout_index", 0)),
        )
        rows[key] = row
    return rows


def completed_rollout_keys(passed_path: Path, failed_path: Path) -> set[str]:
    keys = set()
    for path in (passed_path, failed_path):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("error"):
                continue
            keys.add(
                progress_key(
                    str(row.get("id", "")),
                    str(row.get("rollout_model") or ""),
                    int(row.get("rollout_index", 0)),
                )
            )
    return keys


def completed_judge_keys(judged_path: Path, accepted_path: Path) -> set[str]:
    keys = set()
    for path in (judged_path, accepted_path):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("error"):
                continue
            keys.add(
                progress_key(
                    str(row.get("id", "")),
                    str(row.get("rollout_model") or ""),
                    int(row.get("rollout_index", 0)),
                )
            )
    return keys


def select_rollout_models(
    models: list[ModelConfig],
    variant_id: str,
    rollout_index: int,
    policy: str,
    seed: int,
) -> list[ModelConfig]:
    if not models:
        raise SystemExit("no rollout models configured")
    if policy == "all":
        return models
    order = list(models)
    random.Random(f"{seed}:{variant_id}").shuffle(order)
    return [order[rollout_index % len(order)]]


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return (len(text) + 3) // 4


def quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    values = sorted(values)

    def pct(p: float) -> int:
        return values[min(len(values) - 1, int((len(values) - 1) * p))]

    return {
        "min": values[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": values[-1],
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"sum": 0, "avg": 0, "min": 0, "max": 0}
    total = sum(values)
    return {
        "sum": total,
        "avg": round(total / len(values), 2),
        "min": min(values),
        "max": max(values),
    }


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
