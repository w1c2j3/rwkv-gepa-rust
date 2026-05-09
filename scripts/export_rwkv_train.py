#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cleaned RWKV training jsonl from success / failed synthesis outputs."
    )
    parser.add_argument(
        "--success-path",
        default="data/sft.doubao-compare/done/success.jsonl",
        help="Path to success.jsonl",
    )
    parser.add_argument(
        "--failed-path",
        default="data/sft.doubao-compare/done/failed.jsonl",
        help="Path to failed.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="data/sft.doubao-compare.rwkv_train.jsonl",
        help="Structured output jsonl path",
    )
    parser.add_argument(
        "--text-only-output-path",
        default="data/sft.doubao-compare.rwkv_train.text_only.jsonl",
        help='Text-only output jsonl path. Each row is {"text": "..."}',
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt to prepend to every sample",
    )
    parser.add_argument(
        "--system-file",
        default="",
        help="Optional file whose content is used as the system prompt",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Only export success rows",
    )
    return parser.parse_args()


def read_system_prompt(args: argparse.Namespace) -> str:
    if args.system and args.system_file:
        raise SystemExit("use either --system or --system-file, not both")
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8").strip()
    return args.system.strip()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid jsonl line {line_no} in {path}: {exc}") from exc
    return rows


def normalize_text(text: str) -> str:
    return text.strip() if isinstance(text, str) else ""


def extract_expected_answer(row: dict) -> str:
    raw = normalize_text(row.get("generated_item_json", ""))
    if not raw:
        return ""
    try:
        generated = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return normalize_text(generated.get("answer", ""))


def extract_think_block(assistant: str) -> str:
    assistant = normalize_text(assistant)
    if not assistant:
        return ""

    start = assistant.find("<think>")
    if start == -1:
        return ""

    end = assistant.find("</think>", start)
    if end == -1:
        inner = assistant[start + len("<think>") :].strip()
        if not inner:
            return "<think>\n</think>"
        return f"<think>\n{inner}\n</think>"

    inner = assistant[start + len("<think>") : end].strip()
    if not inner:
        return "<think>\n</think>"
    return f"<think>\n{inner}\n</think>"


def build_assistant(row: dict, split: str) -> tuple[str, str]:
    assistant = normalize_text(row.get("assistant", ""))
    expected_answer = extract_expected_answer(row)

    if split == "success":
        if assistant:
            return assistant, expected_answer
        return expected_answer, expected_answer

    think_block = extract_think_block(assistant)
    if think_block and expected_answer:
        return f"{think_block}\n\n{expected_answer}", expected_answer
    if think_block:
        return think_block, expected_answer
    if expected_answer:
        return expected_answer, expected_answer
    return assistant, expected_answer


def convert_rows(rows: list[dict], split: str, system_prompt: str) -> list[dict]:
    exported: list[dict] = []
    seen_task_ids: set[str] = set()

    for row in rows:
        task_id = normalize_text(row.get("task_id", ""))
        if task_id and task_id in seen_task_ids:
            continue
        if task_id:
            seen_task_ids.add(task_id)

        user = normalize_text(row.get("user", ""))
        if not user:
            continue

        assistant, expected_answer = build_assistant(row, split)
        if not assistant:
            continue

        text_parts: list[str] = []
        if system_prompt:
            text_parts.append(f"System: {system_prompt}")
        text_parts.append(f"User: {user}")
        text_parts.append(f"Assistant: {assistant}")
        text = "\n\n".join(text_parts)
        exported.append(
            {
                "task_id": task_id,
                "source_split": split,
                "source_status": normalize_text(row.get("status", "")),
                "answer_model": normalize_text(row.get("answer_model", "")),
                "expected_answer": expected_answer,
                "user": user,
                "assistant": assistant,
                "text": text,
            }
        )

    return exported


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    system_prompt = read_system_prompt(args)

    success_rows = load_jsonl(Path(args.success_path))
    failed_rows = [] if args.skip_failed else load_jsonl(Path(args.failed_path))

    exported_rows = []
    exported_rows.extend(convert_rows(success_rows, "success", system_prompt))
    exported_rows.extend(convert_rows(failed_rows, "failed", system_prompt))

    write_jsonl(Path(args.output_path), exported_rows)
    write_jsonl(
        Path(args.text_only_output_path),
        [{"text": row["text"]} for row in exported_rows],
    )

    success_count = sum(1 for row in exported_rows if row["source_split"] == "success")
    failed_count = sum(1 for row in exported_rows if row["source_split"] == "failed")

    print(f"success={success_count}")
    print(f"failed={failed_count}")
    print(f"total={len(exported_rows)}")
    print(args.output_path)
    print(args.text_only_output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
