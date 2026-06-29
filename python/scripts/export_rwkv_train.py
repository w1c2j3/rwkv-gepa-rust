#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


ANSWER_LABEL_RE = re.compile(r"([A-Z])\s*\.?\s*$")
OPTION_MARKER_RE = re.compile(r"(?m)(^|[\s:：])([A-J])\.\s+")
THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>", re.IGNORECASE | re.DOTALL
)
THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
TRAILING_LABEL_RE = re.compile(r"\n?\s*[A-J]\s*$")
BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
FORMAT_SELF_CHECK_MARKERS = (
    "Additional output rules",
    "Return the answer",
    "Final answer",
    "final answer must",
    "single uppercase option letter",
    "Bad style examples",
    "provider's native",
    "输出规则",
    "规则要求",
    "最终输出",
    "最后只输出",
    "只输出单个选项字母",
    "以单个选项字母结束",
    "不要添加额外解释",
    "不要输出",
)
DEFAULT_RUN_DIR = "data/sft.doubao-compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cleaned RWKV training jsonl from success / failed synthesis outputs."
    )
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="Synthesis run directory containing done/success.jsonl and done/failed.jsonl.",
    )
    parser.add_argument(
        "--success-path",
        default="",
        help="Path to success.jsonl. Defaults to <run-dir>/done/success.jsonl.",
    )
    parser.add_argument(
        "--failed-path",
        default="",
        help="Path to failed.jsonl. Defaults to <run-dir>/done/failed.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Structured output jsonl path. Defaults to <run-dir>.rwkv_train.jsonl.",
    )
    parser.add_argument(
        "--text-only-output-path",
        default="",
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
        "--include-failed",
        action="store_true",
        help="Also export failed answer rows. Disabled by default because failed rows can contain reasoning for the wrong answer.",
    )
    parser.add_argument(
        "--assistant-style",
        choices=("real-think", "no-think"),
        default="real-think",
        help='RWKV assistant target style. real-think keeps reasoning and appends "Therefore, the answer is X".',
    )
    return parser.parse_args()


def resolve_export_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = Path(args.run_dir)
    success_path = Path(args.success_path) if args.success_path else run_dir / "done/success.jsonl"
    failed_path = Path(args.failed_path) if args.failed_path else run_dir / "done/failed.jsonl"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(f"{run_dir}.rwkv_train.jsonl")
    )
    text_only_output_path = (
        Path(args.text_only_output_path)
        if args.text_only_output_path
        else Path(f"{run_dir}.rwkv_train.text_only.jsonl")
    )
    return success_path, failed_path, output_path, text_only_output_path


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


def normalize_rwkv_content(text: str, role: str = "") -> str:
    text = normalize_text(text).replace("\r\n", "\n").replace("\r", "\n")
    text = BLANK_LINE_RE.sub("\n", text)
    if role:
        prefix = f"{role}:"
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
        if role == "User" and "\nAssistant:" in text:
            text = text.split("\nAssistant:", 1)[0].strip()
    return text.strip()


def extract_expected_answer(row: dict) -> str:
    raw = normalize_text(row.get("generated_item_json", ""))
    if not raw:
        return ""
    try:
        generated = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return normalize_text(generated.get("answer", ""))


def canonical_answer_label(text: str) -> str:
    text = normalize_text(text)
    if len(text) == 1 and text.isalpha():
        return text.upper()
    match = ANSWER_LABEL_RE.search(text)
    return match.group(1).upper() if match else ""


def option_texts(user: str) -> list[tuple[str, str]]:
    matches = list(OPTION_MARKER_RE.finditer(user))
    options: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group(2).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(user)
        text = user[start:end].strip()
        if text:
            options.append((label, text))
    return options


def normalize_option_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_valid_unique_options(user: str, expected_label: str) -> bool:
    options = option_texts(user)
    if not options:
        return True
    labels = {label for label, _ in options}
    if expected_label and expected_label not in labels:
        return False
    seen_texts: set[str] = set()
    for _label, text in options:
        normalized = normalize_option_text(text)
        if normalized in seen_texts:
            return False
        seen_texts.add(normalized)
    return True


def extract_assistant_final_label(assistant: str) -> str:
    assistant = normalize_text(assistant)
    split = THINK_CLOSE_RE.split(assistant, maxsplit=1)
    if len(split) == 2:
        tail = split[1]
        label = canonical_answer_label(tail)
        if label:
            return label
    return canonical_answer_label(assistant)


def extract_think_reasoning(assistant: str) -> str:
    assistant = normalize_text(assistant)
    match = THINK_BLOCK_RE.search(assistant)
    if not match:
        return ""

    reasoning = normalize_rwkv_content(match.group(1))
    return strip_format_self_check(reasoning)


def extract_visible_reasoning(assistant: str) -> str:
    assistant = normalize_rwkv_content(assistant)
    if not assistant or THINK_BLOCK_RE.search(assistant):
        return ""
    reasoning = strip_format_self_check(assistant)
    if len(reasoning) < 80:
        return ""
    return reasoning


def strip_format_self_check(reasoning: str) -> str:
    lines = reasoning.splitlines()
    cut = len(lines)
    for index, line in enumerate(lines):
        lower_line = line.lower()
        if any(marker.lower() in lower_line for marker in FORMAT_SELF_CHECK_MARKERS):
            cut = index
            break
    reasoning = "\n".join(lines[:cut]).strip()
    return strip_trailing_answer_echo(reasoning)


def strip_trailing_answer_echo(reasoning: str) -> str:
    while True:
        stripped = TRAILING_LABEL_RE.sub("", reasoning).strip()
        if stripped == reasoning:
            return stripped
        reasoning = stripped


def build_training_assistant(row: dict, assistant_style: str = "real-think") -> tuple[str, str]:
    assistant = normalize_text(row.get("assistant", ""))
    expected_answer = extract_expected_answer(row)
    expected_label = canonical_answer_label(expected_answer)
    if not expected_label:
        return "", expected_answer

    assistant_label = extract_assistant_final_label(assistant)
    if assistant_label != expected_label:
        return "", expected_label

    if assistant_style == "no-think":
        return assistant_label, expected_label

    reasoning = extract_think_reasoning(assistant)
    if not reasoning:
        reasoning = extract_visible_reasoning(assistant)
    if not reasoning:
        return "", expected_label

    return (
        f"<think>{reasoning}\n</think>Therefore, the answer is {assistant_label}",
        expected_label,
    )


def convert_rows(
    rows: list[dict],
    split: str,
    system_prompt: str,
    assistant_style: str = "real-think",
) -> list[dict]:
    exported: list[dict] = []
    seen_task_ids: set[str] = set()

    for row in rows:
        task_id = normalize_text(row.get("task_id", ""))
        if task_id and task_id in seen_task_ids:
            continue
        if task_id:
            seen_task_ids.add(task_id)

        user = normalize_rwkv_content(row.get("user", ""), role="User")
        if not user:
            continue

        assistant, expected_answer = build_training_assistant(row, assistant_style)
        if not assistant:
            continue
        if not has_valid_unique_options(user, canonical_answer_label(expected_answer)):
            continue

        text_parts: list[str] = []
        if system_prompt:
            text_parts.append(f"System: {normalize_rwkv_content(system_prompt, role='System')}")
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
    success_path, failed_path, output_path, text_only_output_path = resolve_export_paths(args)

    success_rows = load_jsonl(success_path)
    failed_rows = load_jsonl(failed_path) if args.include_failed else []

    exported_rows = []
    exported_rows.extend(
        convert_rows(success_rows, "success", system_prompt, args.assistant_style)
    )
    exported_rows.extend(
        convert_rows(failed_rows, "failed", system_prompt, args.assistant_style)
    )

    write_jsonl(output_path, exported_rows)
    write_jsonl(
        text_only_output_path,
        [{"text": row["text"]} for row in exported_rows],
    )

    success_count = sum(1 for row in exported_rows if row["source_split"] == "success")
    failed_count = sum(1 for row in exported_rows if row["source_split"] == "failed")

    print(f"success={success_count}")
    print(f"failed={failed_count}")
    print(f"total={len(exported_rows)}")
    print(output_path)
    print(text_only_output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
