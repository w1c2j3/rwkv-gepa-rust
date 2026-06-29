#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_RE = re.compile(r"^[A-J]$")
CHOICE_BLOCK_RE = re.compile(
    r"(?ms)^\s*([A-J])\.\s*(.*?)(?=^\s*[A-J]\.\s|\Z)"
)
SUBJECT_RE = re.compile(r"expert in ([^.]+)\.", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:final answer|therefore,\s*the answer is|answer)\b.*$"
)
BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw wrong-answer eval rows into compact synthesis input rows."
    )
    parser.add_argument(
        "--input",
        default="data/inputs/mmlu_pro_wrong_updated.json",
        help="Raw wrong-answer JSON array or JSONL input.",
    )
    parser.add_argument(
        "--output",
        default="data/inputs/mmlu_pro_wrong_updated.cleaned.json",
        help="Cleaned JSON array output.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional cleaning report JSON output. Empty means no report is written.",
    )
    parser.add_argument(
        "--reasoning-chars",
        type=int,
        default=1200,
        help="Maximum chars kept from the model's wrong reasoning excerpt.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise SystemExit(f"{path} must contain a JSON array or JSONL rows")
        return rows
    out = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no} is not a JSON object")
        out.append(row)
    return out


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def canonical_label(value: Any) -> str:
    label = normalize_text(value).upper()
    return label if LABEL_RE.fullmatch(label) else ""


def parse_context(context: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(context)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "context JSON is not an object"
    return payload, None


def extract_user_from_prompt(prompt: str) -> str:
    prompt = normalize_text(prompt)
    if prompt.startswith("User:"):
        prompt = prompt[len("User:") :].strip()
    if "\nAssistant:" in prompt:
        prompt = prompt.split("\nAssistant:", 1)[0].strip()
    elif "Assistant:" in prompt:
        prompt = prompt.split("Assistant:", 1)[0].strip()
    return normalize_text(prompt)


def extract_subject(prompt: str, row: dict[str, Any]) -> str:
    existing = normalize_text(row.get("subject"))
    if existing:
        return existing
    match = SUBJECT_RE.search(prompt)
    return match.group(1).strip().lower() if match else ""


def parse_choices(user: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for label, text in CHOICE_BLOCK_RE.findall(user):
        cleaned = normalize_text(text)
        if cleaned:
            choices[label] = cleaned
    return choices


def stage_value(stage: Any, key: str) -> str:
    if isinstance(stage, dict):
        return normalize_text(stage.get(key))
    return ""


def wrong_reasoning_excerpt(completion: str, limit: int) -> str:
    completion = normalize_text(completion)
    if completion.startswith(">"):
        completion = completion[1:].lstrip()
    lower = completion.lower()
    close = lower.find("</think>")
    if close >= 0:
        completion = completion[:close]
    completion = completion.replace("<think>", "").replace("<thinking>", "")
    completion = completion.replace("</thinking>", "")
    lines: list[str] = []
    seen_recent: set[str] = set()
    for raw_line in completion.splitlines():
        line = normalize_text(FINAL_ANSWER_RE.sub("", raw_line))
        if not line:
            continue
        key = re.sub(r"\s+", " ", line.lower())
        if key in seen_recent:
            continue
        lines.append(line)
        seen_recent.add(key)
        if len(seen_recent) > 80:
            seen_recent = set(list(seen_recent)[-40:])
    excerpt = "\n".join(lines).strip()
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0].strip()
    return excerpt


def clean_row(
    row: dict[str, Any], index: int, reasoning_chars: int
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    context = normalize_text(row.get("context"))
    if not context:
        return None, "missing_context", {}

    payload, parse_error = parse_context(context)
    if payload is None:
        return None, "context_parse_failed", {"parse_error": parse_error}

    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        return None, "missing_stages", {}

    stage1 = stages[0] if isinstance(stages[0], dict) else {}
    prompt = stage_value(stage1, "prompt")
    user = extract_user_from_prompt(prompt)
    if not user:
        return None, "missing_user_prompt", {}

    wrong_answer = canonical_label(row.get("answer"))
    correct_answer = canonical_label(row.get("ref_answer"))
    if not wrong_answer:
        return None, "invalid_wrong_answer", {}
    if not correct_answer:
        return None, "invalid_ref_answer", {}
    if wrong_answer == correct_answer:
        return None, "wrong_equals_ref", {}

    choices = parse_choices(user)
    completion = stage_value(stage1, "completion")
    stage2_completion = ""
    if len(stages) > 1 and isinstance(stages[1], dict):
        stage2_completion = normalize_text(stages[1].get("completion"))

    cleaned: dict[str, Any] = {}
    for key in [
        "name",
        "source",
        "dataset",
        "task_id",
        "completions_id",
        "sample_index",
        "repeat_index",
        "pass_index",
        "fail_reason",
    ]:
        if key in row:
            cleaned[key] = row[key]

    subject = extract_subject(prompt, row)
    if subject:
        cleaned["subject"] = subject

    cleaned.update(
        {
            "answer": wrong_answer,
            "ref_answer": correct_answer,
            "wrong_answer": wrong_answer,
            "correct_answer": correct_answer,
            "option_count": len(choices),
            "option_labels": sorted(choices),
            "wrong_answer_text": choices.get(wrong_answer, ""),
            "correct_answer_text": choices.get(correct_answer, ""),
            "model_final_answer": canonical_label(stage2_completion) or wrong_answer,
            "stage1_stop_reason": normalize_text(stage1.get("stop_reason")),
            "context": user,
            "model_reasoning_excerpt": wrong_reasoning_excerpt(
                completion, reasoning_chars
            ),
            "cleaning": {
                "raw_context_sha256": hashlib.sha256(
                    context.encode("utf-8")
                ).hexdigest(),
                "raw_context_chars": len(context),
                "removed_runtime_metadata": True,
                "choices_detected": len(choices),
                "raw_index": index,
            },
        }
    )

    diagnostics = {
        "choices_detected": len(choices),
        "stage_count": len(stages),
        "has_sampling_config": isinstance(payload.get("sampling_config"), dict),
        "has_decode_params": "sampling_config" in payload,
        "stage1_stop_reason": cleaned["stage1_stop_reason"],
        "stage2_stop_reason": stage_value(stages[1], "stop_reason")
        if len(stages) > 1
        else "",
        "context_chars": len(context),
        "clean_context_chars": len(user),
        "reasoning_excerpt_chars": len(cleaned["model_reasoning_excerpt"]),
        "model_final_matches_row_answer": cleaned["model_final_answer"] == wrong_answer,
    }
    return cleaned, "ok", diagnostics


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input))

    cleaned_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    choices_detected_counts: Counter[str] = Counter()
    stage_count_counts: Counter[str] = Counter()
    stage1_stop_counts: Counter[str] = Counter()
    stage2_stop_counts: Counter[str] = Counter()
    model_final_match_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    ref_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    raw_chars = 0
    clean_chars = 0

    for index, row in enumerate(rows):
        context = normalize_text(row.get("context"))
        raw_chars += len(context)
        cleaned, status, diagnostics = clean_row(row, index, args.reasoning_chars)
        status_counts[status] += 1
        if cleaned is None:
            continue
        cleaned_rows.append(cleaned)
        clean_chars += len(cleaned["context"])
        answer_counts[cleaned["answer"]] += 1
        ref_counts[cleaned["ref_answer"]] += 1
        if cleaned.get("subject"):
            subject_counts[cleaned["subject"]] += 1
        choices_detected_counts[str(diagnostics["choices_detected"])] += 1
        stage_count_counts[str(diagnostics["stage_count"])] += 1
        stage1_stop_counts[diagnostics["stage1_stop_reason"]] += 1
        stage2_stop_counts[diagnostics["stage2_stop_reason"]] += 1
        model_final_match_counts[
            str(diagnostics["model_final_matches_row_answer"])
        ] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(cleaned_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.report:
        report = {
            "input": args.input,
            "output": args.output,
            "rows_in": len(rows),
            "rows_out": len(cleaned_rows),
            "status_counts": dict(status_counts),
            "answer_counts": dict(answer_counts),
            "ref_answer_counts": dict(ref_counts),
            "top_subjects": subject_counts.most_common(30),
            "raw_context_chars": raw_chars,
            "clean_context_chars": clean_chars,
            "raw_to_clean_context_ratio": (raw_chars / clean_chars)
            if clean_chars
            else None,
            "choices_detected_counts": dict(choices_detected_counts),
            "stage_count_counts": dict(stage_count_counts),
            "stage1_stop_counts": dict(stage1_stop_counts),
            "stage2_stop_counts": dict(stage2_stop_counts),
            "model_final_matches_row_answer_counts": dict(model_final_match_counts),
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"rows_in={len(rows)}")
    print(f"rows_out={len(cleaned_rows)}")
    print(f"status_counts={dict(status_counts)}")
    print(f"raw_context_chars={raw_chars}")
    print(f"clean_context_chars={clean_chars}")
    print(out_path)
    if args.report:
        print(args.report)
    return 0 if not status_counts - Counter(ok=status_counts["ok"]) else 0


if __name__ == "__main__":
    sys.exit(main())
