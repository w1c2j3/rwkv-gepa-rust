#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import pipeline  # noqa: E402


LEAKY_META_KEYS = {
    "answer",
    "benchmark_name",
    "benchmark_split",
    "completions_id",
    "cot_mode",
    "evaluator",
    "expected_answer",
    "model_name",
    "model_version",
    "ref_answer",
    "sampling_config",
    "score_cot_mode",
    "source",
    "source_sample_id",
    "task_id",
}

META_ALLOWLIST = {
    "answer_check",
    "answer_style",
    "api_contract",
    "argument_constraint",
    "boundary_condition",
    "call_pattern",
    "changed_factor",
    "cluster_key",
    "complexity_requirement",
    "concept",
    "constraint_type",
    "dataset",
    "difficulty",
    "diversity_signature",
    "domain",
    "edge_case",
    "failure_mode",
    "format_constraint",
    "function_signature",
    "generation_stage",
    "instruction_type",
    "language",
    "numeric_scale",
    "operation_pattern",
    "option_label_distribution",
    "prompt_profile",
    "requires_clarification",
    "semantic_plan",
    "skill_node",
    "subspace_key",
    "task_kind",
    "tool_count",
    "tool_domain",
    "tool_namespace",
    "tool_schema_shape",
    "validation_contract",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final training jsonl from generated synthesis tasks with a hard audit gate."
    )
    parser.add_argument("--generated-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tasks-output-path", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--experiment", default="")
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional synth config; when provided, generated prompts are checked against cleaned source prompts.",
    )
    parser.add_argument(
        "--require-domain-count",
        action="append",
        default=[],
        metavar="DOMAIN=N",
        help="Require exact final row count for a domain. May be repeated.",
    )
    parser.add_argument(
        "--allow-issues",
        action="store_true",
        help="Write outputs even when audit issues exist. Default is hard gate failure.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid jsonl line {line_no} in {path}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def parse_required_counts(values: list[str]) -> dict[str, int]:
    out = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--require-domain-count must be DOMAIN=N, got {value!r}")
        domain, count = value.split("=", 1)
        out[pipeline.canonical_domain(domain)] = int(count)
    return out


def load_source_users(config_path: str) -> dict[str, str]:
    if not config_path:
        return {}
    config = pipeline.load_config(Path(config_path))
    return {
        pipeline.normalize_text(sample.user): sample.sample_id
        for sample in pipeline.load_samples(config)
    }


def generated_payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("generated_item_json")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("row missing generated_item_json")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("generated_item_json is not an object")
    return payload


def clean_meta(meta: dict[str, Any], domain: str, final_task_id: str, experiment: str) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in meta.items()
        if key in META_ALLOWLIST and key not in LEAKY_META_KEYS
    }
    cleaned["task_kind"] = domain
    cleaned["domain"] = domain
    cleaned["final_task_id"] = final_task_id
    if experiment:
        cleaned["experiment"] = experiment
    return cleaned


def convert_rows(
    rows: list[dict[str, Any]], experiment: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = []
    task_rows = []
    issues = []
    for index, row in enumerate(rows):
        try:
            payload = generated_payload(row)
        except Exception as exc:
            issues.append({"row_index": index, "kind": "bad_generated_item_json", "reason": str(exc)})
            continue
        user = str(payload.get("user") or row.get("user") or "").strip()
        assistant = str(payload.get("answer") or row.get("expected_answer") or "").strip()
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        domain = pipeline.canonical_sample_domain(meta)
        final_task_id = f"{experiment or 'synth'}_{index:03d}_{domain}"
        cleaned_meta = clean_meta(meta, domain, final_task_id, experiment)

        train_rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "meta": cleaned_meta,
            }
        )
        task_rows.append(
            {
                "task_id": final_task_id,
                "status": "generated",
                "domain": domain,
                "user": user,
                "expected_answer": assistant,
                "meta": cleaned_meta,
                "gen_input_configs": row.get("gen_input_configs", {}),
            }
        )
    return train_rows, task_rows, issues


def audit_train_rows(
    train_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    source_users: dict[str, str],
    required_counts: dict[str, int],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen_users: dict[str, tuple[str, str]] = {}
    signatures: Counter[str] = Counter()
    domains: Counter[str] = Counter()

    if len(train_rows) != len(task_rows):
        issues.append(
            {"kind": "row_count_mismatch", "train": len(train_rows), "tasks": len(task_rows)}
        )

    for index, row in enumerate(train_rows):
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        task_id = str(meta.get("final_task_id") or f"row_{index}")
        domain = pipeline.canonical_sample_domain(meta)
        domains[domain] += 1
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        if len(messages) != 2:
            issues.append({"task_id": task_id, "kind": "message_count"})
            continue
        user = str(messages[0].get("content") or "").strip()
        assistant = str(messages[1].get("content") or "").strip()
        if not user or not assistant:
            issues.append({"task_id": task_id, "kind": "empty_user_or_assistant"})
            continue

        user_key = pipeline.normalize_text(user)
        if user_key in seen_users:
            other_id, other_answer = seen_users[user_key]
            kind = "answer_conflict" if other_answer != assistant else "duplicate_user"
            issues.append({"task_id": task_id, "kind": kind, "other": other_id})
        seen_users[user_key] = (task_id, assistant)
        if user_key in source_users:
            issues.append(
                {
                    "task_id": task_id,
                    "kind": "identical_to_source",
                    "source_sample_id": source_users[user_key],
                }
            )

        leaked = sorted(key for key in meta if key in LEAKY_META_KEYS)
        if leaked:
            issues.append({"task_id": task_id, "kind": "leaky_meta", "keys": leaked})

        for field, text in (("user", user), ("assistant", assistant)):
            reason = pipeline.benchmark_contamination_reason(text)
            if reason and not (
                field == "assistant"
                and meta.get("answer_style") == "cot"
                and reason == "think_marker"
            ):
                issues.append(
                    {
                        "task_id": task_id,
                        "kind": "benchmark_contamination",
                        "field": field,
                        "reason": reason,
                    }
                )

        try:
            if domain == "function_calling":
                pipeline.validate_function_calling_item(user, assistant, index)
            elif domain == "coding":
                pipeline.validate_coding_item(user, assistant, meta, index)
            elif domain == "math":
                pipeline.validate_math_item(user, assistant, meta.get("answer_style", "final_only"), index)
            elif domain == "knowledge":
                pipeline.validate_knowledge_item(user, assistant, meta.get("answer_style", "final_only"), index)
            elif domain == "instruction_following":
                pipeline.validate_instruction_following_item(
                    user, assistant, meta.get("answer_style", "final_only"), index
                )
            else:
                issues.append({"task_id": task_id, "kind": "unknown_domain", "domain": domain})
        except Exception as exc:
            issues.append({"task_id": task_id, "kind": "local_validation", "reason": str(exc)})

        signature = meta.get("diversity_signature")
        if isinstance(signature, str) and signature.strip():
            signatures[signature] += 1

    for domain, expected in required_counts.items():
        actual = domains.get(domain, 0)
        if actual != expected:
            issues.append(
                {
                    "kind": "domain_count_mismatch",
                    "domain": domain,
                    "expected": expected,
                    "actual": actual,
                }
            )

    return {
        "total": len(train_rows),
        "domain_counts": dict(domains),
        "source_raw_fields_in_final_meta": sorted(
            {key for row in train_rows for key in (row.get("meta") or {}) if key in LEAKY_META_KEYS}
        ),
        "duplicate_diversity_signatures": {
            key: value for key, value in signatures.items() if value > 1
        },
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    generated_path = Path(args.generated_path)
    output_path = Path(args.output_path)
    tasks_output_path = Path(args.tasks_output_path)
    report_path = Path(args.report_path)
    required_counts = parse_required_counts(args.require_domain_count)
    source_users = load_source_users(args.config_path)

    raw_rows = load_jsonl(generated_path)
    train_rows, task_rows, convert_issues = convert_rows(raw_rows, args.experiment)
    report = audit_train_rows(train_rows, task_rows, source_users, required_counts)
    report["input_path"] = str(generated_path)
    report["output_path"] = str(output_path)
    report["tasks_output_path"] = str(tasks_output_path)
    report["convert_issues"] = convert_issues
    report["issue_count"] += len(convert_issues)
    report["issues"] = convert_issues + report["issues"]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if report["issue_count"] and not args.allow_issues:
        print(
            f"audit failed: issue_count={report['issue_count']} report={report_path}",
            file=sys.stderr,
        )
        return 2

    write_jsonl(output_path, train_rows)
    write_jsonl(tasks_output_path, task_rows)
    print(
        json.dumps(
            {
                "total": report["total"],
                "domain_counts": report["domain_counts"],
                "issue_count": report["issue_count"],
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
