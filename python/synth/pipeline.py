from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DOMAINS = {
    "coding",
    "knowledge",
    "math",
    "instruction_following",
    "function_calling",
}

ANSWER_STYLES = {
    "final_only",
    "brief_explanation",
    "json_object",
    "code",
    "function_call",
    "cot",
}

BENCHMARK_WRAPPER_PATTERNS = (
    ("talented_expert_wrapper", re.compile(r"\bYou are a very talented expert\b", re.I)),
    ("assistant_json_prefill", re.compile(r"\bAssistant:\s*```(?:json)?", re.I)),
    (
        "single_option_letter_wrapper",
        re.compile(r"\bAnswer this question and finish with a single option letter\b", re.I),
    ),
    (
        "boxed_solution_eval_wrapper",
        re.compile(r"\bSolve the problem using one clean solution path\b", re.I),
    ),
    (
        "boxed_answer_instruction",
        re.compile(r"\bPut the complete final answer inside\s+\\?\(?\\boxed", re.I),
    ),
    (
        "eval_loop_instruction",
        re.compile(r"\bDo not restart, repeat, enumerate alternative methods\b", re.I),
    ),
    ("think_marker", re.compile(r"\(think\)|<think>", re.I)),
)

COT_PATTERNS = (
    re.compile(r"<think(?:ing)?>|</think(?:ing)?>|\(think\)", re.I),
    re.compile(r"\blet'?s think\b|\bstep by step\b|\bchain[- ]of[- ]thought\b", re.I),
    re.compile(r"思考过程|推理过程|解题步骤|逐步推理"),
)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    endpoint: str
    model_name: str
    api_key: str = ""
    api_key_env: str = ""
    system_prompt: str = ""
    max_completion_tokens: int = 4096
    temperature: float | None = None
    reasoning_effort: str = ""
    thinking: dict[str, Any] | None = None
    enable_thinking: bool | None = None
    json_object_response: bool = False
    merge_reasoning_content: bool = False
    timeout_seconds: float = 240.0

    def resolved_key(self) -> str:
        key = self.api_key.strip()
        if not key and self.api_key_env:
            key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ValueError(
                f"missing API key; set api_key or env {self.api_key_env or '<unset>'}"
            )
        return key


@dataclasses.dataclass(frozen=True)
class PromptProfile:
    name: str
    task_kind: str
    answer_check: str
    answer_style: str
    strategy: str
    axes: tuple[str, ...]
    generation_template: str
    validation_template: str = ""


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    default_profile: Path
    profiles: dict[str, Path]
    selector_keys: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    dataset_path: Path | None
    db_snapshot_path: Path | None
    db_snapshot_command: tuple[str, ...]
    limit: int | None
    start_index: int
    prompt: PromptConfig
    generator: ModelConfig
    validator: ModelConfig | None
    variant_count: int
    generation_attempts: int
    validate_generated_questions: bool
    output_run_dir: Path
    max_concurrency: int = 1
    resume: bool = False
    request_batch_size: int = 0
    source_cluster_limit_per_domain: int = 0
    generated_cluster_limit_per_domain: int = 0


@dataclasses.dataclass(frozen=True)
class SourceSample:
    sample_id: str
    user: str
    meta: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class GeneratedItem:
    task_id: str
    user: str
    answer: str
    meta: dict[str, Any]


class OpenAIChatClient:
    def chat(self, model: ModelConfig, prompt: str) -> str:
        request = urllib.request.Request(
            model.endpoint,
            data=json.dumps(build_chat_payload(model, prompt)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {model.resolved_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=model.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"chat HTTP {err.code}: {detail[:1000]}") from err

        return parse_chat_response_text(body, model.merge_reasoning_content)


def build_chat_payload(model: ModelConfig, prompt: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if model.system_prompt:
        messages.append({"role": "system", "content": model.system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model.model_name,
        "messages": messages,
        "max_completion_tokens": model.max_completion_tokens,
    }
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    if model.reasoning_effort:
        payload["reasoning_effort"] = model.reasoning_effort
    if model.thinking is not None:
        payload["thinking"] = model.thinking
    if model.enable_thinking is not None:
        payload["enable_thinking"] = model.enable_thinking
    if model.json_object_response:
        payload["response_format"] = {"type": "json_object"}
    return payload


def parse_chat_response_text(body: str, merge_reasoning_content: bool) -> str:
    data = json.loads(body)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("chat response contains no choices")
    message = choices[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    reasoning = str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    ).strip()
    if merge_reasoning_content and reasoning:
        if content:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return f"<think>\n{reasoning}\n</think>"
    if not content:
        raise RuntimeError("chat response content is empty")
    return content


def load_config(path: Path) -> PipelineConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    prompt_raw = raw.get("prompt", {})
    input_raw = raw["input"]
    db_snapshot_raw = input_raw.get("db_snapshot", {})
    generator_raw = raw["generator"]
    validator_raw = raw.get("validator")
    output_raw = raw.get("output", {})
    run_raw = raw.get("run", {})

    def resolve(value: str) -> Path:
        out = Path(value)
        return out if out.is_absolute() else base / out

    dataset_path = input_raw.get("dataset_path")
    db_snapshot_path = db_snapshot_raw.get("path")

    profiles = {
        canonical_domain(name): resolve(profile_path)
        for name, profile_path in prompt_raw.get("profiles", {}).items()
    }
    default_profile = resolve(
        prompt_raw.get("profile_path", "prompts/knowledge.toml")
    )

    return PipelineConfig(
        dataset_path=resolve(dataset_path) if dataset_path else None,
        db_snapshot_path=resolve(db_snapshot_path) if db_snapshot_path else None,
        db_snapshot_command=tuple(str(part) for part in db_snapshot_raw.get("command", [])),
        limit=input_raw.get("limit"),
        start_index=int(input_raw.get("start_index", 0)),
        prompt=PromptConfig(
            default_profile=default_profile,
            profiles=profiles,
            selector_keys=tuple(
                prompt_raw.get(
                    "selector_keys",
                    ["profile", "task_kind", "domain", "subject", "dataset", "source"],
                )
            ),
        ),
        generator=model_config_from_raw(generator_raw, run_raw),
        validator=(
            model_config_from_raw(validator_raw, run_raw) if validator_raw else None
        ),
        variant_count=int(generator_raw["variant_count"]),
        generation_attempts=int(generator_raw.get("generation_attempts", 4)),
        validate_generated_questions=bool(
            generator_raw.get("validate_generated_questions", True)
        ),
        output_run_dir=resolve(output_raw.get("run_dir", "data/python_run")),
        max_concurrency=max(1, int(run_raw.get("max_concurrency", 1))),
        resume=bool(output_raw.get("resume", False)),
        request_batch_size=max(
            1,
            int(generator_raw.get("request_batch_size", generator_raw["variant_count"])),
        ),
        source_cluster_limit_per_domain=max(
            0, int(input_raw.get("source_cluster_limit_per_domain", 0))
        ),
        generated_cluster_limit_per_domain=max(
            0, int(generator_raw.get("generated_cluster_limit_per_domain", 0))
        ),
    )


def model_config_from_raw(raw: dict[str, Any], run_raw: dict[str, Any]) -> ModelConfig:
    raw_thinking = raw.get("thinking")
    thinking = raw_thinking if isinstance(raw_thinking, dict) else None
    enable_thinking = raw.get("enable_thinking")
    merge_reasoning_content = bool(raw.get("merge_reasoning_content", False))
    return ModelConfig(
        endpoint=str(raw["endpoint"]).strip(),
        model_name=str(raw["model_name"]).strip(),
        api_key=str(raw.get("api_key", "")).strip(),
        api_key_env=str(raw.get("api_key_env", "")).strip(),
        system_prompt=str(raw.get("system_prompt", "")).strip(),
        max_completion_tokens=int(raw.get("max_completion_tokens", 4096)),
        temperature=raw.get("temperature"),
        reasoning_effort=str(raw.get("reasoning_effort", "")).strip(),
        thinking=thinking,
        enable_thinking=bool(enable_thinking) if enable_thinking is not None else None,
        json_object_response=bool(raw.get("json_object_response", False)),
        merge_reasoning_content=merge_reasoning_content,
        timeout_seconds=float(run_raw.get("request_timeout_seconds", 240.0)),
    )


def load_prompt_profile(path: Path) -> PromptProfile:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    task_kind = canonical_domain(raw.get("task_kind", raw.get("name", "knowledge")))
    if task_kind not in DOMAINS:
        raise ValueError(f"unknown prompt task_kind {task_kind!r} in {path}")
    synthesis = raw.get("synthesis", {})
    return PromptProfile(
        name=str(raw.get("name", task_kind)).strip(),
        task_kind=task_kind,
        answer_check=str(raw.get("answer_check", default_answer_check(task_kind))).strip(),
        answer_style=normalize_answer_style(
            raw.get("answer_style", default_answer_style(task_kind))
        ),
        strategy=str(synthesis.get("strategy", "subspace_contract_diversity")).strip(),
        axes=tuple(str(axis).strip() for axis in synthesis.get("axes", []) if str(axis).strip()),
        generation_template=str(raw["generation"]["template"]).strip(),
        validation_template=str(raw.get("validation", {}).get("template", "")).strip(),
    )


def load_prompt_library(config: PromptConfig) -> dict[str, PromptProfile]:
    library = {"default": load_prompt_profile(config.default_profile)}
    for name, path in config.profiles.items():
        profile = load_prompt_profile(path)
        library[name] = profile
        library[canonical_domain(profile.task_kind)] = profile
        library[normalize_key(profile.name)] = profile
    return library


def load_samples(config: PipelineConfig) -> list[SourceSample]:
    input_path = materialize_input_snapshot(config)
    rows = list(load_json_values(input_path))
    window = rows[config.start_index :]
    samples = [
        normalize_sample(index + config.start_index, row)
        for index, row in enumerate(window)
    ]
    samples = filter_samples_by_source_cluster(samples, config)
    if config.limit is not None:
        samples = samples[: config.limit]
    return samples


def materialize_input_snapshot(config: PipelineConfig) -> Path:
    if config.db_snapshot_command:
        if not config.db_snapshot_path:
            raise ValueError("input.db_snapshot.command requires input.db_snapshot.path")
        result = subprocess.run(
            list(config.db_snapshot_command),
            check=True,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            config.db_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            config.db_snapshot_path.write_text(result.stdout, encoding="utf-8")
    if config.db_snapshot_path:
        return config.db_snapshot_path
    if config.dataset_path:
        return config.dataset_path
    raise ValueError(
        "input requires either input.db_snapshot.path or legacy input.dataset_path"
    )


def load_json_values(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        values = json.loads(text)
        if not isinstance(values, list):
            raise ValueError("JSON input must be an array or JSONL")
        return values
    values = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped:
            values.append(json.loads(stripped))
    return values


def normalize_sample(index: int, row: dict[str, Any]) -> SourceSample:
    if row.get("user"):
        user = sanitize_source_user(str(row["user"]))
    else:
        context = row.get("context", row.get("text", ""))
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False)
        user = sanitize_source_user(extract_user_from_context(context))
    if not user:
        raise ValueError(f"sample {index} has empty user prompt")

    sample_id = str(
        row.get("sample_id")
        or row.get("task_id")
        or row.get("id")
        or f"sample_{index:06d}"
    )
    meta = dict(row.get("meta") or {})
    for key in (
        "task_kind",
        "profile",
        "prompt_profile",
        "domain",
        "subject",
        "dataset",
        "source",
        "answer",
        "ref_answer",
        "expected_answer",
        "answer_style",
        "cot_mode",
        "cot_profile",
        "evaluator",
        "benchmark_name",
        "benchmark_split",
        "sampling_config",
        "task_id",
        "completions_id",
        "score_cot_mode",
        "model_name",
        "model_version",
        "option_labels",
    ):
        if key in row and key not in meta:
            meta[key] = row[key]
    task_kind = canonical_sample_domain(meta)
    meta.setdefault("task_kind", task_kind)
    meta.setdefault("source_cluster_key", cluster_key_for_text(task_kind, user, meta))
    return SourceSample(sample_id=sample_id, user=user, meta=meta)


def filter_samples_by_source_cluster(
    samples: list[SourceSample], config: PipelineConfig
) -> list[SourceSample]:
    limit = config.source_cluster_limit_per_domain
    if limit <= 0:
        return samples
    kept: list[SourceSample] = []
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        domain = canonical_sample_domain(sample.meta)
        cluster = str(
            sample.meta.get("source_cluster_key")
            or cluster_key_for_text(domain, sample.user, sample.meta)
        )
        key = (domain, cluster)
        if counts.get(key, 0) >= limit:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(sample)
    return kept


def choose_profile(
    sample: SourceSample, config: PromptConfig, library: dict[str, PromptProfile]
) -> PromptProfile:
    for key in config.selector_keys:
        value = sample.meta.get(key)
        if isinstance(value, str) and value.strip():
            profile = library.get(canonical_domain(value))
            if profile:
                return profile
            profile = library.get(normalize_key(value))
            if profile:
                return profile
    return library["default"]


def generate_dataset(
    config: PipelineConfig,
    client: Any | None = None,
) -> list[GeneratedItem]:
    client = client or OpenAIChatClient()
    library = load_prompt_library(config.prompt)
    samples = load_samples(config)
    config.output_run_dir.joinpath("generate").mkdir(parents=True, exist_ok=True)
    tasks_path = config.output_run_dir / "generate" / "tasks.jsonl"
    rejected_path = config.output_run_dir / "generate" / "rejected.jsonl"
    progress_path = config.output_run_dir / "generate" / "progress.jsonl"
    progress_enabled = os.environ.get("SYNTH_PROGRESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    accepted_all: list[GeneratedItem] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_signatures: dict[str, int] = {}
    seen_clusters: dict[str, int] = {}
    emit_lock = threading.Lock()
    signature_lock = threading.Lock()
    write_lock = threading.Lock()

    existing_by_source: dict[str, list[GeneratedItem]] = {}
    if config.resume and tasks_path.exists():
        for row in load_json_values(tasks_path):
            item = row_to_item(row)
            if item is None:
                continue
            accepted_all.append(item)
            source_id = str(item.meta.get("source_sample_id") or "")
            if source_id:
                existing_by_source.setdefault(source_id, []).append(item)
            profile = choose_profile(
                SourceSample(source_id or item.task_id, item.user, item.meta),
                config.prompt,
                library,
            )
            signature = diversity_signature(item, profile)
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
            cluster = generated_cluster_key(item, profile)
            seen_clusters[cluster] = seen_clusters.get(cluster, 0) + 1

    if not config.resume:
        tasks_path.write_text("", encoding="utf-8")
        rejected_path.write_text("", encoding="utf-8")

    def emit_progress(event: str, **fields: Any) -> None:
        if not progress_enabled:
            return
        row = {
            "event": event,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **fields,
        }
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with emit_lock:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
            print(line, flush=True)

    if progress_enabled and (not config.resume or not progress_path.exists()):
        progress_path.write_text("", encoding="utf-8")
    emit_progress(
        "run_start",
        samples=len(samples),
        existing=len(accepted_all),
        variant_count=config.variant_count,
        request_batch_size=config.request_batch_size,
        generation_attempts=config.generation_attempts,
        validate_generated_questions=config.validate_generated_questions,
        max_concurrency=config.max_concurrency,
        resume=config.resume,
    )

    def process_sample(
        sample_index: int, sample: SourceSample
    ) -> tuple[int, list[GeneratedItem], list[GeneratedItem], list[dict[str, Any]]]:
        profile = choose_profile(sample, config.prompt, library)
        accepted: list[GeneratedItem] = list(
            existing_by_source.get(sample.sample_id, [])
        )[: config.variant_count]
        if len(accepted) >= config.variant_count:
            emit_progress(
                "sample_skipped",
                sample_index=sample_index,
                sample_count=len(samples),
                sample_id=sample.sample_id,
                task_kind=profile.task_kind,
                prompt_profile=profile.name,
                existing=len(accepted),
            )
            return sample_index, accepted, [], []
        emit_progress(
            "sample_start",
            sample_index=sample_index,
            sample_count=len(samples),
            sample_id=sample.sample_id,
            task_kind=profile.task_kind,
            prompt_profile=profile.name,
            existing=len(accepted),
        )
        new_accepted: list[GeneratedItem] = []
        local_rejected_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for attempt in range(config.generation_attempts):
            remaining = config.variant_count - len(accepted)
            if remaining <= 0:
                break
            batch_count = min(remaining, config.request_batch_size)
            prompt = build_generation_prompt(
                profile,
                sample,
                batch_count,
                accepted,
                errors[-1] if errors else "",
            )
            try:
                raw = client.chat(config.generator, prompt)
                candidates, rejected = parse_generated_items(raw, sample, profile)
                if config.validate_generated_questions and config.validator and candidates:
                    candidates, model_rejected = validate_with_model(
                        client, config.validator, profile, sample, candidates
                    )
                    rejected.extend(model_rejected)
            except Exception as err:
                candidates = []
                rejected = [f"attempt failed: {err}"]
            with signature_lock:
                selected, diversity_rejected = select_diverse_items(
                    candidates,
                    profile,
                    seen_signatures,
                    seen_clusters,
                    config.generated_cluster_limit_per_domain,
                    remaining,
                )
                for item in selected:
                    signature = diversity_signature(item, profile)
                    seen_signatures[signature] = (
                        seen_signatures.get(signature, 0) + 1
                    )
                    cluster = generated_cluster_key(item, profile)
                    seen_clusters[cluster] = seen_clusters.get(cluster, 0) + 1
            selected = assign_task_ids(sample, selected, len(accepted))
            rejected.extend(diversity_rejected)
            attempt_rejected_rows: list[dict[str, Any]] = []
            for reason in rejected:
                attempt_rejected_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task_kind": profile.task_kind,
                        "prompt_profile": profile.name,
                        "attempt": attempt,
                        "reason": reason,
                    }
                )
            if selected or attempt_rejected_rows:
                with write_lock:
                    append_jsonl(
                        tasks_path,
                        [item_to_row(item, config.generator) for item in selected],
                    )
                    append_jsonl(rejected_path, attempt_rejected_rows)
            local_rejected_rows.extend(attempt_rejected_rows)
            accepted.extend(selected)
            new_accepted.extend(selected)
            emit_progress(
                "attempt_done",
                sample_index=sample_index,
                sample_id=sample.sample_id,
                attempt=attempt,
                accepted_this_attempt=len(selected),
                accepted_total=len(accepted),
                rejected_this_attempt=len(rejected),
                remaining=max(config.variant_count - len(accepted), 0),
            )
            errors = prompt_feedback_errors(rejected)
        accepted = accepted[: config.variant_count]
        emit_progress(
            "sample_done",
            sample_index=sample_index,
            sample_id=sample.sample_id,
            task_kind=profile.task_kind,
            accepted=len(accepted),
            accepted_new=len(new_accepted),
        )
        return sample_index, accepted, new_accepted, local_rejected_rows

    results: list[tuple[list[GeneratedItem], list[dict[str, Any]]] | None] = [
        None for _sample in samples
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.max_concurrency
    ) as executor:
        futures = [
            executor.submit(process_sample, sample_index, sample)
            for sample_index, sample in enumerate(samples, 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            sample_index, accepted, new_accepted, local_rejected_rows = future.result()
            results[sample_index - 1] = (new_accepted, local_rejected_rows)

    for result in results:
        if result is None:
            continue
        accepted, local_rejected_rows = result
        accepted_all.extend(accepted)
        rejected_rows.extend(local_rejected_rows)

    emit_progress(
        "run_done",
        generated=len(accepted_all),
        rejected=len(rejected_rows),
        tasks_path=str(tasks_path),
        rejected_path=str(rejected_path),
    )
    return accepted_all


def build_generation_prompt(
    profile: PromptProfile,
    sample: SourceSample,
    variant_count: int,
    accepted: list[GeneratedItem],
    feedback: str,
) -> str:
    source = {
        "sample_id": sample.sample_id,
        "user": sample.user,
        "meta": sample.meta,
    }
    subspace_plan = build_subspace_plan(sample, profile)
    source_style = expected_answer_style(sample, profile) or profile.answer_style
    accepted_json = [accepted_item_summary(item, profile) for item in accepted]
    variables = {
        "profile_name": profile.name,
        "task_kind": profile.task_kind,
        "answer_check": profile.answer_check,
        "answer_style": source_style,
        "variant_count": str(variant_count),
        "source_sample_json": json.dumps(source, ensure_ascii=False, indent=2),
        "accepted_samples_json": json.dumps(
            accepted_json, ensure_ascii=False, indent=2
        ),
        "feedback_block": (
            f"\n上一轮输出未通过：\n{feedback}\n请修复这些问题。\n"
            if feedback
            else ""
        ),
        "synthesis_profile_json": json.dumps(
            {
                "task_kind": profile.task_kind,
                "answer_check": profile.answer_check,
                "answer_style": source_style,
                "profile_default_answer_style": profile.answer_style,
                "strategy": profile.strategy,
                "axes": profile.axes,
                "subspace_plan": subspace_plan,
                "required_meta": [
                    "semantic_plan",
                    "validation_contract",
                    "changed_factor",
                    "cluster_key",
                    "diversity_signature",
                    "answer_style",
                ],
                "selection_policy": "prefer sparse subspaces, non-duplicate cluster_key, and non-duplicate diversity signatures",
            },
            ensure_ascii=False,
            indent=2,
        ),
    }
    return render_template(profile.generation_template, variables)


def accepted_item_summary(item: GeneratedItem, profile: PromptProfile) -> dict[str, Any]:
    meta_keys = (
        "task_kind",
        "answer_style",
        "semantic_plan",
        "changed_factor",
        "validation_contract",
        "cluster_key",
        "diversity_signature",
        *profile.axes,
    )
    meta = {
        key: item.meta[key]
        for key in meta_keys
        if key in item.meta and item.meta[key] not in ("", None)
    }
    return {
        "task_id": item.task_id,
        "meta": meta,
    }


def prompt_feedback_errors(rejected: list[str]) -> list[str]:
    if not rejected:
        return ["not enough accepted candidates"]
    feedback = []
    for reason in rejected:
        if reason.startswith("attempt failed: "):
            if (
                "Remote end closed connection" in reason
                or "timed out" in reason
                or "timeout" in reason.lower()
                or "error sending request" in reason
            ):
                continue
        feedback.append(reason)
    return feedback


def build_subspace_plan(sample: SourceSample, profile: PromptProfile) -> dict[str, Any]:
    values = {}
    missing = []
    for axis in profile.axes:
        value = sample.meta.get(axis)
        if value is None or value == "":
            missing.append(axis)
            values[axis] = "infer_from_prompt"
        else:
            values[axis] = value
    key = "|".join(f"{axis}={values[axis]}" for axis in profile.axes)
    return {
        "method": "subspace_partition",
        "subspace_key": stable_hash(key or sample.user),
        "axis_values": values,
        "missing_axes_to_infer": missing,
    }


def parse_generated_items(
    text: str, sample: SourceSample, profile: PromptProfile
) -> tuple[list[GeneratedItem], list[str]]:
    payload = json.loads(extract_json_object(text))
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("generator output JSON must contain items array")
    items: list[GeneratedItem] = []
    rejected: list[str] = []
    for index, raw in enumerate(raw_items):
        try:
            item = normalize_generated_item(raw, sample, profile, index)
        except ValueError as err:
            rejected.append(str(err))
            continue
        items.append(item)
    return items, rejected


def normalize_generated_item(
    raw: dict[str, Any], sample: SourceSample, profile: PromptProfile, index: int
) -> GeneratedItem:
    if not isinstance(raw, dict):
        raise ValueError(f"item {index} is not an object")
    user = sanitize_user(str(raw.get("user", "")))
    raw_answer = raw.get("answer", "")
    if isinstance(raw_answer, (dict, list)):
        answer = json.dumps(raw_answer, ensure_ascii=False, separators=(",", ":"))
    else:
        answer = str(raw_answer).strip()
    if not user:
        raise ValueError(f"item {index} has empty user")
    if not answer:
        raise ValueError(f"item {index} has empty answer")
    if normalize_text(user) == normalize_text(sample.user):
        raise ValueError(f"item {index} is identical to source")
    reason = benchmark_contamination_reason(user)
    if reason:
        raise ValueError(f"item {index} rejected for benchmark contamination: {reason}")
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)
    meta.setdefault("task_kind", profile.task_kind)
    meta.setdefault("answer_check", profile.answer_check)
    meta["answer_style"] = normalize_answer_style(
        meta.get(
            "answer_style",
            infer_answer_style(profile.task_kind, user, answer),
        )
    )
    meta.setdefault("prompt_profile", profile.name)
    meta.setdefault("subspace_key", build_subspace_plan(sample, profile)["subspace_key"])
    meta.setdefault("source_sample_id", sample.sample_id)
    meta.setdefault("generation_stage", "variant_from_failed_sample")
    meta["cluster_key"] = str(
        meta.get("cluster_key") or cluster_key_for_text(profile.task_kind, user, meta)
    )
    require_meta(meta, index)
    validate_local_item(user, answer, meta, sample, profile, index)
    return GeneratedItem(
        task_id=f"{sample.sample_id}_q{index:03d}",
        user=user,
        answer=answer,
        meta=meta,
    )


def assign_task_ids(
    sample: SourceSample, items: list[GeneratedItem], start_index: int
) -> list[GeneratedItem]:
    return [
        dataclasses.replace(item, task_id=f"{sample.sample_id}_q{start_index + offset:03d}")
        for offset, item in enumerate(items)
    ]


def require_meta(meta: dict[str, Any], index: int) -> None:
    for key in ("semantic_plan", "validation_contract", "changed_factor"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"item {index} missing meta.{key}")
    if not meta.get("diversity_signature"):
        meta["diversity_signature"] = stable_hash(
            "|".join(
                str(meta.get(key, ""))
                for key in (
                    "task_kind",
                    "semantic_plan",
                    "changed_factor",
                    "validation_contract",
                )
            )
        )


def validate_local_item(
    user: str,
    answer: str,
    meta: dict[str, Any],
    sample: SourceSample,
    profile: PromptProfile,
    index: int,
) -> None:
    answer_style = normalize_answer_style(meta.get("answer_style", profile.answer_style))
    meta["answer_style"] = answer_style

    if answer.startswith("Assistant:"):
        raise ValueError(f"item {index} answer includes assistant role prefix")
    if "```" in answer and profile.task_kind == "function_calling":
        raise ValueError(f"item {index} function_calling answer contains markdown fence")

    reason = benchmark_contamination_reason(answer)
    if reason and not (answer_style == "cot" and reason == "think_marker"):
        raise ValueError(f"item {index} answer rejected for benchmark contamination: {reason}")
    if answer_style != "cot" and contains_cot_marker(user + "\n" + answer):
        raise ValueError(f"item {index} leaks CoT/thinking markers without answer_style=cot")

    expected_style = expected_answer_style(sample, profile)
    if expected_style and not compatible_answer_style(expected_style, answer_style):
        raise ValueError(
            f"item {index} answer_style drift: expected {expected_style}, got {answer_style}"
        )

    if profile.task_kind == "function_calling":
        validate_function_calling_item(user, answer, index)
    elif profile.task_kind == "coding":
        validate_coding_item(user, answer, meta, index)
    elif profile.task_kind == "math":
        validate_math_item(user, answer, answer_style, index)
    elif profile.task_kind == "knowledge":
        validate_knowledge_item(user, answer, answer_style, index)
    elif profile.task_kind == "instruction_following":
        validate_instruction_following_item(user, answer, answer_style, index)


def expected_answer_style(sample: SourceSample, profile: PromptProfile) -> str:
    if source_declares_cot(sample.meta):
        return "cot"
    explicit = sample.meta.get("answer_style")
    if isinstance(explicit, str) and explicit.strip():
        return normalize_answer_style(explicit)
    source_answer = source_answer_text(sample)
    if source_answer:
        return infer_answer_style(profile.task_kind, sample.user, source_answer)
    if profile.task_kind in {"math", "knowledge", "coding", "function_calling"}:
        return profile.answer_style
    return ""


def source_declares_cot(meta: dict[str, Any]) -> bool:
    keys = (
        "cot_mode",
        "score_cot_mode",
        "prompt_profile",
        "cot_profile",
        "answer_style",
    )
    for key in keys:
        if value_declares_cot(meta.get(key)):
            return True
    return value_declares_cot(meta.get("sampling_config"))


def value_declares_cot(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        key = normalize_key(text)
        if key in {"nocot", "no_cot", "no_chain_of_thought"}:
            return False
        if key in {"cot", "fakecot", "fake_cot", "chain_of_thought"}:
            return True
        if re.search(r"\b(fake[_ -]?cot|cot|chain[-_ ]of[-_ ]thought)\b", text, re.I):
            return True
        if text.startswith("{") or text.startswith("["):
            try:
                return value_declares_cot(json.loads(text))
            except json.JSONDecodeError:
                return False
        return False
    if isinstance(value, dict):
        return any(value_declares_cot(item) for item in value.values())
    if isinstance(value, list):
        return any(value_declares_cot(item) for item in value)
    return False


def source_answer_text(sample: SourceSample) -> str:
    for key in ("ref_answer", "answer", "expected_answer"):
        value = sample.meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            return str(value).strip()
    return ""


def compatible_answer_style(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if expected == "json_object" and actual == "function_call":
        return True
    if expected == "function_call" and actual == "json_object":
        return True
    return False


def validate_function_calling_item(user: str, answer: str, index: int) -> None:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as err:
        raise ValueError(f"item {index} function_calling answer is not valid JSON") from err
    calls = payload if isinstance(payload, list) else [payload]
    if not calls:
        raise ValueError(f"item {index} function_calling answer contains no calls")
    tool_names = extract_tool_names(user)
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"item {index} function_calling call {call_index} is not an object")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"item {index} function_calling call {call_index} has no name")
        if not isinstance(arguments, dict):
            raise ValueError(
                f"item {index} function_calling call {call_index} arguments must be an object"
            )
        if tool_names and name not in tool_names:
            raise ValueError(
                f"item {index} function_calling call {call_index} uses unknown tool {name!r}"
            )


def extract_tool_names(user: str) -> set[str]:
    names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', user))
    names.update(re.findall(r"\bFunction name:\s*([A-Za-z_][A-Za-z0-9_]*)", user))
    names.update(re.findall(r"\bfunction:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", user, re.I))
    return names


def validate_coding_item(
    user: str, answer: str, meta: dict[str, Any], index: int
) -> None:
    language = str(meta.get("language", "")).strip().lower()
    looks_python = (
        language == "python"
        or "python" in user.lower()
        or re.search(r"(?m)^\s*(def|class|from|import)\s+", answer) is not None
    )
    if looks_python:
        try:
            ast.parse(answer)
        except SyntaxError as err:
            raise ValueError(f"item {index} python answer has syntax error: {err}") from err


def validate_math_item(user: str, answer: str, answer_style: str, index: int) -> None:
    if answer_style not in {"final_only", "cot"}:
        raise ValueError(f"item {index} math answer must keep source answer style")
    if len(user.strip()) < 50:
        raise ValueError(f"item {index} math prompt is too shallow")
    if len(re.findall(r"\d", user)) < 2 and not re.search(r"[a-zA-Z]\s*[=<>]", user):
        raise ValueError(f"item {index} math prompt lacks enough numeric/algebraic constraints")
    if answer_style == "cot":
        if not contains_cot_marker(answer):
            raise ValueError(f"item {index} math answer_style=cot but answer has no CoT marker")
        return
    if looks_like_explanation(answer):
        raise ValueError(f"item {index} math answer drifted from final answer to explanation")


def validate_knowledge_item(user: str, answer: str, answer_style: str, index: int) -> None:
    if answer_style not in {"final_only", "cot"}:
        raise ValueError(f"item {index} knowledge answer must keep source answer style")
    if len(user.strip()) < 40:
        raise ValueError(f"item {index} knowledge prompt is too shallow")
    if answer_style == "cot":
        if not contains_cot_marker(answer):
            raise ValueError(f"item {index} knowledge answer_style=cot but answer has no CoT marker")
        return
    if looks_like_explanation(answer):
        raise ValueError(f"item {index} knowledge answer drifted from final answer to explanation")


def validate_instruction_following_item(
    user: str, answer: str, answer_style: str, index: int
) -> None:
    if answer_style == "cot":
        if not contains_cot_marker(answer):
            raise ValueError(
                f"item {index} instruction answer_style=cot but answer has no CoT marker"
            )
        return
    lower = user.lower()
    if "json" in lower:
        try:
            json.loads(answer)
        except json.JSONDecodeError as err:
            raise ValueError(f"item {index} instruction answer is not valid JSON") from err
    if (
        "tidak boleh mengandungi koma" in lower
        and re.search(r'dimulakan dengan\s+[^"\']*["\'][^"\']*,[^"\']*["\']', user, re.I)
    ):
        raise ValueError(f"item {index} instruction contains contradictory comma rules")
    if contains_cot_marker(answer):
        raise ValueError(f"item {index} instruction answer leaks CoT markers")


def infer_answer_style(task_kind: str, user: str, answer: str) -> str:
    answer = answer.strip()
    if task_kind == "function_calling":
        return "function_call"
    if contains_cot_marker(answer):
        return "cot"
    if is_json_text(answer):
        return "json_object"
    if task_kind == "coding" or looks_like_code(answer):
        return "code"
    if looks_like_explanation(answer):
        return "brief_explanation"
    return "final_only"


def normalize_answer_style(value: Any) -> str:
    style = normalize_key(str(value or "")).strip()
    aliases = {
        "final": "final_only",
        "short_answer": "final_only",
        "answer_only": "final_only",
        "json": "json_object",
        "tool_call": "function_call",
        "function_calling": "function_call",
        "chain_of_thought": "cot",
    }
    style = aliases.get(style, style)
    if style not in ANSWER_STYLES:
        raise ValueError(f"unknown answer_style {value!r}")
    return style


def is_json_text(text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, (dict, list))


def contains_cot_marker(text: str) -> bool:
    return any(pattern.search(text) for pattern in COT_PATTERNS)


def looks_like_code(text: str) -> bool:
    return re.search(r"(?m)^\s*(def|class|from|import)\s+", text) is not None


def looks_like_explanation(text: str) -> bool:
    stripped = text.strip()
    if "\n" in stripped:
        return True
    return bool(
        re.search(
            r"\b(because|therefore|thus|so|hence|let|count|total|probability|common difference)\b|因为|所以|因此|计算|可得|答案是",
            stripped,
            re.I,
        )
    )


def benchmark_contamination_reason(user: str) -> str:
    for name, pattern in BENCHMARK_WRAPPER_PATTERNS:
        if pattern.search(user):
            return name
    return ""


def validate_with_model(
    client: Any,
    model: ModelConfig,
    profile: PromptProfile,
    sample: SourceSample,
    candidates: list[GeneratedItem],
) -> tuple[list[GeneratedItem], list[str]]:
    if not profile.validation_template:
        return candidates, []
    source_style = expected_answer_style(sample, profile) or profile.answer_style
    payload = [
        {"index": index, "user": item.user, "answer": item.answer, "meta": item.meta}
        for index, item in enumerate(candidates)
    ]
    prompt = render_template(
        profile.validation_template,
        {
            "profile_name": profile.name,
            "task_kind": profile.task_kind,
            "answer_check": profile.answer_check,
            "answer_style": source_style,
            "source_sample_json": json.dumps(
                {"sample_id": sample.sample_id, "user": sample.user, "meta": sample.meta},
                ensure_ascii=False,
                indent=2,
            ),
            "generated_candidates_json": json.dumps(
                payload, ensure_ascii=False, indent=2
            ),
            "synthesis_profile_json": json.dumps(
                {
                    "strategy": profile.strategy,
                    "axes": profile.axes,
                    "answer_style": source_style,
                    "profile_default_answer_style": profile.answer_style,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    )
    raw = client.chat(model, prompt)
    decisions = json.loads(extract_json_object(raw)).get("items", [])
    valid_by_index = {
        int(item.get("index")): bool(item.get("valid"))
        for item in decisions
        if isinstance(item, dict) and "index" in item
    }
    accepted = []
    rejected = []
    for index, item in enumerate(candidates):
        if valid_by_index.get(index, True):
            accepted.append(item)
        else:
            rejected.append(f"validator rejected item {index}")
    return accepted, rejected


def select_diverse_items(
    candidates: list[GeneratedItem],
    profile: PromptProfile,
    seen_signatures: dict[str, int],
    seen_clusters: dict[str, int],
    cluster_limit_per_domain: int,
    limit: int,
) -> tuple[list[GeneratedItem], list[str]]:
    ranked = []
    rejected = []
    seen_users = set()
    for item in candidates:
        user_key = normalize_text(item.user)
        cluster = generated_cluster_key(item, profile)
        signature = diversity_signature(item, profile)
        if user_key in seen_users:
            rejected.append(f"duplicate user: {item.task_id}")
            continue
        seen_users.add(user_key)
        if cluster_limit_per_domain > 0 and seen_clusters.get(cluster, 0) >= cluster_limit_per_domain:
            rejected.append(f"duplicate cluster over limit: {cluster}")
            continue
        if seen_signatures.get(signature, 0) > 0:
            rejected.append(f"duplicate diversity signature: {signature}")
            continue
        score = diversity_score(item, profile, seen_signatures)
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in ranked[:limit]], rejected


def diversity_score(
    item: GeneratedItem, profile: PromptProfile, seen_signatures: dict[str, int]
) -> float:
    signature = diversity_signature(item, profile)
    sparse_bonus = 1.0 / (1.0 + seen_signatures.get(signature, 0))
    filled_axes = sum(1 for axis in profile.axes if item.meta.get(axis))
    contract_bonus = 0.2 if item.meta.get("validation_contract") else 0.0
    return sparse_bonus + filled_axes * 0.05 + contract_bonus


def diversity_signature(item: GeneratedItem, profile: PromptProfile) -> str:
    raw = item.meta.get("diversity_signature")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    parts = [profile.task_kind]
    parts.extend(str(item.meta.get(axis, "")) for axis in profile.axes)
    parts.append(str(item.meta.get("changed_factor", "")))
    return stable_hash("|".join(parts))


def generated_cluster_key(item: GeneratedItem, profile: PromptProfile) -> str:
    raw = item.meta.get("cluster_key")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return cluster_key_for_text(profile.task_kind, item.user, item.meta)


def canonical_sample_domain(meta: dict[str, Any]) -> str:
    for key in ("task_kind", "domain", "profile", "prompt_profile"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            domain = canonical_domain(value)
            if domain in DOMAINS:
                return domain
    return "knowledge"


def cluster_key_for_text(task_kind: str, user: str, meta: dict[str, Any] | None = None) -> str:
    task_kind = canonical_domain(task_kind)
    meta = meta or {}
    explicit = meta.get("cluster_key")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    if task_kind == "function_calling":
        tool_part = ",".join(sorted(extract_tool_names(user)))
        intent = user_request_tail(user)
        basis = f"{task_kind}|tools={tool_part}|intent={cluster_text(intent, 24)}"
    elif task_kind == "coding":
        function_names = ",".join(extract_function_names(user))
        basis = f"{task_kind}|funcs={function_names}|task={cluster_text(user, 28)}"
    else:
        basis = f"{task_kind}|task={cluster_text(strip_choice_options(user), 32)}"
    return f"{task_kind}:{stable_hash(basis)}"


def strip_choice_options(text: str) -> str:
    return re.sub(r"(?m)^\s*[A-J]\.\s+.*$", "", text)


def cluster_text(text: str, max_tokens: int) -> str:
    text = text.lower()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"[^a-z0-9_<>一-龥]+", " ", text)
    tokens = [token for token in text.split() if len(token) > 1]
    return " ".join(tokens[:max_tokens])


def user_request_tail(user: str) -> str:
    matches = list(re.finditer(r"(?im)^User:\s*", user))
    if matches:
        return user[matches[-1].end() :].strip()
    return user


def extract_function_names(text: str) -> list[str]:
    names = set(re.findall(r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    names.update(re.findall(r"\bassert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    names.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))
    return sorted(names)


def item_to_row(item: GeneratedItem, model: ModelConfig) -> dict[str, Any]:
    generated_item_json = json.dumps(
        {"user": item.user, "answer": item.answer, "meta": item.meta},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "task_id": item.task_id,
        "status": "generated",
        "user": item.user,
        "expected_answer": item.answer,
        "generated_item_json": generated_item_json,
        "gen_input_configs": {
            "model_name": model.model_name,
            "max_completion_tokens": model.max_completion_tokens,
            "temperature": model.temperature,
            "json_object_response": model.json_object_response,
        },
    }


def row_to_item(row: dict[str, Any]) -> GeneratedItem | None:
    payload = row.get("generated_item_json")
    if isinstance(payload, str) and payload.strip():
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    user = str(data.get("user") or row.get("user") or "").strip()
    answer = str(data.get("answer") or row.get("expected_answer") or "").strip()
    meta = data.get("meta")
    if not user or not answer or not isinstance(meta, dict):
        return None
    task_id = str(row.get("task_id") or meta.get("task_id") or "").strip()
    if not task_id:
        return None
    return GeneratedItem(task_id=task_id, user=user, answer=answer, meta=dict(meta))


def render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if "{{" in rendered:
        raise ValueError("prompt template contains unresolved placeholders")
    return rendered


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for offset, ch in enumerate(stripped[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : offset + 1]
    raise ValueError("unbalanced JSON object")


def extract_user_from_context(context: str) -> str:
    text = context.strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if isinstance(payload.get("prompt"), str):
                return extract_user_from_context(payload["prompt"])
            stages = payload.get("stages")
            if isinstance(stages, list):
                for stage in stages:
                    if isinstance(stage, dict) and isinstance(stage.get("prompt"), str):
                        return extract_user_from_context(stage["prompt"])
    if text.startswith("User:") and "\nAssistant:" in text:
        return text[len("User:") : text.index("\nAssistant:")].strip()
    return text


def sanitize_user(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("User:"):
        text = text[len("User:") :].strip()
    return text


def sanitize_source_user(text: str) -> str:
    return strip_source_benchmark_wrappers(sanitize_user(text)).strip()


def strip_source_benchmark_wrappers(text: str) -> str:
    text = re.sub(
        r"(?is)^solve the problem using one clean solution path\..*?\(think\)\s*",
        "",
        text.strip(),
        count=1,
    )
    text = re.sub(r"(?is)\nAssistant:\s*```(?:json)?\s*$", "", text).strip()
    text = re.sub(r"(?is)\nAssistant:\s*$", "", text).strip()
    text = re.sub(
        r"(?is)^you are a very talented expert[^\n]*\n+",
        "",
        text,
        count=1,
    )
    text = re.sub(
        r"(?im)^answer this question and finish with a single option letter\.?\s*\n+",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^put the complete final answer inside.*\n+",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^do not restart, repeat, enumerate alternative methods.*\n+",
        "",
        text,
    )
    return re.sub(r"(?im)^\(think\)\s*", "", text).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def canonical_domain(value: str) -> str:
    key = normalize_key(value)
    return key


def default_answer_check(task_kind: str) -> str:
    if task_kind == "function_calling":
        return "json_exact"
    if task_kind in {"coding", "instruction_following"}:
        return "disabled"
    return "exact_text"


def default_answer_style(task_kind: str) -> str:
    if task_kind == "function_calling":
        return "function_call"
    if task_kind == "coding":
        return "code"
    return "final_only"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    items = generate_dataset(config)
    print(f"generated {len(items)} items under {config.output_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
