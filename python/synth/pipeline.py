from __future__ import annotations

import argparse
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

BENCHMARK_WRAPPER_PATTERNS = (
    ("talented_expert_wrapper", re.compile(r"\bYou are a very talented expert\b", re.I)),
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

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    endpoint: str
    model_name: str
    api_key: str = ""
    api_key_env: str = ""
    system_prompt: str = ""
    max_completion_tokens: int = 4096
    temperature: float | None = None
    json_object_response: bool = False
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
        if model.json_object_response:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            model.endpoint,
            data=json.dumps(payload).encode("utf-8"),
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

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("chat response contains no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
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
    )


def model_config_from_raw(raw: dict[str, Any], run_raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        endpoint=str(raw["endpoint"]).strip(),
        model_name=str(raw["model_name"]).strip(),
        api_key=str(raw.get("api_key", "")).strip(),
        api_key_env=str(raw.get("api_key_env", "")).strip(),
        system_prompt=str(raw.get("system_prompt", "")).strip(),
        max_completion_tokens=int(raw.get("max_completion_tokens", 4096)),
        temperature=raw.get("temperature"),
        json_object_response=bool(raw.get("json_object_response", False)),
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
    if config.limit is not None:
        window = window[: config.limit]
    return [normalize_sample(index + config.start_index, row) for index, row in enumerate(window)]


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
        user = sanitize_user(str(row["user"]))
    else:
        context = row.get("context", row.get("text", ""))
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False)
        user = sanitize_user(extract_user_from_context(context))
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
        "domain",
        "subject",
        "dataset",
        "source",
        "answer",
        "ref_answer",
        "option_labels",
    ):
        if key in row and key not in meta:
            meta[key] = row[key]
    return SourceSample(sample_id=sample_id, user=user, meta=meta)


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
    emit_lock = threading.Lock()
    signature_lock = threading.Lock()

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

    if progress_enabled:
        progress_path.write_text("", encoding="utf-8")
    emit_progress(
        "run_start",
        samples=len(samples),
        variant_count=config.variant_count,
        generation_attempts=config.generation_attempts,
        validate_generated_questions=config.validate_generated_questions,
        max_concurrency=config.max_concurrency,
    )

    def process_sample(
        sample_index: int, sample: SourceSample
    ) -> tuple[int, list[GeneratedItem], list[dict[str, Any]]]:
        profile = choose_profile(sample, config.prompt, library)
        emit_progress(
            "sample_start",
            sample_index=sample_index,
            sample_count=len(samples),
            sample_id=sample.sample_id,
            task_kind=profile.task_kind,
            prompt_profile=profile.name,
        )
        accepted: list[GeneratedItem] = []
        local_rejected_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for attempt in range(config.generation_attempts):
            remaining = config.variant_count - len(accepted)
            if remaining <= 0:
                break
            prompt = build_generation_prompt(
                profile,
                sample,
                remaining,
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
                    remaining,
                )
                for item in selected:
                    signature = diversity_signature(item, profile)
                    seen_signatures[signature] = (
                        seen_signatures.get(signature, 0) + 1
                    )
            rejected.extend(diversity_rejected)
            for reason in rejected:
                local_rejected_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task_kind": profile.task_kind,
                        "prompt_profile": profile.name,
                        "attempt": attempt,
                        "reason": reason,
                    }
                )
            accepted.extend(selected)
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
            errors = rejected or ["not enough accepted candidates"]
        accepted = accepted[: config.variant_count]
        emit_progress(
            "sample_done",
            sample_index=sample_index,
            sample_id=sample.sample_id,
            task_kind=profile.task_kind,
            accepted=len(accepted),
        )
        return sample_index, accepted, local_rejected_rows

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
            sample_index, accepted, local_rejected_rows = future.result()
            results[sample_index - 1] = (accepted, local_rejected_rows)

    for result in results:
        if result is None:
            continue
        accepted, local_rejected_rows = result
        accepted_all.extend(accepted)
        rejected_rows.extend(local_rejected_rows)

    write_jsonl(tasks_path, [item_to_row(item, config.generator) for item in accepted_all])
    write_jsonl(rejected_path, rejected_rows)
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
    accepted_json = [
        {"user": item.user, "meta": item.meta} for item in accepted
    ]
    variables = {
        "profile_name": profile.name,
        "task_kind": profile.task_kind,
        "answer_check": profile.answer_check,
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
                "strategy": profile.strategy,
                "axes": profile.axes,
                "subspace_plan": subspace_plan,
                "required_meta": [
                    "semantic_plan",
                    "validation_contract",
                    "changed_factor",
                    "diversity_signature",
                ],
                "selection_policy": "prefer sparse subspaces and non-duplicate diversity signatures",
            },
            ensure_ascii=False,
            indent=2,
        ),
    }
    return render_template(profile.generation_template, variables)


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
    if profile.task_kind == "function_calling" and isinstance(
        raw_answer, (dict, list)
    ):
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
    meta.setdefault("prompt_profile", profile.name)
    meta.setdefault("subspace_key", build_subspace_plan(sample, profile)["subspace_key"])
    meta.setdefault("source_sample_id", sample.sample_id)
    meta.setdefault("generation_stage", "variant_from_failed_sample")
    require_meta(meta, index)
    if profile.task_kind == "function_calling":
        json.loads(answer)
    return GeneratedItem(
        task_id=f"{sample.sample_id}_q{index:03d}",
        user=user,
        answer=answer,
        meta=meta,
    )


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
            "source_sample_json": json.dumps(
                {"sample_id": sample.sample_id, "user": sample.user, "meta": sample.meta},
                ensure_ascii=False,
                indent=2,
            ),
            "generated_candidates_json": json.dumps(
                payload, ensure_ascii=False, indent=2
            ),
            "synthesis_profile_json": json.dumps(
                {"strategy": profile.strategy, "axes": profile.axes},
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
    limit: int,
) -> tuple[list[GeneratedItem], list[str]]:
    ranked = []
    rejected = []
    seen_users = set()
    for item in candidates:
        user_key = normalize_text(item.user)
        signature = diversity_signature(item, profile)
        if user_key in seen_users:
            rejected.append(f"duplicate user: {item.task_id}")
            continue
        seen_users.add(user_key)
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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
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
