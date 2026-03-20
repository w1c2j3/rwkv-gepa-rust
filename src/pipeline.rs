use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dashmap::DashSet;
use futures::stream::{self, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::config::{AppConfig, ModelConfig};
use crate::dataset::load_mmlu_samples;
use crate::jsonl::JsonlWriter;
use crate::openai::OpenAiClient;
use crate::types::{ResponseRecord, SampleRecord, VariantRecord};
use crate::util::{is_valid_option_label, log, normalize_answer, normalize_choices_array};

#[derive(Debug, Clone)]
struct NormalizedVariant {
    question: String,
    choices: Vec<String>,
    answer: String,
}

pub async fn run(config: AppConfig) -> Result<()> {
    let config = Arc::new(config);
    ensure_output_dirs(config.as_ref())?;
    let (mut existing_sample_ids, existing_variant_ids, existing_response_keys) =
        build_resume_state(config.as_ref())?;
    log(format!(
        "Runtime worker_threads={} max_blocking_threads={} | dataset_requests={} rewrite_requests={} answer_requests={}",
        config.runtime.worker_threads,
        config.runtime.max_blocking_threads,
        config.concurrency.dataset_requests,
        config.concurrency.rewrite_requests,
        config.concurrency.answer_requests,
    ));

    let samples = load_mmlu_samples(config.as_ref()).await?;
    if samples.is_empty() {
        log("No samples selected. Nothing to do.");
        return Ok(());
    }

    let original_writer = JsonlWriter::open(
        &config.output.original_jsonl,
        config.run.resume,
        config.concurrency.original_writer_capacity,
    )
    .await?;
    let original_sender = original_writer.sender();
    let mut written_originals = 0usize;
    for sample in &samples {
        if existing_sample_ids.insert(sample.sample_id.clone()) {
            original_sender.send(sample.clone()).await?;
            written_originals += 1;
        }
    }
    // Close the last producer clone before awaiting the writer task.
    drop(original_sender);
    original_writer.close().await?;
    log(format!(
        "Wrote {written_originals} new original samples to {}",
        config.output.original_jsonl.display()
    ));

    let variant_ids = Arc::new(DashSet::new());
    for variant_id in existing_variant_ids {
        variant_ids.insert(variant_id);
    }
    let rewrite_failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let rewrite_client = Arc::new(OpenAiClient::new(
        &config.rewrite.model.base_url,
        &config.rewrite.model.api_key,
        &config.run,
        &config.concurrency,
    )?);

    log("Starting rewrite stage");
    let variants_writer = JsonlWriter::open(
        &config.output.variants_jsonl,
        config.run.resume,
        config.concurrency.variant_writer_capacity,
    )
    .await?;
    let variants_sender = variants_writer.sender();
    stream::iter(samples.iter().cloned())
        .for_each_concurrent(config.concurrency.rewrite_requests, |sample| {
            let config = Arc::clone(&config);
            let rewrite_client = rewrite_client.clone();
            let variants_sender = variants_sender.clone();
            let variant_ids = variant_ids.clone();
            let rewrite_failures = rewrite_failures.clone();
            async move {
                if let Err(error) = rewrite_single_sample(
                    sample.clone(),
                    config.as_ref(),
                    &rewrite_client,
                    &variants_sender,
                    &variant_ids,
                )
                .await
                {
                    log(format!(
                        "Rewrite failed for {}: {error:#}",
                        sample.sample_id
                    ));
                    rewrite_failures.lock().await.push(sample.sample_id);
                }
            }
        })
        .await;
    // The background writer exits only after every sender clone is dropped.
    drop(variants_sender);
    variants_writer.close().await?;

    let allowed_sample_ids = samples
        .iter()
        .map(|sample| sample.sample_id.clone())
        .collect::<FxHashSet<_>>();
    let variants = load_variant_records(&config.output.variants_jsonl, &allowed_sample_ids)?;
    if variants.is_empty() {
        log("No variants available after rewrite stage. Skipping answer stage.");
        return Ok(());
    }

    let distribution = summarize_distribution(&variants, config.as_ref())?;
    log(format!(
        "Variant distribution across answer models: {}",
        serde_json::to_string(&distribution)?
    ));

    let response_keys = Arc::new(DashSet::new());
    for response_key in existing_response_keys {
        response_keys.insert(response_key);
    }
    let answer_failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let clients = Arc::new(build_answer_clients(config.as_ref())?);
    let variants_per_model = config.rewrite.variant_count / config.answer_models.len();

    log("Starting answer stage");
    let responses_writer = JsonlWriter::open(
        &config.output.responses_jsonl,
        config.run.resume,
        config.concurrency.response_writer_capacity,
    )
    .await?;
    let responses_sender = responses_writer.sender();
    stream::iter(variants.into_iter())
        .for_each_concurrent(config.concurrency.answer_requests, |variant| {
            let config = Arc::clone(&config);
            let clients = clients.clone();
            let responses_sender = responses_sender.clone();
            let response_keys = response_keys.clone();
            let answer_failures = answer_failures.clone();
            async move {
                if let Err(error) = answer_single_variant(
                    variant.clone(),
                    config.as_ref(),
                    &clients,
                    &responses_sender,
                    &response_keys,
                    variants_per_model,
                )
                .await
                {
                    log(format!(
                        "Answer failed for {}: {error:#}",
                        variant.variant_id
                    ));
                    answer_failures.lock().await.push(variant.variant_id);
                }
            }
        })
        .await;
    // Keep the same shutdown rule for the responses writer.
    drop(responses_sender);
    responses_writer.close().await?;

    let rewrite_failures = rewrite_failures.lock().await.clone();
    let answer_failures = answer_failures.lock().await.clone();
    log(format!(
        "Run finished with {} rewrite failures and {} answer failures",
        rewrite_failures.len(),
        answer_failures.len()
    ));
    if !rewrite_failures.is_empty() {
        log(format!("Rewrite failures: {}", rewrite_failures.join(", ")));
    }
    if !answer_failures.is_empty() {
        log(format!("Answer failures: {}", answer_failures.join(", ")));
    }

    Ok(())
}

async fn rewrite_single_sample(
    sample: SampleRecord,
    config: &AppConfig,
    client: &OpenAiClient,
    writer: &crate::jsonl::JsonlSender<VariantRecord>,
    existing_variant_ids: &DashSet<String>,
) -> Result<()> {
    let missing_variant_ids = (0..config.rewrite.variant_count)
        .map(|rewrite_index| format!("{}_v{rewrite_index:03}", sample.sample_id))
        .filter(|variant_id| !existing_variant_ids.contains(variant_id))
        .collect::<Vec<_>>();
    if missing_variant_ids.is_empty() {
        return Ok(());
    }

    let prompt = build_rewrite_prompt(
        &sample.question,
        &sample.choices,
        &sample.answer,
        config.rewrite.variant_count,
    );

    let mut last_error: Option<anyhow::Error> = None;
    let mut variants = None;
    for attempt in 0..=config.rewrite.parse_retry_times {
        match client
            .chat_completion(
                &config.rewrite.model.name,
                &prompt,
                &config.run,
                config.rewrite.model.enable_thinking,
                config.rewrite.model.reasoning_effort.as_deref(),
            )
            .await
            .and_then(|result| parse_variants(&result.content, config.rewrite.variant_count))
        {
            Ok(parsed) => {
                variants = Some(parsed);
                break;
            }
            Err(error) => {
                if attempt < config.rewrite.parse_retry_times {
                    log(format!(
                        "Rewrite parse failed for {}, retrying ({}/{}): {error:#}",
                        sample.sample_id,
                        attempt + 1,
                        config.rewrite.parse_retry_times
                    ));
                }
                last_error = Some(error);
            }
        }
    }

    let variants = variants.ok_or_else(|| {
        last_error.unwrap_or_else(|| anyhow!("rewrite failed without specific error"))
    })?;

    for (rewrite_index, variant) in variants.into_iter().enumerate() {
        let variant_id = format!("{}_v{rewrite_index:03}", sample.sample_id);
        if !existing_variant_ids.insert(variant_id.clone()) {
            continue;
        }
        writer
            .send(VariantRecord {
                sample_id: sample.sample_id.clone(),
                variant_id,
                subject: sample.subject.clone(),
                rewrite_model: config.rewrite.model.name.clone(),
                question: variant.question,
                choices: variant.choices,
                answer: variant.answer,
            })
            .await?;
    }

    Ok(())
}

async fn answer_single_variant(
    variant: VariantRecord,
    config: &AppConfig,
    clients: &FxHashMap<String, OpenAiClient>,
    writer: &crate::jsonl::JsonlSender<ResponseRecord>,
    response_keys: &DashSet<(String, String)>,
    variants_per_model: usize,
) -> Result<()> {
    let rewrite_index = parse_variant_index(&variant.variant_id)?;
    let model = choose_answer_model(rewrite_index, &config.answer_models, variants_per_model)?;
    let response_key = (variant.variant_id.clone(), model.name.clone());
    if response_keys.contains(&response_key) {
        return Ok(());
    }

    let client = clients
        .get(&model.name)
        .ok_or_else(|| anyhow!("Missing client for answer model {}", model.name))?;

    let prompt = build_prompt(&variant.question, &variant.choices);
    let result = client
        .chat_completion(
            &model.name,
            &prompt,
            &config.run,
            model.enable_thinking,
            model.reasoning_effort.as_deref(),
        )
        .await?;

    writer
        .send(ResponseRecord {
            sample_id: variant.sample_id.clone(),
            variant_id: variant.variant_id.clone(),
            subject: variant.subject.clone(),
            rewrite_model: variant.rewrite_model.clone(),
            answer_model: model.name.clone(),
            prompt,
            answer: variant.answer.clone(),
            model_reasoning: result.reasoning,
            model_response: result.content,
        })
        .await?;
    response_keys.insert(response_key);

    Ok(())
}

fn ensure_output_dirs(config: &AppConfig) -> Result<()> {
    fs::create_dir_all(&config.output.dir).with_context(|| {
        format!(
            "Failed to create output directory {}",
            config.output.dir.display()
        )
    })?;
    for path in [
        &config.output.original_jsonl,
        &config.output.variants_jsonl,
        &config.output.responses_jsonl,
    ] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent directory {}", parent.display())
            })?;
        }
    }
    Ok(())
}

fn build_resume_state(
    config: &AppConfig,
) -> Result<(
    FxHashSet<String>,
    FxHashSet<String>,
    FxHashSet<(String, String)>,
)> {
    if !config.run.resume {
        return Ok((
            FxHashSet::default(),
            FxHashSet::default(),
            FxHashSet::default(),
        ));
    }

    let mut sample_ids = FxHashSet::default();
    read_jsonl_lines(&config.output.original_jsonl, |value| {
        if let Some(sample_id) = value
            .as_object()
            .and_then(|object| object.get("sample_id"))
            .and_then(Value::as_str)
        {
            sample_ids.insert(sample_id.to_owned());
        }
        Ok(())
    })?;

    let mut variant_ids = FxHashSet::default();
    read_jsonl_lines(&config.output.variants_jsonl, |value| {
        if let Ok(record) = variant_record_from_value(&value) {
            variant_ids.insert(record.variant_id);
        }
        Ok(())
    })?;

    let mut response_keys = FxHashSet::default();
    read_jsonl_lines(&config.output.responses_jsonl, |value| {
        let Some(object) = value.as_object() else {
            return Ok(());
        };
        let Some(variant_id) = object.get("variant_id").and_then(Value::as_str) else {
            return Ok(());
        };
        let Some(answer_model) = object.get("answer_model").and_then(Value::as_str) else {
            return Ok(());
        };
        response_keys.insert((variant_id.to_owned(), answer_model.to_owned()));
        Ok(())
    })?;

    Ok((sample_ids, variant_ids, response_keys))
}

fn read_jsonl_lines(path: &Path, mut handler: impl FnMut(Value) -> Result<()>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }

    let file = File::open(path)
        .with_context(|| format!("Failed to open JSONL file {}", path.display()))?;
    let reader = BufReader::new(file);
    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "Failed to read line {} from {}",
                line_number + 1,
                path.display()
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "Invalid JSON at {} line {}",
                path.display(),
                line_number + 1
            )
        })?;
        handler(value)?;
    }
    Ok(())
}

fn build_rewrite_prompt(
    question: &str,
    choices: &[String],
    answer: &str,
    variant_count: usize,
) -> String {
    let original_payload = json!({
        "question": question,
        "choices": choices,
        "answer": answer,
    });
    format!(
        "You are generating training data variants.\n\n\
Carefully read the original multiple-choice question below.\n\
Generate {variant_count} related multiple-choice variants suitable for model training.\n\n\
Requirements:\n\
1. Keep them semantically related to the original question.\n\
2. You may change numbers, entities, wording, scenario, or reasoning path.\n\
3. Preserve roughly similar difficulty.\n\
4. Each variant must be self-contained and unambiguous.\n\
5. Every variant must have exactly 4 choices.\n\
6. The answer must be a single uppercase letter: A, B, C, or D.\n\
7. Return only valid JSON. Do not use markdown fences. Do not include explanations.\n\
8. Output format:\n\
[\n\
  {{\"question\": \"...\", \"choices\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": \"A\"}},\n\
  {{\"question\": \"...\", \"choices\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": \"B\"}}\n\
]\n\n\
Original sample:\n{}",
        serde_json::to_string_pretty(&original_payload)
            .unwrap_or_else(|_| original_payload.to_string())
    )
}

fn build_prompt(question: &str, choices: &[String]) -> String {
    let mut lines = Vec::with_capacity(choices.len() + 1);
    lines.push(format!("Question: {}", question.trim()));
    for (index, choice) in choices.iter().enumerate() {
        lines.push(format!(
            "{}. {}",
            crate::util::index_to_option_label(index),
            choice.trim()
        ));
    }
    lines.join("\n")
}

fn parse_variants(text: &str, expected_count: usize) -> Result<Vec<NormalizedVariant>> {
    let payload = match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(_) => {
            let start = text
                .find('[')
                .ok_or_else(|| anyhow!("Model response does not contain a JSON array"))?;
            let end = text
                .rfind(']')
                .ok_or_else(|| anyhow!("Model response does not contain a JSON array"))?;
            if end <= start {
                bail!("Model response does not contain a valid JSON array");
            }
            serde_json::from_str::<Value>(&text[start..=end])
                .with_context(|| "Failed to parse extracted JSON array")?
        }
    };

    let items = match payload {
        Value::Array(items) => items,
        Value::Object(object) => vec![Value::Object(object)],
        _ => bail!("Model response is not a JSON array"),
    };
    if items.len() != expected_count {
        bail!("Expected {expected_count} variants, got {}", items.len());
    }

    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            normalize_variant_value(item)
                .with_context(|| format!("Variant at index {index} is invalid"))
        })
        .collect()
}

fn normalize_variant_value(value: Value) -> Result<NormalizedVariant> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("Variant must be a JSON object"))?;

    let question = required_string_field(object, "question")?;
    let choices_value = object
        .get("choices")
        .ok_or_else(|| anyhow!("Variant is missing choices"))?;
    let choices = match choices_value {
        Value::Array(items) => normalize_choices_array(items, "choices")?,
        _ => bail!("Variant choices must be an array"),
    };
    if choices.len() != 4 {
        bail!("Variant must contain exactly 4 choices");
    }

    let answer = object
        .get("answer")
        .map(|value| normalize_answer(value, &choices))
        .ok_or_else(|| anyhow!("Variant is missing answer"))?;
    if !is_valid_option_label(&answer, choices.len()) {
        bail!("Variant answer must normalize to one of A/B/C/D");
    }

    Ok(NormalizedVariant {
        question,
        choices,
        answer,
    })
}

fn load_variant_records(
    path: &Path,
    allowed_sample_ids: &FxHashSet<String>,
) -> Result<Vec<VariantRecord>> {
    let mut deduped = FxHashMap::default();
    read_jsonl_lines(path, |value| {
        if let Ok(record) = variant_record_from_value(&value) {
            if allowed_sample_ids.contains(&record.sample_id) {
                deduped.entry(record.variant_id.clone()).or_insert(record);
            }
        }
        Ok(())
    })?;

    let mut records = deduped.into_values().collect::<Vec<_>>();
    records.sort_by(|left, right| left.variant_id.cmp(&right.variant_id));
    Ok(records)
}

fn variant_record_from_value(value: &Value) -> Result<VariantRecord> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("Variant record must be a JSON object"))?;

    let sample_id = required_string_field(object, "sample_id")?;
    let variant_id = required_string_field(object, "variant_id")?;
    let subject = required_string_field(object, "subject")?;
    let rewrite_model = required_string_field(object, "rewrite_model")?;
    let question = required_string_field(object, "question")?;
    let choices = match object.get("choices") {
        Some(Value::Array(items)) => normalize_choices_array(items, "choices")?,
        _ => bail!("Variant record choices must be an array"),
    };
    if choices.len() != 4 {
        bail!("Variant record must contain exactly 4 choices");
    }
    let answer = object
        .get("answer")
        .map(|value| normalize_answer(value, &choices))
        .ok_or_else(|| anyhow!("Variant record is missing answer"))?;
    if !is_valid_option_label(&answer, choices.len()) {
        bail!("Variant record answer must normalize to one of A/B/C/D");
    }

    Ok(VariantRecord {
        sample_id,
        variant_id,
        subject,
        rewrite_model,
        question,
        choices,
        answer,
    })
}

fn required_string_field(object: &Map<String, Value>, key: &str) -> Result<String> {
    let value = object
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow!("{key} must be a non-empty string"))?;
    Ok(value.to_owned())
}

fn build_answer_clients(config: &AppConfig) -> Result<FxHashMap<String, OpenAiClient>> {
    let mut clients = FxHashMap::default();
    for model in &config.answer_models {
        clients.insert(
            model.name.clone(),
            OpenAiClient::new(
                &model.base_url,
                &model.api_key,
                &config.run,
                &config.concurrency,
            )?,
        );
    }
    Ok(clients)
}

fn parse_variant_index(variant_id: &str) -> Result<usize> {
    let (_, suffix) = variant_id
        .rsplit_once("_v")
        .ok_or_else(|| anyhow!("Invalid variant_id format: {variant_id}"))?;
    suffix
        .parse::<usize>()
        .with_context(|| format!("Invalid variant_id format: {variant_id}"))
}

fn choose_answer_model<'a>(
    rewrite_index: usize,
    answer_models: &'a [ModelConfig],
    variants_per_model: usize,
) -> Result<&'a ModelConfig> {
    let model_index = rewrite_index / variants_per_model;
    answer_models
        .get(model_index)
        .ok_or_else(|| anyhow!("No answer model mapped for rewrite index {rewrite_index}"))
}

fn summarize_distribution(
    variants: &[VariantRecord],
    config: &AppConfig,
) -> Result<BTreeMap<String, usize>> {
    let variants_per_model = config.rewrite.variant_count / config.answer_models.len();
    let mut counts = BTreeMap::new();
    for variant in variants {
        let rewrite_index = parse_variant_index(&variant.variant_id)?;
        let model = choose_answer_model(rewrite_index, &config.answer_models, variants_per_model)?;
        *counts.entry(model.name.clone()).or_insert(0) += 1;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc as StdArc, Mutex as StdMutex, mpsc};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::config::{
        AppConfig, ConcurrencyConfig, MmluConfig, ModelConfig, OutputConfig, RewriteConfig,
        RunConfig, RuntimeConfig,
    };

    fn test_config() -> AppConfig {
        AppConfig {
            mmlu: MmluConfig {
                dataset_name: "cais/mmlu".to_owned(),
                subset: "all".to_owned(),
                split: "test".to_owned(),
                rows_api_url: "https://datasets-server.huggingface.co/rows".to_owned(),
                local_jsonl_path: None,
                limit: 1,
                start_index: 0,
            },
            rewrite: RewriteConfig {
                model: ModelConfig {
                    name: "rewrite-model".to_owned(),
                    base_url: "https://api.example.com/v1".to_owned(),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
                variant_count: 4,
                parse_retry_times: 1,
            },
            answer_models: vec![
                ModelConfig {
                    name: "answer-a".to_owned(),
                    base_url: "https://api.example.com/v1".to_owned(),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
                ModelConfig {
                    name: "answer-b".to_owned(),
                    base_url: "https://api.example.com/v1".to_owned(),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
            ],
            output: OutputConfig {
                dir: PathBuf::from("data"),
                original_jsonl: PathBuf::from("data/original.jsonl"),
                variants_jsonl: PathBuf::from("data/variants.jsonl"),
                responses_jsonl: PathBuf::from("data/responses.jsonl"),
            },
            run: RunConfig {
                resume: true,
                request_timeout_seconds: 30.0,
                max_retries: 1,
                retry_backoff_seconds: 1.0,
            },
            concurrency: ConcurrencyConfig {
                dataset_requests: 2,
                dataset_page_size: 16,
                rewrite_requests: 4,
                answer_requests: 8,
                original_writer_capacity: 16,
                variant_writer_capacity: 16,
                response_writer_capacity: 16,
                http_max_idle_per_host: 16,
            },
            runtime: RuntimeConfig {
                worker_threads: 2,
                max_blocking_threads: 8,
            },
        }
    }

    #[test]
    fn parse_variants_accepts_wrapped_json_array() {
        let variants = parse_variants(
            r#"
            leading text
            [
              {"question": "Q1", "choices": ["A1", "B1", "C1", "D1"], "answer": "c"},
              {"question": "Q2", "choices": ["A2", "B2", "C2", "D2"], "answer": 0}
            ]
            trailing text
            "#,
            2,
        )
        .expect("variants should parse");

        assert_eq!(variants[0].answer, "C");
        assert_eq!(variants[1].answer, "A");
    }

    #[test]
    fn summarize_distribution_matches_variant_buckets() {
        let config = test_config();
        let variants = vec![
            VariantRecord {
                sample_id: "mmlu_000001".to_owned(),
                variant_id: "mmlu_000001_v000".to_owned(),
                subject: "subject".to_owned(),
                rewrite_model: "rewrite-model".to_owned(),
                question: "Q1".to_owned(),
                choices: vec![
                    "A".to_owned(),
                    "B".to_owned(),
                    "C".to_owned(),
                    "D".to_owned(),
                ],
                answer: "A".to_owned(),
            },
            VariantRecord {
                sample_id: "mmlu_000001".to_owned(),
                variant_id: "mmlu_000001_v001".to_owned(),
                subject: "subject".to_owned(),
                rewrite_model: "rewrite-model".to_owned(),
                question: "Q2".to_owned(),
                choices: vec![
                    "A".to_owned(),
                    "B".to_owned(),
                    "C".to_owned(),
                    "D".to_owned(),
                ],
                answer: "B".to_owned(),
            },
            VariantRecord {
                sample_id: "mmlu_000001".to_owned(),
                variant_id: "mmlu_000001_v002".to_owned(),
                subject: "subject".to_owned(),
                rewrite_model: "rewrite-model".to_owned(),
                question: "Q3".to_owned(),
                choices: vec![
                    "A".to_owned(),
                    "B".to_owned(),
                    "C".to_owned(),
                    "D".to_owned(),
                ],
                answer: "C".to_owned(),
            },
            VariantRecord {
                sample_id: "mmlu_000001".to_owned(),
                variant_id: "mmlu_000001_v003".to_owned(),
                subject: "subject".to_owned(),
                rewrite_model: "rewrite-model".to_owned(),
                question: "Q4".to_owned(),
                choices: vec![
                    "A".to_owned(),
                    "B".to_owned(),
                    "C".to_owned(),
                    "D".to_owned(),
                ],
                answer: "D".to_owned(),
            },
        ];

        let distribution = summarize_distribution(&variants, &config).expect("should summarize");
        assert_eq!(distribution.get("answer-a"), Some(&2));
        assert_eq!(distribution.get("answer-b"), Some(&2));
    }

    #[test]
    fn invalid_variant_id_is_rejected() {
        assert!(parse_variant_index("mmlu_000001").is_err());
        assert!(parse_variant_index("mmlu_000001_vabc").is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_server_smoke_test_covers_full_pipeline() {
        let responses = StdArc::new(StdMutex::new(VecDeque::from([
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": serde_json::to_string(&vec![
                            serde_json::json!({
                                "question": "Variant question 1",
                                "choices": ["A1", "B1", "C1", "D1"],
                                "answer": "A"
                            }),
                            serde_json::json!({
                                "question": "Variant question 2",
                                "choices": ["A2", "B2", "C2", "D2"],
                                "answer": "B"
                            })
                        ])
                        .expect("rewrite payload")
                    }
                }]
            }),
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "Answer from model A",
                        "reasoning_content": "Reasoning A"
                    }
                }]
            }),
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "Answer from model B",
                        "reasoning_content": "Reasoning B"
                    }
                }]
            }),
        ])));

        let server = match TestServer::start({
            let responses = responses.clone();
            move |request| match (request.method.as_str(), request.path.as_str()) {
                ("GET", path) if path.starts_with("/rows?") => {
                    ResponseSpec::json(serde_json::json!({
                        "rows": [{
                            "row": {
                                "question": "Original question",
                                "choices": ["Alpha", "Beta", "Gamma", "Delta"],
                                "answer": 1,
                                "subject": "business_ethics"
                            }
                        }]
                    }))
                }
                ("POST", "/v1/chat/completions") => {
                    let payload: Value =
                        serde_json::from_str(&request.body).expect("request body should be JSON");
                    let model = payload
                        .get("model")
                        .and_then(Value::as_str)
                        .expect("model should be present");
                    assert!(
                        ["rewrite-model", "answer-a", "answer-b"].contains(&model),
                        "unexpected model {model}"
                    );
                    let next = responses
                        .lock()
                        .expect("response queue lock")
                        .pop_front()
                        .expect("response queue should not be empty");
                    ResponseSpec::json(next)
                }
                _ => ResponseSpec::status(404, serde_json::json!({"error": "not found"})),
            }
        }) {
            Ok(server) => server,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("test server should start: {error}"),
        };

        let temp_dir = temp_dir("pipeline-smoke");
        let server_url = format!("http://{}", server.addr());
        let config = AppConfig {
            mmlu: MmluConfig {
                dataset_name: "local/mmlu".to_owned(),
                subset: "all".to_owned(),
                split: "test".to_owned(),
                rows_api_url: format!("{server_url}/rows"),
                local_jsonl_path: None,
                limit: 1,
                start_index: 0,
            },
            rewrite: RewriteConfig {
                model: ModelConfig {
                    name: "rewrite-model".to_owned(),
                    base_url: format!("{server_url}/v1"),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
                variant_count: 2,
                parse_retry_times: 1,
            },
            answer_models: vec![
                ModelConfig {
                    name: "answer-a".to_owned(),
                    base_url: format!("{server_url}/v1"),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
                ModelConfig {
                    name: "answer-b".to_owned(),
                    base_url: format!("{server_url}/v1"),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
            ],
            output: OutputConfig {
                dir: temp_dir.clone(),
                original_jsonl: temp_dir.join("original.jsonl"),
                variants_jsonl: temp_dir.join("variants.jsonl"),
                responses_jsonl: temp_dir.join("responses.jsonl"),
            },
            run: RunConfig {
                resume: false,
                request_timeout_seconds: 10.0,
                max_retries: 0,
                retry_backoff_seconds: 0.1,
            },
            concurrency: ConcurrencyConfig {
                dataset_requests: 2,
                dataset_page_size: 16,
                rewrite_requests: 2,
                answer_requests: 4,
                original_writer_capacity: 8,
                variant_writer_capacity: 8,
                response_writer_capacity: 8,
                http_max_idle_per_host: 8,
            },
            runtime: RuntimeConfig {
                worker_threads: 2,
                max_blocking_threads: 4,
            },
        };

        run(config).await.expect("pipeline run should succeed");

        let original_lines = read_jsonl_lines(&temp_dir.join("original.jsonl"));
        let variant_lines = read_jsonl_lines(&temp_dir.join("variants.jsonl"));
        let response_lines = read_jsonl_lines(&temp_dir.join("responses.jsonl"));

        assert_eq!(original_lines.len(), 1);
        assert_eq!(variant_lines.len(), 2);
        assert_eq!(response_lines.len(), 2);
        assert!(
            response_lines
                .iter()
                .any(|line| line.contains("\"answer_model\":\"answer-a\""))
        );
        assert!(
            response_lines
                .iter()
                .any(|line| line.contains("\"answer_model\":\"answer-b\""))
        );

        drop(server);
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rwkv-gepa-rust-v1-{name}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("temp dir should be creatable");
        path
    }

    fn read_jsonl_lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .expect("jsonl file should exist")
            .lines()
            .map(ToOwned::to_owned)
            .collect()
    }

    struct TestServer {
        addr: SocketAddr,
        shutdown_tx: mpsc::Sender<()>,
        join: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn start(
            handler: impl Fn(TestRequest) -> ResponseSpec + Send + Sync + 'static,
        ) -> std::io::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            listener
                .set_nonblocking(true)
                .expect("listener should become nonblocking");
            let addr = listener.local_addr().expect("listener addr");
            let handler = StdArc::new(handler);
            let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();

            let join = thread::spawn(move || {
                loop {
                    if shutdown_rx.try_recv().is_ok() {
                        break;
                    }
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let handler = handler.clone();
                            thread::spawn(move || handle_connection(stream, handler));
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });

            Ok(Self {
                addr,
                shutdown_tx,
                join: Some(join),
            })
        }

        fn addr(&self) -> SocketAddr {
            self.addr
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            let _ = self.shutdown_tx.send(());
            let _ = TcpStream::connect(self.addr);
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }

    #[derive(Debug)]
    struct TestRequest {
        method: String,
        path: String,
        body: String,
    }

    struct ResponseSpec {
        status_code: u16,
        body: String,
    }

    impl ResponseSpec {
        fn json(body: Value) -> Self {
            Self {
                status_code: 200,
                body: serde_json::to_string(&body).expect("response body should serialize"),
            }
        }

        fn status(status_code: u16, body: Value) -> Self {
            Self {
                status_code,
                body: serde_json::to_string(&body).expect("response body should serialize"),
            }
        }
    }

    fn handle_connection(
        mut stream: TcpStream,
        handler: StdArc<dyn Fn(TestRequest) -> ResponseSpec + Send + Sync>,
    ) {
        let request = read_request(&mut stream).expect("request should parse");
        let response = handler(request);
        write_response(&mut stream, response).expect("response should write");
    }

    fn read_request(stream: &mut TcpStream) -> std::io::Result<TestRequest> {
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 4096];
        let header_end;
        loop {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before headers",
                ));
            }
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(position) = find_header_end(&buffer) {
                header_end = position;
                break;
            }
        }

        let headers = String::from_utf8_lossy(&buffer[..header_end]);
        let mut lines = headers.lines();
        let request_line = lines.next().unwrap_or_default();
        let mut request_line_parts = request_line.split_whitespace();
        let method = request_line_parts.next().unwrap_or_default().to_owned();
        let path = request_line_parts.next().unwrap_or_default().to_owned();
        let content_length = lines
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);

        while buffer.len() < header_end + 4 + content_length {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
        }

        let body_start = header_end + 4;
        let body_end = body_start + content_length;
        let body = if buffer.len() >= body_end {
            String::from_utf8_lossy(&buffer[body_start..body_end]).into_owned()
        } else {
            String::new()
        };

        Ok(TestRequest { method, path, body })
    }

    fn write_response(stream: &mut TcpStream, response: ResponseSpec) -> std::io::Result<()> {
        let status_text = match response.status_code {
            200 => "OK",
            404 => "Not Found",
            _ => "OK",
        };
        let payload = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.status_code,
            status_text,
            response.body.len(),
            response.body
        );
        stream.write_all(payload.as_bytes())?;
        stream.flush()?;
        Ok(())
    }

    fn find_header_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }
}
