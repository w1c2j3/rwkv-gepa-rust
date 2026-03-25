use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, Subcommand};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use reqwest::StatusCode;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use tokio::io::AsyncWriteExt;
use tokio::process::Command as TokioCommand;

#[rustfmt::skip]
#[derive(Parser)]
#[command(author, version, about = "Generic JSONL -> RWKV training data pipeline")]
struct Cli { #[command(subcommand)] command: Command }

#[rustfmt::skip]
#[derive(Subcommand)]
enum Command {
    Synthesize { #[arg(long, default_value = "config.toml")] config: PathBuf, #[arg(long)] limit: Option<usize> },
    Export { #[arg(long, default_value = "data/rwkv_train.jsonl")] input: PathBuf, #[arg(long, default_value = "data/rwkv_train.parquet")] output: PathBuf },
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
struct Config {
    input: Input,
    generator: Generator,
    answer_models: Vec<Model>,
    #[serde(default)]
    output: Output,
    #[serde(default)]
    run: Run,
    #[serde(default)]
    concurrency: Concurrency,
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
struct Input {
    local_jsonl_path: PathBuf,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    start_index: usize,
    #[serde(default = "default_subject")]
    default_subject: String,
    #[serde(default = "default_sample_id_prefix")]
    sample_id_prefix: String,
    #[serde(default)]
    sample_id_paths: Vec<String>,
    #[serde(default = "default_sample_id_joiner")]
    sample_id_joiner: String,
    #[serde(default)]
    sample_id_path: Option<String>,
    #[serde(default)]
    subject_paths: Vec<String>,
    #[serde(default = "default_subject_joiner")]
    subject_joiner: String,
    #[serde(default)]
    subject_path: Option<String>,
    prompt_paths: Vec<String>,
    ref_answer_path: String,
    #[serde(default = "default_prompt_joiner")]
    prompt_joiner: String,
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
struct Generator {
    #[serde(flatten)]
    model: Model,
    question_count: usize,
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
struct Model {
    #[serde(alias = "model")]
    name: String,
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    enable_thinking: bool,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    reasoning_effort: Option<String>,
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
#[serde(default)]
struct Output { jsonl: PathBuf }

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
#[serde(default)]
struct Run {
    resume: bool,
    request_timeout_seconds: f64,
    disable_env_proxy: bool,
    force_http1: bool,
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
#[serde(default)]
struct Concurrency {
    generator_requests: usize,
    answer_requests: usize,
}

#[rustfmt::skip]
#[derive(Clone)]
struct SourceSample {
    sample_id: String,
    subject: String,
    prompt: String,
    ref_answer: String,
}

#[rustfmt::skip]
#[derive(Clone)]
struct PendingRow {
    record_id: String,
    sample_id: String,
    subject: String,
    prompt: String,
    ref_answer: String,
    generator_model: String,
    user: String,
    generator_usage: UsageStats,
}

#[rustfmt::skip]
#[derive(Clone, Serialize, Deserialize)]
struct TrainingRow {
    generator_model: String,
    answer_model: String,
    user: String,
    assistant: String,
    record_id: String,
    sample_id: String,
    subject: String,
    prompt: String,
    ref_answer: String,
    #[serde(default, skip_serializing_if = "UsageStats::is_empty")]
    generator_usage: UsageStats,
    #[serde(default, skip_serializing_if = "UsageStats::is_empty")]
    answer_usage: UsageStats,
}

#[rustfmt::skip]
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
struct UsageStats {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_tokens: Option<u64>,
}

#[rustfmt::skip]
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 1],
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

#[rustfmt::skip]
#[derive(Serialize)]
struct ChatMessage<'a> { role: &'a str, content: &'a str }

#[rustfmt::skip]
#[derive(Clone)]
struct OpenAiClient { endpoint: String, api_key: String }

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ChatResult {
    content: String,
    reasoning: Option<String>,
    usage: UsageStats,
}

#[derive(Debug)]
struct ChatError {
    kind: ChatErrorKind,
    message: String,
}

#[derive(Debug)]
enum ChatErrorKind {
    Http,
    Api { status: StatusCode, body: String },
    Parse,
}

struct GeneratedBatch {
    rows: Vec<PendingRow>,
    usage: UsageStats,
}

const CURL_HTTP_STATUS_MARKER: &str = "\n__CURL_HTTP_STATUS__:";

#[tokio::main]
#[rustfmt::skip]
async fn main() -> ExitCode {
    match match Cli::parse().command {
        Command::Synthesize { config, limit } => synthesize(&config, limit).await,
        Command::Export { input, output } => export_jsonl(&input, &output),
    } {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => { eprintln!("{err:#}"); ExitCode::FAILURE }
    }
}

#[rustfmt::skip]
async fn synthesize(path: &Path, limit: Option<usize>) -> Result<()> {
    let mut cfg = load_config(path)?;
    if limit.is_some() {
        cfg.input.limit = limit;
    }
    ensure!(!cfg.answer_models.is_empty(), "need at least one answer model");
    ensure!(cfg.generator.question_count > 0, "generator.question_count must be > 0");
    ensure!(!cfg.input.prompt_paths.is_empty(), "input.prompt_paths must not be empty");

    let samples = load_samples(&cfg)?;
    if samples.is_empty() {
        println!("no samples");
        return Ok(());
    }

    prepare_output(&cfg)?;

    let existing_ids: HashSet<_> = if cfg.run.resume {
        read_jsonl_if_exists::<TrainingRow>(&cfg.output.jsonl)?
            .into_iter()
            .map(|row| row.record_id)
            .collect()
    } else {
        HashSet::new()
    };

    let total_samples = samples.len();
    let generator_jobs = samples
        .into_iter()
        .filter(|sample| !sample_is_complete(sample, cfg.generator.question_count, &existing_ids))
        .collect::<Vec<_>>();

    let generator_client = OpenAiClient::new(
        &cfg.generator.model.base_url,
        resolve_api_key(&cfg.generator.model)?,
    );
    let generator = cfg.generator.clone();
    let run = cfg.run.clone();
    let generator_concurrency =
        concurrency_limit(cfg.concurrency.generator_requests, generator_jobs.len());
    let mut pending_rows = Vec::new();
    let mut generator_usage_total = UsageStats::default();
    let generator_pb = progress_bar(generator_jobs.len(), "generate");

    let mut generator_stream = stream::iter(generator_jobs)
        .map(|sample| {
            let client = generator_client.clone();
            let generator = generator.clone();
            let run = run.clone();
            async move { generate_questions(client, generator, run, sample).await }
        })
        .buffer_unordered(generator_concurrency);
    while let Some(result) = generator_stream.next().await {
        let batch = result?;
        generator_usage_total.add_assign(&batch.usage);
        pending_rows.extend(
            batch
                .rows
                .into_iter()
                .filter(|row| !existing_ids.contains(&row.record_id)),
        );
        generator_pb.inc(1);
    }
    generator_pb.finish_with_message("generate done");

    if pending_rows.is_empty() {
        println!("samples={total_samples} new_rows=0");
        return Ok(());
    }

    let answer_clients = cfg
        .answer_models
        .iter()
        .map(|model| Ok(OpenAiClient::new(&model.base_url, resolve_api_key(model)?)))
        .collect::<Result<Vec<_>>>()?;
    let answers = cfg.answer_models.clone();
    let run = cfg.run.clone();
    let answer_concurrency = concurrency_limit(cfg.concurrency.answer_requests, pending_rows.len());
    let generated_row_count = pending_rows.len();
    let mut written = 0usize;
    let mut answer_usage_total = UsageStats::default();
    let answer_pb = progress_bar(pending_rows.len(), "answer");

    let mut answer_stream = stream::iter(pending_rows)
        .map(|row| {
            let idx = pick_answer_model_index(&row, answers.len());
            let client = answer_clients[idx].clone();
            let model = answers[idx].clone();
            let run = run.clone();
            async move { answer_row(client, model, run, row).await }
        })
        .buffer_unordered(answer_concurrency);
    while let Some(result) = answer_stream.next().await {
        let row = result?;
        answer_usage_total.add_assign(&row.answer_usage);
        append_jsonl(&cfg.output.jsonl, std::slice::from_ref(&row))?;
        written += 1;
        answer_pb.inc(1);
    }
    answer_pb.finish_with_message("answer done");

    println!(
        "samples={total_samples} generated={generated_row_count} written={written} generator_usage={} answer_usage={}",
        generator_usage_total.summary(),
        answer_usage_total.summary(),
    );
    Ok(())
}

#[rustfmt::skip]
async fn generate_questions(client: OpenAiClient, generator: Generator, run: Run, sample: SourceSample) -> Result<GeneratedBatch> {
    let prompt = build_generation_prompt(&sample, generator.question_count)?;
    let structured_output = generator_response_format(generator.question_count);
    let result = match client
        .try_chat(
            &generator.model.name,
            &prompt,
            &run,
            generator.model.enable_thinking,
            Some(&structured_output),
            generator.model.max_completion_tokens,
            generator.model.reasoning_effort.as_deref(),
        )
        .await
    {
        Ok(result) => result,
        Err(err) if err.is_response_format_unsupported() => {
            eprintln!(
                "generator model {} does not support response_format; falling back to prompt-only JSON mode",
                generator.model.name,
            );
            client
                .try_chat(
                    &generator.model.name,
                    &prompt,
                    &run,
                    generator.model.enable_thinking,
                    None,
                    generator.model.max_completion_tokens,
                    generator.model.reasoning_effort.as_deref(),
                )
                .await
                .map_err(anyhow::Error::from)?
        }
        Err(err) => return Err(err.into()),
    };
    let rows = parse_generated_questions(
        &sample,
        &generator.model.name,
        generator.question_count,
        &result.content,
        result.usage.clone(),
    )?;
    Ok(GeneratedBatch {
        rows,
        usage: result.usage,
    })
}

#[rustfmt::skip]
async fn answer_row(client: OpenAiClient, model: Model, run: Run, row: PendingRow) -> Result<TrainingRow> {
    let result = client
        .chat(
            &model.name,
            &row.user,
            &run,
            model.enable_thinking,
            None,
            model.max_completion_tokens,
            model.reasoning_effort.as_deref(),
        )
        .await
        .with_context(|| format!("answer model {} failed for record {}", model.name, row.record_id))?;
    Ok(TrainingRow {
        generator_model: row.generator_model,
        answer_model: model.name,
        user: row.user,
        assistant: merge_assistant(result.reasoning, result.content),
        record_id: row.record_id,
        sample_id: row.sample_id,
        subject: row.subject,
        prompt: row.prompt,
        ref_answer: row.ref_answer,
        generator_usage: row.generator_usage,
        answer_usage: result.usage,
    })
}

#[rustfmt::skip]
impl OpenAiClient {
    fn new(base_url: &str, api_key: String) -> Self {
        Self {
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
        }
    }

    async fn chat(&self, model: &str, prompt: &str, run: &Run, enable_thinking: bool, response_format: Option<&Value>, max_completion_tokens: Option<u32>, reasoning_effort: Option<&str>) -> Result<ChatResult> {
        self.try_chat(
            model,
            prompt,
            run,
            enable_thinking,
            response_format,
            max_completion_tokens,
            reasoning_effort,
        )
        .await
        .map_err(Into::into)
    }

    async fn try_chat(&self, model: &str, prompt: &str, run: &Run, enable_thinking: bool, response_format: Option<&Value>, max_completion_tokens: Option<u32>, reasoning_effort: Option<&str>) -> std::result::Result<ChatResult, ChatError> {
        let payload = serde_json::to_string(&ChatRequest {
            model,
            messages: [ChatMessage {
                role: "user",
                content: prompt,
            }],
            enable_thinking,
            response_format,
            max_completion_tokens,
            reasoning_effort,
        })
        .map_err(|err| ChatError::parse(err.into()))?;

        // Chat completions are not idempotent. A transport retry on a second
        // client can duplicate spend, so live requests use a single curl path.
        let (status, body) = self.try_chat_curl(&payload, run).await?;
        self.finish_chat_response(status, body)
    }

    async fn try_chat_curl(&self, payload: &str, run: &Run) -> std::result::Result<(StatusCode, String), ChatError> {
        let mut command = TokioCommand::new("curl");
        command
            .arg("--silent")
            .arg("--show-error")
            .arg("--location")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--max-time")
            .arg(format!("{:.3}", run.request_timeout_seconds))
            .arg("--request")
            .arg("POST")
            .arg("--header")
            .arg("content-type: application/json")
            .arg("--header")
            .arg(format!("authorization: Bearer {}", self.api_key))
            .arg("--data-binary")
            .arg("@-")
            .arg("--output")
            .arg("-")
            .arg("--write-out")
            .arg(format!("{CURL_HTTP_STATUS_MARKER}%{{http_code}}"))
            .arg(&self.endpoint);

        if run.disable_env_proxy {
            command.arg("--noproxy").arg("*");
        }
        if run.force_http1 {
            command.arg("--http1.1");
        }

        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .map_err(|err| ChatError::http_message(format!("failed to spawn curl: {err}")))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(payload.as_bytes())
                .await
                .map_err(|err| ChatError::http_message(format!("failed to write curl request body: {err}")))?;
        }
        let output = child
            .wait_with_output()
            .await
            .map_err(|err| ChatError::http_message(format!("failed to wait for curl: {err}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let message = stderr.trim();
            let message = if message.is_empty() {
                format!("curl exited with status {}", output.status)
            } else {
                format!("curl exited with status {}: {message}", output.status)
            };
            return Err(ChatError::http_message(message));
        }

        split_curl_response(&String::from_utf8_lossy(&output.stdout))
    }

    fn finish_chat_response(&self, status: StatusCode, body: String) -> std::result::Result<ChatResult, ChatError> {
        if !status.is_success() {
            return Err(ChatError::api(status, body));
        }

        let result =
            parse_model_response(&body).with_context(|| response_preview_error(&body)).map_err(ChatError::parse)?;
        if result.content.is_empty() && result.reasoning.is_none() {
            return Err(ChatError::parse(anyhow!("empty model response")));
        }
        Ok(result)
    }
}

fn load_config(path: &Path) -> Result<Config> {
    let mut cfg: Config = toml::from_str(&fs::read_to_string(path)?)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    cfg.input.local_jsonl_path = resolve(base, &cfg.input.local_jsonl_path);
    cfg.output.jsonl = resolve(base, &cfg.output.jsonl);
    Ok(cfg)
}

fn load_samples(cfg: &Config) -> Result<Vec<SourceSample>> {
    let mut samples = Vec::new();
    let rows = load_input_rows(&cfg.input.local_jsonl_path)?;
    for (index, value) in rows.into_iter().enumerate().skip(cfg.input.start_index) {
        if cfg.input.limit.is_some_and(|limit| samples.len() >= limit) {
            break;
        }
        samples.push(normalize_sample(&cfg.input, index, value)?);
    }
    Ok(samples)
}

fn normalize_sample(input: &Input, index: usize, value: Value) -> Result<SourceSample> {
    let sample_id = joined_text(&value, &input.sample_id_paths, &input.sample_id_joiner)
        .or_else(|| {
            input
                .sample_id_path
                .as_deref()
                .and_then(|path| extract_text(&value, path))
        })
        .unwrap_or_else(|| format!("{}_{index:06}", input.sample_id_prefix));
    let subject = joined_text(&value, &input.subject_paths, &input.subject_joiner)
        .or_else(|| {
            input
                .subject_path
                .as_deref()
                .and_then(|path| extract_text(&value, path))
        })
        .unwrap_or_else(|| input.default_subject.clone());
    let prompt_parts = input
        .prompt_paths
        .iter()
        .filter_map(|path| extract_text(&value, path))
        .collect::<Vec<_>>();
    ensure!(
        !prompt_parts.is_empty(),
        "missing prompt parts for sample {sample_id}"
    );
    let ref_answer = extract_text(&value, &input.ref_answer_path)
        .ok_or_else(|| anyhow!("missing ref_answer for sample {sample_id}"))?;
    Ok(SourceSample {
        sample_id,
        subject,
        prompt: prompt_parts.join(&input.prompt_joiner),
        ref_answer,
    })
}

fn parse_generated_questions(
    sample: &SourceSample,
    generator_model: &str,
    n: usize,
    text: &str,
    usage: UsageStats,
) -> Result<Vec<PendingRow>> {
    let mut items = extract_json_array(parse_json_value(text)?)?;
    ensure!(
        items.len() >= n,
        "expected at least {n} generated questions, got {}",
        items.len()
    );
    items.truncate(n);
    items
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            let user = generated_user_text(value)?;
            Ok(PendingRow {
                record_id: record_id(&sample.sample_id, i),
                sample_id: sample.sample_id.clone(),
                subject: sample.subject.clone(),
                prompt: sample.prompt.clone(),
                ref_answer: sample.ref_answer.clone(),
                generator_model: generator_model.to_owned(),
                user,
                generator_usage: usage.clone(),
            })
        })
        .collect()
}

fn extract_json_array(value: Value) -> Result<Vec<Value>> {
    match value {
        Value::Array(items) => Ok(items),
        Value::Object(mut map) => {
            for key in ["items", "questions", "users", "data"] {
                if let Some(Value::Array(items)) = map.remove(key) {
                    return Ok(items);
                }
            }
            bail!("expected JSON array")
        }
        _ => bail!("expected JSON array"),
    }
}

fn generated_user_text(value: Value) -> Result<String> {
    let user = match value {
        Value::String(text) => text.trim().to_owned(),
        Value::Object(map) => ["user", "question", "prompt"]
            .into_iter()
            .filter_map(|key| map.get(key).and_then(value_to_text))
            .next()
            .unwrap_or_default(),
        other => value_to_text(&other).unwrap_or_default(),
    };
    ensure!(!user.is_empty(), "generated user prompt is empty");
    Ok(user)
}

fn build_generation_prompt(sample: &SourceSample, question_count: usize) -> Result<String> {
    Ok(format!(
        "You are generating RWKV training inputs.\n\
Generate {question_count} different user prompts based on the source task.\n\
Requirements:\n\
- Each item must be a standalone user message.\n\
- Do not assume the task is multiple-choice.\n\
- It can be instruction following, coding, extraction, transformation, reasoning, or any nearby task that matches the source prompt.\n\
- Return only JSON.\n\
- Use this exact shape: {{\"questions\":[{{\"user\":\"...\"}}]}}.\n\
- Include exactly {question_count} items in `questions`.\n\n\
Source:\n{}",
        serde_json::to_string_pretty(&json!({
            "subject": sample.subject,
            "prompt": sample.prompt,
            "ref_answer": sample.ref_answer,
        }))?
    ))
}

fn export_jsonl(input: &Path, output: &Path) -> Result<()> {
    let rows = read_jsonl::<TrainingRow>(input)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let schema = Schema::new(vec![
        Field::new("generator_model", DataType::Utf8, false),
        Field::new("answer_model", DataType::Utf8, false),
        Field::new("user", DataType::Utf8, false),
        Field::new("assistant", DataType::Utf8, false),
        Field::new("record_id", DataType::Utf8, false),
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("subject", DataType::Utf8, false),
        Field::new("prompt", DataType::Utf8, false),
        Field::new("ref_answer", DataType::Utf8, false),
        Field::new("generator_prompt_tokens", DataType::UInt64, true),
        Field::new("generator_completion_tokens", DataType::UInt64, true),
        Field::new("generator_total_tokens", DataType::UInt64, true),
        Field::new("generator_reasoning_tokens", DataType::UInt64, true),
        Field::new("answer_prompt_tokens", DataType::UInt64, true),
        Field::new("answer_completion_tokens", DataType::UInt64, true),
        Field::new("answer_total_tokens", DataType::UInt64, true),
        Field::new("answer_reasoning_tokens", DataType::UInt64, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.generator_model),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.answer_model),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(rows.iter().map(|r| &r.user))) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.assistant),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.record_id),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.sample_id),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.subject),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.prompt),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| &r.ref_answer),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.generator_usage.prompt_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.generator_usage.completion_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.generator_usage.total_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.generator_usage.reasoning_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.answer_usage.prompt_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.answer_usage.completion_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.answer_usage.total_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter()
                    .map(|r| r.answer_usage.reasoning_tokens)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    let mut writer = ArrowWriter::try_new(
        File::create(output)?,
        Arc::new(schema),
        Some(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        ),
    )?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(line.trim())?);
        }
    }
    Ok(out)
}

fn read_jsonl_if_exists<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    if path.exists() {
        read_jsonl(path)
    } else {
        Ok(Vec::new())
    }
}

fn append_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    ensure_trailing_newline(path)?;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn prepare_output(cfg: &Config) -> Result<()> {
    if let Some(parent) = cfg.output.jsonl.parent() {
        fs::create_dir_all(parent)?;
    }
    if !cfg.run.resume {
        File::create(&cfg.output.jsonl)?;
    }
    Ok(())
}

fn ensure_trailing_newline(path: &Path) -> Result<()> {
    let mut file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    if file.metadata()?.len() == 0 {
        return Ok(());
    }
    file.seek(SeekFrom::End(-1))?;
    let mut tail = [0; 1];
    file.read_exact(&mut tail)?;
    if tail[0] != b'\n' {
        file.seek(SeekFrom::End(0))?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn parse_json_value(text: &str) -> Result<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Ok(value);
    }
    for (open, close) in [('[', ']'), ('{', '}')] {
        if let (Some(start), Some(end)) = (text.find(open), text.rfind(close)) {
            if start < end {
                if let Ok(value) = serde_json::from_str::<Value>(&text[start..=end]) {
                    return Ok(value);
                }
            }
        }
    }
    Err(anyhow!("model did not return valid JSON"))
}

fn parse_model_response(body: &str) -> Result<ChatResult> {
    let value = serde_json::from_str::<Value>(body)?;
    let (content, reasoning) = extract_chat_completion_response(&value)
        .or_else(|| extract_responses_api_response(&value))
        .ok_or_else(|| anyhow!("unsupported response shape"))?;
    Ok(ChatResult {
        content,
        reasoning,
        usage: extract_usage_stats(&value),
    })
}

fn extract_chat_completion_response(value: &Value) -> Option<(String, Option<String>)> {
    let choice = value.get("choices")?.as_array()?.first()?;
    let message = choice.get("message");
    let content = message
        .and_then(|message| message.get("content"))
        .and_then(extract_response_text)
        .or_else(|| choice.get("text").and_then(extract_response_text))
        .unwrap_or_default();
    let reasoning = message.and_then(|message| {
        ["reasoning_content", "reasoning"]
            .into_iter()
            .filter_map(|key| message.get(key).and_then(extract_response_text))
            .next()
    });
    Some((content, reasoning))
}

fn extract_responses_api_response(value: &Value) -> Option<(String, Option<String>)> {
    let content = value
        .get("output_text")
        .and_then(extract_response_text)
        .or_else(|| {
            value
                .get("output")
                .and_then(Value::as_array)
                .and_then(|items| {
                    join_nonempty_text(items.iter().filter_map(|item| {
                        let item_type = item.get("type").and_then(Value::as_str);
                        if matches!(item_type, Some("reasoning")) {
                            None
                        } else {
                            item.get("content").and_then(extract_response_text)
                        }
                    }))
                })
        })
        .unwrap_or_default();
    let reasoning = value
        .get("output")
        .and_then(Value::as_array)
        .and_then(|items| {
            join_nonempty_text(items.iter().filter_map(|item| {
                if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                    item.get("summary")
                        .and_then(extract_response_text)
                        .or_else(|| item.get("content").and_then(extract_response_text))
                } else {
                    None
                }
            }))
        });
    (!content.is_empty() || reasoning.is_some()).then_some((content, reasoning))
}

fn extract_response_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => normalize_text(text),
        Value::Array(items) => join_nonempty_text(items.iter().filter_map(extract_response_text)),
        Value::Object(map) => ["text", "content", "value", "summary"]
            .into_iter()
            .filter_map(|key| map.get(key).and_then(extract_response_text))
            .next(),
        Value::Bool(_) | Value::Number(_) => normalize_text(&value.to_string()),
    }
}

fn join_nonempty_text(parts: impl Iterator<Item = String>) -> Option<String> {
    let parts = parts
        .map(|part| part.trim().to_owned())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn normalize_text(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn response_preview_error(body: &str) -> String {
    format!("response preview: {}", preview_text(body, 400))
}

fn preview_text(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    let char_count = trimmed.chars().count();
    if char_count <= limit {
        trimmed.to_owned()
    } else {
        format!("{}...", trimmed.chars().take(limit).collect::<String>())
    }
}

fn extract_usage_stats(value: &Value) -> UsageStats {
    let usage = value.get("usage");
    UsageStats {
        prompt_tokens: usage
            .and_then(|usage| usage.get("prompt_tokens"))
            .and_then(Value::as_u64),
        completion_tokens: usage
            .and_then(|usage| usage.get("completion_tokens"))
            .and_then(Value::as_u64),
        total_tokens: usage
            .and_then(|usage| usage.get("total_tokens"))
            .and_then(Value::as_u64),
        reasoning_tokens: usage
            .and_then(|usage| usage.get("completion_tokens_details"))
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64),
    }
}

fn generator_response_format(question_count: usize) -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "generated_questions",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": question_count,
                        "maxItems": question_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "user": { "type": "string" }
                            },
                            "required": ["user"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": false
            }
        }
    })
}

fn extract_text(value: &Value, path: &str) -> Option<String> {
    extract_value(value, path).and_then(|value| value_to_text(&value))
}

fn extract_value(value: &Value, path: &str) -> Option<Value> {
    let mut current = value.clone();
    for segment in path.split('.') {
        if segment.is_empty() {
            continue;
        }
        current = match current {
            Value::Object(map) => map.get(segment).cloned()?,
            Value::Array(items) => items.get(segment.parse::<usize>().ok()?).cloned()?,
            Value::String(text) => parse_context_map(&text)?.get(segment).cloned()?,
            _ => return None,
        };
    }
    Some(current)
}

fn load_input_rows(path: &Path) -> Result<Vec<Value>> {
    let text = fs::read_to_string(path)?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if trimmed.starts_with('[') {
        let value = serde_json::from_str::<Value>(trimmed)
            .with_context(|| format!("invalid JSON array input: {}", path.display()))?;
        return match value {
            Value::Array(items) => Ok(items),
            _ => bail!("expected JSON array in {}", path.display()),
        };
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return match value {
            Value::Array(items) => Ok(items),
            Value::Object(_) => Ok(vec![value]),
            _ => bail!("top-level JSON input must be object or array"),
        };
    }

    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(serde_json::from_str::<Value>(line).with_context(|| {
            format!("invalid JSONL line {} in {}", line_no + 1, path.display())
        })?);
    }
    Ok(out)
}

fn parse_context_map(text: &str) -> Option<Map<String, Value>> {
    let mut out = Map::new();
    let mut current_key = None::<String>;
    let mut current_lines = Vec::new();
    let mut saw_section = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(section) = parse_context_header(trimmed) {
            if let Some(prev) = current_key.replace(section) {
                out.insert(
                    prev,
                    Value::String(current_lines.join("\n").trim().to_owned()),
                );
                current_lines.clear();
            }
            saw_section = true;
        } else if current_key.is_some() {
            current_lines.push(line);
        }
    }

    if let Some(prev) = current_key {
        out.insert(
            prev,
            Value::String(current_lines.join("\n").trim().to_owned()),
        );
    }

    saw_section.then_some(out)
}

fn parse_context_header(line: &str) -> Option<String> {
    line.strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .filter(|section| !section.is_empty())
        .map(ToOwned::to_owned)
}

fn joined_text(value: &Value, paths: &[String], joiner: &str) -> Option<String> {
    let parts = paths
        .iter()
        .filter_map(|path| extract_text(value, path))
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join(joiner))
}

fn value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => {
            let text = text.trim();
            (!text.is_empty()).then(|| text.to_owned())
        }
        Value::Bool(_) | Value::Number(_) => Some(value.to_string()),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value)
            .ok()
            .filter(|text| !text.trim().is_empty()),
    }
}

fn sample_is_complete(
    sample: &SourceSample,
    question_count: usize,
    existing_ids: &HashSet<String>,
) -> bool {
    (0..question_count).all(|i| existing_ids.contains(&record_id(&sample.sample_id, i)))
}

fn record_id(sample_id: &str, index: usize) -> String {
    format!("{sample_id}_q{index:03}")
}

fn pick_answer_model_index(row: &PendingRow, model_count: usize) -> usize {
    let mut hasher = FxHasher::default();
    row.record_id.hash(&mut hasher);
    row.prompt.hash(&mut hasher);
    row.user.hash(&mut hasher);
    (hasher.finish() as usize) % model_count
}

fn merge_assistant(reasoning: Option<String>, content: String) -> String {
    match (reasoning, content.trim()) {
        (Some(reasoning), "") => format!("<think>\n{}\n</think>", reasoning.trim()),
        (Some(reasoning), content) => {
            format!("<think>\n{}\n</think>\n\n{}", reasoning.trim(), content)
        }
        (None, content) => content.to_owned(),
    }
}

fn concurrency_limit(requested: usize, job_count: usize) -> usize {
    match (requested, job_count) {
        (_, 0) => 1,
        (0, n) => n,
        (n, total) => n.min(total),
    }
}

fn progress_bar(total: usize, label: &str) -> ProgressBar {
    if total == 0 {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg:>10} [{bar:40.cyan/blue}] {pos}/{len} {percent:>3}% {elapsed_precise}<{eta_precise}",
        )
        .expect("valid progress bar template")
        .progress_chars("##-"),
    );
    pb.set_message(label.to_owned());
    pb
}

impl UsageStats {
    fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
            && self.reasoning_tokens.is_none()
    }

    fn add_assign(&mut self, other: &Self) {
        self.prompt_tokens = sum_optional_u64(self.prompt_tokens, other.prompt_tokens);
        self.completion_tokens = sum_optional_u64(self.completion_tokens, other.completion_tokens);
        self.total_tokens = sum_optional_u64(self.total_tokens, other.total_tokens);
        self.reasoning_tokens = sum_optional_u64(self.reasoning_tokens, other.reasoning_tokens);
    }

    fn summary(&self) -> String {
        format!(
            "prompt={} completion={} total={} reasoning={}",
            optional_u64_text(self.prompt_tokens),
            optional_u64_text(self.completion_tokens),
            optional_u64_text(self.total_tokens),
            optional_u64_text(self.reasoning_tokens),
        )
    }
}

impl ChatError {
    fn http_message(message: String) -> Self {
        Self {
            kind: ChatErrorKind::Http,
            message,
        }
    }

    fn api(status: StatusCode, body: String) -> Self {
        Self {
            kind: ChatErrorKind::Api {
                status,
                body: body.clone(),
            },
            message: format!("chat completion failed with status {status}: {body}"),
        }
    }

    fn parse(err: anyhow::Error) -> Self {
        Self {
            kind: ChatErrorKind::Parse,
            message: err.to_string(),
        }
    }

    fn is_response_format_unsupported(&self) -> bool {
        match &self.kind {
            ChatErrorKind::Api { status, body } if *status == StatusCode::BAD_REQUEST => {
                let body = body.to_ascii_lowercase();
                body.contains("response_format")
                    || body.contains("json_schema")
                    || body.contains("text.format")
                    || body.contains("format.name")
            }
            _ => false,
        }
    }
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ChatError {}

fn sum_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn split_curl_response(stdout: &str) -> std::result::Result<(StatusCode, String), ChatError> {
    let Some(index) = stdout.rfind(CURL_HTTP_STATUS_MARKER) else {
        return Err(ChatError::http_message(format!(
            "curl response missing status marker; response preview: {}",
            preview_text(stdout, 400),
        )));
    };
    let body = stdout[..index].to_owned();
    let status_text = stdout[index + CURL_HTTP_STATUS_MARKER.len()..].trim();
    let status = StatusCode::from_bytes(status_text.as_bytes()).map_err(|err| {
        ChatError::http_message(format!("invalid curl status code {status_text:?}: {err}"))
    })?;
    Ok((status, body))
}

fn optional_u64_text(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_owned())
}

fn resolve_api_key(model: &Model) -> Result<String> {
    if let Some(api_key) = model
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Ok(api_key.to_owned());
    }

    if let Some(env_name) = model
        .api_key_env
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return std::env::var(env_name).with_context(|| {
            format!(
                "missing environment variable {env_name} for model {}",
                model.name
            )
        });
    }

    if let Ok(value) = std::env::var("RWKV_GEPA_API_KEY") {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }

    if let Ok(value) = std::env::var("OPENAI_API_KEY") {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }

    bail!(
        "missing api key for model {}: set api_key, api_key_env, RWKV_GEPA_API_KEY, or OPENAI_API_KEY",
        model.name
    )
}

fn default_subject() -> String {
    "general".to_owned()
}

fn default_sample_id_prefix() -> String {
    "sample".to_owned()
}

fn default_sample_id_joiner() -> String {
    "#".to_owned()
}

fn default_subject_joiner() -> String {
    "/".to_owned()
}

fn default_prompt_joiner() -> String {
    "\n\n".to_owned()
}

#[rustfmt::skip]
impl Default for Output { fn default() -> Self { Self { jsonl: "data/rwkv_train.jsonl".into() } } }

#[rustfmt::skip]
impl Default for Run {
    fn default() -> Self {
        Self {
            resume: true,
            request_timeout_seconds: 240.0,
            disable_env_proxy: true,
            force_http1: true,
        }
    }
}

#[rustfmt::skip]
impl Default for Concurrency {
    fn default() -> Self {
        Self {
            generator_requests: 8,
            answer_requests: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChatError, ChatMessage, ChatRequest, Input, UsageStats, concurrency_limit, extract_text,
        generated_user_text, generator_response_format, load_config, load_input_rows, load_samples,
        merge_assistant, normalize_sample, parse_model_response, record_id, split_curl_response,
    };
    use reqwest::StatusCode;
    use serde_json::json;
    use std::{fs, path::PathBuf};

    #[test]
    fn zero_requested_uses_all_jobs() {
        assert_eq!(concurrency_limit(0, 100), 100);
        assert_eq!(concurrency_limit(0, 2), 2);
    }

    #[test]
    fn concurrency_is_capped_by_job_count() {
        assert_eq!(concurrency_limit(4, 100), 4);
        assert_eq!(concurrency_limit(32, 10), 10);
        assert_eq!(concurrency_limit(8, 0), 1);
    }

    #[test]
    fn dotted_json_paths_work() {
        let value = json!({
            "meta": { "id": "abc" },
            "messages": [{ "content": "first" }, { "content": "second" }],
        });
        assert_eq!(extract_text(&value, "meta.id").as_deref(), Some("abc"));
        assert_eq!(
            extract_text(&value, "messages.1.content").as_deref(),
            Some("second")
        );
    }

    #[test]
    fn context_paths_work() {
        let value = json!({
            "context": "[lp]\nen-ar_EG\n\n[domain]\nnews\n\n[document_id]\ndoc-1\n\n[segment_id]\n7\n\n[translation_prompt]\nTranslate this.\n\n[reference_target]\nfoo"
        });
        assert_eq!(
            extract_text(&value, "context.lp").as_deref(),
            Some("en-ar_EG")
        );
        assert_eq!(
            extract_text(&value, "context.domain").as_deref(),
            Some("news")
        );
        assert_eq!(
            extract_text(&value, "context.document_id").as_deref(),
            Some("doc-1")
        );
        assert_eq!(
            extract_text(&value, "context.segment_id").as_deref(),
            Some("7")
        );
    }

    #[test]
    fn normalize_sample_builds_prompt_from_multiple_paths() {
        let input = Input {
            local_jsonl_path: "unused.jsonl".into(),
            limit: None,
            start_index: 0,
            default_subject: "general".into(),
            sample_id_prefix: "sample".into(),
            sample_id_paths: Vec::new(),
            sample_id_joiner: "#".into(),
            sample_id_path: Some("id".into()),
            subject_paths: Vec::new(),
            subject_joiner: "/".into(),
            subject_path: Some("task.subject".into()),
            prompt_paths: vec!["task.prompt".into(), "task.question".into()],
            ref_answer_path: "answer".into(),
            prompt_joiner: "\n\n".into(),
        };
        let sample = normalize_sample(
            &input,
            3,
            json!({
                "id": "row-7",
                "task": {
                    "subject": "coding",
                    "prompt": "Write a function.",
                    "question": "Use Rust."
                },
                "answer": "fn main() {}"
            }),
        )
        .unwrap();
        assert_eq!(sample.sample_id, "row-7");
        assert_eq!(sample.subject, "coding");
        assert_eq!(sample.prompt, "Write a function.\n\nUse Rust.");
        assert_eq!(sample.ref_answer, "fn main() {}");
    }

    #[test]
    fn normalize_sample_supports_joined_context_fields() {
        let input = Input {
            local_jsonl_path: "unused.jsonl".into(),
            limit: None,
            start_index: 0,
            default_subject: "general".into(),
            sample_id_prefix: "sample".into(),
            sample_id_paths: vec!["context.document_id".into(), "context.segment_id".into()],
            sample_id_joiner: "#".into(),
            sample_id_path: None,
            subject_paths: vec!["context.lp".into(), "context.domain".into()],
            subject_joiner: "/".into(),
            subject_path: None,
            prompt_paths: vec!["context.translation_prompt".into()],
            ref_answer_path: "context.reference_target".into(),
            prompt_joiner: "\n\n".into(),
        };
        let sample = normalize_sample(
            &input,
            0,
            json!({
                "context": "[lp]\nen-ar_EG\n\n[domain]\nnews\n\n[document_id]\ndoc-1\n\n[segment_id]\n7\n\n[translation_prompt]\nTranslate.\n\n[reference_target]\nfoo"
            }),
        )
        .unwrap();
        assert_eq!(sample.sample_id, "doc-1#7");
        assert_eq!(sample.subject, "en-ar_EG/news");
        assert_eq!(sample.prompt, "Translate.");
        assert_eq!(sample.ref_answer, "foo");
    }

    #[test]
    fn generated_user_accepts_object_shape() {
        let user = generated_user_text(json!({ "question": "Solve this" })).unwrap();
        assert_eq!(user, "Solve this");
    }

    #[test]
    fn assistant_merges_reasoning() {
        let assistant = merge_assistant(Some("step by step".into()), "final answer".into());
        assert_eq!(assistant, "<think>\nstep by step\n</think>\n\nfinal answer");
    }

    #[test]
    fn record_ids_are_stable() {
        assert_eq!(record_id("sample_000001", 5), "sample_000001_q005");
    }

    #[test]
    fn load_input_rows_supports_json_array() {
        let path = std::env::temp_dir().join("rwkv_gepa_v1_test_array.json");
        fs::write(&path, "[{\"a\":1},{\"a\":2}]").unwrap();
        let rows = load_input_rows(&path).unwrap();
        assert_eq!(rows.len(), 2);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn config_targets_one_hundred_rows_from_ten_samples() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.toml");
        let cfg = load_config(&path).unwrap();
        let samples = load_samples(&cfg).unwrap();
        assert_eq!(cfg.generator.model.name, "gpt-5.4");
        assert_eq!(cfg.generator.question_count, 10);
        assert_eq!(cfg.input.limit, Some(10));
        assert_eq!(cfg.concurrency.generator_requests, 8);
        assert_eq!(cfg.concurrency.answer_requests, 32);
        assert!(cfg.run.disable_env_proxy);
        assert_eq!(samples.len(), 10);
        assert_eq!(samples.len() * cfg.generator.question_count, 100);
    }

    #[test]
    fn parses_standard_chat_completion_response() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello"
                }
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 21,
                "total_tokens": 32,
                "completion_tokens_details": {
                    "reasoning_tokens": 14
                }
            }
        })
        .to_string();
        assert_eq!(
            parse_model_response(&body).unwrap(),
            super::ChatResult {
                content: "hello".to_owned(),
                reasoning: None,
                usage: UsageStats {
                    prompt_tokens: Some(11),
                    completion_tokens: Some(21),
                    total_tokens: Some(32),
                    reasoning_tokens: Some(14),
                },
            }
        );
    }

    #[test]
    fn parses_chat_completion_content_parts() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": "first" },
                        { "type": "output_text", "text": "second" }
                    ],
                    "reasoning_content": [
                        { "type": "text", "text": "chain" }
                    ]
                }
            }]
        })
        .to_string();
        assert_eq!(
            parse_model_response(&body).unwrap(),
            super::ChatResult {
                content: "first\n\nsecond".to_owned(),
                reasoning: Some("chain".to_owned()),
                usage: UsageStats::default(),
            }
        );
    }

    #[test]
    fn parses_responses_api_shape() {
        let body = json!({
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{ "type": "summary_text", "text": "thought" }]
                },
                {
                    "type": "message",
                    "content": [{ "type": "output_text", "text": "answer" }]
                }
            ]
        })
        .to_string();
        assert_eq!(
            parse_model_response(&body).unwrap(),
            super::ChatResult {
                content: "answer".to_owned(),
                reasoning: Some("thought".to_owned()),
                usage: UsageStats::default(),
            }
        );
    }

    #[test]
    fn chat_request_serializes_max_completion_tokens_when_set() {
        let value = serde_json::to_value(ChatRequest {
            model: "gpt-5",
            messages: [ChatMessage {
                role: "user",
                content: "hello",
            }],
            enable_thinking: true,
            response_format: Some(&json!({ "type": "json_schema" })),
            max_completion_tokens: Some(512),
            reasoning_effort: Some("medium"),
        })
        .unwrap();
        assert_eq!(value.get("max_completion_tokens"), Some(&json!(512)));
        assert_eq!(
            value.get("response_format"),
            Some(&json!({ "type": "json_schema" }))
        );
    }

    #[test]
    fn usage_stats_addition_keeps_totals() {
        let mut total = UsageStats {
            prompt_tokens: Some(10),
            completion_tokens: None,
            total_tokens: Some(10),
            reasoning_tokens: Some(2),
        };
        total.add_assign(&UsageStats {
            prompt_tokens: Some(5),
            completion_tokens: Some(7),
            total_tokens: Some(12),
            reasoning_tokens: None,
        });
        assert_eq!(
            total,
            UsageStats {
                prompt_tokens: Some(15),
                completion_tokens: Some(7),
                total_tokens: Some(22),
                reasoning_tokens: Some(2),
            }
        );
    }

    #[test]
    fn generator_response_format_uses_questions_schema() {
        let format = generator_response_format(3);
        assert_eq!(format["type"], json!("json_schema"));
        assert_eq!(
            format["json_schema"]["schema"]["properties"]["questions"]["minItems"],
            json!(3)
        );
    }

    #[test]
    fn response_format_unsupported_is_detected() {
        let err = ChatError::api(
            StatusCode::BAD_REQUEST,
            "Unsupported parameter: response_format.json_schema".into(),
        );
        assert!(err.is_response_format_unsupported());
    }

    #[test]
    fn vendor_specific_text_format_error_is_detected() {
        let err = ChatError::api(
            StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"Missing required parameter: '***.***.name'.\",\"param\":\"text.format.name\"}}".into(),
        );
        assert!(err.is_response_format_unsupported());
    }

    #[test]
    fn split_curl_response_extracts_status_and_body() {
        let (status, body) =
            split_curl_response("{\"ok\":true}\n__CURL_HTTP_STATUS__:200").unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "{\"ok\":true}");
    }

    #[test]
    fn split_curl_response_rejects_missing_marker() {
        let err = split_curl_response("{\"ok\":true}").unwrap_err();
        assert!(err.to_string().contains("missing status marker"));
    }
}
