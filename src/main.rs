use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, Subcommand};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use reqwest::Client;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use tokio::time::sleep;

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
    #[serde(default)]
    parse_retry_times: usize,
}

#[rustfmt::skip]
#[derive(Clone, Deserialize)]
struct Model {
    #[serde(alias = "model")]
    name: String,
    base_url: String,
    api_key: String,
    #[serde(default)]
    enable_thinking: bool,
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
    max_retries: usize,
    retry_backoff_seconds: f64,
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
}

#[rustfmt::skip]
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 1],
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

#[rustfmt::skip]
#[derive(Serialize)]
struct ChatMessage<'a> { role: &'a str, content: &'a str }

#[rustfmt::skip]
#[derive(Deserialize)]
struct ChatResponse { choices: Vec<Choice> }

#[rustfmt::skip]
#[derive(Deserialize)]
struct Choice { message: Message }

#[rustfmt::skip]
#[derive(Deserialize)]
struct Message {
    content: String,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
}

#[rustfmt::skip]
#[derive(Clone)]
struct OpenAiClient { http: Client, endpoint: String, api_key: String }

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

    let generator_client =
        OpenAiClient::new(&cfg.generator.model.base_url, &cfg.generator.model.api_key, &cfg.run)?;
    let generator = cfg.generator.clone();
    let run = cfg.run.clone();
    let generator_concurrency =
        concurrency_limit(cfg.concurrency.generator_requests, generator_jobs.len());
    let mut pending_rows = Vec::new();
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
        match result {
            Ok(rows) => pending_rows.extend(
                rows.into_iter()
                    .filter(|row| !existing_ids.contains(&row.record_id)),
            ),
            Err(err) => eprintln!("{err:#}"),
        }
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
        .map(|model| OpenAiClient::new(&model.base_url, &model.api_key, &cfg.run))
        .collect::<Result<Vec<_>>>()?;
    let answers = cfg.answer_models.clone();
    let run = cfg.run.clone();
    let answer_concurrency = concurrency_limit(cfg.concurrency.answer_requests, pending_rows.len());
    let generated_row_count = pending_rows.len();
    let mut new_rows = Vec::new();
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
        match result {
            Ok(row) => new_rows.push(row),
            Err(err) => eprintln!("{err:#}"),
        }
        answer_pb.inc(1);
    }
    answer_pb.finish_with_message("answer done");

    let written = new_rows.len();
    append_jsonl(&cfg.output.jsonl, &new_rows)?;
    println!("samples={total_samples} generated={generated_row_count} written={written}");
    Ok(())
}

#[rustfmt::skip]
async fn generate_questions(client: OpenAiClient, generator: Generator, run: Run, sample: SourceSample) -> Result<Vec<PendingRow>> {
    let prompt = build_generation_prompt(&sample, generator.question_count)?;
    let mut last = None;
    for attempt in 0..=generator.parse_retry_times {
        match client
            .chat(
                &generator.model.name,
                &prompt,
                &run,
                generator.model.enable_thinking,
                generator.model.reasoning_effort.as_deref(),
            )
            .await
            .and_then(|(text, _)| parse_generated_questions(&sample, &generator.model.name, generator.question_count, &text))
        {
            Ok(rows) => return Ok(rows),
            Err(err) => {
                last = Some(err);
                if attempt < generator.parse_retry_times {
                    continue;
                }
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("question generation failed")))
}

#[rustfmt::skip]
async fn answer_row(client: OpenAiClient, model: Model, run: Run, row: PendingRow) -> Result<TrainingRow> {
    let (content, reasoning) = client
        .chat(
            &model.name,
            &row.user,
            &run,
            model.enable_thinking,
            model.reasoning_effort.as_deref(),
        )
        .await?;
    Ok(TrainingRow {
        generator_model: row.generator_model,
        answer_model: model.name,
        user: row.user,
        assistant: merge_assistant(reasoning, content),
        record_id: row.record_id,
        sample_id: row.sample_id,
        subject: row.subject,
        prompt: row.prompt,
        ref_answer: row.ref_answer,
    })
}

#[rustfmt::skip]
impl OpenAiClient {
    fn new(base_url: &str, api_key: &str, run: &Run) -> Result<Self> {
        let mut builder = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs_f64(run.request_timeout_seconds));
        if run.disable_env_proxy {
            builder = builder.no_proxy();
        }
        if run.force_http1 {
            builder = builder.http1_only();
        }
        Ok(Self {
            http: builder.build()?,
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key: api_key.to_owned(),
        })
    }

    async fn chat(&self, model: &str, prompt: &str, run: &Run, enable_thinking: bool, reasoning_effort: Option<&str>) -> Result<(String, Option<String>)> {
        let mut last = None;
        for attempt in 0..=run.max_retries {
            match self.try_chat(model, prompt, enable_thinking, reasoning_effort).await {
                Ok(ok) => return Ok(ok),
                Err(err) => {
                    last = Some(err);
                    if attempt < run.max_retries {
                        sleep(Duration::from_secs_f64(
                            run.retry_backoff_seconds * 2_f64.powi(attempt as i32),
                        ))
                        .await;
                    }
                }
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("chat failed")))
    }

    async fn try_chat(&self, model: &str, prompt: &str, enable_thinking: bool, reasoning_effort: Option<&str>) -> Result<(String, Option<String>)> {
        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&ChatRequest {
                model,
                messages: [ChatMessage {
                    role: "user",
                    content: prompt,
                }],
                enable_thinking,
                reasoning_effort,
            })
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            bail!("chat completion failed with status {status}: {body}");
        }

        let msg = serde_json::from_str::<ChatResponse>(&body)?
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("missing choices[0]"))?
            .message;
        let content = msg.content.trim().to_owned();
        let reasoning = msg
            .reasoning_content
            .or(msg.reasoning)
            .map(|text| text.trim().to_owned())
            .filter(|text| !text.is_empty());
        ensure!(!content.is_empty() || reasoning.is_some(), "empty model response");
        Ok((content, reasoning))
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
- Return only JSON array.\n\
- Each item can be a string, or an object with a `user` field.\n\n\
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
            request_timeout_seconds: 120.0,
            max_retries: 3,
            retry_backoff_seconds: 2.0,
            disable_env_proxy: false,
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
        Input, concurrency_limit, extract_text, generated_user_text, load_input_rows,
        merge_assistant, normalize_sample, record_id,
    };
    use serde_json::json;
    use std::fs;

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
}
