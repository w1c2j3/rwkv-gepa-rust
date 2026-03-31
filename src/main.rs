use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, Subcommand};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Client, StatusCode};
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use serde_jsonlines::{append_json_lines, json_lines};

#[derive(Parser)]
#[command(author, version, about = "Lean JSONL -> RWKV training pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Synthesize {
        #[arg(long, default_value = "mode.toml")]
        config: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: InputConfig,
    generator: GeneratorConfig,
    answer_models: Vec<ModelConfig>,
    #[serde(default)]
    output: OutputConfig,
    #[serde(default)]
    run: RunConfig,
    #[serde(default)]
    concurrency: ConcurrencyConfig,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputConfig {
    dataset_path: PathBuf,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    start_index: usize,
    #[serde(default = "default_subject")]
    default_subject: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorConfig {
    #[serde(flatten)]
    model: ModelConfig,
    variant_count: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    endpoint: String,
    model_name: String,
    api_key: String,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    reasoning_effort: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
struct OutputConfig {
    jsonl_path: PathBuf,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
struct RunConfig {
    resume: bool,
    request_timeout_seconds: f64,
    disable_env_proxy: bool,
    force_http1: bool,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
struct ConcurrencyConfig {
    generate_requests: usize,
    answer_requests: usize,
}

#[derive(Clone)]
struct SourceSample {
    sample_id: String,
    subject: String,
    prompt: String,
    ref_answer: String,
}

#[derive(Clone)]
struct GenerateJob {
    sample: SourceSample,
    missing_indices: Vec<usize>,
}

#[derive(Clone)]
struct PendingTask {
    task_id: String,
    sample_id: String,
    subject: String,
    prompt: String,
    ref_answer: String,
    generator_model: String,
    user: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputRow {
    task_id: String,
    sample_id: String,
    subject: String,
    prompt: String,
    ref_answer: String,
    status: String,
    generator_model: String,
    user: String,
    assistant: String,
    text: String,
    answer_model: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunStatus {
    Generated,
    Done,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Clone)]
struct OpenAiClient {
    http: Client,
    endpoint: String,
    api_key: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ChatResult {
    content: String,
    reasoning: Option<String>,
}

#[derive(Debug)]
struct RequestError {
    transient: bool,
    message: String,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let result = match Cli::parse().command {
        Command::Synthesize { config, limit } => synthesize(&config, limit).await,
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn synthesize(path: &Path, limit: Option<usize>) -> Result<()> {
    let mut cfg = load_config(path)?;
    if let Some(limit) = limit {
        cfg.input.limit = Some(limit);
    }
    validate_config(&cfg)?;

    let samples = load_samples(&cfg)?;
    if samples.is_empty() {
        println!("no samples");
        return Ok(());
    }

    prepare_output(&cfg)?;
    let resume_rows = if cfg.run.resume {
        load_resume_rows(&cfg.output.jsonl_path)?
    } else {
        HashMap::new()
    };
    if !resume_rows.is_empty() {
        let (generated, done) = summarize_resume_rows(&resume_rows)?;
        eprintln!(
            "resuming with {} tracked tasks from {} (generated={}, done={})",
            resume_rows.len(),
            cfg.output.jsonl_path.display(),
            generated,
            done
        );
    }

    let (generator_jobs, mut pending_tasks) =
        build_resume_plan(&samples, cfg.generator.variant_count, &resume_rows)?;
    let mut generated_now = 0usize;
    let resumed_generated = pending_tasks.len();
    let mut skipped_generate = 0usize;

    if !generator_jobs.is_empty() {
        let client = OpenAiClient::new(&cfg.generator.model, &cfg.run)?;
        let generator = cfg.generator.clone();
        let pb = progress_bar(generator_jobs.len(), "generate");
        let concurrency =
            concurrency_limit(cfg.concurrency.generate_requests, generator_jobs.len());
        let mut stream = stream::iter(generator_jobs)
            .map(|job| {
                let client = client.clone();
                let generator = generator.clone();
                async move { generate_tasks(client, generator, job).await }
            })
            .buffer_unordered(concurrency);

        while let Some(result) = stream.next().await {
            match result {
                Ok(tasks) => {
                    let generated_rows = tasks.iter().map(generated_output_row).collect::<Vec<_>>();
                    append_jsonl(&cfg.output.jsonl_path, &generated_rows)?;
                    generated_now += tasks.len();
                    for task in tasks {
                        pending_tasks.push(task);
                    }
                }
                Err(err) => {
                    skipped_generate += 1;
                    eprintln!("skipped generate job: {err:#}");
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("generate done");
    }

    if pending_tasks.is_empty() {
        println!(
            "samples={} total_tasks={} new_tasks=0",
            samples.len(),
            samples.len() * cfg.generator.variant_count
        );
        return Ok(());
    }

    let answer_clients = cfg
        .answer_models
        .iter()
        .map(|model| OpenAiClient::new(model, &cfg.run))
        .collect::<Result<Vec<_>>>()?;
    let answer_models = cfg.answer_models.clone();
    let answer_total = pending_tasks.len();
    let answer_pb = progress_bar(answer_total, "answer");
    let concurrency = concurrency_limit(cfg.concurrency.answer_requests, answer_total);
    let mut written = 0usize;
    let mut skipped_answer = 0usize;

    let mut stream = stream::iter(pending_tasks)
        .map(|task| {
            let idx = pick_model_index(&task.task_id, answer_models.len());
            let client = answer_clients[idx].clone();
            let model = answer_models[idx].clone();
            async move { answer_task(client, model, task).await }
        })
        .buffer_unordered(concurrency);

    while let Some(result) = stream.next().await {
        match result {
            Ok(row) => {
                append_jsonl(&cfg.output.jsonl_path, std::slice::from_ref(&row))?;
                written += 1;
            }
            Err(err) => {
                skipped_answer += 1;
                eprintln!("skipped answer task: {err:#}");
            }
        }
        answer_pb.inc(1);
    }
    answer_pb.finish_with_message("answer done");

    println!(
        "samples={} total_tasks={} resumed_generated={} generated_now={} answered_now={} skipped_generate={} skipped_answer={}",
        samples.len(),
        samples.len() * cfg.generator.variant_count,
        resumed_generated,
        generated_now,
        written,
        skipped_generate,
        skipped_answer
    );
    Ok(())
}

async fn generate_tasks(
    client: OpenAiClient,
    generator: GeneratorConfig,
    job: GenerateJob,
) -> Result<Vec<PendingTask>> {
    let prompt = build_generation_prompt(&job.sample, generator.variant_count)?;
    let result = client
        .chat(&generator.model, &prompt, true)
        .await
        .with_context(|| format!("generator failed for sample {}", job.sample.sample_id))?;
    let users = parse_generated_users(&result.content, generator.variant_count)?;
    job.missing_indices
        .into_iter()
        .map(|index| {
            let user = users.get(index).cloned().ok_or_else(|| {
                anyhow!(
                    "missing generated variant {index} for {}",
                    job.sample.sample_id
                )
            })?;
            Ok(PendingTask {
                task_id: task_id(&job.sample.sample_id, index),
                sample_id: job.sample.sample_id.clone(),
                subject: job.sample.subject.clone(),
                prompt: job.sample.prompt.clone(),
                ref_answer: job.sample.ref_answer.clone(),
                generator_model: generator.model.model_name.clone(),
                user,
            })
        })
        .collect()
}

async fn answer_task(
    client: OpenAiClient,
    model: ModelConfig,
    task: PendingTask,
) -> Result<OutputRow> {
    let result = client
        .chat(&model, &task.user, false)
        .await
        .with_context(|| format!("answer failed for task {}", task.task_id))?;
    let assistant = merge_assistant(result.reasoning, result.content);
    let text = rwkv_text(&task.user, &assistant);
    Ok(OutputRow {
        task_id: task.task_id,
        sample_id: task.sample_id,
        subject: task.subject,
        prompt: task.prompt,
        ref_answer: task.ref_answer,
        status: RunStatus::Done.as_str().to_owned(),
        generator_model: task.generator_model,
        user: task.user,
        assistant,
        text,
        answer_model: model.model_name,
    })
}

impl OpenAiClient {
    fn new(model: &ModelConfig, run: &RunConfig) -> Result<Self> {
        let mut builder =
            Client::builder().timeout(Duration::from_secs_f64(run.request_timeout_seconds));
        if run.disable_env_proxy {
            builder = builder.no_proxy();
        }
        if run.force_http1 {
            builder = builder.http1_only();
        }
        Ok(Self {
            http: builder.build()?,
            endpoint: model.endpoint.clone(),
            api_key: model.api_key.clone(),
        })
    }

    async fn chat(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> Result<ChatResult> {
        let max_attempts = 3usize;
        for attempt in 0..max_attempts {
            match self.try_chat(model, prompt, json_output).await {
                Ok(result) => return Ok(result),
                Err(err) if err.transient && attempt + 1 < max_attempts => {
                    eprintln!(
                        "transient chat error for model {} (attempt {}/{}): {}",
                        model.model_name,
                        attempt + 1,
                        max_attempts,
                        err
                    );
                    tokio::time::sleep(Duration::from_secs((attempt + 1) as u64 * 2)).await;
                }
                Err(err) => return Err(err.into()),
            }
        }
        unreachable!("chat retry loop should either return or fail")
    }

    async fn try_chat(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<ChatResult, RequestError> {
        let body = self.send_chat_request(model, prompt, json_output).await?;
        parse_chat_result(&body).map_err(RequestError::parse)
    }

    async fn send_chat_request(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<String, RequestError> {
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&chat_request(model, prompt, json_output))
            .send()
            .await
            .map_err(RequestError::http)?;
        let status = response.status();
        let body = response.text().await.map_err(RequestError::http)?;
        if !status.is_success() {
            return Err(RequestError::api(status, body));
        }
        Ok(body)
    }
}

impl RequestError {
    fn http(err: reqwest::Error) -> Self {
        Self {
            transient: err.is_timeout() || err.is_connect() || err.is_body(),
            message: err.to_string(),
        }
    }

    fn api(status: StatusCode, body: String) -> Self {
        Self {
            transient: status.is_server_error() || status.as_u16() == 429,
            message: format!(
                "chat completion failed with status {}: {}",
                status,
                preview_text(&body, 400)
            ),
        }
    }

    fn parse(err: anyhow::Error) -> Self {
        Self {
            transient: false,
            message: err.to_string(),
        }
    }
}

impl Display for RequestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

fn chat_request<'a>(model: &'a ModelConfig, prompt: &'a str, json_output: bool) -> ChatRequest<'a> {
    ChatRequest {
        model: &model.model_name,
        messages: [ChatMessage {
            role: "user",
            content: prompt,
        }],
        max_completion_tokens: model.max_completion_tokens,
        reasoning_effort: model.reasoning_effort.as_deref(),
        response_format: json_output.then(json_object_response_format),
    }
}

fn json_object_response_format() -> ResponseFormat {
    ResponseFormat {
        kind: "json_object",
    }
}

fn load_config(path: &Path) -> Result<Config> {
    let mut cfg: Config = toml::from_str(&fs::read_to_string(path)?)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    cfg.input.dataset_path = resolve(base, &cfg.input.dataset_path);
    cfg.output.jsonl_path = resolve(base, &cfg.output.jsonl_path);
    Ok(cfg)
}

fn validate_config(cfg: &Config) -> Result<()> {
    ensure!(
        cfg.generator.variant_count > 0,
        "generator.variant_count must be > 0"
    );
    ensure!(
        !cfg.answer_models.is_empty(),
        "need at least one answer model"
    );
    validate_model(&cfg.generator.model)?;
    for model in &cfg.answer_models {
        validate_model(model)?;
    }
    Ok(())
}

fn validate_model(model: &ModelConfig) -> Result<()> {
    ensure!(
        !model.endpoint.trim().is_empty(),
        "model endpoint must not be empty"
    );
    ensure!(
        !model.model_name.trim().is_empty(),
        "model_name must not be empty"
    );
    ensure!(
        !model.api_key.trim().is_empty(),
        "api_key must not be empty"
    );
    Ok(())
}

fn load_samples(cfg: &Config) -> Result<Vec<SourceSample>> {
    let mut out = Vec::new();
    let mut invalid_count = 0usize;
    let mut invalid_examples = Vec::new();
    for (index, value) in load_input_rows_window(
        &cfg.input.dataset_path,
        cfg.input.start_index,
        cfg.input.limit,
    )? {
        match normalize_sample(&cfg.input, index, value) {
            Ok(sample) => out.push(sample),
            Err(err) => {
                invalid_count += 1;
                if invalid_examples.len() < 5 {
                    invalid_examples.push(format!("sample_index={index}: {err}"));
                }
            }
        }
    }
    if invalid_count > 0 {
        let preview = if invalid_examples.is_empty() {
            String::new()
        } else {
            format!(" examples: {}", invalid_examples.join(" | "))
        };
        eprintln!(
            "skipped {invalid_count} invalid normalized samples from {}{preview}",
            cfg.input.dataset_path.display()
        );
    }
    Ok(out)
}

fn normalize_sample(input: &InputConfig, _index: usize, value: Value) -> Result<SourceSample> {
    let task_id = required_top_level_text(&value, "task_id")?;
    let sample_index = required_top_level_text(&value, "sample_index")?;
    let completions_id = required_top_level_text(&value, "completions_id")?;
    let prompt = required_top_level_text(&value, "context")?;
    let ref_answer = required_top_level_text(&value, "ref_answer")?;
    Ok(SourceSample {
        sample_id: format!("{task_id}_{sample_index}_{completions_id}"),
        subject: input.default_subject.clone(),
        prompt,
        ref_answer,
    })
}

fn build_resume_plan(
    samples: &[SourceSample],
    variant_count: usize,
    resume_rows: &HashMap<String, OutputRow>,
) -> Result<(Vec<GenerateJob>, Vec<PendingTask>)> {
    let mut generator_jobs = Vec::new();
    let mut pending_tasks = Vec::new();

    for sample in samples {
        let mut missing_indices = Vec::new();
        for index in 0..variant_count {
            let id = task_id(&sample.sample_id, index);
            match resume_rows.get(&id) {
                Some(row) => match parse_row_status(row)? {
                    RunStatus::Done => {}
                    RunStatus::Generated => {
                        pending_tasks.push(pending_task_from_output_row(row)?);
                    }
                },
                None => missing_indices.push(index),
            }
        }
        if !missing_indices.is_empty() {
            generator_jobs.push(GenerateJob {
                sample: sample.clone(),
                missing_indices,
            });
        }
    }

    Ok((generator_jobs, pending_tasks))
}

fn build_generation_prompt(sample: &SourceSample, variant_count: usize) -> Result<String> {
    Ok(format!(
        "You are rewriting user questions for RWKV training.\n\
Return only JSON.\n\
The response must be a single JSON object with exactly one key named \"questions\".\n\
The value of \"questions\" must be an array of exactly {variant_count} objects.\n\
Each object must have exactly one key named \"user\".\n\
Each \"user\" value must be a standalone rewritten user question.\n\
Do not answer the question.\n\
Do not include markdown, code fences, commentary, or extra keys.\n\n\
Source task:\n{}",
        serde_json::to_string_pretty(&json!({
            "sample_id": sample.sample_id,
            "subject": sample.subject,
            "prompt": sample.prompt,
            "ref_answer": sample.ref_answer,
        }))?
    ))
}

fn parse_generated_users(text: &str, expected: usize) -> Result<Vec<String>> {
    let value: Value = serde_json::from_str(text).context("generator did not return valid JSON")?;
    let items = value
        .get("questions")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("generator response must be an object with questions[]"))?;
    collect_users(items, expected)
}

fn collect_users(items: &[Value], expected: usize) -> Result<Vec<String>> {
    ensure!(
        items.len() == expected,
        "expected exactly {expected} generated questions, got {}",
        items.len()
    );
    items.iter().map(generated_user_text).collect()
}

fn generated_user_text(value: &Value) -> Result<String> {
    let user = value
        .get("user")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    ensure!(!user.is_empty(), "generated user prompt is empty");
    Ok(user)
}

fn parse_chat_result(body: &str) -> Result<ChatResult> {
    let value = serde_json::from_str::<Value>(body)?;
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| anyhow!("chat response is missing choices[0]"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow!("chat response is missing choices[0].message"))?;
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned);
    ensure!(
        !content.is_empty() || reasoning.is_some(),
        "chat response is empty"
    );
    Ok(ChatResult { content, reasoning })
}

fn merge_assistant(reasoning: Option<String>, content: String) -> String {
    match (reasoning.map(|text| text.trim().to_owned()), content.trim()) {
        (Some(reasoning), "") => format!("<think>\n{reasoning}\n</think>"),
        (Some(reasoning), content) => format!("<think>\n{reasoning}\n</think>\n\n{content}"),
        (None, content) => content.to_owned(),
    }
}

fn rwkv_text(user: &str, assistant: &str) -> String {
    format!("User: {}\nAssistant: {}", user.trim(), assistant.trim())
}

fn generated_output_row(task: &PendingTask) -> OutputRow {
    OutputRow {
        task_id: task.task_id.clone(),
        sample_id: task.sample_id.clone(),
        subject: task.subject.clone(),
        prompt: task.prompt.clone(),
        ref_answer: task.ref_answer.clone(),
        status: RunStatus::Generated.as_str().to_owned(),
        generator_model: task.generator_model.clone(),
        user: task.user.clone(),
        assistant: String::new(),
        text: String::new(),
        answer_model: String::new(),
    }
}

fn task_id(sample_id: &str, variant_index: usize) -> String {
    format!("{sample_id}_q{variant_index:03}")
}

fn pick_model_index(task_id: &str, count: usize) -> usize {
    let mut hasher = FxHasher::default();
    task_id.hash(&mut hasher);
    (hasher.finish() as usize) % count
}

fn load_input_rows_window(
    path: &Path,
    start_index: usize,
    limit: Option<usize>,
) -> Result<Vec<(usize, Value)>> {
    let take_limit = limit.unwrap_or(usize::MAX);
    if take_limit == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut logical_index = 0usize;
    let mut skipped = 0usize;
    let mut examples = Vec::new();
    for (line_no, item) in json_lines::<Value, _>(path)?.enumerate() {
        if out.len() >= take_limit {
            break;
        }
        let value = match item {
            Ok(value) => value,
            Err(err) => {
                skipped += 1;
                if examples.len() < 5 {
                    examples.push(format!("line {}: {err}", line_no + 1));
                }
                continue;
            }
        };
        if logical_index >= start_index {
            out.push((logical_index, value));
        }
        logical_index += 1;
    }
    if skipped > 0 {
        let preview = if examples.is_empty() {
            String::new()
        } else {
            format!(" examples: {}", examples.join(" | "))
        };
        eprintln!(
            "skipped {skipped} invalid input JSONL lines in {}{preview}",
            path.display()
        );
    }
    Ok(out)
}

fn read_jsonl<T: DeserializeOwned>(path: &Path, label: &str) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for (line_no, item) in json_lines::<T, _>(path)?.enumerate() {
        out.push(item.with_context(|| {
            format!(
                "invalid {label} JSONL line {} in {}",
                line_no + 1,
                path.display()
            )
        })?);
    }
    Ok(out)
}

fn read_jsonl_if_exists<T: DeserializeOwned>(path: &Path, label: &str) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_jsonl(path, label)
}

fn append_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    append_json_lines(path, rows.iter())?;
    Ok(())
}

fn prepare_output(cfg: &Config) -> Result<()> {
    if let Some(parent) = cfg.output.jsonl_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if !cfg.run.resume {
        File::create(&cfg.output.jsonl_path)?;
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

fn load_resume_rows(path: &Path) -> Result<HashMap<String, OutputRow>> {
    let mut rows = HashMap::new();
    for row in read_jsonl_if_exists::<OutputRow>(path, "output")? {
        rows.insert(row.task_id.clone(), row);
    }
    Ok(rows)
}

fn summarize_resume_rows(rows: &HashMap<String, OutputRow>) -> Result<(usize, usize)> {
    let mut generated = 0usize;
    let mut done = 0usize;
    for row in rows.values() {
        match parse_row_status(row)? {
            RunStatus::Generated => generated += 1,
            RunStatus::Done => done += 1,
        }
    }
    Ok((generated, done))
}

fn pending_task_from_output_row(row: &OutputRow) -> Result<PendingTask> {
    let user = row.user.trim();
    ensure!(
        !user.is_empty(),
        "generated resume row is missing user for task {}",
        row.task_id
    );
    Ok(PendingTask {
        task_id: row.task_id.clone(),
        sample_id: row.sample_id.clone(),
        subject: row.subject.clone(),
        prompt: row.prompt.clone(),
        ref_answer: row.ref_answer.clone(),
        generator_model: row.generator_model.clone(),
        user: user.to_owned(),
    })
}

fn parse_row_status(row: &OutputRow) -> Result<RunStatus> {
    let status = row.status.trim();
    match status {
        "generated" => Ok(RunStatus::Generated),
        "done" => Ok(RunStatus::Done),
        _ => Err(anyhow!(
            "unsupported output status {:?} for task {}",
            row.status,
            row.task_id
        )),
    }
}

fn required_top_level_text(value: &Value, key: &str) -> Result<String> {
    top_level_text(value, key).ok_or_else(|| anyhow!("missing {key}"))
}

fn top_level_text(value: &Value, key: &str) -> Option<String> {
    scalar_text(value.get(key)?)
}

fn scalar_text(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(text) => text.trim().to_owned(),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Null | Value::Array(_) | Value::Object(_) => return None,
    };
    (!text.is_empty()).then_some(text)
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

fn progress_bar(total: usize, label: &str) -> ProgressBar {
    if total == 0 {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:>10} [{bar:40.cyan/blue}] {pos}/{len} {percent:>3}% {elapsed_precise}<{eta_precise}",
        )
        .expect("valid progress bar template"),
    );
    pb.set_prefix(label.to_owned());
    pb
}

fn concurrency_limit(configured: usize, total: usize) -> usize {
    configured.max(1).min(total.max(1))
}

fn default_subject() -> String {
    "general".to_owned()
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            jsonl_path: PathBuf::from("data/rwkv_train.jsonl"),
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            resume: true,
            request_timeout_seconds: 240.0,
            disable_env_proxy: true,
            force_http1: true,
        }
    }
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            generate_requests: 4,
            answer_requests: 16,
        }
    }
}

impl RunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generated => "generated",
            Self::Done => "done",
        }
    }
}
