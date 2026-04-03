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
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, Response, StatusCode};
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use serde_jsonlines::{append_json_lines, json_lines};

macro_rules! default_value {
    ($name:ident, $ty:ty, $value:expr) => {
        fn $name() -> $ty {
            $value
        }
    };
}

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
    #[serde(default = "OutputConfig::default")]
    output: OutputConfig,
    #[serde(default = "RunConfig::default")]
    run: RunConfig,
    #[serde(default = "ConcurrencyConfig::default")]
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
    #[serde(default = "default_subject", rename = "default_subject")]
    _default_subject: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorConfig {
    #[serde(flatten)]
    model: ModelConfig,
    variant_count: usize,
    #[serde(default = "default_generation_attempts")]
    generation_attempts: usize,
    #[serde(default = "default_validate_generated_questions")]
    validate_generated_questions: bool,
    #[serde(default = "default_generator_json_object_response")]
    json_object_response: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    endpoint: String,
    model_name: String,
    api_key: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    thinking: Option<ThinkingSetting>,
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputConfig {
    #[serde(default = "default_output_jsonl_path")]
    jsonl_path: PathBuf,
}

#[derive(Clone)]
struct OutputPaths {
    legacy_jsonl_path: PathBuf,
    generate_jsonl_path: PathBuf,
    done_jsonl_path: PathBuf,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfig {
    #[serde(default = "default_resume")]
    resume: bool,
    #[serde(default = "default_request_timeout_seconds")]
    request_timeout_seconds: f64,
    #[serde(default = "default_disable_env_proxy")]
    disable_env_proxy: bool,
    #[serde(default = "default_force_http1")]
    force_http1: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConcurrencyConfig {
    #[serde(default = "default_generate_requests")]
    generate_requests: usize,
    #[serde(default = "default_answer_requests")]
    answer_requests: usize,
}

#[derive(Clone)]
struct SourceSample {
    sample_id: String,
    source_user: String,
    source_meta: Value,
}

#[derive(Clone)]
struct GenerateJob {
    sample: SourceSample,
    missing_indices: Vec<usize>,
    accepted_tasks: Vec<PendingTask>,
}

#[derive(Clone)]
struct PendingTask {
    task_id: String,
    user: String,
    expected_answer: String,
    generated_item_json: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct OutputRow {
    task_id: String,
    status: String,
    #[serde(
        default,
        alias = "rewritten_user",
        skip_serializing_if = "String::is_empty"
    )]
    user: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    generated_item_json: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    expected_answer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    predicted_answer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    answer_correct: Option<bool>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    answer_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    assistant: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    text: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneratedItemsEnvelope {
    items: Vec<GeneratedItemDraft>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneratedItemDraft {
    user: String,
    #[serde(default)]
    answer: String,
    #[serde(default)]
    meta: Value,
}

#[derive(Debug)]
struct GeneratedItem {
    user: String,
    answer: String,
    item_json: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValidationEnvelope {
    items: Vec<ValidationItem>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValidationItem {
    index: usize,
    valid: bool,
    reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunStatus {
    Generated,
    Done,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingRequest<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThinkingSetting {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Clone, Copy, Serialize)]
struct ThinkingRequest<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
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

    let output_paths = build_output_paths(&cfg.output)?;
    prepare_output(&output_paths, cfg.run.resume)?;
    let resume_rows = if cfg.run.resume {
        load_resume_rows(&output_paths)?
    } else {
        HashMap::new()
    };
    if !resume_rows.is_empty() {
        let (generated, done) = summarize_resume_rows(&resume_rows)?;
        eprintln!(
            "resuming with {} tracked tasks from {} and {} (generated={}, done={})",
            resume_rows.len(),
            output_paths.generate_jsonl_path.display(),
            output_paths.done_jsonl_path.display(),
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
                    append_jsonl(&output_paths.generate_jsonl_path, &generated_rows)?;
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
                append_jsonl(&output_paths.done_jsonl_path, std::slice::from_ref(&row))?;
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
    let expected = job.missing_indices.len();
    let mut feedback = None::<String>;
    let accepted_norms = job
        .accepted_tasks
        .iter()
        .map(|task| normalize_compare_text(&task.user))
        .collect::<Vec<_>>();

    for attempt in 0..generator.generation_attempts {
        let prompt = build_generation_prompt(
            &job.sample,
            expected,
            &job.accepted_tasks,
            feedback.as_deref(),
        )?;
        let result = client
            .chat(&generator.model, &prompt, generator.json_object_response)
            .await
            .with_context(|| format!("generator failed for sample {}", job.sample.sample_id));

        let generated = match result {
            Ok(result) => parse_generated_items(&result.content, expected, &job.sample),
            Err(err) => Err(err),
        };

        let generated = match generated {
            Ok(items) => items,
            Err(err) if attempt + 1 < generator.generation_attempts => {
                feedback = Some(err.to_string());
                continue;
            }
            Err(err) => return Err(err),
        };

        if let Some(item) = generated.iter().find(|item| {
            let normalized = normalize_compare_text(&item.user);
            accepted_norms
                .iter()
                .any(|existing| existing == &normalized)
        }) {
            let err = anyhow!(
                "generated item duplicates an already accepted variant: {:?}",
                preview_text(&item.user, 120)
            );
            if attempt + 1 < generator.generation_attempts {
                feedback = Some(err.to_string());
                continue;
            }
            return Err(err);
        }

        if generator.validate_generated_questions {
            let drafts = generated
                .iter()
                .map(|item| serde_json::from_str::<GeneratedItemDraft>(&item.item_json))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if let Err(err) =
                validate_generated_items_with_model(&client, &generator.model, &job.sample, &drafts)
                    .await
            {
                if attempt + 1 < generator.generation_attempts {
                    feedback = Some(err.to_string());
                    continue;
                }
                return Err(err);
            }
        }

        let tasks = generated
            .into_iter()
            .zip(job.missing_indices.iter())
            .map(|(item, index)| PendingTask {
                task_id: task_id(&job.sample.sample_id, *index),
                user: item.user,
                generated_item_json: item.item_json,
            })
            .collect::<Vec<_>>();

        return Ok(tasks);
    }

    Err(anyhow!(
        "failed to generate tasks for {} after {} attempts",
        job.sample.sample_id,
        generator.generation_attempts
    ))
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
    let assistant = merge_answer_output(result.reasoning, result.content);
    ensure!(
        !assistant.trim().is_empty(),
        "answer model returned empty output for task {}",
        task.task_id
    );
    let text = rwkv_text(&task.user, &assistant);
    Ok(OutputRow {
        task_id: task.task_id,
        status: RunStatus::Done.as_str().to_owned(),
        user: task.user,
        generated_item_json: task.generated_item_json,
        answer_model: model.model_name,
        assistant,
        text,
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
        let response = self.send_chat_request(model, prompt, json_output).await?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = response.text().await.map_err(RequestError::http)?;
        if !status.is_success() {
            return Err(RequestError::api(status, body));
        }
        if model.stream || content_type_is_event_stream(content_type.as_deref()) {
            parse_stream_chat_result(&body).map_err(RequestError::parse)
        } else {
            parse_chat_result(&body).map_err(RequestError::parse)
        }
    }

    async fn send_chat_request(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<Response, RequestError> {
        self.http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&chat_request(model, prompt, json_output))
            .send()
            .await
            .map_err(RequestError::http)
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
    let mut messages = Vec::with_capacity(2);
    if let Some(system_prompt) = model.system_prompt.as_deref() {
        messages.push(ChatMessage {
            role: "system",
            content: system_prompt,
        });
    }
    messages.push(ChatMessage {
        role: "user",
        content: prompt,
    });

    ChatRequest {
        model: &model.model_name,
        messages,
        max_completion_tokens: model.max_completion_tokens,
        reasoning_effort: model.reasoning_effort.as_deref(),
        stream: model.stream && !json_output,
        thinking: model.thinking.as_ref().map(|thinking| ThinkingRequest {
            kind: thinking.kind.as_str(),
        }),
        enable_thinking: model.enable_thinking,
        response_format: json_output.then(json_object_response_format),
    }
}

fn json_object_response_format() -> ResponseFormat {
    ResponseFormat {
        kind: "json_object",
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn content_type_is_event_stream(content_type: Option<&str>) -> bool {
    content_type
        .map(|value| value.to_ascii_lowercase().contains("text/event-stream"))
        .unwrap_or(false)
}

fn load_config(path: &Path) -> Result<Config> {
    let mut cfg: Config = toml::from_str(&fs::read_to_string(path)?)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    cfg.input.dataset_path = resolve(base, &cfg.input.dataset_path);
    cfg.output.jsonl_path = resolve(base, &cfg.output.jsonl_path);
    trim_model_config(&mut cfg.generator.model);
    cfg.answer_models.iter_mut().for_each(trim_model_config);
    Ok(cfg)
}

fn trim_model_config(model: &mut ModelConfig) {
    model.endpoint = model.endpoint.trim().to_owned();
    model.model_name = model.model_name.trim().to_owned();
    model.api_key = model.api_key.trim().to_owned();
    if let Some(system_prompt) = &mut model.system_prompt {
        *system_prompt = system_prompt.trim().to_owned();
        if system_prompt.is_empty() {
            model.system_prompt = None;
        }
    }
    if let Some(reasoning_effort) = &mut model.reasoning_effort {
        *reasoning_effort = reasoning_effort.trim().to_owned();
    }
    if let Some(thinking) = &mut model.thinking {
        thinking.kind = thinking.kind.trim().to_owned();
    }
}

fn validate_config(cfg: &Config) -> Result<()> {
    ensure!(
        cfg.generator.variant_count > 0,
        "generator.variant_count must be > 0"
    );
    ensure!(
        cfg.generator.generation_attempts > 0,
        "generator.generation_attempts must be > 0"
    );
    ensure!(
        !cfg.answer_models.is_empty(),
        "need at least one answer model"
    );
    validate_model(&cfg.generator.model)?;
    cfg.answer_models.iter().try_for_each(validate_model)?;
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
    if let Some(thinking) = &model.thinking {
        ensure!(
            !thinking.kind.trim().is_empty(),
            "thinking.type must not be empty"
        );
    }
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

fn normalize_sample(_input: &InputConfig, index: usize, value: Value) -> Result<SourceSample> {
    let context = required_top_level_text(&value, "context")
        .or_else(|_| required_top_level_text(&value, "text"))
        .with_context(|| format!("sample_index={index} missing context/text"))?;

    let source_user = parse_single_turn_context(&context)
        .or_else(|_| {
            Ok::<String, anyhow::Error>(normalize_context_text(&context).trim().to_owned())
        })
        .with_context(|| format!("failed to normalize sample at index {index}"))?;

    ensure!(!source_user.is_empty(), "source user content is empty");

    let sample_id = build_sample_id(index, &value);

    let mut meta = serde_json::Map::new();
    for key in [
        "task_id",
        "sample_index",
        "completions_id",
        "subject",
        "ref_answer",
        "source",
        "dataset",
    ] {
        if let Some(v) = value.get(key) {
            meta.insert(key.to_owned(), v.clone());
        }
    }

    Ok(SourceSample {
        sample_id,
        source_user,
        source_meta: Value::Object(meta),
    })
}

fn build_sample_id(index: usize, value: &Value) -> String {
    let task_id = top_level_text(value, "task_id");
    let sample_index = top_level_text(value, "sample_index");
    let completions_id = top_level_text(value, "completions_id");

    match (task_id, sample_index, completions_id) {
        (Some(a), Some(b), Some(c)) => format!("{a}_{b}_{c}"),
        (Some(a), Some(b), None) => format!("{a}_{b}"),
        (Some(a), None, None) => a,
        _ => format!("sample_{index:06}"),
    }
}

fn parse_single_turn_context(context: &str) -> Result<String> {
    let normalized = normalize_context_text(context);
    let trimmed = normalized.trim();
    ensure!(!trimmed.is_empty(), "context is empty");

    let user_headers = header_positions(trimmed, "User:");
    let assistant_headers = header_positions(trimmed, "Assistant:");
    ensure!(
        user_headers.len() == 1,
        "context must contain exactly one line-start User: header, got {} in {:?}",
        user_headers.len(),
        preview_text(trimmed, 120)
    );
    ensure!(
        assistant_headers.len() == 1,
        "context must contain exactly one line-start Assistant: header, got {} in {:?}",
        assistant_headers.len(),
        preview_text(trimmed, 120)
    );

    let user_start = user_headers[0];
    let assistant_start = assistant_headers[0];
    ensure!(user_start == 0, "context must start with User:");
    ensure!(
        assistant_start > user_start,
        "Assistant: header must appear after User: header"
    );

    let user = trimmed["User:".len()..assistant_start].trim();
    ensure!(!user.is_empty(), "context user content is empty");
    Ok(user.to_owned())
}

fn normalize_context_text(text: &str) -> String {
    if !text.contains('\\') {
        return text.to_owned();
    }

    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('/') => out.push('/'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn header_positions(text: &str, header: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut offset = 0usize;
    for segment in text.split_inclusive('\n') {
        let line = segment.trim_end_matches(&['\r', '\n'][..]);
        if line.starts_with(header) {
            positions.push(offset);
        }
        offset += segment.len();
    }
    positions
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
        let mut accepted_tasks = Vec::new();
        for index in 0..variant_count {
            let id = task_id(&sample.sample_id, index);
            match resume_rows.get(&id) {
                Some(row) => {
                    let task = pending_task_from_output_row(row)?;
                    accepted_tasks.push(task.clone());
                    match parse_row_status(row)? {
                        RunStatus::Done => {}
                        RunStatus::Generated => {
                            pending_tasks.push(task);
                        }
                    }
                }
                None => missing_indices.push(index),
            }
        }
        if !missing_indices.is_empty() {
            generator_jobs.push(GenerateJob {
                sample: sample.clone(),
                missing_indices,
                accepted_tasks,
            });
        }
    }

    Ok((generator_jobs, pending_tasks))
}

fn build_generation_prompt(
    sample: &SourceSample,
    count: usize,
    accepted: &[PendingTask],
    feedback: Option<&str>,
) -> Result<String> {
    let accepted_block = if accepted.is_empty() {
        String::new()
    } else {
        format!(
            "\n已接受样本（新生成内容不要与这些重复）：\n{}\n",
            serde_json::to_string_pretty(
                &accepted
                    .iter()
                    .map(|task| json!({ "user": task.user }))
                    .collect::<Vec<_>>()
            )?
        )
    };
    let feedback_block = feedback
        .map(|text| {
            format!("\n上一轮输出未通过，主要原因：\n{text}\n请仅修复这些问题后重新输出。\n")
        })
        .unwrap_or_default();

    Ok(format!(
        "你要基于给定的原始 user prompt，生成 {count} 条新的训练样本。\n\n\
目标：\n\
生成的新样本必须与原样本保持相同的任务类型、语言风格、难度层级和作答方式，但不能只是原句的同义改写、语序调整、浅层替换或局部扰动。\n\
新样本应与原样本属于同一类任务，但在题目对象、输入材料、约束条件、测试重点、求解路径、适用场景或边界条件上，至少有一处本质变化。\n\n\
硬性要求：\n\
1. 输出的是新的 user prompt，不要输出 assistant answer，不要输出解析，不要解释思路。\n\
2. 新样本必须自包含，不能依赖图片、文件、外部网页、历史对话、隐含上下文、未给出的背景事实或“见上文/见下图/联网查询”等信息。\n\
3. 不能泄漏或复用原题答案、参考答案、判题结果、历史 assistant 回复、思维链或任何评测信息。\n\
4. 若原样本是单轮任务，则新样本也必须保持为单轮任务。\n\
5. 若原样本包含格式约束、答题方式约束、输入输出约束、代码补全范式、选项结构、函数签名、样例格式或“只输出答案”等要求，应尽量保留。\n\
6. 若原样本属于以下类型之一，新样本必须保持同类任务，不得改成其他类型：\n\
   - 知识问答\n\
   - 数学题\n\
   - 代码生成\n\
   - 代码补全\n\
   - 算法题\n\
   - 翻译\n\
   - 分类/判断\n\
   - 多项选择题\n\
7. 若原样本包含代码、伪代码、函数接口、输入输出描述、候选选项、样例或结构化约束，新样本应保留这种组织方式，但内容本身要实质变化。\n\
8. 若新样本包含事实性内容，该内容必须满足以下之一：\n\
   - 来自原样本中已给出的信息框架；\n\
   - 不依赖具体外部事实也能自然成立；\n\
   - 是常见、稳定、非冷门、无需检索即可回答的内容。\n\
9. 不要为了“看起来不同”而引入冷门、偏门、拼凑感强、低频或不自然的设定。\n\
10. 新样本应当自然、清晰、完整、可回答，并且足以支持高质量回答。\n\
11. 若原样本本身存在噪声、冗余提示、错误推理、评测残留、无关元信息或低质量表达，新样本不要继承这些噪声，应只保留有效任务意图与必要风格特征。\n\
12. 不要生成与原样本几乎等价的问题；不要只替换数字、实体名、变量名、地名、人名或语言表面形式。\n\n\
多样性要求：\n\
- 与原样本相比，优先改变以下至少一项：\n\
  - 输入材料\n\
  - 判断对象\n\
  - 约束条件\n\
  - 边界情况\n\
  - 题面设定\n\
  - 测试重点\n\
  - 推理路径\n\
  - 输出组织方式\n\
- 但改变后仍要保持同类任务、同级难度、同类回答方式。\n\n\
输出格式：\n\
返回纯 JSON，不要输出额外文字。\n\
{{\n\
  \"items\": [\n\
    {{\n\
      \"user\": \"...\",\n\
      \"meta\": {{}}\n\
    }}\n\
  ]\n\
}}\n\n\
其中：\n\
- items 必须恰好包含 {count} 个对象\n\
- 每个对象的 user 为新的 user prompt\n\
- 每个对象的 meta 可为空对象\n\n\
输出前自检：\n\
- 这些新样本是否只是原样本改写？如果是，重写。\n\
- 这些新样本是否泄漏了原答案、历史回复或评测信息？如果是，重写。\n\
- 这些新样本是否需要外部信息才能回答？如果是，重写。\n\
- 这些新样本是否改变了任务类型或答题方式？如果是，重写。\n\
- 这些新样本是否遵循事实性原则？若包含事实性内容，是否来自原样本已给出的信息框架，或属于无需检索即可回答的常见稳定事实？如果不是，重写。\n\
- 这些新样本是否自然、清晰、完整？如果不是，重写。\n\n\
原始 user prompt：\n{}\n{}{}",
        sample.source_user, accepted_block, feedback_block
    ))
}

fn parse_generated_items(
    text: &str,
    expected: usize,
    sample: &SourceSample,
) -> Result<Vec<GeneratedItem>> {
    let envelope: GeneratedItemsEnvelope = serde_json::from_str(text)
        .or_else(|_| {
            extract_json_object_from_text(text)
                .ok_or_else(|| serde_json::Error::io(std::io::Error::other("no json object")))
                .and_then(|json| serde_json::from_str(&json))
        })
        .with_context(|| {
            format!(
                "generator did not return valid JSON or extractable JSON: {:?}",
                preview_text(text, 200)
            )
        })?;

    ensure!(
        envelope.items.len() == expected,
        "expected exactly {expected} generated items, got {}",
        envelope.items.len()
    );

    let mut seen = Vec::<String>::new();
    let mut out = Vec::new();

    for (index, draft) in envelope.items.into_iter().enumerate() {
        let item = validate_generated_item(draft, sample)
            .with_context(|| format!("generated item #{index} failed validation"))?;
        let normalized = normalize_compare_text(&item.user);
        ensure!(
            !seen.iter().any(|x| x == &normalized),
            "generated item #{index} duplicates another generated item"
        );
        seen.push(normalized);
        out.push(item);
    }

    Ok(out)
}

fn extract_json_object_from_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    for segment in trimmed.split("```").skip(1).step_by(2) {
        if let Some(json) = extract_first_balanced_json_object(segment) {
            return Some(json);
        }
    }

    extract_first_balanced_json_object(trimmed)
}

fn extract_first_balanced_json_object(text: &str) -> Option<String> {
    let mut start = None::<usize>;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in text.char_indices() {
        if let Some(object_start) = start {
            if in_string {
                if escaped {
                    escaped = false;
                    continue;
                }
                match ch {
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
                continue;
            }

            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[object_start..index + ch.len_utf8()].trim().to_owned());
                    }
                }
                _ => {}
            }
        } else if ch == '{' {
            start = Some(index);
            depth = 1;
        }
    }

    None
}

fn validate_generated_item(
    draft: GeneratedItemDraft,
    sample: &SourceSample,
) -> Result<GeneratedItem> {
    let user = draft.user.trim().to_owned();
    ensure!(!user.is_empty(), "generated user is empty");
    ensure!(
        !user.starts_with("Assistant:"),
        "generated item must be a user prompt, not an assistant response"
    );

    let user_norm = normalize_compare_text(&user);
    let original_norm = normalize_compare_text(&sample.source_user);

    ensure!(
        user_norm != original_norm,
        "generated item is effectively identical to the original"
    );

    let item_json = serde_json::to_string(&GeneratedItemDraft {
        user: user.clone(),
        meta: draft.meta,
    })?;

    Ok(GeneratedItem { user, item_json })
}

async fn validate_generated_items_with_model(
    client: &OpenAiClient,
    model: &ModelConfig,
    sample: &SourceSample,
    generated: &[GeneratedItemDraft],
) -> Result<()> {
    let prompt = build_generation_validation_prompt(sample, generated)?;
    let result = client
        .chat(model, &prompt, true)
        .await
        .with_context(|| format!("validator failed for sample {}", sample.sample_id))?;
    let envelope: ValidationEnvelope =
        serde_json::from_str(&result.content).context("validator did not return valid JSON")?;
    ensure!(
        envelope.items.len() == generated.len(),
        "validator returned {} items, expected {}",
        envelope.items.len(),
        generated.len()
    );

    let invalid_reasons = envelope
        .items
        .into_iter()
        .filter(|item| !item.valid)
        .map(|item| format!("#{} {}", item.index, item.reason.trim()))
        .collect::<Vec<_>>();
    ensure!(
        invalid_reasons.is_empty(),
        "model validation rejected generated items: {}",
        invalid_reasons.join(" | ")
    );
    Ok(())
}

fn build_generation_validation_prompt(
    sample: &SourceSample,
    generated: &[GeneratedItemDraft],
) -> Result<String> {
    Ok(format!(
        "你是训练样本质检器。\n\
返回纯 JSON：{{\"items\":[{{\"index\":0,\"valid\":true,\"reason\":\"...\"}}]}}\n\
items 长度必须与候选样本数量一致，每项只能有 index、valid、reason。\n\n\
只有同时满足以下条件时才判 valid=true：\n\
1. 与原样本保持同一知识域、任务风格和语言风格。\n\
2. 不是简单同义改写或表面改写。\n\
3. 新样本与原样本相关但本质不同。\n\
4. 新样本是自包含的，不依赖外部检索、图片、历史上下文或隐藏前提。\n\
5. 新样本本身是一个 user prompt，而不是 assistant answer。\n\
6. 候选样本本身清晰、自然，不是低质量、冷门或明显拼凑的特例。\n\n\
不通过时，reason 只写最关键的失败原因，例如：表述改写 / 与原任务本质相同 / 需要外部知识 / 不是 user prompt / 低质量拼接。\n\n\
原始样本：\n{}\n\n\
候选样本：\n{}\n",
        serde_json::to_string_pretty(&json!({
            "sample_id": sample.sample_id,
            "user": sample.source_user,
            "meta": sample.source_meta,
        }))?,
        serde_json::to_string_pretty(
            &generated
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    json!({
                        "index": index,
                        "user": item.user,
                        "meta": item.meta,
                    })
                })
                .collect::<Vec<_>>()
        )?
    ))
}

fn normalize_compare_text(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            !ch.is_whitespace()
                && !ch.is_ascii_punctuation()
                && !matches!(
                    ch,
                    '，' | '。'
                        | '：'
                        | '；'
                        | '！'
                        | '？'
                        | '（'
                        | '）'
                        | '【'
                        | '】'
                        | '“'
                        | '”'
                        | '‘'
                        | '’'
                        | '、'
                )
        })
        .flat_map(|ch| ch.to_lowercase())
        .collect()
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
    let reasoning = ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|key| {
            message
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        });
    ensure!(
        !content.is_empty() || reasoning.is_some(),
        "chat response is empty"
    );
    Ok(ChatResult { content, reasoning })
}

fn parse_stream_chat_result(body: &str) -> Result<ChatResult> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut saw_chunk = false;

    for raw_line in body.lines() {
        let line = raw_line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }

        let value = serde_json::from_str::<Value>(data).with_context(|| {
            format!(
                "stream chunk is not valid JSON: {:?}",
                preview_text(data, 120)
            )
        })?;
        saw_chunk = true;

        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for choice in choices {
            let delta = choice
                .get("delta")
                .or_else(|| choice.get("message"))
                .unwrap_or(choice);
            append_chunk_text(delta, "reasoning_content", &mut reasoning);
            append_chunk_text(delta, "reasoning", &mut reasoning);
            append_chunk_text(delta, "content", &mut content);
        }
    }

    ensure!(saw_chunk, "stream response did not contain any data chunks");
    let content = content.trim().to_owned();
    let reasoning = reasoning.trim().to_owned();
    ensure!(
        !content.is_empty() || !reasoning.is_empty(),
        "chat stream response is empty"
    );
    Ok(ChatResult {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    })
}

fn append_chunk_text(value: &Value, key: &str, out: &mut String) {
    if let Some(text) = value.get(key).and_then(Value::as_str) {
        out.push_str(text);
    }
}

fn merge_answer_output(reasoning: Option<String>, content: String) -> String {
    match (reasoning.map(|text| text.trim().to_owned()), content.trim()) {
        (Some(reasoning), "") => reasoning,
        (Some(reasoning), content) => format!("{reasoning}\n\n{content}"),
        (None, content) => content.to_owned(),
    }
}

fn rwkv_text(user: &str, assistant: &str) -> String {
    format!("User: {}\nAssistant: {}", user.trim(), assistant.trim())
}

fn generated_output_row(task: &PendingTask) -> OutputRow {
    OutputRow {
        task_id: task.task_id.clone(),
        status: RunStatus::Generated.as_str().to_owned(),
        user: task.user.clone(),
        generated_item_json: task.generated_item_json.clone(),
        answer_model: String::new(),
        assistant: String::new(),
        text: String::new(),
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

fn build_output_paths(output: &OutputConfig) -> Result<OutputPaths> {
    let legacy_jsonl_path = output.jsonl_path.clone();
    let base_dir = output
        .jsonl_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("data"));
    let file_name = output
        .jsonl_path
        .file_name()
        .ok_or_else(|| anyhow!("output.jsonl_path must include a file name"))?;

    Ok(OutputPaths {
        legacy_jsonl_path,
        generate_jsonl_path: base_dir.join("generate").join(file_name),
        done_jsonl_path: base_dir.join("done").join(file_name),
    })
}

fn prepare_output(paths: &OutputPaths, resume: bool) -> Result<()> {
    for parent in [
        paths.generate_jsonl_path.parent(),
        paths.done_jsonl_path.parent(),
    ]
    .into_iter()
    .flatten()
    {
        fs::create_dir_all(parent)?;
    }
    if !resume {
        File::create(&paths.generate_jsonl_path)?;
        File::create(&paths.done_jsonl_path)?;
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

fn load_resume_rows(paths: &OutputPaths) -> Result<HashMap<String, OutputRow>> {
    let mut rows = HashMap::new();

    for (path, label) in [
        (&paths.legacy_jsonl_path, "legacy output"),
        (&paths.generate_jsonl_path, "generated output"),
        (&paths.done_jsonl_path, "done output"),
    ] {
        for row in read_jsonl_if_exists::<OutputRow>(path, label)? {
            rows.insert(row.task_id.clone(), row);
        }
    }

    Ok(rows)
}

fn summarize_resume_rows(rows: &HashMap<String, OutputRow>) -> Result<(usize, usize)> {
    rows.values()
        .try_fold((0usize, 0usize), |(generated, done), row| {
            Ok(match parse_row_status(row)? {
                RunStatus::Generated => (generated + 1, done),
                RunStatus::Done => (generated, done + 1),
            })
        })
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
        user: user.to_owned(),
        generated_item_json: row.generated_item_json.trim().to_owned(),
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

default_value!(default_subject, String, "general".to_owned());
default_value!(
    default_output_jsonl_path,
    PathBuf,
    PathBuf::from("data/rwkv_train.jsonl")
);
default_value!(default_resume, bool, true);
default_value!(default_request_timeout_seconds, f64, 240.0);
default_value!(default_disable_env_proxy, bool, true);
default_value!(default_force_http1, bool, true);
default_value!(default_generate_requests, usize, 4);
default_value!(default_answer_requests, usize, 16);
default_value!(default_generation_attempts, usize, 4);
default_value!(default_validate_generated_questions, bool, true);
default_value!(default_generator_json_object_response, bool, true);

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            jsonl_path: default_output_jsonl_path(),
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            resume: default_resume(),
            request_timeout_seconds: default_request_timeout_seconds(),
            disable_env_proxy: default_disable_env_proxy(),
            force_http1: default_force_http1(),
        }
    }
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            generate_requests: default_generate_requests(),
            answer_requests: default_answer_requests(),
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
