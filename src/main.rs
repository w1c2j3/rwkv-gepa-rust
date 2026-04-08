use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, Subcommand};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, Response, StatusCode};
use rustc_hash::FxHasher;
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeOwned, DeserializeSeed, SeqAccess, Visitor},
};
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
    #[serde(default = "default_validator_json_object_response")]
    validator_json_object_response: bool,
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
    generate_jsonl_path: PathBuf,
    done_success_jsonl_path: PathBuf,
    done_failed_jsonl_path: PathBuf,
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

#[derive(Clone)]
struct PromptTemplates {
    profile_name: String,
    generation: String,
    validation: Option<String>,
}

#[derive(Clone)]
struct GeneratedTaskWriter {
    generate_jsonl_path: PathBuf,
    write_lock: Arc<Mutex<()>>,
}

#[derive(Default)]
struct AnswerBatchStats {
    success: usize,
    failed: usize,
    skipped: usize,
}

struct AnsweredTask {
    row: OutputRow,
    correct: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptProfileFile {
    name: String,
    generation: PromptTemplateSection,
    #[serde(default)]
    validation: Option<PromptTemplateSection>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptTemplateSection {
    template: String,
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
    let prompt_templates = load_prompt_templates(cfg.generator.validate_generated_questions)?;

    let samples = load_samples(&cfg)?;
    if samples.is_empty() {
        println!("no samples");
        return Ok(());
    }

    let output_paths = build_output_paths(&cfg.output)?;
    prepare_output(&output_paths, cfg.run.resume)?;
    let generated_task_writer = GeneratedTaskWriter::new(&output_paths);
    let resume_rows = if cfg.run.resume {
        load_resume_rows(&output_paths)?
    } else {
        HashMap::new()
    };
    if !resume_rows.is_empty() {
        let (generated, done) = summarize_resume_rows(&resume_rows)?;
        eprintln!(
            "resuming with {} tracked tasks from {}, {}, {} (generated={}, done={})",
            resume_rows.len(),
            output_paths.generate_jsonl_path.display(),
            output_paths.done_success_jsonl_path.display(),
            output_paths.done_failed_jsonl_path.display(),
            generated,
            done
        );
    }

    let (generator_jobs, resumed_pending_tasks) =
        build_resume_plan(&samples, cfg.generator.variant_count, &resume_rows)?;
    let answer_clients = cfg
        .answer_models
        .iter()
        .map(|model| OpenAiClient::new(model, &cfg.run))
        .collect::<Result<Vec<_>>>()?;
    let answer_models = cfg.answer_models.clone();
    let mut generated_now = 0usize;
    let resumed_generated = resumed_pending_tasks.len();
    let mut skipped_generate = 0usize;
    let mut answered_success = 0usize;
    let mut answered_failed = 0usize;
    let mut skipped_answer = 0usize;

    if !resumed_pending_tasks.is_empty() {
        let stats = answer_tasks_and_persist(
            answer_clients.clone(),
            answer_models.clone(),
            resumed_pending_tasks,
            &output_paths,
            cfg.concurrency.answer_requests,
        )
        .await?;
        answered_success += stats.success;
        answered_failed += stats.failed;
        skipped_answer += stats.skipped;
    }

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
                let prompt_templates = prompt_templates.clone();
                let generated_task_writer = generated_task_writer.clone();
                async move {
                    generate_tasks(
                        client,
                        generator,
                        prompt_templates,
                        generated_task_writer,
                        job,
                    )
                    .await
                }
            })
            .buffer_unordered(concurrency);

        while let Some(result) = stream.next().await {
            match result {
                Ok(tasks) => {
                    generated_now += tasks.len();
                    let stats = answer_tasks_and_persist(
                        answer_clients.clone(),
                        answer_models.clone(),
                        tasks,
                        &output_paths,
                        cfg.concurrency.answer_requests,
                    )
                    .await?;
                    answered_success += stats.success;
                    answered_failed += stats.failed;
                    skipped_answer += stats.skipped;
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

    println!(
        "samples={} total_tasks={} resumed_generated={} generated_now={} answered_success={} answered_failed={} skipped_generate={} skipped_answer={}",
        samples.len(),
        samples.len() * cfg.generator.variant_count,
        resumed_generated,
        generated_now,
        answered_success,
        answered_failed,
        skipped_generate,
        skipped_answer
    );
    Ok(())
}

async fn generate_tasks(
    client: OpenAiClient,
    generator: GeneratorConfig,
    prompt_templates: PromptTemplates,
    generated_task_writer: GeneratedTaskWriter,
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
            &prompt_templates,
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
            if let Err(err) = validate_generated_items_with_model(
                &client,
                &generator.model,
                generator.validator_json_object_response,
                &prompt_templates,
                &job.sample,
                &drafts,
            )
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
                expected_answer: item.answer,
                generated_item_json: item.item_json,
            })
            .collect::<Vec<_>>();

        generated_task_writer.append_generated_tasks(&tasks)?;
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
) -> Result<AnsweredTask> {
    let answer_prompt = build_answer_prompt(&task.user);
    let result = client
        .chat(&model, &answer_prompt, false)
        .await
        .with_context(|| format!("answer failed for task {}", task.task_id))?;

    let assistant = merge_answer_output(result.reasoning, result.content);

    ensure!(
        !assistant.trim().is_empty(),
        "answer model returned empty output for task {}",
        task.task_id
    );

    // 用完整 assistant 做答案抽取与判对
    let predicted_answer =
        extract_predicted_answer(&task.expected_answer, &assistant, &assistant);
    let answer_correct =
        compare_expected_and_predicted(&task.expected_answer, &predicted_answer);

    // success 的训练文本写完整 assistant，而不是标准答案字母
    let text = if answer_correct {
        rwkv_text(&task.user, &assistant)
    } else {
        String::new()
    };

    Ok(AnsweredTask {
        row: OutputRow {
            task_id: task.task_id,
            status: RunStatus::Done.as_str().to_owned(),

            // 这些别清空，保留下来方便排查和后处理
            user: task.user.clone(),
            generated_item_json: task.generated_item_json.clone(),
            answer_model: model.model_name.clone(),
            assistant: assistant.clone(),
            text,
        },
        correct: answer_correct,
    })
}

async fn answer_tasks_and_persist(
    answer_clients: Vec<OpenAiClient>,
    answer_models: Vec<ModelConfig>,
    tasks: Vec<PendingTask>,
    output_paths: &OutputPaths,
    configured_concurrency: usize,
) -> Result<AnswerBatchStats> {
    if tasks.is_empty() {
        return Ok(AnswerBatchStats::default());
    }

    let concurrency = concurrency_limit(configured_concurrency, tasks.len());
    let mut stats = AnswerBatchStats::default();
    let mut stream = stream::iter(tasks)
        .map(|task| {
            let idx = pick_model_index(&task.task_id, answer_models.len());
            let client = answer_clients[idx].clone();
            let model = answer_models[idx].clone();
            async move { answer_task(client, model, task).await }
        })
        .buffer_unordered(concurrency);

    while let Some(result) = stream.next().await {
        match result {
            Ok(answered) => {
                let path = done_output_path(output_paths, answered.correct);
                append_jsonl(path, std::slice::from_ref(&answered.row))?;
                if answered.correct {
                    stats.success += 1;
                } else {
                    stats.failed += 1;
                }
            }
            Err(err) => {
                stats.skipped += 1;
                eprintln!("skipped answer task: {err:#}");
            }
        }
    }

    Ok(stats)
}

impl OpenAiClient {
    fn new(model: &ModelConfig, run: &RunConfig) -> Result<Self> {
        let mut builder = Client::builder()
            .timeout(Duration::from_secs_f64(run.request_timeout_seconds))
            .user_agent("curl/8.5.0");
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

impl GeneratedTaskWriter {
    fn new(paths: &OutputPaths) -> Self {
        Self {
            generate_jsonl_path: paths.generate_jsonl_path.clone(),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    fn append_generated_tasks(&self, tasks: &[PendingTask]) -> Result<()> {
        if tasks.is_empty() {
            return Ok(());
        }
        let rows = tasks.iter().map(generated_output_row).collect::<Vec<_>>();
        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| anyhow!("generated task writer lock poisoned"))?;
        append_jsonl(&self.generate_jsonl_path, &rows)
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

fn load_prompt_templates(validate_generated_questions: bool) -> Result<PromptTemplates> {
    let profile: PromptProfileFile =
        toml::from_str(include_str!("../prompts/multiple_choice.toml"))
            .context("failed to parse built-in prompt profile TOML prompts/multiple_choice.toml")?;
    ensure!(
        !profile.name.trim().is_empty(),
        "built-in prompt profile name must not be empty"
    );
    ensure!(
        !profile.generation.template.trim().is_empty(),
        "built-in prompt profile generation.template must not be empty"
    );
    if validate_generated_questions {
        let validation = profile.validation.as_ref().ok_or_else(|| {
            anyhow!("built-in prompt profile is missing [validation] while generator.validate_generated_questions=true")
        })?;
        ensure!(
            !validation.template.trim().is_empty(),
            "built-in prompt profile validation.template must not be empty"
        );
    }

    Ok(PromptTemplates {
        profile_name: profile.name.trim().to_owned(),
        generation: profile.generation.template.trim().to_owned(),
        validation: profile
            .validation
            .map(|section| section.template.trim().to_owned())
            .filter(|text| !text.is_empty()),
    })
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

    let source_user = sanitize_training_user_prompt(
        &extract_source_user(&context)
            .with_context(|| format!("failed to normalize sample at index {index}"))?,
    );

    ensure!(!source_user.is_empty(), "source user content is empty");

    let sample_id = build_sample_id(index, &value);

    let mut meta = serde_json::Map::new();
    for key in [
        "task_id",
        "sample_index",
        "repeat_index",
        "pass_index",
        "completions_id",
        "answer",
        "subject",
        "ref_answer",
        "fail_reason",
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
    let repeat_index = top_level_text(value, "repeat_index");
    let pass_index = top_level_text(value, "pass_index");
    let completions_id = top_level_text(value, "completions_id");

    match (
        task_id,
        sample_index,
        repeat_index,
        pass_index,
        completions_id,
    ) {
        (Some(a), Some(b), _, _, Some(c)) => format!("{a}_{b}_{c}"),
        (Some(a), Some(b), _, _, None) => format!("{a}_{b}"),
        (Some(a), None, _, _, None) => a,
        (None, Some(a), Some(b), Some(c), _) => format!("{a}_{b}_{c}"),
        (None, Some(a), Some(b), None, _) => format!("{a}_{b}"),
        (None, Some(a), None, Some(c), _) => format!("{a}_{c}"),
        (None, Some(a), None, None, _) => format!("sample_{a}"),
        _ => format!("sample_{index:06}"),
    }
}

fn extract_source_user(context: &str) -> Result<String> {
    parse_single_turn_context(context)
        .or_else(|_| extract_source_user_from_structured_context(context))
        .or_else(|_| {
            let normalized = normalize_context_text(context);
            let trimmed = normalized.trim();
            ensure!(!trimmed.is_empty(), "context is empty");
            Ok(trimmed.to_owned())
        })
}

fn sanitize_training_user_prompt(text: &str) -> String {
    let mut cleaned = text.trim().to_owned();

    loop {
        let Some(rest) = cleaned.strip_prefix("You are a very talented expert in ") else {
            break;
        };
        let Some(period_index) = rest.find('.') else {
            break;
        };
        cleaned = rest[period_index + 1..].trim_start().to_owned();
    }

    cleaned.trim().to_owned()
}

fn extract_source_user_from_structured_context(context: &str) -> Result<String> {
    let normalized = normalize_context_text(context);
    let payload: Value = serde_json::from_str(normalized.trim())
        .context("context is not parseable structured JSON")?;

    if let Some(prompt) = payload.get("prompt").and_then(Value::as_str) {
        return extract_source_user_from_prompt(prompt);
    }

    if let Some(stages) = payload.get("stages").and_then(Value::as_array) {
        for stage in stages {
            if let Some(prompt) = stage.get("prompt").and_then(Value::as_str) {
                if let Ok(user) = extract_source_user_from_prompt(prompt) {
                    return Ok(user);
                }
            }
        }
    }

    Err(anyhow!(
        "structured context JSON is missing a usable prompt"
    ))
}

fn extract_source_user_from_prompt(prompt: &str) -> Result<String> {
    parse_single_turn_context(prompt).or_else(|_| {
        let normalized = normalize_context_text(prompt);
        let trimmed = normalized.trim();
        let prompt_body = trimmed
            .strip_prefix("User:")
            .map(str::trim)
            .ok_or_else(|| anyhow!("prompt does not start with User:"))?;
        let user = prompt_body
            .split_once("\nAssistant:")
            .map(|(user, _)| user)
            .unwrap_or(prompt_body)
            .trim();
        ensure!(!user.is_empty(), "prompt user content is empty");
        Ok(user.to_owned())
    })
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
    prompt_templates: &PromptTemplates,
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

    let accepted_json = serde_json::to_string_pretty(
        &accepted
            .iter()
            .map(|task| json!({ "user": task.user }))
            .collect::<Vec<_>>(),
    )?;
    let source_sample_json = serde_json::to_string_pretty(&json!({
        "sample_id": sample.sample_id,
        "user": sample.source_user,
        "meta": sample.source_meta,
    }))?;
    Ok(render_prompt_template(
        &prompt_templates.generation,
        &[
            ("profile_name", prompt_templates.profile_name.clone()),
            ("variant_count", count.to_string()),
            ("source_prompt", sample.source_user.clone()),
            ("source_sample_json", source_sample_json),
            ("accepted_samples_json", accepted_json),
            ("accepted_block", accepted_block),
            ("feedback", feedback.unwrap_or_default().to_owned()),
            ("feedback_block", feedback_block),
        ],
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
    let user = sanitize_training_user_prompt(draft.user.trim());
    let answer = draft.answer.trim().to_owned();
    ensure!(!user.is_empty(), "generated user is empty");
    ensure!(!answer.is_empty(), "generated answer is empty");
    ensure!(
        !user.starts_with("Assistant:"),
        "generated item must be a user prompt, not an assistant response"
    );
    ensure!(
        !answer.starts_with("Assistant:"),
        "generated answer must be the final answer only, not a dialogue transcript"
    );

    let user_norm = normalize_compare_text(&user);
    let original_norm = normalize_compare_text(&sample.source_user);

    ensure!(
        user_norm != original_norm,
        "generated item is effectively identical to the original"
    );

    let item_json = serde_json::to_string(&GeneratedItemDraft {
        user: user.clone(),
        answer: answer.clone(),
        meta: draft.meta,
    })?;

    Ok(GeneratedItem {
        user,
        answer,
        item_json,
    })
}

async fn validate_generated_items_with_model(
    client: &OpenAiClient,
    model: &ModelConfig,
    json_output: bool,
    prompt_templates: &PromptTemplates,
    sample: &SourceSample,
    generated: &[GeneratedItemDraft],
) -> Result<()> {
    let prompt = build_generation_validation_prompt(prompt_templates, sample, generated)?;
    let result = client
        .chat(model, &prompt, json_output)
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
    prompt_templates: &PromptTemplates,
    sample: &SourceSample,
    generated: &[GeneratedItemDraft],
) -> Result<String> {
    let validation_template = prompt_templates.validation.as_ref().ok_or_else(|| {
        anyhow!(
            "prompt profile {} is missing validation template",
            prompt_templates.profile_name
        )
    })?;
    let source_sample_json = serde_json::to_string_pretty(&json!({
        "sample_id": sample.sample_id,
        "user": sample.source_user,
        "meta": sample.source_meta,
    }))?;
    let generated_candidates_json = serde_json::to_string_pretty(
        &generated
            .iter()
            .enumerate()
            .map(|(index, item)| {
                json!({
                    "index": index,
                    "user": item.user,
                    "answer": item.answer,
                    "meta": item.meta,
                })
            })
            .collect::<Vec<_>>(),
    )?;
    Ok(render_prompt_template(
        validation_template,
        &[
            ("profile_name", prompt_templates.profile_name.clone()),
            ("source_prompt", sample.source_user.clone()),
            ("source_sample_json", source_sample_json),
            ("generated_candidates_json", generated_candidates_json),
        ],
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
        (Some(reasoning), "") => format!("<think>\n{reasoning}\n</think>"),
        (Some(reasoning), content) => format!("<think>\n{reasoning}\n</think>\n\n{content}"),
        (None, content) => content.to_owned(),
    }
}

fn build_answer_prompt(user: &str) -> String {
    format!(
        r#"{user}

Additional output rules:
1. Keep the provider's native reasoning style if reasoning is produced; do not add a second reasoning block or rewrite the format.
2. If reasoning is produced, keep it very short and only include the key basis for choosing the correct option.
3. Do not repeat or paraphrase the question.
4. Do not restate or enumerate all choices unless absolutely necessary.
5. Do not include hesitation, self-dialogue, meta-commentary, or format self-checks.
6. Do not output phrases like "The answer is F", "Final answer", or "答案是F".
7. Do not use formatting like \boxed{{F}}.
8. The final answer must end with exactly one uppercase option letter only, such as F.
9. Do not output any extra explanation after the final option letter.
10. Bad style examples that must not appear:
- "First, let me analyze this question"
- "Now I will check each option"
- "The question is asking about..."
- "Therefore, the final answer is F"

Return the answer in the provider's native format, with concise reasoning if any, and end with a single uppercase option letter."#
    )
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

fn done_output_path(paths: &OutputPaths, correct: bool) -> &Path {
    if correct {
        &paths.done_success_jsonl_path
    } else {
        &paths.done_failed_jsonl_path
    }
}

fn expected_answer_from_generated_item_json(text: &str) -> Result<String> {
    let draft: GeneratedItemDraft = serde_json::from_str(text)
        .context("generated_item_json is not valid GeneratedItemDraft JSON")?;
    let answer = draft.answer.trim().to_owned();
    ensure!(!answer.is_empty(), "generated_item_json answer is empty");
    Ok(answer)
}

fn extract_predicted_answer(expected_answer: &str, content: &str, assistant: &str) -> String {
    let primary = if !content.trim().is_empty() {
        content.trim()
    } else {
        assistant.trim()
    };

    if canonical_answer_label(expected_answer).is_some() {
        return canonical_answer_label(primary).unwrap_or(label_fallback(primary));
    }

    last_non_empty_line(primary)
        .unwrap_or(primary)
        .trim()
        .to_owned()
}

fn compare_expected_and_predicted(expected: &str, predicted: &str) -> bool {
    if let Some(expected_label) = canonical_answer_label(expected) {
        return canonical_answer_label(predicted)
            .map(|predicted_label| predicted_label == expected_label)
            .unwrap_or(false);
    }

    normalize_compare_text(expected) == normalize_compare_text(predicted)
}

fn canonical_answer_label(text: &str) -> Option<String> {
    let mut labels = text
        .split(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | '.'
                        | ';'
                        | ':'
                        | '，'
                        | '。'
                        | '；'
                        | '：'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '（'
                        | '）'
                        | '【'
                        | '】'
                )
        })
        .filter_map(|token| {
            let token = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
            if token.len() == 1 {
                let ch = token.chars().next()?.to_ascii_uppercase();
                ch.is_ascii_uppercase().then(|| ch.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if let Some(label) = labels.pop() {
        return Some(label);
    }

    let chars = text.chars().collect::<Vec<_>>();
    for index in (0..chars.len()).rev() {
        let ch = chars[index].to_ascii_uppercase();
        if !ch.is_ascii_uppercase() {
            continue;
        }
        let prev_is_alpha = index > 0 && chars[index - 1].is_ascii_alphabetic();
        let next_is_alpha = index + 1 < chars.len() && chars[index + 1].is_ascii_alphabetic();
        if !prev_is_alpha && !next_is_alpha {
            return Some(ch.to_string());
        }
    }
    None
}

fn label_fallback(text: &str) -> String {
    last_non_empty_line(text).unwrap_or(text).trim().to_owned()
}

fn last_non_empty_line(text: &str) -> Option<&str> {
    text.lines().rev().find(|line| !line.trim().is_empty())
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
    match detect_input_format(path)? {
        InputFormat::JsonArray => load_input_rows_window_json_array(path, start_index, limit),
        InputFormat::JsonLines => load_input_rows_window_jsonl(path, start_index, limit),
    }
}

fn detect_input_format(path: &Path) -> Result<InputFormat> {
    let file = File::open(path)
        .with_context(|| format!("failed to open input file {}", path.display()))?;
    let mut reader = BufReader::new(file);

    loop {
        let buf = reader.fill_buf()?;
        ensure!(!buf.is_empty(), "input file {} is empty", path.display());

        let mut consumed = 0usize;
        for byte in buf {
            if byte.is_ascii_whitespace() {
                consumed += 1;
                continue;
            }
            return Ok(if *byte == b'[' {
                InputFormat::JsonArray
            } else {
                InputFormat::JsonLines
            });
        }

        reader.consume(consumed);
    }
}

fn load_input_rows_window_jsonl(
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

fn load_input_rows_window_json_array(
    path: &Path,
    start_index: usize,
    limit: Option<usize>,
) -> Result<Vec<(usize, Value)>> {
    let take_limit = limit.unwrap_or(usize::MAX);
    if take_limit == 0 {
        return Ok(Vec::new());
    }

    let file = File::open(path)
        .with_context(|| format!("failed to open input file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    JsonArrayWindowLoader {
        start_index,
        take_limit,
    }
    .deserialize(&mut deserializer)
    .with_context(|| format!("failed to parse JSON array input from {}", path.display()))
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
    let base_dir = output
        .jsonl_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("data"));
    let dataset_name = output
        .jsonl_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::trim)
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| anyhow!("output.jsonl_path must include a valid file stem"))?;
    let dataset_dir = base_dir.join(dataset_name);

    Ok(OutputPaths {
        generate_jsonl_path: dataset_dir.join("generate").join("tasks.jsonl"),
        done_success_jsonl_path: dataset_dir.join("done").join("success.jsonl"),
        done_failed_jsonl_path: dataset_dir.join("done").join("failed.jsonl"),
    })
}

fn prepare_output(paths: &OutputPaths, resume: bool) -> Result<()> {
    for parent in [
        paths.generate_jsonl_path.parent(),
        paths.done_success_jsonl_path.parent(),
        paths.done_failed_jsonl_path.parent(),
    ]
    .into_iter()
    .flatten()
    {
        fs::create_dir_all(parent)?;
    }
    if !resume {
        File::create(&paths.generate_jsonl_path)?;
        File::create(&paths.done_success_jsonl_path)?;
        File::create(&paths.done_failed_jsonl_path)?;
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
        (&paths.generate_jsonl_path, "generated output"),
        (&paths.done_success_jsonl_path, "done success output"),
        (&paths.done_failed_jsonl_path, "done failed output"),
    ] {
        for row in read_jsonl_if_exists::<OutputRow>(path, label)? {
            if let Some(existing) = rows.get_mut(&row.task_id) {
                merge_output_row(existing, row);
            } else {
                rows.insert(row.task_id.clone(), row);
            }
        }
    }

    Ok(rows)
}

fn merge_output_row(existing: &mut OutputRow, incoming: OutputRow) {
    existing.status = incoming.status;
    if !incoming.user.is_empty() {
        existing.user = incoming.user;
    }
    if !incoming.generated_item_json.is_empty() {
        existing.generated_item_json = incoming.generated_item_json;
    }
    if !incoming.answer_model.is_empty() {
        existing.answer_model = incoming.answer_model;
    }
    if !incoming.assistant.is_empty() {
        existing.assistant = incoming.assistant;
    }
    if !incoming.text.is_empty() {
        existing.text = incoming.text;
    }
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
    let expected_answer = expected_answer_from_generated_item_json(&row.generated_item_json)
        .with_context(|| {
            format!(
                "generated resume row has invalid generated_item_json for task {}",
                row.task_id
            )
        })?;
    Ok(PendingTask {
        task_id: row.task_id.clone(),
        user: user.to_owned(),
        expected_answer,
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

fn render_prompt_template(template: &str, vars: &[(&str, String)]) -> String {
    let mut rendered = template.to_owned();
    for (key, value) in vars {
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), value);
    }
    rendered
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
default_value!(default_validator_json_object_response, bool, true);

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

#[derive(Clone, Copy)]
enum InputFormat {
    JsonArray,
    JsonLines,
}

struct JsonArrayWindowLoader {
    start_index: usize,
    take_limit: usize,
}

impl<'de> DeserializeSeed<'de> for JsonArrayWindowLoader {
    type Value = Vec<(usize, Value)>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for JsonArrayWindowLoader {
    type Value = Vec<(usize, Value)>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a top-level JSON array of objects")
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut out = Vec::new();
        let mut logical_index = 0usize;

        while let Some(value) = seq.next_element::<Value>()? {
            if logical_index >= self.start_index && out.len() < self.take_limit {
                out.push((logical_index, value));
            }
            logical_index += 1;

            if out.len() >= self.take_limit {
                while seq.next_element::<de::IgnoredAny>()?.is_some() {}
                break;
            }
        }

        Ok(out)
    }
}
