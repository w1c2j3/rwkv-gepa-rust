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
    source_mcq: MultipleChoiceQuestion,
    original_correct_answer: String,
}

#[derive(Clone)]
struct GenerateJob {
    sample: SourceSample,
    missing_indices: Vec<usize>,
}

#[derive(Clone)]
struct PendingTask {
    task_id: String,
    user: String,
    generated_correct_answer: String,
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
    generated_correct_answer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    answer_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    assistant: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    text: String,
}

#[derive(Clone, Debug, Serialize)]
struct MultipleChoiceQuestion {
    preamble: String,
    question: String,
    choices: Vec<Choice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Choice {
    label: String,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedQuestionsEnvelope {
    questions: Vec<GeneratedQuestionDraft>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedQuestionDraft {
    new_question: String,
    new_choices: Vec<String>,
    new_correct_answer: String,
}

#[derive(Debug)]
struct GeneratedQuestionItem {
    question: String,
    choices: Vec<Choice>,
    correct_answer: String,
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
    messages: [ChatMessage<'a>; 1],
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
    let mut accepted = Vec::<PendingTask>::new();
    let total_target_count = job.missing_indices.len();

    for index in &job.missing_indices {
        let mut feedback = None::<String>;
        let mut last_error = None::<anyhow::Error>;

        for attempt in 0..generator.generation_attempts {
            let prompt = build_generation_prompt(
                &job.sample,
                total_target_count,
                &accepted,
                feedback.as_deref(),
            )?;
            let result = client
                .chat(
                    &generator.model,
                    &prompt,
                    generator.json_object_response,
                )
                .await
                .with_context(|| format!("generator failed for sample {}", job.sample.sample_id));

            let generated = match result {
                Ok(result) => parse_generated_questions(&result.content, 1, &job.sample)
                    .and_then(|mut items| {
                        items.pop().ok_or_else(|| anyhow!("generator returned no questions"))
                    })
                    .with_context(|| {
                        format!(
                            "invalid generated question for sample {} target index {} on logical attempt {}/{}",
                            job.sample.sample_id,
                            index,
                            attempt + 1,
                            generator.generation_attempts
                        )
                    }),
                Err(err) => Err(err),
            };

            let generated = match generated {
                Ok(generated) => generated,
                Err(err) if attempt + 1 < generator.generation_attempts => {
                    feedback = Some(err.to_string());
                    last_error = Some(err);
                    eprintln!(
                        "retrying generation for {} target index {} after parse/validation failure on attempt {}/{}",
                        job.sample.sample_id,
                        index,
                        attempt + 1,
                        generator.generation_attempts
                    );
                    continue;
                }
                Err(err) => return Err(err),
            };

            if generator.validate_generated_questions {
                if let Err(err) = validate_generated_questions_with_model(
                    &client,
                    &generator.model,
                    &job.sample,
                    std::slice::from_ref(&generated),
                )
                .await
                {
                    if attempt + 1 < generator.generation_attempts {
                        feedback = Some(err.to_string());
                        last_error = Some(err);
                        eprintln!(
                            "retrying generation for {} target index {} after model validation failure on attempt {}/{}",
                            job.sample.sample_id,
                            index,
                            attempt + 1,
                            generator.generation_attempts
                        );
                        continue;
                    }
                    return Err(err);
                }
            }

            let user = render_generated_user(
                &job.sample.source_mcq.preamble,
                &generated.question,
                &generated.choices,
            );
            let normalized_user = normalize_compare_text(&user);
            if accepted
                .iter()
                .any(|task| normalize_compare_text(&task.user) == normalized_user)
            {
                let err = anyhow!(
                    "generated question duplicates an already accepted variant for sample {}",
                    job.sample.sample_id
                );
                if attempt + 1 < generator.generation_attempts {
                    feedback = Some(err.to_string());
                    last_error = Some(err);
                    eprintln!(
                        "retrying generation for {} target index {} after duplicate detection on attempt {}/{}",
                        job.sample.sample_id,
                        index,
                        attempt + 1,
                        generator.generation_attempts
                    );
                    continue;
                }
                return Err(err);
            }

            let is_last_slot = accepted.len() + 1 == total_target_count;
            if total_target_count > 1 && is_last_slot {
                let distinct_labels = accepted
                    .iter()
                    .map(|task| task.generated_correct_answer.as_str())
                    .chain(std::iter::once(generated.correct_answer.as_str()))
                    .collect::<std::collections::BTreeSet<_>>();
                if distinct_labels.len() < 2 {
                    let err = anyhow!(
                        "final generated set would use only one correct-answer label; rewrite this question so the batch covers at least 2 different answer letters"
                    );
                    if attempt + 1 < generator.generation_attempts {
                        feedback = Some(err.to_string());
                        last_error = Some(err);
                        eprintln!(
                            "retrying generation for {} target index {} after answer-label diversity failure on attempt {}/{}",
                            job.sample.sample_id,
                            index,
                            attempt + 1,
                            generator.generation_attempts
                        );
                        continue;
                    }
                    return Err(err);
                }
            }

            accepted.push(PendingTask {
                task_id: task_id(&job.sample.sample_id, *index),
                user,
                generated_correct_answer: generated.correct_answer,
            });
            last_error = None;
            break;
        }

        if let Some(err) = last_error {
            return Err(err);
        }
    }

    Ok(accepted)
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
        status: RunStatus::Done.as_str().to_owned(),
        user: task.user,
        generated_correct_answer: task.generated_correct_answer,
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
    ChatRequest {
        model: &model.model_name,
        messages: [ChatMessage {
            role: "user",
            content: prompt,
        }],
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

fn normalize_sample(_input: &InputConfig, _index: usize, value: Value) -> Result<SourceSample> {
    let task_id = required_top_level_text(&value, "task_id")?;
    let sample_index = required_top_level_text(&value, "sample_index")?;
    let completions_id = required_top_level_text(&value, "completions_id")?;
    let context = required_top_level_text(&value, "context")?;
    let original_correct_answer =
        normalize_answer_label(&required_top_level_text(&value, "ref_answer")?)
            .ok_or_else(|| anyhow!("ref_answer must start with an option label"))?;
    let source_user = parse_single_turn_context(&context).with_context(|| {
        format!(
            "invalid single-turn context for sample {}",
            format!("{task_id}_{sample_index}_{completions_id}")
        )
    })?;
    let source_mcq = parse_multiple_choice_question(&source_user).with_context(|| {
        format!(
            "source user is not a supported multiple-choice prompt for sample {}",
            format!("{task_id}_{sample_index}_{completions_id}")
        )
    })?;
    ensure!(
        choice_text(&source_mcq.choices, &original_correct_answer).is_some(),
        "ref_answer {} does not match any parsed choice",
        original_correct_answer
    );
    Ok(SourceSample {
        sample_id: format!("{task_id}_{sample_index}_{completions_id}"),
        source_mcq,
        original_correct_answer,
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

fn parse_multiple_choice_question(text: &str) -> Result<MultipleChoiceQuestion> {
    let normalized = text.replace("\r\n", "\n");
    let lines = normalized.lines().collect::<Vec<_>>();
    let question_idx = lines
        .iter()
        .position(|line| line.trim_start().starts_with("Question:"))
        .ok_or_else(|| anyhow!("missing Question: line"))?;
    let choices_idx = lines
        .iter()
        .position(|line| line.trim_start().starts_with("Choices:"))
        .ok_or_else(|| anyhow!("missing Choices: line"))?;
    ensure!(
        choices_idx > question_idx,
        "Choices: must appear after Question:"
    );

    let preamble = lines[..question_idx].join("\n").trim().to_owned();

    let mut question_lines = Vec::new();
    let question_head = lines[question_idx].trim_start();
    question_lines.push(
        question_head
            .strip_prefix("Question:")
            .unwrap_or(question_head)
            .trim(),
    );
    question_lines.extend(
        lines[question_idx + 1..choices_idx]
            .iter()
            .map(|line| line.trim_end()),
    );
    let question = question_lines.join("\n").trim().to_owned();
    ensure!(!question.is_empty(), "question stem is empty");

    let mut choices = Vec::<Choice>::new();
    for line in &lines[choices_idx + 1..] {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((label, text)) = parse_choice_line(trimmed) {
            choices.push(Choice { label, text });
            continue;
        }
        if let Some(last) = choices.last_mut() {
            last.text.push(' ');
            last.text.push_str(trimmed);
            continue;
        }
        return Err(anyhow!(
            "invalid choice line before any parsed choice: {:?}",
            trimmed
        ));
    }

    ensure!(choices.len() >= 2, "need at least two parsed choices");
    let expected_labels = expected_labels(choices.len());
    let actual_labels = choices
        .iter()
        .map(|choice| choice.label.clone())
        .collect::<Vec<_>>();
    ensure!(
        actual_labels == expected_labels,
        "choices must use sequential labels {:?}, got {:?}",
        expected_labels,
        actual_labels
    );

    Ok(MultipleChoiceQuestion {
        preamble,
        question,
        choices,
    })
}

fn parse_choice_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    let mut chars = trimmed.char_indices();
    let (_, first) = chars.next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    let label = first.to_ascii_uppercase().to_string();
    let (delimiter_idx, delimiter) = chars.next()?;
    if !matches!(delimiter, '.' | '．' | '、' | ':' | '：' | ')' | '）') {
        return None;
    }
    let text = trimmed[delimiter_idx + delimiter.len_utf8()..].trim();
    (!text.is_empty()).then_some((label, text.to_owned()))
}

fn normalize_answer_label(text: &str) -> Option<String> {
    text.trim()
        .chars()
        .find(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_uppercase().to_string())
}

fn expected_labels(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| ((b'A' + index as u8) as char).to_string())
        .collect()
}

fn choice_text<'a>(choices: &'a [Choice], label: &str) -> Option<&'a str> {
    choices
        .iter()
        .find(|choice| choice.label == label)
        .map(|choice| choice.text.as_str())
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

fn build_generation_prompt(
    sample: &SourceSample,
    total_target_count: usize,
    accepted: &[PendingTask],
    feedback: Option<&str>,
) -> Result<String> {
    let original_correct_text =
        choice_text(&sample.source_mcq.choices, &sample.original_correct_answer)
            .ok_or_else(|| anyhow!("original correct answer is missing from parsed choices"))?;
    let choice_count = sample.source_mcq.choices.len();
    let original_labels = expected_labels(sample.source_mcq.choices.len()).join(", ");
    let original_user = render_generated_user(
        &sample.source_mcq.preamble,
        &sample.source_mcq.question,
        &sample.source_mcq.choices,
    );

    let feedback_block = feedback
        .map(|text| {
            format!(
                "\n上一轮失败原因：{}\n请只修复这些问题，不要重复犯错。\n",
                text
            )
        })
        .unwrap_or_default();
    let accepted_block = if accepted.is_empty() {
        String::new()
    } else {
        format!(
            "已接受题：{}\n",
            serde_json::to_string(
                &accepted
                    .iter()
                    .map(|task| {
                        json!({
                            "user": task.user,
                            "answer": task.generated_correct_answer,
                        })
                    })
                    .collect::<Vec<_>>()
            )?
        )
    };
    let diversity_block = if total_target_count > 1 {
        let accepted_labels = accepted
            .iter()
            .map(|task| task.generated_correct_answer.as_str())
            .collect::<Vec<_>>();
        format!(
            "\n本样本总共需要生成 {total_target_count} 道新题。当前已接受 {} 道，新题正确答案字母依次为 {:?}。最终整组结果至少要覆盖 2 种不同的正确答案字母。\n",
            accepted.len(),
            accepted_labels
        )
    } else {
        String::new()
    };
    Ok(format!(
        "生成1道与原题相关但本质不同的新选择题。\n\
先换掉原正确答案，再重写题干和4个选项，使 new_correct_answer 成为唯一正确答案。\n\
返回纯JSON：{{\"questions\":[{{\"new_question\":\"...\",\"new_choices\":[\"A. ...\",\"B. ...\",\"C. ...\",\"D. ...\"],\"new_correct_answer\":\"A\"}}]}}\n\
要求：同知识域和风格；不是同义改写；不再问原题同一个判断对象；至少改一个决定答案的判定点；new_correct_answer 不能等于原答案，新的正确选项内容也不能等于原答案内容；4个选项全部重写、同层级、不能复用原选项文本或仅换序；题面本身足以唯一判断；不要角色扮演；不要冷门特例；不要与已接受题重复。\n\
自检：不是原题，不是表述改写，先换答案再重写，原答案未保留，只有一个合理答案。\n\
原始样本：{}\n{}{}输出结构：{}\n{}",
        serde_json::to_string_pretty(&json!({
            "sample_id": sample.sample_id,
            "original_user": original_user,
            "preamble": sample.source_mcq.preamble,
            "question": sample.source_mcq.question,
            "choices": sample.source_mcq.choices,
            "original_correct_answer": sample.original_correct_answer,
            "original_correct_choice_text": original_correct_text,
        }))?,
        accepted_block,
        diversity_block,
        serde_json::to_string_pretty(&json!({
            "question_count": 1,
            "choice_count": choice_count,
            "option_labels": original_labels,
        }))?,
        feedback_block
    ))
}

fn parse_generated_questions(
    text: &str,
    expected: usize,
    sample: &SourceSample,
) -> Result<Vec<GeneratedQuestionItem>> {
    let envelope = parse_generated_questions_envelope(text)?;
    ensure!(
        envelope.questions.len() == expected,
        "expected exactly {expected} generated questions, got {}",
        envelope.questions.len()
    );

    let mut seen_rendered = Vec::<String>::new();
    let mut out = Vec::with_capacity(envelope.questions.len());
    for (index, draft) in envelope.questions.into_iter().enumerate() {
        let validated = validate_generated_question(draft, sample)
            .with_context(|| format!("generated question #{index} failed validation"))?;
        let rendered = render_generated_user(
            &sample.source_mcq.preamble,
            &validated.question,
            &validated.choices,
        );
        let normalized_rendered = normalize_compare_text(&rendered);
        ensure!(
            !seen_rendered
                .iter()
                .any(|existing| existing == &normalized_rendered),
            "generated question #{index} is a duplicate of another generated question"
        );
        seen_rendered.push(normalized_rendered);
        out.push(validated);
    }
    Ok(out)
}

fn parse_generated_questions_envelope(text: &str) -> Result<GeneratedQuestionsEnvelope> {
    serde_json::from_str(text)
        .or_else(|_| {
            extract_json_object_from_text(text)
                .ok_or_else(|| serde_json::Error::io(std::io::Error::other("no json object")))
                .and_then(|json| serde_json::from_str(&json))
        })
        .with_context(|| {
            format!(
                "generator did not return valid JSON or an extractable JSON object: {:?}",
                preview_text(text, 200)
            )
        })
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

fn validate_generated_question(
    mut draft: GeneratedQuestionDraft,
    sample: &SourceSample,
) -> Result<GeneratedQuestionItem> {
    draft.new_question = draft.new_question.trim().to_owned();
    draft.new_correct_answer = normalize_answer_label(&draft.new_correct_answer)
        .ok_or_else(|| anyhow!("correct_answer must start with an option label"))?;

    ensure!(
        !draft.new_question.is_empty(),
        "generated question stem is empty"
    );
    ensure!(
        !draft.new_question.contains("Question:"),
        "generated new_question must not contain a Question: prefix"
    );
    ensure!(
        !draft.new_question.contains("Choices:"),
        "generated new_question must not contain a Choices: block"
    );

    let choices = draft
        .new_choices
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let trimmed = line.trim();
            let (label, text) = parse_choice_line(trimmed).ok_or_else(|| {
                anyhow!(
                    "generated new_choices[{}] must look like \"A. ...\", got {:?}",
                    index,
                    preview_text(trimmed, 80)
                )
            })?;
            Ok(Choice { label, text })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut item = GeneratedQuestionItem {
        question: draft.new_question,
        choices,
        correct_answer: draft.new_correct_answer,
    };

    ensure!(
        item.choices.len() == sample.source_mcq.choices.len(),
        "generated question must contain exactly {} choices, got {}",
        sample.source_mcq.choices.len(),
        item.choices.len()
    );

    for choice in &mut item.choices {
        choice.label = normalize_answer_label(&choice.label)
            .ok_or_else(|| anyhow!("choice label must start with an option letter"))?;
        choice.text = choice.text.trim().to_owned();
        ensure!(
            !choice.text.is_empty(),
            "generated choice {} is empty",
            choice.label
        );
    }

    let expected_labels = expected_labels(item.choices.len());
    let actual_labels = item
        .choices
        .iter()
        .map(|choice| choice.label.clone())
        .collect::<Vec<_>>();
    ensure!(
        actual_labels == expected_labels,
        "generated choices must use sequential labels {:?}, got {:?}",
        expected_labels,
        actual_labels
    );
    ensure!(
        item.correct_answer != sample.original_correct_answer,
        "generated correct answer {} is unchanged from the original answer",
        item.correct_answer
    );

    let generated_correct_text =
        choice_text(&item.choices, &item.correct_answer).ok_or_else(|| {
            anyhow!(
                "generated correct answer {} is missing from choices",
                item.correct_answer
            )
        })?;
    let original_correct_text =
        choice_text(&sample.source_mcq.choices, &sample.original_correct_answer)
            .ok_or_else(|| anyhow!("original correct answer is missing from parsed choices"))?;
    ensure!(
        !texts_too_similar(generated_correct_text, original_correct_text),
        "generated correct choice text is still too similar to the original correct answer"
    );

    let generated_question_norm = normalize_compare_text(&item.question);
    let original_question_norm = normalize_compare_text(&sample.source_mcq.question);
    let generated_choice_texts = item
        .choices
        .iter()
        .map(|choice| normalize_compare_text(&choice.text))
        .collect::<Vec<_>>();
    let original_choice_texts = sample
        .source_mcq
        .choices
        .iter()
        .map(|choice| normalize_compare_text(&choice.text))
        .collect::<Vec<_>>();
    ensure!(
        generated_question_norm != original_question_norm
            || generated_choice_texts != original_choice_texts,
        "generated question is effectively identical to the original question"
    );
    ensure!(
        generated_choice_texts != original_choice_texts,
        "generated options are identical to the original ordered options"
    );
    let mut generated_choice_texts_sorted = generated_choice_texts.clone();
    generated_choice_texts_sorted.sort();
    let mut original_choice_texts_sorted = original_choice_texts.clone();
    original_choice_texts_sorted.sort();
    ensure!(
        generated_choice_texts_sorted != original_choice_texts_sorted,
        "generated options are just the original options in a different order"
    );

    let reused_choice_count = item
        .choices
        .iter()
        .filter(|choice| {
            sample
                .source_mcq
                .choices
                .iter()
                .any(|original| texts_too_similar(&choice.text, &original.text))
        })
        .count();
    ensure!(
        reused_choice_count + 1 < item.choices.len(),
        "generated options reuse too many original choices; they must be rewritten around the new decision point"
    );

    let has_duplicate_choice_text =
        generated_choice_texts
            .iter()
            .enumerate()
            .any(|(index, left)| {
                generated_choice_texts
                    .iter()
                    .skip(index + 1)
                    .any(|right| left == right)
            });
    ensure!(
        !has_duplicate_choice_text,
        "generated options contain duplicated choice text"
    );

    ensure!(
        !contains_ambiguous_negative_stem(&item.question),
        "generated question uses an ambiguous negative/exception stem; rewrite it as a positive, uniquely answerable question"
    );

    Ok(item)
}

async fn validate_generated_questions_with_model(
    client: &OpenAiClient,
    model: &ModelConfig,
    sample: &SourceSample,
    generated: &[GeneratedQuestionItem],
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
        "model validation rejected generated questions: {}",
        invalid_reasons.join(" | ")
    );
    Ok(())
}

fn build_generation_validation_prompt(
    sample: &SourceSample,
    generated: &[GeneratedQuestionItem],
) -> Result<String> {
    let original_correct_text =
        choice_text(&sample.source_mcq.choices, &sample.original_correct_answer)
            .ok_or_else(|| anyhow!("original correct answer is missing from parsed choices"))?;
    Ok(format!(
        "你是选择题变体质检器。\n\
返回纯 JSON：{{\"items\":[{{\"index\":0,\"valid\":true,\"reason\":\"...\"}}]}}\n\
items 长度必须与候选题数量一致，每项只能有 index、valid、reason。\n\n\
只有同时满足以下条件时才判 valid=true：\n\
1. 保持同一知识域、题型风格、语言风格，表面形式尽量接近原题。\n\
2. 不是同义改写；如果仍在问原题同一个判断对象，则不通过。\n\
3. 候选题是先换掉原正确答案，再围绕新答案重写的；不是在原题上只做表面改写。\n\
4. 至少改动了一个真正决定答案的判定点，例如比较对象、适用条件、规则对象、制度目的、适用范围、概念边界、例外条件、排序依据。\n\
5. 新正确答案是因为判定依据变化而改变，不是只换字母，也不是原答案内容的近义改写。\n\
6. 选项围绕新判定点重写，不是原选项复用或换序，且四个选项处于同一概念层级。\n\
7. 不需要外部检索；题面本身足以支持唯一答案。\n\
8. 没有退化成特别偏、特别窄或特别冷门的低质量特例题。\n\
9. declared correct_answer 确实是唯一最佳答案。\n\n\
不通过时，reason 只写最关键的失败原因，例如：仍是原问题 / 表述改写 / 未先换答案 / 答案未变 / 需要外部检索 / 选项层级不一致 / 题目有歧义。\n\n\
原始样本：\n{}\n\n\
候选题：\n{}\n",
        serde_json::to_string_pretty(&json!({
            "sample_id": sample.sample_id,
            "question": sample.source_mcq.question,
            "choices": sample.source_mcq.choices,
            "original_correct_answer": sample.original_correct_answer,
            "original_correct_choice_text": original_correct_text,
        }))?,
        serde_json::to_string_pretty(
            &generated
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    json!({
                        "index": index,
                        "question": item.question,
                        "choices": item.choices,
                        "correct_answer": item.correct_answer,
                    })
                })
                .collect::<Vec<_>>()
        )?
    ))
}

fn render_generated_user(preamble: &str, question: &str, choices: &[Choice]) -> String {
    let mut lines = Vec::<String>::new();
    let sanitized_preamble = sanitize_preamble(preamble);
    if !sanitized_preamble.is_empty() {
        lines.push(sanitized_preamble);
    }

    let mut question_lines = question.trim().lines();
    if let Some(first) = question_lines.next() {
        lines.push(format!("Question: {}", first.trim()));
    }
    lines.extend(question_lines.map(|line| line.trim_end().to_owned()));
    lines.push("Choices:".to_owned());
    lines.extend(
        choices
            .iter()
            .map(|choice| format!("{}. {}", choice.label, choice.text.trim())),
    );
    lines.join("\n")
}

fn sanitize_preamble(preamble: &str) -> String {
    preamble
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !looks_like_roleplay_line(line))
        .map(str::to_owned)
        .collect::<Vec<_>>()
        .join("\n")
}

fn looks_like_roleplay_line(line: &str) -> bool {
    let normalized = line.trim().to_ascii_lowercase();
    (normalized.starts_with("you are ")
        || normalized.starts_with("you are a ")
        || normalized.starts_with("you are an "))
        && (normalized.contains("expert")
            || normalized.contains("assistant")
            || normalized.contains("teacher")
            || normalized.contains("tutor")
            || normalized.contains("talented"))
        || ((line.contains("你是") || line.contains("请你作为"))
            && (line.contains("专家")
                || line.contains("老师")
                || line.contains("助手")
                || line.contains("顾问")
                || line.contains("学者")))
}

fn contains_ambiguous_negative_stem(question: &str) -> bool {
    [
        "最不适合",
        "最不是",
        "最不能",
        "最不可能",
        "最不恰当",
        "最不属于",
        "最不正确",
        "错误的是",
        "不正确的是",
    ]
    .iter()
    .any(|pattern| question.contains(pattern))
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

fn texts_too_similar(left: &str, right: &str) -> bool {
    let left = normalize_compare_text(left);
    let right = normalize_compare_text(right);
    if left == right {
        return true;
    }
    let (shorter, longer) = if left.len() <= right.len() {
        (&left, &right)
    } else {
        (&right, &left)
    };
    !shorter.is_empty() && longer.contains(shorter) && shorter.len() * 5 >= longer.len() * 4
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
        status: RunStatus::Generated.as_str().to_owned(),
        user: task.user.clone(),
        generated_correct_answer: task.generated_correct_answer.clone(),
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
    Ok(read_jsonl_if_exists::<OutputRow>(path, "output")?
        .into_iter()
        .map(|row| (row.task_id.clone(), row))
        .collect())
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
        generated_correct_answer: row.generated_correct_answer.trim().to_owned(),
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
