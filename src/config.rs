use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

const ALLOWED_REASONING_EFFORTS: &[&str] = &["none", "low", "medium", "high", "xhigh"];

fn default_rows_api_url() -> String {
    "https://datasets-server.huggingface.co/rows".to_owned()
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub mmlu: MmluConfig,
    pub rewrite: RewriteConfig,
    pub answer_models: Vec<ModelConfig>,
    pub output: OutputConfig,
    pub run: RunConfig,
    pub concurrency: ConcurrencyConfig,
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MmluConfig {
    pub dataset_name: String,
    pub subset: String,
    pub split: String,
    #[serde(default = "default_rows_api_url")]
    pub rows_api_url: String,
    pub local_jsonl_path: Option<PathBuf>,
    pub limit: usize,
    pub start_index: usize,
}

#[derive(Debug, Clone)]
pub struct RewriteConfig {
    pub model: ModelConfig,
    pub variant_count: usize,
    pub parse_retry_times: usize,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub dir: PathBuf,
    pub original_jsonl: PathBuf,
    pub variants_jsonl: PathBuf,
    pub responses_jsonl: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunConfig {
    pub resume: bool,
    pub request_timeout_seconds: f64,
    pub max_retries: usize,
    pub retry_backoff_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    pub dataset_requests: usize,
    pub dataset_page_size: usize,
    pub rewrite_requests: usize,
    pub answer_requests: usize,
    pub original_writer_capacity: usize,
    pub variant_writer_capacity: usize,
    pub response_writer_capacity: usize,
    pub http_max_idle_per_host: usize,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub worker_threads: usize,
    pub max_blocking_threads: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RawAppConfig {
    mmlu: MmluConfig,
    rewrite: RawRewriteConfig,
    #[serde(default)]
    answer: RawAnswerConfig,
    answer_models: Vec<RawModelConfig>,
    output: RawOutputConfig,
    run: RunConfig,
    #[serde(default)]
    concurrency: RawConcurrencyConfig,
    #[serde(default)]
    runtime: RawRuntimeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRewriteConfig {
    model: String,
    base_url: String,
    api_key: String,
    #[serde(default)]
    enable_thinking: bool,
    reasoning_effort: Option<String>,
    variant_count: usize,
    parse_retry_times: usize,
    max_concurrency: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawModelConfig {
    name: String,
    base_url: String,
    api_key: String,
    #[serde(default)]
    enable_thinking: bool,
    reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawOutputConfig {
    dir: String,
    original_jsonl: String,
    variants_jsonl: String,
    responses_jsonl: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawAnswerConfig {
    max_concurrency: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawConcurrencyConfig {
    dataset_requests: Option<usize>,
    dataset_page_size: Option<usize>,
    rewrite_requests: Option<usize>,
    answer_requests: Option<usize>,
    original_writer_capacity: Option<usize>,
    variant_writer_capacity: Option<usize>,
    response_writer_capacity: Option<usize>,
    http_max_idle_per_host: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawRuntimeConfig {
    worker_threads: Option<usize>,
    max_blocking_threads: Option<usize>,
}

pub fn load_config(config_path: &Path) -> Result<AppConfig> {
    let raw_text = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config {}", config_path.display()))?;
    let raw: RawAppConfig =
        toml::from_str(&raw_text).with_context(|| "Failed to parse TOML config")?;
    raw.into_config(config_path.parent().unwrap_or_else(|| Path::new(".")))
}

impl RawAppConfig {
    fn into_config(self, config_dir: &Path) -> Result<AppConfig> {
        let cpu_count = thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(4);
        let dataset_requests = positive_or_default(
            self.concurrency.dataset_requests,
            4,
            "concurrency.dataset_requests",
        )?;
        let dataset_page_size = positive_or_default(
            self.concurrency.dataset_page_size,
            128,
            "concurrency.dataset_page_size",
        )?;
        let rewrite_requests = positive_or_default(
            self.concurrency
                .rewrite_requests
                .or(self.rewrite.max_concurrency),
            cpu_count.max(1),
            "concurrency.rewrite_requests",
        )?;
        let answer_requests = positive_or_default(
            self.concurrency
                .answer_requests
                .or(self.answer.max_concurrency),
            (cpu_count * 4).max(1),
            "concurrency.answer_requests",
        )?;
        let original_writer_capacity = positive_or_default(
            self.concurrency.original_writer_capacity,
            256,
            "concurrency.original_writer_capacity",
        )?;
        let variant_writer_capacity = positive_or_default(
            self.concurrency.variant_writer_capacity,
            1024,
            "concurrency.variant_writer_capacity",
        )?;
        let response_writer_capacity = positive_or_default(
            self.concurrency.response_writer_capacity,
            2048,
            "concurrency.response_writer_capacity",
        )?;
        let http_max_idle_per_host = positive_or_default(
            self.concurrency.http_max_idle_per_host,
            ((cpu_count * 8).max(32)).max(answer_requests),
            "concurrency.http_max_idle_per_host",
        )?;
        let worker_threads = positive_or_default(
            self.runtime.worker_threads,
            cpu_count.max(1),
            "runtime.worker_threads",
        )?;
        let runtime = RuntimeConfig {
            worker_threads,
            max_blocking_threads: positive_or_default(
                self.runtime.max_blocking_threads,
                (worker_threads * 8)
                    .max(answer_requests)
                    .max(rewrite_requests),
                "runtime.max_blocking_threads",
            )?,
        };

        let rewrite_model = ModelConfig {
            name: require_non_empty(self.rewrite.model, "rewrite.model")?,
            base_url: normalize_base_url(self.rewrite.base_url, "rewrite.base_url")?,
            api_key: require_non_empty(self.rewrite.api_key, "rewrite.api_key")?,
            enable_thinking: self.rewrite.enable_thinking,
            reasoning_effort: normalize_reasoning_effort(
                self.rewrite.reasoning_effort,
                "rewrite.reasoning_effort",
            )?,
        };

        let mut answer_models = Vec::with_capacity(self.answer_models.len());
        for (index, model) in self.answer_models.into_iter().enumerate() {
            let label = format!("answer_models[{index}]");
            answer_models.push(ModelConfig {
                name: require_non_empty(model.name, &format!("{label}.name"))?,
                base_url: normalize_base_url(model.base_url, &format!("{label}.base_url"))?,
                api_key: require_non_empty(model.api_key, &format!("{label}.api_key"))?,
                enable_thinking: model.enable_thinking,
                reasoning_effort: normalize_reasoning_effort(
                    model.reasoning_effort,
                    &format!("{label}.reasoning_effort"),
                )?,
            });
        }

        ensure!(
            !answer_models.is_empty(),
            "At least one answer model is required"
        );
        ensure!(
            self.rewrite.variant_count > 0,
            "rewrite.variant_count must be greater than 0"
        );
        ensure!(
            self.rewrite.variant_count % answer_models.len() == 0,
            "rewrite.variant_count must be divisible by len(answer_models)"
        );
        ensure!(
            self.run.request_timeout_seconds > 0.0,
            "run.request_timeout_seconds must be greater than 0"
        );
        ensure!(
            self.run.retry_backoff_seconds >= 0.0,
            "run.retry_backoff_seconds must be non-negative"
        );

        let concurrency = ConcurrencyConfig {
            dataset_requests,
            dataset_page_size,
            rewrite_requests,
            answer_requests,
            original_writer_capacity,
            variant_writer_capacity,
            response_writer_capacity,
            http_max_idle_per_host,
        };

        let output = OutputConfig {
            dir: resolve_path(config_dir, &self.output.dir),
            original_jsonl: resolve_path(config_dir, &self.output.original_jsonl),
            variants_jsonl: resolve_path(config_dir, &self.output.variants_jsonl),
            responses_jsonl: resolve_path(config_dir, &self.output.responses_jsonl),
        };

        Ok(AppConfig {
            mmlu: MmluConfig {
                dataset_name: require_non_empty(self.mmlu.dataset_name, "mmlu.dataset_name")?,
                subset: require_non_empty(self.mmlu.subset, "mmlu.subset")?,
                split: require_non_empty(self.mmlu.split, "mmlu.split")?,
                rows_api_url: require_non_empty(self.mmlu.rows_api_url, "mmlu.rows_api_url")?,
                local_jsonl_path: self
                    .mmlu
                    .local_jsonl_path
                    .map(|path| resolve_path_buf(config_dir, path)),
                limit: self.mmlu.limit,
                start_index: self.mmlu.start_index,
            },
            rewrite: RewriteConfig {
                model: rewrite_model,
                variant_count: self.rewrite.variant_count,
                parse_retry_times: self.rewrite.parse_retry_times,
            },
            answer_models,
            output,
            run: self.run,
            concurrency,
            runtime,
        })
    }
}

fn positive_or_default(value: Option<usize>, default: usize, field: &str) -> Result<usize> {
    let resolved = value.unwrap_or(default);
    ensure!(resolved > 0, "{field} must be greater than 0");
    Ok(resolved)
}

fn require_non_empty(value: String, field: &str) -> Result<String> {
    let trimmed = value.trim();
    ensure!(!trimmed.is_empty(), "{field} must be a non-empty string");
    Ok(trimmed.to_owned())
}

fn normalize_base_url(value: String, field: &str) -> Result<String> {
    let trimmed = require_non_empty(value, field)?;
    Ok(trimmed.trim_end_matches('/').to_owned())
}

fn normalize_reasoning_effort(value: Option<String>, field: &str) -> Result<Option<String>> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let normalized = raw.trim().to_lowercase();
    if normalized.is_empty() {
        bail!("{field} must be a non-empty string when provided");
    }
    ensure!(
        ALLOWED_REASONING_EFFORTS.contains(&normalized.as_str()),
        "{field} must be one of none/low/medium/high/xhigh"
    );
    Ok(Some(normalized))
}

fn resolve_path(config_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    resolve_path_buf(config_dir, path)
}

fn resolve_path_buf(config_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        config_dir.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_config(text: &str) -> AppConfig {
        let raw: RawAppConfig = toml::from_str(text).expect("valid test config");
        raw.into_config(Path::new("/tmp"))
            .expect("config should parse")
    }

    #[test]
    fn legacy_concurrency_fields_still_map_into_new_settings() {
        let config = parse_config(
            r#"
            [mmlu]
            dataset_name = "cais/mmlu"
            subset = "all"
            split = "test"
            limit = 1
            start_index = 0

            [rewrite]
            model = "gpt-5.4"
            base_url = "https://api.example.com/v1"
            api_key = "key"
            variant_count = 4
            parse_retry_times = 1
            max_concurrency = 5

            [answer]
            max_concurrency = 9

            [[answer_models]]
            name = "model-a"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [[answer_models]]
            name = "model-b"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [output]
            dir = "data"
            original_jsonl = "data/original.jsonl"
            variants_jsonl = "data/variants.jsonl"
            responses_jsonl = "data/responses.jsonl"

            [run]
            resume = true
            request_timeout_seconds = 30
            max_retries = 1
            retry_backoff_seconds = 1
            "#,
        );

        assert_eq!(config.concurrency.rewrite_requests, 5);
        assert_eq!(config.concurrency.answer_requests, 9);
        assert!(config.runtime.worker_threads > 0);
        assert!(config.runtime.max_blocking_threads >= config.runtime.worker_threads);
    }

    #[test]
    fn runtime_section_overrides_defaults() {
        let config = parse_config(
            r#"
            [mmlu]
            dataset_name = "cais/mmlu"
            subset = "all"
            split = "test"
            limit = 1
            start_index = 0

            [rewrite]
            model = "gpt-5.4"
            base_url = "https://api.example.com/v1"
            api_key = "key"
            variant_count = 4
            parse_retry_times = 1

            [[answer_models]]
            name = "model-a"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [[answer_models]]
            name = "model-b"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [output]
            dir = "data"
            original_jsonl = "data/original.jsonl"
            variants_jsonl = "data/variants.jsonl"
            responses_jsonl = "data/responses.jsonl"

            [run]
            resume = true
            request_timeout_seconds = 30
            max_retries = 1
            retry_backoff_seconds = 1

            [runtime]
            worker_threads = 3
            max_blocking_threads = 21
            "#,
        );

        assert_eq!(config.runtime.worker_threads, 3);
        assert_eq!(config.runtime.max_blocking_threads, 21);
    }

    #[test]
    fn local_jsonl_path_is_resolved_relative_to_config_dir() {
        let config = parse_config(
            r#"
            [mmlu]
            dataset_name = "local/questions"
            subset = "custom"
            split = "train"
            local_jsonl_path = "fixtures/source_questions.sample.jsonl"
            limit = 2
            start_index = 0

            [rewrite]
            model = "gpt-5.4"
            base_url = "https://api.example.com/v1"
            api_key = "key"
            variant_count = 4
            parse_retry_times = 1

            [[answer_models]]
            name = "model-a"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [[answer_models]]
            name = "model-b"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [output]
            dir = "data"
            original_jsonl = "data/original.jsonl"
            variants_jsonl = "data/variants.jsonl"
            responses_jsonl = "data/responses.jsonl"

            [run]
            resume = true
            request_timeout_seconds = 30
            max_retries = 1
            retry_backoff_seconds = 1
            "#,
        );

        assert_eq!(
            config.mmlu.local_jsonl_path,
            Some(Path::new("/tmp").join("fixtures/source_questions.sample.jsonl"))
        );
    }

    #[test]
    fn invalid_reasoning_effort_is_rejected() {
        let raw: RawAppConfig = toml::from_str(
            r#"
            [mmlu]
            dataset_name = "cais/mmlu"
            subset = "all"
            split = "test"
            limit = 1
            start_index = 0

            [rewrite]
            model = "gpt-5.4"
            base_url = "https://api.example.com/v1"
            api_key = "key"
            variant_count = 4
            parse_retry_times = 1
            reasoning_effort = "turbo"

            [[answer_models]]
            name = "model-a"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [[answer_models]]
            name = "model-b"
            base_url = "https://api.example.com/v1"
            api_key = "key"

            [output]
            dir = "data"
            original_jsonl = "data/original.jsonl"
            variants_jsonl = "data/variants.jsonl"
            responses_jsonl = "data/responses.jsonl"

            [run]
            resume = true
            request_timeout_seconds = 30
            max_retries = 1
            retry_backoff_seconds = 1
            "#,
        )
        .expect("valid raw toml");

        let error = raw.into_config(Path::new("/tmp")).expect_err("should fail");
        assert!(error.to_string().contains("reasoning_effort"));
    }
}
