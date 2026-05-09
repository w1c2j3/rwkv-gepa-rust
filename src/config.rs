use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub(crate) input: InputConfig,
    #[serde(default = "PromptConfig::default")]
    pub(crate) prompt: PromptConfig,
    pub(crate) generator: GeneratorConfig,
    #[serde(default)]
    pub(crate) validator: Option<ValidatorConfig>,
    pub(crate) answer_models: Vec<ModelConfig>,
    #[serde(default = "OutputConfig::default")]
    pub(crate) output: OutputConfig,
    #[serde(default = "RunConfig::default")]
    pub(crate) run: RunConfig,
    #[serde(default = "ConcurrencyConfig::default")]
    pub(crate) concurrency: ConcurrencyConfig,
}

#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromptConfig {
    #[serde(default)]
    pub(crate) profile_path: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputConfig {
    pub(crate) dataset_path: PathBuf,
    #[serde(default)]
    pub(crate) limit: Option<usize>,
    #[serde(default)]
    pub(crate) start_index: usize,
    #[serde(default = "default_subject", rename = "default_subject")]
    _default_subject: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GeneratorConfig {
    #[serde(flatten)]
    pub(crate) model: ModelConfig,
    pub(crate) variant_count: usize,
    #[serde(default = "default_generation_attempts")]
    pub(crate) generation_attempts: usize,
    #[serde(default = "default_validate_generated_questions")]
    pub(crate) validate_generated_questions: bool,
    #[serde(default = "default_generator_json_object_response")]
    pub(crate) json_object_response: bool,
    #[serde(default = "default_validator_json_object_response")]
    pub(crate) validator_json_object_response: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ValidatorConfig {
    #[serde(flatten)]
    pub(crate) model: ModelConfig,
    #[serde(default = "default_validator_json_object_response")]
    pub(crate) json_object_response: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelConfig {
    pub(crate) endpoint: String,
    pub(crate) model_name: String,
    pub(crate) api_key: String,
    #[serde(default)]
    pub(crate) system_prompt: Option<String>,
    #[serde(default)]
    pub(crate) max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) reasoning_effort: Option<String>,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) thinking: Option<ThinkingSetting>,
    #[serde(default)]
    pub(crate) enable_thinking: Option<bool>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputConfig {
    #[serde(default = "default_output_jsonl_path")]
    pub(crate) jsonl_path: PathBuf,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunConfig {
    #[serde(default = "default_resume")]
    pub(crate) resume: bool,
    #[serde(default = "default_answer_generated")]
    pub(crate) answer_generated: bool,
    #[serde(default = "default_request_timeout_seconds")]
    pub(crate) request_timeout_seconds: f64,
    #[serde(default = "default_disable_env_proxy")]
    pub(crate) disable_env_proxy: bool,
    #[serde(default = "default_force_http1")]
    pub(crate) force_http1: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConcurrencyConfig {
    #[serde(default = "default_generate_requests")]
    pub(crate) generate_requests: usize,
    #[serde(default = "default_answer_requests")]
    pub(crate) answer_requests: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ThinkingSetting {
    #[serde(rename = "type")]
    pub(crate) kind: String,
}

pub(crate) fn load_config(path: &Path) -> Result<Config> {
    let mut cfg: Config = toml::from_str(&fs::read_to_string(path)?)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    cfg.input.dataset_path = resolve(base, &cfg.input.dataset_path);
    if let Some(profile_path) = &mut cfg.prompt.profile_path {
        *profile_path = resolve(base, profile_path);
    }
    cfg.output.jsonl_path = resolve(base, &cfg.output.jsonl_path);
    trim_model_config(&mut cfg.generator.model);
    if let Some(validator) = &mut cfg.validator {
        trim_model_config(&mut validator.model);
    }
    cfg.answer_models.iter_mut().for_each(trim_model_config);
    Ok(cfg)
}

pub(crate) fn validate_config(cfg: &Config) -> Result<()> {
    ensure!(
        cfg.generator.variant_count > 0,
        "generator.variant_count must be > 0"
    );
    ensure!(
        cfg.generator.generation_attempts > 0,
        "generator.generation_attempts must be > 0"
    );
    if cfg.run.answer_generated {
        ensure!(
            !cfg.answer_models.is_empty(),
            "need at least one answer model"
        );
    }
    validate_model(&cfg.generator.model)?;
    if let Some(validator) = &cfg.validator {
        validate_model(&validator.model)?;
    }
    cfg.answer_models.iter().try_for_each(validate_model)?;
    Ok(())
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

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn default_subject() -> String {
    "general".to_owned()
}

fn default_output_jsonl_path() -> PathBuf {
    PathBuf::from("data/rwkv_train.jsonl")
}

fn default_resume() -> bool {
    true
}

fn default_answer_generated() -> bool {
    true
}

fn default_request_timeout_seconds() -> f64 {
    240.0
}

fn default_disable_env_proxy() -> bool {
    true
}

fn default_force_http1() -> bool {
    true
}

fn default_generate_requests() -> usize {
    4
}

fn default_answer_requests() -> usize {
    16
}

fn default_generation_attempts() -> usize {
    4
}

fn default_validate_generated_questions() -> bool {
    true
}

fn default_generator_json_object_response() -> bool {
    true
}

fn default_validator_json_object_response() -> bool {
    true
}

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
            answer_generated: default_answer_generated(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_path, write_text};

    fn write_config(text: &str) -> PathBuf {
        let path = temp_path("config/config.toml");
        write_text(&path, text);
        path
    }

    #[test]
    fn load_config_resolves_relative_paths_trims_models_and_applies_defaults() {
        let path = write_config(
            r#"
[input]
dataset_path = "input.jsonl"

[prompt]
profile_path = "prompts/custom.toml"

[generator]
endpoint = " https://generator.example/v1/chat/completions "
model_name = " generator-model "
api_key = " generator-key "
system_prompt = "   "
reasoning_effort = " low "
variant_count = 2

[output]
jsonl_path = "out/train.jsonl"

[validator]
endpoint = " https://validator.example/v1/chat/completions "
model_name = " validator-model "
api_key = " validator-key "
json_object_response = true

[[answer_models]]
endpoint = " https://answer.example/v1/chat/completions "
model_name = " answer-model "
api_key = " answer-key "
thinking = { type = " enabled " }
"#,
        );

        let cfg = load_config(&path).expect("config should load");
        let base = path.parent().expect("config should have parent");

        assert_eq!(cfg.input.dataset_path, base.join("input.jsonl"));
        assert_eq!(
            cfg.prompt.profile_path.as_deref(),
            Some(base.join("prompts/custom.toml").as_path())
        );
        assert_eq!(cfg.output.jsonl_path, base.join("out/train.jsonl"));
        assert_eq!(
            cfg.generator.model.endpoint,
            "https://generator.example/v1/chat/completions"
        );
        assert_eq!(cfg.generator.model.model_name, "generator-model");
        assert_eq!(cfg.generator.model.api_key, "generator-key");
        assert_eq!(cfg.generator.model.system_prompt, None);
        assert_eq!(cfg.generator.model.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(cfg.generator.generation_attempts, 4);
        assert!(cfg.generator.validate_generated_questions);
        let validator = cfg.validator.as_ref().expect("validator should load");
        assert_eq!(
            validator.model.endpoint,
            "https://validator.example/v1/chat/completions"
        );
        assert_eq!(validator.model.model_name, "validator-model");
        assert_eq!(validator.model.api_key, "validator-key");
        assert!(validator.json_object_response);
        assert_eq!(
            cfg.answer_models[0].thinking.as_ref().unwrap().kind,
            "enabled"
        );
        assert!(cfg.run.answer_generated);
        assert_eq!(cfg.run.request_timeout_seconds, 240.0);
        assert_eq!(cfg.concurrency.generate_requests, 4);
    }

    #[test]
    fn validate_config_rejects_invalid_pipeline_settings() {
        let path = write_config(
            r#"
[input]
dataset_path = "input.jsonl"

[generator]
endpoint = "https://generator.example/v1/chat/completions"
model_name = "generator-model"
api_key = "generator-key"
variant_count = 1

[[answer_models]]
endpoint = "https://answer.example/v1/chat/completions"
model_name = "answer-model"
api_key = "answer-key"
"#,
        );
        let mut cfg = load_config(&path).expect("config should load");

        validate_config(&cfg).expect("baseline config should validate");

        cfg.generator.variant_count = 0;
        assert!(
            validate_config(&cfg)
                .expect_err("zero variants should fail")
                .to_string()
                .contains("variant_count")
        );

        cfg.generator.variant_count = 1;
        cfg.answer_models.clear();
        assert!(
            validate_config(&cfg)
                .expect_err("missing answer model should fail")
                .to_string()
                .contains("answer model")
        );
    }
}
