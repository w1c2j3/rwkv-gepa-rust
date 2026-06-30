use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::PromptConfig;
use crate::task::{AnswerCheckMode, AnswerStyle, TaskKind};
use crate::types::{GeneratedItemDraft, PendingTask, SourceSample};

#[derive(Clone)]
pub(crate) struct PromptTemplates {
    pub(crate) profile_name: String,
    pub(crate) task_kind: TaskKind,
    pub(crate) answer_check: AnswerCheckMode,
    pub(crate) answer_style: AnswerStyle,
    synthesis: Option<SynthesisProfile>,
    generation: String,
    validation: Option<String>,
}

#[derive(Clone)]
pub(crate) struct PromptLibrary {
    default: PromptTemplates,
    profiles: HashMap<String, PromptTemplates>,
    selector_keys: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptProfileFile {
    name: String,
    #[serde(default)]
    task_kind: Option<TaskKind>,
    #[serde(default)]
    answer_check: Option<AnswerCheckMode>,
    #[serde(default)]
    answer_style: Option<AnswerStyle>,
    #[serde(default)]
    synthesis: Option<SynthesisProfile>,
    generation: PromptTemplateSection,
    #[serde(default)]
    validation: Option<PromptTemplateSection>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SynthesisProfile {
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    axes: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptTemplateSection {
    template: String,
}

pub(crate) fn load_prompt_library(
    prompt: &PromptConfig,
    validate_generated_questions: bool,
) -> Result<PromptLibrary> {
    let default = load_prompt_templates_from_text(
        &default_profile_text(prompt)?,
        prompt,
        validate_generated_questions,
    )?;
    let mut profiles = HashMap::new();
    for (name, path) in &prompt.profiles {
        let profile_text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt profile TOML {}", path.display()))?;
        let template =
            load_prompt_templates_from_text(&profile_text, prompt, validate_generated_questions)?;
        profiles.insert(normalize_profile_key(name), template.clone());
        profiles.insert(normalize_profile_key(&template.profile_name), template);
    }
    Ok(PromptLibrary {
        default,
        profiles,
        selector_keys: prompt.selector_keys.clone(),
    })
}

fn default_profile_text(prompt: &PromptConfig) -> Result<String> {
    match &prompt.profile_path {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt profile TOML {}", path.display())),
        None => Ok(include_str!("../prompts/knowledge.toml").to_owned()),
    }
}

pub(crate) fn load_prompt_templates(
    prompt: &PromptConfig,
    validate_generated_questions: bool,
) -> Result<PromptTemplates> {
    load_prompt_templates_from_text(
        &default_profile_text(prompt)?,
        prompt,
        validate_generated_questions,
    )
}

fn load_prompt_templates_from_text(
    profile_text: &str,
    prompt: &PromptConfig,
    validate_generated_questions: bool,
) -> Result<PromptTemplates> {
    let profile: PromptProfileFile =
        toml::from_str(profile_text).context("failed to parse prompt profile TOML")?;
    ensure!(
        !profile.name.trim().is_empty(),
        "prompt profile name must not be empty"
    );
    ensure!(
        !profile.generation.template.trim().is_empty(),
        "prompt profile generation.template must not be empty"
    );
    if validate_generated_questions {
        let validation = profile.validation.as_ref().ok_or_else(|| {
            anyhow!("prompt profile is missing [validation] while generator.validate_generated_questions=true")
        })?;
        ensure!(
            !validation.template.trim().is_empty(),
            "prompt profile validation.template must not be empty"
        );
    }
    let task_kind = profile.task_kind.unwrap_or(prompt.task_kind);
    let answer_check = profile
        .answer_check
        .or(prompt.answer_check)
        .unwrap_or_else(|| task_kind.default_answer_check());
    let answer_style = profile
        .answer_style
        .unwrap_or_else(|| task_kind.default_answer_style());

    Ok(PromptTemplates {
        profile_name: profile.name.trim().to_owned(),
        task_kind,
        answer_check,
        answer_style,
        synthesis: profile.synthesis,
        generation: profile.generation.template.trim().to_owned(),
        validation: profile
            .validation
            .map(|section| section.template.trim().to_owned())
            .filter(|text| !text.is_empty()),
    })
}

pub(crate) fn build_generation_prompt(
    prompt_templates: &PromptTemplates,
    sample: &SourceSample,
    count: usize,
    accepted: &[PendingTask],
    feedback: Option<&str>,
) -> Result<String> {
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
    let answer_style = source_answer_style(sample, prompt_templates.answer_style);
    let synthesis_profile_json =
        serde_json::to_string_pretty(&prompt_templates.synthesis_json(answer_style))?;
    Ok(render_prompt_template(
        &prompt_templates.generation,
        &[
            ("profile_name", prompt_templates.profile_name.clone()),
            ("task_kind", prompt_templates.task_kind.as_str().to_owned()),
            (
                "answer_check",
                prompt_templates.answer_check.as_str().to_owned(),
            ),
            ("answer_style", answer_style.as_str().to_owned()),
            ("variant_count", count.to_string()),
            ("source_sample_json", source_sample_json),
            ("accepted_samples_json", accepted_json),
            ("synthesis_profile_json", synthesis_profile_json),
            ("feedback_block", feedback_block),
        ],
    )?)
}

pub(crate) fn build_generation_validation_prompt(
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
    let answer_style = source_answer_style(sample, prompt_templates.answer_style);
    let synthesis_profile_json =
        serde_json::to_string_pretty(&prompt_templates.synthesis_json(answer_style))?;
    Ok(render_prompt_template(
        validation_template,
        &[
            ("profile_name", prompt_templates.profile_name.clone()),
            ("task_kind", prompt_templates.task_kind.as_str().to_owned()),
            (
                "answer_check",
                prompt_templates.answer_check.as_str().to_owned(),
            ),
            ("answer_style", answer_style.as_str().to_owned()),
            ("source_sample_json", source_sample_json),
            ("generated_candidates_json", generated_candidates_json),
            ("synthesis_profile_json", synthesis_profile_json),
        ],
    )?)
}

impl PromptTemplates {
    fn synthesis_json(&self, answer_style: AnswerStyle) -> serde_json::Value {
        let (strategy, axes) = self
            .synthesis
            .as_ref()
            .map(|synthesis| {
                (
                    synthesis
                        .strategy
                        .as_deref()
                        .map(str::trim)
                        .filter(|text| !text.is_empty())
                        .unwrap_or("profile_default")
                        .to_owned(),
                    synthesis
                        .axes
                        .iter()
                        .map(|axis| axis.trim())
                        .filter(|axis| !axis.is_empty())
                        .collect::<Vec<_>>(),
                )
            })
            .unwrap_or_else(|| ("profile_default".to_owned(), Vec::new()));

        json!({
            "task_kind": self.task_kind.as_str(),
            "answer_check": self.answer_check.as_str(),
            "answer_style": answer_style.as_str(),
            "profile_default_answer_style": self.answer_style.as_str(),
            "strategy": strategy,
            "axes": axes,
        })
    }
}

pub(crate) fn source_answer_style(
    sample: &SourceSample,
    profile_default: AnswerStyle,
) -> AnswerStyle {
    if source_declares_cot(&sample.source_meta) {
        return AnswerStyle::Cot;
    }
    if let Some(style) = sample
        .source_meta
        .get("answer_style")
        .and_then(answer_style_from_value)
    {
        return style;
    }
    for key in ["ref_answer", "answer", "expected_answer"] {
        if sample
            .source_meta
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(contains_cot_marker)
        {
            return AnswerStyle::Cot;
        }
    }
    profile_default
}

fn answer_style_from_value(value: &Value) -> Option<AnswerStyle> {
    let raw = value.as_str()?.trim();
    match normalize_profile_key(raw).as_str() {
        "final_only" | "final" | "short_answer" | "answer_only" => Some(AnswerStyle::FinalOnly),
        "brief_explanation" => Some(AnswerStyle::BriefExplanation),
        "json_object" | "json" => Some(AnswerStyle::JsonObject),
        "code" => Some(AnswerStyle::Code),
        "function_call" | "function_calling" | "tool_call" => Some(AnswerStyle::FunctionCall),
        "cot" | "fakecot" | "fake_cot" | "chain_of_thought" => Some(AnswerStyle::Cot),
        _ => None,
    }
}

fn source_declares_cot(meta: &Value) -> bool {
    for key in [
        "cot_mode",
        "score_cot_mode",
        "prompt_profile",
        "cot_profile",
        "answer_style",
    ] {
        if meta.get(key).is_some_and(value_declares_cot) {
            return true;
        }
    }
    meta.get("sampling_config").is_some_and(value_declares_cot)
}

fn value_declares_cot(value: &Value) -> bool {
    match value {
        Value::String(text) => text_declares_cot(text),
        Value::Array(items) => items.iter().any(value_declares_cot),
        Value::Object(object) => object.values().any(value_declares_cot),
        _ => false,
    }
}

fn text_declares_cot(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let key = normalize_profile_key(trimmed);
    if matches!(key.as_str(), "nocot" | "no_cot" | "no_chain_of_thought") {
        return false;
    }
    if matches!(
        key.as_str(),
        "cot" | "fakecot" | "fake_cot" | "chain_of_thought"
    ) {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    lower.contains("fakecot")
        || lower.contains("fake_cot")
        || lower.contains("chain-of-thought")
        || lower.contains("chain of thought")
        || lower
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .any(|part| part == "cot")
}

fn contains_cot_marker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("<think")
        || lower.contains("(think)")
        || lower.contains("step by step")
        || lower.contains("chain-of-thought")
        || lower.contains("chain of thought")
        || text.contains("思考过程")
        || text.contains("推理过程")
        || text.contains("解题步骤")
        || text.contains("逐步推理")
}

impl PromptLibrary {
    pub(crate) fn default_templates(&self) -> &PromptTemplates {
        &self.default
    }

    pub(crate) fn templates_for_sample(&self, sample: &SourceSample) -> PromptTemplates {
        for key in &self.selector_keys {
            let Some(value) = sample
                .source_meta
                .get(key)
                .and_then(scalar_profile_value)
                .map(normalize_profile_key)
            else {
                continue;
            };
            if let Some(profile) = self.profiles.get(&value) {
                return profile.clone();
            }
        }
        self.default.clone()
    }
}

fn scalar_profile_value(value: &serde_json::Value) -> Option<&str> {
    match value {
        serde_json::Value::String(text) => Some(text.trim()).filter(|text| !text.is_empty()),
        _ => None,
    }
}

fn normalize_profile_key(value: &str) -> String {
    let mut out = String::new();
    let mut last_was_sep = false;
    for ch in value.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    out.trim_matches('_').to_owned()
}

fn render_prompt_template(template: &str, vars: &[(&str, String)]) -> Result<String> {
    let mut rendered = template.to_owned();
    for (key, value) in vars {
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), value);
    }
    ensure!(
        !rendered.contains("{{"),
        "prompt template contains unresolved placeholders"
    );
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{pending_task, sample};
    use serde_json::json;

    #[test]
    fn built_in_prompt_profile_loads_generation_and_validation_templates() {
        let templates = load_prompt_templates(&PromptConfig::default(), true)
            .expect("built-in prompt should load");

        assert!(!templates.profile_name.is_empty());
        assert_eq!(templates.task_kind, TaskKind::Knowledge);
        assert_eq!(templates.answer_check, AnswerCheckMode::SingleLabel);
        assert_eq!(templates.answer_style, AnswerStyle::FinalOnly);
        assert!(templates.generation.contains("{{source_sample_json}}"));
        assert!(
            templates
                .validation
                .as_deref()
                .unwrap_or_default()
                .contains("{{generated_candidates_json}}")
        );
    }

    #[test]
    fn prompt_profile_can_be_loaded_from_configured_path() {
        let path = crate::test_support::temp_path("prompt/custom_profile.toml");
        crate::test_support::write_text(
            &path,
            r#"
name = "custom"
task_kind = "coding"
answer_check = "disabled"

[generation]
template = "generate {{task_kind}} {{answer_check}} {{synthesis_profile_json}} {{source_sample_json}}"

[validation]
template = "validate {{generated_candidates_json}}"
"#,
        );
        let config = PromptConfig {
            profile_path: Some(path),
            ..PromptConfig::default()
        };

        let templates = load_prompt_templates(&config, true).expect("custom prompt should load");

        assert_eq!(templates.profile_name, "custom");
        assert_eq!(templates.task_kind, TaskKind::Coding);
        assert_eq!(templates.answer_check, AnswerCheckMode::Disabled);
        assert_eq!(templates.answer_style, AnswerStyle::Code);
        assert_eq!(
            templates.generation,
            "generate {{task_kind}} {{answer_check}} {{synthesis_profile_json}} {{source_sample_json}}"
        );
        assert_eq!(
            templates.validation.as_deref(),
            Some("validate {{generated_candidates_json}}")
        );
    }

    #[test]
    fn prompt_library_routes_samples_by_meta_selector() {
        let default_path = crate::test_support::temp_path("prompt/default_profile.toml");
        let code_path = crate::test_support::temp_path("prompt/code_profile.toml");
        crate::test_support::write_text(
            &default_path,
            r#"
name = "default_mcq"
task_kind = "knowledge"

[generation]
template = "default {{source_sample_json}}"

[validation]
template = "default validate {{generated_candidates_json}}"
"#,
        );
        crate::test_support::write_text(
            &code_path,
            r#"
name = "coding"
task_kind = "coding"
answer_check = "disabled"

[generation]
template = "coding {{task_kind}} {{source_sample_json}}"

[validation]
template = "coding validate {{generated_candidates_json}}"
"#,
        );
        let mut config = PromptConfig {
            profile_path: Some(default_path),
            ..PromptConfig::default()
        };
        config.profiles.insert("coding".to_owned(), code_path);

        let library = load_prompt_library(&config, true).expect("profiles should load");
        let sample = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Write coding".to_owned(),
            source_meta: json!({"task_kind": "coding"}),
        };

        let templates = library.templates_for_sample(&sample);

        assert_eq!(templates.profile_name, "coding");
        assert_eq!(templates.task_kind, TaskKind::Coding);
        assert_eq!(templates.answer_check, AnswerCheckMode::Disabled);
        assert_eq!(templates.answer_style, AnswerStyle::Code);
    }

    #[test]
    fn build_generation_prompt_renders_sample_accepted_and_feedback() {
        let templates = PromptTemplates {
            profile_name: "unit".to_owned(),
            task_kind: TaskKind::Knowledge,
            answer_check: AnswerCheckMode::SingleLabel,
            answer_style: AnswerStyle::FinalOnly,
            synthesis: None,
            generation: "profile={{profile_name}}\ncount={{variant_count}}\nsample={{source_sample_json}}\naccepted={{accepted_samples_json}}\n{{feedback_block}}".to_owned(),
            validation: None,
        };
        let accepted = vec![pending_task("sample_q000", "Existing question", "A")];

        let rendered = build_generation_prompt(
            &templates,
            &sample("sample", "Original question"),
            2,
            &accepted,
            Some("bad JSON"),
        )
        .expect("prompt should render");

        assert!(rendered.contains("profile=unit"));
        assert!(rendered.contains("count=2"));
        assert!(rendered.contains("Original question"));
        assert!(rendered.contains("Existing question"));
        assert!(rendered.contains("bad JSON"));
        assert!(!rendered.contains("{{"));
    }

    #[test]
    fn build_generation_prompt_preserves_source_cot_style() {
        let templates = PromptTemplates {
            profile_name: "unit".to_owned(),
            task_kind: TaskKind::Math,
            answer_check: AnswerCheckMode::ExactText,
            answer_style: AnswerStyle::FinalOnly,
            synthesis: None,
            generation: "style={{answer_style}}\nsynthesis={{synthesis_profile_json}}".to_owned(),
            validation: None,
        };
        let source = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Original math question with visible reasoning format?".to_owned(),
            source_meta: json!({"cot_mode": "CoT"}),
        };
        let accepted: Vec<PendingTask> = Vec::new();

        let rendered = build_generation_prompt(&templates, &source, 1, &accepted, None)
            .expect("prompt should render");

        assert!(rendered.contains("style=cot"));
        assert!(rendered.contains(r#""answer_style": "cot""#));
        assert!(rendered.contains(r#""profile_default_answer_style": "final_only""#));
    }

    #[test]
    fn source_answer_style_detects_cot_from_nested_sampling_config() {
        let source = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Original?".to_owned(),
            source_meta: json!({"sampling_config": {"cot_mode": "FakeCoT"}}),
        };

        assert_eq!(
            source_answer_style(&source, AnswerStyle::FinalOnly),
            AnswerStyle::Cot
        );

        let explicit_default_but_cot_mode = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Original?".to_owned(),
            source_meta: json!({"answer_style": "final_only", "cot_mode": "CoT"}),
        };
        assert_eq!(
            source_answer_style(&explicit_default_but_cot_mode, AnswerStyle::FinalOnly),
            AnswerStyle::Cot
        );

        let no_cot = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Original?".to_owned(),
            source_meta: json!({"cot_mode": "NoCoT"}),
        };
        assert_eq!(
            source_answer_style(&no_cot, AnswerStyle::FinalOnly),
            AnswerStyle::FinalOnly
        );
    }

    #[test]
    fn build_generation_validation_prompt_requires_validation_template() {
        let templates = PromptTemplates {
            profile_name: "unit".to_owned(),
            task_kind: TaskKind::Knowledge,
            answer_check: AnswerCheckMode::ExactText,
            answer_style: AnswerStyle::FinalOnly,
            synthesis: None,
            generation: String::new(),
            validation: None,
        };

        let err = build_generation_validation_prompt(&templates, &sample("sample", "Q"), &[])
            .expect_err("missing validation template should fail");

        assert!(err.to_string().contains("missing validation template"));
    }

    #[test]
    fn build_generation_validation_prompt_renders_candidates() {
        let templates = PromptTemplates {
            profile_name: "unit".to_owned(),
            task_kind: TaskKind::Knowledge,
            answer_check: AnswerCheckMode::SingleLabel,
            answer_style: AnswerStyle::FinalOnly,
            synthesis: None,
            generation: String::new(),
            validation: Some(
                "profile={{profile_name}}\nsample={{source_sample_json}}\ncandidates={{generated_candidates_json}}"
                    .to_owned(),
            ),
        };
        let candidates = vec![GeneratedItemDraft {
            user: "Candidate?".to_owned(),
            answer: "B".to_owned(),
            meta: json!({"difficulty": "easy"}),
        }];

        let rendered = build_generation_validation_prompt(
            &templates,
            &sample("sample", "Original?"),
            &candidates,
        )
        .expect("validation prompt should render");

        assert!(rendered.contains("profile=unit"));
        assert!(rendered.contains("Candidate?"));
        assert!(rendered.contains("\"index\": 0"));
        assert!(!rendered.contains("{{"));
    }
}
