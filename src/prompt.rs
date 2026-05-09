use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::json;

use crate::config::PromptConfig;
use crate::types::{GeneratedItemDraft, PendingTask, SourceSample};

#[derive(Clone)]
pub(crate) struct PromptTemplates {
    profile_name: String,
    generation: String,
    validation: Option<String>,
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

pub(crate) fn load_prompt_templates(
    prompt: &PromptConfig,
    validate_generated_questions: bool,
) -> Result<PromptTemplates> {
    let profile_text = match &prompt.profile_path {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt profile TOML {}", path.display()))?,
        None => include_str!("../prompts/multiple_choice.toml").to_owned(),
    };
    let profile: PromptProfileFile =
        toml::from_str(&profile_text).context("failed to parse prompt profile TOML")?;
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

    Ok(PromptTemplates {
        profile_name: profile.name.trim().to_owned(),
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
    Ok(render_prompt_template(
        &prompt_templates.generation,
        &[
            ("profile_name", prompt_templates.profile_name.clone()),
            ("variant_count", count.to_string()),
            ("source_sample_json", source_sample_json),
            ("accepted_samples_json", accepted_json),
            ("feedback_block", feedback_block),
        ],
    ))
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
    Ok(render_prompt_template(
        validation_template,
        &[
            ("profile_name", prompt_templates.profile_name.clone()),
            ("source_sample_json", source_sample_json),
            ("generated_candidates_json", generated_candidates_json),
        ],
    ))
}

fn render_prompt_template(template: &str, vars: &[(&str, String)]) -> String {
    let mut rendered = template.to_owned();
    for (key, value) in vars {
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), value);
    }
    rendered
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

[generation]
template = "generate {{source_sample_json}}"

[validation]
template = "validate {{generated_candidates_json}}"
"#,
        );
        let config = PromptConfig {
            profile_path: Some(path),
        };

        let templates = load_prompt_templates(&config, true).expect("custom prompt should load");

        assert_eq!(templates.profile_name, "custom");
        assert_eq!(templates.generation, "generate {{source_sample_json}}");
        assert_eq!(
            templates.validation.as_deref(),
            Some("validate {{generated_candidates_json}}")
        );
    }

    #[test]
    fn build_generation_prompt_renders_sample_accepted_and_feedback() {
        let templates = PromptTemplates {
            profile_name: "unit".to_owned(),
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
    fn build_generation_validation_prompt_requires_validation_template() {
        let templates = PromptTemplates {
            profile_name: "unit".to_owned(),
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
