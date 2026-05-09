use std::collections::VecDeque;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::Value;

use crate::config::{GeneratorConfig, ModelConfig};
use crate::openai::OpenAiClient;
use crate::output::GeneratedTaskWriter;
use crate::prompt::{PromptTemplates, build_generation_prompt, build_generation_validation_prompt};
use crate::text::{normalize_compare_text, preview_text, sanitize_training_user_prompt};
use crate::types::{GenerateJob, GeneratedItemDraft, PendingTask, SourceSample};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedItemsEnvelope {
    items: Vec<GeneratedItemDraft>,
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

#[derive(Debug)]
struct GeneratedItem {
    user: String,
    answer: String,
    item_json: String,
}

#[derive(Debug)]
struct CandidateBatch {
    items: Vec<GeneratedItem>,
    rejected_reasons: Vec<String>,
}

#[derive(Debug)]
struct ValidationDecision {
    valid: bool,
    reason: String,
}

#[derive(Clone)]
pub(crate) struct ValidatorRuntime {
    pub(crate) client: OpenAiClient,
    pub(crate) model: ModelConfig,
    pub(crate) json_object_response: bool,
}

pub(crate) struct GenerateTasksResult {
    pub(crate) tasks: Vec<PendingTask>,
    pub(crate) completed: bool,
    pub(crate) rejected_candidates: usize,
    pub(crate) terminal_error: Option<String>,
}

pub(crate) async fn generate_tasks(
    client: OpenAiClient,
    generator: GeneratorConfig,
    validator: Option<ValidatorRuntime>,
    prompt_templates: PromptTemplates,
    generated_task_writer: GeneratedTaskWriter,
    job: GenerateJob,
) -> Result<GenerateTasksResult> {
    let target_count = job.missing_indices.len();
    let mut missing_indices = VecDeque::from(job.missing_indices.clone());
    let mut feedback = None::<String>;
    let mut accepted_tasks = job.accepted_tasks.clone();
    let mut accepted_norms = accepted_tasks
        .iter()
        .map(|task| normalize_compare_text(&task.user))
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut rejected_candidates = 0usize;

    for attempt in 0..generator.generation_attempts {
        if missing_indices.is_empty() {
            break;
        }
        let prompt = build_generation_prompt(
            &prompt_templates,
            &job.sample,
            missing_indices.len(),
            &accepted_tasks,
            feedback.as_deref(),
        )?;
        let result = match client
            .chat(&generator.model, &prompt, generator.json_object_response)
            .await
            .with_context(|| format!("generator failed for sample {}", job.sample.sample_id))
        {
            Ok(result) => result,
            Err(err) => {
                return finish_generation_or_error(out, target_count, rejected_candidates, err);
            }
        };

        let mut batch = match parse_generated_items(&result.content, &job.sample) {
            Ok(batch) => batch,
            Err(err) if attempt + 1 < generator.generation_attempts => {
                feedback = Some(err.to_string());
                continue;
            }
            Err(err) => {
                return finish_generation_or_error(out, target_count, rejected_candidates, err);
            }
        };
        let duplicate_filtered = filter_duplicate_candidates(batch.items, &accepted_norms);
        batch.items = duplicate_filtered.items;
        batch
            .rejected_reasons
            .extend(duplicate_filtered.rejected_reasons);

        let decisions = if generator.validate_generated_questions && !batch.items.is_empty() {
            let validator = validator.as_ref();
            let validator_client = validator
                .map(|validator| &validator.client)
                .unwrap_or(&client);
            let validator_model = validator
                .map(|validator| &validator.model)
                .unwrap_or(&generator.model);
            let validator_json_object_response = validator
                .map(|validator| validator.json_object_response)
                .unwrap_or(generator.validator_json_object_response);
            let drafts = batch
                .items
                .iter()
                .map(|item| serde_json::from_str::<GeneratedItemDraft>(&item.item_json))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            match validate_generated_items_with_model(
                validator_client,
                validator_model,
                validator_json_object_response,
                &prompt_templates,
                &job.sample,
                &drafts,
            )
            .await
            {
                Ok(decisions) => decisions,
                Err(err) => {
                    return finish_generation_or_error(out, target_count, rejected_candidates, err);
                }
            }
        } else {
            batch
                .items
                .iter()
                .map(|_| ValidationDecision {
                    valid: true,
                    reason: String::new(),
                })
                .collect::<Vec<_>>()
        };

        let mut accepted_now = Vec::new();
        for (candidate_index, (item, decision)) in
            batch.items.into_iter().zip(decisions).enumerate()
        {
            if !decision.valid {
                batch.rejected_reasons.push(format!(
                    "#{} {}",
                    candidate_index,
                    decision.reason.trim()
                ));
                continue;
            }
            let Some(index) = missing_indices.pop_front() else {
                break;
            };
            let normalized = normalize_compare_text(&item.user);
            let task = PendingTask {
                task_id: task_id(&job.sample.sample_id, index),
                user: item.user,
                expected_answer: item.answer,
                generated_item_json: item.item_json,
            };
            accepted_norms.push(normalized);
            accepted_tasks.push(task.clone());
            accepted_now.push(task);
        }

        rejected_candidates += batch.rejected_reasons.len();
        generated_task_writer.append_generated_tasks(&accepted_now)?;
        out.extend(accepted_now);
        feedback = if batch.rejected_reasons.is_empty() {
            None
        } else {
            let joined = batch
                .rejected_reasons
                .iter()
                .take(8)
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(" | ");
            Some(format!("部分候选未通过：{joined}"))
        };
    }

    Ok(GenerateTasksResult {
        completed: out.len() == target_count,
        tasks: out,
        rejected_candidates,
        terminal_error: None,
    })
}

fn finish_generation_or_error(
    tasks: Vec<PendingTask>,
    target_count: usize,
    rejected_candidates: usize,
    err: anyhow::Error,
) -> Result<GenerateTasksResult> {
    if tasks.is_empty() {
        Err(err)
    } else {
        Ok(GenerateTasksResult {
            completed: tasks.len() == target_count,
            tasks,
            rejected_candidates,
            terminal_error: Some(err.to_string()),
        })
    }
}

pub(crate) fn task_id(sample_id: &str, variant_index: usize) -> String {
    format!("{sample_id}_q{variant_index:03}")
}

fn parse_generated_items(text: &str, sample: &SourceSample) -> Result<CandidateBatch> {
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

    ensure!(!envelope.items.is_empty(), "generator returned no items");

    let mut items = Vec::new();
    let mut rejected_reasons = Vec::new();
    for (index, draft) in envelope.items.into_iter().enumerate() {
        match validate_generated_item(draft, sample) {
            Ok(item) => items.push(item),
            Err(err) => rejected_reasons.push(format!("#{index} {err}")),
        }
    }

    Ok(CandidateBatch {
        items,
        rejected_reasons,
    })
}

fn filter_duplicate_candidates(
    items: Vec<GeneratedItem>,
    accepted_norms: &[String],
) -> CandidateBatch {
    let mut seen = accepted_norms.to_vec();
    let mut out = Vec::new();
    let mut rejected_reasons = Vec::new();

    for (index, item) in items.into_iter().enumerate() {
        let normalized = normalize_compare_text(&item.user);
        if seen.iter().any(|existing| existing == &normalized) {
            rejected_reasons.push(format!(
                "#{} duplicate variant {:?}",
                index,
                preview_text(&item.user, 120)
            ));
        } else {
            seen.push(normalized);
            out.push(item);
        }
    }

    CandidateBatch {
        items: out,
        rejected_reasons,
    }
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
    if let Some(ref_answer) = sample
        .source_meta
        .get("ref_answer")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|answer| !answer.is_empty())
    {
        ensure!(
            !reuses_reference_answer_content(&sample.source_user, &user, &answer, ref_answer),
            "generated answer reuses original ref_answer content"
        );
    }

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

fn answers_equal(left: &str, right: &str) -> bool {
    let left = left.trim();
    let right = right.trim();
    !left.is_empty() && left.eq_ignore_ascii_case(right)
}

fn reuses_reference_answer_content(
    original_user: &str,
    generated_user: &str,
    generated_answer: &str,
    reference_answer: &str,
) -> bool {
    if !answers_equal(generated_answer, reference_answer) {
        return false;
    }

    match (
        option_text_for_answer(original_user, reference_answer),
        option_text_for_answer(generated_user, generated_answer),
    ) {
        (Some(original), Some(generated)) => {
            normalize_compare_text(&original) == normalize_compare_text(&generated)
        }
        _ => true,
    }
}

fn option_text_for_answer(user: &str, answer: &str) -> Option<String> {
    let answer = single_letter_answer(answer)?;
    let markers = option_markers(user);
    let marker_index = markers.iter().position(|marker| marker.letter == answer)?;
    let start = markers[marker_index].content_start;
    let end = markers
        .get(marker_index + 1)
        .map(|marker| marker.marker_start)
        .unwrap_or(user.len());
    let text = user.get(start..end)?.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn single_letter_answer(answer: &str) -> Option<char> {
    let trimmed = answer.trim().trim_end_matches(['.', ')', '）']).trim();
    let mut chars = trimmed.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    if chars.next().is_some() || !matches!(letter, 'A'..='D') {
        return None;
    }
    Some(letter)
}

#[derive(Clone, Copy)]
struct OptionMarker {
    letter: char,
    marker_start: usize,
    content_start: usize,
}

fn option_markers(user: &str) -> Vec<OptionMarker> {
    let mut markers = Vec::new();
    for (index, ch) in user.char_indices() {
        let letter = ch.to_ascii_uppercase();
        if !matches!(letter, 'A'..='D') {
            continue;
        }
        let after_letter = index + ch.len_utf8();
        if !user[after_letter..].starts_with('.') {
            continue;
        }
        let content_start = after_letter + '.'.len_utf8();
        let Some(after_dot) = user[content_start..].chars().next() else {
            continue;
        };
        if !after_dot.is_whitespace() {
            continue;
        }
        let before_ok = index == 0
            || user[..index]
                .chars()
                .next_back()
                .is_some_and(|before| before.is_whitespace() || before == ':' || before == '：');
        if before_ok {
            markers.push(OptionMarker {
                letter,
                marker_start: index,
                content_start,
            });
        }
    }
    markers
}

async fn validate_generated_items_with_model(
    client: &OpenAiClient,
    model: &ModelConfig,
    json_output: bool,
    prompt_templates: &PromptTemplates,
    sample: &SourceSample,
    generated: &[GeneratedItemDraft],
) -> Result<Vec<ValidationDecision>> {
    let prompt = build_generation_validation_prompt(prompt_templates, sample, generated)?;
    let result = client
        .chat(model, &prompt, json_output)
        .await
        .with_context(|| format!("validator failed for sample {}", sample.sample_id))?;
    let envelope = parse_validation_envelope(&result.content)?;
    ensure!(
        envelope.items.len() == generated.len(),
        "validator returned {} items, expected {}",
        envelope.items.len(),
        generated.len()
    );

    let mut decisions = (0..generated.len())
        .map(|_| None)
        .collect::<Vec<Option<ValidationDecision>>>();
    for item in envelope.items {
        ensure!(
            item.index < generated.len(),
            "validator returned out-of-range item index {}",
            item.index
        );
        ensure!(
            decisions[item.index].is_none(),
            "validator returned duplicate item index {}",
            item.index
        );
        decisions[item.index] = Some(ValidationDecision {
            valid: item.valid,
            reason: item.reason,
        });
    }

    decisions
        .into_iter()
        .enumerate()
        .map(|(index, decision)| {
            decision.ok_or_else(|| anyhow!("validator omitted item index {index}"))
        })
        .collect()
}

fn parse_validation_envelope(text: &str) -> Result<ValidationEnvelope> {
    serde_json::from_str(text)
        .or_else(|_| {
            extract_json_object_from_text(text)
                .ok_or_else(|| serde_json::Error::io(std::io::Error::other("no json object")))
                .and_then(|json| serde_json::from_str(&json))
        })
        .context("validator did not return valid JSON or extractable JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::sample;
    use serde_json::json;

    #[test]
    fn task_id_uses_stable_zero_padded_variant_suffix() {
        assert_eq!(task_id("sample", 7), "sample_q007");
    }

    #[test]
    fn parse_generated_items_accepts_fenced_json_and_serializes_clean_items() {
        let text = format!(
            "before```json\n{}\n```after",
            json!({
                "items": [
                    {"user": "New question A?", "answer": "A", "meta": {"kind": "a"}},
                    {"user": "New question B?", "answer": "B"}
                ]
            })
        );

        let batch = parse_generated_items(&text, &sample("sample", "Original question?"))
            .expect("items should parse");

        assert_eq!(batch.items.len(), 2);
        assert!(batch.rejected_reasons.is_empty());
        assert_eq!(batch.items[0].user, "New question A?");
        assert_eq!(batch.items[0].answer, "A");
        assert!(batch.items[0].item_json.contains("\"kind\":\"a\""));
        assert_eq!(batch.items[1].answer, "B");
    }

    #[test]
    fn parse_validation_envelope_accepts_fenced_json() {
        let text = r#"```json
{"items":[{"index":0,"valid":true,"reason":"ok"}]}
```"#;

        let envelope = parse_validation_envelope(text).expect("fenced validator JSON should parse");

        assert_eq!(envelope.items.len(), 1);
        assert_eq!(envelope.items[0].index, 0);
        assert!(envelope.items[0].valid);
        assert_eq!(envelope.items[0].reason, "ok");
    }

    #[test]
    fn parse_generated_items_accepts_available_candidates_without_count_lockstep() {
        let short_batch = json!({
            "items": [{"user": "New question?", "answer": "A"}]
        })
        .to_string();
        let batch = parse_generated_items(&short_batch, &sample("sample", "Original question?"))
            .expect("short batch should still parse");
        assert_eq!(batch.items.len(), 1);

        let mixed = json!({
            "items": [
                {"user": "Original question?", "answer": "A"},
                {"user": "New question?", "answer": "B"}
            ]
        })
        .to_string();
        let batch = parse_generated_items(&mixed, &sample("sample", "Original question?"))
            .expect("valid candidates should survive local rejection");
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.items[0].user, "New question?");
        assert_eq!(batch.rejected_reasons.len(), 1);
    }

    #[test]
    fn filter_duplicate_candidates_rejects_duplicates_without_dropping_unique_items() {
        let duplicate = json!({
            "items": [
                {"user": "Same question?", "answer": "A"},
                {"user": "Same question!", "answer": "B"}
            ]
        })
        .to_string();
        let batch = parse_generated_items(&duplicate, &sample("sample", "Original question?"))
            .expect("duplicates are filtered after parsing");

        let filtered = filter_duplicate_candidates(batch.items, &[]);

        assert_eq!(filtered.items.len(), 1);
        assert_eq!(filtered.items[0].user, "Same question?");
        assert_eq!(filtered.rejected_reasons.len(), 1);
        assert!(filtered.rejected_reasons[0].contains("duplicate"));
    }

    #[test]
    fn validate_generated_item_rejects_assistant_and_identical_originals() {
        let original = sample("sample", "Original question?");

        let assistant = GeneratedItemDraft {
            user: "Assistant: answer".to_owned(),
            answer: "A".to_owned(),
            meta: serde_json::Value::Null,
        };
        assert!(
            validate_generated_item(assistant, &original)
                .expect_err("assistant prompt should fail")
                .to_string()
                .contains("user prompt")
        );

        let identical = GeneratedItemDraft {
            user: "Original question?".to_owned(),
            answer: "A".to_owned(),
            meta: serde_json::Value::Null,
        };
        assert!(
            validate_generated_item(identical, &original)
                .expect_err("identical prompt should fail")
                .to_string()
                .contains("identical")
        );
    }

    #[test]
    fn validate_generated_item_rejects_answer_equal_to_source_ref_answer() {
        let original = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Original question?".to_owned(),
            source_meta: json!({"ref_answer": "B", "answer": "A"}),
        };

        let err = validate_generated_item(
            GeneratedItemDraft {
                user: "New question?".to_owned(),
                answer: " b ".to_owned(),
                meta: Value::Null,
            },
            &original,
        )
        .expect_err("same ref_answer should fail");

        assert!(err.to_string().contains("ref_answer"));
    }

    #[test]
    fn validate_generated_item_allows_same_letter_when_option_content_changes() {
        let original = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Question: original\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
                .to_owned(),
            source_meta: json!({"ref_answer": "B"}),
        };

        let item = validate_generated_item(
            GeneratedItemDraft {
                user: "Question: changed\nChoices:\nA. Alpha\nB. PCR\nC. Gamma\nD. Delta"
                    .to_owned(),
                answer: "B".to_owned(),
                meta: Value::Null,
            },
            &original,
        )
        .expect("same letter with changed option content should pass");

        assert_eq!(item.answer, "B");
    }

    #[test]
    fn validate_generated_item_rejects_same_letter_and_same_option_content() {
        let original = SourceSample {
            sample_id: "sample".to_owned(),
            source_user: "Question: original Choices: A. Alpha B. Beta C. Gamma D. Delta"
                .to_owned(),
            source_meta: json!({"ref_answer": "B"}),
        };

        let err = validate_generated_item(
            GeneratedItemDraft {
                user: "Question: changed Choices: A. Other B. Beta C. Gamma D. Delta".to_owned(),
                answer: "B".to_owned(),
                meta: Value::Null,
            },
            &original,
        )
        .expect_err("same letter with same option content should fail");

        assert!(err.to_string().contains("ref_answer"));
    }

    #[test]
    fn extract_first_balanced_json_object_handles_braces_inside_strings() {
        let object = extract_first_balanced_json_object(
            r#"prefix {"items":[{"user":"contains } brace","answer":"A"}]} suffix"#,
        )
        .expect("balanced object should be extracted");

        assert_eq!(
            object,
            r#"{"items":[{"user":"contains } brace","answer":"A"}]}"#
        );
    }
}
