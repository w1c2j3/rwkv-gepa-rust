use std::hash::{Hash, Hasher};

use anyhow::{Context, Error, Result, ensure};
use futures::{StreamExt, stream};
use rustc_hash::FxHasher;

use crate::config::ModelConfig;
use crate::openai::OpenAiClient;
use crate::output::{OutputPaths, RunStatus, append_jsonl, done_output_path};
use crate::task::{AnswerCheckMode, TaskKind};
use crate::text::normalize_compare_text;
use crate::types::{OutputRow, PendingTask};
use crate::util::concurrency_limit;

#[derive(Default)]
pub(crate) struct AnswerBatchStats {
    pub(crate) success: usize,
    pub(crate) failed: usize,
    pub(crate) skipped: usize,
}

struct AnsweredTask {
    row: OutputRow,
    correct: bool,
}

enum AnswerTaskOutcome {
    Answered(AnsweredTask),
    Skipped { row: OutputRow, error: Error },
}

pub(crate) async fn answer_tasks_and_persist(
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
            async move {
                let skipped_row = skipped_answer_row(&model, &task, "");
                let task_kind = task.task_kind;
                let answer_check = task.answer_check;
                match answer_task(client, model, task, task_kind, answer_check).await {
                    Ok(answered) => AnswerTaskOutcome::Answered(answered),
                    Err(error) => AnswerTaskOutcome::Skipped {
                        row: skipped_row,
                        error,
                    },
                }
            }
        })
        .buffer_unordered(concurrency);

    while let Some(outcome) = stream.next().await {
        match outcome {
            AnswerTaskOutcome::Answered(answered) => {
                let path = done_output_path(output_paths, answered.correct);
                append_jsonl(path, std::slice::from_ref(&answered.row))?;
                if answered.correct {
                    stats.success += 1;
                } else {
                    stats.failed += 1;
                }
            }
            AnswerTaskOutcome::Skipped { mut row, error } => {
                row.assistant = format!("answer_error: {error:#}");
                append_jsonl(
                    done_output_path(output_paths, false),
                    std::slice::from_ref(&row),
                )?;
                stats.skipped += 1;
                eprintln!("skipped answer task: {error:#}");
            }
        }
    }

    Ok(stats)
}

fn skipped_answer_row(model: &ModelConfig, task: &PendingTask, assistant: &str) -> OutputRow {
    OutputRow {
        task_id: task.task_id.clone(),
        status: RunStatus::Done.as_str().to_owned(),
        user: task.user.clone(),
        generated_item_json: task.generated_item_json.clone(),
        answer_model: model.model_name.clone(),
        assistant: assistant.to_owned(),
        text: String::new(),
    }
}

async fn answer_task(
    client: OpenAiClient,
    model: ModelConfig,
    task: PendingTask,
    task_kind: TaskKind,
    answer_check: AnswerCheckMode,
) -> Result<AnsweredTask> {
    let answer_prompt = build_answer_prompt(task_kind, &task.user);
    let result = client
        .chat(&model, &answer_prompt, false)
        .await
        .with_context(|| format!("answer failed for task {}", task.task_id))?;

    let visible_content = result.content.clone();
    let assistant = merge_answer_output(result.reasoning, result.content);

    ensure!(
        !assistant.trim().is_empty(),
        "answer model returned empty output for task {}",
        task.task_id
    );

    let predicted_answer = extract_predicted_answer(
        answer_check,
        &task.expected_answer,
        &visible_content,
        &assistant,
    );
    let answer_correct =
        compare_expected_and_predicted(answer_check, &task.expected_answer, &predicted_answer);
    let text = if answer_correct {
        format!(
            "User: {}\nAssistant: {}",
            task.user.trim(),
            assistant.trim()
        )
    } else {
        String::new()
    };

    Ok(AnsweredTask {
        row: OutputRow {
            task_id: task.task_id,
            status: RunStatus::Done.as_str().to_owned(),
            user: task.user.clone(),
            generated_item_json: task.generated_item_json.clone(),
            answer_model: model.model_name.clone(),
            assistant: assistant.clone(),
            text,
        },
        correct: answer_correct,
    })
}

fn merge_answer_output(reasoning: Option<String>, content: String) -> String {
    match (reasoning.map(|text| text.trim().to_owned()), content.trim()) {
        (Some(reasoning), "") => format!("<think>\n{reasoning}\n</think>"),
        (Some(reasoning), content) => format!("<think>\n{reasoning}\n</think>\n\n{content}"),
        (None, content) => content.to_owned(),
    }
}

fn build_answer_prompt(task_kind: TaskKind, user: &str) -> String {
    let rules = match task_kind {
        TaskKind::Knowledge => {
            r#"Additional output rules:
1. If this is a single-choice question, finish with exactly "Therefore, the answer is X", replacing X with one uppercase option letter.
2. If this is a multi-choice question, finish with exactly "Therefore, the answers are X", replacing X with comma-separated uppercase labels in sorted order, for example A,C,F.
3. If this is not a choice question, finish with exactly one final line: Final answer: <answer>.
4. If no separate reasoning channel is available, use this training format for non-trivial reasoning:
<think>
focused reasoning, concise and directly relevant
</think><final line>
5. Do not repeat or paraphrase the question.
6. Do not output extra text after the final answer line."#
        }
        TaskKind::Math => {
            r#"Additional output rules:
1. Solve the problem carefully and keep the reasoning concise.
2. Finish with exactly one final line: Final answer: <answer>.
3. Do not add text after the final answer line."#
        }
        TaskKind::Coding => {
            r#"Additional output rules:
1. Return the requested code, patch, SQL, or implementation result in the format requested by the user.
2. If this is a GitHub issue task, answer with a concrete cause, minimal fix, and patch-level detail when useful.
3. If explanation is requested, keep it concise and grounded in the code.
4. Do not invent dependencies, files, APIs, schemas, or hidden context."#
        }
        TaskKind::InstructionFollowing => {
            r#"Additional output rules:
1. Follow every user instruction, formatting constraint, ordering rule, and negative constraint exactly.
2. For translation tasks, preserve meaning, tone, locale, names, numbers, and formatting constraints.
3. Do not add commentary unless the user asks for it."#
        }
        TaskKind::FunctionCalling => {
            r#"Additional output rules:
1. Output only valid JSON matching the requested function-calling/tool-call shape.
2. Do not use markdown fences.
3. Do not add explanations, comments, or trailing text."#
        }
    };
    format!("{user}\n\n{rules}")
}

fn extract_predicted_answer(
    answer_check: AnswerCheckMode,
    expected_answer: &str,
    content: &str,
    assistant: &str,
) -> String {
    let primary = if !content.trim().is_empty() {
        content.trim()
    } else {
        assistant.trim()
    };

    match answer_check {
        AnswerCheckMode::SingleLabel if canonical_answer_label(expected_answer).is_some() => {
            return canonical_answer_label(primary).unwrap_or_else(|| {
                last_non_empty_line(primary)
                    .unwrap_or(primary)
                    .trim()
                    .to_owned()
            });
        }
        AnswerCheckMode::LabelSet => {
            return canonical_answer_label_set(primary)
                .map(|labels| labels.join(","))
                .unwrap_or_else(|| {
                    last_non_empty_line(primary)
                        .unwrap_or(primary)
                        .trim()
                        .to_owned()
                });
        }
        AnswerCheckMode::JsonExact => {
            return extract_json_value_text(primary).unwrap_or_else(|| {
                last_non_empty_line(primary)
                    .unwrap_or(primary)
                    .trim()
                    .to_owned()
            });
        }
        AnswerCheckMode::SingleLabel | AnswerCheckMode::ExactText | AnswerCheckMode::Disabled => {}
    }

    if answer_check == AnswerCheckMode::SingleLabel
        && canonical_answer_label(expected_answer).is_some()
    {
        return canonical_answer_label(primary).unwrap_or_else(|| {
            last_non_empty_line(primary)
                .unwrap_or(primary)
                .trim()
                .to_owned()
        });
    }

    last_non_empty_line(primary)
        .unwrap_or(primary)
        .trim()
        .to_owned()
}

fn compare_expected_and_predicted(
    answer_check: AnswerCheckMode,
    expected: &str,
    predicted: &str,
) -> bool {
    match answer_check {
        AnswerCheckMode::Disabled => true,
        AnswerCheckMode::SingleLabel => canonical_answer_label(expected)
            .and_then(|expected_label| {
                canonical_answer_label(predicted)
                    .map(|predicted_label| predicted_label == expected_label)
            })
            .unwrap_or(false),
        AnswerCheckMode::LabelSet => {
            canonical_answer_label_set(expected) == canonical_answer_label_set(predicted)
        }
        AnswerCheckMode::ExactText => {
            normalize_compare_text(expected) == normalize_compare_text(predicted)
        }
        AnswerCheckMode::JsonExact => match (normalized_json(expected), normalized_json(predicted))
        {
            (Some(expected), Some(predicted)) => expected == predicted,
            _ => false,
        },
    }
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

fn canonical_answer_label_set(text: &str) -> Option<Vec<String>> {
    let mut labels = text
        .split(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';'
                        | '/'
                        | '|'
                        | '，'
                        | '；'
                        | '、'
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
                matches!(ch, 'A'..='J').then(|| ch.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    (!labels.is_empty()).then_some(labels)
}

fn normalized_json(text: &str) -> Option<String> {
    let json_text = extract_json_value_text(text)?;
    let value = serde_json::from_str::<serde_json::Value>(&json_text).ok()?;
    serde_json::to_string(&value).ok()
}

fn extract_json_value_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed.to_owned());
    }

    let mut start = None::<usize>;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in trimmed.char_indices() {
        if start.is_none() && matches!(ch, '{' | '[') {
            start = Some(index);
            depth = 1;
            continue;
        }
        if start.is_none() {
            continue;
        }
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
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    let start = start?;
                    return Some(trimmed[start..index + ch.len_utf8()].to_owned());
                }
            }
            _ => {}
        }
    }
    None
}

fn last_non_empty_line(text: &str) -> Option<&str> {
    text.lines().rev().find(|line| !line.trim().is_empty())
}

fn pick_model_index(task_id: &str, count: usize) -> usize {
    let mut hasher = FxHasher::default();
    task_id.hash(&mut hasher);
    (hasher.finish() as usize) % count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_answer_output_preserves_reasoning_and_content() {
        assert_eq!(
            merge_answer_output(Some(" short reason ".to_owned()), " A ".to_owned()),
            "<think>\nshort reason\n</think>\n\nA"
        );
        assert_eq!(
            merge_answer_output(Some(" short reason ".to_owned()), "  ".to_owned()),
            "<think>\nshort reason\n</think>"
        );
        assert_eq!(merge_answer_output(None, " B ".to_owned()), "B");
    }

    #[test]
    fn answer_prompt_includes_original_user_and_final_letter_rule() {
        let prompt = build_answer_prompt(TaskKind::Knowledge, "Question?");

        assert!(prompt.starts_with("Question?"));
        assert!(prompt.contains("<think>"));
        assert!(prompt.contains("Therefore, the answer is X"));
    }

    #[test]
    fn answer_prompt_uses_task_specific_rules() {
        let code_prompt = build_answer_prompt(TaskKind::Coding, "Fix this function");
        assert!(code_prompt.contains("requested code"));
        assert!(!code_prompt.contains("Therefore, the answer is X"));

        let json_prompt = build_answer_prompt(TaskKind::FunctionCalling, "Call the tool");
        assert!(json_prompt.contains("only valid JSON"));
    }

    #[test]
    fn expected_label_comparison_uses_final_option_letter() {
        assert!(compare_expected_and_predicted(
            AnswerCheckMode::SingleLabel,
            "A",
            "Reasoning...\nA"
        ));
        assert!(compare_expected_and_predicted(
            AnswerCheckMode::SingleLabel,
            "answer: c",
            "therefore C"
        ));
        assert!(!compare_expected_and_predicted(
            AnswerCheckMode::SingleLabel,
            "A",
            "B"
        ));
    }

    #[test]
    fn label_set_comparison_uses_sorted_unique_labels() {
        assert!(compare_expected_and_predicted(
            AnswerCheckMode::LabelSet,
            "A, C, F",
            "Therefore, the answers are F,A,C"
        ));
        assert!(!compare_expected_and_predicted(
            AnswerCheckMode::LabelSet,
            "A,C",
            "A,D"
        ));
    }

    #[test]
    fn free_text_comparison_uses_normalized_text() {
        assert!(compare_expected_and_predicted(
            AnswerCheckMode::ExactText,
            "Hello, world!",
            "hello world"
        ));
        assert!(!compare_expected_and_predicted(
            AnswerCheckMode::ExactText,
            "Hello",
            "Goodbye"
        ));
        assert!(compare_expected_and_predicted(
            AnswerCheckMode::Disabled,
            "expected",
            "anything nonempty was already checked"
        ));
    }

    #[test]
    fn json_comparison_normalizes_objects_and_rejects_invalid_json() {
        assert!(compare_expected_and_predicted(
            AnswerCheckMode::JsonExact,
            r#"{"name":"search","arguments":{"q":"rwkv"}}"#,
            "```json\n{\"name\":\"search\",\"arguments\":{\"q\":\"rwkv\"}}\n```"
        ));
        assert!(!compare_expected_and_predicted(
            AnswerCheckMode::JsonExact,
            "not-json",
            "also-not-json"
        ));
    }

    #[test]
    fn extract_predicted_answer_falls_back_to_last_non_empty_line() {
        assert_eq!(
            extract_predicted_answer(
                AnswerCheckMode::ExactText,
                "free text answer",
                "reason\n\nFinal text",
                ""
            ),
            "Final text"
        );
        assert_eq!(
            extract_predicted_answer(AnswerCheckMode::SingleLabel, "A", "analysis\nB", ""),
            "B"
        );
        assert_eq!(
            extract_predicted_answer(
                AnswerCheckMode::LabelSet,
                "A,C",
                "Therefore, the answers are C,A",
                ""
            ),
            "A,C"
        );
    }

    #[test]
    fn model_picker_is_deterministic_and_in_range() {
        let first = pick_model_index("sample_q000", 3);
        let second = pick_model_index("sample_q000", 3);

        assert_eq!(first, second);
        assert!(first < 3);
    }
}
