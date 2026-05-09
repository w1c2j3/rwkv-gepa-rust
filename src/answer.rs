use std::hash::{Hash, Hasher};

use anyhow::{Context, Result, ensure};
use futures::{StreamExt, stream};
use rustc_hash::FxHasher;

use crate::config::ModelConfig;
use crate::openai::OpenAiClient;
use crate::output::{OutputPaths, RunStatus, append_jsonl, done_output_path};
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

    let predicted_answer = extract_predicted_answer(&task.expected_answer, &assistant, &assistant);
    let answer_correct = compare_expected_and_predicted(&task.expected_answer, &predicted_answer);
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

fn extract_predicted_answer(expected_answer: &str, content: &str, assistant: &str) -> String {
    let primary = if !content.trim().is_empty() {
        content.trim()
    } else {
        assistant.trim()
    };

    if canonical_answer_label(expected_answer).is_some() {
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
        let prompt = build_answer_prompt("Question?");

        assert!(prompt.starts_with("Question?"));
        assert!(prompt.contains("single uppercase option letter"));
        assert!(prompt.contains("Return the answer"));
    }

    #[test]
    fn expected_label_comparison_uses_final_option_letter() {
        assert!(compare_expected_and_predicted("A", "Reasoning...\nA"));
        assert!(compare_expected_and_predicted("answer: c", "therefore C"));
        assert!(!compare_expected_and_predicted("A", "B"));
    }

    #[test]
    fn free_text_comparison_uses_normalized_text() {
        assert!(compare_expected_and_predicted(
            "Hello, world!",
            "hello world"
        ));
        assert!(!compare_expected_and_predicted("Hello", "Goodbye"));
    }

    #[test]
    fn extract_predicted_answer_falls_back_to_last_non_empty_line() {
        assert_eq!(
            extract_predicted_answer("free text answer", "reason\n\nFinal text", ""),
            "Final text"
        );
        assert_eq!(extract_predicted_answer("A", "analysis\nB", ""), "B");
    }

    #[test]
    fn model_picker_is_deterministic_and_in_range() {
        let first = pick_model_index("sample_q000", 3);
        let second = pick_model_index("sample_q000", 3);

        assert_eq!(first, second);
        assert!(first < 3);
    }
}
