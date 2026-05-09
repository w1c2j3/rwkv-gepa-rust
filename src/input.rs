use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use serde::de::{self, DeserializeSeed, SeqAccess, Visitor};
use serde_json::Value;
use serde_jsonlines::json_lines;

use crate::config::{Config, InputConfig};
use crate::text::{preview_text, sanitize_training_user_prompt};
use crate::types::SourceSample;

pub(crate) fn load_samples(cfg: &Config) -> Result<Vec<SourceSample>> {
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

fn extract_source_user_from_structured_context(context: &str) -> Result<String> {
    let payload: Value = match serde_json::from_str(context.trim()) {
        Ok(value) => value,
        Err(raw_err) => {
            let normalized = normalize_context_text(context);
            serde_json::from_str(normalized.trim()).with_context(|| {
                format!("context is not parseable structured JSON; raw parse error: {raw_err}")
            })?
        }
    };

    if let Some(prompt) = payload.get("prompt").and_then(Value::as_str) {
        return extract_source_user_from_prompt(prompt);
    }

    if let Some(stages) = payload.get("stages").and_then(Value::as_array) {
        for stage in stages {
            if let Some(prompt) = stage.get("prompt").and_then(Value::as_str)
                && let Ok(user) = extract_source_user_from_prompt(prompt)
            {
                return Ok(user);
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

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_config;
    use crate::test_support::{temp_path, write_text};
    use serde_json::json;

    #[test]
    fn load_input_rows_window_jsonl_skips_invalid_lines_and_applies_window() {
        let path = temp_path("input/window.jsonl");
        write_text(
            &path,
            r#"{"context":"first"}
not-json
{"context":"second"}
{"context":"third"}
"#,
        );

        let rows = load_input_rows_window(&path, 1, Some(1)).expect("rows should load");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 1);
        assert_eq!(rows[0].1["context"], "second");
    }

    #[test]
    fn load_input_rows_window_json_array_detects_format_and_applies_window() {
        let path = temp_path("input/window_array.json");
        write_text(
            &path,
            r#"[
  {"context":"first"},
  {"context":"second"},
  {"context":"third"}
]"#,
        );

        let rows = load_input_rows_window(&path, 2, Some(5)).expect("rows should load");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 2);
        assert_eq!(rows[0].1["context"], "third");
    }

    #[test]
    fn normalize_sample_extracts_user_prompt_and_metadata() {
        let config_path = temp_path("input/config.toml");
        write_text(
            &config_path,
            r#"
[input]
dataset_path = "unused.jsonl"

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
        let cfg = load_config(&config_path).expect("config should load");
        let value = json!({
            "task_id": "task",
            "sample_index": 7,
            "completions_id": "abc",
            "context": "User: You are a very talented expert in math. Question?\nAssistant: old",
            "answer": "A",
            "subject": "math"
        });

        let sample = normalize_sample(&cfg.input, 3, value).expect("sample should normalize");

        assert_eq!(sample.sample_id, "task_7_abc");
        assert_eq!(sample.source_user, "Question?");
        assert_eq!(sample.source_meta["answer"], "A");
        assert_eq!(sample.source_meta["subject"], "math");
    }

    #[test]
    fn extract_source_user_reads_structured_stage_prompts() {
        let context = json!({
            "stages": [
                {"prompt": "not usable"},
                {"prompt": "User: Structured question?\nAssistant: answer"}
            ]
        })
        .to_string();

        let user = extract_source_user(&context).expect("structured prompt should parse");

        assert_eq!(user, "Structured question?");
    }

    #[test]
    fn scalar_text_accepts_scalar_json_values_only() {
        assert_eq!(scalar_text(&json!(true)).as_deref(), Some("true"));
        assert_eq!(scalar_text(&json!(42)).as_deref(), Some("42"));
        assert_eq!(scalar_text(&json!(" text ")).as_deref(), Some("text"));
        assert_eq!(scalar_text(&json!([])), None);
    }
}
