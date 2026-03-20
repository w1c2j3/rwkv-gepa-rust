use anyhow::{Result, ensure};
use chrono::Local;
use serde_json::Value;

pub fn log(message: impl AsRef<str>) {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    println!("[{timestamp}] {}", message.as_ref());
}

pub fn index_to_option_label(index: usize) -> String {
    let mut label = String::new();
    let mut current = index;
    loop {
        label.insert(0, (b'A' + (current % 26) as u8) as char);
        if current < 26 {
            break;
        }
        current = current / 26 - 1;
    }
    label
}

pub fn normalize_answer(value: &Value, choices: &[String]) -> String {
    match value {
        Value::Number(number) => number
            .as_u64()
            .and_then(|raw| usize::try_from(raw).ok())
            .filter(|index| *index < choices.len())
            .map(index_to_option_label)
            .unwrap_or_else(|| number.to_string()),
        Value::String(text) => {
            let trimmed = text.trim();
            let uppercase = trimmed.to_uppercase();
            if is_valid_option_label(&uppercase, choices.len()) {
                return uppercase;
            }
            if let Ok(index) = trimmed.parse::<usize>() {
                if index < choices.len() {
                    return index_to_option_label(index);
                }
            }
            for (index, choice) in choices.iter().enumerate() {
                if trimmed == choice {
                    return index_to_option_label(index);
                }
            }
            trimmed.to_owned()
        }
        Value::Bool(raw) => raw.to_string(),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}

pub fn normalize_choices_array(values: &[Value], field: &str) -> Result<Vec<String>> {
    let mut choices = Vec::with_capacity(values.len());
    for value in values {
        let Some(text) = stringify_scalar(value) else {
            anyhow::bail!("{field} must contain only scalar string-like values");
        };
        let trimmed = text.trim();
        ensure!(
            !trimmed.is_empty(),
            "{field} must not contain empty strings"
        );
        choices.push(trimmed.to_owned());
    }
    Ok(choices)
}

pub fn stringify_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.to_owned()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(raw) => Some(raw.to_string()),
        _ => None,
    }
}

pub fn is_valid_option_label(label: &str, choice_count: usize) -> bool {
    (0..choice_count).any(|index| index_to_option_label(index) == label)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn option_labels_follow_excel_style_sequence() {
        assert_eq!(index_to_option_label(0), "A");
        assert_eq!(index_to_option_label(25), "Z");
        assert_eq!(index_to_option_label(26), "AA");
        assert_eq!(index_to_option_label(27), "AB");
    }

    #[test]
    fn normalize_answer_handles_indices_labels_and_choice_text() {
        let choices = vec![
            "alpha".to_owned(),
            "beta".to_owned(),
            "gamma".to_owned(),
            "delta".to_owned(),
        ];

        assert_eq!(normalize_answer(&json!(2), &choices), "C");
        assert_eq!(normalize_answer(&json!("1"), &choices), "B");
        assert_eq!(normalize_answer(&json!("delta"), &choices), "D");
        assert_eq!(normalize_answer(&json!("c"), &choices), "C");
    }
}
