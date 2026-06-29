pub(crate) fn normalize_compare_text(text: &str) -> String {
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

pub(crate) fn sanitize_training_user_prompt(text: &str) -> String {
    let mut cleaned = text.trim().to_owned();

    loop {
        let Some(rest) = cleaned.strip_prefix("You are a very talented expert in ") else {
            break;
        };
        let Some(period_index) = rest.find('.') else {
            break;
        };
        cleaned = rest[period_index + 1..].trim_start().to_owned();
    }

    cleaned.trim().to_owned()
}

pub(crate) fn preview_text(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    let char_count = trimmed.chars().count();
    if char_count <= limit {
        trimmed.to_owned()
    } else {
        format!("{}...", trimmed.chars().take(limit).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_compare_text_removes_spacing_and_punctuation() {
        assert_eq!(normalize_compare_text(" A， B. C "), "abc");
        assert_eq!(normalize_compare_text("选择：甲、乙"), "选择甲乙");
    }

    #[test]
    fn sanitize_training_user_prompt_strips_expert_prefixes() {
        let text = "You are a very talented expert in math. Question: 1+1?";
        assert_eq!(sanitize_training_user_prompt(text), "Question: 1+1?");
    }

    #[test]
    fn preview_text_trims_and_truncates_by_chars() {
        assert_eq!(preview_text("  abc  ", 3), "abc");
        assert_eq!(preview_text("你好世界", 2), "你好...");
    }
}
