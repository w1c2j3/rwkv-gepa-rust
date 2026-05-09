use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone)]
pub(crate) struct SourceSample {
    pub(crate) sample_id: String,
    pub(crate) source_user: String,
    pub(crate) source_meta: Value,
}

#[derive(Clone)]
pub(crate) struct GenerateJob {
    pub(crate) sample: SourceSample,
    pub(crate) missing_indices: Vec<usize>,
    pub(crate) accepted_tasks: Vec<PendingTask>,
}

#[derive(Clone)]
pub(crate) struct PendingTask {
    pub(crate) task_id: String,
    pub(crate) user: String,
    pub(crate) expected_answer: String,
    pub(crate) generated_item_json: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct OutputRow {
    pub(crate) task_id: String,
    pub(crate) status: String,
    #[serde(
        default,
        alias = "rewritten_user",
        skip_serializing_if = "String::is_empty"
    )]
    pub(crate) user: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) generated_item_json: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) answer_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) assistant: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) text: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GeneratedItemDraft {
    pub(crate) user: String,
    pub(crate) answer: String,
    #[serde(default)]
    pub(crate) meta: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_row_accepts_legacy_rewritten_user_alias() {
        let row: OutputRow = serde_json::from_value(json!({
            "task_id": "task",
            "status": "generated",
            "rewritten_user": "legacy user",
            "generated_item_json": "",
            "answer_model": "",
            "assistant": "",
            "text": ""
        }))
        .expect("row should deserialize");

        assert_eq!(row.user, "legacy user");
    }

    #[test]
    fn generated_item_draft_defaults_missing_meta_to_null() {
        let draft: GeneratedItemDraft =
            serde_json::from_str(r#"{"user":"u","answer":"A"}"#).expect("draft should parse");

        assert_eq!(draft.meta, Value::Null);
    }
}
