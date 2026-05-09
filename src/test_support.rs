use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::config::ModelConfig;
use crate::types::{GeneratedItemDraft, PendingTask, SourceSample};

pub(crate) fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rwkv_gepa_rust_v1_test_{}_{}_{}",
        std::process::id(),
        nanos,
        name
    ))
}

pub(crate) fn write_text(path: &PathBuf, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("test temp parent should be creatable");
    }
    fs::write(path, text).expect("test temp file should be writable");
}

pub(crate) fn sample(sample_id: &str, user: &str) -> SourceSample {
    SourceSample {
        sample_id: sample_id.to_owned(),
        source_user: user.to_owned(),
        source_meta: json!({"subject": "unit"}),
    }
}

pub(crate) fn generated_item_json(user: &str, answer: &str) -> String {
    serde_json::to_string(&GeneratedItemDraft {
        user: user.to_owned(),
        answer: answer.to_owned(),
        meta: Value::Null,
    })
    .expect("test generated item should serialize")
}

pub(crate) fn pending_task(task_id: &str, user: &str, answer: &str) -> PendingTask {
    PendingTask {
        task_id: task_id.to_owned(),
        user: user.to_owned(),
        expected_answer: answer.to_owned(),
        generated_item_json: generated_item_json(user, answer),
    }
}

pub(crate) fn model_config() -> ModelConfig {
    ModelConfig {
        endpoint: "https://example.invalid/v1/chat/completions".to_owned(),
        model_name: "unit-model".to_owned(),
        api_key: "unit-key".to_owned(),
        system_prompt: Some("system".to_owned()),
        max_completion_tokens: Some(16),
        reasoning_effort: Some("low".to_owned()),
        stream: false,
        thinking: None,
        enable_thinking: None,
    }
}
