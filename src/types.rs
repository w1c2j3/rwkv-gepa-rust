use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRecord {
    pub sample_id: String,
    pub subject: String,
    pub question: String,
    pub choices: Vec<String>,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantRecord {
    pub sample_id: String,
    pub variant_id: String,
    pub subject: String,
    pub rewrite_model: String,
    pub question: String,
    pub choices: Vec<String>,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseRecord {
    pub sample_id: String,
    pub variant_id: String,
    pub subject: String,
    pub rewrite_model: String,
    pub answer_model: String,
    pub prompt: String,
    pub answer: String,
    pub model_reasoning: Option<String>,
    pub model_response: String,
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub content: String,
    pub reasoning: Option<String>,
}
