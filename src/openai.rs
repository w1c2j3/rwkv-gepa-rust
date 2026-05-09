use std::fmt::{Display, Formatter};
use std::time::Duration;

use anyhow::{Result, anyhow, ensure};
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, Response, StatusCode};
use serde::Serialize;
use serde_json::Value;

use crate::config::{ModelConfig, RunConfig};
use crate::text::preview_text;

#[derive(Clone)]
pub(crate) struct OpenAiClient {
    http: Client,
    endpoint: String,
    api_key: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChatResult {
    pub(crate) content: String,
    pub(crate) reasoning: Option<String>,
}

#[derive(Debug)]
struct RequestError {
    transient: bool,
    message: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingRequest<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct ThinkingRequest<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl OpenAiClient {
    pub(crate) fn new(model: &ModelConfig, run: &RunConfig) -> Result<Self> {
        let mut builder = Client::builder()
            .timeout(Duration::from_secs_f64(run.request_timeout_seconds))
            .user_agent("curl/8.5.0");
        if run.disable_env_proxy {
            builder = builder.no_proxy();
        }
        if run.force_http1 {
            builder = builder.http1_only();
        }
        Ok(Self {
            http: builder.build()?,
            endpoint: model.endpoint.clone(),
            api_key: model.api_key.clone(),
        })
    }

    pub(crate) async fn chat(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> Result<ChatResult> {
        let max_attempts = 3usize;
        for attempt in 0..max_attempts {
            match self.try_chat(model, prompt, json_output).await {
                Ok(result) => return Ok(result),
                Err(err) if err.transient && attempt + 1 < max_attempts => {
                    eprintln!(
                        "transient chat error for model {} (attempt {}/{}): {}",
                        model.model_name,
                        attempt + 1,
                        max_attempts,
                        err
                    );
                    tokio::time::sleep(Duration::from_secs((attempt + 1) as u64 * 2)).await;
                }
                Err(err) => return Err(err.into()),
            }
        }
        unreachable!("chat retry loop should either return or fail")
    }

    async fn try_chat(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<ChatResult, RequestError> {
        let response = self.send_chat_request(model, prompt, json_output).await?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = response.text().await.map_err(RequestError::http)?;
        if !status.is_success() {
            return Err(RequestError::api(status, body));
        }
        if model.stream || content_type_is_event_stream(content_type.as_deref()) {
            parse_stream_chat_result(&body).map_err(RequestError::parse)
        } else {
            parse_chat_result(&body).map_err(RequestError::parse)
        }
    }

    async fn send_chat_request(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<Response, RequestError> {
        self.http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&chat_request(model, prompt, json_output))
            .send()
            .await
            .map_err(RequestError::http)
    }
}

impl RequestError {
    fn http(err: reqwest::Error) -> Self {
        Self {
            transient: err.is_timeout() || err.is_connect() || err.is_body(),
            message: err.to_string(),
        }
    }

    fn api(status: StatusCode, body: String) -> Self {
        Self {
            transient: status.is_server_error() || status.as_u16() == 429,
            message: format!(
                "chat completion failed with status {}: {}",
                status,
                preview_text(&body, 400)
            ),
        }
    }

    fn parse(err: anyhow::Error) -> Self {
        Self {
            transient: false,
            message: err.to_string(),
        }
    }
}

impl Display for RequestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

fn chat_request<'a>(model: &'a ModelConfig, prompt: &'a str, json_output: bool) -> ChatRequest<'a> {
    let mut messages = Vec::with_capacity(2);
    if let Some(system_prompt) = model.system_prompt.as_deref() {
        messages.push(ChatMessage {
            role: "system",
            content: system_prompt,
        });
    }
    messages.push(ChatMessage {
        role: "user",
        content: prompt,
    });

    ChatRequest {
        model: &model.model_name,
        messages,
        max_completion_tokens: model.max_completion_tokens,
        reasoning_effort: model.reasoning_effort.as_deref(),
        stream: model.stream && !json_output,
        thinking: model.thinking.as_ref().map(|thinking| ThinkingRequest {
            kind: thinking.kind.as_str(),
        }),
        enable_thinking: model.enable_thinking,
        response_format: json_output.then_some(ResponseFormat {
            kind: "json_object",
        }),
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn content_type_is_event_stream(content_type: Option<&str>) -> bool {
    content_type
        .map(|value| value.to_ascii_lowercase().contains("text/event-stream"))
        .unwrap_or(false)
}

fn parse_chat_result(body: &str) -> Result<ChatResult> {
    let value = serde_json::from_str::<Value>(body)?;
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| anyhow!("chat response is missing choices[0]"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow!("chat response is missing choices[0].message"))?;
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    let reasoning = ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|key| {
            message
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        });
    ensure!(
        !content.is_empty() || reasoning.is_some(),
        "chat response is empty"
    );
    Ok(ChatResult { content, reasoning })
}

fn parse_stream_chat_result(body: &str) -> Result<ChatResult> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut saw_chunk = false;

    for raw_line in body.lines() {
        let line = raw_line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }

        let value = serde_json::from_str::<Value>(data).map_err(|err| {
            anyhow!(
                "stream chunk is not valid JSON: {:?}: {}",
                preview_text(data, 120),
                err
            )
        })?;
        saw_chunk = true;

        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for choice in choices {
            let delta = choice
                .get("delta")
                .or_else(|| choice.get("message"))
                .unwrap_or(choice);
            append_chunk_text(delta, "reasoning_content", &mut reasoning);
            append_chunk_text(delta, "reasoning", &mut reasoning);
            append_chunk_text(delta, "content", &mut content);
        }
    }

    ensure!(saw_chunk, "stream response did not contain any data chunks");
    let content = content.trim().to_owned();
    let reasoning = reasoning.trim().to_owned();
    ensure!(
        !content.is_empty() || !reasoning.is_empty(),
        "chat stream response is empty"
    );
    Ok(ChatResult {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    })
}

fn append_chunk_text(value: &Value, key: &str, out: &mut String) {
    if let Some(text) = value.get(key).and_then(Value::as_str) {
        out.push_str(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ThinkingSetting, load_config};
    use crate::test_support::model_config;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn chat_request_includes_system_user_and_json_response_format() {
        let mut model = model_config();
        model.stream = true;
        model.thinking = Some(ThinkingSetting {
            kind: "enabled".to_owned(),
        });

        let value =
            serde_json::to_value(chat_request(&model, "hello", true)).expect("request serializes");

        assert_eq!(value["model"], "unit-model");
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["content"], "hello");
        assert!(
            value.get("stream").is_none(),
            "false stream is intentionally omitted from the request body"
        );
        assert_eq!(value["thinking"]["type"], "enabled");
        assert_eq!(value["response_format"]["type"], "json_object");
    }

    #[test]
    fn parse_chat_result_extracts_content_or_reasoning() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": "  B  ",
                    "reasoning_content": " because "
                }
            }]
        })
        .to_string();

        let result = parse_chat_result(&body).expect("chat result should parse");

        assert_eq!(result.content, "B");
        assert_eq!(result.reasoning.as_deref(), Some("because"));
    }

    #[test]
    fn parse_stream_chat_result_merges_content_and_reasoning_chunks() {
        let body = concat!(
            r#"data: {"choices":[{"delta":{"reasoning_content":"why ","content":"A"}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"reasoning":"now","content":"B"}}]}"#,
            "\n\n",
            "data: [DONE]\n"
        );

        let result = parse_stream_chat_result(body).expect("stream result should parse");

        assert_eq!(result.content, "AB");
        assert_eq!(result.reasoning.as_deref(), Some("why now"));
    }

    #[test]
    fn content_type_detection_is_case_insensitive() {
        assert!(content_type_is_event_stream(Some(
            "TEXT/EVENT-STREAM; charset=utf-8"
        )));
        assert!(!content_type_is_event_stream(Some("application/json")));
        assert!(!content_type_is_event_stream(None));
    }

    #[tokio::test]
    #[ignore = "requires usable credentials in config.cmmlu_task122.bulk.toml"]
    async fn chat_with_real_openai_compatible_api_from_toml_config() {
        let cfg = load_config(Path::new("config.cmmlu_task122.bulk.toml"))
            .expect("test TOML config should load");
        let mut model = cfg.generator.model;
        model.max_completion_tokens = Some(16);
        model.stream = false;
        let client = OpenAiClient::new(&model, &cfg.run).expect("client should build");

        let result = client
            .chat(&model, "Return exactly this text: OK", false)
            .await
            .expect("real API chat should succeed");

        assert!(!result.content.trim().is_empty() || result.reasoning.is_some());
    }
}
