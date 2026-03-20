use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::config::{ConcurrencyConfig, RunConfig};
use crate::types::ChatResult;

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: Client,
    endpoint: String,
    api_key: String,
}

impl OpenAiClient {
    pub fn new(
        base_url: &str,
        api_key: &str,
        run: &RunConfig,
        concurrency: &ConcurrencyConfig,
    ) -> Result<Self> {
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs_f64(run.request_timeout_seconds))
            .pool_max_idle_per_host(concurrency.http_max_idle_per_host)
            .http2_adaptive_window(true)
            .tcp_nodelay(true)
            .build()
            .with_context(|| "Failed to build reqwest client")?;

        Ok(Self {
            http,
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key: api_key.to_owned(),
        })
    }

    pub async fn chat_completion(
        &self,
        model: &str,
        prompt: &str,
        run: &RunConfig,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<ChatResult> {
        let mut last_error: Option<anyhow::Error> = None;
        for attempt in 0..=run.max_retries {
            match self
                .send_request(model, prompt, enable_thinking, reasoning_effort)
                .await
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error);
                    if attempt >= run.max_retries {
                        break;
                    }
                    let delay = run.retry_backoff_seconds * 2_f64.powi(attempt as i32);
                    sleep(Duration::from_secs_f64(delay)).await;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("chat completion failed without error")))
    }

    async fn send_request(
        &self,
        model: &str,
        prompt: &str,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<ChatResult> {
        let mut payload = json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt,
            }],
        });

        if enable_thinking {
            payload["enable_thinking"] = Value::Bool(true);
        }
        if let Some(reasoning_effort) = reasoning_effort {
            payload["reasoning_effort"] = Value::String(reasoning_effort.to_owned());
        }

        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("Request to {} failed", self.endpoint))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .with_context(|| "Failed to read chat completion response body")?;
        if !status.is_success() {
            bail!("chat completion failed with status {status}: {body}");
        }

        let payload: Value =
            serde_json::from_str(&body).with_context(|| "Failed to decode chat completion JSON")?;
        let message = payload
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .ok_or_else(|| anyhow!("Response is missing choices[0].message"))?;

        let content = extract_message_text(message.get("content"))
            .ok_or_else(|| anyhow!("Model response content is empty"))?;
        let reasoning = message
            .get("reasoning_content")
            .and_then(|value| extract_message_text(Some(value)))
            .or_else(|| {
                message
                    .get("reasoning")
                    .and_then(|value| extract_message_text(Some(value)))
            });

        Ok(ChatResult { content, reasoning })
    }
}

fn extract_message_text(value: Option<&Value>) -> Option<String> {
    let value = value?;
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_owned())
            }
        }
        Value::Array(parts) => {
            let mut output = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    output.push_str(text);
                } else if let Some(text) = part.get("content").and_then(Value::as_str) {
                    output.push_str(text);
                }
            }
            let trimmed = output.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_owned())
            }
        }
        Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned),
        _ => None,
    }
}
