use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use futures::stream::{self, StreamExt, TryStreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

use crate::config::AppConfig;
use crate::types::SampleRecord;
use crate::util::{log, normalize_answer, normalize_choices_array};

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<RowEnvelope>,
}

#[derive(Debug, Deserialize)]
struct RowEnvelope {
    row: Value,
}

pub async fn load_mmlu_samples(config: &AppConfig) -> Result<Vec<SampleRecord>> {
    if config.mmlu.limit == 0 {
        return Ok(Vec::new());
    }

    if let Some(local_jsonl_path) = config.mmlu.local_jsonl_path.as_ref() {
        log(format!(
            "Loading local dataset {} subset={} split={} from {}",
            config.mmlu.dataset_name,
            config.mmlu.subset,
            config.mmlu.split,
            local_jsonl_path.display()
        ));
        return load_local_samples(config, local_jsonl_path);
    }

    log(format!(
        "Loading dataset {} subset={} split={}",
        config.mmlu.dataset_name, config.mmlu.subset, config.mmlu.split
    ));

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs_f64(config.run.request_timeout_seconds))
        .pool_max_idle_per_host(config.concurrency.http_max_idle_per_host)
        .http2_adaptive_window(true)
        .tcp_nodelay(true)
        .build()
        .with_context(|| "Failed to build dataset HTTP client")?;

    let start = config.mmlu.start_index;
    let end = start + config.mmlu.limit;
    let page_size = config.concurrency.dataset_page_size;
    let requests = (start..end)
        .step_by(page_size)
        .map(|offset| {
            let length = (end - offset).min(page_size);
            (offset, length)
        })
        .collect::<Vec<_>>();

    let mut pages = stream::iter(requests)
        .map(|(offset, length)| {
            let client = client.clone();
            let dataset_name = config.mmlu.dataset_name.clone();
            let subset = config.mmlu.subset.clone();
            let split = config.mmlu.split.clone();
            let rows_api_url = config.mmlu.rows_api_url.clone();
            async move {
                let rows = fetch_rows(
                    &client,
                    &rows_api_url,
                    &dataset_name,
                    &subset,
                    &split,
                    offset,
                    length,
                )
                .await?;
                Ok::<_, anyhow::Error>((offset, rows))
            }
        })
        .buffer_unordered(config.concurrency.dataset_requests)
        .try_collect::<Vec<_>>()
        .await?;

    pages.sort_by_key(|(offset, _)| *offset);

    let mut samples = Vec::with_capacity(config.mmlu.limit);
    for (offset, rows) in pages {
        for (index, row) in rows.into_iter().enumerate() {
            let global_index = offset + index;
            samples.push(row_to_sample(
                global_index,
                row,
                &config.mmlu.subset,
                "mmlu",
            )?);
            if samples.len() >= config.mmlu.limit {
                break;
            }
        }
        if samples.len() >= config.mmlu.limit {
            break;
        }
    }

    log(format!("Loaded {} MMLU samples", samples.len()));
    Ok(samples)
}

fn load_local_samples(config: &AppConfig, path: &Path) -> Result<Vec<SampleRecord>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open local dataset {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut samples = Vec::with_capacity(config.mmlu.limit);
    let mut record_index = 0usize;

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "Failed to read line {} from local dataset {}",
                line_number + 1,
                path.display()
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if record_index < config.mmlu.start_index {
            record_index += 1;
            continue;
        }
        if samples.len() >= config.mmlu.limit {
            break;
        }

        let row: Value = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "Invalid JSON on line {} in local dataset {}",
                line_number + 1,
                path.display()
            )
        })?;
        samples.push(row_to_sample(
            record_index,
            row,
            &config.mmlu.subset,
            "local",
        )?);
        record_index += 1;
    }

    log(format!("Loaded {} MMLU samples", samples.len()));
    Ok(samples)
}

async fn fetch_rows(
    client: &Client,
    rows_api_url: &str,
    dataset_name: &str,
    subset: &str,
    split: &str,
    offset: usize,
    length: usize,
) -> Result<Vec<Value>> {
    let response = client
        .get(rows_api_url)
        .query(&[
            ("dataset", dataset_name),
            ("config", subset),
            ("split", split),
            ("offset", &offset.to_string()),
            ("length", &length.to_string()),
        ])
        .send()
        .await
        .with_context(|| format!("Failed to fetch dataset rows offset={offset} length={length}"))?
        .error_for_status()
        .with_context(|| format!("Dataset rows request failed offset={offset} length={length}"))?;

    let payload: RowsResponse = response
        .json()
        .await
        .with_context(|| "Failed to decode dataset rows JSON")?;
    Ok(payload.rows.into_iter().map(|entry| entry.row).collect())
}

fn row_to_sample(
    global_index: usize,
    row: Value,
    default_subject: &str,
    generated_prefix: &str,
) -> Result<SampleRecord> {
    let object = row
        .as_object()
        .ok_or_else(|| anyhow!("Dataset row {global_index} is not a JSON object"))?;

    let question = object
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow!("Dataset row {global_index} is missing question"))?
        .to_owned();

    let raw_choices = object
        .get("choices")
        .ok_or_else(|| anyhow!("Dataset row {global_index} is missing choices"))?;
    let choices = extract_choices(raw_choices)
        .with_context(|| format!("Dataset row {global_index} has invalid choices"))?;
    ensure!(
        !choices.is_empty(),
        "Dataset row {global_index} must contain at least one choice"
    );

    let answer = object
        .get("answer")
        .map(|value| normalize_answer(value, &choices))
        .ok_or_else(|| anyhow!("Dataset row {global_index} is missing answer"))?;

    let subject = object
        .get("subject")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .unwrap_or(default_subject)
        .to_owned();
    let sample_id = object
        .get("sample_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("{generated_prefix}_{global_index:06}"));

    Ok(SampleRecord {
        sample_id,
        subject,
        question,
        choices,
        answer,
    })
}

fn extract_choices(value: &Value) -> Result<Vec<String>> {
    match value {
        Value::Array(items) => normalize_choices_array(items, "choices"),
        Value::Object(map) => {
            for key in ["text", "choices", "values", "labels", "value"] {
                if let Some(candidate) = map.get(key) {
                    return extract_choices(candidate);
                }
            }
            Err(anyhow!("Unsupported choices object layout"))
        }
        _ => Err(anyhow!("choices must be an array or array-like object")),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::config::ModelConfig;
    use crate::config::{
        AppConfig, ConcurrencyConfig, MmluConfig, OutputConfig, RewriteConfig, RunConfig,
        RuntimeConfig,
    };

    #[tokio::test(flavor = "current_thread")]
    async fn local_jsonl_source_supports_explicit_and_generated_sample_ids() {
        let path = temp_path("local-source");
        std::fs::write(
            &path,
            concat!(
                "{\"sample_id\":\"custom_001\",\"subject\":\"ethics\",\"question\":\"Q1\",\"choices\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":\"B\"}\n",
                "{\"subject\":\"ethics\",\"question\":\"Q2\",\"choices\":[\"A2\",\"B2\",\"C2\",\"D2\"],\"answer\":2}\n"
            ),
        )
        .expect("source fixture should be writable");

        let config = test_config(Some(path.clone()), 0, 2);
        let samples = load_mmlu_samples(&config)
            .await
            .expect("local samples should load");

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].sample_id, "custom_001");
        assert_eq!(samples[1].sample_id, "local_000001");
        assert_eq!(samples[1].answer, "C");

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn local_jsonl_source_respects_start_index_and_limit() {
        let path = temp_path("local-slice");
        std::fs::write(
            &path,
            concat!(
                "{\"question\":\"Q0\",\"choices\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":\"A\"}\n",
                "{\"question\":\"Q1\",\"choices\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":\"B\"}\n",
                "{\"question\":\"Q2\",\"choices\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":\"C\"}\n"
            ),
        )
        .expect("source fixture should be writable");

        let config = test_config(Some(path.clone()), 1, 1);
        let samples = load_mmlu_samples(&config)
            .await
            .expect("local samples should load");

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].question, "Q1");
        assert_eq!(samples[0].sample_id, "local_000001");

        let _ = std::fs::remove_file(path);
    }

    fn test_config(
        local_jsonl_path: Option<PathBuf>,
        start_index: usize,
        limit: usize,
    ) -> AppConfig {
        AppConfig {
            mmlu: MmluConfig {
                dataset_name: "local/questions".to_owned(),
                subset: "all".to_owned(),
                split: "train".to_owned(),
                rows_api_url: "https://datasets-server.huggingface.co/rows".to_owned(),
                local_jsonl_path,
                limit,
                start_index,
            },
            rewrite: RewriteConfig {
                model: ModelConfig {
                    name: "rewrite-model".to_owned(),
                    base_url: "https://api.example.com/v1".to_owned(),
                    api_key: "key".to_owned(),
                    enable_thinking: false,
                    reasoning_effort: None,
                },
                variant_count: 1,
                parse_retry_times: 0,
            },
            answer_models: vec![ModelConfig {
                name: "answer-model".to_owned(),
                base_url: "https://api.example.com/v1".to_owned(),
                api_key: "key".to_owned(),
                enable_thinking: false,
                reasoning_effort: None,
            }],
            output: OutputConfig {
                dir: PathBuf::from("data"),
                original_jsonl: PathBuf::from("data/original.jsonl"),
                variants_jsonl: PathBuf::from("data/variants.jsonl"),
                responses_jsonl: PathBuf::from("data/responses.jsonl"),
            },
            run: RunConfig {
                resume: false,
                request_timeout_seconds: 30.0,
                max_retries: 0,
                retry_backoff_seconds: 0.1,
            },
            concurrency: ConcurrencyConfig {
                dataset_requests: 1,
                dataset_page_size: 16,
                rewrite_requests: 1,
                answer_requests: 1,
                original_writer_capacity: 8,
                variant_writer_capacity: 8,
                response_writer_capacity: 8,
                http_max_idle_per_host: 8,
            },
            runtime: RuntimeConfig {
                worker_threads: 1,
                max_blocking_threads: 1,
            },
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rwkv-gepa-rust-v1-{name}-{}-{unique}.jsonl",
            std::process::id()
        ))
    }
}
