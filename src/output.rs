use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, ensure};
use serde::{Serialize, de::DeserializeOwned};
use serde_jsonlines::{append_json_lines, json_lines};

use crate::config::OutputConfig;
use crate::types::{GeneratedItemDraft, OutputRow, PendingTask};

#[derive(Clone)]
pub(crate) struct OutputPaths {
    pub(crate) generate_jsonl_path: PathBuf,
    pub(crate) done_success_jsonl_path: PathBuf,
    pub(crate) done_failed_jsonl_path: PathBuf,
}

#[derive(Clone)]
pub(crate) struct GeneratedTaskWriter {
    generate_jsonl_path: PathBuf,
    write_lock: Arc<Mutex<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunStatus {
    Generated,
    Done,
}

impl GeneratedTaskWriter {
    pub(crate) fn new(paths: &OutputPaths) -> Self {
        Self {
            generate_jsonl_path: paths.generate_jsonl_path.clone(),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    pub(crate) fn append_generated_tasks(&self, tasks: &[PendingTask]) -> Result<()> {
        if tasks.is_empty() {
            return Ok(());
        }
        let rows = tasks.iter().map(generated_output_row).collect::<Vec<_>>();
        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| anyhow!("generated task writer lock poisoned"))?;
        append_jsonl(&self.generate_jsonl_path, &rows)
    }
}

impl RunStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Generated => "generated",
            Self::Done => "done",
        }
    }
}

fn generated_output_row(task: &PendingTask) -> OutputRow {
    OutputRow {
        task_id: task.task_id.clone(),
        status: RunStatus::Generated.as_str().to_owned(),
        user: task.user.clone(),
        generated_item_json: task.generated_item_json.clone(),
        answer_model: String::new(),
        assistant: String::new(),
        text: String::new(),
    }
}

pub(crate) fn done_output_path(paths: &OutputPaths, correct: bool) -> &Path {
    if correct {
        &paths.done_success_jsonl_path
    } else {
        &paths.done_failed_jsonl_path
    }
}

fn read_jsonl<T: DeserializeOwned>(path: &Path, label: &str) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for (line_no, item) in json_lines::<T, _>(path)?.enumerate() {
        out.push(item.with_context(|| {
            format!(
                "invalid {label} JSONL line {} in {}",
                line_no + 1,
                path.display()
            )
        })?);
    }
    Ok(out)
}

pub(crate) fn append_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    append_json_lines(path, rows.iter())?;
    Ok(())
}

pub(crate) fn build_output_paths(output: &OutputConfig) -> Result<OutputPaths> {
    let base_dir = output
        .jsonl_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("data"));
    let dataset_name = output
        .jsonl_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::trim)
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| anyhow!("output.jsonl_path must include a valid file stem"))?;
    let dataset_dir = base_dir.join(dataset_name);

    Ok(OutputPaths {
        generate_jsonl_path: dataset_dir.join("generate").join("tasks.jsonl"),
        done_success_jsonl_path: dataset_dir.join("done").join("success.jsonl"),
        done_failed_jsonl_path: dataset_dir.join("done").join("failed.jsonl"),
    })
}

pub(crate) fn prepare_output(paths: &OutputPaths, resume: bool) -> Result<()> {
    for parent in [
        paths.generate_jsonl_path.parent(),
        paths.done_success_jsonl_path.parent(),
        paths.done_failed_jsonl_path.parent(),
    ]
    .into_iter()
    .flatten()
    {
        fs::create_dir_all(parent)?;
    }
    if !resume {
        File::create(&paths.generate_jsonl_path)?;
        File::create(&paths.done_success_jsonl_path)?;
        File::create(&paths.done_failed_jsonl_path)?;
    }
    Ok(())
}

pub(crate) fn load_resume_rows(paths: &OutputPaths) -> Result<HashMap<String, OutputRow>> {
    let mut rows = HashMap::new();

    for (path, label) in [
        (&paths.generate_jsonl_path, "generated output"),
        (&paths.done_success_jsonl_path, "done success output"),
        (&paths.done_failed_jsonl_path, "done failed output"),
    ] {
        if !path.exists() {
            continue;
        }
        for row in read_jsonl::<OutputRow>(path, label)? {
            if let Some(existing) = rows.get_mut(&row.task_id) {
                merge_output_row(existing, row);
            } else {
                rows.insert(row.task_id.clone(), row);
            }
        }
    }

    Ok(rows)
}

pub(crate) fn summarize_resume_rows(rows: &HashMap<String, OutputRow>) -> Result<(usize, usize)> {
    rows.values()
        .try_fold((0usize, 0usize), |(generated, done), row| {
            Ok(match parse_row_status(row)? {
                RunStatus::Generated => (generated + 1, done),
                RunStatus::Done => (generated, done + 1),
            })
        })
}

pub(crate) fn pending_task_from_output_row(row: &OutputRow) -> Result<PendingTask> {
    let user = row.user.trim();
    ensure!(
        !user.is_empty(),
        "generated resume row is missing user for task {}",
        row.task_id
    );
    let expected_answer = expected_answer_from_generated_item_json(&row.generated_item_json)
        .with_context(|| {
            format!(
                "generated resume row has invalid generated_item_json for task {}",
                row.task_id
            )
        })?;
    Ok(PendingTask {
        task_id: row.task_id.clone(),
        user: user.to_owned(),
        expected_answer,
        generated_item_json: row.generated_item_json.trim().to_owned(),
    })
}

pub(crate) fn parse_row_status(row: &OutputRow) -> Result<RunStatus> {
    let status = row.status.trim();
    match status {
        "generated" => Ok(RunStatus::Generated),
        "done" => Ok(RunStatus::Done),
        _ => Err(anyhow!(
            "unsupported output status {:?} for task {}",
            row.status,
            row.task_id
        )),
    }
}

fn expected_answer_from_generated_item_json(text: &str) -> Result<String> {
    let draft: GeneratedItemDraft = serde_json::from_str(text)
        .context("generated_item_json is not valid GeneratedItemDraft JSON")?;
    let answer = draft.answer.trim().to_owned();
    ensure!(!answer.is_empty(), "generated_item_json answer is empty");
    Ok(answer)
}

fn merge_output_row(existing: &mut OutputRow, incoming: OutputRow) {
    existing.status = incoming.status;
    if !incoming.user.is_empty() {
        existing.user = incoming.user;
    }
    if !incoming.generated_item_json.is_empty() {
        existing.generated_item_json = incoming.generated_item_json;
    }
    if !incoming.answer_model.is_empty() {
        existing.answer_model = incoming.answer_model;
    }
    if !incoming.assistant.is_empty() {
        existing.assistant = incoming.assistant;
    }
    if !incoming.text.is_empty() {
        existing.text = incoming.text;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{pending_task, temp_path};

    fn paths(name: &str) -> OutputPaths {
        let base = temp_path(name);
        OutputPaths {
            generate_jsonl_path: base.join("generate/tasks.jsonl"),
            done_success_jsonl_path: base.join("done/success.jsonl"),
            done_failed_jsonl_path: base.join("done/failed.jsonl"),
        }
    }

    #[test]
    fn build_output_paths_uses_dataset_stem_directory() {
        let output = OutputConfig {
            jsonl_path: PathBuf::from("data/train_set.jsonl"),
        };

        let paths = build_output_paths(&output).expect("paths should build");

        assert_eq!(
            paths.generate_jsonl_path,
            PathBuf::from("data/train_set/generate/tasks.jsonl")
        );
        assert_eq!(
            paths.done_success_jsonl_path,
            PathBuf::from("data/train_set/done/success.jsonl")
        );
        assert_eq!(
            paths.done_failed_jsonl_path,
            PathBuf::from("data/train_set/done/failed.jsonl")
        );
    }

    #[test]
    fn prepare_output_creates_and_truncates_output_files_when_not_resuming() {
        let paths = paths("output/prepare");
        append_jsonl(
            &paths.generate_jsonl_path,
            &[generated_output_row(&pending_task(
                "task_q000",
                "User?",
                "A",
            ))],
        )
        .expect("preexisting row should write");

        prepare_output(&paths, false).expect("output should prepare");

        assert_eq!(
            fs::read_to_string(&paths.generate_jsonl_path).expect("file should exist"),
            ""
        );
        assert!(paths.done_success_jsonl_path.exists());
        assert!(paths.done_failed_jsonl_path.exists());
    }

    #[test]
    fn generated_writer_and_resume_loader_merge_generated_and_done_rows() {
        let paths = paths("output/resume");
        prepare_output(&paths, false).expect("output should prepare");
        let task = pending_task("sample_q000", "User?", "A");
        GeneratedTaskWriter::new(&paths)
            .append_generated_tasks(std::slice::from_ref(&task))
            .expect("generated task should write");
        append_jsonl(
            &paths.done_success_jsonl_path,
            &[OutputRow {
                task_id: task.task_id.clone(),
                status: RunStatus::Done.as_str().to_owned(),
                user: String::new(),
                generated_item_json: String::new(),
                answer_model: "answer-model".to_owned(),
                assistant: "A".to_owned(),
                text: "User: User?\nAssistant: A".to_owned(),
            }],
        )
        .expect("done row should write");

        let rows = load_resume_rows(&paths).expect("resume rows should load");
        let row = rows.get("sample_q000").expect("merged row should exist");

        assert_eq!(rows.len(), 1);
        assert_eq!(row.status, "done");
        assert_eq!(row.user, "User?");
        assert_eq!(row.generated_item_json, task.generated_item_json);
        assert_eq!(row.answer_model, "answer-model");
        assert_eq!(
            summarize_resume_rows(&rows).expect("summary should compute"),
            (0, 1)
        );
        let pending = pending_task_from_output_row(row).expect("pending task should rebuild");
        assert_eq!(pending.expected_answer, "A");
    }

    #[test]
    fn parse_row_status_rejects_unknown_values() {
        let mut row = generated_output_row(&pending_task("task_q000", "User?", "A"));
        assert_eq!(
            parse_row_status(&row).expect("generated should parse"),
            RunStatus::Generated
        );

        row.status = "weird".to_owned();
        assert!(
            parse_row_status(&row)
                .expect_err("unknown status should fail")
                .to_string()
                .contains("unsupported output status")
        );
    }
}
