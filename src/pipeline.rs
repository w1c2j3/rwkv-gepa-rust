use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};

use crate::answer::answer_tasks_and_persist;
use crate::config::{load_config, validate_config};
use crate::generation::{ValidatorRuntime, generate_tasks, task_id};
use crate::input::load_samples;
use crate::openai::OpenAiClient;
use crate::output::{
    GeneratedTaskWriter, RunStatus, build_output_paths, load_resume_rows, parse_row_status,
    pending_task_from_output_row, prepare_output, summarize_resume_rows,
};
use crate::prompt::load_prompt_templates;
use crate::types::{GenerateJob, OutputRow, PendingTask, SourceSample};
use crate::util::concurrency_limit;

pub(crate) async fn synthesize(path: &Path, limit: Option<usize>) -> Result<()> {
    let mut cfg = load_config(path)?;
    if let Some(limit) = limit {
        cfg.input.limit = Some(limit);
    }
    validate_config(&cfg)?;
    let prompt_templates =
        load_prompt_templates(&cfg.prompt, cfg.generator.validate_generated_questions)?;

    let samples = load_samples(&cfg)?;
    if samples.is_empty() {
        println!("no samples");
        return Ok(());
    }

    let output_paths = build_output_paths(&cfg.output)?;
    prepare_output(&output_paths, cfg.run.resume)?;
    let generated_task_writer = GeneratedTaskWriter::new(&output_paths);
    let resume_rows = if cfg.run.resume {
        load_resume_rows(&output_paths)?
    } else {
        HashMap::new()
    };
    if !resume_rows.is_empty() {
        let (generated, done) = summarize_resume_rows(&resume_rows)?;
        eprintln!(
            "resuming with {} tracked tasks from {}, {}, {} (generated={}, done={})",
            resume_rows.len(),
            output_paths.generate_jsonl_path.display(),
            output_paths.done_success_jsonl_path.display(),
            output_paths.done_failed_jsonl_path.display(),
            generated,
            done
        );
    }

    let (generator_jobs, resumed_pending_tasks) =
        build_resume_plan(&samples, cfg.generator.variant_count, &resume_rows)?;
    let (answer_clients, answer_models) = if cfg.run.answer_generated {
        (
            cfg.answer_models
                .iter()
                .map(|model| OpenAiClient::new(model, &cfg.run))
                .collect::<Result<Vec<_>>>()?,
            cfg.answer_models.clone(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut generated_now = 0usize;
    let resumed_generated = resumed_pending_tasks.len();
    let mut skipped_generate = 0usize;
    let mut answered_success = 0usize;
    let mut answered_failed = 0usize;
    let mut skipped_answer = 0usize;

    if cfg.run.answer_generated && !resumed_pending_tasks.is_empty() {
        let stats = answer_tasks_and_persist(
            answer_clients.clone(),
            answer_models.clone(),
            resumed_pending_tasks,
            &output_paths,
            cfg.concurrency.answer_requests,
        )
        .await?;
        answered_success += stats.success;
        answered_failed += stats.failed;
        skipped_answer += stats.skipped;
    } else if !cfg.run.answer_generated {
        skipped_answer += resumed_pending_tasks.len();
    }

    if !generator_jobs.is_empty() {
        let client = OpenAiClient::new(&cfg.generator.model, &cfg.run)?;
        let generator = cfg.generator.clone();
        let validator = if generator.validate_generated_questions {
            cfg.validator
                .as_ref()
                .map(|validator| {
                    Ok::<_, anyhow::Error>(ValidatorRuntime {
                        client: OpenAiClient::new(&validator.model, &cfg.run)?,
                        model: validator.model.clone(),
                        json_object_response: validator.json_object_response,
                    })
                })
                .transpose()?
        } else {
            None
        };
        let pb = progress_bar(generator_jobs.len(), "generate");
        let concurrency =
            concurrency_limit(cfg.concurrency.generate_requests, generator_jobs.len());
        let mut stream = stream::iter(generator_jobs)
            .map(|job| {
                let client = client.clone();
                let generator = generator.clone();
                let validator = validator.clone();
                let prompt_templates = prompt_templates.clone();
                let generated_task_writer = generated_task_writer.clone();
                async move {
                    generate_tasks(
                        client,
                        generator,
                        validator,
                        prompt_templates,
                        generated_task_writer,
                        job,
                    )
                    .await
                }
            })
            .buffer_unordered(concurrency);

        while let Some(result) = stream.next().await {
            match result {
                Ok(tasks) => {
                    if !tasks.completed {
                        skipped_generate += 1;
                        eprintln!(
                            "incomplete generate job: accepted {} task(s), rejected {} candidate(s){}",
                            tasks.tasks.len(),
                            tasks.rejected_candidates,
                            tasks
                                .terminal_error
                                .as_deref()
                                .map_or(String::new(), |err| format!(", terminal error: {err}"))
                        );
                    }
                    generated_now += tasks.tasks.len();
                    if cfg.run.answer_generated {
                        let stats = answer_tasks_and_persist(
                            answer_clients.clone(),
                            answer_models.clone(),
                            tasks.tasks,
                            &output_paths,
                            cfg.concurrency.answer_requests,
                        )
                        .await?;
                        answered_success += stats.success;
                        answered_failed += stats.failed;
                        skipped_answer += stats.skipped;
                    } else {
                        skipped_answer += tasks.tasks.len();
                    }
                }
                Err(err) => {
                    skipped_generate += 1;
                    eprintln!("skipped generate job: {err:#}");
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("generate done");
    }

    println!(
        "samples={} total_tasks={} resumed_generated={} generated_now={} answered_success={} answered_failed={} skipped_generate={} skipped_answer={}",
        samples.len(),
        samples.len() * cfg.generator.variant_count,
        resumed_generated,
        generated_now,
        answered_success,
        answered_failed,
        skipped_generate,
        skipped_answer
    );
    Ok(())
}

fn build_resume_plan(
    samples: &[SourceSample],
    variant_count: usize,
    resume_rows: &HashMap<String, OutputRow>,
) -> Result<(Vec<GenerateJob>, Vec<PendingTask>)> {
    let mut generator_jobs = Vec::new();
    let mut pending_tasks = Vec::new();

    for sample in samples {
        let mut missing_indices = Vec::new();
        let mut accepted_tasks = Vec::new();
        for index in 0..variant_count {
            let id = task_id(&sample.sample_id, index);
            match resume_rows.get(&id) {
                Some(row) => {
                    let task = pending_task_from_output_row(row)?;
                    accepted_tasks.push(task.clone());
                    match parse_row_status(row)? {
                        RunStatus::Done => {}
                        RunStatus::Generated => {
                            pending_tasks.push(task);
                        }
                    }
                }
                None => missing_indices.push(index),
            }
        }
        if !missing_indices.is_empty() {
            generator_jobs.push(GenerateJob {
                sample: sample.clone(),
                missing_indices,
                accepted_tasks,
            });
        }
    }

    Ok((generator_jobs, pending_tasks))
}

fn progress_bar(total: usize, label: &str) -> ProgressBar {
    if total == 0 {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:>10} [{bar:40.cyan/blue}] {pos}/{len} {percent:>3}% {elapsed_precise}<{eta_precise}",
        )
        .expect("valid progress bar template"),
    );
    pb.set_prefix(label.to_owned());
    pb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{generated_item_json, sample};

    fn row(task_id: &str, status: &str, user: &str, answer: &str) -> OutputRow {
        OutputRow {
            task_id: task_id.to_owned(),
            status: status.to_owned(),
            user: user.to_owned(),
            generated_item_json: generated_item_json(user, answer),
            answer_model: String::new(),
            assistant: String::new(),
            text: String::new(),
        }
    }

    #[test]
    fn build_resume_plan_separates_missing_generated_and_done_tasks() {
        let samples = vec![sample("sample", "Original?")];
        let mut rows = HashMap::new();
        rows.insert(
            "sample_q000".to_owned(),
            row("sample_q000", "generated", "Generated?", "A"),
        );
        rows.insert(
            "sample_q001".to_owned(),
            row("sample_q001", "done", "Done?", "B"),
        );

        let (jobs, pending) =
            build_resume_plan(&samples, 3, &rows).expect("resume plan should build");

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].task_id, "sample_q000");
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].missing_indices, vec![2]);
        assert_eq!(jobs[0].accepted_tasks.len(), 2);
    }

    #[test]
    fn build_resume_plan_rejects_invalid_resume_status() {
        let samples = vec![sample("sample", "Original?")];
        let mut rows = HashMap::new();
        rows.insert(
            "sample_q000".to_owned(),
            row("sample_q000", "unknown", "Generated?", "A"),
        );

        let err = match build_resume_plan(&samples, 1, &rows) {
            Ok(_) => panic!("invalid status should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("unsupported output status"));
    }
}
