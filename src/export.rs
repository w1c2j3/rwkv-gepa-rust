use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::{Map, Value};

use crate::util::{is_valid_option_label, normalize_answer, normalize_choices_array};

#[derive(Debug, Clone)]
struct ExportRow {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: String,
}

pub fn run(input: &Path, output: &Path) -> Result<()> {
    let rows = load_rows(input)?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory {}", parent.display()))?;
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("question", DataType::Utf8, false),
        Field::new("subject", DataType::Utf8, false),
        Field::new(
            "choices",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("answer", DataType::Utf8, false),
    ]));

    let questions = Arc::new(StringArray::from(
        rows.iter()
            .map(|row| row.question.clone())
            .collect::<Vec<String>>(),
    )) as ArrayRef;
    let subjects = Arc::new(StringArray::from(
        rows.iter()
            .map(|row| row.subject.clone())
            .collect::<Vec<String>>(),
    )) as ArrayRef;
    let answers = Arc::new(StringArray::from(
        rows.iter()
            .map(|row| row.answer.clone())
            .collect::<Vec<String>>(),
    )) as ArrayRef;

    let mut choices_builder = ListBuilder::new(StringBuilder::new());
    for row in &rows {
        for choice in &row.choices {
            choices_builder.values().append_value(choice);
        }
        choices_builder.append(true);
    }
    let choices = Arc::new(choices_builder.finish()) as ArrayRef;

    let batch = RecordBatch::try_new(schema.clone(), vec![questions, subjects, choices, answers])
        .with_context(|| "Failed to build Arrow record batch")?;

    let file = File::create(output)
        .with_context(|| format!("Failed to create output file {}", output.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .with_context(|| "Failed to create parquet writer")?;
    writer
        .write(&batch)
        .with_context(|| "Failed to write parquet batch")?;
    writer
        .close()
        .with_context(|| "Failed to close parquet writer")?;

    println!("Exported {} rows to {}", rows.len(), output.display());
    Ok(())
}

fn load_rows(path: &Path) -> Result<Vec<ExportRow>> {
    if !path.exists() {
        bail!("Input file not found: {}", path.display());
    }

    let file =
        File::open(path).with_context(|| format!("Failed to open input {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "Failed to read line {} from {}",
                line_number + 1,
                path.display()
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let payload: Value = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "Line {} in {} is not valid JSON",
                line_number + 1,
                path.display()
            )
        })?;
        rows.push(validate_record(&payload, line_number + 1)?);
    }

    Ok(rows)
}

fn validate_record(value: &Value, line_number: usize) -> Result<ExportRow> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("Line {line_number}: expected a JSON object"))?;

    let question = required_string_field(object, "question", line_number)?;
    let subject = required_string_field(object, "subject", line_number)?;
    let choices = match object.get("choices") {
        Some(Value::Array(items)) => normalize_choices_array(items, "choices")
            .with_context(|| format!("Line {line_number}: invalid choices"))?,
        _ => bail!("Line {line_number}: choices must be a list"),
    };
    if choices.len() != 4 {
        bail!("Line {line_number}: choices must contain exactly 4 items");
    }

    let answer = object
        .get("answer")
        .map(|value| normalize_answer(value, &choices))
        .ok_or_else(|| anyhow!("Line {line_number}: answer is required"))?;
    if !is_valid_option_label(&answer, choices.len()) {
        bail!("Line {line_number}: answer must be one of A/B/C/D");
    }

    Ok(ExportRow {
        question,
        subject,
        choices,
        answer,
    })
}

fn required_string_field(
    object: &Map<String, Value>,
    field: &str,
    line_number: usize,
) -> Result<String> {
    let value = object
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("Line {line_number}: {field} must be a non-empty string"))?;
    Ok(value.to_owned())
}
