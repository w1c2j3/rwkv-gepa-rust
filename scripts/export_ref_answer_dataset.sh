#!/usr/bin/env bash
set -euo pipefail

input_path="${1:-data/cmmlu_task122_bulk_v5.jsonl}"
all_output_path="${2:-data/cmmlu_task122_bulk_v5.ref_answer_all.jsonl}"
subset_output_path="${3:-data/cmmlu_task122_bulk_v5.ref_answer_8k.jsonl}"
subset_size="${4:-8000}"

jq -s -c '
  reduce .[] as $row ({}; .[$row.task_id] = $row)
  | to_entries[]
  | .value
  | select((.user | type == "string") and (.user != "") and (.ref_answer | test("^[ABCD]$")))
  | {
      task_id,
      sample_id,
      source_status: .status,
      source_answer_model: .answer_model,
      answer_source: "ref_answer",
      user: (.user | sub("^\\s+"; "") | sub("\\s+$"; "")),
      assistant: .ref_answer,
      text: ("User: " + (.user | sub("^\\s+"; "") | sub("\\s+$"; "")) + "\nAssistant: " + .ref_answer)
    }
' "$input_path" > "$all_output_path"

shuf -n "$subset_size" "$all_output_path" > "$subset_output_path"

wc -l "$all_output_path" "$subset_output_path"
