#!/usr/bin/env bash
set -euo pipefail

input_path="${1:-data/cmmlu_task122_bulk_v5.jsonl}"
output_path="${2:-data/cmmlu_task122_bulk_v5.done_train.jsonl}"
text_only_output_path="${3:-data/cmmlu_task122_bulk_v5.done_text_train.jsonl}"

jq -c '
  select(.status == "done")
  | {
      user: (.user // .rewritten_user),
      assistant: .assistant,
      text: ("User: " + ((.user // .rewritten_user) | tostring) + "\nAssistant: " + (.assistant | tostring))
    }
' "$input_path" > "$output_path"

jq -c '
  select(.status == "done")
  | {
      text: ("User: " + ((.user // .rewritten_user) | tostring) + "\nAssistant: " + (.assistant | tostring))
    }
' "$input_path" > "$text_only_output_path"

wc -l "$output_path"
wc -l "$text_only_output_path"
