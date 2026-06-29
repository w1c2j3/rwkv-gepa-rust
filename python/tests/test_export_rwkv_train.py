import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_rwkv_train.py"
SPEC = importlib.util.spec_from_file_location("export_rwkv_train", SCRIPT_PATH)
export_rwkv_train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_rwkv_train)


def row(task_id="task", assistant="", answer="B"):
    return {
        "task_id": task_id,
        "status": "done",
        "user": "Question?\nChoices:\nA. x\nB. y",
        "assistant": assistant,
        "answer_model": "unit",
        "generated_item_json": json.dumps({"answer": answer}),
    }


class ExportRwkvTrainTests(unittest.TestCase):
    def test_resolve_export_paths_derives_paths_from_run_dir(self):
        args = SimpleNamespace(
            run_dir="data/custom.run",
            success_path="",
            failed_path="",
            output_path="",
            text_only_output_path="",
        )

        paths = export_rwkv_train.resolve_export_paths(args)

        self.assertEqual(
            paths,
            (
                Path("data/custom.run/done/success.jsonl"),
                Path("data/custom.run/done/failed.jsonl"),
                Path("data/custom.run.rwkv_train.jsonl"),
                Path("data/custom.run.rwkv_train.text_only.jsonl"),
            ),
        )

    def test_resolve_export_paths_accepts_explicit_overrides(self):
        args = SimpleNamespace(
            run_dir="data/custom.run",
            success_path="custom/success.jsonl",
            failed_path="custom/failed.jsonl",
            output_path="custom/train.jsonl",
            text_only_output_path="custom/train.text.jsonl",
        )

        paths = export_rwkv_train.resolve_export_paths(args)

        self.assertEqual(
            paths,
            (
                Path("custom/success.jsonl"),
                Path("custom/failed.jsonl"),
                Path("custom/train.jsonl"),
                Path("custom/train.text.jsonl"),
            ),
        )

    def test_normalize_rwkv_content_strips_crlf_and_blank_lines(self):
        text = "  line 1\r\n\r\nline 2\n\n\nline 3  "

        normalized = export_rwkv_train.normalize_rwkv_content(text)

        self.assertEqual(normalized, "line 1\nline 2\nline 3")

    def test_normalize_rwkv_content_strips_outer_user_role(self):
        text = " User: Question?\nAssistant: old answer "

        normalized = export_rwkv_train.normalize_rwkv_content(text, role="User")

        self.assertEqual(normalized, "Question?")

    def test_build_training_assistant_keeps_real_think_and_final_label(self):
        assistant_text = """<think>
reasoning line 1

reasoning line 2
输出规则：最后只输出单个选项字母。
</think>

extra explanation that should be discarded
B"""

        assistant, expected = export_rwkv_train.build_training_assistant(
            row(assistant=assistant_text, answer="b")
        )

        self.assertEqual(expected, "B")
        self.assertEqual(
            assistant,
            "<think>reasoning line 1\nreasoning line 2\n</think>Therefore, the answer is B",
        )

    def test_build_training_assistant_wraps_visible_reasoning_without_think(self):
        assistant, expected = export_rwkv_train.build_training_assistant(
            row(
                assistant=(
                    "Use the definition and eliminate the distractors carefully. "
                    "The only option matching the described condition is B."
                ),
                answer="b",
            )
        )

        self.assertEqual(expected, "B")
        self.assertEqual(
            assistant,
            (
                "<think>Use the definition and eliminate the distractors carefully. "
                "The only option matching the described condition is B.\n"
                "</think>Therefore, the answer is B"
            ),
        )

    def test_build_training_assistant_rejects_short_missing_think(self):
        assistant, expected = export_rwkv_train.build_training_assistant(
            row(assistant="B", answer="b")
        )

        self.assertEqual(expected, "B")
        self.assertEqual(assistant, "")

    def test_build_training_assistant_accepts_thinking_tag_alias(self):
        assistant, expected = export_rwkv_train.build_training_assistant(
            row(assistant="<thinking>\nreal reasoning\n</thinking>\nB", answer="b")
        )

        self.assertEqual(expected, "B")
        self.assertEqual(
            assistant, "<think>real reasoning\n</think>Therefore, the answer is B"
        )

    def test_build_training_assistant_keeps_answer_phrase_inside_think(self):
        assistant, expected = export_rwkv_train.build_training_assistant(
            row(
                assistant="<thinking>\nreal reasoning\nTherefore, the answer is B.\nB\n</thinking>\nB",
                answer="b",
            )
        )

        self.assertEqual(expected, "B")
        self.assertEqual(
            assistant,
            "<think>real reasoning\nTherefore, the answer is B.\n</think>Therefore, the answer is B",
        )

    def test_build_training_assistant_can_emit_no_think_target(self):
        assistant, expected = export_rwkv_train.build_training_assistant(
            row(assistant="<think>\nreasoning\n</think>\nB", answer="b"),
            assistant_style="no-think",
        )

        self.assertEqual(expected, "B")
        self.assertEqual(assistant, "B")

    def test_convert_rows_emits_rwkv_real_think_template(self):
        rows = [
            row("valid", "<think>\nreasoning\n</think>\nB", "B"),
        ]

        exported = export_rwkv_train.convert_rows(rows, "success", "")

        self.assertEqual([item["task_id"] for item in exported], ["valid"])
        self.assertEqual(
            exported[0]["assistant"], "<think>reasoning\n</think>Therefore, the answer is B"
        )
        self.assertTrue(
            exported[0]["text"].endswith(
                "Assistant: <think>reasoning\n</think>Therefore, the answer is B"
            )
        )

    def test_convert_rows_uses_no_think_format(self):
        rows = [
            row("valid", "<think>\nreasoning\n</think>\nB", "B"),
        ]

        exported = export_rwkv_train.convert_rows(
            rows, "success", "", assistant_style="no-think"
        )

        self.assertEqual(exported[0]["assistant"], "B")
        self.assertTrue(exported[0]["text"].endswith("Assistant: B"))


if __name__ == "__main__":
    unittest.main()
