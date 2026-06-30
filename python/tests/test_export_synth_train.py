import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_synth_train.py"
SPEC = importlib.util.spec_from_file_location("export_synth_train", SCRIPT_PATH)
export_synth_train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_synth_train)


def generated_row(
    user=(
        "A box contains 12 red balls and 8 blue balls. If 5 red balls are added "
        "and 3 blue balls are removed, how many balls are in the box?"
    ),
    answer="22",
    meta=None,
):
    payload = {
        "user": user,
        "answer": answer,
        "meta": {
            "task_kind": "math",
            "answer_style": "final_only",
            "semantic_plan": "simple arithmetic",
            "validation_contract": "exact numeric answer",
            "changed_factor": "numbers",
            "diversity_signature": "math-addition",
            **(meta or {}),
        },
    }
    return {
        "task_id": "generated",
        "status": "generated",
        "user": user,
        "expected_answer": answer,
        "generated_item_json": json.dumps(payload),
    }


class ExportSynthTrainTests(unittest.TestCase):
    def test_convert_rows_drops_source_metadata(self):
        rows = [
            generated_row(
                meta={
                    "source_sample_id": "raw-source",
                    "benchmark_name": "heldout",
                    "skill_node": "addition",
                }
            )
        ]

        train_rows, task_rows, issues = export_synth_train.convert_rows(
            rows, "unit_experiment"
        )

        self.assertEqual(issues, [])
        self.assertNotIn("source_sample_id", train_rows[0]["meta"])
        self.assertNotIn("benchmark_name", train_rows[0]["meta"])
        self.assertEqual(train_rows[0]["meta"]["skill_node"], "addition")
        self.assertEqual(task_rows[0]["domain"], "math")

    def test_audit_rejects_duplicate_prompt_answer_conflict(self):
        rows = [
            generated_row(answer="4", meta={"diversity_signature": "a"}),
            generated_row(answer="5", meta={"diversity_signature": "b"}),
        ]
        train_rows, task_rows, _ = export_synth_train.convert_rows(rows, "unit")

        report = export_synth_train.audit_train_rows(train_rows, task_rows, {}, {})

        self.assertEqual(report["issue_count"], 1)
        self.assertEqual(report["issues"][0]["kind"], "answer_conflict")

    def test_main_hard_gate_writes_report_not_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated_path = root / "tasks.jsonl"
            output_path = root / "train.jsonl"
            tasks_output_path = root / "train_tasks.jsonl"
            report_path = root / "audit.json"
            generated_path.write_text(
                "\n".join(
                    [
                        json.dumps(generated_row(answer="4", meta={"diversity_signature": "a"})),
                        json.dumps(generated_row(answer="5", meta={"diversity_signature": "b"})),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original_argv = export_synth_train.sys.argv
            export_synth_train.sys.argv = [
                "export_synth_train.py",
                "--generated-path",
                str(generated_path),
                "--output-path",
                str(output_path),
                "--tasks-output-path",
                str(tasks_output_path),
                "--report-path",
                str(report_path),
                "--experiment",
                "unit",
            ]
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = export_synth_train.main()
            finally:
                export_synth_train.sys.argv = original_argv

            self.assertEqual(code, 2)
            self.assertTrue(report_path.exists())
            self.assertFalse(output_path.exists())
            self.assertFalse(tasks_output_path.exists())


if __name__ == "__main__":
    unittest.main()
