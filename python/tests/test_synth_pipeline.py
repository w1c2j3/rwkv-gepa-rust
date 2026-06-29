import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import pipeline


def profile(task_kind="knowledge"):
    return pipeline.PromptProfile(
        name=task_kind,
        task_kind=task_kind,
        answer_check=pipeline.default_answer_check(task_kind),
        strategy="subspace_contract_diversity",
        axes=("domain", "skill_node", "changed_factor"),
        generation_template=(
            "profile={{profile_name}}\n"
            "kind={{task_kind}}\n"
            "synthesis={{synthesis_profile_json}}\n"
            "sample={{source_sample_json}}\n"
            "accepted={{accepted_samples_json}}\n"
            "{{feedback_block}}"
        ),
        validation_template="",
    )


def generated_json(user="New question?", answer="B", signature="sig-a"):
    return json.dumps(
        {
            "items": [
                {
                    "user": user,
                    "answer": answer,
                    "meta": {
                        "semantic_plan": "same domain, changed key condition",
                        "validation_contract": "answer must follow changed condition",
                        "changed_factor": "key condition",
                        "diversity_signature": signature,
                    },
                }
            ]
        }
    )


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def chat(self, model, prompt):
        self.prompts.append(prompt)
        return self.text


class SynthPipelineTests(unittest.TestCase):
    def test_build_generation_prompt_includes_synthesis_controls(self):
        source = pipeline.SourceSample(
            sample_id="s1",
            user="Original question?",
            meta={"domain": "biology", "skill_node": "temperature"},
        )

        prompt_text = pipeline.build_generation_prompt(
            profile(), source, 2, [], "duplicate signature"
        )

        self.assertIn("subspace_plan", prompt_text)
        self.assertIn("semantic_plan", prompt_text)
        self.assertIn("validation_contract", prompt_text)
        self.assertIn("selection_policy", prompt_text)
        self.assertIn("duplicate signature", prompt_text)

    def test_parse_generated_items_requires_circuit_meta(self):
        source = pipeline.SourceSample("s1", "Original?", {})

        items, rejected = pipeline.parse_generated_items(
            generated_json(), source, profile()
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].meta["task_kind"], "knowledge")
        self.assertEqual(items[0].meta["answer_check"], "exact_text")

        _, rejected = pipeline.parse_generated_items(
            json.dumps({"items": [{"user": "New?", "answer": "A", "meta": {}}]}),
            source,
            profile(),
        )
        self.assertIn("missing meta.semantic_plan", rejected[0])

    def test_parse_generated_items_accepts_nested_function_calling_answer(self):
        source = pipeline.SourceSample("s1", "Call a tool?", {})
        raw = json.dumps(
            {
                "items": [
                    {
                        "user": "Call search for x.",
                        "answer": {"name": "search", "arguments": {"q": "x"}},
                        "meta": {
                            "semantic_plan": "Call one search tool.",
                            "validation_contract": "JSON object with name and arguments.",
                            "changed_factor": "query",
                            "diversity_signature": "fc-nested",
                        },
                    }
                ]
            }
        )

        items, rejected = pipeline.parse_generated_items(
            raw, source, profile("function_calling")
        )

        self.assertEqual(items[0].answer, '{"name":"search","arguments":{"q":"x"}}')
        self.assertEqual(rejected, [])

    def test_parse_generated_items_rejects_benchmark_wrapper_pollution(self):
        source = pipeline.SourceSample("s1", "Original?", {})
        raw = generated_json(
            "You are a very talented expert in math.\n"
            "Answer this question and finish with a single option letter.\n"
            "Question: 1+1?\nChoices:\nA. 1\nB. 2",
            "B",
            "wrapped",
        )

        items, rejected = pipeline.parse_generated_items(raw, source, profile())

        self.assertEqual(items, [])
        self.assertIn("benchmark contamination", rejected[0])

    def test_parse_generated_items_allows_clean_multiple_choice(self):
        source = pipeline.SourceSample("s1", "Original?", {})
        raw = generated_json(
            "What is 1 + 1?\nA. 1\nB. 2\nReturn only the correct option letter.",
            "B",
            "clean-mc",
        )

        items, rejected = pipeline.parse_generated_items(raw, source, profile())

        self.assertEqual(len(items), 1)
        self.assertEqual(rejected, [])

    def test_profile_routing_uses_exact_five_domain_names(self):
        config = pipeline.PromptConfig(
            default_profile=Path("unused"),
            profiles={},
            selector_keys=("task_kind",),
        )
        library = {
            "default": profile("knowledge"),
            "coding": profile("coding"),
            "function_calling": profile("function_calling"),
        }

        code_sample = pipeline.SourceSample("s1", "Fix issue", {"task_kind": "coding"})
        fc_sample = pipeline.SourceSample(
            "s2", "Call tool", {"task_kind": "function_calling"}
        )
        invalid_sample = pipeline.SourceSample(
            "s3", "Invalid metadata", {"task_kind": "not_a_domain"}
        )

        self.assertEqual(
            pipeline.choose_profile(code_sample, config, library).task_kind, "coding"
        )
        self.assertEqual(
            pipeline.choose_profile(fc_sample, config, library).task_kind,
            "function_calling",
        )
        self.assertEqual(
            pipeline.choose_profile(invalid_sample, config, library).task_kind,
            "knowledge",
        )

    def test_select_diverse_items_rejects_duplicate_signature(self):
        source = pipeline.SourceSample("s1", "Original?", {})
        items, _ = pipeline.parse_generated_items(
            json.dumps(
                {
                    "items": [
                        json.loads(generated_json("New A?", "A", "same"))["items"][0],
                        json.loads(generated_json("New B?", "B", "same"))["items"][0],
                    ]
                }
            ),
            source,
            profile(),
        )

        selected, rejected = pipeline.select_diverse_items(
            items, profile(), {"same": 1}, 2
        )

        self.assertEqual(selected, [])
        self.assertEqual(len(rejected), 2)

    def test_generate_dataset_writes_tasks_jsonl_with_fake_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "input.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "user": "Original math question?",
                        "task_kind": "math",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            prompt_text = """
name = "math"
task_kind = "math"

[synthesis]
strategy = "subspace_contract_diversity"
axes = ["domain", "skill_node", "changed_factor"]

[generation]
template = "generate {{synthesis_profile_json}} {{source_sample_json}} {{accepted_samples_json}} {{feedback_block}}"
"""
            (prompt_dir / "math.toml").write_text(prompt_text, encoding="utf-8")

            config = pipeline.PipelineConfig(
                dataset_path=None,
                db_snapshot_path=dataset,
                db_snapshot_command=(),
                limit=None,
                start_index=0,
                prompt=pipeline.PromptConfig(
                    default_profile=prompt_dir / "math.toml",
                    profiles={"math": prompt_dir / "math.toml"},
                    selector_keys=("task_kind",),
                ),
                generator=pipeline.ModelConfig(
                    endpoint="https://example.invalid/v1/chat/completions",
                    model_name="unit",
                    api_key="unit",
                    json_object_response=True,
                ),
                validator=None,
                variant_count=1,
                generation_attempts=1,
                validate_generated_questions=False,
                output_run_dir=root / "out",
            )
            client = FakeClient(generated_json("New math question?", "42", "math-sig"))

            items = pipeline.generate_dataset(config, client)

            self.assertEqual(len(items), 1)
            output = root / "out" / "generate" / "tasks.jsonl"
            self.assertTrue(output.exists())
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["task_id"], "s1_q000")
            self.assertEqual(row["expected_answer"], "42")
            self.assertIn("subspace_plan", client.prompts[0])

    def test_materialize_input_snapshot_can_capture_db_export_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.jsonl"
            row = json.dumps({"sample_id": "db1", "user": "From DB?", "task_kind": "knowledge"})
            config = pipeline.PipelineConfig(
                dataset_path=None,
                db_snapshot_path=snapshot,
                db_snapshot_command=(
                    sys.executable,
                    "-c",
                    f"import sys; sys.stdout.write({row + os.linesep!r})",
                ),
                limit=None,
                start_index=0,
                prompt=pipeline.PromptConfig(
                    default_profile=Path("unused"),
                    profiles={},
                    selector_keys=("task_kind",),
                ),
                generator=pipeline.ModelConfig(
                    endpoint="https://example.invalid/v1/chat/completions",
                    model_name="unit",
                    api_key="unit",
                ),
                validator=None,
                variant_count=1,
                generation_attempts=1,
                validate_generated_questions=False,
                output_run_dir=root / "out",
            )

            materialized = pipeline.materialize_input_snapshot(config)
            samples = pipeline.load_samples(config)

            self.assertEqual(materialized, snapshot)
            self.assertEqual(samples[0].sample_id, "db1")
            self.assertEqual(samples[0].user, "From DB?")

    def test_normalize_sample_extracts_prompt_from_db_stages_context(self):
        sample = pipeline.normalize_sample(
            0,
            {
                "sample_id": "db2",
                "context": {
                    "stages": [
                        {"prompt": "User: What is 2 + 2?\nAssistant:"},
                    ],
                },
                "task_kind": "math",
            },
        )

        self.assertEqual(sample.sample_id, "db2")
        self.assertEqual(sample.user, "What is 2 + 2?")
        self.assertEqual(sample.meta["task_kind"], "math")


if __name__ == "__main__":
    unittest.main()
