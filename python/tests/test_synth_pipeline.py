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
        answer_style=pipeline.default_answer_style(task_kind),
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


def generated_json(
    user=(
        "In a biology lab, a cell organelle has a double membrane and produces "
        "most ATP through oxidative phosphorylation. Which organelle is it?"
    ),
    answer="mitochondrion",
    signature="sig-a",
):
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
                        "answer_style": pipeline.infer_answer_style(
                            "knowledge", user, answer
                        ),
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
    def test_build_chat_payload_includes_reasoning_controls(self):
        model = pipeline.ModelConfig(
            endpoint="https://example.invalid/v1/chat/completions",
            model_name="unit",
            api_key="unit",
            max_completion_tokens=128,
            temperature=0.2,
            reasoning_effort="medium",
            thinking={"type": "enabled"},
            enable_thinking=True,
            json_object_response=True,
        )

        payload = pipeline.build_chat_payload(model, "Prompt")

        self.assertEqual(payload["model"], "unit")
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertIs(payload["enable_thinking"], True)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_parse_chat_response_can_merge_reasoning_content(self):
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "Compute 17 * 23 = 391.",
                            "content": "391",
                        }
                    }
                ]
            }
        )

        merged = pipeline.parse_chat_response_text(body, merge_reasoning_content=True)
        visible_only = pipeline.parse_chat_response_text(
            body, merge_reasoning_content=False
        )

        self.assertEqual(
            merged,
            "<think>\nCompute 17 * 23 = 391.\n</think>\n\n391",
        )
        self.assertEqual(visible_only, "391")

    def test_parse_chat_response_can_return_reasoning_only_when_merge_enabled(self):
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "Reason only.",
                            "content": "",
                        }
                    }
                ]
            }
        )

        self.assertEqual(
            pipeline.parse_chat_response_text(body, merge_reasoning_content=True),
            "<think>\nReason only.\n</think>",
        )
        with self.assertRaisesRegex(RuntimeError, "content is empty"):
            pipeline.parse_chat_response_text(body, merge_reasoning_content=False)

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

    def test_build_generation_prompt_preserves_source_cot_style(self):
        source = pipeline.SourceSample(
            sample_id="s1",
            user="Original math question with visible reasoning format?",
            meta={"domain": "math", "cot_mode": "CoT"},
        )

        prompt_text = pipeline.build_generation_prompt(
            profile("math"), source, 1, [], ""
        )

        self.assertIn('"answer_style": "cot"', prompt_text)
        self.assertIn('"profile_default_answer_style": "final_only"', prompt_text)

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
                            "answer_style": "function_call",
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
                        json.loads(
                            generated_json(
                                "In a plant cell, which organelle contains chlorophyll and carries out photosynthesis?",
                                "chloroplast",
                                "same",
                            )
                        )["items"][0],
                        json.loads(
                            generated_json(
                                "In a neuron, which cellular extension usually carries impulses away from the cell body?",
                                "axon",
                                "same",
                            )
                        )["items"][0],
                    ]
                }
            ),
            source,
            profile(),
        )

        selected, rejected = pipeline.select_diverse_items(
            items, profile(), {"same": 1}, {}, 0, 2
        )

        self.assertEqual(selected, [])
        self.assertEqual(len(rejected), 2)

    def test_cluster_key_groups_numeric_variants(self):
        first = pipeline.cluster_key_for_text(
            "knowledge",
            "The carrier swing is 150 kHz. Find the percentage modulation.\nA. 50%\nB. 100%",
            {},
        )
        second = pipeline.cluster_key_for_text(
            "knowledge",
            "The carrier swing is 100 kHz. Find the percentage modulation.\nA. 50%\nB. 100%",
            {},
        )

        self.assertEqual(first, second)

    def test_filter_samples_by_source_cluster_limits_duplicate_sources(self):
        samples = [
            pipeline.SourceSample(
                "s1",
                "The carrier swing is 150 kHz. Find the percentage modulation.",
                {"task_kind": "knowledge"},
            ),
            pipeline.SourceSample(
                "s2",
                "The carrier swing is 100 kHz. Find the percentage modulation.",
                {"task_kind": "knowledge"},
            ),
            pipeline.SourceSample(
                "s3",
                "Which organelle produces ATP through aerobic respiration?",
                {"task_kind": "knowledge"},
            ),
        ]
        config = pipeline.PipelineConfig(
            dataset_path=None,
            db_snapshot_path=None,
            db_snapshot_command=(),
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
            output_run_dir=Path("unused"),
            source_cluster_limit_per_domain=1,
        )

        kept = pipeline.filter_samples_by_source_cluster(samples, config)

        self.assertEqual([sample.sample_id for sample in kept], ["s1", "s3"])

    def test_select_diverse_items_rejects_duplicate_cluster_over_limit(self):
        source = pipeline.SourceSample("s1", "Original?", {})
        raw = json.dumps(
            {
                "items": [
                    {
                        "user": "What is the capital city of Canada?\nA. Toronto\nB. Ottawa",
                        "answer": "B",
                        "meta": {
                            "semantic_plan": "capital city",
                            "validation_contract": "single option",
                            "changed_factor": "country",
                            "cluster_key": "knowledge:capital_city",
                            "diversity_signature": "capital-canada",
                            "answer_style": "final_only",
                        },
                    }
                ]
            }
        )
        items, rejected = pipeline.parse_generated_items(raw, source, profile())

        selected, diversity_rejected = pipeline.select_diverse_items(
            items,
            profile(),
            {},
            {"knowledge:capital_city": 2},
            2,
            1,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(selected, [])
        self.assertIn("duplicate cluster over limit", diversity_rejected[0])

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
            client = FakeClient(
                generated_json(
                    "A sequence has first term 3 and common difference 4. What is the 10th term?",
                    "39",
                    "math-sig",
                )
            )

            items = pipeline.generate_dataset(config, client)

            self.assertEqual(len(items), 1)
            output = root / "out" / "generate" / "tasks.jsonl"
            self.assertTrue(output.exists())
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["task_id"], "s1_q000")
            self.assertEqual(row["expected_answer"], "39")
            self.assertIn("subspace_plan", client.prompts[0])

    def test_generate_dataset_resume_skips_completed_source(self):
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
            output_dir = root / "out"
            generate_dir = output_dir / "generate"
            generate_dir.mkdir(parents=True)
            existing = pipeline.GeneratedItem(
                task_id="s1_q000",
                user=(
                    "A sequence has first term 3 and common difference 4. "
                    "What is the 10th term?"
                ),
                answer="39",
                meta={
                    "source_sample_id": "s1",
                    "task_kind": "math",
                    "answer_style": "final_only",
                    "semantic_plan": "sequence term",
                    "validation_contract": "answer is exact",
                    "changed_factor": "index",
                    "diversity_signature": "math-sig",
                },
            )
            pipeline.write_jsonl(
                generate_dir / "tasks.jsonl",
                [
                    pipeline.item_to_row(
                        existing,
                        pipeline.ModelConfig(
                            endpoint="https://example.invalid/v1/chat/completions",
                            model_name="unit",
                            api_key="unit",
                        ),
                    )
                ],
            )

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
                output_run_dir=output_dir,
                resume=True,
            )
            client = FakeClient(generated_json(signature="new-sig"))

            items = pipeline.generate_dataset(config, client)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].task_id, "s1_q000")
            self.assertEqual(client.prompts, [])
            self.assertEqual(
                len((generate_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()),
                1,
            )

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

    def test_normalize_sample_strips_source_benchmark_math_wrapper(self):
        sample = pipeline.normalize_sample(
            0,
            {
                "sample_id": "wrapped_math",
                "context": {
                    "stages": [
                        {
                            "prompt": (
                                "User: Solve the problem using one clean solution path. "
                                "Keep each step concise. Put the complete final answer "
                                "inside \\(\\boxed{...}\\). Do not restart, repeat, "
                                "enumerate alternative methods. (think)\n"
                                "Let x + 7 = 12. Find x.\nAssistant:"
                            )
                        }
                    ]
                },
                "task_kind": "math",
            },
        )

        self.assertEqual(sample.user, "Let x + 7 = 12. Find x.")
        self.assertNotIn("Solve the problem using one clean solution path", sample.user)

    def test_normalize_sample_strips_source_assistant_prefill(self):
        sample = pipeline.normalize_sample(
            0,
            {
                "sample_id": "fc1",
                "context": {
                    "stages": [
                        {
                            "prompt": (
                                'System: Tools:\n[{"name":"search","arguments":{}}]\n'
                                "User: Search for train times.\nAssistant: ```json"
                            )
                        }
                    ]
                },
                "task_kind": "function_calling",
            },
        )

        self.assertEqual(
            sample.user,
            'System: Tools:\n[{"name":"search","arguments":{}}]\nUser: Search for train times.',
        )
        self.assertNotIn("Assistant:", sample.user)

    def test_normalize_sample_preserves_db_cot_format_fields(self):
        sample = pipeline.normalize_sample(
            0,
            {
                "sample_id": "db3",
                "user": "What is 12 + 30?",
                "task_kind": "math",
                "cot_mode": "CoT",
                "sampling_config": {"cot_mode": "CoT"},
                "benchmark_name": "unit_math",
            },
        )

        self.assertEqual(sample.meta["cot_mode"], "CoT")
        self.assertEqual(sample.meta["sampling_config"]["cot_mode"], "CoT")
        self.assertEqual(sample.meta["benchmark_name"], "unit_math")
        self.assertEqual(pipeline.expected_answer_style(sample, profile("math")), "cot")
        self.assertEqual(
            pipeline.expected_answer_style(
                pipeline.SourceSample(
                    "s",
                    "Original?",
                    {"answer_style": "final_only", "cot_mode": "CoT"},
                ),
                profile("math"),
            ),
            "cot",
        )
        self.assertEqual(
            pipeline.expected_answer_style(
                pipeline.SourceSample("s", "Original?", {"cot_mode": "NoCoT"}),
                profile("math"),
            ),
            "final_only",
        )

    def test_parse_generated_items_rejects_final_only_answer_style_drift(self):
        source = pipeline.SourceSample(
            "s1",
            "The sum of two numbers is 10 and one number is 6. What is the other?",
            {"task_kind": "math", "ref_answer": "4"},
        )
        raw = generated_json(
            "An arithmetic sequence has 5th term 17 and 12th term 45. What is the sum of its first 20 terms?",
            "The common difference is 4, so the final sum is 780.",
            "math-drift",
        )

        items, rejected = pipeline.parse_generated_items(raw, source, profile("math"))

        self.assertEqual(items, [])
        self.assertIn("answer_style drift", rejected[0])

    def test_parse_generated_items_requires_cot_when_source_is_cot(self):
        source = pipeline.SourceSample(
            "s1",
            "An original arithmetic benchmark asks for visible reasoning before the final answer.",
            {"task_kind": "math", "cot_mode": "CoT"},
        )
        raw = generated_json(
            "A sequence starts at 4 and increases by 3 each step. What is the 9th term?",
            "28",
            "missing-cot",
        )

        items, rejected = pipeline.parse_generated_items(raw, source, profile("math"))

        self.assertEqual(items, [])
        self.assertIn("answer_style drift: expected cot", rejected[0])

    def test_parse_generated_items_accepts_cot_when_source_is_cot(self):
        source = pipeline.SourceSample(
            "s1",
            "An original arithmetic benchmark asks for visible reasoning before the final answer.",
            {"task_kind": "math", "cot_mode": "CoT"},
        )
        raw = generated_json(
            "A sequence starts at 4 and increases by 3 each step. What is the 9th term?",
            "<think>The 9th term is 4 + 8 * 3 = 28.</think>\n28",
            "with-cot",
        )

        items, rejected = pipeline.parse_generated_items(raw, source, profile("math"))

        self.assertEqual(rejected, [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].meta["answer_style"], "cot")

    def test_parse_generated_items_rejects_shallow_knowledge_prompt(self):
        source = pipeline.SourceSample("s1", "Original?", {})
        raw = generated_json("What is the powerhouse of the cell?", "mitochondria", "shallow")

        items, rejected = pipeline.parse_generated_items(raw, source, profile("knowledge"))

        self.assertEqual(items, [])
        self.assertIn("too shallow", rejected[0])

    def test_parse_generated_items_rejects_unknown_function_tool(self):
        source = pipeline.SourceSample("s1", "Call a tool?", {})
        raw = json.dumps(
            {
                "items": [
                    {
                        "user": (
                            "System: Tools:\n"
                            '[{"name":"search","arguments":{"q":{"type":"string"}}}]\n'
                            "User: Search for cats."
                        ),
                        "answer": {"name": "lookup", "arguments": {"q": "cats"}},
                        "meta": {
                            "semantic_plan": "Call one tool.",
                            "validation_contract": "Tool name must be in schema.",
                            "changed_factor": "tool",
                            "diversity_signature": "bad-tool",
                            "answer_style": "function_call",
                        },
                    }
                ]
            }
        )

        items, rejected = pipeline.parse_generated_items(
            raw, source, profile("function_calling")
        )

        self.assertEqual(items, [])
        self.assertIn("unknown tool", rejected[0])

    def test_parse_generated_items_rejects_instruction_contradiction(self):
        source = pipeline.SourceSample("s1", "Original?", {})
        raw = json.dumps(
            {
                "items": [
                    {
                        "user": (
                            'Jawab dalam JSON. Alasan mesti dimulakan dengan "Pada pendapat saya," '
                            "dan tidak boleh mengandungi koma (,)."
                        ),
                        "answer": {"jawapan": "D", "alasan": "Pada pendapat saya contoh"},
                        "meta": {
                            "semantic_plan": "Contradictory instruction.",
                            "validation_contract": "No comma conflict.",
                            "changed_factor": "format",
                            "diversity_signature": "comma-conflict",
                            "answer_style": "json_object",
                        },
                    }
                ]
            }
        )

        items, rejected = pipeline.parse_generated_items(
            raw, source, profile("instruction_following")
        )

        self.assertEqual(items, [])
        self.assertIn("contradictory comma", rejected[0])


if __name__ == "__main__":
    unittest.main()
