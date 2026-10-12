"""
Adversarial Empirical Challenge Suite for Milestone 2 (M2) Iteration Engine.
Agent: teamwork_preview_challenger_m2_2

Adversarially challenges and empirically stress-tests:
1. ProposerCriticEngine.run_refinement_loop:
   - Early exit on 0 blocking flaws (Turn 1 exit).
   - Multi-turn refinement progression (Turn 1 flaws -> Turn 2/3 refinement).
   - Strict turn capping (<= 3 turns, bounds checking on max_turns).
   - Deterministic post-loop healing (cycles, self-loops, missing mandatory phases).
   - Metadata attachment (critique_trace and scorecard).
2. ProposerCriticEngine robustness under adversarial inputs:
   - Malformed/non-JSON LLM responses, empty outputs, network/Ollama exceptions.
   - None/empty/huge context payloads (>50k characters).
   - Intersecting and multi-node cycles.
3. Persistent vector storage invariant (75,039 chunks in physics_papers).
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import chromadb

from research_assistant.config import (
    COLLECTION_NAME,
    VECTORDB_PATH,
)
from research_assistant.tasks.critic import (
    CritiqueFlaw,
    FlawSeverity,
    MacroCritic,
    MicroCritic,
    RubricScorecard,
    TaskCritic,
)
from research_assistant.tasks.iteration import (
    ProposerCriticEngine,
    _build_heuristic_7phase_fallback,
)
from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)


def _build_valid_6phase_7task_graph(paper_id: str = "valid_paper") -> TaskGraph:
    """Builds a verified, fully compliant 7-task DAG covering all 6 mandatory phases."""
    g = TaskGraph(paper_id=paper_id, paper_title="Valid Replication DAG")
    g.add_task(Task(
        task_id="task_01_env",
        title="Setup Environment & Lock Dependencies",
        category=TaskCategory.ENVIRONMENT_SETUP,
        assigned_role=AgentRole.DEVOPS_AGENT,
        outputs=["env_ready.lock"],
        description="Configure Python runtime, install required computational packages, and lock thread bounds (OMP_NUM_THREADS=1).",
        acceptance_criteria=["Exit code 0", "All core libraries import cleanly without error"],
    ))
    g.add_task(Task(
        task_id="task_02_data",
        title="Data Acquisition & Splits",
        category=TaskCategory.DATA_PREPROCESSING,
        assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"],
        outputs=["data/train.pt", "data/test.pt"],
        description="Download raw dataset and partition into deterministic train, validation, and test splits with fixed seed 42.",
        acceptance_criteria=["Splits non-empty", "Zero missing values or NaN entries detected"],
    ))
    g.add_task(Task(
        task_id="task_03_model",
        title="Model Architecture & Operators",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"],
        outputs=["src/model.py", "src/loss.py"],
        description="Implement core neural architecture, forward pass tensor operations, and differentiable loss functions.",
        acceptance_criteria=["Forward pass completes with correct tensor dimensions", "Loss is differentiable and finite"],
    ))
    g.add_task(Task(
        task_id="task_04_smoke",
        title="Sanity Smoke-Test Forward Pass",
        category=TaskCategory.SANITY_SMOKE_TEST,
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_02_data", "task_03_model"],
        outputs=["logs/smoke_test.log"],
        description="Execute single forward pass and backward step on mini-batch to verify gradient flow and tensor shapes.",
        acceptance_criteria=["Forward pass completes with exit code 0", "Finite non-zero gradients verified"],
    ))
    g.add_task(Task(
        task_id="task_05_train",
        title="Model Training Loop",
        category=TaskCategory.TRAINING_OR_SIMULATION,
        assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_04_smoke"],
        outputs=["checkpoints/best_model.pt", "logs/train_metrics.csv"],
        description="Execute training loop using specified hyperparameters with CPU thread limits (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
        acceptance_criteria=["Loss converges over epochs", "Model checkpoint saved successfully"],
    ))
    g.add_task(Task(
        task_id="task_06_eval",
        title="Metric Evaluation on Test Split",
        category=TaskCategory.METRIC_EVALUATION,
        assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_05_train"],
        outputs=["metrics/evaluation_report.json"],
        description="Evaluate model on held-out test split, compute accuracy and F1 score against reported paper targets.",
        acceptance_criteria=["Primary metric computed on test set", "Comparison table generated against paper baseline"],
    ))
    g.add_task(Task(
        task_id="task_07_audit",
        title="Verification Audit & Scorecard",
        category=TaskCategory.VERIFICATION_AUDIT,
        assigned_role=AgentRole.AUDIT_AGENT,
        dependencies=["task_06_eval"],
        outputs=["results/replication_scorecard.md"],
        description="Verify output deliverable checksums and compile final replication scorecard.",
        acceptance_criteria=["Replication scorecard generated", "All artifact checksums certified non-empty"],
    ))
    return g


class TestProposerCriticEarlyExit(unittest.TestCase):
    """Empirically test early exit behavior in run_refinement_loop."""

    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_early_exit_turn_1_on_zero_blocking_flaws(self):
        """Turn 1 terminates immediately when candidate graph has 0 blocking flaws."""
        valid_graph = _build_valid_6phase_7task_graph("test_p1")
        with patch.object(self.engine, "_synthesize_turn", return_value=valid_graph):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="test_p1",
                paper_title="Test Early Exit",
                context={},
                max_turns=3,
            )
            self.assertEqual(len(scorecards), 1, "Must exit on Turn 1 when 0 blocking flaws exist.")
            self.assertEqual(scorecards[0].blocking_flaws_count, 0)
            self.assertTrue(scorecards[0].is_passing)
            self.assertIn("critique_trace", graph.metadata)
            self.assertEqual(len(graph.metadata["critique_trace"]), 1)
            self.assertEqual(graph.metadata["critique_trace"][0]["turn"], 1)

    def test_early_exit_turn_1_with_non_blocking_warnings(self):
        """Graph with non-blocking warnings (e.g. subjective phrase, vague desc) still exits on Turn 1 if blocking=0."""
        graph_with_warnings = _build_valid_6phase_7task_graph("test_warnings")
        # Add subjective criterion (warning) and slightly short description (warning)
        graph_with_warnings.tasks["task_01_env"].acceptance_criteria.append("Output looks good as expected")
        graph_with_warnings.tasks["task_02_data"].description = "Download raw dataset and partition into deterministic train splits."  # ~71 chars, safe, but let's test short desc
        graph_with_warnings.tasks["task_03_model"].description = "Implement core neural network model architecture."  # 49 chars (<60 -> warning)

        with patch.object(self.engine, "_synthesize_turn", return_value=graph_with_warnings):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="test_warnings",
                paper_title="Test Warnings",
                context={},
                max_turns=3,
            )
            self.assertEqual(len(scorecards), 1, "Must exit early despite non-blocking warnings.")
            self.assertEqual(scorecards[0].blocking_flaws_count, 0)
            self.assertGreater(scorecards[0].warning_flaws_count, 0)
            self.assertTrue(scorecards[0].is_passing)

    def test_no_early_exit_when_single_blocking_flaw_present(self):
        """Graph with even 1 blocking flaw does NOT exit on Turn 1."""
        flawed_turn1 = _build_valid_6phase_7task_graph("flawed_t1")
        # Remove outputs from task_01_env (blocking contract flaw)
        flawed_turn1.tasks["task_01_env"].outputs = []

        valid_turn2 = _build_valid_6phase_7task_graph("repaired_t2")

        turns = [flawed_turn1, valid_turn2]
        with patch.object(self.engine, "_synthesize_turn", side_effect=lambda *args, **kwargs: turns.pop(0)):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="flawed_t1",
                paper_title="Flawed T1",
                context={},
                max_turns=3,
            )
            self.assertEqual(len(scorecards), 2, "Must proceed to Turn 2 when Turn 1 had blocking flaws.")
            self.assertGreater(scorecards[0].blocking_flaws_count, 0)
            self.assertEqual(scorecards[1].blocking_flaws_count, 0)
            self.assertTrue(scorecards[1].is_passing)


class TestProposerCriticMultiTurnRefinement(unittest.TestCase):
    """Empirically test multi-turn refinement progression and prompt feedback injection."""

    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_refinement_across_3_turns_until_passing(self):
        """Simulate Turn 1 and Turn 2 failing with blocking flaws, and Turn 3 succeeding."""
        # Turn 1: missing sanity smoke test
        g1 = _build_valid_6phase_7task_graph("g1")
        del g1.tasks["task_04_smoke"]
        g1.tasks["task_05_train"].dependencies = ["task_02_data", "task_03_model"]

        # Turn 2: added smoke test but missing outputs on task_05_train
        g2 = _build_valid_6phase_7task_graph("g2")
        g2.tasks["task_05_train"].outputs = []

        # Turn 3: fully valid graph
        g3 = _build_valid_6phase_7task_graph("g3")

        turn_sequence = [g1, g2, g3]
        with patch.object(self.engine, "_synthesize_turn", side_effect=lambda *args, **kwargs: turn_sequence.pop(0)):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="three_turns",
                paper_title="Three Turns Paper",
                context={},
                max_turns=3,
            )
            self.assertEqual(len(scorecards), 3)
            self.assertFalse(scorecards[0].is_passing)
            self.assertFalse(scorecards[1].is_passing)
            self.assertTrue(scorecards[2].is_passing)
            self.assertEqual(len(graph.metadata["critique_trace"]), 3)

    def test_feedback_flaws_formatting_into_subsequent_prompts(self):
        """Assert Turn 2 receives previous Turn 1 flaws, categories, and suggestions verbatim."""
        flawed_scorecard = RubricScorecard(
            topology_score=70.0,
            contract_score=60.0,
            testability_score=50.0,
            overall_score=60.0,
            blocking_flaws_count=2,
            warning_flaws_count=1,
            flaws=[
                CritiqueFlaw(
                    flaw_id="f_cycle",
                    task_id="task_05_train",
                    critic_level="macro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="topology",
                    description="Circular dependency detected between training and eval",
                    suggestion="Remove backward edge from eval to train",
                ),
                CritiqueFlaw(
                    flaw_id="f_contract",
                    task_id="task_02_data",
                    critic_level="micro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="contract",
                    description="Task produces no output deliverables",
                    suggestion="Add outputs=['data/train.pt', 'data/test.pt']",
                ),
                CritiqueFlaw(
                    flaw_id="f_wave",
                    task_id=None,
                    critic_level="macro",
                    severity=FlawSeverity.WARNING.value,
                    category="wave",
                    description="All tasks serialized",
                    suggestion="Parallelize data and model tasks",
                ),
            ],
        )

        captured_messages = []
        def mock_chat(messages, **kwargs):
            captured_messages.extend(messages)
            return MagicMock(content='{"paper_title": "P", "tasks": []}')

        engine = ProposerCriticEngine(chat_fn=mock_chat)
        engine._synthesize_turn(
            paper_id="paper_prompt_test",
            paper_title="Prompt Inspection Test",
            context={"abstract": "Adversarial test abstract."},
            turn=2,
            previous_scorecard=flawed_scorecard,
        )

        user_content = next((m["content"] for m in captured_messages if m["role"] == "user"), "")
        self.assertIn("Circular dependency detected", user_content)
        self.assertIn("Remove backward edge from eval to train", user_content)
        self.assertIn("Task produces no output deliverables", user_content)
        self.assertIn("Add outputs=['data/train.pt', 'data/test.pt']", user_content)
        self.assertIn("🔴 BLOCKING", user_content)
        self.assertIn("🟡 WARNING", user_content)


class TestProposerCriticTurnCappingAndBounds(unittest.TestCase):
    """Empirically test strict turn capping and boundary values."""

    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_strict_cap_at_3_turns_when_flaws_persist(self):
        """When candidate graph persistently fails, loop caps at exactly 3 turns."""
        always_flawed = TaskGraph("always_flawed", "Always Flawed")
        always_flawed.add_task(Task(
            task_id="t1",
            title="Vague Step",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="too short",
            outputs=[],
            acceptance_criteria=[],
        ))

        with patch.object(self.engine, "_synthesize_turn", return_value=always_flawed):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="p_capped",
                paper_title="Capped Paper",
                context={},
                max_turns=3,
            )
            self.assertLessEqual(len(scorecards), 3)

    def test_boundary_max_turns_greater_than_3_is_capped_at_3(self):
        """Passing max_turns=10 or max_turns=100 is strictly bounded to <= 3 rounds."""
        call_count = [0]
        def counting_synthesize(*args, **kwargs):
            call_count[0] += 1
            # Return flawed graph so it would continue if not capped
            g = TaskGraph("g_flawed", "Flawed")
            g.add_task(Task(task_id="t1", title="T1", description="desc", outputs=[], acceptance_criteria=[]))
            return g

        with patch.object(self.engine, "_synthesize_turn", side_effect=counting_synthesize):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="p_max10",
                paper_title="Max 10 Paper",
                context={},
                max_turns=10,
            )
            self.assertEqual(call_count[0], 3, "Loop must execute at most 3 turns even if max_turns=10.")
            self.assertLessEqual(len(scorecards), 3)

    def test_boundary_max_turns_1(self):
        """Passing max_turns=1 executes exactly 1 turn."""
        call_count = [0]
        def counting_synthesize(*args, **kwargs):
            call_count[0] += 1
            g = TaskGraph("g_flawed", "Flawed")
            g.add_task(Task(task_id="t1", title="T1", description="desc", outputs=[], acceptance_criteria=[]))
            return g

        with patch.object(self.engine, "_synthesize_turn", side_effect=counting_synthesize):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="p_max1",
                paper_title="Max 1 Paper",
                context={},
                max_turns=1,
            )
            self.assertEqual(call_count[0], 1)
            self.assertLessEqual(len(scorecards), 1)

    def test_boundary_max_turns_zero_or_negative(self):
        """Passing max_turns=0 or -5 executes at least 1 turn without infinite loop or crash."""
        valid_graph = _build_valid_6phase_7task_graph("p_zero")
        with patch.object(self.engine, "_synthesize_turn", return_value=valid_graph):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="p_zero",
                paper_title="Zero Turns",
                context={},
                max_turns=0,
            )
            self.assertEqual(len(scorecards), 1)

            graph_neg, scorecards_neg = self.engine.run_refinement_loop(
                paper_id="p_neg",
                paper_title="Negative Turns",
                context={},
                max_turns=-5,
            )
            self.assertEqual(len(scorecards_neg), 1)


class TestProposerCriticDeterministicHealing(unittest.TestCase):
    """Empirically test post-loop deterministic healing of cycles and missing phases."""

    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_healing_severs_simple_2node_cycle(self):
        """2-node cycle (t1 -> t2 -> t1) is severed deterministically."""
        g = TaskGraph("g_2cycle", "Two Node Cycle")
        g.add_task(Task(
            task_id="t1",
            title="Task 1",
            category=TaskCategory.ENVIRONMENT_SETUP,
            dependencies=["t2"],
            outputs=["env.lock"],
            description="Environment configuration with thread limits OMP_NUM_THREADS=1.",
            acceptance_criteria=["Exit code 0"],
        ))
        g.add_task(Task(
            task_id="t2",
            title="Task 2",
            category=TaskCategory.DATA_PREPROCESSING,
            dependencies=["t1"],
            outputs=["data.pt"],
            description="Data acquisition step with reproducible random seeds.",
            acceptance_criteria=["Exit code 0"],
        ))
        self.assertGreater(len(g.validate()), 0)

        scorecard = self.engine.critic.evaluate(g)
        healed = self.engine._heal_remaining_flaws(g, scorecard, {})
        self.assertEqual(healed.validate(), [], "Validation errors must be zero after cycle healing.")

    def test_healing_severs_3node_cycle(self):
        """3-node cycle (A -> B -> C -> A) is severed deterministically."""
        g = TaskGraph("g_3cycle", "Three Node Cycle")
        g.add_task(Task(
            task_id="A",
            title="Step A",
            category=TaskCategory.ENVIRONMENT_SETUP,
            dependencies=["B"],
            outputs=["a.lock"],
            description="Setup environment with thread caps.",
            acceptance_criteria=["Exit code 0"],
        ))
        g.add_task(Task(
            task_id="B",
            title="Step B",
            category=TaskCategory.DATA_PREPROCESSING,
            dependencies=["C"],
            outputs=["b.pt"],
            description="Data preprocessing.",
            acceptance_criteria=["Exit code 0"],
        ))
        g.add_task(Task(
            task_id="C",
            title="Step C",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            dependencies=["A"],
            outputs=["c.py"],
            description="Model implementation.",
            acceptance_criteria=["Exit code 0"],
        ))
        self.assertGreater(len(g.validate()), 0)

        scorecard = self.engine.critic.evaluate(g)
        healed = self.engine._heal_remaining_flaws(g, scorecard, {})
        self.assertEqual(healed.validate(), [])

    def test_healing_severs_self_loop(self):
        """Task depending on itself (self-loop: t1 -> t1) is severed deterministically."""
        g = TaskGraph("g_self", "Self Loop")
        g.add_task(Task(
            task_id="t1",
            title="Self Dependent Task",
            category=TaskCategory.ENVIRONMENT_SETUP,
            dependencies=["t1"],
            outputs=["self.lock"],
            description="Self dependent task environment configuration.",
            acceptance_criteria=["Exit code 0"],
        ))
        self.assertGreater(len(g.validate()), 0)

        scorecard = self.engine.critic.evaluate(g)
        healed = self.engine._heal_remaining_flaws(g, scorecard, {})
        self.assertEqual(healed.validate(), [])
        self.assertEqual(g.tasks["t1"].dependencies, [])

    def test_healing_fills_missing_mandatory_phases(self):
        """Graph missing 4 mandatory phases has all missing phases injected with valid dependencies."""
        g = TaskGraph("g_missing", "Missing Phases")
        # Only environment and data tasks provided
        g.add_task(Task(
            task_id="task_01_env",
            title="Setup Environment",
            category=TaskCategory.ENVIRONMENT_SETUP,
            assigned_role=AgentRole.DEVOPS_AGENT,
            outputs=["env.lock"],
            description="Configure runtime environment.",
            acceptance_criteria=["Exit code 0"],
        ))
        g.add_task(Task(
            task_id="task_02_data",
            title="Data Preprocessing",
            category=TaskCategory.DATA_PREPROCESSING,
            assigned_role=AgentRole.DATA_ENGINEER,
            dependencies=["task_01_env"],
            outputs=["data.pt"],
            description="Preprocess dataset.",
            acceptance_criteria=["Exit code 0"],
        ))

        self.assertFalse(g.has_all_mandatory_phases())
        missing_before = g.get_missing_phases()
        self.assertIn("sanity_smoke_test", missing_before)
        self.assertIn("training_or_simulation", missing_before)
        self.assertIn("metric_evaluation", missing_before)
        self.assertIn("verification_audit", missing_before)

        scorecard = self.engine.critic.evaluate(g)
        healed = self.engine._heal_remaining_flaws(g, scorecard, {})

        self.assertTrue(healed.has_all_mandatory_phases())
        self.assertEqual(healed.get_missing_phases(), [])
        self.assertEqual(healed.validate(), [])

    def test_healing_records_deterministic_healing_trace_entry(self):
        """When healing is applied post-loop, critique_trace logs an action='deterministic_healing' entry."""
        flawed_g = TaskGraph("flawed_trace", "Flawed Trace")
        flawed_g.add_task(Task(
            task_id="t1",
            title="Setup",
            category=TaskCategory.ENVIRONMENT_SETUP,
            dependencies=["t1"],  # self-loop
            outputs=["env.lock"],
            description="Configure environment with thread limits OMP_NUM_THREADS=1.",
            acceptance_criteria=["Exit code 0"],
        ))

        with patch.object(self.engine, "_synthesize_turn", return_value=flawed_g):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="flawed_trace",
                paper_title="Flawed Trace Paper",
                context={},
                max_turns=1,
            )
            # Critique trace must have the healing entry
            actions = [entry.get("action") for entry in graph.metadata["critique_trace"]]
            self.assertIn("deterministic_healing", actions)


class TestProposerCriticMetadataAttachment(unittest.TestCase):
    """Empirically verify critique_trace and scorecard metadata attached to output TaskGraph."""

    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_metadata_fields_and_schema(self):
        """TaskGraph metadata must contain critique_trace and scorecard dicts with valid numeric scores."""
        valid_graph = _build_valid_6phase_7task_graph("p_meta")
        with patch.object(self.engine, "_synthesize_turn", return_value=valid_graph):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="p_meta",
                paper_title="Metadata Paper",
                context={},
                max_turns=3,
            )

            # 1. critique_trace verification
            self.assertIn("critique_trace", graph.metadata)
            trace = graph.metadata["critique_trace"]
            self.assertIsInstance(trace, list)
            self.assertEqual(len(trace), 1)

            entry = trace[0]
            required_trace_keys = {
                "turn",
                "topology_score",
                "contract_score",
                "testability_score",
                "overall_score",
                "blocking_flaws_count",
                "warning_flaws_count",
                "flaws",
            }
            self.assertTrue(required_trace_keys.issubset(set(entry.keys())))
            self.assertEqual(entry["blocking_flaws_count"], 0)
            self.assertGreaterEqual(entry["overall_score"], 80.0)

            # 2. scorecard verification
            self.assertIn("scorecard", graph.metadata)
            sc = graph.metadata["scorecard"]
            self.assertIsInstance(sc, dict)
            required_sc_keys = {
                "topology_score",
                "contract_score",
                "testability_score",
                "overall_score",
                "blocking_flaws_count",
                "warning_flaws_count",
                "is_passing",
                "flaws",
            }
            self.assertTrue(required_sc_keys.issubset(set(sc.keys())))
            self.assertTrue(sc["is_passing"])


class TestProposerCriticRobustnessAndFallbacks(unittest.TestCase):
    """Empirically test parser robustness, extreme contexts, and exception handling."""

    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_parse_llm_json_markdown_code_fences(self):
        """JSON enclosed in ```json ... ``` markdown code fences is parsed cleanly."""
        raw = """```json
        {
            "paper_title": "Parsed Paper",
            "tasks": [
                {
                    "task_id": "task_01_env",
                    "title": "Env Setup",
                    "category": "environment_setup",
                    "assigned_role": "devops_agent",
                    "dependencies": [],
                    "description": "Environment configuration with OMP_NUM_THREADS=1.",
                    "inputs": ["requirements.txt"],
                    "outputs": ["env.lock"],
                    "acceptance_criteria": ["Exit code 0"]
                }
            ]
        }
        ```"""
        graph = self.engine._parse_llm_json(raw, "pid", "Parsed Paper")
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.tasks), 1)
        self.assertEqual(graph.tasks["task_01_env"].category, TaskCategory.ENVIRONMENT_SETUP)
        self.assertEqual(graph.tasks["task_01_env"].assigned_role, AgentRole.DEVOPS_AGENT)

    def test_parse_llm_json_malformed_returns_none(self):
        """Malformed non-JSON strings return None without raising exceptions."""
        self.assertIsNone(self.engine._parse_llm_json("Not JSON text", "pid", "title"))
        self.assertIsNone(self.engine._parse_llm_json("", "pid", "title"))
        self.assertIsNone(self.engine._parse_llm_json('{"tasks": "not a list"}', "pid", "title"))
        self.assertIsNone(self.engine._parse_llm_json('{"tasks": []}', "pid", "title"))

    def test_offline_fallback_on_network_or_ollama_exception(self):
        """When chat_fn raises exception, synthesis falls back cleanly to heuristic 7-phase DAG."""
        def broken_chat(*args, **kwargs):
            raise ConnectionError("Ollama service unavailable")

        engine = ProposerCriticEngine(chat_fn=broken_chat, full_phases=True)
        graph, scorecards = engine.run_refinement_loop(
            paper_id="offline_p",
            paper_title="Offline Paper",
            context={},
            max_turns=3,
        )
        self.assertIsInstance(graph, TaskGraph)
        self.assertEqual(len(graph.tasks), 7)
        self.assertTrue(graph.has_all_mandatory_phases())
        self.assertEqual(graph.validate(), [])
        self.assertGreaterEqual(scorecards[0].overall_score, 90.0)

    def test_format_context_text_handles_none_and_empty_safely(self):
        """Context with None or empty fields formats without raising TypeError."""
        empty_ctx = {}
        self.assertIsInstance(self.engine._format_context_text(empty_ctx), str)

        none_ctx = {"abstract": None, "sections": None, "tables": None}
        self.assertIsInstance(self.engine._format_context_text(none_ctx), str)

    def test_format_context_text_handles_huge_50k_payload(self):
        """Context exceeding 50,000 characters formats completely without crashing."""
        huge_ctx = {
            "formatted_context": "# Faceted Context\n" + ("Scientific parameters: alpha=0.01, beta=0.9.\n" * 1500),
        }
        text = self.engine._format_context_text(huge_ctx)
        self.assertGreater(len(text), 50000)

    def test_build_heuristic_7phase_fallback_quality(self):
        """Canonical 7-task heuristic fallback satisfies all 6 phases across 6 waves with score >= 90."""
        graph = _build_heuristic_7phase_fallback("h7", "Heuristic Paper")
        self.assertEqual(len(graph.tasks), 7)
        self.assertEqual(graph.validate(), [])
        self.assertTrue(graph.has_all_mandatory_phases())
        self.assertEqual(graph.wave_count(), 6)
        self.assertTrue(graph.is_parallelizable())

        critic = TaskCritic()
        scorecard = critic.evaluate(graph)
        self.assertEqual(scorecard.blocking_flaws_count, 0)
        self.assertTrue(scorecard.is_passing)
        self.assertGreaterEqual(scorecard.overall_score, 90.0)


class TestPersistentStorageChunkCountInvariant(unittest.TestCase):
    """Empirically verify ChromaDB persistent storage chunk count is intact."""

    def test_chromadb_physics_papers_chunk_count_equals_75039(self):
        """Assert persistent collection 'physics_papers' in data/physics_vectordb has exactly 75,039 chunks."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        self.assertEqual(
            count,
            75039,
            f"Persistent storage count violated! Expected 75039, got {count}",
        )


if __name__ == "__main__":
    unittest.main()
