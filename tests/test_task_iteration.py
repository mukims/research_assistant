"""
Unit tests for ProposerCriticEngine multi-turn iteration loop.
"""

import unittest
from unittest.mock import MagicMock, patch

from research_assistant.tasks.critic import (
    CritiqueFlaw,
    FlawSeverity,
    RubricScorecard,
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
)


def _build_valid_7phase_graph(paper_id: str = "valid_paper") -> TaskGraph:
    """Builds a verified 7-task DAG covering all 6 mandatory scientific phases."""
    g = TaskGraph(paper_id=paper_id, paper_title="Valid Scientific Replication Workflow")
    g.add_task(Task(
        task_id="task_01_env", title="Setup Environment",
        category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT,
        outputs=["env_ready.lock"],
        description="Configure Python runtime, install required packages, and enforce thread caps (OMP_NUM_THREADS=1).",
        acceptance_criteria=["Exit code 0", "All libraries import cleanly"],
    ))
    g.add_task(Task(
        task_id="task_02_data", title="Data Acquisition & Splits",
        category=TaskCategory.DATA_PREPROCESSING, assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"], outputs=["data/train.pt", "data/test.pt"],
        description="Download dataset and create deterministic train/val/test splits with seed 42.",
        acceptance_criteria=["Splits non-empty", "Zero NaN values"],
    ))
    g.add_task(Task(
        task_id="task_03_model", title="Model Architecture",
        category=TaskCategory.MODEL_IMPLEMENTATION, assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"], outputs=["src/model.py"],
        description="Implement neural network architecture, forward pass, and differentiable loss.",
        acceptance_criteria=["Forward pass completes", "Loss differentiable"],
    ))
    g.add_task(Task(
        task_id="task_04_smoke", title="Sanity Smoke-Test",
        category=TaskCategory.SANITY_SMOKE_TEST, assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_02_data", "task_03_model"], outputs=["logs/smoke.log"],
        description="Execute single mini-batch forward pass to verify gradient flow and tensor shapes.",
        acceptance_criteria=["Forward pass exits with code 0", "Non-zero finite gradients"],
    ))
    g.add_task(Task(
        task_id="task_05_train", title="Model Training",
        category=TaskCategory.TRAINING_OR_SIMULATION, assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_04_smoke"], outputs=["checkpoints/best.pt"],
        description="Train model according to paper hyperparameters with thread bounds (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
        acceptance_criteria=["Loss converges", "Model checkpoint saved"],
    ))
    g.add_task(Task(
        task_id="task_06_eval", title="Metric Evaluation",
        category=TaskCategory.METRIC_EVALUATION, assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_05_train"], outputs=["results/eval.json"],
        description="Evaluate model on test split and compute accuracy, F1 score against reported numbers.",
        acceptance_criteria=["Accuracy >= 0.85", "Comparison table generated"],
    ))
    g.add_task(Task(
        task_id="task_07_audit", title="Verification Audit",
        category=TaskCategory.VERIFICATION_AUDIT, assigned_role=AgentRole.AUDIT_AGENT,
        dependencies=["task_06_eval"], outputs=["scorecard.md"],
        description="Audit output deliverable checksums and compile replication scorecard.",
        acceptance_criteria=["Scorecard generated", "Deliverables verified"],
    ))
    return g


class TestProposerCriticEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ProposerCriticEngine()

    def test_early_exit_on_turn_1_when_zero_blocking_flaws(self):
        valid_graph = _build_valid_7phase_graph()
        with patch.object(self.engine, "_synthesize_turn", return_value=valid_graph):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="test_early_exit",
                paper_title="Test Early Exit",
                context={},
                max_turns=3,
            )
            self.assertEqual(len(scorecards), 1)
            self.assertEqual(scorecards[0].blocking_flaws_count, 0)
            self.assertTrue(scorecards[0].is_passing)
            self.assertIn("critique_trace", graph.metadata)
            self.assertEqual(len(graph.metadata["critique_trace"]), 1)

    def test_refinement_across_turns_until_passing(self):
        # Turn 1: graph missing smoke test (1 blocking flaw)
        flawed_graph = _build_valid_7phase_graph("flawed")
        del flawed_graph.tasks["task_04_smoke"]
        flawed_graph.tasks["task_05_train"].dependencies = ["task_02_data", "task_03_model"]

        # Turn 2: complete valid graph
        repaired_graph = _build_valid_7phase_graph("repaired")

        synthesized_returns = [flawed_graph, repaired_graph]

        def fake_synthesize(*args, **kwargs):
            return synthesized_returns.pop(0)

        with patch.object(self.engine, "_synthesize_turn", side_effect=fake_synthesize):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="test_refine",
                paper_title="Test Refine",
                context={},
                max_turns=3,
            )
            # Should take 2 turns and exit early on turn 2
            self.assertEqual(len(scorecards), 2)
            self.assertGreater(scorecards[0].blocking_flaws_count, 0)
            self.assertEqual(scorecards[1].blocking_flaws_count, 0)
            self.assertTrue(scorecards[1].is_passing)
            self.assertEqual(len(graph.metadata["critique_trace"]), 2)

    def test_strict_cap_at_max_turns(self):
        # Graph that always has blocking flaws
        flawed_graph = TaskGraph("always_flawed", "Always Flawed")
        flawed_graph.add_task(Task(
            task_id="t1", title="Vague", category=TaskCategory.TRAINING_OR_SIMULATION,
            description="short", outputs=[], acceptance_criteria=[],
        ))

        with patch.object(self.engine, "_synthesize_turn", return_value=flawed_graph):
            graph, scorecards = self.engine.run_refinement_loop(
                paper_id="p1", paper_title="P1", context={}, max_turns=3,
            )
            self.assertLessEqual(len(scorecards), 3)

    def test_feedback_injected_into_prompt_for_turn_2(self):
        # Verify that previous scorecard flaws are formatted into messages on turn >= 2
        scorecard = RubricScorecard(
            topology_score=80.0, contract_score=80.0, testability_score=80.0, overall_score=80.0,
            blocking_flaws_count=1, warning_flaws_count=1,
            flaws=[
                CritiqueFlaw(
                    flaw_id="f1", task_id="t1", critic_level="macro",
                    severity=FlawSeverity.BLOCKING.value, category="missing_phase",
                    description="Missing sanity smoke test phase",
                    suggestion="Add task_04_smoke for forward pass verification",
                ),
                CritiqueFlaw(
                    flaw_id="f2", task_id="t2", critic_level="micro",
                    severity=FlawSeverity.WARNING.value, category="vague_instruction",
                    description="Short description", suggestion="Expand description",
                ),
            ],
        )

        recorded_messages = []

        def mock_chat_fn(messages, **kwargs):
            recorded_messages.extend(messages)
            return MagicMock(content='{"paper_title": "P", "tasks": []}')

        engine = ProposerCriticEngine(chat_fn=mock_chat_fn)
        engine._synthesize_turn(
            paper_id="paper_feed",
            paper_title="Paper Feedback Test",
            context={"abstract": "Test abstract"},
            turn=2,
            previous_scorecard=scorecard,
        )

        user_content = next((m["content"] for m in recorded_messages if m["role"] == "user"), "")
        self.assertIn("Missing sanity smoke test phase", user_content)
        self.assertIn("Add task_04_smoke", user_content)
        self.assertIn("BLOCKING", user_content)

    def test_offline_fallback_on_llm_exception(self):
        def failing_chat(*args, **kwargs):
            raise ConnectionError("Ollama daemon unreachable")

        engine = ProposerCriticEngine(chat_fn=failing_chat, full_phases=True)
        graph = engine._synthesize_turn(
            paper_id="offline_paper",
            paper_title="Offline Paper",
            context={"abstract": "Offline test"},
            turn=1,
        )

        self.assertIsInstance(graph, TaskGraph)
        self.assertEqual(len(graph.tasks), 7)
        self.assertTrue(graph.has_all_mandatory_phases())
        self.assertEqual(graph.validate(), [])

    def test_deterministic_cycle_severing_healing(self):
        # Create a graph with cycle: t1 -> t2 -> t1
        cyclic_graph = TaskGraph("cycle_heal", "Cycle Heal")
        cyclic_graph.add_task(Task(
            task_id="t1", title="Task 1", category=TaskCategory.ENVIRONMENT_SETUP,
            dependencies=["t2"], outputs=["env.lock"], description="Environment setup step.",
            acceptance_criteria=["Exit code 0"],
        ))
        cyclic_graph.add_task(Task(
            task_id="t2", title="Task 2", category=TaskCategory.DATA_PREPROCESSING,
            dependencies=["t1"], outputs=["data.pt"], description="Data acquisition step.",
            acceptance_criteria=["Exit code 0"],
        ))
        self.assertTrue(len(cyclic_graph.validate()) > 0)

        # Apply healing
        scorecard = self.engine.critic.evaluate(cyclic_graph)
        healed = self.engine._heal_remaining_flaws(cyclic_graph, scorecard, {})
        self.assertEqual(healed.validate(), [])

    def test_heuristic_7phase_fallback_builder(self):
        graph = _build_heuristic_7phase_fallback("h7", "Heuristic 7 Phase")
        self.assertEqual(len(graph.tasks), 7)
        self.assertEqual(graph.validate(), [])
        self.assertTrue(graph.has_all_mandatory_phases())
        waves = graph.topological_waves()
        self.assertEqual(len(waves), 6)
        scorecard = self.engine.critic.evaluate(graph)
        self.assertEqual(scorecard.blocking_flaws_count, 0)
        self.assertTrue(scorecard.is_passing)
        self.assertGreaterEqual(scorecard.overall_score, 90.0)


if __name__ == "__main__":
    unittest.main()
