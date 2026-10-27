"""
Unit tests for Two-Level Hierarchical Critic Engine (MacroCritic, MicroCritic, RubricScorecard).
"""

import unittest

from research_assistant.tasks.critic import (
    CritiqueFlaw,
    FlawSeverity,
    MacroCritic,
    MicroCritic,
    RubricScorecard,
    TaskCritic,
    evaluate_workflow,
)
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskCategory,
    TaskGraph,
)


def _make_sample_task(
    task_id: str,
    title: str = "Test Task",
    category: TaskCategory = TaskCategory.MODEL_IMPLEMENTATION,
    assigned_role: AgentRole = AgentRole.MODEL_ARCHITECT,
    dependencies: list = None,
    inputs: list = None,
    outputs: list = None,
    acceptance_criteria: list = None,
    description: str = "Implement mathematical formulation, forward pass, and gradient checks with high fidelity.",
) -> Task:
    return Task(
        task_id=task_id,
        title=title,
        category=category,
        assigned_role=assigned_role,
        dependencies=dependencies if dependencies is not None else [],
        inputs=inputs if inputs is not None else ["input.pt"],
        outputs=outputs if outputs is not None else ["output.pt"],
        acceptance_criteria=acceptance_criteria if acceptance_criteria is not None else ["Exit code 0", "Non-empty output"],
        description=description,
    )


def _build_full_canonical_graph() -> TaskGraph:
    """Builds a verified 7-task DAG covering all 6 mandatory scientific phases."""
    g = TaskGraph(paper_id="canon_paper", paper_title="Canonical Replication Workflow")
    g.add_task(_make_sample_task(
        "task_01_env", "Provision Environment & Lock Dependencies",
        category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT,
        outputs=["env_ready.lock"],
        description="Configure Python runtime, install required packages, verify CUDA, and enforce thread limits (OMP_NUM_THREADS=1).",
    ))
    g.add_task(_make_sample_task(
        "task_02_data", "Acquire Raw Data & Split Sets",
        category=TaskCategory.DATA_PREPROCESSING, assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"], outputs=["data/train.pt", "data/test.pt"],
        description="Download raw scientific dataset, normalize features, and produce deterministic train/val/test splits with seed 42.",
    ))
    g.add_task(_make_sample_task(
        "task_03_model", "Implement Model Architecture",
        category=TaskCategory.MODEL_IMPLEMENTATION, assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"], outputs=["src/model.py"],
        description="Implement core neural network architecture, forward pass, and differentiable loss formulation in PyTorch.",
    ))
    g.add_task(_make_sample_task(
        "task_04_smoke", "Sanity Smoke-Test Forward Pass",
        category=TaskCategory.SANITY_SMOKE_TEST, assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_02_data", "task_03_model"], outputs=["logs/smoke_pass.log"],
        description="Run a single mini-batch forward and backward pass to verify gradient flow, tensor shapes, and memory usage.",
    ))
    g.add_task(_make_sample_task(
        "task_05_train", "Execute Training Loop",
        category=TaskCategory.TRAINING_OR_SIMULATION, assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_04_smoke"], outputs=["checkpoints/best_model.pt"],
        description="Train model according to paper hyperparameters while enforcing CPU thread caps (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
    ))
    g.add_task(_make_sample_task(
        "task_06_eval", "Compute Performance Metrics",
        category=TaskCategory.METRIC_EVALUATION, assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_05_train"], outputs=["results/metrics.json"],
        description="Compute test split accuracy, F1 score, and compare against baseline replication numbers reported in paper.",
    ))
    g.add_task(_make_sample_task(
        "task_07_audit", "Audit Artifacts & Compile Replication Scorecard",
        category=TaskCategory.VERIFICATION_AUDIT, assigned_role=AgentRole.AUDIT_AGENT,
        dependencies=["task_06_eval"], outputs=["scorecard.md"],
        description="Certify deliverable checksums, verify replication claims, and compile formal replication scorecard.",
    ))
    return g


class TestMacroCritic(unittest.TestCase):
    def setUp(self):
        self.critic = MacroCritic()

    def test_empty_graph_flagged_as_blocking(self):
        empty_graph = TaskGraph(paper_id="empty", paper_title="Empty Graph")
        flaws = self.critic.evaluate(empty_graph)
        self.assertTrue(any(f.severity == FlawSeverity.BLOCKING.value and f.category == "topology" for f in flaws))
        score = self.critic.compute_topology_score(empty_graph, flaws)
        self.assertEqual(score, 0.0)

    def test_cycle_detected_as_blocking(self):
        g = _build_full_canonical_graph()
        g.tasks["task_01_env"].dependencies = ["task_07_audit"]  # Create cycle
        flaws = self.critic.evaluate(g)
        cycle_flaws = [f for f in flaws if f.category == "topology" and "Cycle" in f.description]
        self.assertTrue(len(cycle_flaws) > 0)
        self.assertEqual(cycle_flaws[0].severity, FlawSeverity.BLOCKING.value)
        self.assertIn("Circular dependency", cycle_flaws[0].description)

    def test_self_dependency_detected_as_blocking(self):
        g = TaskGraph(paper_id="self_dep", paper_title="Self Dependency")
        g.add_task(_make_sample_task("t1", "Task 1", dependencies=["t1"]))
        flaws = self.critic.evaluate(g)
        self.assertTrue(any(f.category == "topology" and "cannot depend on itself" in f.description for f in flaws))

    def test_missing_dependency_detected_as_blocking(self):
        g = TaskGraph(paper_id="missing_dep", paper_title="Missing Dependency")
        g.add_task(_make_sample_task("t1", "Task 1", dependencies=["ghost_task"]))
        flaws = self.critic.evaluate(g)
        self.assertTrue(any(f.category == "topology" and "depends on missing task" in f.description for f in flaws))

    def test_missing_mandatory_phases_flagged(self):
        g = TaskGraph(paper_id="incomplete", paper_title="Incomplete Pipeline")
        # Only env and data, missing smoke, train, eval, audit
        g.add_task(_make_sample_task("task_01_env", "Env", category=TaskCategory.ENVIRONMENT_SETUP))
        g.add_task(_make_sample_task("task_02_data", "Data", category=TaskCategory.DATA_PREPROCESSING, dependencies=["task_01_env"]))

        flaws = self.critic.evaluate(g)
        phase_flaws = [f for f in flaws if f.category == "missing_phase"]
        self.assertGreaterEqual(len(phase_flaws), 4)
        for f in phase_flaws:
            self.assertEqual(f.severity, FlawSeverity.BLOCKING.value)

    def test_canonical_graph_has_zero_blocking_macro_flaws(self):
        g = _build_full_canonical_graph()
        flaws = self.critic.evaluate(g)
        blocking = [f for f in flaws if f.severity == FlawSeverity.BLOCKING.value]
        self.assertEqual(blocking, [])
        score = self.critic.compute_topology_score(g, flaws)
        self.assertEqual(score, 100.0)

    def test_serialization_warning_when_unnecessarily_sequential(self):
        g = TaskGraph(paper_id="seq", paper_title="Strictly Serialized Graph")
        g.add_task(_make_sample_task("task_01_env", "Env", category=TaskCategory.ENVIRONMENT_SETUP))
        g.add_task(_make_sample_task("task_02_data", "Data", category=TaskCategory.DATA_PREPROCESSING, dependencies=["task_01_env"]))
        g.add_task(_make_sample_task("task_03_model", "Model", category=TaskCategory.MODEL_IMPLEMENTATION, dependencies=["task_02_data"]))
        g.add_task(_make_sample_task("task_04_smoke", "Smoke", category=TaskCategory.SANITY_SMOKE_TEST, dependencies=["task_03_model"]))
        g.add_task(_make_sample_task("task_05_train", "Train", category=TaskCategory.TRAINING_OR_SIMULATION, dependencies=["task_04_smoke"]))
        g.add_task(_make_sample_task("task_06_eval", "Eval", category=TaskCategory.METRIC_EVALUATION, dependencies=["task_05_train"]))
        g.add_task(_make_sample_task("task_07_audit", "Audit", category=TaskCategory.VERIFICATION_AUDIT, dependencies=["task_06_eval"]))

        flaws = self.critic.evaluate(g)
        # All 7 tasks are strictly sequential; wave warning should be emitted
        wave_flaws = [f for f in flaws if f.category == "wave"]
        self.assertTrue(len(wave_flaws) > 0)
        self.assertEqual(wave_flaws[0].severity, FlawSeverity.WARNING.value)

    def test_cross_task_duplicate_output_flagged_blocking(self):
        g = _build_full_canonical_graph()
        # Make task_02_data also claim results/metrics.json which belongs to task_06_eval
        g.tasks["task_02_data"].outputs.append("results/metrics.json")
        flaws = self.critic.evaluate(g)
        dup_flaws = [f for f in flaws if "macro_duplicate_output" in f.flaw_id]
        self.assertTrue(len(dup_flaws) > 0)
        self.assertEqual(dup_flaws[0].severity, FlawSeverity.BLOCKING.value)
        self.assertIn("results/metrics.json", dup_flaws[0].description)

    def test_graph_complexity_bloat_flagged_warning(self):
        g = TaskGraph(paper_id="bloated", paper_title="Bloated Graph")
        for i in range(13):
            g.add_task(_make_sample_task(f"t_{i}", f"Task {i}", category=TaskCategory.DATA_PREPROCESSING))
        flaws = self.critic.evaluate(g)
        bloat_flaws = [f for f in flaws if f.flaw_id == "macro_graph_complexity_bloat"]
        self.assertTrue(len(bloat_flaws) > 0)
        self.assertEqual(bloat_flaws[0].severity, FlawSeverity.WARNING.value)


class TestMicroCritic(unittest.TestCase):
    def setUp(self):
        self.critic = MicroCritic()

    def test_empty_description_flagged_blocking(self):
        t = _make_sample_task("t1", description="")
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        desc_flaws = [f for f in flaws if f.category == "vague_instruction"]
        self.assertTrue(len(desc_flaws) > 0)
        self.assertEqual(desc_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_vague_short_description_flagged_warning(self):
        t = _make_sample_task("t1", description="Short description under 60 chars.")
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        desc_flaws = [f for f in flaws if f.category == "vague_instruction"]
        self.assertTrue(len(desc_flaws) > 0)
        self.assertEqual(desc_flaws[0].severity, FlawSeverity.WARNING.value)
        self.assertGreater(len(desc_flaws[0].suggestion), 10)

    def test_empty_outputs_flagged_contract_blocking(self):
        t = _make_sample_task("t1", outputs=[])
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        contract_flaws = [f for f in flaws if f.category == "contract"]
        self.assertTrue(len(contract_flaws) > 0)
        self.assertEqual(contract_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_output_missing_extension_flagged_warning(self):
        t = _make_sample_task("t1", outputs=["checkpoints/best_model"])
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        ext_flaws = [f for f in flaws if f.category == "contract" and "extension" in f.description]
        self.assertTrue(len(ext_flaws) > 0)
        self.assertEqual(ext_flaws[0].severity, FlawSeverity.WARNING.value)

    def test_empty_acceptance_criteria_flagged_blocking(self):
        t = _make_sample_task("t1", acceptance_criteria=[])
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        test_flaws = [f for f in flaws if f.category == "testability"]
        self.assertTrue(len(test_flaws) > 0)
        self.assertEqual(test_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_subjective_acceptance_criteria_flagged_warning(self):
        t = _make_sample_task("t1", acceptance_criteria=["Must look good and perform nicely"])
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        test_flaws = [f for f in flaws if f.category == "testability" and "subjective" in f.description]
        self.assertTrue(len(test_flaws) > 0)
        self.assertEqual(test_flaws[0].severity, FlawSeverity.WARNING.value)

    def test_unbounded_threads_flagged_cpu_oversubscription_blocking(self):
        t = _make_sample_task(
            "t1_train", "Train Model",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Train neural network using PyTorch without thread limits across all cores.",
        )
        g = TaskGraph("p1", "P1", tasks={"t1_train": t})
        flaws = self.critic.evaluate(g)
        thread_flaws = [f for f in flaws if f.category == "cpu_oversubscription"]
        self.assertTrue(len(thread_flaws) > 0)
        self.assertEqual(thread_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_explicit_safe_threads_has_zero_oversubscription_flaws(self):
        t = _make_sample_task(
            "t1_train", "Train Model",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Train model according to paper hyperparameters while enforcing CPU thread caps (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
        )
        g = TaskGraph("p1", "P1", tasks={"t1_train": t})
        flaws = self.critic.evaluate(g)
        thread_flaws = [f for f in flaws if f.category == "cpu_oversubscription"]
        self.assertEqual(thread_flaws, [])

    def test_intra_task_duplicate_output_flagged_blocking(self):
        t = _make_sample_task("t1", outputs=["data/features.pt", "data/features.pt"])
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        dup_flaws = [f for f in flaws if "micro_contract_duplicate_out" in f.flaw_id]
        self.assertTrue(len(dup_flaws) > 0)
        self.assertEqual(dup_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_intra_task_deliverable_bloat_flagged_warning(self):
        outputs = [f"output_{i}.pt" for i in range(9)]
        t = _make_sample_task("t1", outputs=outputs)
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        bloat_flaws = [f for f in flaws if "micro_contract_bloat" in f.flaw_id]
        self.assertTrue(len(bloat_flaws) > 0)
        self.assertEqual(bloat_flaws[0].severity, FlawSeverity.WARNING.value)

    def test_tensor_dim_bloat_flagged_warning(self):
        contract = ArtifactContract(
            path="embeddings.pt",
            artifact_type=ArtifactType.TENSOR,
            shape=[100, 60000],
            dtype="float32",
        )
        t = _make_sample_task("t1", outputs=["embeddings.pt"])
        t.artifact_contracts = [contract]
        g = TaskGraph("p1", "P1", tasks={"t1": t})
        flaws = self.critic.evaluate(g)
        dim_flaws = [f for f in flaws if "micro_tensor_dim_bloat" in f.flaw_id]
        self.assertTrue(len(dim_flaws) > 0)
        self.assertEqual(dim_flaws[0].severity, FlawSeverity.WARNING.value)


class TestTaskCriticAndScorecard(unittest.TestCase):
    def setUp(self):
        self.critic = TaskCritic()

    def test_canonical_graph_produces_perfect_passing_scorecard(self):
        g = _build_full_canonical_graph()
        scorecard = self.critic.evaluate(g)
        self.assertIsInstance(scorecard, RubricScorecard)
        self.assertEqual(scorecard.blocking_flaws_count, 0)
        self.assertTrue(scorecard.is_passing)
        self.assertEqual(scorecard.blocking_count, 0)
        self.assertGreaterEqual(scorecard.overall_score, 90.0)

    def test_scorecard_markdown_rendering(self):
        g = _build_full_canonical_graph()
        scorecard = self.critic.evaluate(g)
        md = scorecard.render_markdown()
        self.assertIn("Replication Workflow Rubric Scorecard", md)
        self.assertIn("DAG Topology & 6 Scientific Phases", md)
        self.assertIn("Artifact Deliverables & Contracts", md)
        self.assertIn("Deterministic Testability & CPU Bounds", md)

    def test_scorecard_serialization_roundtrip(self):
        g = _build_full_canonical_graph()
        scorecard = self.critic.evaluate(g)
        d = scorecard.to_dict()
        self.assertIn("topology_score", d)
        self.assertIn("contract_score", d)
        self.assertIn("testability_score", d)
        self.assertIn("overall_score", d)
        self.assertTrue(d["is_passing"])

    def test_evaluate_workflow_helper(self):
        g = _build_full_canonical_graph()
        scorecard = evaluate_workflow(g)
        self.assertEqual(scorecard.blocking_flaws_count, 0)
        self.assertTrue(scorecard.is_passing)


if __name__ == "__main__":
    unittest.main()
