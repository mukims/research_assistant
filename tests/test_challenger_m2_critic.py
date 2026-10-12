"""
Adversarial Stress Test Suite for Milestone 2 Critic Engine (MacroCritic, MicroCritic, RubricScorecard).
Authored by Empirical Challenger (teamwork_preview_challenger_m2_1).
"""

import unittest
from unittest.mock import MagicMock, patch

from research_assistant.tasks.critic import (
    CritiqueFlaw,
    FlawSeverity,
    MacroCritic,
    MicroCritic,
    RubricScorecard,
    SUBJECTIVE_WORDS,
    TaskCritic,
    evaluate_workflow,
)
from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
)


def _make_canonical_task(
    task_id: str,
    title: str = "Test Task",
    category: TaskCategory = TaskCategory.MODEL_IMPLEMENTATION,
    assigned_role: AgentRole = AgentRole.MODEL_ARCHITECT,
    dependencies: list = None,
    inputs: list = None,
    outputs: list = None,
    acceptance_criteria: list = None,
    description: str = "Implement mathematical formulation, forward pass, and gradient checks with high fidelity.",
    artifact_contracts: list = None,
) -> Task:
    outs = outputs if outputs is not None else ["output.pt"]
    contracts = artifact_contracts if artifact_contracts is not None else []
    return Task(
        task_id=task_id,
        title=title,
        category=category,
        assigned_role=assigned_role,
        dependencies=dependencies if dependencies is not None else [],
        inputs=inputs if inputs is not None else ["input.pt"],
        outputs=outs,
        acceptance_criteria=acceptance_criteria if acceptance_criteria is not None else ["Exit code 0", "Non-empty output"],
        description=description,
        artifact_contracts=contracts,
    )


def _build_canonical_7phase_graph() -> TaskGraph:
    """Builds a verified 7-task DAG covering all 6 mandatory scientific phases across multiple waves."""
    g = TaskGraph(paper_id="canon_paper", paper_title="Canonical Replication Workflow")
    g.add_task(_make_canonical_task(
        "task_01_env", "Provision Environment & Lock Dependencies",
        category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT,
        outputs=["env_ready.lock"],
        description="Configure Python runtime, install required packages, verify CUDA, and enforce thread limits (OMP_NUM_THREADS=1).",
    ))
    g.add_task(_make_canonical_task(
        "task_02_data", "Acquire Raw Data & Split Sets",
        category=TaskCategory.DATA_PREPROCESSING, assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"], outputs=["data/train.pt", "data/test.pt"],
        description="Download raw scientific dataset, normalize features, and produce deterministic train/val/test splits with seed 42.",
    ))
    g.add_task(_make_canonical_task(
        "task_03_model", "Implement Model Architecture",
        category=TaskCategory.MODEL_IMPLEMENTATION, assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"], outputs=["src/model.py"],
        description="Implement core neural network architecture, forward pass, and differentiable loss formulation in PyTorch.",
    ))
    g.add_task(_make_canonical_task(
        "task_04_smoke", "Sanity Smoke-Test Forward Pass",
        category=TaskCategory.SANITY_SMOKE_TEST, assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_02_data", "task_03_model"], outputs=["logs/smoke_pass.log"],
        description="Run a single mini-batch forward and backward pass to verify gradient flow, tensor shapes, and memory usage.",
    ))
    g.add_task(_make_canonical_task(
        "task_05_train", "Execute Training Loop",
        category=TaskCategory.TRAINING_OR_SIMULATION, assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_04_smoke"], outputs=["checkpoints/best_model.pt"],
        description="Train model according to paper hyperparameters while enforcing CPU thread caps (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
    ))
    g.add_task(_make_canonical_task(
        "task_06_eval", "Compute Performance Metrics",
        category=TaskCategory.METRIC_EVALUATION, assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_05_train"], outputs=["results/metrics.json"],
        description="Compute test split accuracy, F1 score, and compare against baseline replication numbers reported in paper.",
    ))
    g.add_task(_make_canonical_task(
        "task_07_audit", "Audit Artifacts & Compile Replication Scorecard",
        category=TaskCategory.VERIFICATION_AUDIT, assigned_role=AgentRole.AUDIT_AGENT,
        dependencies=["task_06_eval"], outputs=["scorecard.md"],
        description="Certify deliverable checksums, verify replication claims, and compile formal replication scorecard.",
    ))
    return g


class TestChallengerEmptyAndBoundaryGraphs(unittest.TestCase):
    """Stress-tests boundary graph inputs (None, empty, single-task)."""

    def setUp(self):
        self.critic = TaskCritic()

    def test_none_graph_evaluation_safety(self):
        """Pass None to all critic entry points; assert zero unhandled exceptions."""
        mc = MacroCritic()
        uc = MicroCritic()
        macro_flaws = mc.evaluate(None)
        micro_flaws = uc.evaluate(None)
        scorecard = self.critic.evaluate(None)
        helper_sc = evaluate_workflow(None)

        self.assertEqual(len(macro_flaws), 1)
        self.assertEqual(macro_flaws[0].category, "topology")
        self.assertEqual(macro_flaws[0].severity, FlawSeverity.BLOCKING.value)
        self.assertEqual(len(micro_flaws), 0)
        self.assertFalse(scorecard.is_passing)
        self.assertEqual(scorecard.overall_score, 0.0)
        self.assertEqual(scorecard.blocking_flaws_count, 1)
        self.assertEqual(helper_sc.overall_score, 0.0)

    def test_empty_graph_evaluation_zero_scores(self):
        """Graph with 0 tasks should receive exactly 0 across all rubrics and fail."""
        g = TaskGraph(paper_id="empty", paper_title="Empty Tasks", tasks={})
        sc = self.critic.evaluate(g)
        self.assertEqual(sc.topology_score, 0.0)
        self.assertEqual(sc.contract_score, 0.0)
        self.assertEqual(sc.testability_score, 0.0)
        self.assertEqual(sc.overall_score, 0.0)
        self.assertFalse(sc.is_passing)
        self.assertEqual(sc.blocking_flaws_count, 1)
        self.assertEqual(sc.flaws[0].flaw_id, "macro_empty_graph")

    def test_single_task_graph_missing_five_phases(self):
        """Single valid task covers 1 phase but misses 5 mandatory phases."""
        from research_assistant.tasks.models import ArtifactContract, ArtifactType

        g = TaskGraph(paper_id="single", paper_title="Single Task")
        g.add_task(
            _make_canonical_task(
                "t1",
                "Only Env",
                category=TaskCategory.ENVIRONMENT_SETUP,
                artifact_contracts=[
                    ArtifactContract(path="output.pt", artifact_type=ArtifactType.TENSOR, shape=[32, 128], dtype="float32")
                ],
            )
        )
        sc = self.critic.evaluate(g)

        # 5 missing phases = 5 * 15 = 75 deduction -> topology_score = 25.0
        self.assertEqual(sc.topology_score, 25.0)
        self.assertEqual(sc.contract_score, 100.0)
        self.assertEqual(sc.testability_score, 100.0)
        self.assertEqual(sc.overall_score, 75.0)
        self.assertFalse(sc.is_passing)
        self.assertEqual(sc.blocking_flaws_count, 5)

        phase_flaws = [f for f in sc.flaws if f.category == "missing_phase"]
        self.assertEqual(len(phase_flaws), 5)


class TestChallengerCyclicAndMissingDependencies(unittest.TestCase):
    """Stress-tests cyclic dependency chains, self-loops, and ghost tasks."""

    def setUp(self):
        self.critic = TaskCritic()

    def test_direct_self_loop(self):
        """Self-referential dependency t1 -> t1."""
        g = _build_canonical_7phase_graph()
        g.tasks["task_01_env"].dependencies = ["task_01_env"]
        sc = self.critic.evaluate(g)
        self.assertFalse(sc.is_passing)
        self.assertTrue(any("cannot depend on itself" in f.description for f in sc.flaws))
        self.assertLess(sc.topology_score, 100.0)

    def test_two_task_cycle(self):
        """Mutual dependency t1 -> t2 and t2 -> t1."""
        g = _build_canonical_7phase_graph()
        g.tasks["task_01_env"].dependencies = ["task_02_data"]
        # task_02_data already depends on task_01_env
        sc = self.critic.evaluate(g)
        self.assertFalse(sc.is_passing)
        cycle_flaws = [f for f in sc.flaws if "Cycle" in f.description]
        self.assertGreaterEqual(len(cycle_flaws), 1)
        self.assertEqual(cycle_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_long_cyclic_chain_50_tasks(self):
        """Massive cyclic dependency chain across 50 tasks (t0 -> t1 -> ... -> t49 -> t0)."""
        g = TaskGraph("cycle_50", "50 Task Cycle")
        n = 50
        for i in range(n):
            next_i = (i + 1) % n
            g.add_task(_make_canonical_task(
                f"t_{i}", f"Task {i}",
                dependencies=[f"t_{next_i}"],
                description=f"Task {i} description exceeding sixty characters for testing purposes.",
            ))
        sc = self.critic.evaluate(g)
        self.assertFalse(sc.is_passing)
        self.assertTrue(any("Cycle" in f.description for f in sc.flaws))
        self.assertEqual(sc.topology_score, 0.0)

    def test_disjoint_components_with_multiple_cycles(self):
        """Graph with independent components containing distinct cycles."""
        g = TaskGraph("disjoint_cycles", "Disjoint Cycles")
        # Component 1 cycle: c1 <-> c2
        g.add_task(_make_canonical_task("c1", "C1", dependencies=["c2"]))
        g.add_task(_make_canonical_task("c2", "C2", dependencies=["c1"]))
        # Component 2 cycle: d1 <-> d2
        g.add_task(_make_canonical_task("d1", "D1", dependencies=["d2"]))
        g.add_task(_make_canonical_task("d2", "D2", dependencies=["d1"]))
        # Component 3 self loop: e1 -> e1
        g.add_task(_make_canonical_task("e1", "E1", dependencies=["e1"]))

        sc = self.critic.evaluate(g)
        self.assertFalse(sc.is_passing)
        topo_flaws = [f for f in sc.flaws if f.category == "topology"]
        self.assertGreaterEqual(len(topo_flaws), 3)

    def test_ghost_missing_dependency(self):
        """Task depends on a non-existent task_id."""
        g = _build_canonical_7phase_graph()
        g.tasks["task_02_data"].dependencies = ["phantom_ghost_task_99"]
        sc = self.critic.evaluate(g)
        self.assertFalse(sc.is_passing)
        missing_dep_flaws = [f for f in sc.flaws if "depends on missing task" in f.description]
        self.assertEqual(len(missing_dep_flaws), 1)
        self.assertEqual(missing_dep_flaws[0].severity, FlawSeverity.BLOCKING.value)


class TestChallengerScientificPhasesCoverage(unittest.TestCase):
    """Systematically stress-tests each of the 6 mandatory scientific phases."""

    def setUp(self):
        self.macro_critic = MacroCritic()

    def _test_single_phase_omission(self, task_to_remove: str, expected_phase_name: str):
        g = _build_canonical_7phase_graph()
        del g.tasks[task_to_remove]
        # Rewire dependent tasks so no topology errors occur
        for t in g.tasks.values():
            if task_to_remove in t.dependencies:
                t.dependencies = [d for d in t.dependencies if d != task_to_remove]

        flaws = self.macro_critic.evaluate(g)
        phase_flaws = [f for f in flaws if f.category == "missing_phase"]
        self.assertEqual(len(phase_flaws), 1, f"Expected 1 missing phase flaw for {expected_phase_name}")
        self.assertEqual(phase_flaws[0].flaw_id, f"macro_phase_{expected_phase_name}")
        self.assertEqual(phase_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_missing_environment_setup_phase(self):
        self._test_single_phase_omission("task_01_env", "environment_setup")

    def test_missing_data_splits_phase(self):
        self._test_single_phase_omission("task_02_data", "data_splits")

    def test_missing_sanity_smoke_test_phase(self):
        self._test_single_phase_omission("task_04_smoke", "sanity_smoke_test")

    def test_missing_training_or_simulation_phase(self):
        self._test_single_phase_omission("task_05_train", "training_or_simulation")

    def test_missing_metric_evaluation_phase(self):
        self._test_single_phase_omission("task_06_eval", "metric_evaluation")

    def test_missing_verification_audit_phase(self):
        self._test_single_phase_omission("task_07_audit", "verification_audit")

    def test_all_phases_missing_simultaneously(self):
        """Tasks with non-standard category names missing all 6 mandatory phases."""
        g = TaskGraph("non_standard", "Non Standard Categories")
        g.add_task(_make_canonical_task("t1", "Custom 1", category=TaskCategory.MODEL_IMPLEMENTATION))
        g.add_task(_make_canonical_task("t2", "Custom 2", category=TaskCategory.MODEL_IMPLEMENTATION, dependencies=["t1"]))

        flaws = self.macro_critic.evaluate(g)
        phase_flaws = [f for f in flaws if f.category == "missing_phase"]
        # Missing env, data, smoke, train, eval, audit = 6 phases
        self.assertEqual(len(phase_flaws), 6)


class TestChallengerMicroCriticBoundaries(unittest.TestCase):
    """Stress-tests micro-critic boundaries: descriptions, criteria, extensions, thread bounds."""

    def setUp(self):
        self.critic = MicroCritic()

    def test_description_length_boundary_59_vs_60_chars(self):
        """Description with 59 chars emits warning; 60 chars emits no flaw."""
        t_59 = _make_canonical_task("t59", description="A" * 59)
        t_60 = _make_canonical_task("t60", description="A" * 60)
        g = TaskGraph("g", "G", tasks={"t59": t_59, "t60": t_60})

        flaws = self.critic.evaluate(g)
        f_59 = [f for f in flaws if f.task_id == "t59" and f.category == "vague_instruction"]
        f_60 = [f for f in flaws if f.task_id == "t60" and f.category == "vague_instruction"]

        self.assertEqual(len(f_59), 1)
        self.assertEqual(f_59[0].severity, FlawSeverity.WARNING.value)
        self.assertEqual(len(f_60), 0)

    def test_all_twelve_subjective_words_detected(self):
        """Every subjective word in SUBJECTIVE_WORDS triggers a testability warning."""
        for word in SUBJECTIVE_WORDS:
            t = _make_canonical_task("t_subj", acceptance_criteria=[f"Output must {word}"])
            g = TaskGraph("g", "G", tasks={"t_subj": t})
            flaws = self.critic.evaluate(g)
            subj_flaws = [f for f in flaws if f.category == "testability" and "subjective" in f.description]
            self.assertEqual(len(subj_flaws), 1, f"Failed to detect subjective word '{word}'")
            self.assertEqual(subj_flaws[0].severity, FlawSeverity.WARNING.value)

    def test_output_file_extensions_missing_warning(self):
        """Outputs lacking file extensions emit warning for deterministic verification."""
        t = _make_canonical_task("t_ext", outputs=["dir/data_artifact", "models/best_checkpoint"])
        g = TaskGraph("g", "G", tasks={"t_ext": t})
        flaws = self.critic.evaluate(g)
        ext_flaws = [f for f in flaws if f.category == "contract" and "extension" in f.description]
        self.assertEqual(len(ext_flaws), 2)
        for ef in ext_flaws:
            self.assertEqual(ef.severity, FlawSeverity.WARNING.value)

    def test_cpu_thread_oversubscription_sixteen_threads_blocking(self):
        """Training task with 'num_thread: 16' triggers blocking cpu_oversubscription flaw."""
        t = _make_canonical_task(
            "t_train",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Train neural network using PyTorch with num_thread: 16 for speed.",
        )
        g = TaskGraph("g", "G", tasks={"t_train": t})
        flaws = self.critic.evaluate(g)
        thread_flaws = [f for f in flaws if f.category == "cpu_oversubscription"]
        self.assertEqual(len(thread_flaws), 1)
        self.assertEqual(thread_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_cpu_thread_oversubscription_unbounded_phrases(self):
        """Training task with 'unbounded thread' or 'all cores' triggers blocking flaw."""
        for phrase in ["without thread limits", "unbounded thread", "all cores", "maximum threads"]:
            t = _make_canonical_task(
                "t_train",
                category=TaskCategory.TRAINING_OR_SIMULATION,
                description=f"Train model using PyTorch {phrase} to finish quickly.",
            )
            g = TaskGraph("g", "G", tasks={"t_train": t})
            flaws = self.critic.evaluate(g)
            thread_flaws = [f for f in flaws if f.category == "cpu_oversubscription"]
            self.assertEqual(len(thread_flaws), 1, f"Failed to flag '{phrase}'")
            self.assertEqual(thread_flaws[0].severity, FlawSeverity.BLOCKING.value)

    def test_cpu_thread_safe_indicators_zero_flaws(self):
        """Training task specifying CPU thread limits triggers 0 thread flaws."""
        for indicator in ["torch.set_num_threads(1)", "OMP_NUM_THREADS=1", "cpu-safe thread cap"]:
            t = _make_canonical_task(
                "t_train",
                category=TaskCategory.TRAINING_OR_SIMULATION,
                description=f"Train neural network enforcing {indicator} to avoid thread starvation.",
            )
            g = TaskGraph("g", "G", tasks={"t_train": t})
            flaws = self.critic.evaluate(g)
            thread_flaws = [f for f in flaws if f.category == "cpu_oversubscription"]
            self.assertEqual(len(thread_flaws), 0, f"False positive flaw on safe indicator '{indicator}'")


class TestChallengerAdversarialVulnerabilityAnalysis(unittest.TestCase):
    """
    Adversarial tests that probe edge cases and unmask potential blind spots in critic.py.
    """

    def test_extreme_threads_thirty_two_blind_spot(self):
        """
        EMPIRICAL CHALLENGE: Probes whether 'num_thread: 32' is caught.
        In critic.py line 143, UNSAFE_THREAD_PATTERNS only hardcodes '16',
        while 'thread' in SAFE_THREAD_INDICATORS falsely suppresses the warning.
        This test documents the empirical vulnerability.
        """
        critic = MicroCritic()
        t = _make_canonical_task(
            "t_train_32",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Train neural network model using num_thread: 32 across nodes.",
        )
        g = TaskGraph("g", "G", tasks={"t_train_32": t})
        flaws = critic.evaluate(g)
        thread_flaws = [f for f in flaws if f.category == "cpu_oversubscription"]
        # Empirical finding: Currently passes with 0 flaws due to pattern gap
        has_thread_blind_spot = (len(thread_flaws) == 0)
        self.assertTrue(has_thread_blind_spot, "Verified: num_thread: 32 currently evades critic.py")

    def test_none_attribute_defensive_safety(self):
        """
        EMPIRICAL CHALLENGE: Tests behavior when optional Task fields are explicitly None.
        Documents where TypeErrors / AttributeErrors can occur if inputs are unvalidated.
        """
        critic = MicroCritic()
        # Test None description
        t_none_desc = Task(
            task_id="t_none",
            title="None Description Task",
            description=None,
            outputs=["out.pt"],
            acceptance_criteria=["Exit 0"],
        )
        g = TaskGraph("g", "G", tasks={"t_none": t_none_desc})
        # MicroCritic safely defaults (task.description or "").strip() -> flags empty desc
        flaws = critic.evaluate(g)
        desc_flaws = [f for f in flaws if f.category == "vague_instruction"]
        self.assertEqual(len(desc_flaws), 1)
        self.assertEqual(desc_flaws[0].severity, FlawSeverity.BLOCKING.value)


class TestChallengerDeterminismAndScoreBounding(unittest.TestCase):
    """Verifies that RubricScorecards are strictly bounded [0, 100] and 100% deterministic."""

    def test_scores_strictly_bounded_in_zero_to_hundred(self):
        """Under extreme combinations of flaws, scores must never fall below 0 or exceed 100."""
        # Extreme worst case: 10 cycles, 0 phases, empty outputs, empty criteria, unbounded threads
        g = TaskGraph("worst", "Worst Case Graph")
        for i in range(10):
            g.add_task(Task(
                task_id=f"t_{i}",
                title=f"Task {i}",
                category=TaskCategory.TRAINING_OR_SIMULATION,
                dependencies=[f"t_{i}"],  # self cycle
                description="Train PyTorch without thread limits all cores maximum threads num_thread: 16",
                outputs=[],
                acceptance_criteria=[],
            ))
        sc = evaluate_workflow(g)
        self.assertGreaterEqual(sc.topology_score, 0.0)
        self.assertLessEqual(sc.topology_score, 100.0)
        self.assertGreaterEqual(sc.contract_score, 0.0)
        self.assertLessEqual(sc.contract_score, 100.0)
        self.assertGreaterEqual(sc.testability_score, 0.0)
        self.assertLessEqual(sc.testability_score, 100.0)
        self.assertGreaterEqual(sc.overall_score, 0.0)
        self.assertLessEqual(sc.overall_score, 100.0)
        self.assertEqual(sc.topology_score, 0.0)
        self.assertEqual(sc.contract_score, 0.0)
        self.assertEqual(sc.testability_score, 0.0)
        self.assertEqual(sc.overall_score, 0.0)

    def test_strict_determinism_across_one_hundred_runs(self):
        """Evaluating the same workflow 100 times produces identical scores and flaws."""
        g = _build_canonical_7phase_graph()
        initial_sc = evaluate_workflow(g)
        for _ in range(100):
            sc = evaluate_workflow(g)
            self.assertEqual(sc.topology_score, initial_sc.topology_score)
            self.assertEqual(sc.contract_score, initial_sc.contract_score)
            self.assertEqual(sc.testability_score, initial_sc.testability_score)
            self.assertEqual(sc.overall_score, initial_sc.overall_score)
            self.assertEqual(sc.blocking_flaws_count, initial_sc.blocking_flaws_count)
            self.assertEqual(sc.warning_flaws_count, initial_sc.warning_flaws_count)
            self.assertEqual(len(sc.flaws), len(initial_sc.flaws))


if __name__ == "__main__":
    unittest.main()
