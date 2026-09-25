"""
Unit tests for the Task and TaskGraph data structures, topological sorting, and visualizers.
"""

import os
import tempfile
import unittest

from research_assistant.tasks.graph import (
    export_antigravity_specs,
    load_task_graph,
    render_mermaid,
    render_markdown_plan,
    save_task_graph,
)
from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)


class TestTaskGraphModels(unittest.TestCase):
    """Tests core Task and TaskGraph mechanics."""

    def setUp(self):
        self.graph = TaskGraph(
            paper_id="2108.10114v3",
            paper_title="Topological Quantum Wires Under Disorder",
        )

        # Build a standard 5-task diamond DAG:
        # T1 (env) -> T2 (data) -> T4 (train) -> T5 (eval)
        #         \-> T3 (model) /
        self.t1 = Task(
            task_id="t1_env",
            title="Setup Environment",
            category=TaskCategory.ENVIRONMENT_SETUP,
            description="Create virtualenv and install dependencies.",
            assigned_role=AgentRole.DEVOPS_AGENT,
            dependencies=[],
            inputs=["requirements.txt"],
            outputs=["venv_ready.lock"],
            acceptance_criteria=["Python 3.10 verified", "PyTorch with CUDA available"],
        )
        self.t2 = Task(
            task_id="t2_data",
            title="Data Preprocessing",
            category=TaskCategory.DATA_PREPROCESSING,
            description="Process raw wire lattice coordinates.",
            assigned_role=AgentRole.DATA_ENGINEER,
            dependencies=["t1_env"],
            inputs=["raw_lattice.csv"],
            outputs=["processed_lattice.h5"],
            acceptance_criteria=["Output contains 50,000 samples"],
        )
        self.t3 = Task(
            task_id="t3_model",
            title="Model Architecture",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            description="Implement tight-binding Hamiltonian operator.",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            dependencies=["t1_env"],
            inputs=["processed_lattice.h5"],
            outputs=["model.py"],
            acceptance_criteria=["Hermitian symmetry test passes"],
        )
        self.t4 = Task(
            task_id="t4_train",
            title="Simulate Majorana Transport",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Run disorder transport sweep.",
            assigned_role=AgentRole.EXPERIMENT_RUNNER,
            dependencies=["t2_data", "t3_model"],
            inputs=["processed_lattice.h5", "model.py"],
            outputs=["conductance_curve.npy"],
            acceptance_criteria=["Disorder parameter W from 0.0 to 2.0 completed"],
        )
        self.t5 = Task(
            task_id="t5_eval",
            title="Verify Quantized Conductance",
            category=TaskCategory.METRIC_EVALUATION,
            description="Compute zero-bias conductance peak and compare against Fig. 3.",
            assigned_role=AgentRole.EVALUATION_AGENT,
            dependencies=["t4_train"],
            inputs=["conductance_curve.npy"],
            outputs=["comparison_table.json"],
            acceptance_criteria=["Peak matches 2e^2/h within 1%"],
        )

        for t in [self.t1, self.t2, self.t3, self.t4, self.t5]:
            self.graph.add_task(t)

    def test_validation_clean_graph(self):
        errors = self.graph.validate()
        self.assertEqual(errors, [])

    def test_validation_catches_missing_dependency(self):
        bad_task = Task(
            task_id="t_bad",
            title="Bad Task",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            description="Missing parent",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            dependencies=["nonexistent_task"],
        )
        self.graph.add_task(bad_task)
        errors = self.graph.validate()
        self.assertTrue(any("nonexistent_task" in e for e in errors))

    def test_validation_catches_self_dependency(self):
        bad_task = Task(
            task_id="t_self",
            title="Self Loop",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            description="Loops on itself",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            dependencies=["t_self"],
        )
        self.graph.add_task(bad_task)
        errors = self.graph.validate()
        self.assertTrue(any("cannot depend on itself" in e for e in errors))

    def test_validation_catches_cycle(self):
        # Create cycle: T1 -> T2 -> T4 -> T1
        self.graph.tasks["t1_env"].dependencies = ["t4_train"]
        errors = self.graph.validate()
        self.assertTrue(any("Cycle detected" in e for e in errors))

    def test_topological_waves(self):
        waves = self.graph.topological_waves()
        self.assertEqual(len(waves), 4)

        # Wave 0: Root (t1_env)
        self.assertEqual([t.task_id for t in waves[0]], ["t1_env"])
        # Wave 1: Parallel branches (t2_data, t3_model)
        wave1_ids = {t.task_id for t in waves[1]}
        self.assertEqual(wave1_ids, {"t2_data", "t3_model"})
        # Wave 2: Join (t4_train)
        self.assertEqual([t.task_id for t in waves[2]], ["t4_train"])
        # Wave 3: Final evaluation (t5_eval)
        self.assertEqual([t.task_id for t in waves[3]], ["t5_eval"])

    def test_ready_tasks_progression(self):
        # Initially only t1_env has all dependencies completed (it has none)
        ready = self.graph.ready_tasks()
        self.assertEqual([t.task_id for t in ready], ["t1_env"])

        # Complete t1_env
        self.graph.tasks["t1_env"].status = TaskStatus.COMPLETED

        # Now both t2_data and t3_model are ready
        ready_ids = {t.task_id for t in self.graph.ready_tasks()}
        self.assertEqual(ready_ids, {"t2_data", "t3_model"})

        # Complete only t2_data -> t4_train is NOT ready yet (still waiting on t3_model)
        self.graph.tasks["t2_data"].status = TaskStatus.COMPLETED
        self.assertEqual([t.task_id for t in self.graph.ready_tasks()], ["t3_model"])

        # Complete t3_model -> now t4_train is ready!
        self.graph.tasks["t3_model"].status = TaskStatus.COMPLETED
        self.assertEqual([t.task_id for t in self.graph.ready_tasks()], ["t4_train"])

    def test_serialization_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name

        try:
            save_task_graph(self.graph, path)
            loaded = load_task_graph(path)
            self.assertEqual(loaded.paper_id, self.graph.paper_id)
            self.assertEqual(loaded.paper_title, self.graph.paper_title)
            self.assertEqual(len(loaded.tasks), len(self.graph.tasks))
            self.assertEqual(loaded.tasks["t4_train"].category, TaskCategory.TRAINING_OR_SIMULATION)
            self.assertEqual(loaded.tasks["t4_train"].assigned_role, AgentRole.EXPERIMENT_RUNNER)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_render_mermaid(self):
        mermaid = render_mermaid(self.graph)
        self.assertIn("flowchart TD", mermaid)
        self.assertIn('t1_env["🛠️ Setup Environment"]', mermaid)
        self.assertIn("t1_env --> t2_data", mermaid)
        self.assertIn("t1_env --> t3_model", mermaid)
        self.assertIn("t2_data --> t4_train", mermaid)
        self.assertIn("t3_model --> t4_train", mermaid)
        self.assertIn("t4_train --> t5_eval", mermaid)

    def test_render_mermaid_sanitizes_special_characters(self):
        g = TaskGraph(paper_id="special", paper_title="Special Paper")
        t1 = Task(task_id="task-01:env (setup)", title="Setup Env", category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT)
        t2 = Task(task_id="task-02/data", title="Prep Data", category=TaskCategory.DATA_PREPROCESSING, assigned_role=AgentRole.DATA_ENGINEER, dependencies=["task-01:env (setup)"])
        g.add_task(t1)
        g.add_task(t2)
        mermaid = render_mermaid(g)
        self.assertIn("task_01_env__setup_", mermaid)
        self.assertIn("task_02_data", mermaid)
        self.assertIn("task_01_env__setup_ --> task_02_data", mermaid)

    def test_render_mermaid_leading_digits_and_bracketed_titles(self):
        g = TaskGraph(paper_id="digits_test", paper_title="Digits Paper")
        t1 = Task(task_id="1_setup", title="Setup [PyTorch] & GPU", category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT)
        t2 = Task(task_id="2_model", title="Model [Transformer] Arch\nSecond Line", category=TaskCategory.MODEL_IMPLEMENTATION, assigned_role=AgentRole.MODEL_ARCHITECT, dependencies=["1_setup"])
        g.add_task(t1)
        g.add_task(t2)
        mermaid = render_mermaid(g)
        self.assertIn("n_1_setup", mermaid)
        self.assertIn("n_2_model", mermaid)
        self.assertIn("n_1_setup --> n_2_model", mermaid)
        self.assertIn("(PyTorch)", mermaid)
        self.assertIn("(Transformer)", mermaid)
        self.assertNotIn("[PyTorch]", mermaid)

    def test_render_markdown_plan(self):
        md = render_markdown_plan(self.graph)
        self.assertIn("# Research Execution Plan: Topological Quantum Wires Under Disorder", md)
        self.assertIn("| Wave 1 | `t1_env` |", md)
        self.assertIn("### `t4_train`: Simulate Majorana Transport", md)
        self.assertIn("- [ ] Peak matches 2e^2/h within 1%", md)

    def test_export_antigravity_specs(self):
        specs = export_antigravity_specs(self.graph)
        self.assertEqual(len(specs), 5)
        t1_spec = next(s for s in specs if s["Metadata"]["task_id"] == "t1_env")
        self.assertEqual(t1_spec["Role"], "Devops Agent")
        self.assertIn("Mission: Setup Environment", t1_spec["Prompt"])
        self.assertIn("Python 3.10 verified", t1_spec["Prompt"])


if __name__ == "__main__":
    unittest.main()
