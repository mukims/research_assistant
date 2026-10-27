"""
Unit tests for tasks Streamlit UI component.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.tasks.ui import (
    _list_candidate_papers,
    _render_mermaid_interactive,
)


class TestTaskUI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_list_candidate_papers(self):
        pulled = os.path.join(self.tmp_dir, "pulled_pdfs")
        os.makedirs(pulled, exist_ok=True)
        with open(os.path.join(pulled, "paper1.pdf"), "w") as f:
            f.write("dummy pdf")

        raw = os.path.join(self.tmp_dir, "raw")
        os.makedirs(raw, exist_ok=True)
        with open(os.path.join(raw, "paper2.pdf"), "w") as f:
            f.write("dummy raw pdf")

        audit_tei = os.path.join(self.tmp_dir, "audit", "tei")
        os.makedirs(audit_tei, exist_ok=True)
        with open(os.path.join(audit_tei, "paper3.tei.xml"), "w") as f:
            f.write("<TEI/>")

        from research_assistant import config
        with patch.object(config, "DATA_DIR", self.tmp_dir), \
             patch.object(config, "PULLED_PDFS_DIR", pulled), \
             patch.object(config, "RAW_DIR", raw):
            candidates = _list_candidate_papers()
            self.assertEqual(len(candidates), 3)
            basenames = [os.path.basename(c) for c in candidates]
            self.assertIn("paper1.pdf", basenames)
            self.assertIn("paper2.pdf", basenames)
            self.assertIn("paper3.tei.xml", basenames)

    @patch("streamlit.components.v1.html")
    def test_render_mermaid_interactive(self, mock_html):
        code = "flowchart TD\n    A --> B"
        _render_mermaid_interactive(code, height=300)
        mock_html.assert_called_once()
        args, kwargs = mock_html.call_args
        self.assertIn("mermaid.min.js", args[0])
        self.assertIn(code, args[0])
        self.assertEqual(kwargs["height"], 300)

    def test_render_planner_tab_select_flow(self):
        import streamlit as st
        from research_assistant.tasks.ui import render_planner_tab

        pulled = os.path.join(self.tmp_dir, "pulled_pdfs")
        os.makedirs(pulled, exist_ok=True)
        with open(os.path.join(pulled, "sample_corpus_paper.pdf"), "wb") as f:
            f.write(b"%PDF-1.4\n")

        from research_assistant import config
        with patch.object(config, "DATA_DIR", self.tmp_dir), \
             patch.object(config, "PULLED_PDFS_DIR", pulled), \
             patch.object(config, "RAW_DIR", os.path.join(self.tmp_dir, "raw")):

            st.session_state.clear()
            with patch("streamlit.radio", return_value="Select Ingested Paper"), \
                 patch("streamlit.selectbox", return_value=0), \
                 patch("streamlit.checkbox", return_value=True), \
                 patch("streamlit.button", return_value=True), \
                 patch("streamlit.spinner"), \
                 patch("streamlit.success"):

                render_planner_tab()
                self.assertIn("planner_active_graph", st.session_state)
                graph = st.session_state["planner_active_graph"]
                self.assertIsNotNone(graph)
                self.assertGreaterEqual(len(graph.tasks), 5)
                self.assertEqual(graph.paper_id, "sample_corpus_paper")

    def test_render_planner_tab_upload_flow(self):
        import streamlit as st
        from research_assistant.tasks.ui import render_planner_tab

        from research_assistant import config
        with patch.object(config, "DATA_DIR", self.tmp_dir):
            st.session_state.clear()
            mock_upload = MagicMock()
            mock_upload.name = "../../paper_upload [2024].txt"
            mock_upload.getvalue.return_value = b"Method: Experimental setup and evaluation."

            with patch("streamlit.radio", return_value="Upload New Paper / TEI XML / Text"), \
                 patch("streamlit.file_uploader", return_value=mock_upload), \
                 patch("streamlit.checkbox", return_value=True), \
                 patch("streamlit.button", return_value=True), \
                 patch("streamlit.spinner"), \
                 patch("streamlit.success"):

                render_planner_tab()
                self.assertIn("planner_active_graph", st.session_state)
                graph = st.session_state["planner_active_graph"]
                self.assertIsNotNone(graph)
                self.assertGreaterEqual(len(graph.tasks), 5)
                self.assertNotIn("..", graph.paper_id)
                self.assertIn("paper_upload", graph.paper_id)

    def test_render_planner_tab_duplicate_acceptance_criteria_keys(self):
        import streamlit as st
        from research_assistant.tasks.models import AgentRole, Task, TaskCategory, TaskGraph
        from research_assistant.tasks.ui import render_planner_tab

        graph = TaskGraph(paper_id="dup_test", paper_title="Duplicate Criteria Paper")
        task = Task(
            task_id="task_dup",
            title="Task with Duplicate Criteria",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            assigned_role=AgentRole.MODEL_ARCHITECT,
            acceptance_criteria=["Criterion A", "Criterion A", "Criterion B"],
        )
        graph.add_task(task)

        st.session_state.clear()
        st.session_state["planner_active_graph"] = graph

        with patch("streamlit.checkbox") as mock_cb:
            render_planner_tab()
            crit_keys = [
                call.kwargs.get("key")
                for call in mock_cb.call_args_list
                if call.kwargs.get("key", "").startswith("crit_")
            ]
            self.assertEqual(len(crit_keys), 3)
            self.assertEqual(len(set(crit_keys)), 3, f"Duplicate widget keys detected: {crit_keys}")

    def test_render_planner_tab_invalid_graph_waves_graceful(self):
        import streamlit as st
        from research_assistant.tasks.models import AgentRole, Task, TaskCategory, TaskGraph
        from research_assistant.tasks.ui import render_planner_tab

        graph = TaskGraph(paper_id="cyclic_test", paper_title="Cyclic Paper")
        t1 = Task(task_id="t1", title="T1", category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT, dependencies=["t2"])
        t2 = Task(task_id="t2", title="T2", category=TaskCategory.DATA_PREPROCESSING, assigned_role=AgentRole.DATA_ENGINEER, dependencies=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)

        st.session_state.clear()
        st.session_state["planner_active_graph"] = graph

        with patch("streamlit.error") as mock_err:
            render_planner_tab()
            self.assertTrue(mock_err.called)

    def test_render_planner_tab_execute_button_workspace_safety(self):
        import streamlit as st
        from research_assistant import config
        from research_assistant.tasks.models import AgentRole, Task, TaskCategory, TaskGraph
        from research_assistant.tasks.ui import render_planner_tab

        graph = TaskGraph(paper_id="2608_11188v1", paper_title="Sample Physics Paper")
        t1 = Task(task_id="t1", title="Setup", category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT)
        graph.add_task(t1)

        st.session_state.clear()
        st.session_state["planner_active_graph"] = graph

        def mock_button(label, *args, **kwargs):
            if kwargs.get("key") == "planner_btn_exec":
                return True
            return False

        with patch("streamlit.button", side_effect=mock_button), \
             patch("streamlit.rerun"), \
             patch("research_assistant.tasks.ui.TaskRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_report = MagicMock()
            mock_report.scorecard = {
                "rating": "CERTIFIED_REPLICABLE",
                "replication_rate": 1.0,
                "total_tasks": 1,
                "completed_tasks": 1,
                "failed_tasks": 0,
                "blocked_tasks": 0,
            }
            mock_runner.execute.return_value = mock_report
            mock_runner_cls.return_value = mock_runner

            render_planner_tab()

            mock_runner_cls.assert_called_once()
            called_ws = mock_runner_cls.call_args.kwargs.get("workspace_dir")
            self.assertIsNotNone(called_ws)
            abs_data = os.path.abspath(config.DATA_DIR)
            abs_ws = os.path.abspath(called_ws)
            self.assertFalse(
                abs_ws == abs_data or abs_ws.startswith(abs_data + os.sep),
                f"Workspace '{abs_ws}' must not be inside DATA_DIR '{abs_data}'",
            )
            self.assertIn("2608_11188v1", abs_ws)


if __name__ == "__main__":
    unittest.main()
