"""
Unit tests for tasks Streamlit UI component.
"""

import os
from unittest.mock import patch, MagicMock
import pytest

from research_assistant.tasks.ui import (
    _list_candidate_papers,
    _render_mermaid_interactive,
)


def test_list_candidate_papers(tmp_path, monkeypatch):
    pulled = tmp_path / "pulled_pdfs"
    pulled.mkdir()
    (pulled / "paper1.pdf").write_text("dummy pdf")

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "paper2.pdf").write_text("dummy raw pdf")

    audit_tei = tmp_path / "audit" / "tei"
    audit_tei.mkdir(parents=True)
    (audit_tei / "paper3.tei.xml").write_text("<TEI/>")

    from research_assistant import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PULLED_PDFS_DIR", str(pulled))
    monkeypatch.setattr(config, "RAW_DIR", str(raw))

    candidates = _list_candidate_papers()
    assert len(candidates) == 3
    basenames = [os.path.basename(c) for c in candidates]
    assert "paper1.pdf" in basenames
    assert "paper2.pdf" in basenames
    assert "paper3.tei.xml" in basenames


@patch("streamlit.components.v1.html")
def test_render_mermaid_interactive(mock_html):
    code = "flowchart TD\n    A --> B"
    _render_mermaid_interactive(code, height=300)
    mock_html.assert_called_once()
    args, kwargs = mock_html.call_args
    assert "mermaid.min.js" in args[0]
    assert code in args[0]
    assert kwargs["height"] == 300


def test_render_planner_tab_select_flow(tmp_path, monkeypatch):
    import streamlit as st
    from research_assistant.tasks.ui import render_planner_tab

    pulled = tmp_path / "pulled_pdfs"
    pulled.mkdir()
    (pulled / "sample_corpus_paper.pdf").write_bytes(b"%PDF-1.4\n")

    from research_assistant import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PULLED_PDFS_DIR", str(pulled))
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path / "raw"))

    st.session_state.clear()
    with patch("streamlit.radio", return_value="Select Ingested Paper"), \
         patch("streamlit.selectbox", return_value=0), \
         patch("streamlit.checkbox", return_value=True), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.spinner"), \
         patch("streamlit.success"):

        render_planner_tab()
        assert "planner_active_graph" in st.session_state
        graph = st.session_state["planner_active_graph"]
        assert graph is not None
        assert len(graph.tasks) == 5
        assert graph.paper_id == "sample_corpus_paper"


def test_render_planner_tab_upload_flow(tmp_path, monkeypatch):
    import streamlit as st
    from research_assistant.tasks.ui import render_planner_tab

    from research_assistant import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

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
        assert "planner_active_graph" in st.session_state
        graph = st.session_state["planner_active_graph"]
        assert graph is not None
        assert len(graph.tasks) == 5
        assert ".." not in graph.paper_id
        assert "paper_upload" in graph.paper_id


def test_render_planner_tab_upload_slash_only_name(tmp_path, monkeypatch):
    import streamlit as st
    from research_assistant.tasks.ui import render_planner_tab

    from research_assistant import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

    st.session_state.clear()
    mock_upload = MagicMock()
    mock_upload.name = "///"
    mock_upload.getvalue.return_value = b"Method: Testing slash-only name."

    with patch("streamlit.radio", return_value="Upload New Paper / TEI XML / Text"), \
         patch("streamlit.file_uploader", return_value=mock_upload), \
         patch("streamlit.checkbox", return_value=True), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.spinner"), \
         patch("streamlit.success"):

        render_planner_tab()
        assert "planner_active_graph" in st.session_state
        graph = st.session_state["planner_active_graph"]
        assert graph is not None
        assert graph.paper_id == "uploaded_paper"


def test_list_candidate_papers_uppercase_extensions(tmp_path, monkeypatch):
    pulled = tmp_path / "pulled_pdfs"
    pulled.mkdir()
    (pulled / "paper_upper.PDF").write_text("dummy")

    audit_tei = tmp_path / "audit" / "tei"
    audit_tei.mkdir(parents=True)
    (audit_tei / "paper_upper.XML").write_text("<TEI/>")

    from research_assistant import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PULLED_PDFS_DIR", str(pulled))
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path / "raw"))

    candidates = _list_candidate_papers()
    basenames = [os.path.basename(c) for c in candidates]
    assert "paper_upper.PDF" in basenames
    assert "paper_upper.XML" in basenames


def test_render_planner_tab_upload_nameless_text_content_sniffing(tmp_path, monkeypatch):
    import streamlit as st
    from research_assistant.tasks.ui import render_planner_tab

    from research_assistant import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

    st.session_state.clear()
    mock_upload = MagicMock()
    mock_upload.name = ""
    # Plain text content without %PDF-
    mock_upload.getvalue.return_value = b"Method: Experimental setup and evaluation in plain text."

    with patch("streamlit.radio", return_value="Upload New Paper / TEI XML / Text"), \
         patch("streamlit.file_uploader", return_value=mock_upload), \
         patch("streamlit.checkbox", return_value=True), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.spinner"), \
         patch("streamlit.success"):

        render_planner_tab()
        assert "planner_active_graph" in st.session_state
        graph = st.session_state["planner_active_graph"]
        assert graph is not None
        assert graph.paper_id == "uploaded_paper"
        # Verify file saved on disk has .txt extension
        saved_files = os.listdir(tmp_path / "tmp_uploads")
        assert any(f.endswith(".txt") for f in saved_files)


def test_render_planner_tab_duplicate_acceptance_criteria_keys():
    import streamlit as st
    from research_assistant.tasks.models import Task, TaskGraph, TaskCategory, AgentRole
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
        # Extract keys used for acceptance criteria
        crit_keys = [
            call.kwargs.get("key")
            for call in mock_cb.call_args_list
            if call.kwargs.get("key", "").startswith("crit_")
        ]
        assert len(crit_keys) == 3
        assert len(set(crit_keys)) == 3, f"Duplicate widget keys detected: {crit_keys}"


def test_render_planner_tab_invalid_graph_waves_graceful():
    import streamlit as st
    from research_assistant.tasks.models import Task, TaskGraph, TaskCategory, AgentRole
    from research_assistant.tasks.ui import render_planner_tab

    # Create a cyclic graph that causes topological_waves to fail
    graph = TaskGraph(paper_id="cyclic_test", paper_title="Cyclic Paper")
    t1 = Task(task_id="t1", title="T1", category=TaskCategory.ENVIRONMENT_SETUP, assigned_role=AgentRole.DEVOPS_AGENT, dependencies=["t2"])
    t2 = Task(task_id="t2", title="T2", category=TaskCategory.DATA_PREPROCESSING, assigned_role=AgentRole.DATA_ENGINEER, dependencies=["t1"])
    graph.add_task(t1)
    graph.add_task(t2)

    st.session_state.clear()
    st.session_state["planner_active_graph"] = graph

    with patch("streamlit.error") as mock_err:
        render_planner_tab()
        # Should catch the error and display clean notification rather than raising uncaught exception
        assert mock_err.called




