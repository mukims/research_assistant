# tests/test_app_render.py
"""The pure pieces of the Streamlit front-end: which synthesis warnings the
reader sees, how an audit item is badged, how a run is found again. Everything
else in research_assistant/ui/ is Streamlit calls."""

import sys
import unittest

from research_assistant.ui import audit as ui_audit
from research_assistant.ui import jobs as ui_jobs
from research_assistant.ui import results as ui_results


class TestSynthesisWarnings(unittest.TestCase):
    def test_clean_answer_has_no_warnings(self):
        self.assertEqual(ui_results._synthesis_warnings({"unverified_citations": [], "irrelevant_cited": []}), [])

    def test_unverified_and_irrelevant_keys_are_named(self):
        w = ui_results._synthesis_warnings({"unverified_citations": ["P7"], "irrelevant_cited": ["P2"], "keys": {"P2": "Paper 2"}})
        self.assertEqual(len(w), 2)
        self.assertIn("P7", w[0]); self.assertIn("P2", w[1]); self.assertIn("Paper 2", w[1])

    def test_old_answers_without_the_fields_are_fine(self):
        self.assertEqual(ui_results._synthesis_warnings({"suggestion": "x"}), [])


class TestSynthesisExport(unittest.TestCase):
    def test_build_synthesis_markdown(self):
        answer = {
            "suggestion": "Topological edge states are protected against backscattering [P1].",
            "selected": [
                {
                    "key": "P1",
                    "citation": "Hasan 2010 Colloquium: Topological insulators",
                    "document": "hasan2010.pdf",
                    "score": 0.92,
                    "summary": "Reviews topological insulator theory and phenomenology.",
                }
            ],
            "notes": [
                {
                    "key": "P1",
                    "citation": "Hasan 2010 Colloquium: Topological insulators",
                    "notes": "Edge states exist inside the bulk energy gap.",
                    "relevant": True,
                    "passages_used": 3,
                }
            ],
            "citations": ["Hasan 2010 Colloquium: Topological insulators"],
        }
        md = ui_results._build_synthesis_markdown("quantum wires", answer)
        self.assertIn("# 🔬 Literature Synthesis: quantum wires", md)
        self.assertIn("Topological edge states are protected", md)
        self.assertIn("Hasan 2010 Colloquium", md)
        self.assertIn("Edge states exist inside the bulk energy gap", md)
        self.assertIn("## 📑 Cited Sources", md)

    def test_build_synthesis_bibtex(self):
        answer = {
            "selected": [
                {
                    "key": "P1",
                    "citation": "Kane 2005 Z2 Topological Order and the Quantum Spin Hall Effect",
                    "document": "kane2005.pdf",
                }
            ]
        }
        bib = ui_results._build_synthesis_bibtex(answer)
        self.assertIn("@article{kane2005_p1,", bib)
        self.assertIn("author = {Kane},", bib)
        self.assertIn("year = {2005},", bib)
        self.assertIn("note = {Document: kane2005.pdf},", bib)

    def test_build_synthesis_bibtex_empty(self):
        bib = ui_results._build_synthesis_bibtex({})
        self.assertIn("No references found", bib)


class TestAuditItemBadge(unittest.TestCase):
    def test_badges_follow_outcome_before_judgement(self):
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "judged", "judgement": "Supports"}), "🟢 Supports")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "judged", "judgement": "Does not support"}), "🟠 Does Not Support")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "judged", "judgement": "Unclear / insufficient evidence"}), "⚪ Unclear / Insufficient Evidence")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "cap_exceeded", "judgement": None}), "⏸ Not Assessed (budget)")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "not_attempted", "judgement": None}), "⏸ Not Assessed (backend down)")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "not_a_claim", "judgement": None, "role": "software"}), "🔧 Not a Claim (software)")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "deferred_paywalled", "judgement": None}), "⏳ Deferred (Pending Evidence)")
        self.assertEqual(ui_audit._audit_item_badge({"outcome": "not_downloaded", "judgement": None}), "🔒 Paywalled / Not In Corpus")
        # No outcome ever falls through to a verdict-looking label.
        self.assertNotIn("Unclear", ui_audit._audit_item_badge({"outcome": "retrieval_failed", "judgement": None}))


class TestSeedAndDownloadsScoping(unittest.TestCase):
    """Ensures reference counts and lists in the UI are scoped to the active seed paper."""

    def test_render_seed_and_downloads_scopes_references_to_seed_paper(self):
        from unittest.mock import patch
        final = {
            "seed_path": "data/raw/processed/paper_a.pdf",
            "extraction": {"references_ok": True},
        }
        fake_extracted = {
            "references": [
                {"source_file": "paper_a.pdf", "title": "Ref A1", "raw_reference": "Ref A1", "doi": "10.1/a1"},
                {"source_file": "paper_a.pdf", "title": "Ref A2", "raw_reference": "Ref A2", "doi": "10.1/a2"},
                {"source_file": "paper_b.pdf", "title": "Ref B1", "raw_reference": "Ref B1", "doi": "10.1/b1"},
            ]
        }
        fake_downloaded = {
            "doi:10.1/a1": {"title": "Ref A1", "key": "doi:10.1/a1"},
            "doi:10.1/b1": {"title": "Ref B1", "key": "doi:10.1/b1"},
        }
        fake_failed = {
            "doi:10.1/a2": {"title": "Ref A2", "key": "doi:10.1/a2", "reason": "paywalled"},
        }

        def fake_manifest(path):
            if "extracted_citations" in str(path):
                return fake_extracted
            if "downloaded" in str(path):
                return fake_downloaded
            if "failed" in str(path):
                return fake_failed
            return {}

        with patch.object(ui_results, "_manifest", side_effect=fake_manifest), \
             patch.object(ui_results.st, "expander") as mock_exp, \
             patch.object(ui_results.st, "caption"), \
             patch.object(ui_results.st, "markdown"), \
             patch.object(ui_results.st, "subheader"):
            ui_results._render_seed_and_downloads("test query", final)
            self.assertTrue(mock_exp.called)
            label = mock_exp.call_args[0][0]
            self.assertIn("1 fetched", label)
            self.assertIn("1 unavailable", label)
            self.assertIn("2 references found", label)
            self.assertNotIn("3 references found", label)


class TestJobResolution(unittest.TestCase):
    """How a tab finds a run after a refresh: registry first, disk second."""

    def setUp(self):
        import tempfile
        import research_assistant.shared.seed_audit as sa
        from research_assistant.shared import run_jobs
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        from unittest.mock import patch
        p = patch.object(sa, "AUDIT_DIR", self.tmp.name); p.start(); self.addCleanup(p.stop)
        run_jobs._reset_registry_for_tests()
        self.run_jobs = run_jobs

    def test_unknown_job_id(self):
        self.assertEqual(ui_jobs._resolve_job("nope"), (None, None, None, None))

    def test_running_and_finished_jobs_come_from_the_registry(self):
        import threading, time
        gate = threading.Event()
        job = self.run_jobs.start_job("j", "p.pdf", lambda: (gate.wait(5), {"seed_path": "p.pdf"})[1], origin="audit")
        self.assertEqual(ui_jobs._resolve_job("j")[0], "running")
        gate.set()
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        state, final, error, origin = ui_jobs._resolve_job("j")
        self.assertEqual((state, final["seed_path"], error, origin), ("done", "p.pdf", None, "audit"))

    def test_a_finished_run_survives_a_server_restart_via_disk(self):
        self.run_jobs.save_run("k", {"seed_path": "q.pdf", "citation_audit": {"totals": {}}}, label="q.pdf", origin="idea")
        self.run_jobs._reset_registry_for_tests()
        state, final, error, origin = ui_jobs._resolve_job("k")
        self.assertEqual((state, final["seed_path"], origin), ("done", "q.pdf", "idea"))

    def test_failed_job_reports_its_error(self):
        import time
        job = self.run_jobs.start_job("f", "p.pdf", lambda: (_ for _ in ()).throw(RuntimeError("boom")), origin="audit")
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        state, _, error, _ = ui_jobs._resolve_job("f")
        self.assertEqual(state, "failed"); self.assertIn("boom", error)

    def test_start_audit_job_registers_and_clears_stale_state(self):
        import time
        from unittest.mock import patch
        import streamlit as st
        st.session_state["audit_result"] = {"stale": True}
        st.session_state["build_result"] = {"stale": True}
        with patch("research_assistant.shared.seed_audit.audit_seed_citations") as mock_audit:
            mock_audit.return_value = {"totals": {"total": 1}, "results": []}
            job_id = ui_audit._start_audit_job("p.pdf", "p.pdf", origin="audit", force=True, existing_final={"answer": "saved"})
            self.assertIsNotNone(job_id)
            self.assertNotIn("audit_result", st.session_state)
            self.assertNotIn("build_result", st.session_state)
            self.assertEqual(st.session_state.get("active_job"), job_id)
            job = self.run_jobs.get_job(job_id)
            self.assertIsNotNone(job)
            deadline = time.time() + 5
            while job.state == "running" and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(job.state, "done")
            mock_audit.assert_called_once_with("p.pdf", force=True, skip_if_cached=False)
            self.assertEqual(job.result.get("answer"), "saved")
            self.assertEqual(job.result.get("citation_audit"), {"totals": {"total": 1}, "results": []})



class TestPipelineExecutionIsStreamlitFree(unittest.TestCase):
    def test_execute_pipeline_never_calls_streamlit(self):
        """It runs in a job thread with no ScriptRunContext. One st.* call
        from there — the old progress callback — is exactly what a browser
        refresh used to kill the run through."""
        import inspect, re
        src = inspect.getsource(ui_jobs._execute_pipeline)
        self.assertIsNone(re.search(r"\bst\.", src), src)


class TestInterruptedJobPrompt(unittest.TestCase):
    """Verifies that an interrupted or failed audit job prompts to restart or move to new paper."""

    def test_handle_job_state_renders_prompt_on_failed_audit(self):
        from unittest.mock import patch
        import streamlit as st
        st.session_state["active_job"] = "test_job_failed"
        with patch("research_assistant.ui.jobs._resolve_job", return_value=("failed", None, "GROBID connection refused", "audit")), \
             patch("research_assistant.ui.jobs._render_interrupted_run_prompt") as mock_prompt:
            res = ui_jobs._handle_job_state("audit")
            self.assertTrue(res)
            mock_prompt.assert_called_once()
            args = mock_prompt.call_args[1]
            self.assertEqual(args["state"], "failed")
            self.assertEqual(args["error"], "GROBID connection refused")

    def test_handle_job_state_renders_prompt_on_cancelled_audit(self):
        from unittest.mock import patch
        import streamlit as st
        st.session_state["active_job"] = "test_job_cancelled"
        with patch("research_assistant.ui.jobs._resolve_job", return_value=("cancelled", None, "User cancelled", "audit")), \
             patch("research_assistant.ui.jobs._render_interrupted_run_prompt") as mock_prompt:
            res = ui_jobs._handle_job_state("audit")
            self.assertTrue(res)
            mock_prompt.assert_called_once()
            args = mock_prompt.call_args[1]
            self.assertEqual(args["state"], "cancelled")

    def test_render_interrupted_run_prompt_structure(self):
        from unittest.mock import patch, MagicMock
        with patch.object(ui_jobs.st, "container"), \
             patch.object(ui_jobs.st, "error") as mock_err, \
             patch.object(ui_jobs.st, "caption"), \
             patch.object(ui_jobs.st, "markdown") as mock_md, \
             patch.object(ui_jobs.st, "write") as mock_write, \
             patch.object(ui_jobs.st, "columns", return_value=(MagicMock(), MagicMock())), \
             patch.object(ui_jobs.st, "button", return_value=False) as mock_btn:
            ui_jobs._render_interrupted_run_prompt(
                origin="audit",
                error="Network timeout",
                state="failed",
                job_id="job123",
                final={"seed_path": "paper.pdf"},
            )
            mock_err.assert_called_once()
            self.assertIn("Network timeout", mock_err.call_args[0][0])
            btn_labels = [call[0][0] for call in mock_btn.call_args_list]
            self.assertTrue(any("Restart" in lbl for lbl in btn_labels))
            self.assertTrue(any("New Paper" in lbl for lbl in btn_labels))

    def test_job_restart_clears_stale_cancel_request_and_runs_cleanly(self):
        """Regression test: restarting a previously cancelled job via _start_pipeline_job
        must clear stale cancel flags and execute to completion rather than aborting with
        'cancelled before execution'."""
        from research_assistant.shared import run_jobs, pipeline_status
        from unittest.mock import patch
        import time

        job_id = run_jobs.job_id_for("restart_test_doc.pdf")
        pipeline_status.request_cancel("User stopped previous run", job_id=job_id)
        self.assertTrue(pipeline_status.is_cancel_requested(job_id))

        # Clean in-memory registry if test ran before
        run_jobs.remove_job(job_id)

        with patch("research_assistant.ui.jobs._execute_pipeline", return_value={"status": "success", "result": 42}):
            started_id = ui_jobs._start_pipeline_job(
                query="",
                seed_file_path="/tmp/restart_test_doc.pdf",
                origin="audit",
                force=True,
            )
            self.assertEqual(started_id, job_id)
            job = run_jobs.get_job(job_id)
            self.assertIsNotNone(job)
            job.thread.join(timeout=5.0)

            self.assertEqual(job.state, "done")
            self.assertIsNone(job.error)
            self.assertEqual(job.result.get("status"), "success")
            self.assertFalse(pipeline_status.is_cancel_requested(job_id))
            run_jobs.remove_job(job_id)


def _render_build_script(final, origin):
    """AppTest script: render one run's results exactly as the Results tab does."""
    from research_assistant.ui.results import _render_build
    _render_build(final, "q", origin=origin)


def _run_app_test(at):
    """Run an AppTest, then restore ``sys.modules["__main__"]``.

    Streamlit's script runner swaps it for the script and never swaps it back.
    multiprocessing's spawn children re-run __main__, so a stale one kills every
    later spawn test in the same process."""
    main = sys.modules["__main__"]
    try:
        at.run()
    finally:
        sys.modules["__main__"] = main
    return at


class TestUngroundedLabel(unittest.TestCase):
    """An answer from general knowledge (no seed paper) carries `ungrounded`. The label
    and the missing export buttons must follow that flag, not the run-level `stopped`
    note that only some ungrounded runs also set."""

    LABEL = "Answer (not grounded in retrieved papers)"

    def _render(self, final, origin="idea"):
        from streamlit.testing.v1 import AppTest
        at = _run_app_test(AppTest.from_function(_render_build_script, args=(final, origin)))
        self.assertEqual([e.value for e in at.exception], [])
        return at

    def _labels(self, at):
        return [m.value for m in at.markdown if self.LABEL in m.value]

    def test_ungrounded_answer_without_stopped_is_still_labelled(self):
        at = self._render({"answer": {"suggestion": "x", "ungrounded": True}})
        self.assertEqual(len(self._labels(at)), 1)
        self.assertEqual(len(at.get("download_button")), 0)

    def test_ungrounded_answer_with_stopped_is_labelled_once(self):
        at = self._render({"answer": {"suggestion": "x", "ungrounded": True},
                           "stopped": "Pipeline stopped by user via UI"})
        self.assertEqual(len(self._labels(at)), 1)
        self.assertEqual(len(at.warning), 1)
        self.assertEqual(len(at.get("download_button")), 0)

    def test_ungrounded_answer_in_an_uploaded_collection_is_labelled_too(self):
        at = self._render({"batch_uploaded": [{"filename": "a.pdf"}],
                           "answer": {"suggestion": "x", "ungrounded": True}})
        self.assertEqual(len(self._labels(at)), 1)
        self.assertEqual(len(at.get("download_button")), 0)

    def test_grounded_answer_keeps_its_synthesis_and_exports(self):
        at = self._render({"answer": {"suggestion": "x", "citations": [], "passages": []}})
        self.assertEqual(self._labels(at), [])
        self.assertEqual(len(at.get("download_button")), 2)


def _sidebar_script():
    from research_assistant.ui.common import _render_sidebar_backend_controls
    _render_sidebar_backend_controls()


class TestSidebarModelIsPerSession(unittest.TestCase):
    """The sidebar picker used to assign config.LLM_MODEL / LLM_BACKEND: a process-wide
    write that leaked into other browser sessions and into jobs already running."""

    def test_sidebar_choice_does_not_touch_global_config(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from research_assistant import config
        before = (config.LLM_MODEL, config.LLM_BACKEND)
        at = AppTest.from_function(_sidebar_script)
        at.session_state["chat_model"] = "llama3.2:latest"
        with patch("research_assistant.shared.llm.list_ollama_models", return_value=["llama3.2:latest"]):
            _run_app_test(at)
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual((config.LLM_MODEL, config.LLM_BACKEND), before)
        # The choice itself still lives in the session.
        self.assertEqual(at.session_state["llm_model"], "llama3.2:latest")
        self.assertEqual(at.session_state["llm_backend"], "ollama")

    def test_sidebar_says_which_calls_the_model_governs(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_function(_sidebar_script)
        with patch("research_assistant.shared.llm.list_ollama_models", return_value=[]):
            _run_app_test(at)
        self.assertTrue(any("Applies to chat, synthesis and the planner." in c.value for c in at.caption))


class TestJobCarriesTheSessionModel(unittest.TestCase):
    def test_start_pipeline_job_puts_the_session_model_in_the_inputs(self):
        import streamlit as st
        from unittest.mock import patch
        from research_assistant.shared import run_jobs

        job_id = run_jobs.job_id_for("model_inputs_doc.pdf")
        run_jobs.remove_job(job_id)
        self.addCleanup(run_jobs.remove_job, job_id)
        st.session_state["llm_model"] = "m"
        st.session_state["llm_backend"] = "ollama"
        self.addCleanup(st.session_state.pop, "llm_model", None)
        self.addCleanup(st.session_state.pop, "llm_backend", None)
        with patch("research_assistant.ui.jobs._execute_pipeline", return_value={"status": "success"}) as run:
            ui_jobs._start_pipeline_job(query="", seed_file_path="/tmp/model_inputs_doc.pdf", origin="audit")
            run_jobs.get_job(job_id).thread.join(timeout=5.0)
        inputs = run.call_args.args[0]
        self.assertEqual((inputs["model"], inputs["backend"]), ("m", "ollama"))


class TestAuditTabSynthesisModel(unittest.TestCase):
    def test_collection_upload_synthesis_uses_the_session_model(self):
        import streamlit as st
        from unittest.mock import patch
        from research_assistant.shared import retrieve
        from research_assistant.ui import audit_tab

        st.session_state["llm_model"] = "m"
        st.session_state["llm_backend"] = "ollama"
        self.addCleanup(st.session_state.pop, "llm_model", None)
        self.addCleanup(st.session_state.pop, "llm_backend", None)
        progress = object()
        with patch.object(retrieve, "research_answer", return_value={"suggestion": "s"}) as ra:
            audit_tab._research_answer_for_session("q", progress)
        self.assertEqual(ra.call_args.args, ("q",))
        self.assertEqual(ra.call_args.kwargs["model"], "m")
        self.assertEqual(ra.call_args.kwargs["backend"], "ollama")
        self.assertIs(ra.call_args.kwargs["on_progress"], progress)
