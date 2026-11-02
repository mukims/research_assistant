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


def _adopt_draft_script(final):
    """AppTest script: adopt a finished draft job, then report what landed in session state."""
    import streamlit as st
    from research_assistant.ui.jobs import _adopt_job_result
    st.session_state["active_job"] = "j1"
    st.session_state["active_job_origin"] = "draft"
    _adopt_job_result(final, "draft")


class TestAdoptDraftResult(unittest.TestCase):
    def _adopt(self, final):
        from streamlit.testing.v1 import AppTest
        at = _run_app_test(AppTest.from_function(_adopt_draft_script, args=(final,)))
        self.assertEqual([e.value for e in at.exception], [])
        return at

    def test_adopt_draft_result(self):
        at = self._adopt({"written": "/x_cited.txt"})
        self.assertEqual(at.session_state["batch_result"], "/x_cited.txt")
        # A draft run is not an audit or an idea run.
        for key in ("audit_result", "audit_query", "build_result", "build_query", "idea_result", "idea_query"):
            self.assertNotIn(key, at.session_state)
        self.assertNotIn("active_job", at.session_state)
        self.assertNotIn("active_job_origin", at.session_state)

    def test_adopting_a_run_that_wrote_nothing_keeps_the_none(self):
        at = self._adopt({"written": None})
        self.assertIn("batch_result", at.session_state)
        self.assertIsNone(at.session_state["batch_result"])

    def test_adopt_verification_result_is_keyed_by_the_draft(self):
        verification = {"totals": {"total": 1, "judged": 1}}
        at = self._adopt({"verification": verification, "written": "/x_cited.txt"})
        self.assertEqual(at.session_state["verification:/x_cited.txt"], verification)


class TestDraftJobs(unittest.TestCase):
    """The draft tab's citing and verification are server-side jobs, with one error per failure."""

    KEYS = ("active_job", "active_job_origin", "batch_result", "idea_result", "audit_result")

    def setUp(self):
        import tempfile
        from unittest.mock import patch
        import streamlit as st
        import research_assistant.shared.seed_audit as sa
        from research_assistant import config
        from research_assistant.shared import run_jobs
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        for target, attr in ((sa, "AUDIT_DIR"), (config, "DRAFTS_DIR")):
            p = patch.object(target, attr, self.tmp.name); p.start(); self.addCleanup(p.stop)
        run_jobs._reset_registry_for_tests()
        self.run_jobs = run_jobs
        self.st = st
        self.addCleanup(self._clear_state)
        self._clear_state()

    def _clear_state(self):
        for key in list(self.st.session_state.keys()):
            if key in self.KEYS or key.startswith("verification:"):
                del self.st.session_state[key]
        self.st.query_params.clear()

    def _wait(self, job_id):
        import time
        job = self.run_jobs.get_job(job_id)
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        return job

    def test_cite_job_runs_the_citer_and_returns_the_written_path(self):
        from unittest.mock import patch
        from research_assistant.ui import draft_tab
        resources = ("coll", "bm25", ["t"], [{}])
        with patch.object(draft_tab, "get_cached_search_resources", return_value=resources), \
             patch("research_assistant.agents.agent5_batch_citer.run_batch_citer", side_effect=lambda path, out, search_resources=None: out) as citer:
            job_id = draft_tab._start_draft_job("cite", "A claim. Another claim.")
            self.assertIsNotNone(job_id)
            job = self._wait(job_id)
        self.assertEqual(job.state, "done")
        self.assertEqual(job.origin, "draft")
        path, out = citer.call_args[0]
        self.assertTrue(out.endswith("_cited.txt"))
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "A claim. Another claim.")
        # The job thread got the resources the session resolved; it never resolves them itself.
        self.assertEqual(citer.call_args[1]["search_resources"], resources)
        self.assertEqual(job.result, {"written": out})
        self.assertEqual(self.st.session_state["active_job"], job_id)
        self.assertEqual(self.st.session_state["active_job_origin"], "draft")
        self.assertEqual(self.st.query_params["job"], job_id)

    def test_verify_job_returns_the_verification_for_that_draft(self):
        from unittest.mock import patch
        from research_assistant.ui import draft_tab
        verification = {"totals": {"total": 2, "judged": 2}}
        with patch.object(draft_tab, "verify_draft", return_value=verification) as verify:
            job_id = draft_tab._start_draft_job("verify", "/d/x_cited.txt")
            job = self._wait(job_id)
        verify.assert_called_once_with("/d/x_cited.txt")
        self.assertEqual(job.result["verification"], verification)
        self.assertEqual(job.result["written"], "/d/x_cited.txt")

    def test_a_second_run_while_one_is_active_warns_and_starts_nothing(self):
        import threading
        from unittest.mock import patch
        from research_assistant.ui import draft_tab
        gate = threading.Event()
        self.run_jobs.start_job("busy", "Other run", lambda: (gate.wait(5), {})[1], origin="audit")
        self.addCleanup(gate.set)
        with patch.object(draft_tab.st, "warning") as warn:
            self.assertIsNone(draft_tab._start_draft_job("verify", "/d/x_cited.txt"))
        warn.assert_called_once()
        self.assertNotIn("active_job", self.st.session_state)

    def test_failed_citing_job_shows_one_error_and_no_traceback(self):
        from unittest.mock import patch
        from research_assistant.ui import draft_tab
        with patch.object(draft_tab, "get_cached_search_resources", return_value=None), \
             patch("research_assistant.agents.agent5_batch_citer.run_batch_citer", side_effect=RuntimeError("model down")):
            job_id = draft_tab._start_draft_job("cite", "A claim.")
            self.assertEqual(self._wait(job_id).state, "failed")
        with patch.object(ui_jobs.st, "error") as err, \
             patch.object(ui_jobs.st, "exception") as exc, \
             patch.object(ui_jobs.st, "warning") as warn:
            self.assertFalse(ui_jobs._handle_job_state("draft"))
        err.assert_called_once()
        self.assertTrue(err.call_args[0][0].startswith("Citing failed: "))
        self.assertIn("model down", err.call_args[0][0])
        exc.assert_not_called()
        warn.assert_not_called()
        self.assertNotIn("active_job", self.st.session_state)

    def test_a_failed_job_is_reported_once_not_on_every_rerun(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from research_assistant.ui import draft_tab

        def script():
            from research_assistant.ui.draft_tab import render_draft_tab
            render_draft_tab(5)

        with patch.object(draft_tab, "get_cached_search_resources", return_value=None), \
             patch("research_assistant.agents.agent5_batch_citer.run_batch_citer", side_effect=RuntimeError("model down")):
            job_id = draft_tab._start_draft_job("cite", "A claim.")
            self.assertEqual(self._wait(job_id).state, "failed")
        # The job id rides in the URL; this is the refresh (or next rerun) that finds it.
        at = AppTest.from_function(script)
        at.query_params["job"] = job_id
        at = _run_app_test(at)
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], ["Citing failed: model down"])
        self.assertNotIn("job", at.query_params)
        # The next rerun, and a fresh upload error beside it, see no leftover.
        at.text_area[0].set_value("   ")
        at.button[0].click()
        at = _run_app_test(at)
        self.assertEqual([e.value for e in at.error], ["The draft is empty."])

    def test_a_cancelled_draft_job_is_forgotten_after_its_warning(self):
        from unittest.mock import patch
        self.st.session_state["active_job"] = "j"
        self.st.session_state["active_job_origin"] = "draft"
        self.st.query_params["job"] = "j"
        with patch("research_assistant.ui.jobs._resolve_job", return_value=("cancelled", None, "User cancelled", "draft")), \
             patch.object(ui_jobs.st, "warning") as warn:
            self.assertFalse(ui_jobs._handle_job_state("draft"))
            warn.assert_called_once()
            self.assertNotIn("active_job", self.st.session_state)
            self.assertNotIn("active_job_origin", self.st.session_state)
            self.assertNotIn("job", self.st.query_params)
            ui_jobs._handle_job_state("draft")
            warn.assert_called_once()

    def test_failed_verification_job_says_verification_failed(self):
        from unittest.mock import patch
        from research_assistant.ui import draft_tab
        with patch.object(draft_tab, "verify_draft", side_effect=FileNotFoundError("No citation mapping")):
            job_id = draft_tab._start_draft_job("verify", "/d/x_cited.txt")
            self._wait(job_id)
        with patch.object(ui_jobs.st, "error") as err:
            ui_jobs._handle_job_state("draft")
        err.assert_called_once()
        self.assertTrue(err.call_args[0][0].startswith("Verification failed: "))

    def test_alignment_message_only_when_the_job_succeeded_without_writing(self):
        from streamlit.testing.v1 import AppTest
        def script():
            import streamlit as st
            from research_assistant.ui.draft_tab import render_draft_tab
            if "seeded" not in st.session_state:
                st.session_state["seeded"] = True
                st.session_state["batch_result"] = None
            render_draft_tab(5)
        at = _run_app_test(AppTest.from_function(script))
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.error), 1)
        self.assertIn("could not be aligned", at.error[0].value)

    def test_no_message_at_all_before_any_run(self):
        from streamlit.testing.v1 import AppTest
        def script():
            from research_assistant.ui.draft_tab import render_draft_tab
            render_draft_tab(5)
        at = _run_app_test(AppTest.from_function(script))
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.error), 0)

    def test_the_draft_tab_ignores_other_tabs_jobs_and_they_ignore_its(self):
        from unittest.mock import patch
        self.st.session_state["active_job"] = "j"
        with patch("research_assistant.ui.jobs._resolve_job", return_value=("running", None, None, "audit")), \
             patch.object(ui_jobs, "_render_job_progress") as progress:
            self.assertFalse(ui_jobs._handle_job_state("draft"))
        progress.assert_not_called()
        for other in ("audit", "idea"):
            with patch("research_assistant.ui.jobs._resolve_job", return_value=("done", {"written": "/x"}, None, "draft")), \
                 patch.object(ui_jobs, "_adopt_job_result") as adopt:
                self.assertFalse(ui_jobs._handle_job_state(other))
            adopt.assert_not_called()

    def test_a_finished_draft_job_is_adopted_once_per_session(self):
        from unittest.mock import patch
        self.st.session_state["active_job"] = "j"
        with patch("research_assistant.ui.jobs._resolve_job", return_value=("done", {"written": "/x_cited.txt"}, None, "draft")):
            self.assertFalse(ui_jobs._handle_job_state("draft"))
            self.assertEqual(self.st.session_state["batch_result"], "/x_cited.txt")
            self.assertNotIn("active_job", self.st.session_state)
            # A later rerun finds the job id in the URL; it must not clobber what the user has since.
            self.st.session_state["batch_result"] = "/y_cited.txt"
            self.st.query_params["job"] = "j"
            ui_jobs._handle_job_state("draft")
        self.assertEqual(self.st.session_state["batch_result"], "/y_cited.txt")

    def test_a_refreshed_session_adopts_a_finished_verification_and_its_draft(self):
        from unittest.mock import patch
        self.st.query_params["job"] = "j"
        final = {"verification": {"totals": {"total": 1, "judged": 1}}, "written": "/x_cited.txt"}
        with patch("research_assistant.ui.jobs._resolve_job", return_value=("done", final, None, "draft")):
            ui_jobs._handle_job_state("draft")
        self.assertEqual(self.st.session_state["verification:/x_cited.txt"], final["verification"])
        self.assertEqual(self.st.session_state["batch_result"], "/x_cited.txt")

    def test_draft_runners_never_call_streamlit(self):
        """They run in a job thread with no ScriptRunContext."""
        import inspect, re
        from research_assistant.ui import draft_tab
        for fn in (draft_tab._run_cite, draft_tab._run_verify):
            self.assertIsNone(re.search(r"\bst\.", inspect.getsource(fn)), fn.__name__)


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
