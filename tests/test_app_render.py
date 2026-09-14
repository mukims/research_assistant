# tests/test_app_render.py
"""The one pure piece of Tab 1's synthesis rendering: which warnings the
reader sees. Everything else in app.py is Streamlit calls."""

import unittest

import app


class TestSynthesisWarnings(unittest.TestCase):
    def test_clean_answer_has_no_warnings(self):
        self.assertEqual(app._synthesis_warnings({"unverified_citations": [], "irrelevant_cited": []}), [])

    def test_unverified_and_irrelevant_keys_are_named(self):
        w = app._synthesis_warnings({"unverified_citations": ["P7"], "irrelevant_cited": ["P2"], "keys": {"P2": "Paper 2"}})
        self.assertEqual(len(w), 2)
        self.assertIn("P7", w[0]); self.assertIn("P2", w[1]); self.assertIn("Paper 2", w[1])

    def test_old_answers_without_the_fields_are_fine(self):
        self.assertEqual(app._synthesis_warnings({"suggestion": "x"}), [])


class TestAuditItemBadge(unittest.TestCase):
    def test_badges_follow_outcome_before_judgement(self):
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Supports"}), "🟢 Supports")
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Does not support"}), "🟠 Does Not Support")
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Unclear / insufficient evidence"}), "⚪ Unclear / Insufficient Evidence")
        self.assertEqual(app._audit_item_badge({"outcome": "cap_exceeded", "judgement": None}), "⏸ Not Assessed (budget)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_attempted", "judgement": None}), "⏸ Not Assessed (backend down)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_a_claim", "judgement": None, "role": "software"}), "🔧 Not a Claim (software)")
        self.assertEqual(app._audit_item_badge({"outcome": "deferred_paywalled", "judgement": None}), "⏳ Deferred (Pending Evidence)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_downloaded", "judgement": None}), "🔒 Paywalled / Not In Corpus")
        # No outcome ever falls through to a verdict-looking label.
        self.assertNotIn("Unclear", app._audit_item_badge({"outcome": "retrieval_failed", "judgement": None}))


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
        self.assertEqual(app._resolve_job("nope"), (None, None, None, None))

    def test_running_and_finished_jobs_come_from_the_registry(self):
        import threading, time
        gate = threading.Event()
        job = self.run_jobs.start_job("j", "p.pdf", lambda: (gate.wait(5), {"seed_path": "p.pdf"})[1], origin="audit")
        self.assertEqual(app._resolve_job("j")[0], "running")
        gate.set()
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        state, final, error, origin = app._resolve_job("j")
        self.assertEqual((state, final["seed_path"], error, origin), ("done", "p.pdf", None, "audit"))

    def test_a_finished_run_survives_a_server_restart_via_disk(self):
        self.run_jobs.save_run("k", {"seed_path": "q.pdf", "citation_audit": {"totals": {}}}, label="q.pdf", origin="idea")
        self.run_jobs._reset_registry_for_tests()
        state, final, error, origin = app._resolve_job("k")
        self.assertEqual((state, final["seed_path"], origin), ("done", "q.pdf", "idea"))

    def test_failed_job_reports_its_error(self):
        import time
        job = self.run_jobs.start_job("f", "p.pdf", lambda: (_ for _ in ()).throw(RuntimeError("boom")), origin="audit")
        deadline = time.time() + 5
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.01)
        state, _, error, _ = app._resolve_job("f")
        self.assertEqual(state, "failed"); self.assertIn("boom", error)


class TestPipelineExecutionIsStreamlitFree(unittest.TestCase):
    def test_execute_pipeline_never_calls_streamlit(self):
        """It runs in a job thread with no ScriptRunContext. One st.* call
        from there — the old progress callback — is exactly what a browser
        refresh used to kill the run through."""
        import inspect, re
        src = inspect.getsource(app._execute_pipeline)
        self.assertIsNone(re.search(r"\bst\.", src), src)
