"""Background pipeline runs that outlive the browser session.

A browser refresh ends a Streamlit session and stops its script at the
next widget call — which, with the pipeline running inside the session,
was the pipeline's own progress callback. These tests pin the contract
that fixes it: a run is a thread in the server process, its state is in a
registry any session can read, and its result is on disk under a job id
that survives a refresh (it rides in the URL).
"""

import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from research_assistant.shared import run_jobs
from research_assistant.shared.pipeline_status import PipelineCancelledError


def _wait(job, timeout=5.0):
    deadline = time.time() + timeout
    while job.state == "running" and time.time() < deadline:
        time.sleep(0.01)
    return job


class _RunsDir(unittest.TestCase):
    def setUp(self):
        import research_assistant.shared.seed_audit as sa
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(sa, "AUDIT_DIR", self.tmp.name)
        p.start(); self.addCleanup(p.stop)
        run_jobs._reset_registry_for_tests()


class TestStartJob(_RunsDir):
    def test_runner_runs_in_another_thread_and_result_is_kept(self):
        seen = {}
        gate = threading.Event()

        def runner():
            seen["thread"] = threading.current_thread().name
            gate.wait(5)
            return {"seed_path": "x.pdf", "citation_audit": {"totals": {"total": 1}}}

        job = run_jobs.start_job("job1", "x.pdf", runner, origin="audit")
        self.assertEqual(job.state, "running")
        gate.set(); _wait(job)
        self.assertEqual(job.state, "done")
        self.assertNotEqual(seen["thread"], threading.current_thread().name)
        self.assertEqual(job.result["seed_path"], "x.pdf")
        self.assertIs(run_jobs.get_job("job1"), job)

    def test_result_is_saved_to_disk_and_loadable_without_the_registry(self):
        job = run_jobs.start_job("job2", "paper.pdf", lambda: {"seed_path": "paper.pdf", "answer": {"suggestion": "s"}},
                                 origin="audit")
        _wait(job)
        run_jobs._reset_registry_for_tests()
        self.assertIsNone(run_jobs.get_job("job2"))
        saved = run_jobs.load_run("job2")
        self.assertEqual(saved["final"]["seed_path"], "paper.pdf")
        self.assertEqual(saved["label"], "paper.pdf")
        self.assertEqual(saved["origin"], "audit")
        self.assertIn("saved_at", saved)

    def test_snapshot_is_updated_while_running(self):
        gate = threading.Event()

        def runner(on_snapshot):
            on_snapshot({"seed_path": "p.pdf", "downloaded": 3})
            gate.wait(5)
            return {"seed_path": "p.pdf", "downloaded": 3, "citation_audit": {}}

        job = run_jobs.start_job("job3", "p.pdf", runner, origin="audit")
        deadline = time.time() + 5
        while not job.snapshot and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(job.snapshot.get("downloaded"), 3)
        self.assertEqual(job.state, "running")
        gate.set(); _wait(job)
        self.assertEqual(job.state, "done")

    def test_failure_is_recorded_and_nothing_is_saved(self):
        def runner():
            raise RuntimeError("chroma is unreachable")

        job = _wait(run_jobs.start_job("job4", "p.pdf", runner, origin="audit"))
        self.assertEqual(job.state, "failed")
        self.assertIn("chroma is unreachable", job.error)
        self.assertIsNone(run_jobs.load_run("job4"))

    def test_cancellation_is_its_own_state(self):
        def runner():
            raise PipelineCancelledError("stopped via sidebar")

        job = _wait(run_jobs.start_job("job5", "p.pdf", runner, origin="audit"))
        self.assertEqual(job.state, "cancelled")
        self.assertIsNone(run_jobs.load_run("job5"))


class TestSingleFlight(_RunsDir):
    def test_same_id_while_running_attaches_to_the_existing_job(self):
        gate = threading.Event()
        first = run_jobs.start_job("same", "p.pdf", lambda: (gate.wait(5), {"seed_path": "p.pdf"})[1], origin="audit")
        again = run_jobs.start_job("same", "p.pdf", lambda: {"seed_path": "other"}, origin="audit")
        self.assertIs(again, first)
        gate.set(); _wait(first)
        self.assertEqual(first.result["seed_path"], "p.pdf")

    def test_a_different_job_while_one_runs_is_refused(self):
        gate = threading.Event()
        first = run_jobs.start_job("a", "a.pdf", lambda: (gate.wait(5), {})[1], origin="audit")
        with self.assertRaises(run_jobs.JobBusy) as ctx:
            run_jobs.start_job("b", "b.pdf", lambda: {}, origin="audit")
        self.assertEqual(ctx.exception.job.job_id, "a")
        gate.set(); _wait(first)
        # Once it finishes, a new job may start.
        second = _wait(run_jobs.start_job("b", "b.pdf", lambda: {"seed_path": "b.pdf"}, origin="audit"))
        self.assertEqual(second.state, "done")

    def test_finished_job_with_same_id_is_rerun(self):
        first = _wait(run_jobs.start_job("r", "p.pdf", lambda: {"n": 1}, origin="audit"))
        second = _wait(run_jobs.start_job("r", "p.pdf", lambda: {"n": 2}, origin="audit"))
        self.assertIsNot(second, first)
        self.assertEqual(second.result["n"], 2)


class TestRunsOnDisk(_RunsDir):
    def test_list_runs_is_newest_first_with_labels(self):
        for i, (jid, label) in enumerate((("old", "first.pdf"), ("new", "second.pdf"))):
            run_jobs.save_run(jid, {"seed_path": f"/x/{label}", "seed_name": label}, label=label, origin="audit")
            os.utime(run_jobs.run_path(jid), (1_700_000_000 + i, 1_700_000_000 + i))
        runs = run_jobs.list_runs()
        self.assertEqual([r["job_id"] for r in runs], ["new", "old"])
        self.assertEqual(runs[0]["label"], "second.pdf")
        self.assertEqual(runs[0]["origin"], "audit")

    def test_unserialisable_values_do_not_lose_the_run(self):
        class Odd:
            def __str__(self):
                return "odd-object"
        run_jobs.save_run("odd", {"seed_path": "p.pdf", "weird": Odd()}, label="p.pdf", origin="audit")
        self.assertEqual(run_jobs.load_run("odd")["final"]["weird"], "odd-object")

    def test_corrupt_run_file_is_skipped_not_fatal(self):
        os.makedirs(run_jobs.runs_dir(), exist_ok=True)
        with open(run_jobs.run_path("bad"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertIsNone(run_jobs.load_run("bad"))
        self.assertEqual(run_jobs.list_runs(), [])


class TestJobId(unittest.TestCase):
    def test_job_id_is_the_graph_thread_id(self):
        import hashlib
        self.assertEqual(run_jobs.job_id_for("file_abc123.pdf"), hashlib.sha1(b"file_abc123.pdf").hexdigest()[:16])
        self.assertEqual(run_jobs.job_id_for("file_abc123.pdf"), run_jobs.job_id_for("file_abc123.pdf"))


class TestNoStreamlitInTheModule(unittest.TestCase):
    def test_module_never_touches_streamlit(self):
        """A job thread has no ScriptRunContext; one st.* call from it would
        raise or silently do nothing. The module must not know Streamlit."""
        import inspect, re
        src = inspect.getsource(run_jobs)
        self.assertNotIn("import streamlit", src)
        self.assertIsNone(re.search(r"\bst\.", src))


if __name__ == "__main__":
    unittest.main()
