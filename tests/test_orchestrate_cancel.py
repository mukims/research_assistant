"""A cancel left in pipeline_status by the sidebar must not kill the next CLI run.

The sidebar's Stop button sets cancel_requested in the shared status file. Nothing
clears it when the run it was meant for has already ended, so the next `orchestrate.py`
or `scripts/transfer.py` invocation saw it at its first checkpoint and stopped before
doing any work. CLI entry points start a fresh run; a cancel that predates it is stale.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import orchestrate
from research_assistant.shared import pipeline_status
from research_assistant.transfer import engine
from scripts import transfer as transfer_cli


class TestStaleCancel(unittest.TestCase):
    def setUp(self):
        pipeline_status.clear_cancel_request()
        self.addCleanup(pipeline_status.clear_cancel_request)

    @staticmethod
    def _leave_stale_cancel_on_a_dead_active_run():
        """A sidebar click landed on a status file that still says a run is active (the run
        died without going idle). set_status's clean-start rule only clears a cancel when
        the file was idle, so this one survives into the next run."""
        pipeline_status.set_status(active=True, stage="fetch", stage_label="Fetching", current_step=3)
        with pipeline_status._status_lock():
            status = pipeline_status._load_status_from_disk()
            status["cancel_requested"] = True
            pipeline_status._persist_status_locked(status)
        assert pipeline_status.is_cancel_requested()

    def test_cli_run_ignores_a_stale_sidebar_cancel(self):
        self._leave_stale_cancel_on_a_dead_active_run()
        seen = {}

        class FakeApp:
            def stream(self, inputs, config=None, stream_mode="values"):  # run() streams whole states
                seen["cancel_at_start"] = pipeline_status.is_cancel_requested()
                yield {**inputs, "seed_path": None, "stopped": "no seed"}

        with patch.object(orchestrate, "build_graph", return_value=FakeApp()):
            orchestrate.run(query="q")
        self.assertIs(seen["cancel_at_start"], False)

    def test_transfer_cli_ignores_a_stale_sidebar_cancel(self):
        pipeline_status.request_cancel("old sidebar click")
        seen = {}

        def fake_live_deps(*args, **kwargs):
            seen["cancel_at_start"] = pipeline_status.is_cancel_requested()
            raise SystemExit(0)

        with patch.object(engine, "live_deps", side_effect=fake_live_deps), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                transfer_cli.main(["-q", "x"])
        self.assertIs(seen["cancel_at_start"], False)

    def test_transfer_cli_busy_refusal_leaves_the_running_jobs_cancel_alone(self):
        pipeline_status.request_cancel("stop the live run")
        with patch.object(engine, "ensure_idle", side_effect=engine.BusyError("Another run is active: x")), \
             patch.object(engine, "live_deps") as live, patch("sys.stderr", io.StringIO()):
            self.assertEqual(transfer_cli.main(["-q", "x"]), 2)
        live.assert_not_called()
        self.assertTrue(pipeline_status.is_cancel_requested())


if __name__ == "__main__":
    unittest.main()
