import io
from contextlib import redirect_stderr, redirect_stdout
import unittest
from unittest.mock import patch

from research_assistant.transfer import engine, stages
from scripts import transfer as cli
from tests.transfer_fixtures import world


class TestTransferCli(unittest.TestCase):
    def test_busy_exits_2_and_uncovered_exits_3(self):
        with patch.object(cli.engine, "ensure_idle", side_effect=engine.BusyError("Another run is active: Ingesting")), \
             patch.object(cli.engine, "live_deps") as ld, redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["-q", "x"]), 2)
        ld.assert_not_called()
        refusal = stages.CorpusCoverageError("x", [], [{"document": "a", "citation": "Paper A", "summary": "", "score": 0.4}])
        with patch.object(cli.engine, "ensure_idle"), patch.object(cli.engine, "live_deps"), \
             patch.object(cli.engine, "run_best_of_n", side_effect=refusal), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(["-q", "x"]), 3)
        self.assertIn("Paper A", err.getvalue())

    def test_show_cards_writes_no_transfer(self):
        deps, log = world()
        with patch.object(cli.engine, "ensure_idle"), patch.object(cli.engine, "live_deps", return_value=deps), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["-q", "q", "--show-cards"]), 0)
        self.assertEqual(log.critic_calls, 0)
        self.assertIn("s1", out.getvalue())
