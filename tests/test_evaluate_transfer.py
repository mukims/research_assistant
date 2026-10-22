import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from research_assistant.eval.transfer_cases import Case
from scripts import evaluate_transfer
from tests.transfer_fixtures import world


class TestEvaluateTransfer(unittest.TestCase):
    def test_retrieval_only_never_calls_a_model(self):
        deps, log = world()
        case = Case(
            id="test_case",
            kind="positive",
            query="test query",
            generic="generic bottleneck",
            cutoff_year=2020,
            corpus="main",
            papers={"source": ["Source Paper"], "hidden": []},
            exclude_documents=[],
            expected={
                "source_titles": ["Source Paper"],
                "technique_terms": ["latent"],
                "mapping_terms": [],
                "critic_terms": [],
                "set_aside": None,
            },
        )
        with patch.object(evaluate_transfer, "load_cases", return_value=[case]), \
             patch.object(evaluate_transfer, "case_runner", side_effect=lambda c, mode="retrieval", **kw: evaluate_transfer.run_case_in_process(c, mode, deps=deps)), \
             patch.object(evaluate_transfer, "get_case_resolved", return_value={"Source Paper": "s1"}), \
             redirect_stdout(io.StringIO()) as out:
            rc = evaluate_transfer.main(["--retrieval-only"])
            self.assertEqual(log.chat_calls, 0)
            self.assertEqual(log.critic_calls, 0)
            self.assertIn("F2", out.getvalue())

    def test_a_case_with_target_papers_uses_exactly_them_as_its_near_set(self):
        deps, _ = world()
        case_deps = evaluate_transfer.near_from_targets(deps, {"n4", "n5"})
        shortlist = case_deps.shortlist("q", 6)
        self.assertEqual({r["document"] for r in shortlist}, {"n4", "n5"})
        self.assertEqual(case_deps.gate("q", shortlist), shortlist)

    def test_score_runs_the_leak_guard_before_any_case(self):
        case = Case(
            id="test_leak",
            kind="positive",
            query="test query",
            generic="generic",
            cutoff_year=2020,
            corpus="main",
            papers={"source": [], "hidden": ["Hidden Paper"]},
            exclude_documents=[],
            expected={
                "source_titles": [],
                "technique_terms": [],
                "mapping_terms": [],
                "critic_terms": [],
                "set_aside": None,
            },
        )
        with patch.object(evaluate_transfer, "load_cases", return_value=[case]), \
             patch.object(evaluate_transfer.transfer_metrics, "leak_guard", side_effect=RuntimeError("many-body hidden leak")), \
             patch.object(evaluate_transfer.engine, "ensure_idle"), \
             patch.object(evaluate_transfer.engine, "run_best_of_n") as mock_run:
            rc = evaluate_transfer.main(["--score", "--runs", "2"])
            self.assertEqual(rc, 1)
            mock_run.assert_not_called()

    def test_compare_prints_case_differences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "a.json")
            file_b = os.path.join(tmpdir, "b.json")
            with open(file_a, "w") as f:
                json.dump({"summary": {"f3": {"passed": 2, "of": 2}}, "cases": {"nqs": [{"found_rank": 1, "passed": True}]}}, f)
            with open(file_b, "w") as f:
                json.dump({"summary": {"f3": {"passed": 1, "of": 2}}, "cases": {"nqs": [{"found_rank": 4, "passed": False}]}}, f)

            with redirect_stdout(io.StringIO()) as out:
                rc = evaluate_transfer.main(["--compare", file_a, file_b])
                self.assertEqual(rc, 0)
                val = out.getvalue()
                self.assertIn("nqs", val)
                self.assertIn("Comparison", val)
