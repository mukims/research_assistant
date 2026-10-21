import io
from contextlib import redirect_stderr
import unittest
from unittest.mock import patch

from research_assistant import config
from research_assistant.transfer import engine
from research_assistant.transfer.deps import TransferSettings
from tests.transfer_fixtures import world


class TestEngine(unittest.TestCase):
    def test_two_bottlenecks_times_three_cards_make_six_ranked_transfers(self):
        r = engine.run_best_of_n("q", world()[0])
        self.assertEqual(sorted(t.id for t in r.transfers), ["B1.T1", "B1.T2", "B1.T3", "B2.T1", "B2.T2", "B2.T3"])
        self.assertTrue(all(t.scores.set_aside is None for t in r.transfers))

    def test_calls_are_counted_by_kind(self):
        r = engine.run_best_of_n("q", world()[0])
        self.assertEqual((r.calls["llm"], r.calls["critic"]), (14, 6))      # gate + bottlenecks + 6 * (write + revise)

    def test_an_excluded_document_is_never_used_anywhere(self):
        deps, log = world()
        r = engine.run_best_of_n("q", deps, TransferSettings(exclude_documents=frozenset({"s1"})))
        self.assertNotIn("s1", {t.card_document for t in r.transfers})
        self.assertTrue(all("s1" in (excl or set()) for _, excl in log.searches))

    def test_done_before_is_ranked_last_with_its_reason(self):
        r = engine.run_best_of_n("q", world(card_names={"s2": "known method"})[0])
        self.assertEqual((r.transfers[-1].card_document, r.transfers[-1].scores.set_aside), ("s2", "done_before"))

    def test_progress_reaches_its_total(self):
        deps, log = world()
        engine.run_best_of_n("q", deps)
        self.assertEqual(log.progress[-1][:2], (8, 8))

    def test_ensure_idle_refuses_while_another_run_is_active(self):
        with patch.object(engine.pipeline_status, "get_status", return_value={"active": True, "stage_label": "Ingesting and summarizing papers"}):
            with self.assertRaisesRegex(engine.BusyError, "Ingesting"):
                engine.ensure_idle()

    def test_every_live_search_excludes_figures(self):
        with patch.object(engine, "hybrid_search", return_value=[]) as hs:
            engine.make_search(("col", "bm25", ["t"], [{}]))("q", 3, {"a.pdf"}, {"x.pdf"})
        kw = hs.call_args.kwargs
        self.assertEqual((kw["exclude_types"], kw["doc_filter"], kw["exclude_docs"]), (config.EXCLUDED_FROM_EVIDENCE, {"a.pdf"}, {"x.pdf"}))

    def test_estimate_calls_for_three_by_three(self):
        self.assertEqual(
            engine.estimate_calls(TransferSettings(bottlenecks=3, techniques_per_bottleneck=3)),
            {"llm": 20, "critic": 9, "judge_low": 21, "judge_high": 45},
        )
