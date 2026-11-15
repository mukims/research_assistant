import io
import tempfile
from contextlib import redirect_stderr
import unittest
from unittest.mock import patch

from research_assistant import config
from research_assistant.transfer import engine, store
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

    def test_a_run_with_excluded_documents_saves_and_loads(self):
        r = engine.run_best_of_n("q", world()[0], TransferSettings(exclude_documents=frozenset({"s1", "n9"})))
        with tempfile.TemporaryDirectory() as d:
            loaded = store.load_result(store.save_result(r, directory=d))
        self.assertEqual(loaded.settings["exclude_documents"], ["n9", "s1"])

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

    def test_transfer_engine_run_clears_stale_cancel_after_idle_guard(self):
        deps, _ = world()
        engine.pipeline_status.request_cancel()
        self.assertTrue(engine.pipeline_status.is_cancel_requested())

        eng = engine.TransferEngine(deps=deps)
        result = eng.run("q")

        self.assertFalse(engine.pipeline_status.is_cancel_requested())
        self.assertEqual(len(result.transfers), 6)

    def test_transfer_engine_run_does_not_clear_cancel_when_busy(self):
        deps, _ = world()
        eng = engine.TransferEngine(deps=deps)
        with patch.object(engine.pipeline_status, "get_status", return_value={"active": True, "stage_label": "Fetching"}), \
             patch.object(engine.pipeline_status, "clear_cancel_request") as clear_mock:
            with self.assertRaises(engine.BusyError):
                eng.run("q")
        clear_mock.assert_not_called()

    def test_parallel_run_best_of_n_is_deterministic_and_merges_without_collision(self):
        r_seq = engine.run_best_of_n("q", world()[0], TransferSettings(max_workers=1))
        r_par = engine.run_best_of_n("q", world()[0], TransferSettings(max_workers=3))
        self.assertEqual([t.id for t in r_par.transfers], [t.id for t in r_seq.transfers])
        self.assertEqual(r_par.calls, r_seq.calls)
        self.assertEqual(sorted(r_par.passages.keys()), sorted(r_seq.passages.keys()))
        for t in r_par.transfers:
            for p in t.premises:
                self.assertIn(p.key, r_par.passages)

    def test_transfer_engine_run_tracks_transfer_stage_and_clears_on_exit(self):
        deps, _ = world()
        observed_stages = []
        orig_chat = deps.chat

        def spying_chat(msgs, temperature=0.7):
            observed_stages.append(engine.pipeline_status.get_status().get("stage"))
            return orig_chat(msgs, temperature)

        deps.chat = spying_chat
        eng = engine.TransferEngine(deps=deps)
        eng.run("q")
        self.assertTrue(observed_stages)
        self.assertTrue(all(st == "transfer" for st in observed_stages))
        self.assertEqual(engine.pipeline_status.get_status().get("stage"), "idle")

    def test_chunk_store_lance_and_sequence_adapters(self):
        from types import SimpleNamespace
        from research_assistant.transfer.chunk_store import LanceChunkStore, SequenceChunkStore, chunk_store_for

        # SequenceChunkStore
        texts = ["alpha", "beta", "gamma"]
        metas = [
            {"document": "d1.pdf", "seq": 0, "type": "text_chunk"},
            {"document": "d1.pdf", "seq": 1, "type": "text_chunk"},
            {"document": "d2.pdf", "seq": 0, "type": "figure", "section": "caption"},
        ]
        seq_store = chunk_store_for(texts, metas)
        self.assertIsInstance(seq_store, SequenceChunkStore)
        self.assertEqual(seq_store.get_chunk("d1.pdf", 1), (1, "beta", metas[1]))
        self.assertEqual(len(seq_store.get_document_chunks("d1.pdf")), 2)
        self.assertEqual(seq_store.get_neighbours("d1.pdf", "alpha", 0, window=1, position=0), ("", "beta"))
        self.assertEqual(seq_store.get_document_chunks("d2.pdf"), [])

        # LanceChunkStore delegates to table_adapter
        fake_adapter = SimpleNamespace(
            find_chunk_by_seq=lambda doc, seq, exclude_types=None: (42, "found", {"document": doc, "seq": seq}),
            find_document_chunks=lambda doc, exclude_types=None: [(5, "found", {"document": doc, "seq": 5})],
            get_neighbours=lambda doc, stored_form, seq, window=1, exclude_types=None: ("prev", "next"),
        )
        proxy_like = SimpleNamespace(table_adapter=fake_adapter)
        lance_store = chunk_store_for(proxy_like, proxy_like)
        self.assertIsInstance(lance_store, LanceChunkStore)
        self.assertEqual(lance_store.get_chunk("x.pdf", 5)[0], 42)
        self.assertEqual(lance_store.get_document_chunks("x.pdf")[0][0], 5)
        self.assertEqual(lance_store.get_neighbours("x.pdf", "found", 5, window=1), ("prev", "next"))
