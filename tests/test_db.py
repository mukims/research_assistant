# tests/test_db.py
"""The BM25 pickle carries its tokenizer version, and both formats load.

A pickle built by tokenize() searched with legacy_tokenize() silently loses
most keyword recall — the stems don't match the raw words. So the pickle
says which built it, the loader tags the object, and search reads the tag.
"""

import os
import pickle
import sys
import tempfile
import time
import types

import unittest
from unittest.mock import patch

import research_assistant.shared.db as db


class _Bm25Stub:
    def get_scores(self, q):
        return [0.0]


class _FakeCollection:
    def get(self, include=None, limit=None, offset=0):
        if offset:
            return {"ids": [], "documents": [], "metadatas": []}
        return {"ids": ["chunk_0"], "documents": ["x"], "metadatas": [{"document": "a.pdf"}]}


class LoadTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkl = os.path.join(self.tmp.name, "bm25.pkl")
        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = lambda path: types.SimpleNamespace(
            get_collection=lambda name: _FakeCollection()
        )
        p = patch.dict(sys.modules, {"chromadb": chroma}); p.start(); self.addCleanup(p.stop)
        p = patch.object(db, "BM25_INDEX_PATH", self.pkl); p.start(); self.addCleanup(p.stop)

    def test_legacy_bare_pickle_is_tagged_v1(self):
        with open(self.pkl, "wb") as f:
            pickle.dump(_Bm25Stub(), f)
        _, bm25, texts, _ = db.load_search_resources()
        self.assertEqual(bm25.tokenizer_version, "v1")
        self.assertEqual(texts, ["x"])

    def test_v2_pickle_is_unwrapped_and_tagged(self):
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v2", "built_at": "now", "bm25": _Bm25Stub()}, f)
        _, bm25, _, _ = db.load_search_resources()
        self.assertEqual(bm25.tokenizer_version, "v2")
        self.assertTrue(hasattr(bm25, "get_scores"))

    def test_unknown_tokenizer_warns_but_loads(self):
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v9", "built_at": "now", "bm25": _Bm25Stub()}, f)
        with self.assertLogs("db", level="WARNING") as cm:
            _, bm25, _, _ = db.load_search_resources()
        self.assertEqual(bm25.tokenizer_version, "v9")
        self.assertTrue(any("tokenizer" in line for line in cm.output))


class TestSearchResourceCache(LoadTestCase):
    def setUp(self):
        super().setUp()
        db.clear_search_cache()
        self.addCleanup(db.clear_search_cache)
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v2", "built_at": "now", "bm25": _Bm25Stub()}, f)

    def test_second_call_returns_the_same_objects_without_reloading(self):
        first = db.load_search_resources()
        with patch.object(db, "_load_search_resources_uncached", side_effect=AssertionError("reloaded")):
            second = db.load_search_resources()
        self.assertIs(first, second)

    def test_a_rebuilt_pickle_invalidates(self):
        first = db.load_search_resources()
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v2", "built_at": "later", "bm25": _Bm25Stub()}, f)
        os.utime(self.pkl, (time.time() + 5, time.time() + 5))
        second = db.load_search_resources()
        self.assertIsNot(first, second)

    def test_clear_forces_a_reload(self):
        first = db.load_search_resources()
        db.clear_search_cache()
        self.assertIsNot(first, db.load_search_resources())



class OpenStoreTestCase(unittest.TestCase):
    """Every process reaches the store through open_store(). The local store is not safe across
    processes: on 2026-10-21 one process saved its older copy of the vector index over another's
    writes and 17,201 vectors were lost. A Chroma server owns the store; the files are opened
    directly only for tests, scratch trials and offline repair."""

    def _chroma(self, heartbeat=lambda: 1):
        calls = []
        chroma = types.ModuleType("chromadb")

        def http(**kwargs):
            calls.append(("http", kwargs))
            return types.SimpleNamespace(heartbeat=heartbeat)

        chroma.HttpClient = http
        chroma.PersistentClient = lambda path: calls.append(("persistent", path)) or "local"
        p = patch.dict(sys.modules, {"chromadb": chroma}); p.start(); self.addCleanup(p.stop)
        return calls

    def test_a_configured_server_is_used_over_http(self):
        calls = self._chroma()
        with patch.object(db, "CHROMA_SERVER", "http://localhost:8000"):
            client = db.open_store()
        self.assertEqual(calls, [("http", {"host": "localhost", "port": 8000, "ssl": False})])
        self.assertTrue(hasattr(client, "heartbeat"))

    def test_a_server_that_does_not_answer_says_how_to_start_it(self):
        def down():
            raise ConnectionError("refused")

        self._chroma(heartbeat=down)
        with patch.object(db, "CHROMA_SERVER", "http://localhost:8000"):
            with self.assertRaises(db.StoreUnavailable) as ctx:
                db.open_store()
        self.assertIn("http://localhost:8000", str(ctx.exception))
        self.assertIn(f"chroma run --path {db.VECTORDB_PATH} --port 8000", str(ctx.exception))

    def test_no_server_opens_the_files(self):
        calls = self._chroma()
        with patch.object(db, "CHROMA_SERVER", ""):
            self.assertEqual(db.open_store(), "local")
        self.assertEqual(calls, [("persistent", db.VECTORDB_PATH)])


class ChromaServerSettingTestCase(unittest.TestCase):
    def test_the_server_is_the_default(self):
        from research_assistant.config import chroma_server_from

        self.assertEqual(chroma_server_from({}), "http://localhost:8000")

    def test_a_scratch_data_dir_never_reaches_the_real_store_through_the_server(self):
        from research_assistant.config import chroma_server_from

        self.assertEqual(chroma_server_from({"CITATION_DATA_DIR": "/scratch"}), "")

    def test_an_explicit_setting_wins(self):
        from research_assistant.config import chroma_server_from

        self.assertEqual(chroma_server_from({"CITATION_CHROMA_SERVER": "http://h:9"}), "http://h:9")
        self.assertEqual(
            chroma_server_from({"CITATION_DATA_DIR": "/scratch", "CITATION_CHROMA_SERVER": "http://h:9"}),
            "http://h:9",
        )
        self.assertEqual(chroma_server_from({"CITATION_CHROMA_SERVER": ""}), "")
