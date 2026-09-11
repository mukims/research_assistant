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
