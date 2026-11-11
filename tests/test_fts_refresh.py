"""After an ingest the full-text index is kept, not rebuilt from scratch.

A full rebuild over every chunk after each batch left an old index copy on
disk per batch for the 7-day version window (3.8 GB of _indices). LanceDB
already searches rows added after the index was built, and optimize() folds
them into it, so a batch only needs the index to exist.
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import lancedb

from research_assistant.shared import ingestion as ing
from research_assistant.shared.lance_store import LanceTableAdapter, get_papers_schema


def _rows(start, n, text="ordinary filler sentence"):
    return [{
        "chunk_id": f"chk_{i:024d}", "document": "doc1", "citation_source": "Some paper", "stored_form": "f",
        "section": "methods", "seq": i, "page": 1, "page_first": 1, "page_last": 1, "tier": "A",
        "text": f"{text} {i}", "content_hash": f"h{i}", "vector": [0.1, 0.2, 0.3, 0.4], "metadata_json": "{}",
    } for i in range(start, start + n)]


class TestEnsureFtsIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_fts_")
        self.tbl = lancedb.connect(self.tmp).create_table("papers", schema=get_papers_schema(dim=4))
        self.tbl.add(_rows(0, 50))
        self.adapter = LanceTableAdapter(self.tbl)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fts_columns(self):
        return sorted(c for i in self.tbl.list_indices() if i.index_type == "FTS" for c in i.columns)

    def test_a_missing_index_is_built(self):
        self.assertTrue(self.adapter.ensure_fts_index())
        self.assertEqual(self._fts_columns(), ["citation_source", "text"])

    def test_an_existing_index_is_not_rebuilt(self):
        self.adapter.ensure_fts_index()
        with patch.object(self.tbl, "create_index", wraps=self.tbl.create_index) as ci:
            self.assertFalse(self.adapter.ensure_fts_index())
        ci.assert_not_called()

    def test_rows_added_after_the_index_are_found(self):
        # The library behaviour this design leans on; re-check it on a lancedb upgrade.
        self.adapter.ensure_fts_index()
        self.tbl.add(_rows(50, 1, text="zebrafish magnetoresistance"))
        hits = self.tbl.search("magnetoresistance", query_type="fts").to_list()
        self.assertEqual([h["seq"] for h in hits], [50])


class TestRebuildBm25(unittest.TestCase):
    def _run(self, **kw):
        col = MagicMock()
        store = MagicMock()
        store.get_collection.return_value = col
        with patch.object(ing, "open_store", return_value=store), \
             patch("research_assistant.shared.db.clear_search_cache") as clear:
            ing.rebuild_bm25(**kw)
        return col, clear

    def test_after_an_ingest_it_only_makes_sure_the_index_exists(self):
        col, clear = self._run()
        col.ensure_fts_index.assert_called_once()
        col.create_fts_index.assert_not_called()
        clear.assert_called_once()

    def test_full_rebuild_on_request(self):
        col, clear = self._run(full=True)
        col.create_fts_index.assert_called_once_with("text", replace=True)
        clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
