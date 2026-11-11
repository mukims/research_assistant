"""Post-ingest compaction runs only when the table is fragmented enough to pay for it.

Every 64-chunk add is a new fragment, and compaction merges them into the big
one, rewriting the whole table (~600 MB at 157k chunks). Compacting after every
ingest batch left ~36 dead copies (21 GB) waiting out the 7-day version window.
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import lancedb

from research_assistant.shared.lance_store import LanceTableAdapter, get_papers_schema


def _rows(start, n):
    return [{
        "chunk_id": f"chk_{i:024d}", "document": "doc1", "citation_source": "s", "stored_form": "f",
        "section": "methods", "seq": i, "page": 1, "page_first": 1, "page_last": 1, "tier": "A",
        "text": f"text {i}", "content_hash": f"h{i}", "vector": [0.1, 0.2, 0.3, 0.4], "metadata_json": "{}",
    } for i in range(start, start + n)]


class TestCompactIfFragmented(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_lance_compact_")
        self.tbl = lancedb.connect(self.tmp).create_table("papers", schema=get_papers_schema(dim=4))
        self.tbl.add(_rows(0, 100))                       # the big fragment
        self.adapter = LanceTableAdapter(self.tbl)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fragments(self):
        return self.tbl.stats()["fragment_stats"]["num_fragments"]

    def test_one_fragment_is_left_alone(self):
        self.assertFalse(self.adapter.compact_if_fragmented())
        self.assertEqual(self._fragments(), 1)

    def test_a_small_append_is_not_worth_a_rewrite(self):
        self.tbl.add(_rows(100, 5))                       # 5 of 105 rows outside the big fragment
        self.assertFalse(self.adapter.compact_if_fragmented(min_share=0.10))
        self.assertEqual(self._fragments(), 2)

    def test_enough_small_fragments_are_compacted(self):
        for start in (100, 106):
            self.tbl.add(_rows(start, 6))                 # 12 of 112 rows outside the big fragment
        self.assertTrue(self.adapter.compact_if_fragmented(min_share=0.10))
        self.assertEqual(self._fragments(), 1)
        self.assertEqual(self.adapter.count(), 112)

    def test_too_many_fragments_are_compacted_whatever_their_share(self):
        for start in (100, 101, 102):
            self.tbl.add(_rows(start, 1))
        self.assertTrue(self.adapter.compact_if_fragmented(min_share=0.5, max_fragments=4))
        self.assertEqual(self._fragments(), 1)

    def test_unreadable_stats_fall_back_to_optimize(self):
        table = MagicMock()
        table.name = "physics_papers_v3"
        table.stats.side_effect = RuntimeError("no stats")
        self.assertTrue(LanceTableAdapter(table).compact_if_fragmented())
        table.optimize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
