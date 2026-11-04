# tests/test_db.py
"""Tests for LanceDB store opening, resource loading, caching, and embedder validation."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import research_assistant.shared.db as db


class TestOpenStore(unittest.TestCase):
    def test_open_store_returns_lance_client(self):
        with patch.object(db, "open_lance_store", return_value="lance_client"):
            client = db.open_store()
            self.assertEqual(client, "lance_client")

    def test_open_store_raises_store_unavailable_when_lancedb_missing(self):
        with patch.object(db, "lancedb", None):
            with self.assertRaises(db.StoreUnavailable) as ctx:
                db.open_store()
            self.assertIn("lancedb package is required", str(ctx.exception))

    def test_open_store_raises_store_unavailable_on_open_failure(self):
        with patch.object(db, "open_lance_store", side_effect=RuntimeError("disk unreadable")):
            with self.assertRaises(db.StoreUnavailable) as ctx:
                db.open_store()
            self.assertIn("Failed to open LanceDB store", str(ctx.exception))


class TestSearchResourceCache(unittest.TestCase):
    def setUp(self):
        db.clear_search_cache()
        self.addCleanup(db.clear_search_cache)

    def test_clear_forces_a_reload(self):
        table = MagicMock()
        table.name = db.LANCE_TABLE_NAME
        table.metadata = {"embedder": db.EMBED_MODEL}
        client = MagicMock()
        client.get_collection.return_value = table

        with patch.object(db, "open_store", return_value=client):
            first = db.load_search_resources()
            self.assertIs(first, db.load_search_resources())
            db.clear_search_cache()
            second = db.load_search_resources()
            self.assertIsNot(first, second)


class TestCheckEmbedder(unittest.TestCase):
    def test_mismatched_embedder_raises_runtime_error(self):
        table = MagicMock()
        table.name = "physics_papers_v3"
        table.metadata = {"embedder": "incompatible-model"}
        with self.assertRaisesRegex(RuntimeError, "CITATION_EMBED_MODEL"):
            db.check_embedder(table)

    def test_matching_embedder_passes(self):
        table = MagicMock()
        table.name = "physics_papers_v3"
        table.metadata = {"embedder": db.EMBED_MODEL}
        # Does not raise
        db.check_embedder(table)

    def test_non_dict_metadata_passes(self):
        table = MagicMock()
        table.metadata = None
        db.check_embedder(table)


class TestGetMaxChunkIndex(unittest.TestCase):
    def test_lancedb_collection_uses_count(self):
        col = MagicMock()
        col.is_lancedb = True
        col.count.return_value = 42
        self.assertEqual(db.get_max_chunk_index(col), 42)


if __name__ == "__main__":
    unittest.main()
