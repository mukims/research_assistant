"""Tests for Agent 3 skipping abstract-only records already in the store."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.agents import agent3_ingestor
from research_assistant.shared.db import StoreUnavailable
from research_assistant.shared.paper_index import PaperEntry, PaperIndex


class FakeCollection:
    def __init__(self, records=None):
        self.records = records or []

    def get(self, where=None, limit=None, offset=0, include=None):
        offset = offset or 0
        limit = limit or len(self.records)
        sub = self.records
        if where and "type" in where:
            target_type = where["type"]
            sub = [
                r for r in sub
                if r.get("type") == target_type or r.get("stored_form") == target_type
            ]
        sliced = sub[offset : offset + limit]
        return {"metadatas": sliced}


class TestAgent3AbstractDedupe(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.downloaded_path = os.path.join(self.temp_dir.name, "downloaded.json")
        self.manifest_path = os.path.join(self.temp_dir.name, "ingested_v3.json")
        self.index_path = os.path.join(self.temp_dir.name, "paper_index.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_downloaded(self, data: dict):
        with open(self.downloaded_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_stub_matching_ingested_pdf_title_is_skipped(self):
        title = "A Comprehensive Survey of Deep Learning for Graphs"
        # Index has PDF p.pdf titled T and in the manifest
        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="p.pdf", path="/path/p.pdf", title=title))
        idx.save()

        manifest_data = {"p.pdf": "2026-11-06T00:00:00"}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        # Record doi:x titled T in downloaded.json
        self._write_downloaded({
            "doi:10.1000/x": {
                "key": "doi:10.1000/x",
                "evidence_tier": "abstract_only",
                "title": title,
                "abstract": "Deep learning on graphs has emerged as a powerful tool...",
            }
        })

        mock_store = MagicMock()
        mock_store.get_collection.return_value = FakeCollection()

        with patch("research_assistant.agents.agent3_ingestor.DOWNLOADED_JSON_PATH", self.downloaded_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.shared.ingestion.upsert_corpus") as mock_upsert:
            agent3_ingestor.run_ingestor()

            # upsert_corpus is NOT called with any doi:10.1000/x entry
            if mock_upsert.called:
                corpus = mock_upsert.call_args[0][0]
                docs = {entry["document"] for entry in corpus}
                self.assertNotIn("doi:10.1000/x", docs)
            else:
                self.assertFalse(mock_upsert.called)

    def test_stub_matching_ingested_stub_is_skipped(self):
        title = "A Comprehensive Survey of Deep Learning for Graphs"
        # The store has an abstract-only row doi:x / citation_source T
        fake_col = FakeCollection([
            {
                "document": "doi:10.1000/x",
                "citation_source": title,
                "type": "abstract_only",
            }
        ])
        manifest_data = {"doi:10.1000/x": "2026-11-06T00:00:00"}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        idx = PaperIndex(self.index_path)
        idx.save()

        # Record raw:y titled T
        self._write_downloaded({
            "raw:y": {
                "key": "raw:y",
                "evidence_tier": "abstract_only",
                "title": title,
                "abstract": "Another abstract on graph representation...",
            }
        })

        mock_store = MagicMock()
        mock_store.get_collection.return_value = fake_col

        with patch("research_assistant.agents.agent3_ingestor.DOWNLOADED_JSON_PATH", self.downloaded_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.shared.ingestion.upsert_corpus") as mock_upsert:
            agent3_ingestor.run_ingestor()

            if mock_upsert.called:
                corpus = mock_upsert.call_args[0][0]
                docs = {entry["document"] for entry in corpus}
                self.assertNotIn("raw:y", docs)
            else:
                self.assertFalse(mock_upsert.called)

    def test_two_new_stubs_same_title_only_first_ingested(self):
        title = "A Comprehensive Survey of Deep Learning for Graphs"
        manifest_data = {}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        idx = PaperIndex(self.index_path)
        idx.save()

        self._write_downloaded({
            "doi:10.1000/first": {
                "key": "doi:10.1000/first",
                "evidence_tier": "abstract_only",
                "title": title,
                "abstract": "First abstract...",
            },
            "doi:10.1000/second": {
                "key": "doi:10.1000/second",
                "evidence_tier": "abstract_only",
                "title": title,
                "abstract": "Second abstract...",
            },
        })

        mock_store = MagicMock()
        mock_store.get_collection.return_value = FakeCollection()

        with patch("research_assistant.agents.agent3_ingestor.DOWNLOADED_JSON_PATH", self.downloaded_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.shared.ingestion.upsert_corpus") as mock_upsert:
            agent3_ingestor.run_ingestor()

            self.assertTrue(mock_upsert.called)
            corpus = mock_upsert.call_args[0][0]
            docs = {entry["document"] for entry in corpus}
            self.assertIn("doi:10.1000/first", docs)
            self.assertNotIn("doi:10.1000/second", docs)

    def test_stub_with_short_title_is_ingested(self):
        # A record titled "Results" (< TITLE_MIN) is ingested as today
        manifest_data = {}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        idx = PaperIndex(self.index_path)
        idx.save()

        self._write_downloaded({
            "doi:10.1000/short": {
                "key": "doi:10.1000/short",
                "evidence_tier": "abstract_only",
                "title": "Results",
                "abstract": "Brief results abstract...",
            }
        })

        mock_store = MagicMock()
        mock_store.get_collection.return_value = FakeCollection()

        with patch("research_assistant.agents.agent3_ingestor.DOWNLOADED_JSON_PATH", self.downloaded_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.shared.ingestion.upsert_corpus") as mock_upsert:
            agent3_ingestor.run_ingestor()

            self.assertTrue(mock_upsert.called)
            corpus = mock_upsert.call_args[0][0]
            docs = {entry["document"] for entry in corpus}
            self.assertIn("doi:10.1000/short", docs)

    def test_store_unavailable_ingests_as_before(self):
        # The store raises StoreUnavailable -> the record is ingested
        manifest_data = {}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        self._write_downloaded({
            "doi:10.1000/fallback": {
                "key": "doi:10.1000/fallback",
                "evidence_tier": "abstract_only",
                "title": "A Very Long Paper Title For Fallback Testing",
                "abstract": "Fallback abstract...",
            }
        })

        with patch("research_assistant.agents.agent3_ingestor.DOWNLOADED_JSON_PATH", self.downloaded_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.db.open_store", side_effect=StoreUnavailable("db locked")), \
             patch("research_assistant.shared.ingestion.upsert_corpus") as mock_upsert:
            agent3_ingestor.run_ingestor()

            self.assertTrue(mock_upsert.called)
            corpus = mock_upsert.call_args[0][0]
            docs = {entry["document"] for entry in corpus}
            self.assertIn("doi:10.1000/fallback", docs)

    def test_force_reingests_stub_already_in_store(self):
        title = "A Comprehensive Survey of Deep Learning for Graphs"
        # The store has an abstract-only row doi:x / citation_source T
        fake_col = FakeCollection([
            {
                "document": "doi:10.1000/x",
                "citation_source": title,
                "type": "abstract_only",
            }
        ])
        manifest_data = {"doi:10.1000/x": "2026-11-06T00:00:00"}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        idx = PaperIndex(self.index_path)
        idx.save()

        # Record doi:x titled T in downloaded.json
        self._write_downloaded({
            "doi:10.1000/x": {
                "key": "doi:10.1000/x",
                "evidence_tier": "abstract_only",
                "title": title,
                "abstract": "Updated or forced abstract...",
            }
        })

        mock_store = MagicMock()
        mock_store.get_collection.return_value = fake_col

        with patch("research_assistant.agents.agent3_ingestor.DOWNLOADED_JSON_PATH", self.downloaded_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.shared.ingestion.upsert_corpus") as mock_upsert:
            agent3_ingestor.run_ingestor(force=True)

            self.assertTrue(mock_upsert.called)
            corpus = mock_upsert.call_args[0][0]
            docs = {entry["document"] for entry in corpus}
            self.assertIn("doi:10.1000/x", docs)


if __name__ == "__main__":
    unittest.main()
