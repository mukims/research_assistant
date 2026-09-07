"""Tests for Agent 0's seed manifest, which now routes through SeedPaper.

The one thing actually changed by the port: seeds[query] used to be a bare
dict any caller could grow arbitrary keys into (the old **extra splat on
_download_and_record was a direct route for that). Routing both manifest
sites through SeedPaper(...).to_dict() means a malformed record fails at
write time, and the manifest a caller like orchestrate.py reads back is
always a valid SeedPaper shape.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.agents.agent0_discoverer as a0
from research_assistant.schemas import SeedPaper


class TestDownloadAndRecordSeedPaperWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir_patch = patch.object(a0, "RAW_DIR", self.tmp.name)
        self.seed_path_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.raw_dir_patch.start()
        self.seed_path_patch.start()
        self.addCleanup(self.raw_dir_patch.stop)
        self.addCleanup(self.seed_path_patch.stop)

    def test_manifest_record_round_trips_through_seed_paper(self):
        with patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0._download_and_record(
                "some query", "arxiv:1234.5678", "https://arxiv.org/pdf/1234.5678",
                {}, force=False, arxiv_id="1234.5678", source="manual-url",
            )
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        record = seeds["some query"]
        # Must be exactly the shape SeedPaper produces, not a hand-rolled dict:
        # a malformed record would raise here instead of round-tripping.
        self.assertEqual(SeedPaper.from_dict(record).to_dict(), record)
        self.assertEqual(record["source"], "manual-url")
        self.assertEqual(record["arxiv_id"], "1234.5678")

    def test_rejects_undeclared_keyword_arguments(self):
        """The **extra splat is gone — an unknown kwarg must now be a TypeError,
        not a key silently smuggled into the manifest."""
        with patch.object(a0, "download_pdf", return_value=(True, None)):
            with self.assertRaises(TypeError):
                a0._download_and_record(
                    "q", "k", "https://example.com/p.pdf", {}, False,
                    unexpected_field="sneaky",
                )


if __name__ == "__main__":
    unittest.main()
