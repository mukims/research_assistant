"""CITATION_INDEX_VERSION selects every on-disk name the index uses.

Checked in a subprocess: config.py is imported by most of the package, so
reloading it in-process would change module attributes other tests already
bound. A fresh interpreter per case is the honest way to read it.
"""

import json
import os
import subprocess
import sys
import unittest

_PROBE = """
import json
from research_assistant import config as c
print(json.dumps({
    "v": c.INDEX_VERSION,
    "chunks": c.COLLECTION_NAME,
    "summaries": c.SUMMARY_COLLECTION_NAME,
    "bm25": c.BM25_INDEX_PATH.rsplit("/", 1)[-1],
    "manifest": c.INGESTED_MANIFEST_PATH.rsplit("/", 1)[-1],
    "target": c.CHUNK_TARGET_CHARS,
    "max": c.CHUNK_MAX_CHARS,
    "tei_dir": c.GROBID_TEI_DIR.rsplit("/", 1)[-1],
}))
"""


def _probe(**env):
    # Start from a copy of the environment with every CITATION_* knob these
    # tests set removed, so the outer shell cannot leak a value into the
    # "default" case; then apply this case's values.
    full = {k: v for k, v in os.environ.items()
            if k not in ("CITATION_INDEX_VERSION", "CITATION_CHUNK_TARGET_CHARS", "CITATION_CHUNK_MAX_CHARS")}
    full.update({"CITATION_LOG_FILE": "0", **env})
    out = subprocess.run([sys.executable, "-c", _PROBE], env=full, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestIndexVersionNames(unittest.TestCase):
    def test_default_is_v1_with_the_existing_names(self):
        got = _probe()
        self.assertEqual(got["v"], 1)
        self.assertEqual(got["chunks"], "physics_papers")
        self.assertEqual(got["summaries"], "physics_summaries")
        self.assertEqual(got["bm25"], "bm25_index.pkl")
        self.assertEqual(got["manifest"], "ingested.json")

    def test_v2_suffixes_every_name(self):
        got = _probe(CITATION_INDEX_VERSION="2")
        self.assertEqual(got["v"], 2)
        self.assertEqual(got["chunks"], "physics_papers_v2")
        self.assertEqual(got["summaries"], "physics_summaries_v2")
        self.assertEqual(got["bm25"], "bm25_index_v2.pkl")
        self.assertEqual(got["manifest"], "ingested_v2.json")

    def test_chunk_sizes_have_spec_defaults_and_are_overridable(self):
        got = _probe()
        self.assertEqual((got["target"], got["max"]), (1200, 1800))
        got = _probe(CITATION_CHUNK_TARGET_CHARS="900", CITATION_CHUNK_MAX_CHARS="1400")
        self.assertEqual((got["target"], got["max"]), (900, 1400))

    def test_tei_cache_dir_is_the_one_agent1_uses(self):
        self.assertEqual(_probe()["tei_dir"], "grobid_output")
