"""Tests for durable file replacement.

Every JSON manifest in the pipeline is a resume point: downloaded.json says
which references have already been pulled, seed_papers.json which queries have
been seeded, extracted_citations.json what Agent 2 still has to work through.
Each was written with a plain open(path, "w"), which truncates the previous
contents before the first byte of the new ones is written — so a crash, a full
disk or a Ctrl-C mid-dump left invalid JSON behind, and every reader of these
files treats invalid JSON as "start over".

shared.manifest already had the durable pattern (temp file in the same
directory, then os.replace). This is that pattern, extracted so the other
writers can share it.
"""

import json
import os
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import research_assistant.agents.agent0_discoverer as agent0
import research_assistant.agents.agent2_fetcher as agent2
import research_assistant.shared.ingestion as ing
from research_assistant.shared.atomic import atomic_write


class AtomicTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "state.json")

    def _write(self, text):
        with open(self.path, "w") as fh:
            fh.write(text)

    def _siblings(self):
        return sorted(os.listdir(self.tmp.name))


class TestAtomicWrite(AtomicTestCase):
    def test_writes_the_new_content(self):
        with atomic_write(self.path) as fh:
            fh.write("hello")
        with open(self.path) as fh:
            self.assertEqual(fh.read(), "hello")

    def test_failure_leaves_the_previous_content_intact(self):
        """The whole point: a half-written file must never reach the real path."""
        self._write('{"kept": true}')
        with self.assertRaises(ValueError):
            with atomic_write(self.path) as fh:
                fh.write('{"partial": ')
                raise ValueError("crash mid-write")
        with open(self.path) as fh:
            self.assertEqual(json.load(fh), {"kept": True})

    def test_failure_leaves_no_temp_file_behind(self):
        self._write("original")
        with self.assertRaises(ValueError):
            with atomic_write(self.path) as fh:
                fh.write("partial")
                raise ValueError("crash mid-write")
        self.assertEqual(self._siblings(), ["state.json"])

    def test_creates_the_parent_directory(self):
        nested = os.path.join(self.tmp.name, "a", "b", "state.json")
        with atomic_write(nested) as fh:
            fh.write("x")
        self.assertTrue(os.path.exists(nested))

    def test_binary_mode_round_trips(self):
        path = os.path.join(self.tmp.name, "index.pkl")
        with atomic_write(path, binary=True) as fh:
            pickle.dump({"a": 1}, fh)
        with open(path, "rb") as fh:
            self.assertEqual(pickle.load(fh), {"a": 1})


class TestFetcherCheckpoint(AtomicTestCase):
    """Agent 2 checkpoints after every paper so a crashed run resumes.

    A truncated downloaded.json is worse than no checkpoint at all: _load_state
    reports it unreadable and starts fresh, so every PDF downloads again.
    """

    def setUp(self):
        super().setUp()
        self.downloaded = os.path.join(self.tmp.name, "downloaded.json")
        self.failed = os.path.join(self.tmp.name, "failed.json")
        for target, path in [
            ("DOWNLOADED_JSON_PATH", self.downloaded),
            ("FAILED_DOWNLOADS_PATH", self.failed),
        ]:
            patcher = patch.object(agent2, target, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_checkpoint_survives_a_crash_mid_write(self):
        agent2._checkpoint({"doi:10.1/a": {"path": "a.pdf"}}, {})

        # A set is not JSON-serialisable: json.dump writes the opening bytes
        # and then raises, which is exactly the shape of a crash mid-dump.
        with self.assertRaises(TypeError):
            agent2._checkpoint({"doi:10.1/b": {"paths": {"unserialisable"}}}, {})

        self.assertEqual(
            agent2._load_state(self.downloaded, {}),
            {"doi:10.1/a": {"path": "a.pdf"}},
            "the previous checkpoint was destroyed — the run restarts from zero",
        )


class TestSeedManifest(AtomicTestCase):
    def setUp(self):
        super().setUp()
        self.seeds = os.path.join(self.tmp.name, "seed_papers.json")
        patcher = patch.object(agent0, "SEED_PAPERS_PATH", self.seeds)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_seed_manifest_survives_a_crash_mid_write(self):
        agent0._save_seeds({"graphene": {"key": "arxiv:1"}})

        with self.assertRaises(TypeError):
            agent0._save_seeds({"graphene": {"key": {"unserialisable"}}})

        self.assertEqual(agent0._load_seeds(), {"graphene": {"key": "arxiv:1"}})


class TestBm25Rebuild(AtomicTestCase):
    """A truncated BM25 pickle takes down every search path at once.

    load_search_resources() unpickles it on every query, so a partial write is
    not a degraded index — it is an exception on the next search, and the only
    documented recovery is re-ingesting the corpus.
    """

    def setUp(self):
        super().setUp()
        self.index = os.path.join(self.tmp.name, "bm25.pkl")
        patcher = patch.object(ing, "BM25_INDEX_PATH", self.index)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rebuild_leaves_the_old_index_loadable_when_it_fails(self):
        # The stored index is only ever data this pipeline pickled itself.
        with open(self.index, "wb") as fh:
            pickle.dump("previous index", fh)

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("disk full mid-dump")

        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = lambda path: types.SimpleNamespace(
            get_collection=lambda name: types.SimpleNamespace(
                get=lambda **kw: {"ids": [], "documents": []}
            )
        )
        bm25_mod = types.ModuleType("rank_bm25")
        bm25_mod.BM25Okapi = lambda corpus: Unpicklable()

        with patch.dict(sys.modules, {"chromadb": chroma, "rank_bm25": bm25_mod}):
            with self.assertRaises(RuntimeError):
                ing.rebuild_bm25()

        with open(self.index, "rb") as fh:
            self.assertEqual(pickle.load(fh), "previous index")


if __name__ == "__main__":
    unittest.main()
