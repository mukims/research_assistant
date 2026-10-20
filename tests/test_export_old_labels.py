"""Exporting the citation labels the old indexes hold, straight from Chroma's sqlite file.

The store belongs to a long-running app, so the reader must never write to it: it opens the file
read-only and immutable, refuses a store with a live write-ahead log, and a test pins that the file's
bytes are untouched and no sidecar file appears."""

import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts import export_old_labels as ex


def _make_store(directory, rows):
    """rows: (collection, document, citation_source). A Chroma-shaped sqlite file, as much of it as is read."""
    path = os.path.join(directory, "chroma.sqlite3")
    db = sqlite3.connect(path)
    db.executescript("""
        create table collections (id text primary key, name text not null);
        create table segments (id text primary key, type text, scope text, collection text);
        create table embeddings (id integer primary key, segment_id text not null, embedding_id text not null);
        create table embedding_metadata (id integer, key text not null, string_value text, primary key (id, key));
    """)
    colls, n = {}, 0
    for coll, document, label in rows:
        if coll not in colls:
            colls[coll] = f"c{len(colls)}"
            db.execute("insert into collections values (?, ?)", (colls[coll], coll))
            db.execute("insert into segments values (?, 'meta', 'METADATA', ?)", (f"s-{colls[coll]}", colls[coll]))
        n += 1
        db.execute("insert into embeddings values (?, ?, ?)", (n, f"s-{colls[coll]}", f"chunk_{n}"))
        db.execute("insert into embedding_metadata values (?, 'document', ?)", (n, document))
        if label is not None:
            db.execute("insert into embedding_metadata values (?, 'citation_source', ?)", (n, label))
    db.commit()
    db.close()
    return path


class TestMergeLabels(unittest.TestCase):
    def test_the_newer_collection_wins_and_a_disagreement_is_recorded(self):
        rows = [("physics_papers", "a.pdf", "Old A"), ("physics_papers_v2", "a.pdf", "New A"),
                ("physics_papers", "b.pdf", "Only B"), ("physics_papers_v2", "c.pdf", "Only C")]
        labels, conflicts = ex.merge_labels(rows, ["physics_papers_v2", "physics_papers"])
        self.assertEqual(labels, {"a.pdf": "New A", "b.pdf": "Only B", "c.pdf": "Only C"})
        self.assertEqual(conflicts, {"a.pdf": {"physics_papers_v2": "New A", "physics_papers": "Old A"}})

    def test_agreeing_collections_are_not_a_conflict_and_blank_labels_are_skipped(self):
        rows = [("physics_papers", "a.pdf", "Same"), ("physics_papers_v2", "a.pdf", "Same"),
                ("physics_papers", "b.pdf", None), ("physics_papers_v2", "b.pdf", "  ")]
        labels, conflicts = ex.merge_labels(rows, ["physics_papers_v2", "physics_papers"])
        self.assertEqual((labels, conflicts), ({"a.pdf": "Same"}, {}))


class TestReadLabels(unittest.TestCase):
    def test_reads_every_distinct_label_per_collection(self):
        with tempfile.TemporaryDirectory() as d:
            _make_store(d, [("physics_papers", "a.pdf", "A"), ("physics_papers", "a.pdf", "A"),
                            ("physics_papers_v2", "a.pdf", "A2"), ("physics_summaries", "a.pdf", "S")])
            rows = ex.read_labels(d, ["physics_papers", "physics_papers_v2"])
        self.assertEqual(sorted(rows), [("physics_papers", "a.pdf", "A"), ("physics_papers_v2", "a.pdf", "A2")])

    def test_the_store_is_left_byte_for_byte_alone_and_no_sidecar_file_appears(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_store(d, [("physics_papers", "a.pdf", "A")])
            before = hashlib.sha256(open(path, "rb").read()).hexdigest()
            before_files = sorted(os.listdir(d))
            ex.read_labels(d, ["physics_papers"])
            self.assertEqual(hashlib.sha256(open(path, "rb").read()).hexdigest(), before)
            self.assertEqual(sorted(os.listdir(d)), before_files)

    def test_a_live_write_ahead_log_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _make_store(d, [("physics_papers", "a.pdf", "A")])
            open(os.path.join(d, "chroma.sqlite3-wal"), "wb").write(b"x" * 64)
            with self.assertRaisesRegex(RuntimeError, "write-ahead log"):
                ex.read_labels(d, ["physics_papers"])

    def test_a_missing_store_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, "chroma.sqlite3"):
                ex.read_labels(d, ["physics_papers"])


if __name__ == "__main__":
    unittest.main()
