"""The skip conditions that keep CI green without the heavy stack.

CI installs requirements-test.txt only: no chromadb, no torch, and no corpus.
tests/optional_deps.py decides which tests are skipped there instead of erroring.
"""

import os
import sqlite3
import tempfile
import unittest

from tests import optional_deps


class TestAvailable(unittest.TestCase):
    def test_reports_an_installed_module(self):
        self.assertTrue(optional_deps.available("os"))

    def test_reports_a_missing_module(self):
        self.assertFalse(optional_deps.available("no_such_module_for_this_test"))


class TestStoreHasData(unittest.TestCase):
    """A file on disk is not enough: with chromadb installed, importing app.py creates an
    empty store at config.VECTORDB_PATH as a side effect, and an earlier version of the
    guard mistook that for the maintainer's corpus and let the real one be opened."""

    @staticmethod
    def _store(folder, rows):
        con = sqlite3.connect(os.path.join(folder, "chroma.sqlite3"))
        con.execute("create table embeddings (id integer primary key)")
        con.executemany("insert into embeddings (id) values (?)", [(i,) for i in range(rows)])
        con.commit()
        con.close()

    def test_a_store_with_embeddings_has_data(self):
        with tempfile.TemporaryDirectory() as folder:
            self._store(folder, rows=3)
            self.assertTrue(optional_deps.store_has_data(folder))

    def test_an_empty_store_has_no_data(self):
        with tempfile.TemporaryDirectory() as folder:
            self._store(folder, rows=0)
            self.assertFalse(optional_deps.store_has_data(folder))

    def test_a_file_that_is_not_a_chroma_database_has_no_data(self):
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, "chroma.sqlite3"), "wb").close()
            self.assertFalse(optional_deps.store_has_data(folder))

    def test_a_missing_directory_has_no_data(self):
        self.assertFalse(optional_deps.store_has_data(os.path.join(tempfile.gettempdir(), "no_such_store_dir")))

    def test_checking_does_not_modify_the_store(self):
        with tempfile.TemporaryDirectory() as folder:
            self._store(folder, rows=3)
            path = os.path.join(folder, "chroma.sqlite3")
            before = (os.path.getsize(path), os.path.getmtime(path))
            optional_deps.store_has_data(folder)
            self.assertEqual((os.path.getsize(path), os.path.getmtime(path)), before)


if __name__ == "__main__":
    unittest.main()
