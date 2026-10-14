"""The skip conditions that keep CI green without the heavy stack.

CI installs requirements-test.txt only: no chromadb, no torch, and no corpus.
tests/optional_deps.py decides which tests are skipped there instead of erroring.
"""

import os
import tempfile
import unittest

from tests import optional_deps


class TestAvailable(unittest.TestCase):
    def test_reports_an_installed_module(self):
        self.assertTrue(optional_deps.available("os"))

    def test_reports_a_missing_module(self):
        self.assertFalse(optional_deps.available("no_such_module_for_this_test"))


class TestStoreFilePresent(unittest.TestCase):
    def test_needs_the_chroma_database_file(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(optional_deps.store_file_present(folder))
            open(os.path.join(folder, "chroma.sqlite3"), "wb").close()
            self.assertTrue(optional_deps.store_file_present(folder))

    def test_a_missing_directory_is_not_a_store(self):
        self.assertFalse(optional_deps.store_file_present(os.path.join(tempfile.gettempdir(), "no_such_store_dir")))


if __name__ == "__main__":
    unittest.main()
