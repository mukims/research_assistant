"""The skip conditions that keep CI green without the heavy stack.

CI installs requirements-test.txt only: no torch, and no corpus.
tests/optional_deps.py decides which tests are skipped there instead of erroring.
"""

import os
import tempfile
import unittest

from tests import optional_deps
from research_assistant import config


class TestAvailable(unittest.TestCase):
    def test_reports_an_installed_module(self):
        self.assertTrue(optional_deps.available("os"))

    def test_reports_a_missing_module(self):
        self.assertFalse(optional_deps.available("no_such_module_for_this_test"))


class TestStoreHasData(unittest.TestCase):
    @staticmethod
    def _store(folder, has_data):
        table_dir = os.path.join(folder, f"{config.COLLECTION_NAME}.lance")
        if has_data:
            os.makedirs(table_dir, exist_ok=True)
            with open(os.path.join(table_dir, "data.lance"), "wb") as f:
                f.write(b"data")

    def test_a_store_with_data_has_data(self):
        with tempfile.TemporaryDirectory() as folder:
            self._store(folder, has_data=True)
            self.assertTrue(optional_deps.store_has_data(folder))

    def test_an_empty_store_has_no_data(self):
        with tempfile.TemporaryDirectory() as folder:
            self._store(folder, has_data=False)
            self.assertFalse(optional_deps.store_has_data(folder))

    def test_a_missing_directory_has_no_data(self):
        self.assertFalse(optional_deps.store_has_data(os.path.join(tempfile.gettempdir(), "no_such_store_dir")))


if __name__ == "__main__":
    unittest.main()
