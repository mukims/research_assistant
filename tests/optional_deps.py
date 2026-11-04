"""Skip decorators for tests that need what CI does not install.

CI installs requirements-test.txt only: no torch, and no corpus.
A test that needs one of those should skip there, not error, and an unguarded
import at the top of a test file aborts collection for the whole run.

    from tests.optional_deps import requires_real_store, requires_torch

requires_real_store also protects a developer's corpus: it is evaluated against
config.LANCEDB_PATH, so running the suite with CITATION_DATA_DIR pointed at a
scratch directory skips the tests that would otherwise open the real store.
"""

import importlib.util
import os
import unittest

from research_assistant import config


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def store_has_data(lancedb_path: str) -> bool:
    """True only for a lance store that holds data."""
    if not os.path.isdir(lancedb_path):
        return False
    table_dir = os.path.join(lancedb_path, f"{config.COLLECTION_NAME}.lance")
    if not os.path.isdir(table_dir):
        return False
    try:
        return len(os.listdir(table_dir)) > 0
    except OSError:
        return False


requires_torch = unittest.skipUnless(
    available("torch"), "torch is not installed (CI installs requirements-test.txt only)"
)

requires_real_store = unittest.skipUnless(
    os.environ.get("CITATION_TEST_REAL_STORE") == "1"
    and available("lancedb") and store_has_data(config.LANCEDB_PATH),
    "needs the maintainer's LanceDB corpus at config.LANCEDB_PATH and CITATION_TEST_REAL_STORE=1",
)
