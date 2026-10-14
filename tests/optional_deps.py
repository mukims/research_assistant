"""Skip decorators for tests that need what CI does not install.

CI installs requirements-test.txt only: no chromadb, no torch, and no corpus.
A test that needs one of those should skip there, not error, and an unguarded
`import chromadb` at the top of a test file aborts collection for the whole run.

    from tests.optional_deps import requires_real_store, requires_torch

requires_real_store also protects a developer's corpus: it is evaluated against
config.VECTORDB_PATH, so running the suite with CITATION_DATA_DIR pointed at a
scratch directory skips the tests that would otherwise open the real store.
"""

import importlib.util
import os
import unittest

from research_assistant import config


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def store_file_present(vectordb_path: str) -> bool:
    return os.path.isfile(os.path.join(vectordb_path, "chroma.sqlite3"))


requires_torch = unittest.skipUnless(
    available("torch"), "torch is not installed (CI installs requirements-test.txt only)"
)
requires_real_store = unittest.skipUnless(
    available("chromadb") and store_file_present(config.VECTORDB_PATH),
    "needs the maintainer's chromadb corpus at config.VECTORDB_PATH",
)
