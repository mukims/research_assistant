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
import pathlib
import sqlite3
import unittest

from research_assistant import config


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def store_has_data(vectordb_path: str) -> bool:
    """True only for a chroma store that holds embeddings. A file is not enough: with chromadb
    installed, importing app.py creates an empty store at config.VECTORDB_PATH as a side effect.
    Read-only sqlite, never a chroma client, so checking cannot create or migrate anything."""
    database = os.path.join(vectordb_path, "chroma.sqlite3")
    if not os.path.isfile(database):
        return False
    try:
        connection = sqlite3.connect(f"{pathlib.Path(database).as_uri()}?mode=ro", uri=True)
        try:
            return connection.execute("select count(*) from embeddings").fetchone()[0] > 0
        finally:
            connection.close()
    except sqlite3.Error:
        return False


requires_torch = unittest.skipUnless(
    available("torch"), "torch is not installed (CI installs requirements-test.txt only)"
)
requires_real_store = unittest.skipUnless(
    available("chromadb") and store_has_data(config.VECTORDB_PATH),
    "needs the maintainer's chromadb corpus at config.VECTORDB_PATH",
)
