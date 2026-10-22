"""Tests never touch the maintainer's corpus or the Chroma server that owns it.

Every run gets its own empty data directory, so a test whose mocks fall through to
load_search_resources() opens an empty store, not the real one. Opening the real store
while the server owns it is what lost 17,201 vectors on 2026-10-21. The tests marked
requires_real_store run against the real corpus only with CITATION_TEST_REAL_STORE=1,
and then the server and the app must be stopped.

Set before any test module imports research_assistant.config.
"""

import atexit
import os
import shutil
import tempfile

if os.environ.get("CITATION_TEST_REAL_STORE") != "1" and "CITATION_DATA_DIR" not in os.environ:
    _data_dir = tempfile.mkdtemp(prefix="marvin-tests-")
    atexit.register(shutil.rmtree, _data_dir, True)
    os.environ["CITATION_DATA_DIR"] = _data_dir
os.environ.setdefault("CITATION_CHROMA_SERVER", "")
# The fixture world (tests/transfer_fixtures.py) is laid out for 3 cards per query
# and 3 techniques per bottleneck; the live defaults are 5.
os.environ.setdefault("CITATION_TRANSFER_CARDS_PER_QUERY", "3")
os.environ.setdefault("CITATION_TRANSFER_TECHNIQUES_PER_BOTTLENECK", "3")
