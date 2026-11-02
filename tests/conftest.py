"""Tests never touch the maintainer's corpus, the network, or each other's state.

Every run gets its own empty data directory, so a test whose mocks fall through to
load_search_resources() opens an empty store, not the real one. Opening the real store
while the server owns it is what lost 17,201 vectors on 2026-10-21. The tests marked
requires_real_store run against the real corpus only with CITATION_TEST_REAL_STORE=1,
and then the server and the app must be stopped.

Hermetic by construction (2026-11-02): a live GEMINI_API_KEY in a gitignored `.ENV` (a
case-insensitive filesystem serves it as `.env`) used to switch the suite to the Gemini
backend and spend real money, and planner tests reached a live Ollama on :11434. So:
no .env files, no API keys, every provider URL points at a closed port, TCP is blocked
per test (opt out with @pytest.mark.allow_network), and the temp data/workspace dirs
and in-process caches are reset after every test.

Set before any test module imports research_assistant.config.
"""

import atexit
import os
import shutil
import socket
import sys
import tempfile

import pytest

# TEST_DATA_DIR is the directory this module created with mkdtemp for DATA_DIR, or None
# when it did not (CITATION_TEST_REAL_STORE=1, or CITATION_DATA_DIR was already set -- then
# DATA_DIR may be the real corpus). _wipe_contents refuses every path but the dirs created
# here, so it cannot delete anything it did not make.
TEST_DATA_DIR = None
if os.environ.get("CITATION_TEST_REAL_STORE") != "1" and "CITATION_DATA_DIR" not in os.environ:
    TEST_DATA_DIR = tempfile.mkdtemp(prefix="marvin-tests-")
    atexit.register(shutil.rmtree, TEST_DATA_DIR, True)
    os.environ["CITATION_DATA_DIR"] = TEST_DATA_DIR

_WORKSPACE_DIR = tempfile.mkdtemp(prefix="marvin-tests-workspace-")
atexit.register(shutil.rmtree, _WORKSPACE_DIR, True)
os.environ["CITATION_WORKSPACE_DIR"] = _WORKSPACE_DIR

# Keys and .env files never reach the test process (see config.load_env_files).
os.environ["CITATION_DISABLE_DOTENV"] = "1"
for _key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "S2_API_KEY"):
    os.environ.pop(_key, None)
# A subprocess does not inherit the socket patch below, so point every provider at a closed
# port (discard, 9) and let it fail fast.
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["GROBID_SERVER"] = "http://127.0.0.1:9"

os.environ.setdefault("CITATION_CHROMA_SERVER", "")
# The legacy seam tests target the Chroma path; LanceDB has its own dedicated coverage.
os.environ.setdefault("CITATION_USE_LANCEDB", "0")
# The fixture world (tests/transfer_fixtures.py) is laid out for 3 cards per query
# and 3 techniques per bottleneck; the live defaults are 5.
os.environ.setdefault("CITATION_TRANSFER_CARDS_PER_QUERY", "3")
os.environ.setdefault("CITATION_TRANSFER_TECHNIQUES_PER_BOTTLENECK", "3")


def _wipe_contents(path):
    """Delete everything inside `path`, keeping `path` itself.

    Only the directories this conftest created are allowed: with
    CITATION_TEST_REAL_STORE=1 the DATA_DIR is the real corpus and must never be emptied.
    """
    allowed = {os.path.realpath(p) for p in (TEST_DATA_DIR, _WORKSPACE_DIR) if p}
    if not path or os.path.realpath(path) not in allowed:
        raise RuntimeError(f"refusing to wipe {path!r}: not a directory conftest created")
    if not os.path.isdir(path):
        return
    leftovers = []
    for name in os.listdir(path):
        child = os.path.join(path, name)
        try:
            if os.path.isdir(child) and not os.path.islink(child):
                shutil.rmtree(child)
            else:
                os.unlink(child)
        except OSError as exc:
            leftovers.append(f"{child} ({exc})")
    if leftovers:
        # A reset that silently leaves files behind shows up later as an unrelated failure.
        raise RuntimeError("could not wipe: " + "; ".join(leftovers))


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    if request.node.get_closest_marker("allow_network"):
        return

    def _blocked(*_args, **_kwargs):
        raise OSError("network disabled in tests")

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _guard(real):
        def wrapper(self, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                _blocked()
            return real(self, *args, **kwargs)  # AF_UNIX: multiprocessing needs it
        return wrapper

    monkeypatch.setattr(socket.socket, "connect", _guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(real_connect_ex))
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def _release_chroma_clients():
    """Drop chromadb's process-wide client cache, so no client still holds a wiped store.

    Chroma keeps one system per path; emptying the directory under it makes the next test's
    writes fail with "attempt to write a readonly database".
    """
    chromadb = sys.modules.get("chromadb")
    if chromadb is None:
        return
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except (ImportError, AttributeError):
        pass  # a chromadb without the shared-client cache has nothing to release


def _clear_in_process_caches():
    """Reset module-level caches, only for modules a test has already imported."""
    mods = sys.modules
    db = mods.get("research_assistant.shared.db")
    if db is not None:
        db.clear_search_cache()
        db._build_cache.clear()
    llm = mods.get("research_assistant.shared.llm")
    if llm is not None:
        llm._CLIENT_CACHE.clear()
        llm._embeddings_singleton = None
    run_jobs = mods.get("research_assistant.shared.run_jobs")
    if run_jobs is not None:
        run_jobs._JOBS.clear()
    pipeline_status = mods.get("research_assistant.shared.pipeline_status")
    if pipeline_status is not None:
        pipeline_status._local.current_job_id = None
    st = mods.get("streamlit")
    if st is not None:
        st.cache_data.clear()
        st.cache_resource.clear()


@pytest.fixture(autouse=True)
def _reset_shared_state():
    yield
    _release_chroma_clients()
    if TEST_DATA_DIR is not None:
        _wipe_contents(TEST_DATA_DIR)
    _wipe_contents(os.environ["CITATION_WORKSPACE_DIR"])
    _clear_in_process_caches()
