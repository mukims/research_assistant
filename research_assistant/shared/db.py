"""
Shared database loading utilities.

Consolidates the repeated ChromaDB + BM25 loading boilerplate
used by Agents 4, 5 and 7.

Usage:
    from research_assistant.shared.db import load_search_resources
    collection, bm25, texts, metadatas = load_search_resources()
"""

import os
import pickle

try:
    import lancedb
except ImportError:
    lancedb = None

from research_assistant.config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    LANCEDB_PATH,
    LANCE_TABLE_NAME,
    LANCE_SUMMARY_TABLE_NAME,
)
from research_assistant.shared.log import get_logger
from research_assistant.shared.tokenize import TOKENIZER_VERSION

logger = get_logger("db")

# One process, one copy of the corpus in memory. The synthesis path calls
# hybrid_search once per shortlisted paper, and each call used to page the
# whole collection out of Chroma and unpickle 13 MB — per query, per paper.
# Keyed on the pickle's mtime so a rebuild (ingest) invalidates it.
_cache: dict = {}


def open_lance_store(path=None):
    """Open the LanceDB store."""
    from research_assistant.shared.lance_store import LanceStoreClient
    return LanceStoreClient(path or LANCEDB_PATH)


def is_lance_ready() -> bool:
    """Check if LanceDB is enabled and has the main table initialized."""
    if lancedb is None or not os.path.exists(LANCEDB_PATH):
        return False
    try:
        from research_assistant.shared.lance_store import get_lance_connection, list_table_names
        conn = get_lance_connection(LANCEDB_PATH)
        tables = list_table_names(conn)
        return LANCE_TABLE_NAME in tables
    except Exception:
        return False


class LanceSequenceProxy:
    """Lazy zero-copy sequence proxy over a LanceDB table with chunked prefetch cache."""

    def __init__(self, table_adapter, field_name: str, cache_size: int = 256):
        self.table_adapter = table_adapter
        self.field_name = field_name
        self._cache_start = -1
        self._cache_data = []
        self._chunk_size = cache_size

    def __len__(self) -> int:
        return self.table_adapter.count()

    def _fetch_chunk(self, start_idx: int):
        try:
            q = self.table_adapter.table.search().offset(start_idx).limit(self._chunk_size)
            rows = q.to_list()
        except Exception:
            rows = []
        self._cache_start = start_idx
        self._cache_data = rows

    def __getitem__(self, idx: int | slice):
        total = len(self)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(total)
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx += total
        if not (0 <= idx < total):
            raise IndexError(f"Index {idx} out of range (length {total})")

        if not (self._cache_start <= idx < self._cache_start + len(self._cache_data)):
            self._fetch_chunk(idx)

        offset = idx - self._cache_start
        if not (0 <= offset < len(self._cache_data)):
            raise IndexError(f"Index {idx} out of range")
        r = self._cache_data[offset]
        if self.field_name == "metadata":
            m = {
                "document": r.get("document", ""),
                "citation_source": r.get("citation_source", ""),
                "stored_form": r.get("stored_form", ""),
                "section": r.get("section", ""),
                "seq": r.get("seq", 0),
                "page": r.get("page", 1),
                "type": r.get("stored_form", ""),
                "tier": r.get("tier", "full_text"),
            }
            if r.get("metadata_json"):
                try:
                    import json
                    m.update(json.loads(r["metadata_json"]))
                except Exception:
                    pass
            return m
        return r.get("text", "")


class LanceBM25Proxy:
    """Lightweight proxy for BM25 when backed by native LanceDB FTS."""

    tokenizer_version = "v2"
    is_lancedb = True

    def get_scores(self, q):
        return []

    def get_batch_scores(self, q, pool):
        return []


def _cache_key():
    lance_mtime = None
    if os.path.exists(LANCEDB_PATH):
        try:
            candidates = [os.path.getmtime(LANCEDB_PATH)]
            table_dir = os.path.join(LANCEDB_PATH, f"{COLLECTION_NAME}.lance")
            if os.path.exists(table_dir):
                candidates.append(os.path.getmtime(table_dir))
                versions_dir = os.path.join(table_dir, "_versions")
                if os.path.exists(versions_dir):
                    candidates.append(os.path.getmtime(versions_dir))
                    try:
                        for vf in os.scandir(versions_dir):
                            candidates.append(vf.stat().st_mtime)
                    except OSError:
                        pass
                hint_json = os.path.join(table_dir, "latest_version_hint.json")
                if os.path.exists(hint_json):
                    candidates.append(os.path.getmtime(hint_json))
                hint_file = os.path.join(table_dir, "_latest_version")
                if os.path.exists(hint_file):
                    candidates.append(os.path.getmtime(hint_file))
            lance_mtime = max(candidates) if candidates else None
        except OSError:
            pass
    return (LANCEDB_PATH, COLLECTION_NAME, EMBED_MODEL, lance_mtime)


class StoreUnavailable(RuntimeError):
    """The store is not available or could not be opened."""


def open_store():
    """The client for the LanceDB store. Every process goes through here."""
    if lancedb is None:
        raise StoreUnavailable("lancedb package is required but not installed.")
    try:
        return open_lance_store()
    except Exception as exc:
        raise StoreUnavailable(f"Failed to open LanceDB store at {LANCEDB_PATH}: {exc}") from exc


def embedder_record(embeddings) -> dict:
    """Collection metadata naming the embedder, its vector width, and its build where it can say."""
    from research_assistant.shared.llm import embedding_dim

    record = {"embedder": EMBED_MODEL, "embedding_dim": embedding_dim(embeddings)}
    build = getattr(embeddings, "build_id", lambda: None)()
    if build:
        record["embedder_build"] = build
    return record


_build_cache: dict = {}


def _current_build():
    """The configured embedder's build id, looked up once per process."""
    if "build" not in _build_cache:
        from research_assistant.shared.llm import get_embeddings

        _build_cache["build"] = getattr(get_embeddings(), "build_id", lambda: None)()
    return _build_cache["build"]


def check_embedder(collection) -> None:
    """Refuse a collection built with a different embedder than the configured one.

    A query embedded by one model against vectors from another returns plausible
    but wrong neighbours and no error, so the embedder is recorded when a
    collection is created and compared here whenever one is opened. A collection
    that predates the record (index v1, v2) is not checked.
    """
    meta = getattr(collection, "metadata", None)
    if not isinstance(meta, dict):
        return
    stored = meta.get("embedder")
    if isinstance(stored, str) and stored and stored != EMBED_MODEL:
        raise RuntimeError(
            f"Collection {collection.name!r} was built with the embedder {stored!r} but "
            f"CITATION_EMBED_MODEL is {EMBED_MODEL!r}. Set CITATION_EMBED_MODEL={stored} to "
            "read it, or build a new index version for the new embedder."
        )
    stored_build = meta.get("embedder_build")
    if isinstance(stored_build, str) and stored_build:
        current = _current_build()
        if current and current != stored_build:
            logger.warning(
                "Collection %r was embedded with build %s of %s but this machine has build %s. "
                "Same name, different weights or quantisation: retrieval will be slightly off. "
                "Pull the same build, or re-embed.", collection.name, stored_build, EMBED_MODEL, current,
            )


def clear_search_cache() -> None:
    _cache.clear()


def load_search_resources():
    """Load and cache LanceDB search table and proxies."""
    key = _cache_key()
    if key not in _cache:
        _cache.clear()
        client = open_store()
        try:
            table = client.get_collection(LANCE_TABLE_NAME)
        except Exception as e:
            logger.error("Collection '%s' not found in LanceDB at '%s': %s", LANCE_TABLE_NAME, LANCEDB_PATH, e)
            raise RuntimeError(f"Database not initialized. Please ingest papers first: {e}") from e
        
        check_embedder(table)
        _cache[key] = (
            table,
            LanceBM25Proxy(),
            LanceSequenceProxy(table, "text"),
            LanceSequenceProxy(table, "metadata"),
        )
    return _cache[key]


def get_max_chunk_index(collection) -> int:
    """
    Read the current maximum chunk index from the collection.

    Returns the next available index (max + 1), or 0 if the collection is empty.
    """
    if getattr(collection, "is_lancedb", False):
        try:
            return collection.count()
        except Exception:
            return 0
    max_idx = -1
    limit, offset = 5000, 0
    while True:
        # include=[] fetches ids only. The ChromaDB default is
        # ["metadatas", "documents"], so without this the whole corpus — every
        # chunk's text and metadata — was pulled across the wire and discarded
        # just to read the largest id.
        batch = collection.get(limit=limit, offset=offset, include=[])
        if not batch or not batch["ids"]:
            break
        for cid in batch["ids"]:
            try:
                max_idx = max(max_idx, int(cid.split("_")[1]))
            except (ValueError, IndexError):
                pass
        offset += limit
    return max_idx + 1
