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
    VECTORDB_PATH,
    COLLECTION_NAME,
    BM25_INDEX_PATH,
    EMBED_MODEL,
    CHROMA_SERVER,
    USE_LANCEDB,
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
    if not USE_LANCEDB or lancedb is None:
        return False
    if not os.path.exists(LANCEDB_PATH):
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
    try:
        mtime = os.path.getmtime(BM25_INDEX_PATH)
    except OSError:
        mtime = None
    lance_mtime = None
    if os.path.exists(LANCEDB_PATH):
        try:
            lance_mtime = os.path.getmtime(LANCEDB_PATH)
        except OSError:
            pass
    return (USE_LANCEDB, LANCEDB_PATH, VECTORDB_PATH, COLLECTION_NAME, BM25_INDEX_PATH, mtime, lance_mtime)


class StoreUnavailable(RuntimeError):
    """The Chroma server that owns the store is not answering."""


def open_store():
    """The client for the store (LanceDB if ready, ChromaDB as fallback). Every process goes through here.

    With config.CHROMA_SERVER set, the server owns VECTORDB_PATH and this connects to it
    (see config.chroma_server_from for why); otherwise the files are opened directly,
    which is only safe when no other process has them open."""
    if is_lance_ready():
        return open_lance_store()

    import chromadb
    from urllib.parse import urlparse

    if not CHROMA_SERVER:
        return chromadb.PersistentClient(path=VECTORDB_PATH)
    url = urlparse(CHROMA_SERVER)
    client = chromadb.HttpClient(host=url.hostname, port=url.port or 8000, ssl=url.scheme == "https")
    try:
        client.heartbeat()
    except Exception as exc:  # noqa: BLE001 — any failure means no usable server
        raise StoreUnavailable(
            f"The Chroma server at {CHROMA_SERVER} is not answering ({exc}). Start it with: "
            f"chroma run --path {VECTORDB_PATH} --port {url.port or 8000}"
        ) from exc
    return client


def embedder_record(embeddings) -> dict:
    """Collection metadata naming the embedder, and its build where it can say."""
    record = {"embedder": EMBED_MODEL}
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
    if getattr(collection, "is_lancedb", False):
        return
    stored = (getattr(collection, "metadata", None) or {}).get("embedder")
    if stored and stored != EMBED_MODEL:
        raise RuntimeError(
            f"Collection {collection.name!r} was built with the embedder {stored!r} but "
            f"CITATION_EMBED_MODEL is {EMBED_MODEL!r}. Set CITATION_EMBED_MODEL={stored} to "
            "read it, or build a new index version for the new embedder."
        )
    stored_build = (getattr(collection, "metadata", None) or {}).get("embedder_build")
    if stored_build:
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
    """Cached: see _load_search_resources_uncached() for what is loaded."""
    key = _cache_key()
    if key not in _cache:
        _cache.clear()
        if is_lance_ready():
            try:
                client = open_lance_store()
                table = client.get_collection(LANCE_TABLE_NAME)
                _cache[key] = (
                    table,
                    LanceBM25Proxy(),
                    LanceSequenceProxy(table, "text"),
                    LanceSequenceProxy(table, "metadata"),
                )
                return _cache[key]
            except Exception as e:
                logger.warning("Failed to open LanceDB store (%s), falling back to ChromaDB.", e)
        _cache[key] = _load_search_resources_uncached()
    return _cache[key]


def _load_search_resources_uncached():
    """
    Connect to ChromaDB, fetch all chunks in index order, and load the
    BM25 pickle.


    Returns:
        tuple: (collection, bm25, texts, metadatas)
    """
    # Imported here rather than at module scope: chromadb pulls in a large
    # dependency tree, and agents that only need this module's pure helpers
    # (or their tests) should not have to install it.
    import chromadb

    logger.info("Connecting to ChromaDB at %s…", VECTORDB_PATH)
    chroma_client = open_store()

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        logger.error(
            "Collection '%s' not found in ChromaDB at '%s'.\n"
            "  → You need to run Agent 3 (python agent3_ingestor.py) first to\n"
            "    ingest papers and create the vector database.",
            COLLECTION_NAME,
            VECTORDB_PATH,
        )
        raise RuntimeError("Database not initialized. Please ingest papers first.")
    check_embedder(collection)

    # Paginate all chunks from the collection
    paired_data = []
    limit, offset = 5000, 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"], limit=limit, offset=offset
        )
        if not batch["ids"]:
            break
        for doc, meta, chunk_id in zip(
            batch["documents"], batch["metadatas"], batch["ids"]
        ):
            try:
                idx = int(chunk_id.split("_")[1])
                paired_data.append((idx, doc, meta))
            except (ValueError, IndexError):
                logger.warning("Skipping chunk with unexpected ID format: %s", chunk_id)
        offset += limit

    paired_data.sort(key=lambda x: x[0])
    texts = [item[1] for item in paired_data]
    metadatas = [item[2] for item in paired_data]

    logger.info("Loaded %d chunks from ChromaDB.", len(texts))

    if not os.path.exists(BM25_INDEX_PATH):
        logger.error(
            "BM25 index not found at '%s'.\n"
            "  → Run Agent 3 (python agent3_ingestor.py) to build it.",
            BM25_INDEX_PATH,
        )
        raise RuntimeError("BM25 index not found. Please ingest papers first.")

    logger.info("Loading BM25 index from %s…", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)

    # Two on-disk shapes: a bare BM25Okapi (built before the tokenizer was
    # versioned, with `\w+`) and a dict carrying the tokenizer tag. The tag is
    # copied onto the object so hybrid_search can pick the matching query
    # tokenizer without a second return value.
    if isinstance(payload, dict) and "bm25" in payload:
        bm25 = payload["bm25"]
        version = payload.get("tokenizer", "v1")
    else:
        bm25 = payload
        version = "v1"
    bm25.tokenizer_version = version
    if version not in ("v1", TOKENIZER_VERSION):
        logger.warning(
            "BM25 index was built with tokenizer %s but this code knows %s — "
            "keyword recall may be degraded; re-run ingestion to rebuild it.",
            version, TOKENIZER_VERSION,
        )

    return collection, bm25, texts, metadatas


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
