"""
Which documents have already been ingested.

This used to be two sources of truth unioned on every call: an append-only
newline-delimited text file inside the ChromaDB directory (no deduplication,
growing without bound) and a full paginated scan of every metadata record in
the collection. The scan is now a repair path — rebuild_from_collection() —
rather than something ingestion pays for on every run.

Writes are atomic: a temp file in the same directory, then os.replace, the same
pattern shared/fetch.py uses for downloads. A half-written manifest would make
the pipeline re-parse an entire corpus.
"""

import json
import os
import tempfile
import time
from typing import Iterable

from research_assistant.config import (
    COLLECTION_NAME,
    INGESTED_MANIFEST_PATH,
    VECTORDB_PATH,
)
from research_assistant.shared.log import get_logger

logger = get_logger("manifest")

MANIFEST_PATH = INGESTED_MANIFEST_PATH


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load() -> dict[str, str]:
    """Return ``{pdf_key: iso8601}``. Unreadable or absent means empty."""
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Ingestion manifest at %s is unreadable (%s) — treating as empty. "
            "Already-ingested PDFs may be re-parsed; rebuild_from_collection() "
            "recovers it from the vector store.",
            MANIFEST_PATH, exc,
        )
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "Ingestion manifest at %s is a %s, expected an object — treating as "
            "empty.", MANIFEST_PATH, type(data).__name__,
        )
        return {}
    return data


def _write(data: dict[str, str]) -> None:
    directory = os.path.dirname(MANIFEST_PATH) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, MANIFEST_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def contains(pdf_key: str) -> bool:
    return pdf_key in load()


def add(pdf_key: str) -> None:
    data = load()
    if pdf_key in data:
        return
    data[pdf_key] = _now()
    _write(data)


def add_many(pdf_keys: Iterable[str]) -> None:
    """Add several keys in one read-modify-write.

    ingest_pdfs() marks every candidate at the end of a batch; doing that one
    at a time would re-read and rewrite the whole manifest per PDF.
    """
    data = load()
    now = _now()
    added = False
    for key in pdf_keys:
        if key not in data:
            data[key] = now
            added = True
    if added:
        _write(data)


def rebuild_from_collection() -> int:
    """Repair the manifest from ChromaDB metadata. Returns the number of keys.

    Only for recovering a lost or corrupt manifest — ingestion does not call
    this. Reads every metadata record in the collection.
    """
    import chromadb

    try:
        collection = chromadb.PersistentClient(path=VECTORDB_PATH).get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:  # noqa: BLE001 — no collection yet is not an error
        logger.warning("No collection to rebuild from (%s).", exc)
        return 0

    documents, limit, offset = set(), 5000, 0
    while True:
        batch = collection.get(include=["metadatas"], limit=limit, offset=offset)
        if not batch or not batch["metadatas"]:
            break
        for meta in batch["metadatas"]:
            if meta and meta.get("document"):
                documents.add(meta["document"])
        offset += limit

    now = _now()
    _write({doc: now for doc in sorted(documents)})
    logger.info("Rebuilt ingestion manifest with %d document(s).", len(documents))
    return len(documents)
