"""Chunk lookup abstraction decoupling transfer logic from list proxies and LanceDB adapters."""

from __future__ import annotations

from typing import Protocol

from research_assistant.config import EXCLUDED_FROM_EVIDENCE
from research_assistant.shared.search import expand_neighbours


class ChunkStore(Protocol):
    """Read-only interface for resolving corpus chunks and their document neighbours."""

    def get_chunk(
        self,
        document: str,
        seq: int,
        position: int | None = None,
    ) -> tuple[int, str, dict] | None: ...

    def get_document_chunks(self, document: str) -> list[tuple[int, str, dict]]: ...

    def get_neighbours(
        self,
        document: str,
        stored_form: str,
        seq: int,
        window: int,
        position: int | None = None,
    ) -> tuple[str, str]: ...


def chunk_seq(meta: dict | None, position: int) -> int:
    """Return a chunk's per-document sequence number, falling back to list position."""
    seq = (meta or {}).get("seq")
    return position if seq is None else int(seq)


class LanceChunkStore:
    """ChunkStore backed directly by LanceStoreClient SQL queries (zero offset scans)."""

    def __init__(self, table_adapter):
        self.table_adapter = table_adapter

    def get_chunk(
        self,
        document: str,
        seq: int,
        position: int | None = None,
    ) -> tuple[int, str, dict] | None:
        if not document:
            return None
        return self.table_adapter.find_chunk_by_seq(
            document, int(seq), exclude_types=EXCLUDED_FROM_EVIDENCE
        )

    def get_document_chunks(self, document: str) -> list[tuple[int, str, dict]]:
        if not document:
            return []
        return [
            (seq_val, txt, meta)
            for seq_val, txt, meta in self.table_adapter.find_document_chunks(
                document, exclude_types=EXCLUDED_FROM_EVIDENCE
            )
            if meta.get("section") != "caption"
        ]

    def get_neighbours(
        self,
        document: str,
        stored_form: str,
        seq: int,
        window: int,
        position: int | None = None,
    ) -> tuple[str, str]:
        if not document or window <= 0:
            return "", ""
        return self.table_adapter.get_neighbours(
            document,
            stored_form,
            int(seq),
            window=window,
            exclude_types=EXCLUDED_FROM_EVIDENCE,
        )


class SequenceChunkStore:
    """ChunkStore backed by in-memory sequences (used by tests and legacy list corpora)."""

    def __init__(self, texts: list[str], metadatas: list[dict]):
        self.texts = texts
        self.metadatas = metadatas

    def get_chunk(
        self,
        document: str,
        seq: int,
        position: int | None = None,
    ) -> tuple[int, str, dict] | None:
        if not document:
            return None
        n = len(self.texts)
        for guess in dict.fromkeys(g for g in (position, seq) if g is not None):
            if 0 <= guess < n and guess < len(self.metadatas):
                meta = self.metadatas[guess]
                if meta.get("document") == document and chunk_seq(meta, guess) == seq:
                    return guess, self.texts[guess], meta

        for idx, meta in enumerate(self.metadatas):
            if meta.get("document") == document and chunk_seq(meta, idx) == seq:
                return idx, self.texts[idx], meta
        return None

    def get_document_chunks(self, document: str) -> list[tuple[int, str, dict]]:
        if not document:
            return []
        rows: list[tuple[int, str, dict]] = []
        for idx, meta in enumerate(self.metadatas):
            if meta.get("document") != document:
                continue
            if meta.get("type") in EXCLUDED_FROM_EVIDENCE:
                continue
            if meta.get("section") == "caption":
                continue
            rows.append((idx, self.texts[idx], meta))
        return rows

    def get_neighbours(
        self,
        document: str,
        stored_form: str,
        seq: int,
        window: int,
        position: int | None = None,
    ) -> tuple[str, str]:
        if window <= 0:
            return "", ""
        resolved = None
        if (
            position is not None
            and 0 <= position < len(self.texts)
            and position < len(self.metadatas)
            and self.metadatas[position].get("document") == document
        ):
            resolved = (position, self.texts[position], self.metadatas[position])
        if resolved is None:
            resolved = self.get_chunk(document, seq, position=position)
        if resolved is None:
            return "", ""
        idx, text, meta = resolved
        hit = {"chunk_index": idx, "text": text, "metadata": meta}
        expand_neighbours([hit], self.texts, self.metadatas, window=window)
        return hit.get("context_before", ""), hit.get("context_after", "")


def chunk_store_for(texts: list[str], metadatas: list[dict]) -> ChunkStore:
    """Select LanceChunkStore when a table_adapter is attached, else SequenceChunkStore."""
    table_adapter = getattr(texts, "table_adapter", None) or getattr(metadatas, "table_adapter", None)
    if table_adapter is not None and hasattr(table_adapter, "find_chunk_by_seq"):
        return LanceChunkStore(table_adapter)
    return SequenceChunkStore(texts, metadatas)
