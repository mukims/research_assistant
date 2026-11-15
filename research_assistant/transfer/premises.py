"""Premise checking for technique transfer using the citer's acceptance rule."""

from __future__ import annotations

import threading
from typing import Callable

from research_assistant.judgement.evidence import assemble_evidence
from research_assistant.config import (
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
)
from research_assistant.transfer.chunk_store import ChunkStore, chunk_seq, chunk_store_for
from research_assistant.transfer.models import Premise


def resolve_chunk(
    texts: list[str],
    metadatas: list[dict],
    document: str,
    seq: int | None,
    position: int | None = None,
    chunk_store: ChunkStore | None = None,
) -> tuple[int, str, dict] | None:
    """Find chunk (document, seq) as (position in texts, text, metadata); None when there is none.

    `position` is where the caller expects the chunk to sit. A row is taken without a search only
    when it is proven to be the chunk: same paper and same seq. Otherwise the paper's chunk with that
    seq is searched for. There is deliberately no fallback to another chunk: that would misattribute
    the evidence.
    """
    if seq is None or not document:
        return None
    store = chunk_store or chunk_store_for(texts, metadatas)
    return store.get_chunk(document, int(seq), position=position)


def resolve_premise(
    texts: list[str],
    metadatas: list[dict],
    p: Premise,
    chunk_store: ChunkStore | None = None,
) -> tuple[int, str, dict] | None:
    """Find the chunk a premise cites. A premise saved before seq existed is card evidence, whose
    chunk_index was always a position in texts: that row is used if it is still the same paper's,
    and is never re-read as a seq."""
    if p.seq is not None:
        return resolve_chunk(texts, metadatas, p.document, p.seq, position=p.chunk_index, chunk_store=chunk_store)
    idx = p.chunk_index
    if idx is None or not p.document or not (0 <= idx < len(texts)) or idx >= len(metadatas):
        return None
    meta = metadatas[idx]
    if meta.get("document") != p.document:
        return None
    return idx, texts[idx], meta


class PremiseChecker:
    """Thread-safe judge for candidate premises against source chunks and their neighbours."""

    def __init__(
        self,
        judge: Callable[[str, str], dict],
        texts: list[str],
        metadatas: list[dict],
        chunk_store: ChunkStore | None = None,
    ):
        self.judge = judge
        self.texts = texts
        self.metadatas = metadatas
        self.chunk_store = chunk_store or chunk_store_for(texts, metadatas)
        self.calls: int = 0
        self._cache: dict[tuple, tuple[dict | None, str | None]] = {}
        self._lock = threading.RLock()

    def check(self, p: Premise) -> Premise:
        """Check a single premise against its cited chunk and neighbours."""
        claim_norm = " ".join(p.claim.lower().split())
        # Keyed by the chunk's identity: a premise saved before seq existed has only its position.
        identity = ("seq", p.seq) if p.seq is not None else ("position", p.chunk_index)
        cache_key = (claim_norm, p.document, identity)

        with self._lock:
            cached = self._cache.get(cache_key)

        if cached is not None:
            verdict, err = cached
        else:
            resolved = resolve_premise(self.texts, self.metadatas, p, chunk_store=self.chunk_store)
            if resolved is None:
                # Nothing to judge against: report it rather than judge some other paper's chunk.
                verdict, err = None, f"evidence chunk not found in {p.document}"
                with self._lock:
                    self._cache[cache_key] = (verdict, err)
            else:
                idx, text, meta = resolved
                seq = chunk_seq(meta, idx)
                stored_form = meta.get("stored_form") or meta.get("type") or "text_chunk"
                before, after = self.chunk_store.get_neighbours(
                    p.document,
                    stored_form,
                    seq,
                    window=JUDGEMENT_NEIGHBOUR_WINDOW,
                    position=idx,
                )
                hit = {
                    "chunk_index": idx,
                    "text": text,
                    "metadata": meta,
                    "context_before": before,
                    "context_after": after,
                }
                evidence = assemble_evidence([hit], JUDGEMENT_EVIDENCE_MAX_CHARS)

                with self._lock:
                    self.calls += 1
                try:
                    verdict = self.judge(p.claim, evidence)
                    err = None
                    with self._lock:
                        self._cache[cache_key] = (verdict, None)
                except Exception as exc:
                    verdict = None
                    err = f"{type(exc).__name__}: {exc}"
                    with self._lock:
                        self._cache[cache_key] = (None, err)

        if verdict is not None:
            p.judgement = verdict.get("judgement")
            p.span_verified = verdict.get("span_verified")
            p.supporting_span = verdict.get("supporting_span")
            p.reason = verdict.get("reason")
            p.error = None
        else:
            p.judgement = None
            p.span_verified = None
            p.supporting_span = None
            p.reason = None
            p.error = err

        return p

    def check_all(self, ps: list[Premise]) -> list[Premise]:
        return [self.check(p) for p in ps]
