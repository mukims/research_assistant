"""Premise checking for technique transfer using the citer's acceptance rule."""

from __future__ import annotations

from typing import Callable

from research_assistant.judgement.evidence import assemble_evidence
from research_assistant.config import (
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
)
from research_assistant.shared.search import expand_neighbours
from research_assistant.transfer.models import Premise


def chunk_seq(meta: dict | None, position: int) -> int:
    """A chunk's identity within its paper: metadata["seq"], the per-document sequence number.

    LanceDB and v2/v3 Chroma ingestion record it. A store that records none (v1 Chroma chunks, the
    list-backed test corpora) has only the chunk's position in texts, which is also what its search
    hits report as chunk_index, so the position is that chunk's seq.
    """
    seq = (meta or {}).get("seq")
    return position if seq is None else int(seq)


def resolve_chunk(
    texts: list[str],
    metadatas: list[dict],
    document: str,
    seq: int | None,
    position: int | None = None,
) -> tuple[int, str, dict] | None:
    """Find chunk (document, seq) as (position in texts, text, metadata); None when there is none.

    `position` is where the caller expects the chunk to sit. A row is taken without a search only
    when it is proven to be the chunk: same paper and same seq. Otherwise the paper's chunk with that
    seq is searched for. There is deliberately no fallback to another chunk: that would misattribute
    the evidence.
    """
    if seq is None or not document:
        return None

    n = len(texts)
    for guess in dict.fromkeys(g for g in (position, seq) if g is not None):
        if 0 <= guess < n and guess < len(metadatas):
            meta = metadatas[guess]
            if meta.get("document") == document and chunk_seq(meta, guess) == seq:
                return guess, texts[guess], meta

    # Metadata only: on LanceDB every item read is a table query, so texts is read once, on a match.
    for idx, m in enumerate(metadatas):
        if m.get("document") == document and chunk_seq(m, idx) == seq:
            return idx, texts[idx], m
    return None


def resolve_premise(texts: list[str], metadatas: list[dict], p: Premise) -> tuple[int, str, dict] | None:
    """Find the chunk a premise cites. A premise saved before seq existed is card evidence, whose
    chunk_index was always a position in texts: that row is used if it is still the same paper's,
    and is never re-read as a seq."""
    if p.seq is not None:
        return resolve_chunk(texts, metadatas, p.document, p.seq, position=p.chunk_index)
    idx = p.chunk_index
    if idx is None or not p.document or not (0 <= idx < len(texts)) or idx >= len(metadatas):
        return None
    meta = metadatas[idx]
    if meta.get("document") != p.document:
        return None
    return idx, texts[idx], meta


class PremiseChecker:
    """Judges candidate premises against source chunks and their neighbours."""

    def __init__(
        self,
        judge: Callable[[str, str], dict],
        texts: list[str],
        metadatas: list[dict],
    ):
        self.judge = judge
        self.texts = texts
        self.metadatas = metadatas
        self.calls: int = 0
        self._cache: dict[tuple, tuple[dict | None, str | None]] = {}

    def check(self, p: Premise) -> Premise:
        """Check a single premise against its cited chunk and neighbours."""
        claim_norm = " ".join(p.claim.lower().split())
        # Keyed by the chunk's identity: a premise saved before seq existed has only its position.
        identity = ("seq", p.seq) if p.seq is not None else ("position", p.chunk_index)
        cache_key = (claim_norm, p.document, identity)

        if cache_key in self._cache:
            verdict, err = self._cache[cache_key]
        else:
            resolved = resolve_premise(self.texts, self.metadatas, p)
            if resolved is None:
                # Nothing to judge against: report it rather than judge some other paper's chunk.
                verdict, err = None, f"evidence chunk not found in {p.document}"
                self._cache[cache_key] = (verdict, err)
            else:
                idx, text, meta = resolved
                # "chunk_index" is the resolved global position, which expand_neighbours indexes by.
                hit = {"chunk_index": idx, "text": text, "metadata": meta}
                expand_neighbours([hit], self.texts, self.metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
                evidence = assemble_evidence([hit], JUDGEMENT_EVIDENCE_MAX_CHARS)

                self.calls += 1
                try:
                    verdict = self.judge(p.claim, evidence)
                    err = None
                    self._cache[cache_key] = (verdict, None)
                except Exception as exc:
                    verdict = None
                    err = f"{type(exc).__name__}: {exc}"
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
