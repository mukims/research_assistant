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


def resolve_chunk(
    texts: list[str],
    metadatas: list[dict],
    document: str,
    chunk_index: int | None,
) -> tuple[int, str, dict] | None:
    """Find a paper's chunk as (position in texts, text, metadata).

    chunk_index is a global position on the Chroma backend but the per-document seq on LanceDB, so
    the positional entry is trusted only when it belongs to `document`; otherwise the paper's chunk
    whose seq or chunk_index equals it is used. Returns None when neither exists. There is
    deliberately no fallback to another chunk: that would misattribute the evidence.
    """
    if chunk_index is None or not document:
        return None

    if 0 <= chunk_index < len(texts):
        meta = metadatas[chunk_index] if chunk_index < len(metadatas) else {}
        if meta.get("document") == document:
            return chunk_index, texts[chunk_index], meta

    for idx, (txt, m) in enumerate(zip(texts, metadatas)):
        if m.get("document") == document and (m.get("seq") == chunk_index or m.get("chunk_index") == chunk_index):
            return idx, txt, m
    return None


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
        self._cache: dict[tuple[str, str, int | None], tuple[dict | None, str | None]] = {}

    def check(self, p: Premise) -> Premise:
        """Check a single premise against its cited chunk and neighbours."""
        claim_norm = " ".join(p.claim.lower().split())
        cache_key = (claim_norm, p.document, p.chunk_index)

        if cache_key in self._cache:
            verdict, err = self._cache[cache_key]
        else:
            resolved = resolve_chunk(self.texts, self.metadatas, p.document, p.chunk_index)
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
