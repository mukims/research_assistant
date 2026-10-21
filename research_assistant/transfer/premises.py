"""Premise checking for technique transfer using the citer's acceptance rule."""

from __future__ import annotations

from typing import Callable

from research_assistant.agents.agent8_verifier import assemble_evidence
from research_assistant.config import (
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
)
from research_assistant.shared.search import expand_neighbours
from research_assistant.transfer.models import Premise


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
        self._cache: dict[tuple[str, int | None], tuple[dict | None, str | None]] = {}

    def check(self, p: Premise) -> Premise:
        """Check a single premise against its cited chunk and neighbours."""
        claim_norm = " ".join(p.claim.lower().split())
        cache_key = (claim_norm, p.chunk_index)

        if cache_key in self._cache:
            verdict, err = self._cache[cache_key]
        else:
            evidence = ""
            if p.chunk_index is not None and 0 <= p.chunk_index < len(self.texts):
                hit = {
                    "chunk_index": p.chunk_index,
                    "text": self.texts[p.chunk_index],
                    "metadata": self.metadatas[p.chunk_index] if p.chunk_index < len(self.metadatas) else {},
                }
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
