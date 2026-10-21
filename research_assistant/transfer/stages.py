"""Pipeline stages for technique transfer: grounding, candidate search, development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_assistant.config import DOC_SELECT_K
from research_assistant.transfer.deps import Deps, TransferSettings
from research_assistant.transfer.models import Bottleneck, Passage, Premise
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prompts import BOTTLENECKS_USER, SYSTEM, format_passages
from research_assistant.transfer.replies import ask_json, resolve_premises


class CorpusCoverageError(RuntimeError):
    """Raised when the literature corpus does not adequately cover the query topic."""

    def __init__(self, query: str, kept: list[dict], shortlist: list[dict]):
        self.query = query
        self.kept = kept
        self.shortlist = shortlist
        citations = [r.get("citation") or r.get("document", "unknown") for r in shortlist]
        cit_str = ", ".join(citations) if citations else "none"
        super().__init__(
            f"Corpus does not sufficiently cover query {query!r}: kept {len(kept)}/{len(shortlist)} papers ({cit_str}). "
            f"Build a corpus in the Research idea tab first."
        )


@dataclass
class Grounding:
    query: str
    near: list[dict]
    passages: dict[str, Passage]
    bottlenecks: list[Bottleneck]
    warnings: list[str]


def ground(
    query: str,
    deps: Deps,
    settings: TransferSettings,
    checker: PremiseChecker,
) -> Grounding:
    """
    Stage S1: Ground query in literature, gate relevance, and extract bottlenecks.

    1. Shortlist documents excluding settings.exclude_documents.
    2. Gate documents. Raise CorpusCoverageError if fewer than min_near_papers kept.
    3. Retrieve passages for kept papers, tagged C1... with role 'near'.
    4. Prompt LLM for technical bottlenecks with generic abstraction levels.
    5. Resolve and validate bottleneck premises.
    6. Judge premises via checker.
    """
    shortlist = [
        r for r in deps.shortlist(query, DOC_SELECT_K)
        if r.get("document") not in settings.exclude_documents
    ]

    kept = deps.gate(query, shortlist)
    if len(kept) < settings.min_near_papers:
        raise CorpusCoverageError(query, kept, shortlist)

    kept_docs = {r["document"] for r in kept}
    exclude_docs = set(settings.exclude_documents) if settings.exclude_documents else None
    hits = deps.search(query, settings.near_passages, kept_docs, exclude_docs)

    passages: dict[str, Passage] = {}
    seen_chunks: set[int] = set()
    p_num = 1
    for h in hits:
        c_idx = h["chunk_index"]
        if c_idx in seen_chunks:
            continue
        seen_chunks.add(c_idx)
        key = f"C{p_num}"
        meta = h.get("metadata") or {}
        passages[key] = Passage(
            key=key,
            chunk_index=c_idx,
            document=h["document"],
            citation=meta.get("citation", f"Citation {h['document']}"),
            section=meta.get("section", "methods"),
            text=h.get("text", ""),
            role="near",
        )
        p_num += 1

    passages_text = format_passages(list(passages.values()), settings.passage_max_chars)
    prompt = BOTTLENECKS_USER.format(
        query=query,
        passages=passages_text,
        n=settings.bottlenecks,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    warnings: list[str] = []
    try:
        reply = ask_json(
            lambda msgs: deps.chat(msgs, settings.bottleneck_temperature),
            messages,
            required=("bottlenecks",),
        )
        raw_bottlenecks = reply.get("bottlenecks", [])
        if not isinstance(raw_bottlenecks, list):
            warnings.append(f"Expected list for 'bottlenecks', got {type(raw_bottlenecks).__name__}")
            raw_bottlenecks = []
    except Exception as exc:
        warnings.append(f"Failed to parse bottlenecks reply: {exc}")
        raw_bottlenecks = []

    bottlenecks: list[Bottleneck] = []
    all_premises_to_check: list[Premise] = []
    b_idx = 1
    levels = ("specific", "mid", "broad")

    for item in raw_bottlenecks[: settings.bottlenecks]:
        statement = item.get("statement", "").strip() if isinstance(item.get("statement"), str) else ""
        quantities = [q.strip() for q in item.get("quantities", []) if isinstance(q, str) and q.strip()]
        current_method = item.get("current_method", "").strip() if isinstance(item.get("current_method"), str) else ""
        generic = item.get("generic") if isinstance(item.get("generic"), dict) else {}
        raw_premises = item.get("premises", []) if isinstance(item.get("premises"), list) else []

        if not quantities:
            warnings.append(f"Dropped bottleneck with no quantities: {statement!r}")
            continue

        if not all(isinstance(generic.get(lvl), str) and generic.get(lvl).strip() for lvl in levels):
            warnings.append(f"Dropped bottleneck with incomplete generic levels: {statement!r}")
            continue

        resolved_p, p_warns = resolve_premises(raw_premises, passages, limit=3)
        warnings.extend(p_warns)
        if not resolved_p:
            warnings.append(f"Dropped bottleneck with no valid premises: {statement!r}")
            continue

        b_id = f"B{b_idx}"
        b = Bottleneck(
            id=b_id,
            statement=statement,
            quantities=quantities,
            current_method=current_method,
            generic={lvl: generic[lvl].strip() for lvl in levels},
            premises=resolved_p,
        )
        bottlenecks.append(b)
        all_premises_to_check.extend(resolved_p)
        b_idx += 1

    checker.check_all(all_premises_to_check)

    return Grounding(
        query=query,
        near=kept,
        passages=passages,
        bottlenecks=bottlenecks,
        warnings=warnings,
    )
