"""Pipeline stages for technique transfer: grounding, candidate search, development."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from research_assistant.config import DOC_SELECT_K
from research_assistant.shared import pipeline_status
from research_assistant.transfer.deps import Deps, TransferSettings
from research_assistant.transfer.models import (
    Bottleneck,
    Card,
    Critique,
    Identifiability,
    MappingRow,
    Passage,
    Premise,
    PriorArt,
    Step,
    TechniqueLead,
    Transfer,
)
from research_assistant.transfer.chunk_store import chunk_seq
from research_assistant.transfer.premises import PremiseChecker, resolve_chunk, resolve_premise
from research_assistant.transfer.prompts import (
    BOTTLENECKS_USER,
    CRITIQUE_USER,
    REVISE_USER,
    SYSTEM,
    TRANSFER_USER,
    VERIFY_REBUTTAL_USER,
    format_passages,
)
from research_assistant.transfer.replies import (
    ask_json,
    resolve_premises,
    validate_critique_reply,
    validate_revise_reply,
    validate_transfer_reply,
    validate_verify_rebuttal_reply,
)


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


def _passage_key(document: str, seq: int) -> tuple[str, int]:
    """Identity of a passage: its paper and its seq within it (see Passage). Never chunk_index,
    which is a position in texts for a card passage but the seq for a LanceDB search hit."""
    return (document, seq)


def _passage_of(passages: dict[str, Passage], document: str, seq: int) -> Passage | None:
    return next((p for p in passages.values() if p.document == document and p.seq == seq), None)


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
    seen_chunks: set[tuple[str, int]] = set()
    p_num = 1
    for h in hits:
        c_idx = h["chunk_index"]
        # hybrid_search keeps the paper and its citation in the chunk metadata.
        meta = h.get("metadata") or {}
        doc = meta.get("document", "")
        seq = chunk_seq(meta, c_idx)
        passage_id = _passage_key(doc, seq)
        if passage_id in seen_chunks:
            continue
        seen_chunks.add(passage_id)
        key = f"C{p_num}"
        passages[key] = Passage(
            key=key,
            chunk_index=c_idx,
            seq=seq,
            document=doc,
            citation=meta.get("citation_source") or doc,
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

        present = {lvl: generic[lvl].strip() for lvl in levels if isinstance(generic.get(lvl), str) and generic[lvl].strip()}
        if len(present) < len(levels):
            # A missing level borrows the mid form, then another level, then the statement.
            fallback = next((present[lvl] for lvl in ("mid", "specific", "broad") if lvl in present), statement)
            if not fallback:
                warnings.append(f"Dropped bottleneck with no generic forms and no statement: {item!r}"[:200])
                continue
            warnings.append(f"Repaired generic levels {sorted(set(levels) - set(present))} of: {statement!r}")
            generic = {lvl: present.get(lvl, fallback) for lvl in levels}

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
            generic={lvl: generic[lvl].strip() for lvl in levels} | inverse_form(item),
            premises=resolved_p,
            # Anything but an explicit false or an explicit forward-problem statement keeps the identifiability gate on.
            inference=_is_inference_bottleneck(query, statement, item.get("inference")),
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


_FORWARD_PROBLEM_RE = re.compile(
    r"\b(forward\s+problem|forward\s+simulation|forward\s+solver|accelerat\w*\s+the\s+forward)\b",
    re.IGNORECASE,
)
_INVERSE_CUE_RE = re.compile(
    r"\b(inverse|invert|infer\w*|recover\w*|reconstruct\w*|identifi\w*|deconvol\w*)\b",
    re.IGNORECASE,
)


def _is_inference_bottleneck(query: str, statement: str, raw_flag: Any) -> bool:
    """Determine whether a bottleneck is an inverse/inference problem subject to identifiability gating."""
    if raw_flag is False:
        return False
    combined = f"{query} {statement}"
    if _FORWARD_PROBLEM_RE.search(combined) and not _INVERSE_CUE_RE.search(statement):
        return False
    return True


def inverse_form(item: dict) -> dict[str, str]:
    """{"inverse": sentence} for an inference bottleneck that gives one, else {}.

    It states the problem's shape (what unobserved thing is inferred from what
    measured one), which reaches inverse-problem techniques the three levels miss.
    """
    generic = item.get("generic") if isinstance(item.get("generic"), dict) else {}
    text = generic.get("inverse")
    if item.get("inference") is False or not isinstance(text, str) or not text.strip():
        return {}
    return {"inverse": text.strip()}


def _title_key(title: str) -> str:
    """Letters and digits only, lowercased: "MoS 2 films" and "MoS2 films" match."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _is_baseline_echo(card_name: str, current_method: str) -> bool:
    """Return True when card_name names the bottleneck's own current_method."""
    c_key = _title_key(card_name)
    m_key = _title_key(current_method)
    if len(c_key) < 6 or len(m_key) < 6:
        return False
    return c_key == m_key or c_key in m_key


_GENERIC_DOMAIN_STOPWORDS = frozenset({
    "method", "methods", "model", "models", "system", "systems", "problem", "problems",
    "data", "based", "using", "approach", "analysis", "theory", "study", "network",
    "networks", "learning", "algorithm", "algorithms", "function", "functions", "state",
    "states", "large", "scale", "high", "low", "fast", "new", "from", "with", "for",
    "citation", "context", "paper", "source", "target",
})


def _domain_echo_factor(card: Card, near: list[dict], quantities: list[str]) -> float:
    """Discount factor in [0.85, 1.0] when card's application context/citation echoes the target domain."""
    near_text = " ".join(str(r.get("citation", "")) for r in near) + " " + " ".join(quantities)
    target_domain_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z]{5,}", near_text)
        if w.lower() not in _GENERIC_DOMAIN_STOPWORDS
    }
    if len(target_domain_words) < 2:
        return 1.0
    card_domain_text = f"{card.citation} {card.context}"
    card_domain_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z]{5,}", card_domain_text)
        if w.lower() not in _GENERIC_DOMAIN_STOPWORDS
    }
    shared = len(target_domain_words & card_domain_words)
    if shared >= 3:
        return 0.85
    if shared == 2:
        return 0.92
    return 1.0


def copies_of(near: list[dict], cards: list[Card]) -> set[str]:
    """Documents that hold another copy of a near paper, matched by title.

    The corpus stores some papers more than once (a DOI copy and a title
    copy); a near paper's other copies must not come back as technique sources.
    """
    near_docs = {r["document"] for r in near}
    keys = {_title_key(r.get("citation", "")) for r in near}
    keys |= {_title_key(c.citation) for c in cards if c.document in near_docs}
    keys.discard("")
    return {c.document for c in cards if _title_key(c.citation) in keys} - near_docs


@dataclass
class Seed:
    bottleneck_id: str
    card: Card
    score: float
    level: str
    rank: int
    guidance: str = ""


def _topic_match_bonus(card: Card, source_topic: str) -> float:
    """Bonus when a card's discipline/citation/context/method matches a requested source_topic."""
    if not source_topic or not source_topic.strip():
        return 0.0
    topic_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9]{4,}", source_topic)
        if w.lower() not in _GENERIC_DOMAIN_STOPWORDS
    }
    if not topic_words:
        return 0.0
    card_text = f"{card.name} {card.citation} {card.context} {card.problem_solved}".lower()
    card_words = set(re.findall(r"[a-zA-Z0-9]{4,}", card_text))
    overlap = len(topic_words & card_words) / len(topic_words)
    return 0.12 * overlap


def candidate_cards(
    grounding: Grounding,
    deps: Deps,
    settings: TransferSettings,
    leads: list[TechniqueLead] | None = None,
) -> tuple[list[Seed], list[str]]:
    """
    Stage S2: Retrieve and rank candidate technique cards for each bottleneck.

    1. For each level in ("specific", "mid", "broad", "inverse") and any LLM-proposed TechniqueLead,
       search the cards index.
    2. Merge hits by paper title keeping the highest similarity score, level, rank, and LLM guidance.
    3. Keep top settings.techniques_per_bottleneck cards per bottleneck.
    4. Warn if no verified cards found for a bottleneck.
    """
    near_docs = {r["document"] for r in grounding.near}
    exclude = frozenset(near_docs | copies_of(grounding.near, deps.cards.cards) | set(settings.exclude_documents))
    seeds: list[Seed] = []
    warnings: list[str] = []
    leads_list = leads or []

    for b in grounding.bottlenecks:
        if deps.cancelled():
            raise pipeline_status.PipelineCancelledError("Technique transfer stopped by user")
        best_by_paper: dict[str, tuple[Card, float, str, int, str]] = {}

        search_specs: list[tuple[str, str, str, set[str]]] = []
        for level in ("specific", "mid", "broad", "inverse"):
            generic_text = b.generic.get(level, "")
            if generic_text:
                search_specs.append((level, generic_text, "", set()))

        for ld in leads_list:
            if ld.bottleneck_id == b.id:
                lead_q = f"{ld.technique_name}. {ld.resolution_guidance}".strip()
                guidance_str = f"[{ld.technique_name} — {ld.source_discipline}] {ld.resolution_guidance}"
                search_specs.append(("llm_guided", lead_q, guidance_str, set(ld.built_card_ids)))

        for level, query_str, guidance_str, lead_card_ids in search_specs:
            query_vec = deps.embed_query(query_str)
            try:
                hits = deps.cards.search(
                    query_vec,
                    settings.cards_per_query,
                    exclude_documents=exclude,
                    query_text=query_str,
                )
            except TypeError:
                hits = deps.cards.search(
                    query_vec,
                    settings.cards_per_query,
                    exclude_documents=exclude,
                )
            for rank_idx, (card_obj, score) in enumerate(hits, start=1):
                if _is_baseline_echo(card_obj.name, b.current_method):
                    continue
                adj_score = score * _domain_echo_factor(card_obj, grounding.near, b.quantities)
                adj_score += _topic_match_bonus(card_obj, settings.source_topic)
                if level == "llm_guided":
                    if card_obj.card_id in lead_card_ids:
                        adj_score += 0.12
                    else:
                        adj_score += 0.05
                # Deduplicate by normalized paper title so duplicate PDFs of the same
                # paper cannot occupy multiple top-k slots for the same bottleneck.
                paper_key = _title_key(card_obj.citation) or card_obj.document
                prev = best_by_paper.get(paper_key)
                if prev is None or adj_score > prev[1]:
                    kept_guidance = guidance_str or (prev[4] if prev else "")
                    best_by_paper[paper_key] = (card_obj, adj_score, level, rank_idx, kept_guidance)
                elif guidance_str and not prev[4]:
                    best_by_paper[paper_key] = (prev[0], prev[1], prev[2], prev[3], guidance_str)

        if not best_by_paper:
            mid_text = b.generic.get("mid", "")
            warnings.append(f"corpus gap for {b.id}: no verified card matches '{mid_text}'")
            continue

        sorted_candidates = sorted(best_by_paper.values(), key=lambda x: round(x[1], 8), reverse=True)
        top_candidates = sorted_candidates[: settings.techniques_per_bottleneck]

        for card_obj, score, level, rank_idx, guidance_str in top_candidates:
            seeds.append(
                Seed(
                    bottleneck_id=b.id,
                    card=card_obj,
                    score=score,
                    level=level,
                    rank=rank_idx,
                    guidance=guidance_str,
                )
            )

    return seeds, warnings


def resolve_card_evidence(deps: Deps, card: Card) -> tuple[int, str, dict] | None:
    """Find the chunk a card's evidence points at, as (position in deps.texts, text, metadata).

    See resolve_premise: None when the chunk cannot be found, never the paper's first chunk.
    """
    ev = card.evidence
    if ev.document != card.document:
        ev = replace(ev, document=card.document)
    return resolve_premise(deps.texts, deps.metadatas, ev, chunk_store=deps.chunk_store)


def _check_cancelled(deps: Deps) -> None:
    if deps.cancelled():
        raise pipeline_status.PipelineCancelledError("Technique transfer stopped by user")


def _gather_source_passages(
    card: Card,
    base_passages: dict[str, Passage],
    deps: Deps,
    settings: TransferSettings,
    transfer_id: str,
) -> tuple[list[Passage], dict[str, Passage], list[str]]:
    """Collect the card's primary evidence passage and additional search passages from its paper."""
    new_passages: dict[str, Passage] = {}
    new_warnings: list[str] = []

    def _next_key() -> str:
        idx = len(base_passages) + len(new_passages) + 1
        while f"C{idx}" in base_passages or f"C{idx}" in new_passages:
            idx += 1
        return f"C{idx}"

    def _find_existing(doc: str, seq_val: int) -> Passage | None:
        return _passage_of(base_passages, doc, seq_val) or _passage_of(new_passages, doc, seq_val)

    # 1. Card evidence chunk comes first. If it cannot be found it is reported, never replaced:
    #    citing some other chunk of the paper would misattribute the evidence.
    source_passages: list[Passage] = []
    seen_chunks: set[tuple[str, int]] = set()

    resolved = resolve_card_evidence(deps, card)
    if resolved is None:
        new_warnings.append(
            f"{transfer_id}: card evidence not found in {card.document}; premise unverified"
        )
    else:
        resolved_chunk_idx, resolved_text, resolved_meta = resolved
        # The resolved row's own seq: a card saved before seq existed carries only a position.
        ev_seq = chunk_seq(resolved_meta, resolved_chunk_idx)
        seen_chunks.add(_passage_key(card.document, ev_seq))
        existing = _find_existing(card.document, ev_seq)
        if existing:
            p_ev = existing
        else:
            k = _next_key()
            p_ev = Passage(
                key=k,
                chunk_index=resolved_chunk_idx,
                seq=ev_seq,
                document=card.document,
                citation=card.citation,
                section=resolved_meta.get("section", "methods"),
                text=resolved_text,
                role="source",
            )
            new_passages[k] = p_ev
        source_passages.append(p_ev)

    # 2. Search for additional source passages
    search_hits = deps.search(
        card.problem_solved,
        settings.source_passages,
        {card.document},
        settings.exclude_documents or None,
    )
    for hit in search_hits:
        c_idx = hit.get("chunk_index")
        if c_idx is None:
            continue
        meta = hit.get("metadata") or {}
        seq = chunk_seq(meta, c_idx)
        if _passage_key(card.document, seq) in seen_chunks:
            continue
        seen_chunks.add(_passage_key(card.document, seq))
        existing = _find_existing(card.document, seq)
        if existing:
            p_src = existing
        else:
            k = _next_key()
            text = hit.get("text")
            if text is None:
                found = resolve_chunk(
                    deps.texts,
                    deps.metadatas,
                    card.document,
                    seq,
                    position=c_idx,
                    chunk_store=deps.chunk_store,
                )
                text = found[1] if found else ""
            p_src = Passage(
                key=k,
                chunk_index=c_idx,
                seq=seq,
                document=card.document,
                citation=meta.get("citation", card.citation),
                section=meta.get("section", "methods"),
                text=text,
                role="source",
            )
            new_passages[k] = p_src
        source_passages.append(p_src)

    return source_passages, new_passages, new_warnings


def _run_critique(
    reply_dict: dict[str, Any],
    mapping_list: list[dict],
    steps_list: list[dict],
    ident_dict: dict[str, Any],
    deps: Deps,
    transfer_id: str,
    transfer_warnings: list[str],
) -> Critique:
    """Run adversarial critique (Stage S4) on a proposed transfer."""
    mapping_str = "\n".join(
        f"Row {i}: source_role='{r.get('source_role')}', target_role='{r.get('target_role')}', "
        f"assumption='{r.get('assumption')}', status='{r.get('status')}', core={r.get('core')}, note='{r.get('note', '')}'"
        for i, r in enumerate(mapping_list)
        if isinstance(r, dict)
    )
    steps_str = "\n".join(
        f"Step {i+1}: action='{s.get('action')}', input='{s.get('input')}', output='{s.get('output')}', tool='{s.get('tool')}'"
        for i, s in enumerate(steps_list)
        if isinstance(s, dict)
    )

    critique_prompt = CRITIQUE_USER.format(
        title=reply_dict.get("title", ""),
        summary=reply_dict.get("summary", ""),
        mapping=mapping_str,
        measurements=ident_dict.get("measurements", ""),
        unknowns=ident_dict.get("unknowns", ""),
        symmetries=", ".join(ident_dict.get("symmetries", [])) if isinstance(ident_dict.get("symmetries"), list) else str(ident_dict.get("symmetries", "")),
        identifiability_verdict=ident_dict.get("verdict", ""),
        identifiability_argument=ident_dict.get("argument", ""),
        steps=steps_str,
        baseline=reply_dict.get("baseline", ""),
        metric=reply_dict.get("metric", ""),
        prediction=reply_dict.get("prediction", ""),
        refuting_result=reply_dict.get("refuting_result", ""),
    )

    critique_messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": critique_prompt},
    ]

    _check_cancelled(deps)
    try:
        crit_dict = ask_json(
            deps.critic,
            critique_messages,
            required=("shape_match",),
            validator=validate_critique_reply,
        )
        crit_objections_raw = crit_dict.get("objections", [])
        crit_objections = []
        for obj in crit_objections_raw:
            text = obj if isinstance(obj, str) else (obj.get("text", "") if isinstance(obj, dict) else str(obj))
            crit_objections.append({
                "text": text,
                "row": None,
                "answered": False,
                "response": "",
                "conceded": False,
            })
        # Each row the critic breaks is also an objection, so the reviser must
        # rebut it or concede it; scoring reads the answer by its row.
        for cr in crit_dict.get("rows", []):
            if not isinstance(cr, dict) or str(cr.get("status", "")).strip().lower() != "breaks":
                continue
            idx = cr.get("row")
            if not (isinstance(idx, int) and 0 <= idx < len(mapping_list) and isinstance(mapping_list[idx], dict)):
                continue
            row = mapping_list[idx]
            crit_objections.append({
                "text": (
                    f"Mapping row {idx} ({row.get('source_role', '')} -> {row.get('target_role', '')}) "
                    f"breaks: {cr.get('reason', '')}"
                ),
                "row": idx,
                "answered": False,
                "response": "",
                "conceded": False,
            })
        kind = str(crit_dict.get("kind", "")).strip().lower()
        return Critique(
            shape_match=crit_dict.get("shape_match", "unknown"),
            kind=kind if kind in ("replace", "augment", "tune") else "unknown",
            rows=crit_dict.get("rows", []),
            identifiability=crit_dict.get("identifiability"),
            objections=crit_objections,
        )
    except Exception as err:
        transfer_warnings.append(f"critic failed for {transfer_id}: {err}")
        return Critique()


def _remap_critic_rows_after_revision(
    old_mapping: list[dict],
    new_mapping: list[dict],
    critique: Critique,
) -> None:
    """Remap critic row indices if revised_mapping reordered, inserted, or dropped rows."""
    # Map each old row index to its matching index in new_mapping by (source_role, target_role) or source_role
    old_to_new: dict[int, int] = {}
    for old_idx, old_row in enumerate(old_mapping):
        if not isinstance(old_row, dict):
            continue
        s_role = str(old_row.get("source_role", "")).strip().lower()
        t_role = str(old_row.get("target_role", "")).strip().lower()
        matched_idx = next(
            (
                new_idx
                for new_idx, nr in enumerate(new_mapping)
                if isinstance(nr, dict)
                and str(nr.get("source_role", "")).strip().lower() == s_role
                and str(nr.get("target_role", "")).strip().lower() == t_role
            ),
            None,
        )
        if matched_idx is None and s_role:
            matched_idx = next(
                (
                    new_idx
                    for new_idx, nr in enumerate(new_mapping)
                    if isinstance(nr, dict)
                    and str(nr.get("source_role", "")).strip().lower() == s_role
                ),
                None,
            )
        if matched_idx is not None:
            old_to_new[old_idx] = matched_idx

    for cr in critique.rows:
        if isinstance(cr, dict) and isinstance(cr.get("row"), int):
            r_idx = cr["row"]
            if r_idx in old_to_new:
                cr["row"] = old_to_new[r_idx]
            elif r_idx >= len(new_mapping):
                cr["row"] = -1

    for obj in critique.objections:
        if isinstance(obj, dict) and isinstance(obj.get("row"), int):
            r_idx = obj["row"]
            if r_idx in old_to_new:
                obj["row"] = old_to_new[r_idx]
            elif r_idx >= len(new_mapping):
                obj["row"] = -1


def _run_revision(
    reply_dict: dict[str, Any],
    critique: Critique,
    mapping_list: list[dict],
    steps_list: list[dict],
    ident_dict: dict[str, Any],
    deps: Deps,
    settings: TransferSettings,
    transfer_id: str,
    transfer_warnings: list[str],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Run revision (Stage S5a) over critic objections and return updated (mapping_list, steps_list, ident_dict)."""
    if not critique.objections:
        return mapping_list, steps_list, ident_dict

    _check_cancelled(deps)
    objections_text = "\n".join(f"{i}. {o['text']}" for i, o in enumerate(critique.objections))
    revise_prompt = REVISE_USER.format(
        title=reply_dict.get("title", ""),
        summary=reply_dict.get("summary", ""),
        objections=objections_text,
    )
    revise_messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": revise_prompt},
    ]
    try:
        rev_dict = ask_json(
            lambda msgs: deps.chat(msgs, settings.temperature),
            revise_messages,
            required=("responses",),
            validator=validate_revise_reply,
        )
        for item in rev_dict.get("responses", []):
            idx = item.get("objection")
            if isinstance(idx, int) and 0 <= idx < len(critique.objections):
                critique.objections[idx]["answered"] = True
                critique.objections[idx]["response"] = item.get("response", "")
                critique.objections[idx]["conceded"] = bool(item.get("concede", False))

        # Apply revisions to transfer fields if supplied by the reviser
        rev_mapping = rev_dict.get("mapping") or rev_dict.get("revised_mapping")
        if rev_mapping and isinstance(rev_mapping, list):
            valid_rows = [r for r in rev_mapping if isinstance(r, dict)]
            if valid_rows:
                _remap_critic_rows_after_revision(mapping_list, valid_rows, critique)
                mapping_list = valid_rows
        rev_steps = rev_dict.get("steps") or rev_dict.get("revised_steps")
        if rev_steps and isinstance(rev_steps, list):
            valid_steps = [s for s in rev_steps if isinstance(s, dict)]
            if valid_steps:
                steps_list = valid_steps
        rev_pred = rev_dict.get("prediction") or rev_dict.get("revised_prediction")
        if rev_pred and isinstance(rev_pred, str):
            reply_dict["prediction"] = rev_pred
        rev_summary = rev_dict.get("summary") or rev_dict.get("revised_summary")
        if rev_summary and isinstance(rev_summary, str):
            reply_dict["summary"] = rev_summary
        rev_title = rev_dict.get("title") or rev_dict.get("revised_title")
        if rev_title and isinstance(rev_title, str):
            reply_dict["title"] = rev_title
        rev_metric = rev_dict.get("metric") or rev_dict.get("revised_metric")
        if rev_metric and isinstance(rev_metric, str):
            reply_dict["metric"] = rev_metric
        rev_ident = rev_dict.get("identifiability") or rev_dict.get("revised_identifiability")
        if isinstance(rev_ident, dict) and rev_ident:
            ident_dict = {**ident_dict, **rev_ident}
    except Exception as err:
        transfer_warnings.append(f"revision failed for {transfer_id}: {err}")

    return mapping_list, steps_list, ident_dict


def _verify_broken_rebuttals(
    reply_dict: dict[str, Any],
    card: Card,
    critique: Critique,
    card_mechanism_str: str,
    deps: Deps,
    settings: TransferSettings,
    transfer_id: str,
    transfer_warnings: list[str],
) -> None:
    """Ask the critic (Stage S5b) whether broken-row rebuttals hold within the source mechanism."""
    broken_rebuttals = [
        (i, o)
        for i, o in enumerate(critique.objections)
        if o.get("row") is not None and o.get("answered") and not o.get("conceded")
    ]
    if not (settings.verify_rebuttals and broken_rebuttals):
        return

    rebuttals_text = "\n".join(
        f"{i}. [Row {o['row']}] Objection: {o['text']}\n   Rebuttal: {o.get('response', '')}"
        for i, o in broken_rebuttals
    )
    verify_prompt = VERIFY_REBUTTAL_USER.format(
        title=reply_dict.get("title", ""),
        card_name=card.name,
        card_mechanism=card_mechanism_str,
        rebuttals=rebuttals_text,
    )
    verify_messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": verify_prompt},
    ]
    try:
        ver_dict = ask_json(
            deps.critic,
            verify_messages,
            required=("verdicts",),
            validator=validate_verify_rebuttal_reply,
        )
        for item in ver_dict.get("verdicts", []):
            idx = item.get("objection")
            if isinstance(idx, int) and 0 <= idx < len(critique.objections):
                if critique.objections[idx].get("row") is not None:
                    critique.objections[idx]["rebuttal_accepted"] = bool(item.get("resolved", False))
                    critique.objections[idx]["rebuttal_reason"] = str(item.get("reason", ""))
    except Exception as err:
        transfer_warnings.append(f"rebuttal verification skipped for {transfer_id}: {err}")


def develop_isolated(
    seed: Seed,
    n: int,
    grounding: Grounding,
    deps: Deps,
    settings: TransferSettings,
    checker: PremiseChecker,
) -> tuple[Transfer | None, dict[str, Passage], list[str]]:
    """Thread-safe Stage S3-S5 runner that returns (transfer, new_passages, new_warnings) without mutating grounding."""
    _check_cancelled(deps)

    transfer_id = f"{seed.bottleneck_id}.T{n}"
    bottleneck = next((b for b in grounding.bottlenecks if b.id == seed.bottleneck_id), None)
    if bottleneck is None:
        return None, {}, [f"{transfer_id}: bottleneck {seed.bottleneck_id} not found"]

    card = seed.card
    base_passages = dict(grounding.passages)
    source_passages, new_passages, new_warnings = _gather_source_passages(
        card, base_passages, deps, settings, transfer_id
    )

    # 3. Prompt passages: passages behind bottleneck premises, then source passages
    near_passages: list[Passage] = []
    seen_near_keys: set[str] = set()
    for p in bottleneck.premises:
        if p.key in base_passages and p.key not in seen_near_keys:
            near_passages.append(base_passages[p.key])
            seen_near_keys.add(p.key)

    prompt_passages: list[Passage] = near_passages + source_passages

    bottleneck_premises_str = "\n".join(f"- {p.claim} ({p.key})" for p in bottleneck.premises)
    passages_str = format_passages(prompt_passages, settings.passage_max_chars)
    card_mechanism_str = "\n".join(f"- {step}" for step in card.mechanism) if card.mechanism else "- N/A"
    guidance_block = (
        f"\nLLM Cross-Disciplinary Guidance:\n{seed.guidance}\n"
        if seed.guidance
        else ""
    )

    user_prompt = TRANSFER_USER.format(
        bottleneck_statement=bottleneck.statement,
        bottleneck_quantities=", ".join(bottleneck.quantities),
        bottleneck_current_method=bottleneck.current_method,
        bottleneck_premises=bottleneck_premises_str,
        card_name=card.name,
        card_citation=card.citation,
        card_problem_solved=card.problem_solved,
        card_context=card.context,
        card_inputs=card.inputs,
        card_outputs=card.outputs,
        card_assumptions=", ".join(card.assumptions),
        card_mechanism=card_mechanism_str,
        llm_guidance_block=guidance_block,
        passages=passages_str,
    )

    required_fields = (
        "title",
        "summary",
        "mapping",
        "identifiability",
        "steps",
        "data_requirement",
        "baseline",
        "metric",
        "prediction",
        "refuting_result",
        "premises",
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    _check_cancelled(deps)
    try:
        reply_dict = ask_json(
            lambda msgs: deps.chat(msgs, settings.temperature),
            messages,
            required=required_fields,
            validator=validate_transfer_reply,
        )
    except Exception:
        new_warnings.append(f"{transfer_id}: the model's reply was not usable JSON; skipped")
        return None, new_passages, new_warnings

    transfer_warnings: list[str] = []

    # 4. Premises: resolve against prompt_passages only, then judge
    prompt_passages_dict = {p.key: p for p in prompt_passages}
    raw_premises = reply_dict.get("premises", [])
    resolved, p_warnings = resolve_premises(raw_premises, prompt_passages_dict, limit=4)
    transfer_warnings.extend(p_warnings)
    judged_premises = checker.check_all(resolved)

    mapping_list = reply_dict.get("mapping", [])
    ident_dict = reply_dict.get("identifiability", {})
    if not isinstance(ident_dict, dict):
        ident_dict = {}
    steps_list = reply_dict.get("steps", [])

    # 5. Critique
    critique = _run_critique(
        reply_dict, mapping_list, steps_list, ident_dict, deps, transfer_id, transfer_warnings
    )

    # 6. Revision
    mapping_list, steps_list, ident_dict = _run_revision(
        reply_dict, critique, mapping_list, steps_list, ident_dict, deps, settings, transfer_id, transfer_warnings
    )

    # 6b. Verify rebuttals to broken mapping rows with the critic
    _verify_broken_rebuttals(
        reply_dict, card, critique, card_mechanism_str, deps, settings, transfer_id, transfer_warnings
    )

    # 7. Prior art
    _check_cancelled(deps)
    summary = reply_dict.get("summary", "")
    try:
        prior_art = deps.prior_art(
            card.name,
            bottleneck.quantities,
            summary,
            settings.prior_art_before_year,
        )
        if getattr(prior_art, "error", None):
            transfer_warnings.append(f"prior art warning for {transfer_id}: {prior_art.error}")
    except Exception as err:
        transfer_warnings.append(f"prior art check failed for {transfer_id}: {err}")
        prior_art = PriorArt(status="unchecked", error=str(err))

    mapping_objs = [
        MappingRow(
            source_role=r.get("source_role", ""),
            target_role=r.get("target_role", ""),
            assumption=r.get("assumption", ""),
            status=r.get("status", "unknown"),
            core=bool(r.get("core", False)),
            note=r.get("note", ""),
        )
        for r in mapping_list
        if isinstance(r, dict)
    ]

    ident_obj = Identifiability(
        measurements=ident_dict.get("measurements", ""),
        unknowns=ident_dict.get("unknowns", ""),
        symmetries=ident_dict.get("symmetries", []) if isinstance(ident_dict.get("symmetries"), list) else [],
        verdict=ident_dict.get("verdict", "unclear"),
        argument=ident_dict.get("argument", ""),
    )

    step_objs = [
        Step(
            action=s.get("action", ""),
            input=s.get("input", ""),
            output=s.get("output", ""),
            tool=s.get("tool", ""),
        )
        for s in steps_list
        if isinstance(s, dict)
    ]

    transfer = Transfer(
        id=transfer_id,
        bottleneck_id=seed.bottleneck_id,
        card_id=card.card_id,
        card_document=card.document,
        level=seed.level,
        retrieval_rank=seed.rank,
        title=reply_dict.get("title", ""),
        summary=summary,
        mapping=mapping_objs,
        identifiability=ident_obj,
        steps=step_objs,
        data_requirement=reply_dict.get("data_requirement", ""),
        baseline=reply_dict.get("baseline", ""),
        metric=reply_dict.get("metric", ""),
        prediction=reply_dict.get("prediction", ""),
        refuting_result=reply_dict.get("refuting_result", ""),
        premises=judged_premises,
        critique=critique,
        prior_art=prior_art,
        scores=None,
        warnings=transfer_warnings,
    )
    return transfer, new_passages, new_warnings


def merge_developed(
    grounding: Grounding,
    transfer: Transfer | None,
    new_passages: dict[str, Passage],
    new_warnings: list[str],
) -> Transfer | None:
    """Merge thread-local passages and warnings into grounding with collision-free canonical keys."""
    grounding.warnings.extend(new_warnings)
    key_remap: dict[str, str] = {}
    for old_key, p in new_passages.items():
        existing = _passage_of(grounding.passages, p.document, p.seq) if p.seq is not None else None
        if existing is not None:
            key_remap[old_key] = existing.key
            continue
        if old_key not in grounding.passages:
            canonical = old_key
        else:
            idx = len(grounding.passages) + 1
            while f"C{idx}" in grounding.passages:
                idx += 1
            canonical = f"C{idx}"
        key_remap[old_key] = canonical
        grounding.passages[canonical] = replace(p, key=canonical)

    if transfer is not None and key_remap:
        for prem in transfer.premises:
            if prem.key in key_remap:
                prem.key = key_remap[prem.key]
    return transfer


def develop(
    seed: Seed,
    n: int,
    grounding: Grounding,
    deps: Deps,
    settings: TransferSettings,
    checker: PremiseChecker,
) -> Transfer | None:
    """Stage S3-S5: Write, judge, critique, and revise one technique transfer."""
    transfer, new_passages, new_warnings = develop_isolated(
        seed, n, grounding, deps, settings, checker
    )
    return merge_developed(grounding, transfer, new_passages, new_warnings)


