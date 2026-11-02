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
    Transfer,
)
from research_assistant.transfer.premises import PremiseChecker, chunk_seq, resolve_chunk, resolve_premise
from research_assistant.transfer.prompts import (
    BOTTLENECKS_USER,
    CRITIQUE_USER,
    REVISE_USER,
    SYSTEM,
    TRANSFER_USER,
    format_passages,
)
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
            # Anything but an explicit false keeps the identifiability gate on.
            inference=item.get("inference") is not False,
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


def candidate_cards(
    grounding: Grounding,
    deps: Deps,
    settings: TransferSettings,
) -> tuple[list[Seed], list[str]]:
    """
    Stage S2: Retrieve and rank candidate technique cards for each bottleneck.

    1. For each level in ("specific", "mid", "broad"), search cards index.
    2. Merge hits by card_id keeping the highest similarity score, level, and 1-based rank.
    3. Keep top settings.techniques_per_bottleneck cards per bottleneck.
    4. Warn if no verified cards found for a bottleneck.
    """
    near_docs = {r["document"] for r in grounding.near}
    exclude = frozenset(near_docs | copies_of(grounding.near, deps.cards.cards) | set(settings.exclude_documents))
    seeds: list[Seed] = []
    warnings: list[str] = []

    for b in grounding.bottlenecks:
        best_by_card: dict[str, tuple[Card, float, str, int]] = {}

        for level in ("specific", "mid", "broad", "inverse"):
            generic_text = b.generic.get(level, "")
            if not generic_text:
                continue
            query_vec = deps.embed_query(generic_text)
            hits = deps.cards.search(
                query_vec,
                settings.cards_per_query,
                exclude_documents=exclude,
            )
            for rank_idx, (card_obj, score) in enumerate(hits, start=1):
                cid = card_obj.card_id
                if cid not in best_by_card or score > best_by_card[cid][1]:
                    best_by_card[cid] = (card_obj, score, level, rank_idx)

        if not best_by_card:
            mid_text = b.generic.get("mid", "")
            warnings.append(f"corpus gap for {b.id}: no verified card matches '{mid_text}'")
            continue

        sorted_candidates = sorted(best_by_card.values(), key=lambda x: round(x[1], 8), reverse=True)
        top_candidates = sorted_candidates[: settings.techniques_per_bottleneck]

        for card_obj, score, level, rank_idx in top_candidates:
            seeds.append(
                Seed(
                    bottleneck_id=b.id,
                    card=card_obj,
                    score=score,
                    level=level,
                    rank=rank_idx,
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
    return resolve_premise(deps.texts, deps.metadatas, ev)


def develop(
    seed: Seed,
    n: int,
    grounding: Grounding,
    deps: Deps,
    settings: TransferSettings,
    checker: PremiseChecker,
) -> Transfer | None:
    """
    Stage S3–S5: Write, judge, critique, and revise one technique transfer.
    """
    if deps.cancelled():
        raise pipeline_status.PipelineCancelledError("Technique transfer stopped by user")

    transfer_id = f"{seed.bottleneck_id}.T{n}"
    bottleneck = next((b for b in grounding.bottlenecks if b.id == seed.bottleneck_id), None)
    if bottleneck is None:
        grounding.warnings.append(f"{transfer_id}: bottleneck {seed.bottleneck_id} not found")
        return None

    card = seed.card

    # Source passages:
    # 1. Card evidence chunk comes first. If it cannot be found it is reported, never replaced:
    #    citing some other chunk of the paper would misattribute the evidence.
    source_passages: list[Passage] = []
    seen_chunks: set[tuple[str, int]] = set()

    resolved = resolve_card_evidence(deps, card)
    if resolved is None:
        grounding.warnings.append(
            f"{transfer_id}: card evidence not found in {card.document}; premise unverified"
        )
    else:
        resolved_chunk_idx, resolved_text, resolved_meta = resolved
        # The resolved row's own seq: a card saved before seq existed carries only a position.
        ev_seq = chunk_seq(resolved_meta, resolved_chunk_idx)
        seen_chunks.add(_passage_key(card.document, ev_seq))
        existing = _passage_of(grounding.passages, card.document, ev_seq)
        if existing:
            p_ev = existing
        else:
            k = f"C{len(grounding.passages) + 1}"
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
            grounding.passages[k] = p_ev
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
        existing = _passage_of(grounding.passages, card.document, seq)
        if existing:
            p_src = existing
        else:
            k = f"C{len(grounding.passages) + 1}"
            text = hit.get("text")
            if text is None:
                found = resolve_chunk(deps.texts, deps.metadatas, card.document, seq, position=c_idx)
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
            grounding.passages[k] = p_src
        source_passages.append(p_src)

    # 3. Prompt passages:
    # Passages behind bottleneck premises, then source passages
    near_passages: list[Passage] = []
    seen_near_keys: set[str] = set()
    for p in bottleneck.premises:
        if p.key in grounding.passages and p.key not in seen_near_keys:
            near_passages.append(grounding.passages[p.key])
            seen_near_keys.add(p.key)

    prompt_passages: list[Passage] = near_passages + source_passages

    bottleneck_premises_str = "\n".join(f"- {p.claim} ({p.key})" for p in bottleneck.premises)
    passages_str = format_passages(prompt_passages, settings.passage_max_chars)
    card_mechanism_str = "\n".join(f"- {step}" for step in card.mechanism) if card.mechanism else "- N/A"

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

    try:
        reply_dict = ask_json(
            lambda msgs: deps.chat(msgs, settings.temperature),
            messages,
            required=required_fields,
        )
    except Exception:
        grounding.warnings.append(f"{transfer_id}: the model's reply was not usable JSON; skipped")
        return None

    transfer_warnings: list[str] = []

    # 4. Premises: resolve against prompt_passages only, then judge
    prompt_passages_dict = {p.key: p for p in prompt_passages}
    raw_premises = reply_dict.get("premises", [])
    resolved, p_warnings = resolve_premises(raw_premises, prompt_passages_dict, limit=4)
    transfer_warnings.extend(p_warnings)
    judged_premises = checker.check_all(resolved)

    # 5. Critique: CRITIQUE_USER goes to deps.critic
    mapping_list = reply_dict.get("mapping", [])
    mapping_str = "\n".join(
        f"Row {i}: source_role='{r.get('source_role')}', target_role='{r.get('target_role')}', "
        f"assumption='{r.get('assumption')}', status='{r.get('status')}', core={r.get('core')}, note='{r.get('note', '')}'"
        for i, r in enumerate(mapping_list)
        if isinstance(r, dict)
    )
    ident_dict = reply_dict.get("identifiability", {})
    if not isinstance(ident_dict, dict):
        ident_dict = {}
    steps_list = reply_dict.get("steps", [])
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

    critique = Critique()
    try:
        crit_dict = ask_json(deps.critic, critique_messages, required=("shape_match",))
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
        critique = Critique(
            shape_match=crit_dict.get("shape_match", "unknown"),
            kind=kind if kind in ("replace", "augment", "tune") else "unknown",
            rows=crit_dict.get("rows", []),
            identifiability=crit_dict.get("identifiability"),
            objections=crit_objections,
        )
    except Exception as err:
        transfer_warnings.append(f"critic failed for {transfer_id}: {err}")
        critique = Critique()

    # 6. Revision: REVISE_USER goes to deps.chat with the objections
    if critique.objections:
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
                mapping_list = rev_mapping
            rev_steps = rev_dict.get("steps") or rev_dict.get("revised_steps")
            if rev_steps and isinstance(rev_steps, list):
                steps_list = rev_steps
            rev_pred = rev_dict.get("prediction") or rev_dict.get("revised_prediction")
            if rev_pred and isinstance(rev_pred, str):
                reply_dict["prediction"] = rev_pred
            rev_summary = rev_dict.get("summary") or rev_dict.get("revised_summary")
            if rev_summary and isinstance(rev_summary, str):
                reply_dict["summary"] = rev_summary
            rev_title = rev_dict.get("title") or rev_dict.get("revised_title")
            if rev_title and isinstance(rev_title, str):
                reply_dict["title"] = rev_title
        except Exception as err:
            transfer_warnings.append(f"revision failed for {transfer_id}: {err}")

    # 7. Prior art
    summary = reply_dict.get("summary", "")
    try:
        prior_art = deps.prior_art(
            card.name,
            bottleneck.quantities,
            summary,
            settings.prior_art_before_year,
        )
    except Exception as err:
        transfer_warnings.append(f"prior art check failed for {transfer_id}: {err}")
        prior_art = PriorArt(status="unchecked")

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

    return Transfer(
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


