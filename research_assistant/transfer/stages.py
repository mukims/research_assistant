"""Pipeline stages for technique transfer: grounding, candidate search, development."""

from __future__ import annotations

from dataclasses import dataclass
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
from research_assistant.transfer.premises import PremiseChecker
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
        # hybrid_search keeps the paper and its citation in the chunk metadata.
        meta = h.get("metadata") or {}
        doc = meta.get("document", "")
        passages[key] = Passage(
            key=key,
            chunk_index=c_idx,
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
    exclude = frozenset(near_docs | set(settings.exclude_documents))
    seeds: list[Seed] = []
    warnings: list[str] = []

    for b in grounding.bottlenecks:
        best_by_card: dict[str, tuple[Card, float, str, int]] = {}

        for level in ("specific", "mid", "broad"):
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
    # 1. Card evidence chunk comes first.
    source_passages: list[Passage] = []
    seen_chunks: set[int] = set()

    ev_chunk = card.evidence.chunk_index
    if ev_chunk is not None and 0 <= ev_chunk < len(deps.texts):
        seen_chunks.add(ev_chunk)
        existing = next(
            (p for p in grounding.passages.values() if p.chunk_index == ev_chunk and p.document == card.document),
            None,
        )
        if existing:
            p_ev = existing
        else:
            meta = deps.metadatas[ev_chunk] if ev_chunk < len(deps.metadatas) else {}
            k = f"C{len(grounding.passages) + 1}"
            p_ev = Passage(
                key=k,
                chunk_index=ev_chunk,
                document=card.document,
                citation=card.citation,
                section=meta.get("section", "methods"),
                text=deps.texts[ev_chunk],
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
        if c_idx is None or c_idx in seen_chunks:
            continue
        seen_chunks.add(c_idx)
        existing = next(
            (p for p in grounding.passages.values() if p.chunk_index == c_idx and p.document == card.document),
            None,
        )
        if existing:
            p_src = existing
        else:
            k = f"C{len(grounding.passages) + 1}"
            meta = hit.get("metadata", {})
            p_src = Passage(
                key=k,
                chunk_index=c_idx,
                document=card.document,
                citation=meta.get("citation", card.citation),
                section=meta.get("section", "methods"),
                text=hit.get("text", deps.texts[c_idx] if c_idx < len(deps.texts) else ""),
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
                "answered": False,
                "response": "",
                "conceded": False,
            })
        critique = Critique(
            shape_match=crit_dict.get("shape_match", "unknown"),
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


