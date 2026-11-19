"""LLM-guided cross-disciplinary scouting, literature fetching, and dynamic card expansion."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable

from research_assistant import config
from research_assistant.agents import agent0_discoverer
from research_assistant.shared import academic_apis, batch_uploader, ingestion, manifest, pipeline_status
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.fetch import download_pdf, filename_for, is_saved_pdf
from research_assistant.shared.log import get_logger
from research_assistant.shared.source_key import pdf_key, source_key
from research_assistant.transfer import cards as cards_mod
from research_assistant.transfer.chunk_store import chunk_store_for
from research_assistant.transfer.deps import Deps, TransferSettings
from research_assistant.transfer.models import Bottleneck, Premise, TechniqueLead
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prompts import (
    SCOUT_TARGET_USER,
    SCOUT_TECHNIQUES_USER,
    SYSTEM,
)
from research_assistant.transfer.replies import ask_json
from research_assistant.transfer.stages import (
    Grounding,
    _is_inference_bottleneck,
    inverse_form,
)

logger = get_logger("transfer.scout")


def known_corpus_documents(deps: Deps) -> set[str]:
    """Return document IDs currently present in *deps* without scanning lazy LanceDB proxies."""
    docs: set[str] = {c.document for c in deps.cards.cards if c.document}
    if isinstance(deps.metadatas, list):
        for m in deps.metadatas:
            if isinstance(m, dict) and m.get("document"):
                docs.add(str(m["document"]))
    else:
        try:
            docs.update(manifest.load().keys())
        except Exception:
            pass
    return docs


def default_scholarly_search(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """Query arXiv and OpenAlex for open-access candidates matching *query*."""
    hits: list[dict[str, Any]] = []
    for name, fn in (
        ("arxiv", agent0_discoverer._search_arxiv),
        ("openalex", agent0_discoverer._search_openalex),
    ):
        try:
            provider_hits = fn(query, limit)
            if isinstance(provider_hits, list):
                hits.extend(provider_hits)
        except Exception as exc:
            logger.debug("Scout search on %s failed for %r: %s", name, query, exc)

    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for h in hits:
        if not isinstance(h, dict):
            continue
        key = source_key(h)
        if not key or key in seen_keys:
            continue
        pdf_url = h.get("pdf_url")
        if not pdf_url and h.get("doi") and config.UNPAYWALL_EMAIL:
            try:
                pdf_url = academic_apis.fetch_open_access_pdf_url(h["doi"])
            except Exception:
                pdf_url = None
        if not pdf_url:
            continue
        seen_keys.add(key)
        item = dict(h)
        item["key"] = key
        item["pdf_url"] = pdf_url
        out.append(item)
    return out


def fetch_and_ingest_queries(
    queries: list[str],
    *,
    max_papers: int,
    existing_docs: set[str],
    dest_dir: str | None = None,
    search_fn: Callable[[str, int], list[dict[str, Any]]] = default_scholarly_search,
    download_fn: Callable[[str, str], tuple[bool, str | None]] = download_pdf,
    ingest_fn: Callable[[dict[str, str]], dict] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Search scholarly APIs, download up to *max_papers* new OA PDFs, and ingest into LanceDB."""
    if max_papers <= 0 or not queries:
        return [], []

    target_dir = dest_dir or config.PULLED_PDFS_DIR
    os.makedirs(target_dir, exist_ok=True)
    warnings: list[str] = []

    candidates: list[dict[str, Any]] = []
    seen_doc_names: set[str] = set(existing_docs)

    for q in queries:
        if cancelled():
            raise pipeline_status.PipelineCancelledError("Technique transfer scout stopped by user")
        q_clean = str(q or "").strip()
        if not q_clean:
            continue
        try:
            found = search_fn(q_clean, max(5, max_papers * 2))
        except Exception as exc:
            warnings.append(f"Scout search failed for {q_clean!r}: {exc}")
            continue
        for cand in found:
            key = cand.get("key") or source_key(cand)
            if not key:
                continue
            fname = filename_for(key)
            doc_id = pdf_key(fname)
            if doc_id in seen_doc_names or fname in seen_doc_names:
                continue
            pdf_url = str(cand.get("pdf_url") or "").strip()
            if not pdf_url or not batch_uploader.is_safe_url(pdf_url):
                continue
            seen_doc_names.add(doc_id)
            cand_copy = dict(cand)
            cand_copy["doc_filename"] = fname
            cand_copy["doc_id"] = doc_id
            candidates.append(cand_copy)

    downloaded_map: dict[str, str] = {}
    ingested_pairs: list[tuple[str, str]] = []

    for cand in candidates:
        if len(downloaded_map) >= max_papers:
            break
        if cancelled():
            raise pipeline_status.PipelineCancelledError("Technique transfer scout stopped by user")
        dest_path = os.path.join(target_dir, cand["doc_filename"])
        title = str(cand.get("title") or "").strip()
        year = cand.get("year")
        authors = cand.get("authors") or []
        first_author = str(authors[0]).strip() if authors else ""
        if title and first_author and year:
            label = f"{first_author} ({year}) — {title}"
        elif title:
            label = title
        else:
            label = cand["doc_id"]

        if is_saved_pdf(dest_path):
            downloaded_map[dest_path] = label
            ingested_pairs.append((pdf_key(dest_path), label))
            continue

        ok, err = download_fn(cand["pdf_url"], dest_path)
        if ok and is_saved_pdf(dest_path):
            downloaded_map[dest_path] = label
            ingested_pairs.append((pdf_key(dest_path), label))
        else:
            warnings.append(f"Scout download skipped {cand.get('title') or cand['doc_id']!r}: {err}")

    if not downloaded_map:
        return [], warnings

    run_ingest = ingest_fn or (
        lambda pdf_map: ingestion.ingest_pdfs(
            pdf_map,
            workers=1,
            skip_ingested=True,
            rebuild_index=True,
            log_prefix="[TransferScout] ",
        )
    )
    try:
        ing_res = run_ingest(downloaded_map)
        failed_paths = {os.path.abspath(p) for p in (ing_res.get("failed") or [])}
        final_pairs = [
            (pdf_key(p), lbl)
            for p, lbl in downloaded_map.items()
            if os.path.abspath(p) not in failed_paths
        ]
        return final_pairs, warnings
    except Exception as exc:
        warnings.append(f"Scout ingestion failed: {exc}")
        return [], warnings


def _parse_provisional_bottlenecks(query: str, raw_items: Any, max_n: int) -> list[Bottleneck]:
    """Parse provisional bottlenecks when local + fetched target PDFs remain below min_near_papers."""
    if not isinstance(raw_items, list):
        return []
    levels = ("specific", "mid", "broad")
    out: list[Bottleneck] = []
    idx = 1
    for item in raw_items[:max_n]:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "").strip()
        quantities = [str(q).strip() for q in (item.get("quantities") or []) if str(q).strip()]
        current_method = str(item.get("current_method") or "standard domain baseline").strip()
        if not statement or not quantities:
            continue
        raw_gen = item.get("generic") if isinstance(item.get("generic"), dict) else {}
        present = {
            lvl: str(raw_gen[lvl]).strip()
            for lvl in levels
            if isinstance(raw_gen.get(lvl), str) and str(raw_gen[lvl]).strip()
        }
        fallback = next((present[lvl] for lvl in ("mid", "specific", "broad") if lvl in present), statement)
        generic = {lvl: present.get(lvl, fallback) for lvl in levels} | inverse_form(item)
        # Mark a clearly labelled provisional premise so downstream stages know its provenance
        prov_premise = Premise(
            claim=statement,
            key="C_QUERY",
            document="query_context",
            chunk_index=0,
            judgement="Partially supports",
            span_verified=True,
            supporting_span=statement,
            reason="Provisional problem formulation from user query (sparse local target corpus)",
            seq=0,
        )
        out.append(
            Bottleneck(
                id=f"B{idx}",
                statement=statement,
                quantities=quantities,
                current_method=current_method,
                generic=generic,
                premises=[prov_premise],
                inference=_is_inference_bottleneck(query, statement, item.get("inference")),
            )
        )
        idx += 1
    return out


def scout_target_topic(
    query: str,
    deps: Deps,
    settings: TransferSettings,
    *,
    existing_docs: set[str],
    search_fn: Callable[[str, int], list[dict[str, Any]]] = default_scholarly_search,
    download_fn: Callable[[str, str], tuple[bool, str | None]] = download_pdf,
    ingest_fn: Callable[[dict[str, str]], dict] | None = None,
) -> tuple[list[str], Grounding | None, list[str]]:
    """Ask LLM for target-problem search queries, fetch & ingest OA papers, and prepare fallback grounding."""
    warnings: list[str] = []
    prompt = SCOUT_TARGET_USER.format(query=query, n=settings.bottlenecks)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    search_queries: list[str] = [query]
    provisional_bottlenecks: list[Bottleneck] = []
    try:
        reply = ask_json(
            lambda msgs: deps.chat(msgs, settings.bottleneck_temperature),
            messages,
            required=("search_queries",),
        )
        raw_qs = reply.get("search_queries") or []
        if isinstance(raw_qs, list):
            parsed_qs = [str(q).strip() for q in raw_qs if str(q).strip()]
            if parsed_qs:
                search_queries = parsed_qs[:3]
        provisional_bottlenecks = _parse_provisional_bottlenecks(
            query, reply.get("provisional_bottlenecks"), settings.bottlenecks
        )
    except Exception as exc:
        warnings.append(f"Scout target query generation fell back to raw query: {exc}")

    ingested_pairs, fetch_warns = fetch_and_ingest_queries(
        search_queries,
        max_papers=settings.scout_max_papers,
        existing_docs=existing_docs,
        search_fn=search_fn,
        download_fn=download_fn,
        ingest_fn=ingest_fn,
        cancelled=deps.cancelled,
    )
    warnings.extend(fetch_warns)
    fetched_docs = [doc_id for doc_id, _ in ingested_pairs]

    fallback_grounding: Grounding | None = None
    if provisional_bottlenecks:
        from research_assistant.transfer.models import Passage
        q_passage = Passage(
            key="C_QUERY",
            chunk_index=0,
            seq=0,
            document="query_context",
            citation="User Problem Statement",
            section="problem",
            text=query,
            role="near",
        )
        fallback_grounding = Grounding(
            query=query,
            near=[],
            passages={"C_QUERY": q_passage},
            bottlenecks=provisional_bottlenecks,
            warnings=[
                "Target problem grounded provisionally via LLM scout because fewer than "
                f"{settings.min_near_papers} open-access target papers were available."
            ],
        )

    return fetched_docs, fallback_grounding, warnings


def propose_technique_leads(
    query: str,
    grounding: Grounding,
    deps: Deps,
    settings: TransferSettings,
) -> tuple[list[TechniqueLead], list[str]]:
    """Prompt LLM to suggest cross-disciplinary techniques and explain how they resolve each bottleneck."""
    if not grounding.bottlenecks:
        return [], []

    if deps.cancelled():
        raise pipeline_status.PipelineCancelledError("Technique transfer stopped by user")

    if settings.source_topic.strip():
        directive = (
            f"Source discipline / topic constraint: {settings.source_topic.strip()}\n"
            "(Prioritize techniques originating from or foundational to this discipline/topic.)"
        )
    else:
        directive = (
            "Source discipline / topic constraint: Open-ended across external scientific, mathematical, "
            "statistical, and engineering disciplines."
        )

    b_blocks = []
    valid_b_ids = {b.id for b in grounding.bottlenecks}
    default_b_id = grounding.bottlenecks[0].id
    for b in grounding.bottlenecks:
        b_blocks.append(
            f"- {b.id}: {b.statement}\n"
            f"  Quantities: {', '.join(b.quantities)}\n"
            f"  Current method: {b.current_method}\n"
            f"  Abstract problem: {b.generic.get('mid') or b.statement}"
        )

    per_b = max(1, min(3, settings.techniques_per_bottleneck))
    prompt = SCOUT_TECHNIQUES_USER.format(
        query=query,
        source_topic_directive=directive,
        bottlenecks="\n".join(b_blocks),
        k=per_b,
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
            required=("leads",),
        )
        raw_leads = reply.get("leads") or []
    except Exception as exc:
        warnings.append(f"LLM technique guidance generation failed: {exc}")
        return [], warnings

    leads: list[TechniqueLead] = []
    if not isinstance(raw_leads, list):
        return [], warnings

    topic_clean = settings.source_topic.strip()
    for item in raw_leads:
        if not isinstance(item, dict):
            continue
        t_name = str(item.get("technique_name") or "").strip()
        guidance = str(item.get("resolution_guidance") or "").strip()
        discipline = str(item.get("source_discipline") or topic_clean or "External discipline").strip()
        b_id = str(item.get("bottleneck_id") or "").strip().upper()
        if b_id not in valid_b_ids:
            b_id = default_b_id
        if not t_name or not guidance:
            continue
        raw_qs = item.get("search_queries") or []
        qs = [str(q).strip() for q in raw_qs if isinstance(q, str) and str(q).strip()][:3]
        if not qs:
            qs = [f"{t_name} {discipline}".strip()]
        if topic_clean and not any(topic_clean.lower() in q.lower() for q in qs):
            qs.insert(0, f"{topic_clean} {t_name}".strip())
        leads.append(
            TechniqueLead(
                bottleneck_id=b_id,
                technique_name=t_name,
                source_discipline=discipline,
                resolution_guidance=guidance,
                search_queries=qs[:3],
            )
        )

    return leads, warnings


def reload_live_deps(
    deps: Deps,
    card_dir: str | None = None,
    reload_resources_fn: Callable[[], tuple] = load_search_resources,
) -> Deps:
    """Refresh search resources, ChunkStore, and CardIndex on *deps* after dynamic ingestion."""
    from research_assistant.transfer.engine import make_search

    resources = reload_resources_fn()
    collection, bm25, texts, metadatas = resources
    cards_ix = cards_mod.CardStore(card_dir or config.TRANSFER_DIR).load(check_model=False)
    return dataclasses.replace(
        deps,
        search=make_search(resources),
        texts=texts,
        metadatas=metadatas,
        chunk_store=chunk_store_for(texts, metadatas),
        cards=cards_ix,
    )


def materialize_leads_into_cards(
    leads: list[TechniqueLead],
    deps: Deps,
    settings: TransferSettings,
    *,
    card_dir: str | None = None,
    existing_docs: set[str] | None = None,
    search_fn: Callable[[str, int], list[dict[str, Any]]] = default_scholarly_search,
    download_fn: Callable[[str, str], tuple[bool, str | None]] = download_pdf,
    ingest_fn: Callable[[dict[str, str]], dict] | None = None,
    reload_resources_fn: Callable[[], tuple] = load_search_resources,
) -> tuple[Deps, list[str], int, list[str]]:
    """Fetch & ingest missing source papers for *leads*, build verified technique cards, and reload *deps*."""
    if not leads or settings.scout_max_papers <= 0:
        return deps, [], 0, []

    target_card_dir = card_dir or config.TRANSFER_DIR
    store = cards_mod.CardStore(target_card_dir)
    known_docs = set(existing_docs or set()) | store.done_documents() | {c.document for c in deps.cards.cards}

    # Collect unique queries across leads, interleaved round-robin so every bottleneck gets coverage
    interleaved_queries: list[tuple[TechniqueLead, str]] = []
    max_q_len = max((len(ld.search_queries) for ld in leads), default=0)
    for q_idx in range(max_q_len):
        for ld in leads:
            if q_idx < len(ld.search_queries):
                interleaved_queries.append((ld, ld.search_queries[q_idx]))

    warnings: list[str] = []
    all_ingested: list[tuple[str, str]] = []
    doc_to_lead: dict[str, TechniqueLead] = {}

    # Cap total source papers fetched across the run to max(scout_max_papers, min(8, len(leads) * 2))
    budget = min(8, max(settings.scout_max_papers, len(leads)))

    for ld, q in interleaved_queries:
        if len(all_ingested) >= budget:
            break
        pairs, w = fetch_and_ingest_queries(
            [q],
            max_papers=min(2, budget - len(all_ingested)),
            existing_docs=known_docs,
            search_fn=search_fn,
            download_fn=download_fn,
            ingest_fn=ingest_fn,
            cancelled=deps.cancelled,
        )
        warnings.extend(w)
        for doc_id, label in pairs:
            known_docs.add(doc_id)
            all_ingested.append((doc_id, label))
            ld.fetched_documents.append(doc_id)
            doc_to_lead[doc_id] = ld

    if not all_ingested:
        return deps, [], 0, warnings

    # Reload corpus search resources so newly ingested chunks are visible for card building
    try:
        refreshed_deps = reload_live_deps(deps, card_dir=target_card_dir, reload_resources_fn=reload_resources_fn)
    except Exception as exc:
        warnings.append(f"Could not reload search resources after scouting: {exc}")
        return deps, [d for d, _ in all_ingested], 0, warnings

    checker = PremiseChecker(
        refreshed_deps.judge,
        refreshed_deps.texts,
        refreshed_deps.metadatas,
        chunk_store=refreshed_deps.chunk_store,
    )
    build_res = cards_mod.build_all(
        all_ingested,
        chat=lambda msgs: refreshed_deps.chat(msgs, 0.2),
        checker=checker,
        embed_documents=refreshed_deps.embed_documents,
        store=store,
        texts=refreshed_deps.texts,
        metadatas=refreshed_deps.metadatas,
        cancelled=refreshed_deps.cancelled,
    )
    warnings.extend(build_res.get("warnings") or [])

    updated_index = store.load(check_model=False)
    for c in updated_index.cards:
        owner_lead = doc_to_lead.get(c.document)
        if owner_lead is not None and c.card_id not in owner_lead.built_card_ids:
            owner_lead.built_card_ids.append(c.card_id)

    refreshed_deps = dataclasses.replace(refreshed_deps, cards=updated_index)
    return (
        refreshed_deps,
        [d for d, _ in all_ingested],
        int(build_res.get("cards", 0)),
        warnings,
    )
