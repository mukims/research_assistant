"""Engine: guards, live dependencies, call estimation, and best-of-N runner."""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

from research_assistant import config
from research_assistant.judgement.judge import judge
from research_assistant.shared import llm, pipeline_status, retrieve
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.log import get_logger
from research_assistant.shared.retry import retry
from research_assistant.shared.search import hybrid_search
from research_assistant.transfer import scout
from research_assistant.transfer.cards import CardStore
from research_assistant.transfer.chunk_store import chunk_store_for
from research_assistant.transfer.deps import Deps, TransferSettings
from research_assistant.transfer.models import Card, RunResult, TechniqueLead, Transfer, new_run_id
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prior_art import PriorArtClient
from research_assistant.transfer.scoring import rank, score_transfer
from research_assistant.transfer.stages import (
    CorpusCoverageError,
    Seed,
    candidate_cards,
    develop,
    develop_isolated,
    ground,
    merge_developed,
)

logger = get_logger("transfer")


class BusyError(RuntimeError):
    """Raised when an operation is attempted while another pipeline run is active."""


def ensure_idle() -> None:
    """Ensure no other pipeline run is currently active, raising BusyError otherwise."""
    status = pipeline_status.get_status()
    if status.get("active"):
        stage_label = status.get("stage_label", "active run")
        raise BusyError(f"Another run is active: {stage_label}. Wait for it to finish.")


def make_search(resources: tuple) -> Callable:
    """Return search function using hybrid_search with figures and non-text excluded."""
    collection, bm25, texts, metadatas = resources
    return lambda q, k, docs, excl: hybrid_search(
        q,
        collection,
        bm25,
        texts,
        metadatas,
        top_k=k,
        doc_filter=set(docs) if docs else None,
        exclude_types=config.EXCLUDED_FROM_EVIDENCE,
        exclude_docs=set(excl) if excl else None,
    )


def live_deps(search_resources=None, card_dir=None) -> Deps:
    """Assemble live production dependencies for technique transfer."""
    if search_resources is None:
        search_resources = load_search_resources()
    collection, bm25, texts, metadatas = search_resources

    search_fn = make_search(search_resources)

    def live_chat(messages: list[dict], temperature: float = 0.7) -> str:
        res = llm.chat(messages, temperature=temperature, options=config.TRANSFER_OLLAMA_OPTIONS)
        return res.content or ""

    def live_critic(messages: list[dict]) -> str:
        res = llm.chat(
            messages,
            model=config.TRANSFER_CRITIC_MODEL or None,
            temperature=0.0,
            options=config.TRANSFER_OLLAMA_OPTIONS,
        )
        return res.content or ""

    retried_judge = retry(max_retries=2, backoff=1.0)(judge)

    embedder = llm.get_embeddings()

    cards_index = CardStore(card_dir or config.TRANSFER_DIR).load()
    prior_art_client = PriorArtClient(embed_documents=embedder.embed_documents)

    return Deps(
        chat=live_chat,
        critic=live_critic,
        judge=retried_judge,
        embed_query=embedder.embed_query,
        embed_documents=embedder.embed_documents,
        shortlist=retrieve.rank_documents,
        gate=retrieve.gate_documents,
        search=search_fn,
        prior_art=prior_art_client.check,
        texts=texts,
        metadatas=metadatas,
        chunk_store=chunk_store_for(texts, metadatas),
        cards=cards_index,
        cancelled=pipeline_status.is_cancel_requested,
    )


def estimate_calls(settings: TransferSettings) -> dict[str, int]:
    """Estimate model and judge call counts for a transfer run with given settings."""
    b = settings.bottlenecks
    t = settings.techniques_per_bottleneck
    scout_llm = (2 if settings.enable_scout else 0) or (1 if settings.source_topic.strip() else 0)
    return {
        "llm": 2 + 2 * b * t + scout_llm,
        "critic": b * t,
        "judge_low": 1 * b + 2 * b * t,
        "judge_high": 3 * b + 4 * b * t,
    }


def run_best_of_n(
    query: str,
    deps: Deps,
    settings: TransferSettings | None = None,
    *,
    card_dir: str | None = None,
    scout_hooks: dict | None = None,
) -> RunResult:
    """Run best-of-N technique transfer search across candidate cards."""
    if settings is None:
        settings = TransferSettings()

    start_time = time.time()
    calls_lock = threading.Lock()
    calls = {"llm": 0, "critic": 0, "judge": 0, "embed": 0, "prior_art": 0}
    logger.info(
        "Starting technique transfer run for query=%r (bottlenecks=%d, techniques=%d, cards_per_query=%d, scout=%s, source_topic=%r)",
        query,
        settings.bottlenecks,
        settings.techniques_per_bottleneck,
        settings.cards_per_query,
        settings.enable_scout,
        settings.source_topic,
    )

    def _inc(key: str) -> None:
        with calls_lock:
            calls[key] += 1

    def counting_chat(msgs: list[dict], temperature: float = 0.7) -> str:
        _inc("llm")
        return deps.chat(msgs, temperature)

    def counting_gate(q: str, shortlist: list[dict]) -> list[dict]:
        _inc("llm")
        return deps.gate(q, shortlist)

    def counting_critic(msgs: list[dict]) -> str:
        _inc("critic")
        return deps.critic(msgs)

    def counting_judge(claim: str, evidence: str) -> dict:
        _inc("judge")
        return deps.judge(claim, evidence)

    def counting_embed_query(text: str) -> list[float]:
        _inc("embed")
        return deps.embed_query(text)

    def counting_embed_docs(texts: list[str]) -> list[list[float]]:
        _inc("embed")
        return deps.embed_documents(texts)

    def counting_prior_art(tech: str, quant: list[str], summary: str, before_year: int | None = None):
        _inc("prior_art")
        return deps.prior_art(tech, quant, summary, before_year)

    wrapped_deps = dataclasses.replace(
        deps,
        chat=counting_chat,
        gate=counting_gate,
        critic=counting_critic,
        judge=counting_judge,
        embed_query=counting_embed_query,
        embed_documents=counting_embed_docs,
        prior_art=counting_prior_art,
    )

    checker = PremiseChecker(
        wrapped_deps.judge,
        wrapped_deps.texts,
        wrapped_deps.metadatas,
        chunk_store=wrapped_deps.chunk_store,
    )

    scout_warnings: list[str] = []
    target_fetched: list[str] = []
    source_fetched: list[str] = []
    cards_built = 0
    leads: list[TechniqueLead] = []
    hooks = scout_hooks or {}

    # 1. Grounding (with automatic target literature scouting if sparse and enable_scout=True)
    try:
        grounding = ground(query, wrapped_deps, settings, checker)
    except CorpusCoverageError:
        if not settings.enable_scout:
            raise
        deps.progress(0, 3, "Scouting target literature (arXiv / OpenAlex)")
        target_kw = {k: hooks[k] for k in ("search_fn", "download_fn", "ingest_fn") if k in hooks}
        target_fetched, fallback_grounding, t_warns = scout.scout_target_topic(
            query,
            wrapped_deps,
            settings,
            existing_docs=scout.known_corpus_documents(wrapped_deps),
            **target_kw,
        )
        scout_warnings.extend(t_warns)
        grounding = None
        if target_fetched:
            reload_kw = {"reload_resources_fn": hooks["reload_resources_fn"]} if "reload_resources_fn" in hooks else {}
            try:
                wrapped_deps = scout.reload_live_deps(wrapped_deps, card_dir=card_dir, **reload_kw)
                checker = PremiseChecker(
                    wrapped_deps.judge,
                    wrapped_deps.texts,
                    wrapped_deps.metadatas,
                    chunk_store=wrapped_deps.chunk_store,
                )
                grounding = ground(query, wrapped_deps, settings, checker)
            except CorpusCoverageError:
                grounding = None
            except Exception as reload_exc:
                scout_warnings.append(f"Target corpus reload warning: {reload_exc}")
                grounding = None
        if grounding is None:
            if fallback_grounding is not None:
                grounding = fallback_grounding
            else:
                raise

    # 1b. LLM Cross-Disciplinary Guidance & Dynamic Source Card Materialization
    if settings.enable_scout or settings.source_topic.strip():
        leads, l_warns = scout.propose_technique_leads(query, grounding, wrapped_deps, settings)
        scout_warnings.extend(l_warns)
        if settings.enable_scout and leads:
            deps.progress(1, 3, "Fetching cross-disciplinary papers & building technique cards")
            mat_kw = {
                k: hooks[k]
                for k in ("search_fn", "download_fn", "ingest_fn", "reload_resources_fn")
                if k in hooks
            }
            wrapped_deps, source_fetched, cards_built, m_warns = scout.materialize_leads_into_cards(
                leads,
                wrapped_deps,
                settings,
                card_dir=card_dir,
                existing_docs=scout.known_corpus_documents(wrapped_deps),
                **mat_kw,
            )
            scout_warnings.extend(m_warns)
            checker = PremiseChecker(
                wrapped_deps.judge,
                wrapped_deps.texts,
                wrapped_deps.metadatas,
                chunk_store=wrapped_deps.chunk_store,
            )

    # 2. Candidate cards
    seeds, cand_warnings = candidate_cards(grounding, wrapped_deps, settings, leads=leads)

    total_progress = 2 + len(seeds)
    deps.progress(1, total_progress, "Grounding bottlenecks")
    deps.progress(2, total_progress, "Retrieving candidate technique cards")

    # 3. Develop each seed (in parallel across independent seeds when max_workers > 1)
    transfers: list[Transfer] = []
    card_map: dict[str, Card] = {}
    bottleneck_map = {b.id: b for b in grounding.bottlenecks}
    bottleneck_counts: dict[str, int] = {}
    work: list[tuple[int, Seed, int]] = []
    for idx, seed in enumerate(seeds, start=1):
        b_id = seed.bottleneck_id
        bottleneck_counts[b_id] = bottleneck_counts.get(b_id, 0) + 1
        work.append((idx, seed, bottleneck_counts[b_id]))

    completed: dict[int, tuple] = {}
    if settings.max_workers <= 1 or len(work) <= 1:
        for idx, seed, n in work:
            completed[idx] = develop_isolated(seed, n, grounding, wrapped_deps, settings, checker)
            deps.progress(2 + idx, total_progress, f"Developing transfer {seed.bottleneck_id}.T{n}")
    else:
        workers = min(settings.max_workers, len(work))
        done_count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(develop_isolated, seed, n, grounding, wrapped_deps, settings, checker): (idx, seed, n)
                for idx, seed, n in work
            }
            try:
                for future in as_completed(future_map):
                    idx, seed, n = future_map[future]
                    completed[idx] = future.result()
                    done_count += 1
                    deps.progress(2 + done_count, total_progress, f"Developing transfer {seed.bottleneck_id}.T{n}")
            except Exception:
                for pending in future_map:
                    pending.cancel()
                raise

    for idx, seed, n in work:
        t, new_passages, new_warnings = completed[idx]
        merged = merge_developed(grounding, t, new_passages, new_warnings)
        if merged is not None:
            card_map[seed.card.card_id] = seed.card
            b_obj = bottleneck_map.get(seed.bottleneck_id)
            if b_obj is not None:
                score_transfer(merged, b_obj, seed.card)
            transfers.append(merged)

    ranked_transfers = rank(transfers)

    generator_model = getattr(config, "LLM_MODEL", "unknown")
    critic_model = getattr(config, "TRANSFER_CRITIC_MODEL", None) or generator_model
    judge_model = getattr(config, "JUDGEMENT_MODEL", "unknown")
    models = {
        "generator": generator_model,
        "critic": critic_model,
        "judge": judge_model,
    }

    run_warnings = scout_warnings + list(grounding.warnings) + cand_warnings
    if getattr(config, "TRANSFER_CRITIC_MODEL", None) is None:
        run_warnings.append("critic shares the generator's model")

    cards_used = {t.card_id: card_map[t.card_id] for t in ranked_transfers if t.card_id in card_map}
    discovery_info = (
        {
            "enabled": bool(settings.enable_scout),
            "target_fetched": target_fetched,
            "source_fetched": source_fetched,
            "cards_built": cards_built,
        }
        if (settings.enable_scout or settings.source_topic.strip() or leads)
        else {}
    )

    result = RunResult(
        run_id=new_run_id(query),
        query=query,
        search="best_of_n",
        created_at=datetime.now(timezone.utc).isoformat(),
        models=models,
        # JSON has no sets: the saved run lists the excluded documents.
        settings={
            **dataclasses.asdict(settings),
            "exclude_documents": sorted(settings.exclude_documents),
        },
        near=grounding.near,
        passages=grounding.passages,
        bottlenecks=grounding.bottlenecks,
        cards=cards_used,
        transfers=ranked_transfers,
        calls=calls,
        seconds=round(time.time() - start_time, 2),
        warnings=run_warnings,
        source_topic=settings.source_topic.strip(),
        leads=leads,
        discovery=discovery_info,
    )
    logger.info(
        "Completed technique transfer run %s in %.2fs (%d transfers, %d leads, %d warnings)",
        result.run_id,
        result.seconds,
        len(result.transfers),
        len(result.leads),
        len(result.warnings),
    )
    return result


class TransferEngine:
    """High-level runner for Cross-Disciplinary Technique Transfer."""

    def __init__(self, deps: Deps | None = None, card_dir: str | None = None):
        self._deps = deps
        self._live_loaded = deps is None
        self._card_dir = card_dir

    def get_deps(self) -> Deps:
        if self._deps is None:
            self._deps = live_deps(card_dir=self._card_dir)
        return self._deps

    def run(
        self,
        query: str,
        candidates: int = 3,
        temperature: float = 0.2,
        prior_art_before_year: int | None = None,
        bottlenecks: int | None = None,
        techniques_per_bottleneck: int | None = None,
        source_topic: str = "",
        enable_scout: bool = False,
        scout_max_papers: int | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> RunResult:
        ensure_idle()
        # Only past the busy guard: a cancel pending against another live run is not stale,
        # and clearing it here would un-stop that run. With no run active, it is.
        pipeline_status.clear_cancel_request()
        deps = self.get_deps()
        if progress_cb is not None:
            deps = dataclasses.replace(deps, progress=progress_cb)

        kw: dict = {
            "cards_per_query": candidates,
            "temperature": temperature,
            "prior_art_before_year": prior_art_before_year,
            "source_topic": source_topic.strip(),
            "enable_scout": bool(enable_scout),
        }
        if bottlenecks is not None:
            kw["bottlenecks"] = bottlenecks
        if techniques_per_bottleneck is not None:
            kw["techniques_per_bottleneck"] = techniques_per_bottleneck
        if scout_max_papers is not None:
            kw["scout_max_papers"] = scout_max_papers

        settings = TransferSettings(**kw)
        with pipeline_status.track_stage(
            "transfer",
            f"Technique transfer: {query[:60]}",
            total_steps=1,
            mark_idle_on_exit=True,
        ):
            res = run_best_of_n(query, deps, settings, card_dir=self._card_dir)
            if self._live_loaded and (
                res.discovery.get("target_fetched")
                or res.discovery.get("source_fetched")
                or res.discovery.get("cards_built")
            ):
                self._deps = None
            return res


def default_engine() -> TransferEngine:
    """Return a standard TransferEngine instance."""
    return TransferEngine()

