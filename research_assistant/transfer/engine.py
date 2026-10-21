"""Engine: guards, live dependencies, call estimation, and best-of-N runner."""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone
from typing import Callable

from research_assistant import config
from research_assistant.judgement.judge import judge
from research_assistant.shared import llm, pipeline_status, retrieve
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.retry import retry
from research_assistant.shared.search import hybrid_search
from research_assistant.transfer.cards import CardStore
from research_assistant.transfer.deps import Deps, TransferSettings
from research_assistant.transfer.models import Card, RunResult, Transfer, new_run_id
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prior_art import PriorArtClient
from research_assistant.transfer.scoring import rank, score_transfer
from research_assistant.transfer.stages import candidate_cards, develop, ground


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
        cards=cards_index,
        cancelled=pipeline_status.is_cancel_requested,
    )


def estimate_calls(settings: TransferSettings) -> dict[str, int]:
    """Estimate model and judge call counts for a transfer run with given settings."""
    b = settings.bottlenecks
    t = settings.techniques_per_bottleneck
    return {
        "llm": 2 + 2 * b * t,
        "critic": b * t,
        "judge_low": 1 * b + 2 * b * t,
        "judge_high": 3 * b + 4 * b * t,
    }


def run_best_of_n(
    query: str,
    deps: Deps,
    settings: TransferSettings | None = None,
) -> RunResult:
    """Run best-of-N technique transfer search across candidate cards."""
    if settings is None:
        settings = TransferSettings()

    start_time = time.time()
    calls = {"llm": 0, "critic": 0, "judge": 0, "embed": 0, "prior_art": 0}

    def counting_chat(msgs: list[dict], temperature: float = 0.7) -> str:
        calls["llm"] += 1
        return deps.chat(msgs, temperature)

    def counting_gate(q: str, shortlist: list[dict]) -> list[dict]:
        calls["llm"] += 1
        return deps.gate(q, shortlist)

    def counting_critic(msgs: list[dict]) -> str:
        calls["critic"] += 1
        return deps.critic(msgs)

    def counting_judge(claim: str, evidence: str) -> dict:
        calls["judge"] += 1
        return deps.judge(claim, evidence)

    def counting_embed_query(text: str) -> list[float]:
        calls["embed"] += 1
        return deps.embed_query(text)

    def counting_embed_docs(texts: list[str]) -> list[list[float]]:
        calls["embed"] += 1
        return deps.embed_documents(texts)

    def counting_prior_art(tech: str, quant: list[str], summary: str, before_year: int | None = None):
        calls["prior_art"] += 1
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

    checker = PremiseChecker(wrapped_deps.judge, wrapped_deps.texts, wrapped_deps.metadatas)

    # 1. Grounding
    grounding = ground(query, wrapped_deps, settings, checker)

    # 2. Candidate cards
    seeds, cand_warnings = candidate_cards(grounding, wrapped_deps, settings)

    total_progress = 2 + len(seeds)
    deps.progress(1, total_progress, "Grounding bottlenecks")
    deps.progress(2, total_progress, "Retrieving candidate technique cards")

    # 3. Develop each seed
    transfers: list[Transfer] = []
    card_map: dict[str, Card] = {}
    bottleneck_map = {b.id: b for b in grounding.bottlenecks}
    bottleneck_counts: dict[str, int] = {}

    for idx, seed in enumerate(seeds, start=1):
        b_id = seed.bottleneck_id
        bottleneck_counts[b_id] = bottleneck_counts.get(b_id, 0) + 1
        n = bottleneck_counts[b_id]
        t = develop(seed, n, grounding, wrapped_deps, settings, checker)
        if t is not None:
            card_map[seed.card.card_id] = seed.card
            b_obj = bottleneck_map.get(b_id)
            if b_obj is not None:
                score_transfer(t, b_obj, seed.card)
            transfers.append(t)
        deps.progress(2 + idx, total_progress, f"Developing transfer {b_id}.T{n}")

    ranked_transfers = rank(transfers)

    generator_model = getattr(config, "LLM_MODEL", "unknown")
    critic_model = getattr(config, "TRANSFER_CRITIC_MODEL", None) or generator_model
    judge_model = getattr(config, "JUDGEMENT_MODEL", "unknown")
    models = {
        "generator": generator_model,
        "critic": critic_model,
        "judge": judge_model,
    }

    run_warnings = list(grounding.warnings) + cand_warnings
    if getattr(config, "TRANSFER_CRITIC_MODEL", None) is None:
        run_warnings.append("critic shares the generator's model")

    cards_used = {t.card_id: card_map[t.card_id] for t in ranked_transfers if t.card_id in card_map}

    return RunResult(
        run_id=new_run_id(query),
        query=query,
        search="best_of_n",
        created_at=datetime.now(timezone.utc).isoformat(),
        models=models,
        settings=dataclasses.asdict(settings),
        near=grounding.near,
        passages=grounding.passages,
        bottlenecks=grounding.bottlenecks,
        cards=cards_used,
        transfers=ranked_transfers,
        calls=calls,
        seconds=round(time.time() - start_time, 2),
        warnings=run_warnings,
    )
