#!/usr/bin/env python3
"""
CLI runner for building technique cards across the indexed corpus.

Usage:
    python scripts/build_cards.py --sample 20 --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.config import TRANSFER_DIR, TRANSFER_OLLAMA_OPTIONS
from research_assistant.judgement.judge import judge
from research_assistant.shared import pipeline_status
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.llm import chat, get_embeddings
from research_assistant.shared.retry import retry
from research_assistant.transfer.cards import CardStore, build_all
from research_assistant.transfer.premises import PremiseChecker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build technique cards from indexed corpus.")
    parser.add_argument("--documents", nargs="*", default=None, help="Specific document IDs to process.")
    parser.add_argument("--sample", type=int, default=None, help="Sample N documents randomly.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Limit total documents to process.")

    args = parser.parse_args(argv)

    # 1. Guard against active pipelines
    status = pipeline_status.get_status()
    if status.get("active"):
        stage = status.get("stage_label", "active job")
        sys.stderr.write(f"Cannot build technique cards: pipeline is busy ({stage}).\n")
        return 2

    # 2. Load search resources
    collection, bm25, texts, metadatas = load_search_resources()

    # Collect unique documents and their citation sources
    doc_map: dict[str, str] = {}
    for meta in metadatas:
        doc = meta.get("document")
        if doc and doc not in doc_map:
            doc_map[doc] = meta.get("citation_source") or doc

    all_docs = list(doc_map.items())

    if args.documents:
        selected_set = set(args.documents)
        docs_to_build = [(d, c) for d, c in all_docs if d in selected_set]
    elif args.sample is not None:
        rng = random.Random(args.seed)
        sample_size = min(args.sample, len(all_docs))
        docs_to_build = rng.sample(all_docs, sample_size)
    else:
        docs_to_build = all_docs

    if args.limit is not None:
        docs_to_build = docs_to_build[: args.limit]

    # Setup checker and chat
    retried_judge = retry(max_retries=2, backoff=1.0)(judge)
    checker = PremiseChecker(retried_judge, texts, metadatas)

    def model_chat(messages: list[dict]) -> str:
        res = chat(messages, temperature=0.2, options=TRANSFER_OLLAMA_OPTIONS)
        return res.content or ""

    embedder = get_embeddings()
    store = CardStore(TRANSFER_DIR)

    def print_progress(cur: int, tot: int, label: str):
        sys.stderr.write(f"[{cur}/{tot}] {label}\n")

    results = build_all(
        docs_to_build,
        chat=model_chat,
        checker=checker,
        embed_documents=embedder.embed_documents,
        store=store,
        texts=texts,
        metadatas=metadatas,
        progress=print_progress,
    )

    print(
        f"Built technique cards: {results['built']} papers processed, "
        f"{results['skipped']} skipped, {results['cards']} cards created."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
