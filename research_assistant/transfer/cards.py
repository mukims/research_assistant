"""Technique cards: selection, extraction, storage, and search."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from research_assistant.config import (
    EXCLUDED_FROM_EVIDENCE,
    TRANSFER_CARD_MAX_CHARS,
    TRANSFER_CARD_MAX_TECHNIQUES,
    TRANSFER_CARD_MIN_MECHANISM_STEPS,
    TRANSFER_PASSAGE_MAX_CHARS,
)
from research_assistant.shared.atomic import atomic_write, atomic_write_json
from research_assistant.transfer.models import Card, Passage, card_from_dict, card_to_dict
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prompts import CARDS_USER, SYSTEM, format_passages
from research_assistant.transfer.replies import ask_json, resolve_premises


def select_card_passages(
    document: str,
    citation: str,
    texts: list[str],
    metadatas: list[dict],
    max_chars: int = TRANSFER_CARD_MAX_CHARS,
) -> list[Passage]:
    """
    Select candidate passages from a document for technique extraction.

    Order:
      1. abstract chunks
      2. methods chunks
      3. first chunk under each section_raw heading not already seen
    Excludes figure images, captions, and non-document chunks.
    Stops before character budget is exceeded.
    """
    doc_indices = []
    for idx, meta in enumerate(metadatas):
        if meta.get("document") != document:
            continue
        if meta.get("type") in EXCLUDED_FROM_EVIDENCE:
            continue
        if meta.get("section") == "caption":
            continue
        doc_indices.append(idx)

    # 1. abstract chunks
    abstract_indices = [
        i for i in doc_indices if metadatas[i].get("section") == "abstract"
    ]
    # 2. methods chunks
    methods_indices = [
        i for i in doc_indices if metadatas[i].get("section") == "methods"
    ]

    # 3. first chunk under each distinct section_raw not already used
    used = set(abstract_indices) | set(methods_indices)
    heading_indices = []
    seen_headings = set()
    for i in doc_indices:
        if i in used:
            continue
        raw = metadatas[i].get("section_raw") or metadatas[i].get("section") or ""
        if raw and raw not in seen_headings:
            seen_headings.add(raw)
            heading_indices.append(i)

    candidate_indices = abstract_indices + methods_indices + heading_indices

    passages: list[Passage] = []
    total_chars = 0
    key_counter = 1

    for idx in candidate_indices:
        txt = texts[idx]
        if total_chars + len(txt) > max_chars:
            break
        meta = metadatas[idx]
        sec = meta.get("section_raw") or meta.get("section") or "other"
        passages.append(
            Passage(
                key=f"C{key_counter}",
                chunk_index=idx,
                document=document,
                citation=citation,
                section=sec,
                text=txt,
                role="card",
            )
        )
        total_chars += len(txt)
        key_counter += 1

    return passages


def build_card(
    document: str,
    citation: str,
    passages: list[Passage],
    *,
    chat: Callable[[list[dict]], str],
    checker: PremiseChecker,
) -> tuple[list[Card], list[str]]:
    """Prompt the model for reusable techniques in a paper and verify them."""
    if not passages:
        return [], []

    passages_text = format_passages(passages, TRANSFER_PASSAGE_MAX_CHARS)
    prompt = CARDS_USER.format(
        citation=citation,
        passages=passages_text,
        n=TRANSFER_CARD_MAX_TECHNIQUES,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    reply_data = ask_json(chat, messages, required=("techniques",))
    raw_techniques = reply_data.get("techniques", [])

    passages_by_key = {p.key: p for p in passages}
    cards: list[Card] = []
    warnings: list[str] = []
    card_index = 0

    for tech in raw_techniques:
        mechanism = [
            str(s).strip() for s in tech.get("mechanism") or [] if str(s).strip()
        ]
        if len(mechanism) < TRANSFER_CARD_MIN_MECHANISM_STEPS:
            warnings.append(
                f"Technique {tech.get('name', '')!r} dropped: its mechanism has "
                f"{len(mechanism)} step(s), {TRANSFER_CARD_MIN_MECHANISM_STEPS} needed"
            )
            continue

        ev = tech.get("evidence", {})
        resolved, w = resolve_premises([ev], passages_by_key, limit=1)
        warnings.extend(w)
        if not resolved:
            warnings.append(
                f"Evidence for technique {tech.get('name', '')!r} did not resolve: {ev}"
            )
            continue

        prem = resolved[0]
        checker.check(prem)

        card_id = f"{document}#{card_index}"
        cards.append(
            Card(
                card_id=card_id,
                document=document,
                citation=citation,
                name=tech.get("name", "Unknown technique"),
                problem_solved=tech.get("problem_solved", ""),
                context=tech.get("context", ""),
                inputs=tech.get("inputs", ""),
                outputs=tech.get("outputs", ""),
                assumptions=tech.get("assumptions", []),
                evidence=prem,
                mechanism=mechanism,
            )
        )
        card_index += 1

    return cards, warnings


class CardIndex:
    """In-memory index of technique cards and their problem_solved vectors."""

    def __init__(self, cards: list[Card], vectors: np.ndarray):
        self.cards = cards
        self.vectors = vectors

    def search(
        self,
        vector: np.ndarray,
        k: int,
        exclude_documents: frozenset[str] = frozenset(),
        verified_only: bool = True,
    ) -> list[tuple[Card, float]]:
        """Search cards by vector cosine similarity."""
        results: list[tuple[Card, float]] = []
        if len(self.cards) == 0 or len(self.vectors) == 0:
            return results

        vec = np.asarray(vector, dtype=float)
        v_norm = np.linalg.norm(vec)
        if v_norm == 0:
            v_norm = 1.0

        for idx, c in enumerate(self.cards):
            if c.document in exclude_documents:
                continue
            if verified_only and not c.verified:
                continue

            card_vec = self.vectors[idx]
            c_norm = np.linalg.norm(card_vec)
            if c_norm == 0:
                c_norm = 1.0

            sim = float(np.dot(card_vec, vec) / (c_norm * v_norm))
            results.append((c, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class CardStore:
    """Persistent storage for technique cards and vectors under DATA_DIR/transfer."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.jsonl_path = os.path.join(directory, "cards.jsonl")
        self.npy_path = os.path.join(directory, "cards.npy")
        self.manifest_path = os.path.join(directory, "manifest.json")

    def done_documents(self) -> set[str]:
        if not os.path.exists(self.manifest_path):
            return set()
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return set(data.keys())
        except Exception:
            return set()

    def append(self, document: str, cards: list[Card], vectors: np.ndarray) -> None:
        """Append cards and vectors for a document, updating manifest atomically."""
        existing_index = self.load()
        all_cards = list(existing_index.cards) + list(cards)

        if len(existing_index.vectors) > 0 and len(vectors) > 0:
            all_vectors = np.vstack([existing_index.vectors, vectors])
        elif len(vectors) > 0:
            all_vectors = np.asarray(vectors, dtype=float)
        elif len(existing_index.vectors) > 0:
            all_vectors = existing_index.vectors
        else:
            dim = vectors.shape[1] if vectors.ndim == 2 else 0
            all_vectors = np.empty((0, dim), dtype=float)

        manifest = {}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except Exception:
                manifest = {}

        manifest[document] = {
            "cards": len(cards),
            "at": datetime.now(timezone.utc).isoformat(),
        }

        # Write cards.jsonl
        lines = [json.dumps(card_to_dict(c)) + "\n" for c in all_cards]
        with atomic_write(self.jsonl_path) as fh:
            fh.write("".join(lines))

        # Write cards.npy
        temp_npy = os.path.join(self.directory, "cards_tmp.npy")
        np.save(temp_npy, all_vectors)
        os.replace(temp_npy, self.npy_path)

        # Write manifest.json
        atomic_write_json(self.manifest_path, manifest)

    def load(self) -> CardIndex:
        cards: list[Card] = []
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        cards.append(card_from_dict(json.loads(line)))

        vectors = np.empty((0, 0), dtype=float)
        if os.path.exists(self.npy_path):
            try:
                vectors = np.load(self.npy_path)
            except Exception:
                vectors = np.empty((0, 0), dtype=float)

        return CardIndex(cards, vectors)


def build_all(
    documents: list[tuple[str, str]],
    *,
    chat: Callable[[list[dict]], str],
    checker: PremiseChecker,
    embed_documents: Callable[[list[str]], list[list[float]]],
    store: CardStore,
    texts: list[str],
    metadatas: list[dict],
    max_chars: int = TRANSFER_CARD_MAX_CHARS,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int, str], None] = lambda d, t, label: None,
) -> dict:
    """Build technique cards across a list of (document, citation) pairs."""
    done = store.done_documents()
    built = 0
    skipped = 0
    total_cards = 0
    failed: list[str] = []
    all_warnings: list[str] = []

    total_docs = len(documents)
    for i, (doc, cit) in enumerate(documents, start=1):
        if cancelled():
            break
        if doc in done:
            skipped += 1
            progress(i, total_docs, f"Skipped (already done): {doc}")
            continue

        progress(i, total_docs, f"Building cards for: {doc}")
        passages = select_card_passages(doc, cit, texts, metadatas, max_chars)
        try:
            new_cards, warnings = build_card(
                doc, cit, passages, chat=chat, checker=checker
            )
        except Exception as exc:
            # Not recorded as done, so the next run retries this paper.
            failed.append(doc)
            all_warnings.append(f"{doc}: no cards built ({type(exc).__name__}: {exc})")
            continue
        all_warnings.extend(warnings)

        if new_cards:
            vecs = embed_documents([c.problem_solved for c in new_cards])
            vecs_arr = np.asarray(vecs, dtype=float)
            norms = np.linalg.norm(vecs_arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs_norm = vecs_arr / norms
            store.append(doc, new_cards, vecs_norm)
        else:
            store.append(doc, [], np.empty((0, 0), dtype=float))

        built += 1
        total_cards += len(new_cards)

    return {
        "built": built,
        "skipped": skipped,
        "cards": total_cards,
        "failed": failed,
        "warnings": all_warnings,
    }
