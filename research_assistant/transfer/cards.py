"""Technique cards: selection, extraction, storage, and search."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from research_assistant import config
from research_assistant.config import (
    EXCLUDED_FROM_EVIDENCE,
    TRANSFER_CARD_MAX_CHARS,
    TRANSFER_CARD_MAX_TECHNIQUES,
    TRANSFER_CARD_MIN_MECHANISM_STEPS,
    TRANSFER_PASSAGE_MAX_CHARS,
)
from research_assistant.shared.atomic import atomic_write, atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.transfer.chunk_store import ChunkStore, chunk_seq, chunk_store_for
from research_assistant.transfer.models import Card, Passage, card_from_dict, card_to_dict
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prompts import CARDS_USER, SYSTEM, format_passages
from research_assistant.transfer.replies import ask_json, resolve_premises

logger = get_logger("transfer")
_STORE_LOCK = threading.RLock()


def select_card_passages(
    document: str,
    citation: str,
    texts: list[str],
    metadatas: list[dict],
    max_chars: int = TRANSFER_CARD_MAX_CHARS,
    chunk_store: ChunkStore | None = None,
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
    store = chunk_store or chunk_store_for(texts, metadatas)
    doc_rows = store.get_document_chunks(document)
    source_texts = [txt for _, txt, _ in doc_rows]
    source_metas = [meta for _, _, meta in doc_rows]
    row_positions = [pos for pos, _, _ in doc_rows]
    doc_indices = list(range(len(doc_rows)))

    # 1. abstract chunks
    abstract_indices = [
        i for i in doc_indices if source_metas[i].get("section") == "abstract"
    ]
    # 2. methods chunks
    methods_indices = [
        i for i in doc_indices if source_metas[i].get("section") == "methods"
    ]

    # 3. first chunk under each distinct section_raw not already used
    used = set(abstract_indices) | set(methods_indices)
    heading_indices = []
    seen_headings = set()
    for i in doc_indices:
        if i in used:
            continue
        raw = source_metas[i].get("section_raw") or source_metas[i].get("section") or ""
        if raw and raw not in seen_headings:
            seen_headings.add(raw)
            heading_indices.append(i)

    candidate_indices = abstract_indices + methods_indices + heading_indices

    passages: list[Passage] = []
    total_chars = 0
    key_counter = 1

    for i in candidate_indices:
        txt = source_texts[i]
        if total_chars + len(txt) > max_chars:
            break
        meta = source_metas[i]
        pos = row_positions[i]
        seq = chunk_seq(meta, pos)
        sec = meta.get("section_raw") or meta.get("section") or "other"
        passages.append(
            Passage(
                key=f"C{key_counter}",
                chunk_index=pos,
                seq=seq,
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


def card_search_text(card: Card) -> str:
    """Build composite text for embedding and lexical matching of a technique card."""
    parts = [card.problem_solved.strip()]
    if card.inputs:
        parts.append(f"Inputs: {card.inputs.strip()}")
    if card.outputs:
        parts.append(f"Outputs: {card.outputs.strip()}")
    if card.mechanism:
        mech = "; ".join(s.strip() for s in card.mechanism if s.strip())
        if mech:
            parts.append(f"Mechanism: {mech}")
    return "\n".join(p for p in parts if p)


def _search_tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", text or "")}


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
        query_text: str = "",
    ) -> list[tuple[Card, float]]:
        """Search cards by dense cosine similarity, blended with lexical overlap when query_text is set."""
        results: list[tuple[Card, float]] = []
        if len(self.cards) == 0 or len(self.vectors) == 0:
            return results

        vec = np.asarray(vector, dtype=float)
        v_norm = np.linalg.norm(vec)
        if v_norm == 0:
            v_norm = 1.0

        q_tokens = _search_tokens(query_text) if query_text else set()

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
            if q_tokens:
                c_tokens = _search_tokens(card_search_text(c))
                lex = len(q_tokens & c_tokens) / len(q_tokens) if c_tokens else 0.0
                sim = 0.85 * sim + 0.15 * lex
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
        self.meta_path = os.path.join(directory, "cards_meta.json")

    def done_documents(self) -> set[str]:
        if not os.path.exists(self.manifest_path):
            return set()
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return set(data.keys())
        except Exception:
            return set()

    def load_meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_meta(self, meta: dict) -> None:
        atomic_write_json(self.meta_path, meta)

    def check_embedder(self) -> None:
        """Validate stored card vectors match configured EMBED_MODEL (mirroring db.check_embedder)."""
        meta = self.load_meta()
        stored = meta.get("embedder")
        if isinstance(stored, str) and stored and stored != config.EMBED_MODEL:
            raise RuntimeError(
                f"CardStore at {self.directory!r} was embedded with {stored!r} but "
                f"CITATION_EMBED_MODEL is {config.EMBED_MODEL!r}. Set CITATION_EMBED_MODEL={stored} "
                "or rebuild cards."
            )

    def _write_all_locked(self, cards: list[Card], vectors: np.ndarray, manifest: dict) -> None:
        lines = [json.dumps(card_to_dict(c)) + "\n" for c in cards]
        with atomic_write(self.jsonl_path) as fh:
            fh.write("".join(lines))

        temp_npy = os.path.join(self.directory, "cards_tmp.npy")
        np.save(temp_npy, vectors)
        os.replace(temp_npy, self.npy_path)

        atomic_write_json(self.manifest_path, manifest)

    def save_all(self, cards: list[Card], vectors: np.ndarray, manifest: dict) -> None:
        """Atomically overwrite cards, vectors, and manifest."""
        with _STORE_LOCK:
            self._write_all_locked(cards, vectors, manifest)

    def prune(self, keep_documents: set[str]) -> tuple[int, int]:
        """Remove cards and manifest entries whose document is not in keep_documents.

        Returns (kept_cards, removed_cards).
        """
        with _STORE_LOCK:
            index = self.load(check_model=False)
            kept_cards: list[Card] = []
            kept_indices: list[int] = []
            for idx, c in enumerate(index.cards):
                if c.document in keep_documents:
                    kept_cards.append(c)
                    if idx < len(index.vectors):
                        kept_indices.append(idx)

            removed = len(index.cards) - len(kept_cards)
            if kept_indices and len(index.vectors) > 0:
                kept_vectors = index.vectors[kept_indices]
            elif len(index.vectors.shape) == 2:
                kept_vectors = np.empty((0, index.vectors.shape[1]), dtype=float)
            else:
                kept_vectors = np.empty((0, 0), dtype=float)

            manifest = {}
            if os.path.exists(self.manifest_path):
                try:
                    with open(self.manifest_path, "r", encoding="utf-8") as fh:
                        raw_manifest = json.load(fh)
                    if isinstance(raw_manifest, dict):
                        manifest = {
                            k: v for k, v in raw_manifest.items()
                            if k in keep_documents
                        }
                except Exception:
                    manifest = {}

            self._write_all_locked(kept_cards, kept_vectors, manifest)
            return len(kept_cards), removed

    def append(self, document: str, cards: list[Card], vectors: np.ndarray) -> None:
        """Append cards and vectors for a document, updating manifest atomically."""
        with _STORE_LOCK:
            existing_index = self.load(check_model=False)
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

            self._write_all_locked(all_cards, all_vectors, manifest)

    def load(self, check_model: bool = True) -> CardIndex:
        if check_model:
            self.check_embedder()
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
            vecs = embed_documents([card_search_text(c) for c in new_cards])
            vecs_arr = np.asarray(vecs, dtype=float)
            norms = np.linalg.norm(vecs_arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs_norm = vecs_arr / norms
            store.append(doc, new_cards, vecs_norm)
            if vecs_norm.ndim == 2 and vecs_norm.shape[1] > 0:
                meta = store.load_meta()
                if not meta:
                    store.save_meta({
                        "embedder": config.EMBED_MODEL,
                        "embedding_dim": int(vecs_norm.shape[1]),
                    })
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
