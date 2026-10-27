"""
Shared hybrid search implementation.

Used by Agent 4 (interactive), Agent 5 (batch) and Agent 7 (chat).

The embeddings model is accepted as a parameter so callers can reuse a
singleton instance across many calls (needed for the planned Agent 5's
per-sentence loop).
"""

import numpy as np

from research_assistant.config import RRF_K, DEFAULT_TOP_K, EXCLUDED_FROM_EVIDENCE
from research_assistant.shared.log import get_logger
from research_assistant.shared.tokenize import tokenize, legacy_tokenize

logger = get_logger("search")


def _get_embeddings():
    # shared.llm keeps the process-wide singleton and picks the backend
    # (ollama / openai / huggingface) from the environment.
    from research_assistant.shared.llm import get_embeddings

    return get_embeddings()


def _where(doc_filter, exclude_types, exclude_docs=None):
    """ChromaDB where-clause for the dense side. Two or more conditions need
    $and; one stands alone; none is None (Chroma rejects an empty dict).
    doc_filter sorts for a stable clause; the exclusions sort likewise."""
    clauses = []
    if doc_filter:
        clauses.append({"document": {"$in": sorted(doc_filter)}})
    if exclude_docs:
        clauses.append({"document": {"$nin": sorted(exclude_docs)}})
    if exclude_types:
        clauses.append({"type": {"$nin": sorted(exclude_types)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def hybrid_search(
    query: str,
    collection,
    bm25,
    texts: list[str],
    metadatas: list[dict],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
    embeddings_model=None,
    doc_filter: set | None = None,
    exclude_types: set | None = None,
    exclude_docs: set | None = None,
) -> list[dict]:
    """
    Perform hybrid (BM25 sparse + dense vector) search with Reciprocal Rank Fusion.

    Args:
        query:            The search query string.
        collection:       ChromaDB collection handle.
        bm25:             BM25Okapi instance.
        texts:            Ordered list of all document texts (aligned with BM25 index).
        metadatas:        Ordered list of metadata dicts (aligned with texts).
        top_k:            Number of final results to return.
        rrf_k:            RRF fusion constant (higher = less rank bias).
        embeddings_model: Optional pre-initialised embeddings instance.
        doc_filter:       If given, restrict results to chunks whose
                          ``metadata["document"]`` is in this set (stage-2 of
                          two-stage retrieval).
        exclude_types:    Chunk ``metadata["type"]`` values to leave out of
                          both retrievers (e.g. ``EXCLUDED_FROM_EVIDENCE``
                          — Agent 8 never takes a model's reading of a plot
                          as evidence).
        exclude_docs:     Chunk ``metadata["document"]`` values to leave out of
                          both retrievers — the citer's evaluation keeps the
                          seed paper out of its own search.

    Returns:
        list[dict]: Each entry has keys: chunk_index, text, metadata, rrf_score.
    """
    if embeddings_model is None:
        embeddings_model = _get_embeddings()

    # Fast path: Native LanceDB hybrid search (zero-RAM paging, native Tantivy FTS + vector RRF)
    if getattr(collection, "is_lancedb", False):
        query_emb = embeddings_model.embed_query(query)
        return collection.search_hybrid(
            query=query,
            query_vector=query_emb,
            top_k=top_k,
            rrf_k=rrf_k,
            doc_filter=doc_filter,
            exclude_types=exclude_types,
            exclude_docs=exclude_docs,
        )

    # k_cand is the dense side's candidate count (Chroma filters before it ranks).
    k_cand = max(60, top_k * 10) if (doc_filter or exclude_docs) else max(15, top_k * 3)

    # 1. Sparse (BM25) retrieval — tokenised the way the pickle was built
    # (db.load_search_resources tags the object; an untagged object is v1).
    if getattr(bm25, "tokenizer_version", "v1") == "v1":
        tokenized_query = legacy_tokenize(query)
    else:
        tokenized_query = tokenize(query)

    def _meta(i):
        return metadatas[i] if i < len(metadatas) else {}

    def _top(scores, k):
        """Positions of the k highest scores, best first. Partitions instead of
        sorting everything: Agent 5 calls this once per sentence needing a
        citation, and a full argsort is O(n log n) over every chunk each time."""
        if k < len(scores):
            top_unordered = np.argpartition(scores, -k)[-k:]
            return top_unordered[np.argsort(scores[top_unordered])[::-1]]
        return np.argsort(scores)[::-1]

    if doc_filter or exclude_types or exclude_docs:
        # Rank within the allowed chunks. Cutting a global top-k first and
        # filtering after meant that on a large corpus none of the cited
        # paper's chunks were in the pool, and the sparse side silently
        # returned nothing for the one paper the auditor had asked about.
        pool = [
            i for i in range(len(texts))
            if (not doc_filter or _meta(i).get("document") in doc_filter)
            and (not exclude_types or _meta(i).get("type") not in exclude_types)
            and (not exclude_docs or _meta(i).get("document") not in exclude_docs)
        ]
        if hasattr(bm25, "get_batch_scores"):          # rank_bm25: scores only these, not every chunk
            pool_scores = np.asarray(bm25.get_batch_scores(tokenized_query, pool), dtype=float)
        else:
            pool_scores = np.asarray(bm25.get_scores(tokenized_query), dtype=float)[pool]
        sparse_top_indices = [pool[j] for j in _top(pool_scores, k_cand)]
    else:
        sparse_top_indices = _top(np.asarray(bm25.get_scores(tokenized_query), dtype=float), k_cand)

    # 2. Dense (embedding) retrieval
    query_emb = embeddings_model.embed_query(query)
    dense_results = collection.query(
        query_embeddings=[query_emb],
        n_results=k_cand,
        include=["documents", "metadatas", "distances"],
        where=_where(doc_filter, exclude_types, exclude_docs),
    )

    # The integer in a "chunk_N" id doubles as the position of that chunk in
    # `texts`, because both the BM25 corpus and `texts` are built by sorting the
    # collection on that same integer. That holds only while the ids are
    # contiguous from zero, which is true by construction (they are assigned
    # sequentially and never deleted) but is not enforced anywhere. Anything out
    # of range is dropped rather than silently indexing the wrong chunk.
    dense_ids_ordered = []
    if dense_results["ids"] and dense_results["ids"][0]:
        for id_str in dense_results["ids"][0]:
            try:
                idx = int(id_str.split("_")[1])
            except (ValueError, IndexError):
                continue
            if 0 <= idx < len(texts):
                dense_ids_ordered.append(idx)
            else:
                logger.warning(
                    "Dense hit %s is outside the %d loaded chunks — the id space "
                    "and the BM25 ordering have diverged; rebuild the index.",
                    id_str, len(texts),
                )

    # 3. Reciprocal Rank Fusion
    fused_scores = {}
    for rank, idx in enumerate(sparse_top_indices):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rank + 1 + rrf_k))
    for rank, idx in enumerate(dense_ids_ordered):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rank + 1 + rrf_k))

    # 4. Rank and return top_k
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, rrf_score in ranked:
        if idx < len(texts):
            results.append({
                # Cast: sparse indices arrive as numpy integers and dense ones
                # as Python ints, so the field's type depended on which
                # retriever surfaced the chunk.
                "chunk_index": int(idx),
                "text": texts[idx],
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "rrf_score": round(rrf_score, 5),
            })

    return results


def expand_neighbours(results, texts, metadatas, window: int = 1):
    """Attach the adjacent chunks of the same document to each result.

    A paper's chunks are minted contiguously (ARCHITECTURE.md §4.4), so
    chunk_index ± 1 with the same ``metadata["document"]`` is the neighbouring
    text. Adds ``context_before`` / ``context_after`` (joined with newlines,
    nearest last / nearest first respectively); empty strings at a document
    boundary. Callers that ignore the two keys see no change.

    Entries that are never evidence (a figure's image, an old VLM description)
    are skipped: they are stored beside their caption and would otherwise reach
    the judge as its context.
    """
    table_adapter = getattr(texts, "table_adapter", None)
    if table_adapter is not None:
        for r in results:
            doc = (r.get("metadata") or {}).get("document")
            stored_form = (
                (r.get("metadata") or {}).get("stored_form")
                or (r.get("metadata") or {}).get("type")
                or "text_chunk"
            )
            seq = (r.get("metadata") or {}).get("seq")
            if seq is None and isinstance(r.get("chunk_index"), int):
                seq = r.get("chunk_index")
            if doc and seq is not None and window > 0:
                b, a = table_adapter.get_neighbours(
                    doc, stored_form, seq, window=window, exclude_types=EXCLUDED_FROM_EVIDENCE
                )
                r["context_before"] = b
                r["context_after"] = a
            else:
                r["context_before"] = ""
                r["context_after"] = ""
        return results

    def _meta(i):
        return metadatas[i] if 0 <= i < len(metadatas) else None

    for r in results:
        idx = r.get("chunk_index")
        doc = (r.get("metadata") or {}).get("document")
        before, after = [], []
        if isinstance(idx, int) and 0 <= idx < len(texts) and window > 0:
            for i in range(1, window + 1):
                m = _meta(idx - i)
                if m is None or m.get("document") != doc:
                    break
                if m.get("type") in EXCLUDED_FROM_EVIDENCE:
                    continue
                before.insert(0, texts[idx - i])
            for i in range(1, window + 1):
                m = _meta(idx + i)
                if m is None or m.get("document") != doc:
                    break
                if m.get("type") in EXCLUDED_FROM_EVIDENCE:
                    continue
                after.append(texts[idx + i])
        r["context_before"] = "\n".join(before)
        r["context_after"] = "\n".join(after)
    return results


_CROSS_ENCODER = None
_CROSS_ENCODER_TRIED = False


def get_cross_encoder():
    """Lazily load cross-encoder model for candidate re-ranking if enabled."""
    global _CROSS_ENCODER, _CROSS_ENCODER_TRIED
    if _CROSS_ENCODER_TRIED:
        return _CROSS_ENCODER
    _CROSS_ENCODER_TRIED = True

    try:
        from research_assistant.config import USE_CROSS_ENCODER
        if not USE_CROSS_ENCODER:
            _CROSS_ENCODER = None
            return None
    except Exception:
        pass

    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as exc:
        logger.debug("CrossEncoder not available (%s); using original ranking.", exc)
        _CROSS_ENCODER = None
    return _CROSS_ENCODER


def rerank_candidates(query: str, candidates: list[dict], top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Re-ranks candidates using a neural Cross-Encoder when available,
    falling back cleanly to the existing RRF score.
    """
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_k]

    model = get_cross_encoder()
    if model is None:
        return candidates[:top_k]

    try:
        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = model.predict(pairs)
        ranked = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["cross_encoder_score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return ranked[:top_k]
    except Exception as exc:
        logger.debug("Cross-encoder re-ranking failed (%s); returning original candidates.", exc)
        return candidates[:top_k]
