"""
Shared hybrid search implementation.

Used by Agent 4 (interactive), Agent 5 (batch) and Agent 7 (chat).

The embeddings model is accepted as a parameter so callers can reuse a
singleton instance across many calls (needed for the planned Agent 5's
per-sentence loop).
"""

import os
import re
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


def resolve_cross_encoder_target(
    use_flag: bool | None = None,
    model_name: str | None = None,
    local_dir: str | None = None,
    explicit_env: str | None = None,
) -> str | None:
    """Return the local directory or model identifier to load for CrossEncoder, or None if disabled.

    Auto-activates an offline local model under ``data/models/cross-encoder`` when
    present unless ``CITATION_USE_CROSS_ENCODER=0`` was explicitly set.
    """
    from research_assistant import config as _cfg

    if use_flag is None:
        use_flag = getattr(_cfg, "USE_CROSS_ENCODER", False)
    if model_name is None:
        model_name = getattr(_cfg, "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if local_dir is None:
        local_dir = getattr(_cfg, "CROSS_ENCODER_LOCAL_DIR", "")
    if explicit_env is None:
        explicit_env = os.environ.get("CITATION_USE_CROSS_ENCODER")

    if explicit_env is not None and explicit_env.strip().lower() in ("0", "false", "no", "off"):
        return None
    if local_dir and os.path.isfile(os.path.join(local_dir, "config.json")):
        return local_dir
    if model_name and os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "config.json")):
        return model_name
    if use_flag:
        return model_name
    return None


def get_cross_encoder():
    """Lazily load cross-encoder model for candidate re-ranking if enabled or cached locally."""
    global _CROSS_ENCODER, _CROSS_ENCODER_TRIED
    if _CROSS_ENCODER_TRIED:
        return _CROSS_ENCODER
    _CROSS_ENCODER_TRIED = True

    target = resolve_cross_encoder_target()
    if not target:
        _CROSS_ENCODER = None
        return None

    try:
        from sentence_transformers import CrossEncoder
        try:
            _CROSS_ENCODER = CrossEncoder(target, local_files_only=True)
        except Exception:
            if os.path.isdir(target):
                raise
            _CROSS_ENCODER = CrossEncoder(target)
    except Exception as exc:
        logger.debug("CrossEncoder not available (%s); using deterministic lexical reranker.", exc)
        _CROSS_ENCODER = None
    return _CROSS_ENCODER


_RERANK_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "by", "of", "from",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "as",
    "it", "its", "et", "al", "which", "we", "our", "their", "paper", "study", "studies",
    "propose", "proposed", "show", "shows", "shown", "demonstrate", "demonstrates", "demonstrated",
    "report", "reports", "reported", "using", "used", "into", "also", "can", "could", "may",
    "might", "will", "would", "shall", "should", "not", "no", "such", "than", "then", "have",
    "has", "had", "more", "most", "very", "both", "between", "within", "across", "under", "over",
})
_SALIENT_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9@_\-\.]{1,}")
_TECH_TOKEN_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?(?:\s*(?:wt%|at\.?%|mol%|eV|meV|keV|GeV|K|C|Hz|GHz|THz|nm|um|mm|cm|mA|uA|V|mV|F/g|Wh/kg|W/kg|cm2/Vs|%|s|ms|ns|ps|fs))?\b"
    r"|\b(?:[A-Z]{2,}[0-9A-Za-z\-]*|[A-Za-z]+[0-9]+[A-Za-z0-9@\-]*|[A-Za-z]+-[A-Za-z0-9\-]+)\b)"
)
_OUTBOUND_CITE_RE = re.compile(r"\[\d+[^\]]*\]|\bet\s+al\.")
_PRIMARY_BODY_SECTIONS = frozenset({"results", "methods", "discussion", "experimental", "methodology", "analysis"})
_INTRO_SECTIONS = frozenset({"abstract", "introduction", "background", "related_work", "overview"})


def extract_salient_claim_terms(claim: str, max_terms: int = 14) -> list[str]:
    """Extract informative domain words, numbers, chemical formulas, and acronyms from *claim*
    without polluting intra-paper search with the cited paper's own title."""
    if not claim:
        return []
    tech = [m.group(0).strip(".,;:()") for m in _TECH_TOKEN_RE.finditer(claim)]
    words = [m.group(0).strip(".,;:()") for m in _SALIENT_WORD_RE.finditer(claim)]
    seen: set[str] = set()
    out: list[str] = []
    for tok in tech + words:
        tl = tok.lower()
        if len(tl) < 2 or tl in _RERANK_STOPWORDS or tl in seen:
            continue
        if len(tl) == 2 and not any(c.isdigit() or c.isupper() for c in tok):
            continue
        seen.add(tl)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def supplement_intra_doc_lexical_hits(
    queries: list[str],
    doc_filter: set | None,
    collection,
    texts,
    metadatas,
    existing_hits: list[dict],
    exclude_types: set | None = None,
    max_extra: int = 4,
) -> list[dict]:
    """Deterministically rescue high-overlap chunks (exact numbers, formulas, rare claim words)
    inside *doc_filter* that dense/contextualized search missed, with zero embedding API calls."""
    if not doc_filter or not queries:
        return existing_hits
    combined_q = " ".join(q for q in queries if q)
    salient = [t.lower() for t in extract_salient_claim_terms(combined_q, max_terms=18)]
    tech = {m.group(0).strip(".,;:()").lower() for m in _TECH_TOKEN_RE.finditer(combined_q) if len(m.group(0).strip()) >= 2}
    if not salient:
        return existing_hits

    doc_chunks: list[tuple[int, str, dict]] = []
    if getattr(collection, "is_lancedb", False) and hasattr(collection, "get"):
        for doc_name in sorted(doc_filter):
            try:
                res = collection.get(where={"document": doc_name}, limit=160, include=["documents", "metadatas"])
                docs_l = (res or {}).get("documents") or []
                metas_l = (res or {}).get("metadatas") or []
                for i, d_txt in enumerate(docs_l):
                    m = metas_l[i] if i < len(metas_l) and isinstance(metas_l[i], dict) else {"document": doc_name}
                    if exclude_types and m.get("type") in exclude_types:
                        continue
                    cidx = m.get("seq") if isinstance(m.get("seq"), int) else -1
                    if d_txt and str(d_txt).strip():
                        doc_chunks.append((cidx, str(d_txt), m))
            except Exception:
                pass
    elif isinstance(texts, list) and isinstance(metadatas, list) and texts and metadatas:
        limit = min(len(texts), len(metadatas))
        for idx in range(limit):
            m = metadatas[idx]
            if not isinstance(m, dict) or m.get("document") not in doc_filter:
                continue
            if exclude_types and m.get("type") in exclude_types:
                continue
            d_txt = texts[idx]
            if d_txt and str(d_txt).strip():
                doc_chunks.append((idx, str(d_txt), m))

    if not doc_chunks:
        return existing_hits

    seen_cidx = {h.get("chunk_index") for h in existing_hits if isinstance(h.get("chunk_index"), int) and h.get("chunk_index") >= 0}
    seen_texts = {h.get("text") for h in existing_hits if h.get("text")}

    scored_extras: list[tuple[float, dict]] = []
    min_rrf = min((float(h.get("rrf_score", 0.01)) for h in existing_hits), default=0.01)
    for cidx, d_txt, m in doc_chunks:
        if (cidx >= 0 and cidx in seen_cidx) or d_txt in seen_texts:
            continue
        low_txt = d_txt.lower()
        sal_hits = sum(1 for t in salient if t in low_txt)
        tech_hits = sum(1 for t in tech if t in low_txt)
        if tech_hits >= 1 and sal_hits >= 2 or sal_hits >= max(3, len(salient) // 2):
            lex_score = (sal_hits / len(salient)) + (0.5 * tech_hits / max(1, len(tech)))
            scored_extras.append((
                lex_score,
                {
                    "chunk_index": int(cidx),
                    "text": d_txt,
                    "metadata": m,
                    "rrf_score": round(min_rrf * 0.95, 5),
                },
            ))

    if not scored_extras:
        return existing_hits

    scored_extras.sort(key=lambda x: x[0], reverse=True)
    merged = list(existing_hits)
    for _, extra_hit in scored_extras[:max_extra]:
        merged.append(extra_hit)
    return merged


def rerank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int = DEFAULT_TOP_K,
    claim_text: str | None = None,
    neighbour_window: int = 1,
    primary_k: int | None = None,
) -> list[dict]:
    """
    Re-ranks candidates using a neural Cross-Encoder when enabled, or a fast
    deterministic lexical + technical/numeric + section-diversity reranker
    when Cross-Encoder is disabled.
    """
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_k]

    model = get_cross_encoder()
    if model is not None:
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
            logger.debug("Cross-encoder re-ranking failed (%s); falling back to lexical reranker.", exc)

    if claim_text is None:
        return candidates[:top_k]

    combined_q = f"{claim_text or ''} {query or ''}".strip()
    salient = [t.lower() for t in extract_salient_claim_terms(combined_q, max_terms=16)]
    tech = [
        m.group(0).strip(".,;:()").lower()
        for m in _TECH_TOKEN_RE.finditer(combined_q)
        if len(m.group(0).strip()) >= 2
    ]

    has_explicit_scores = any(("rrf_score" in c or "score" in c) for c in candidates)
    raw_scores = [
        float(c.get("rrf_score", c.get("score", 1.0 / (pos + 1 + 60.0))))
        for pos, c in enumerate(candidates)
    ]
    min_s, max_s = min(raw_scores), max(raw_scores)

    scored: list[tuple[float, float, int, dict]] = []
    for pos, (cand, r_s) in enumerate(zip(candidates, raw_scores)):
        if has_explicit_scores:
            if min_s >= 0 and max_s > 0:
                base = r_s / max_s
            elif max_s > min_s:
                base = (r_s - min_s) / (max_s - min_s)
            else:
                base = 0.5
        else:
            base = 1.0 / (pos + 1 + 10.0)

        text = str(cand.get("text") or "")
        low_txt = text.lower()
        sal_frac = (sum(1 for t in salient if t in low_txt) / len(salient)) if salient else 0.0
        tech_frac = (sum(1 for t in tech if t in low_txt) / len(tech)) if tech else 0.0
        lex_signal = 0.30 * sal_frac + 0.22 * tech_frac

        meta = cand.get("metadata") or {}
        sec = str(meta.get("section") or "").strip().lower()
        sec_adj = 0.0
        if lex_signal > 0:
            if sec in _PRIMARY_BODY_SECTIONS:
                sec_adj += 0.06
            outbound_cites = len(_OUTBOUND_CITE_RE.findall(text))
            if outbound_cites >= 4 and sec in (_INTRO_SECTIONS | {""}):
                sec_adj -= 0.06

        total_score = (0.55 * base) + lex_signal + sec_adj - (pos * 1e-6)
        scored.append((total_score, lex_signal, pos, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    focus_k = primary_k if (primary_k is not None and 0 < primary_k <= top_k) else top_k

    # When pool exceeds focus_k and focus_k >= 3, avoid wasting primary slots on adjacent chunks
    # (which expand_neighbours merges anyway) and ensure body-section representation when supported.
    if len(scored) > focus_k and focus_k >= 3 and neighbour_window > 0:
        selected: list[tuple[float, float, int, dict]] = []
        deferred_adj: list[tuple[float, float, int, dict]] = []
        for entry in scored:
            if len(selected) >= focus_k:
                deferred_adj.append(entry)
                continue
            cand = entry[3]
            cidx = cand.get("chunk_index")
            doc = (cand.get("metadata") or {}).get("document")
            is_adj = False
            if isinstance(cidx, int) and cidx >= 0:
                for sel in selected:
                    s_cand = sel[3]
                    s_cidx = s_cand.get("chunk_index")
                    s_doc = (s_cand.get("metadata") or {}).get("document")
                    if isinstance(s_cidx, int) and s_cidx >= 0 and s_doc == doc and abs(cidx - s_cidx) <= neighbour_window:
                        is_adj = True
                        break
            if is_adj:
                deferred_adj.append(entry)
            else:
                selected.append(entry)

        while len(selected) < focus_k and deferred_adj:
            selected.append(deferred_adj.pop(0))

        # Section diversity: if every selected chunk is abstract/intro, swap last slot for best body chunk
        sel_secs = {str((s[3].get("metadata") or {}).get("section") or "").lower() for s in selected}
        if sel_secs and sel_secs.issubset(_INTRO_SECTIONS):
            for idx_d, d_entry in enumerate(deferred_adj):
                d_sec = str((d_entry[3].get("metadata") or {}).get("section") or "").lower()
                if d_sec in _PRIMARY_BODY_SECTIONS and d_entry[1] > 0 and (selected[-1][0] - d_entry[0]) <= 0.20:
                    displaced = selected[-1]
                    selected[-1] = d_entry
                    deferred_adj[idx_d] = displaced
                    break

        ordered_all = [x[3] for x in (selected + deferred_adj)]
        return ordered_all[:top_k]

    return [x[3] for x in scored[:top_k]]
