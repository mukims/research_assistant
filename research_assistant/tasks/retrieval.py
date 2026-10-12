"""
Grounded Multi-Facet Vector DB Retrieval for Scientific Task Planning.

Retrieves evidence across 4 distinct methodological facets from local ChromaDB
(physics_papers collection with 75k chunks) and supports transparent auto-ingestion,
corpus-wide fallback for external benchmarks/protocols, and un-truncated context formatting.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from research_assistant.shared.db import load_search_resources
from research_assistant.shared.ingestion import (
    get_ingested_documents,
    ingest_pdfs,
    pdf_key,
)
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import hybrid_search

logger = get_logger("retrieval")

# 4 Distinct Methodological Facets
FACET_QUERIES = {
    "data_pipeline": (
        "data acquisition provenance dataset download train validation test split "
        "normalization preprocessing schema cleaning tokenization sample dimensions"
    ),
    "model_architecture": (
        "core model architecture neural network layer dimensions mathematical operators "
        "loss function objective forward pass equation formulation tensor shapes"
    ),
    "hyperparameters": (
        "hyperparameters learning rate schedule optimizer configuration batch size "
        "random seeds weight decay epochs iterations gradient clipping"
    ),
    "evaluation": (
        "evaluation metrics benchmark test procedures comparative baseline target values "
        "accuracy F1 score loss error rate table performance replication bounds"
    ),
}

FACET_LABELS = {
    "data_pipeline": "Data Acquisition, Provenance & Preprocessing Schemas",
    "model_architecture": "Core Model Architecture, Mathematical Operators & Loss Formulations",
    "hyperparameters": "Hyperparameters, Optimizer Configurations & Training Schedules",
    "evaluation": "Evaluation Metrics, Benchmark Procedures & Target Replication Bounds",
}


def check_or_ingest_paper(
    tei_or_path: Union[str, os.PathLike],
    title: Optional[str] = None,
) -> str:
    """
    Verifies that the target paper exists in ChromaDB (COLLECTION_NAME, e.g. 'physics_papers').
    If absent and tei_or_path points to an existing file, triggers transparent auto-ingestion
    with CPU-safe limits (workers=1, num_ctx: 4096).

    Args:
        tei_or_path: Path to the paper file (PDF, TEI XML) or filename / doc string.
        title: Optional paper title to associate during ingestion.

    Returns:
        Normalized document key (doc_key).
    """
    if not tei_or_path:
        return ""

    path_str = os.fspath(tei_or_path)
    doc_key = pdf_key(path_str)

    try:
        ingested_docs = get_ingested_documents()
    except Exception as e:
        logger.warning("Could not read ingested documents manifest: %s", e)
        ingested_docs = set()

    if doc_key in ingested_docs:
        logger.info("Paper %s is already grounded in ChromaDB (doc_key=%s)", path_str, doc_key)
        return doc_key

    # Check if paper file exists on disk to trigger auto-ingestion
    if os.path.isfile(path_str):
        logger.info("Target paper %s not found in ChromaDB; triggering auto-ingestion...", path_str)
        try:
            ingest_res = ingest_pdfs(
                {path_str: title or ""},
                workers=1,
                skip_ingested=True,
                rebuild_index=True,
                log_prefix="[TaskAutoIngest] ",
            )
            logger.info("Auto-ingestion completed for %s: %s", path_str, ingest_res)
        except Exception as e:
            logger.error("Auto-ingestion failed for %s: %s", path_str, e)
    else:
        logger.warning(
            "Target paper '%s' is not an accessible file on disk and not found in ChromaDB.",
            path_str,
        )

    return doc_key


def ensure_target_paper_grounded(
    tei_or_path: Union[str, os.PathLike],
    title: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Verifies if target paper is already grounded in ChromaDB / manifest.
    If present, returns (doc_key, False).
    If absent, calls check_or_ingest_paper(tei_or_path, title=title) and returns (doc_key, True).

    Args:
        tei_or_path: Path to the paper file (PDF, TEI XML) or filename / doc string.
        title: Optional paper title to associate during ingestion.

    Returns:
        Tuple of (doc_key, was_ingested).
    """
    if not tei_or_path:
        return ("", False)

    path_str = os.fspath(tei_or_path)
    doc_key = pdf_key(path_str)

    try:
        ingested_docs = get_ingested_documents()
    except Exception as e:
        logger.warning("Could not read ingested documents manifest: %s", e)
        ingested_docs = set()

    if doc_key in ingested_docs:
        logger.info("Paper %s is already grounded in ChromaDB (doc_key=%s)", path_str, doc_key)
        return (doc_key, False)

    # Absent: ingest paper via check_or_ingest_paper
    actual_key = check_or_ingest_paper(tei_or_path, title=title)
    return (actual_key or doc_key, True)


def format_faceted_context(facets: Dict[str, Any]) -> str:
    """
    Accepts dictionary containing facets (e.g. 'abstract', 'facets', 'tables', 'equations',
    or facet results dictionary). Formats clean markdown string with headers
    '### Methodological Facet: <facet_name>' and deduplicates overlapping chunks.

    Args:
        facets: Dictionary containing extracted facets, tables, equations, and/or abstract.

    Returns:
        Formatted un-truncated Markdown string.
    """
    if not facets or not isinstance(facets, dict):
        return ""

    sections: List[str] = ["# Methodological Facets"]
    seen_texts: set = set()
    seen_ids: set = set()

    # 1. Abstract
    if "abstract" in facets and facets["abstract"]:
        abstract_text = str(facets["abstract"]).strip()
        sections.append(f"## Abstract / Overview\n\n{abstract_text}")

    # 2. Extract facet mapping
    facet_data_map: Dict[str, Any] = {}
    if "facets" in facets and isinstance(facets["facets"], dict):
        facet_data_map.update(facets["facets"])

    for k, v in facets.items():
        if k not in ("abstract", "tables", "equations", "corpus_fallback", "facets", "doc_key", "formatted_context"):
            if isinstance(v, (dict, list, str)):
                facet_data_map[k] = v

    ordered_keys = [k for k in FACET_LABELS if k in facet_data_map]
    for k in facet_data_map:
        if k not in ordered_keys:
            ordered_keys.append(k)

    for facet_key in ordered_keys:
        val = facet_data_map[facet_key]
        facet_label = FACET_LABELS.get(facet_key, facet_key.replace("_", " ").title())
        facet_header = f"### Methodological Facet: {facet_label}"
        facet_body_parts: List[str] = []

        if isinstance(val, dict):
            chunks = val.get("chunks", [])
            if isinstance(chunks, list) and chunks:
                for idx, chunk in enumerate(chunks, 1):
                    if isinstance(chunk, dict):
                        cid = chunk.get("id") or chunk.get("chunk_id")
                        text = (chunk.get("text") or "").strip()
                        if cid and cid in seen_ids:
                            continue
                        if text and text in seen_texts:
                            continue
                        if cid:
                            seen_ids.add(cid)
                        if text:
                            seen_texts.add(text)

                        meta = chunk.get("metadata", {})
                        page = meta.get("page", "?")
                        chunk_type = meta.get("type", "text")
                        sec_title = meta.get("section", "")
                        rrf_score = chunk.get("rrf_score", 0.0)

                        header_parts = [f"Chunk {idx}", f"Page {page}", f"Type: {chunk_type}"]
                        if sec_title:
                            header_parts.append(f"Section: {sec_title}")
                        if rrf_score:
                            header_parts.append(f"RRF: {rrf_score:.4f}")
                        facet_body_parts.append(f"#### [{' | '.join(header_parts)}]\n{text}")
                    elif isinstance(chunk, str):
                        text = chunk.strip()
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            facet_body_parts.append(text)

            text_val = val.get("text", "")
            if isinstance(text_val, str) and text_val.strip():
                clean_text = text_val.strip()
                if clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    facet_body_parts.append(clean_text)

        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip() and item.strip() not in seen_texts:
                    seen_texts.add(item.strip())
                    facet_body_parts.append(item.strip())
        elif isinstance(val, str) and val.strip():
            clean_text = val.strip()
            if clean_text not in seen_texts:
                seen_texts.add(clean_text)
                facet_body_parts.append(clean_text)

        if facet_body_parts:
            sections.append(f"{facet_header}\n\n" + "\n\n".join(facet_body_parts))

    # 3. Tables
    if "tables" in facets and facets["tables"]:
        table_items = facets["tables"]
        if isinstance(table_items, list):
            sections.append("## Methodological Tables\n\n" + "\n\n".join(str(t).strip() for t in table_items))
        elif isinstance(table_items, str):
            sections.append(f"## Methodological Tables\n\n{table_items.strip()}")

    # 4. Equations
    if "equations" in facets and facets["equations"]:
        eq_items = facets["equations"]
        if isinstance(eq_items, list):
            sections.append("## Mathematical Formulations & Loss Equations\n\n" + "\n\n".join(str(e).strip() for e in eq_items))
        elif isinstance(eq_items, str):
            sections.append(f"## Mathematical Formulations & Loss Equations\n\n{eq_items.strip()}")

    # 5. External Corpus Fallback
    corpus_fallback = facets.get("corpus_fallback")
    if corpus_fallback and isinstance(corpus_fallback, list):
        fallback_parts = []
        for idx, chunk in enumerate(corpus_fallback, 1):
            if isinstance(chunk, dict):
                cid = chunk.get("id") or chunk.get("chunk_id")
                text = (chunk.get("text") or "").strip()
                if cid and cid in seen_ids:
                    continue
                if text and text in seen_texts:
                    continue
                if cid:
                    seen_ids.add(cid)
                if text:
                    seen_texts.add(text)
                meta = chunk.get("metadata", {})
                doc = meta.get("document", "external")
                page = meta.get("page", "?")
                fallback_parts.append(f"#### [External Chunk {idx} | Document: {doc} | Page: {page}]\n{text}")
            elif isinstance(chunk, str) and chunk.strip() and chunk.strip() not in seen_texts:
                seen_texts.add(chunk.strip())
                fallback_parts.append(chunk.strip())
        if fallback_parts:
            sections.append("## External Corpus Protocol & Benchmark Fallback\n\n" + "\n\n".join(fallback_parts))

    return "\n\n".join(sections)


def format_untruncated_context(
    facet_results: Dict[str, Any],
    corpus_fallback: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Formats an un-truncated markdown string preserving all extracted passages,
    mathematical formulas, architecture tables, and hyperparameter grids without
    the 10k character truncation barrier.
    """
    sections = []
    for facet_name, label in FACET_LABELS.items():
        facet_data = facet_results.get(facet_name, {})
        chunks = facet_data.get("chunks", [])
        if chunks:
            sections.append(f"## Facet: {label}")
            for idx, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                page = meta.get("page", "?")
                chunk_type = meta.get("type", "text")
                sec_title = meta.get("section", "")
                rrf_score = chunk.get("rrf_score", 0.0)
                chunk_idx = chunk.get("chunk_index", idx)

                header_parts = [f"Chunk {chunk_idx}", f"Page {page}", f"Type: {chunk_type}"]
                if sec_title:
                    header_parts.append(f"Section: {sec_title}")
                header_parts.append(f"RRF Score: {rrf_score:.4f}")
                header_str = " | ".join(header_parts)

                sections.append(f"### [{header_str}]\n{chunk.get('text', '').strip()}")
        elif facet_data.get("text"):
            sections.append(f"## Facet: {label}\n{facet_data['text'].strip()}")

    if corpus_fallback:
        sections.append("## External Corpus Protocol & Benchmark Fallback")
        for idx, chunk in enumerate(corpus_fallback, 1):
            meta = chunk.get("metadata", {})
            doc = meta.get("document", "external")
            page = meta.get("page", "?")
            citation = meta.get("citation_source", "")
            rrf_score = chunk.get("rrf_score", 0.0)
            chunk_idx = chunk.get("chunk_index", idx)

            header_parts = [f"External Chunk {chunk_idx}", f"Document: {doc}", f"Page {page}"]
            if citation:
                header_parts.append(f"Citation: {citation}")
            header_parts.append(f"RRF Score: {rrf_score:.4f}")
            header_str = " | ".join(header_parts)

            sections.append(f"### [{header_str}]\n{chunk.get('text', '').strip()}")

    return "\n\n".join(sections)


def retrieve_faceted_paper_context(
    doc_key: str,
    query_hint: Optional[str] = None,
    top_k_per_facet: int = 5,
    enable_corpus_fallback: bool = True,
    search_resources: Optional[Tuple[Any, Any, List[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Retrieves evidence across 4 distinct methodological facets:
    - 'data_pipeline': Dict with chunks, scores, text
    - 'model_architecture': Dict with chunks, scores, text
    - 'hyperparameters': Dict with chunks, scores, text
    - 'evaluation': Dict with chunks, scores, text
    - 'corpus_fallback': List of external protocol chunks
    - 'formatted_context': Clean un-truncated markdown string with formulas and tables

    Args:
        doc_key: Normalized document key to filter target paper chunks.
        query_hint: Optional query augmentation string (e.g. paper title or research topic).
        top_k_per_facet: Number of chunks to retrieve per facet (default: 5).
        enable_corpus_fallback: Whether to search across the full 75k corpus chunks
                                for external baselines and benchmark protocols.
        search_resources: Optional pre-loaded tuple of (collection, bm25, texts, metadatas).
                          If omitted, loaded via load_search_resources().

    Returns:
        Structured context dictionary.
    """
    empty_result = {
        "doc_key": doc_key,
        "data_pipeline": {"chunks": [], "scores": [], "text": ""},
        "model_architecture": {"chunks": [], "scores": [], "text": ""},
        "hyperparameters": {"chunks": [], "scores": [], "text": ""},
        "evaluation": {"chunks": [], "scores": [], "text": ""},
        "corpus_fallback": [],
        "formatted_context": "",
    }

    if search_resources is None:
        try:
            search_resources = load_search_resources()
        except Exception as e:
            logger.warning("Failed to load search resources for faceted retrieval: %s", e)
            return empty_result

    try:
        collection, bm25, texts, metadatas = search_resources
    except Exception as e:
        logger.warning("Invalid search resources structure: %s", e)
        return empty_result

    doc_filter = {doc_key} if doc_key else None

    facet_results = {}
    for facet_name, base_query in FACET_QUERIES.items():
        facet_query = f"{base_query} {query_hint}" if query_hint else base_query
        try:
            hits = hybrid_search(
                query=facet_query,
                collection=collection,
                bm25=bm25,
                texts=texts,
                metadatas=metadatas,
                top_k=top_k_per_facet,
                doc_filter=doc_filter,
            )
        except Exception as e:
            logger.warning("Faceted hybrid search failed for facet '%s': %s", facet_name, e)
            hits = []

        facet_results[facet_name] = {
            "chunks": hits,
            "scores": [hit.get("rrf_score", 0.0) for hit in hits],
            "text": "\n\n".join(hit["text"] for hit in hits if hit.get("text")),
        }

    corpus_hits = []
    if enable_corpus_fallback:
        fallback_query = (
            f"{query_hint} baseline benchmark protocol dataset implementation replication evaluation"
            if query_hint
            else "baseline benchmark protocol dataset implementation replication evaluation"
        )
        try:
            corpus_hits = hybrid_search(
                query=fallback_query,
                collection=collection,
                bm25=bm25,
                texts=texts,
                metadatas=metadatas,
                top_k=top_k_per_facet,
                doc_filter=None,
                exclude_docs={doc_key} if doc_key else None,
            )
        except Exception as e:
            logger.warning("Corpus fallback hybrid search failed: %s", e)
            corpus_hits = []

    formatted_context = format_untruncated_context(facet_results, corpus_hits)

    return {
        "doc_key": doc_key,
        "data_pipeline": facet_results["data_pipeline"],
        "model_architecture": facet_results["model_architecture"],
        "hyperparameters": facet_results["hyperparameters"],
        "evaluation": facet_results["evaluation"],
        "corpus_fallback": corpus_hits,
        "formatted_context": formatted_context,
    }
