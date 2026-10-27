"""
Grounded Multi-Facet Vector DB Retrieval for Scientific Task Planning.

Retrieves evidence across 4 distinct methodological facets from local ChromaDB
(physics_papers collection with 75k chunks) and supports transparent auto-ingestion,
corpus-wide fallback for external benchmarks/protocols, and un-truncated context formatting.
"""

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from research_assistant.config import DATA_DIR, PULLED_PDFS_DIR, RAW_DIR
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.ingestion import (
    get_ingested_documents,
    ingest_pdfs,
    pdf_key,
)
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import expand_neighbours, hybrid_search
from research_assistant.tasks.models import Task, TaskContextPack, TaskGraph

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


def budget_context_blocks(
    blocks: List[Dict[str, Any]],
    max_tokens: int = 6000,
    chars_per_token: float = 3.5,
) -> List[Dict[str, Any]]:
    """
    Dynamically budgets evidence blocks to fit within model context limits.
    Prioritizes highest-scoring / most relevant passages and truncates when budget is exceeded.
    """
    if not blocks:
        return []

    max_chars = int(max_tokens * chars_per_token)
    accumulated_chars = 0
    budgeted: List[Dict[str, Any]] = []

    # Sort blocks by score descending if scores exist, preserving order otherwise
    sorted_blocks = sorted(blocks, key=lambda b: float(b.get("score") or b.get("rrf_score") or 0.0), reverse=True) if any(b.get("score") is not None or b.get("rrf_score") is not None for b in blocks) else blocks

    for b in sorted_blocks:
        b_text = b.get("text") or b.get("content") or ""
        b_len = len(b_text)
        if accumulated_chars + b_len <= max_chars:
            budgeted.append(b)
            accumulated_chars += b_len
        else:
            remaining = max_chars - accumulated_chars
            if remaining > 200:
                truncated_b = dict(b)
                truncated_b["text"] = b_text[:remaining] + "... [truncated to fit context budget]"
                budgeted.append(truncated_b)
                accumulated_chars += remaining
            break

    return budgeted


def format_untruncated_context(
    facet_results: Dict[str, Any],
    corpus_fallback: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Formats an un-truncated markdown string preserving all extracted passages,
    mathematical formulas, architecture tables, and hyperparameter grids without
    the 10k character truncation barrier. Applies dynamic context budgeting when max_tokens is specified.
    """
    sections = []
    for facet_name, label in FACET_LABELS.items():
        facet_data = facet_results.get(facet_name, {})
        chunks = facet_data.get("chunks", [])
        if chunks:
            if max_tokens is not None:
                chunks = budget_context_blocks(chunks, max_tokens=max_tokens)
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
    fallback_doc_filter: Optional[set] = None,
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
        fallback_doc_filter: Optional set of cited document keys to scope the corpus
                             fallback search to the paper's reference subgraph.

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
        fb_filter = set(fallback_doc_filter) if fallback_doc_filter else None
        try:
            corpus_hits = hybrid_search(
                query=fallback_query,
                collection=collection,
                bm25=bm25,
                texts=texts,
                metadatas=metadatas,
                top_k=top_k_per_facet,
                doc_filter=fb_filter,
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


_METHOD_SECTION_RE = re.compile(
    r"\b(method\w*|model\w*|architect\w*|implement\w*|experiment\w*|setup\w*|data\w*|train\w*|"
    r"algorithm\w*|protocol\w*|formulat\w*|procedur\w*|baseline\w*|evaluat\w*|simulat\w*)\b",
    re.IGNORECASE,
)


def resolve_tei_source(tei_or_path: Union[str, bytes, os.PathLike]) -> Optional[Union[str, bytes]]:
    """
    Locates GROBID TEI XML content or file path for a given paper identifier, PDF path,
    or inline XML string.
    """
    if not tei_or_path:
        return None

    if isinstance(tei_or_path, bytes):
        if b"<TEI" in tei_or_path or b"<?xml" in tei_or_path:
            return tei_or_path
        return None

    raw_str = os.fspath(tei_or_path) if isinstance(tei_or_path, os.PathLike) else str(tei_or_path)
    stripped = raw_str.strip()
    if stripped.startswith("<?xml") or stripped.startswith("<TEI") or "<teiHeader" in stripped:
        return raw_str

    if os.path.isfile(raw_str) and (raw_str.endswith(".xml") or raw_str.endswith(".tei")):
        return raw_str

    doc_key = pdf_key(raw_str)
    stem = doc_key[:-4] if doc_key.lower().endswith(".pdf") else doc_key
    candidates = [
        os.path.join(RAW_DIR, "grobid_output", f"{stem}.grobid.tei.xml"),
        os.path.join(RAW_DIR, "grobid_output", f"{stem}.tei.xml"),
        os.path.join(DATA_DIR, "raw", "grobid_output", f"{stem}.grobid.tei.xml"),
    ]
    if os.path.isfile(raw_str):
        base_dir = os.path.dirname(os.path.abspath(raw_str))
        candidates.append(os.path.join(base_dir, f"{stem}.grobid.tei.xml"))
        candidates.append(os.path.join(base_dir, "grobid_output", f"{stem}.grobid.tei.xml"))

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    return None


def _ensure_references_built_for_pdf(pdf_path: str) -> None:
    """
    Optionally invokes Agent 1 (GROBID citation extractor), Agent 2 (citation fetcher),
    and Agent 3 (PDF ingestion) to build the reference database around a target PDF.
    """
    if not pdf_path or not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        return
    try:
        from research_assistant.agents.agent1_extractor import extract_citations_from_pdf
        from research_assistant.agents.agent2_fetcher import fetch_missing_citations

        logger.info("Building reference database around seed PDF: %s", pdf_path)
        extract_citations_from_pdf(pdf_path)
        newly_downloaded = fetch_missing_citations()
        if newly_downloaded:
            ingest_pdfs(
                newly_downloaded,
                workers=1,
                skip_ingested=True,
                rebuild_index=True,
                log_prefix="[TaskRefBuild] ",
            )
    except Exception as e:
        logger.warning("Reference database build step skipped or failed for %s: %s", pdf_path, e)


def build_paper_reference_subgraph(
    tei_or_path: Union[str, bytes, os.PathLike],
    paper_title: Optional[str] = None,
    auto_fetch_refs: bool = False,
    max_method_refs: int = 25,
) -> Dict[str, Any]:
    """
    Builds a paper-scoped reference subgraph by extracting in-text citations from the
    target paper's GROBID TEI XML, classifying citation roles ('method', 'definition',
    'software', 'pointer', 'evidential'), and resolving each cited reference to its
    downloaded/ingested document key (`doc_key`) in the local database.

    Returns:
        Dictionary with keys:
        - 'seed_doc_key': Normalized doc_key of the target paper
        - 'paper_title': Resolved paper title
        - 'cited_doc_keys': Sorted list of all resolved cited document keys
        - 'method_doc_keys': Sorted list of methodological cited document keys
        - 'all_subgraph_keys': [seed_doc_key] + cited_doc_keys
        - 'method_citations': List of resolved methodological citation dicts
        - 'unresolved_method_citations': List of methodological citations missing from corpus
        - 'citation_map': Mapping of citation xml_id / cite_text to reference metadata
    """
    raw_str = (
        os.fspath(tei_or_path)
        if isinstance(tei_or_path, os.PathLike)
        else (tei_or_path if isinstance(tei_or_path, str) else "seed_paper.pdf")
    )
    if isinstance(raw_str, str) and (raw_str.strip().startswith("<") or "\n" in raw_str):
        seed_doc_key = pdf_key(paper_title or "seed_paper.pdf")
    else:
        seed_doc_key = pdf_key(str(raw_str))

    should_fetch = auto_fetch_refs or (
        os.environ.get("TASK_ENABLE_REFERENCE_BUILD", "0").strip().lower() in ("1", "true", "yes")
    )
    if should_fetch and isinstance(raw_str, str) and os.path.isfile(raw_str):
        _ensure_references_built_for_pdf(raw_str)

    tei_source = resolve_tei_source(tei_or_path)
    empty_subgraph: Dict[str, Any] = {
        "seed_doc_key": seed_doc_key,
        "paper_title": paper_title or seed_doc_key,
        "cited_doc_keys": [],
        "method_doc_keys": [],
        "all_subgraph_keys": [seed_doc_key] if seed_doc_key else [],
        "method_citations": [],
        "unresolved_method_citations": [],
        "citation_map": {},
    }
    if not tei_source:
        return empty_subgraph

    temp_tei_path: Optional[str] = None
    try:
        from research_assistant.shared.seed_audit import (
            _load_downloaded_manifest,
            _match_downloaded_paper,
            extract_seed_citation_claims,
        )

        if isinstance(tei_source, bytes) or (
            isinstance(tei_source, str)
            and (tei_source.strip().startswith("<") or "\n" in tei_source or not os.path.exists(tei_source))
        ):
            import tempfile
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".tei.xml", delete=False) as tf:
                raw_bytes = tei_source if isinstance(tei_source, bytes) else tei_source.encode("utf-8", errors="replace")
                tf.write(raw_bytes)
                temp_tei_path = tf.name
            claims = extract_seed_citation_claims(temp_tei_path)
        else:
            claims = extract_seed_citation_claims(str(tei_source))
    except Exception as e:
        logger.debug("Could not extract citation claims for subgraph (%s): %s", seed_doc_key, e)
        return empty_subgraph
    finally:
        if temp_tei_path and os.path.exists(temp_tei_path):
            try:
                os.remove(temp_tei_path)
            except Exception:
                pass

    if not claims:
        return empty_subgraph

    try:
        manifest = _load_downloaded_manifest()
    except Exception:
        manifest = {}

    try:
        ingested_docs = get_ingested_documents()
    except Exception:
        ingested_docs = set()

    cited_doc_keys_set: set = set()
    method_doc_keys_set: set = set()
    method_citations: List[Dict[str, Any]] = []
    unresolved_method_citations: List[Dict[str, Any]] = []
    citation_map: Dict[str, Dict[str, Any]] = {}
    seen_unresolved_keys: set = set()

    for c in claims:
        ref = c.get("ref") or {}
        cite_text = str(c.get("cite_text") or "").strip()
        role = str(c.get("role") or "evidential").strip()
        section = str(c.get("section") or "").strip()
        heading = str(c.get("section_heading") or section).strip()
        sentence = str(c.get("sentence") or c.get("claim") or "").strip()
        ref_title = str(ref.get("title") or "").strip()
        ref_doi = str(ref.get("doi") or "").strip()
        ref_year = str(ref.get("year") or "").strip()
        ref_xml_id = str(ref.get("xml_id") or c.get("target") or cite_text).strip()

        matched_entry = _match_downloaded_paper(ref, seed_doc_key, manifest) if c.get("resolved") else None
        cited_doc_key = ""
        if matched_entry and matched_entry.get("path"):
            cited_doc_key = pdf_key(matched_entry["path"])
        elif ref_doi:
            cand_key = pdf_key(f"doi_{ref_doi.replace('/', '_')}.pdf")
            if cand_key in ingested_docs:
                cited_doc_key = cand_key

        is_ingested = bool(cited_doc_key and (not ingested_docs or cited_doc_key in ingested_docs))
        if cited_doc_key and cited_doc_key != seed_doc_key:
            cited_doc_keys_set.add(cited_doc_key)

        ref_record = {
            "xml_id": ref_xml_id,
            "cite_text": cite_text,
            "title": ref_title,
            "doi": ref_doi,
            "year": ref_year,
            "authors": ref.get("authors") or [],
            "venue": ref.get("venue") or "",
            "cited_doc_key": cited_doc_key,
            "ingested": is_ingested,
        }
        if ref_xml_id:
            citation_map.setdefault(ref_xml_id, ref_record)
        if cite_text:
            citation_map.setdefault(cite_text, ref_record)

        is_methodological = (
            role in ("method", "definition", "software", "pointer")
            or bool(_METHOD_SECTION_RE.search(heading))
            or bool(_METHOD_SECTION_RE.search(section))
        )
        if not is_methodological:
            continue

        entry = {
            "cite_text": cite_text,
            "xml_id": ref_xml_id,
            "role": role,
            "section": heading or section,
            "sentence": sentence,
            "claim": str(c.get("claim") or sentence).strip(),
            "context": str(c.get("context") or sentence).strip(),
            "ref_title": ref_title,
            "ref_doi": ref_doi,
            "ref_year": ref_year,
            "cited_doc_key": cited_doc_key,
            "ingested": is_ingested,
            "resolved_in_corpus": bool(cited_doc_key),
        }

        if cited_doc_key and cited_doc_key != seed_doc_key:
            method_doc_keys_set.add(cited_doc_key)
            if len(method_citations) < max_method_refs:
                method_citations.append(entry)
        elif ref_title or cite_text:
            dedup_u = ref_xml_id or ref_title or cite_text
            if dedup_u not in seen_unresolved_keys and len(unresolved_method_citations) < max_method_refs:
                seen_unresolved_keys.add(dedup_u)
                unresolved_method_citations.append(entry)

    cited_doc_keys = sorted(cited_doc_keys_set)
    method_doc_keys = sorted(method_doc_keys_set)
    all_keys = ([seed_doc_key] if seed_doc_key else []) + [k for k in cited_doc_keys if k != seed_doc_key]

    return {
        "seed_doc_key": seed_doc_key,
        "paper_title": paper_title or seed_doc_key,
        "cited_doc_keys": cited_doc_keys,
        "method_doc_keys": method_doc_keys,
        "all_subgraph_keys": all_keys,
        "method_citations": method_citations,
        "unresolved_method_citations": unresolved_method_citations,
        "citation_map": citation_map,
    }


def dereference_method_citations(
    subgraph: Dict[str, Any],
    search_resources: Optional[Tuple[Any, Any, List[str], List[Dict[str, Any]]]] = None,
    top_k_per_ref: int = 2,
    expand_window: int = 1,
) -> List[Dict[str, Any]]:
    """
    Dereferences in-text methodological citations against the cited papers in ChromaDB/BM25.
    For each resolved methodological citation, retrieves the specific passages from the
    cited document (`doc_filter={cited_doc_key}`) matching the citing sentence and reference title,
    expanding neighboring chunks so equations and hyperparameter context are preserved intact.
    """
    if not subgraph or not isinstance(subgraph, dict):
        return []

    method_citations = subgraph.get("method_citations") or []
    resolved = [c for c in method_citations if c.get("cited_doc_key")]
    if not resolved:
        return []

    if search_resources is None:
        try:
            search_resources = load_search_resources()
        except Exception as e:
            logger.warning("Could not load search resources for citation dereferencing: %s", e)
            return []

    try:
        collection, bm25, texts, metadatas = search_resources
    except Exception as e:
        logger.warning("Invalid search resources for citation dereferencing: %s", e)
        return []

    dereferenced: List[Dict[str, Any]] = []
    seen_doc_queries: set = set()

    for cite in resolved:
        cited_doc_key = str(cite.get("cited_doc_key") or "").strip()
        if not cited_doc_key:
            continue
        sentence = str(cite.get("claim") or cite.get("sentence") or "").strip()
        ref_title = str(cite.get("ref_title") or "").strip()
        query = f"{sentence} {ref_title}".strip() or ref_title or cited_doc_key

        dedup_key = (cited_doc_key, query[:120])
        if dedup_key in seen_doc_queries:
            continue
        seen_doc_queries.add(dedup_key)

        try:
            hits = hybrid_search(
                query=query,
                collection=collection,
                bm25=bm25,
                texts=texts,
                metadatas=metadatas,
                top_k=top_k_per_ref,
                doc_filter={cited_doc_key},
            )
            if hits and expand_window > 0 and texts and metadatas:
                expand_neighbours(hits, texts, metadatas, window=expand_window)
        except Exception as e:
            logger.warning("Citation dereference search failed for %s: %s", cited_doc_key, e)
            hits = []

        if not hits:
            continue

        passages = []
        for h in hits:
            meta = h.get("metadata") or {}
            before = (h.get("context_before") or "").strip()
            core = (h.get("text") or "").strip()
            after = (h.get("context_after") or "").strip()
            full_text = "\n".join(part for part in (before, core, after) if part)
            passages.append({
                "text": full_text or core,
                "core_text": core,
                "section": meta.get("section") or "text",
                "page": meta.get("page_first", meta.get("page", "?")),
                "chunk_index": h.get("chunk_index", -1),
                "rrf_score": h.get("rrf_score", 0.0),
                "document": meta.get("document") or cited_doc_key,
            })

        dereferenced.append({
            "cite_text": cite.get("cite_text") or "",
            "role": cite.get("role") or "method",
            "seed_section": cite.get("section") or "",
            "citing_sentence": cite.get("sentence") or sentence,
            "cited_doc_key": cited_doc_key,
            "cited_title": ref_title or cited_doc_key,
            "cited_doi": cite.get("ref_doi") or "",
            "passages": passages,
        })

    return dereferenced


def format_dereferenced_citations(dereferenced: List[Dict[str, Any]]) -> str:
    """Formats dereferenced methodological citations as Markdown for planner prompts."""
    if not dereferenced:
        return ""
    blocks = ["## Dereferenced Methodological Citations (Subgraph Protocols)"]
    for idx, item in enumerate(dereferenced, 1):
        cite_marker = item.get("cite_text") or f"Ref #{idx}"
        role = item.get("role") or "method"
        cited_title = item.get("cited_title") or item.get("cited_doc_key") or "Unknown Reference"
        cited_doc = item.get("cited_doc_key") or ""
        seed_sec = item.get("seed_section") or "Methods"
        citing_sent = item.get("citing_sentence") or ""

        header = f"### [{cite_marker} | Role: {role} | Section: {seed_sec}] -> {cited_title} ({cited_doc})"
        lines = [header]
        if citing_sent:
            lines.append(f"- **In-Text Usage in Target Paper:** \"{citing_sent}\"")
        for p_idx, p in enumerate(item.get("passages") or [], 1):
            sec = p.get("section") or "text"
            page = p.get("page", "?")
            txt = (p.get("text") or "").strip()
            lines.append(f"- **Retrieved Protocol Passage {p_idx} (p.{page}, {sec}):**\n  {txt}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def load_subgraph_technique_cards(doc_keys: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Loads verified TechniqueCards from `DATA_DIR/transfer*/cards.jsonl` that belong
    to any document in the paper's reference subgraph (`doc_keys`).
    """
    wanted = {str(k) for k in (doc_keys or []) if k}
    if not wanted or not os.path.isdir(DATA_DIR):
        return []

    patterns = [
        os.path.join(DATA_DIR, "transfer*", "cards.jsonl"),
        os.path.join(DATA_DIR, "cards.jsonl"),
    ]
    card_files: List[str] = []
    for pat in patterns:
        card_files.extend(sorted(glob.glob(pat)))

    cards: List[Dict[str, Any]] = []
    seen_card_ids: set = set()
    for path in card_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    doc = obj.get("document") or (obj.get("evidence") or {}).get("document")
                    cid = obj.get("card_id") or f"{doc}#{obj.get('name', '')}"
                    if doc in wanted and cid not in seen_card_ids:
                        seen_card_ids.add(cid)
                        cards.append({
                            "card_id": cid,
                            "document": doc,
                            "name": obj.get("name", ""),
                            "problem_solved": obj.get("problem_solved", ""),
                            "context": obj.get("context", ""),
                            "inputs": obj.get("inputs", ""),
                            "outputs": obj.get("outputs", ""),
                            "supporting_span": (obj.get("evidence") or {}).get("supporting_span", ""),
                        })
        except Exception as e:
            logger.debug("Skipping technique cards file %s: %s", path, e)

    return cards


def compile_task_context_pack(
    task: Task,
    graph: Optional[TaskGraph] = None,
    subgraph: Optional[Dict[str, Any]] = None,
    dereferenced_citations: Optional[List[Dict[str, Any]]] = None,
    search_resources: Optional[Tuple[Any, Any, List[str], List[Dict[str, Any]]]] = None,
    top_k_seed: int = 3,
    top_k_cited: int = 2,
) -> TaskContextPack:
    """
    Compiles a grounded `TaskContextPack` for a single `Task` by retrieving:
    1. Seed paper passages tailored to the task's title, description, inputs, and outputs
    2. Dereferenced protocol passages from cited papers in the paper's reference subgraph
    3. Unresolved methodological citations relevant to the task
    4. Verified TechniqueCards from the paper subgraph
    5. Target benchmarks and acceptance criteria
    """
    if subgraph is None and graph is not None:
        subgraph = (graph.metadata or {}).get("subgraph") or {}
    subgraph = subgraph or {}

    seed_doc_key = subgraph.get("seed_doc_key") or (graph.paper_id if graph else "")
    cited_doc_keys = list(subgraph.get("cited_doc_keys") or [])
    all_subgraph_keys = list(subgraph.get("all_subgraph_keys") or ([seed_doc_key] if seed_doc_key else []))

    query_parts = [
        task.title,
        task.category.value if hasattr(task.category, "value") else str(task.category),
        task.description,
        " ".join(task.inputs[:4]),
        " ".join(task.outputs[:4]),
    ]
    task_query = " ".join(p for p in query_parts if p).strip()
    query_terms = [w for w in re.findall(r"[A-Za-z0-9_\-]{4,}", f"{task.title} {task.description}")][:12]

    seed_passages: List[Dict[str, Any]] = []
    cited_protocols: List[Dict[str, Any]] = []

    if search_resources is not None:
        try:
            collection, bm25, texts, metadatas = search_resources
        except Exception:
            collection = bm25 = texts = metadatas = None
    else:
        collection = bm25 = texts = metadatas = None

    if collection is not None and bm25 is not None and texts is not None and metadatas is not None and task_query:
        if seed_doc_key:
            try:
                s_hits = hybrid_search(
                    query=task_query,
                    collection=collection,
                    bm25=bm25,
                    texts=texts,
                    metadatas=metadatas,
                    top_k=top_k_seed,
                    doc_filter={seed_doc_key},
                )
                if s_hits and texts and metadatas:
                    expand_neighbours(s_hits, texts, metadatas, window=1)
                for h in s_hits:
                    meta = h.get("metadata") or {}
                    seed_passages.append({
                        "document": meta.get("document") or seed_doc_key,
                        "section": meta.get("section") or "text",
                        "page": meta.get("page_first", meta.get("page", "?")),
                        "chunk_index": h.get("chunk_index", -1),
                        "rrf_score": h.get("rrf_score", 0.0),
                        "text": (h.get("text") or "").strip(),
                        "context_before": (h.get("context_before") or "").strip(),
                        "context_after": (h.get("context_after") or "").strip(),
                    })
            except Exception as e:
                logger.debug("Task seed passage retrieval failed for %s: %s", task.task_id, e)

        if cited_doc_keys:
            try:
                c_hits = hybrid_search(
                    query=task_query,
                    collection=collection,
                    bm25=bm25,
                    texts=texts,
                    metadatas=metadatas,
                    top_k=top_k_cited,
                    doc_filter=set(cited_doc_keys),
                )
                if c_hits and texts and metadatas:
                    expand_neighbours(c_hits, texts, metadatas, window=1)
                for h in c_hits:
                    meta = h.get("metadata") or {}
                    doc = meta.get("document") or ""
                    cited_protocols.append({
                        "cite_text": meta.get("citation_source") or doc,
                        "role": "method",
                        "cited_doc_key": doc,
                        "cited_title": meta.get("citation_source") or doc,
                        "section": meta.get("section") or "text",
                        "page": meta.get("page_first", meta.get("page", "?")),
                        "chunk_index": h.get("chunk_index", -1),
                        "rrf_score": h.get("rrf_score", 0.0),
                        "text": (h.get("text") or "").strip(),
                    })
            except Exception as e:
                logger.debug("Task cited passage retrieval failed for %s: %s", task.task_id, e)

    # Always fold in task.paper_context_excerpts if seed_passages is empty
    if not seed_passages and task.paper_context_excerpts:
        for exc in task.paper_context_excerpts:
            if exc and str(exc).strip():
                seed_passages.append({
                    "document": seed_doc_key or "target_paper",
                    "section": "excerpt",
                    "page": "?",
                    "text": str(exc).strip(),
                })

    # Match dereferenced citations relevant to this task
    if dereferenced_citations:
        task_lower = f"{task.title} {task.description} {' '.join(task.paper_context_excerpts)}".lower()
        existing_docs = {c.get("cited_doc_key") for c in cited_protocols if c.get("cited_doc_key")}
        for d in dereferenced_citations:
            c_doc = d.get("cited_doc_key")
            c_marker = str(d.get("cite_text") or "").lower()
            c_title_words = [w for w in re.findall(r"[a-z0-9]{4,}", str(d.get("cited_title") or "").lower())][:5]
            marker_match = bool(c_marker and c_marker in task_lower)
            word_match = any(w in task_lower for w in c_title_words)
            if marker_match or word_match or len(cited_protocols) < top_k_cited:
                for p in (d.get("passages") or [])[:1]:
                    if c_doc not in existing_docs or marker_match:
                        cited_protocols.append({
                            "cite_text": d.get("cite_text") or "",
                            "role": d.get("role") or "method",
                            "citing_sentence": d.get("citing_sentence") or "",
                            "cited_doc_key": c_doc or "",
                            "cited_title": d.get("cited_title") or "",
                            "section": p.get("section") or "text",
                            "page": p.get("page", "?"),
                            "text": p.get("text") or "",
                        })
                        existing_docs.add(c_doc)

    unresolved_all = subgraph.get("unresolved_method_citations") or []
    unresolved_for_task: List[Dict[str, Any]] = []
    for u in unresolved_all[:5]:
        unresolved_for_task.append({
            "cite_text": u.get("cite_text") or "",
            "role": u.get("role") or "method",
            "title": u.get("ref_title") or "",
            "doi": u.get("ref_doi") or "",
            "sentence": u.get("sentence") or "",
        })

    cards = load_subgraph_technique_cards(all_subgraph_keys)
    benchmarks = list(task.acceptance_criteria or [])

    return TaskContextPack(
        seed_passages=seed_passages,
        cited_protocols=cited_protocols,
        unresolved_citations=unresolved_for_task,
        technique_cards=cards[:5],
        target_benchmarks=benchmarks,
        query_terms=query_terms,
    )


def populate_graph_context_packs(
    graph: TaskGraph,
    subgraph: Optional[Dict[str, Any]] = None,
    dereferenced_citations: Optional[List[Dict[str, Any]]] = None,
    search_resources: Optional[Tuple[Any, Any, List[str], List[Dict[str, Any]]]] = None,
) -> TaskGraph:
    """
    Populates `task.context_pack` for every task in `graph` and records the paper's
    reference subgraph metadata on `graph.metadata["subgraph"]`.
    """
    if subgraph is not None:
        graph.metadata["subgraph"] = {
            "seed_doc_key": subgraph.get("seed_doc_key", graph.paper_id),
            "cited_doc_keys": list(subgraph.get("cited_doc_keys") or []),
            "method_doc_keys": list(subgraph.get("method_doc_keys") or []),
            "all_subgraph_keys": list(subgraph.get("all_subgraph_keys") or []),
            "method_citations_count": len(subgraph.get("method_citations") or []),
            "unresolved_method_citations": list(subgraph.get("unresolved_method_citations") or [])[:10],
        }
        graph.metadata["subgraph_doc_keys"] = list(subgraph.get("all_subgraph_keys") or [])

    for task in graph.tasks.values():
        task.context_pack = compile_task_context_pack(
            task=task,
            graph=graph,
            subgraph=subgraph,
            dereferenced_citations=dereferenced_citations,
            search_resources=search_resources,
        )
        # Ensure paper_context_excerpts also has grounded text for backward compatibility
        if not task.paper_context_excerpts and task.context_pack and task.context_pack.seed_passages:
            task.paper_context_excerpts = [
                p["text"] for p in task.context_pack.seed_passages[:2] if p.get("text")
            ]

    return graph


def query_task_database(
    query: str,
    graph_or_subgraph: Optional[Union[TaskGraph, Dict[str, Any]]] = None,
    task_id: Optional[str] = None,
    scope: str = "subgraph",
    top_k: int = 5,
    expand_window: int = 1,
    search_resources: Optional[Tuple[Any, Any, List[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Executes an on-demand hybrid search against the paper's reference database for an
    executing agent.

    Args:
        query: Natural-language or keyword/equation query from the agent.
        graph_or_subgraph: Optional TaskGraph or subgraph dict to scope the search.
        task_id: Optional task_id to augment query context and inspect task context_pack.
        scope: One of 'subgraph' (seed + cited papers, default), 'seed' (target paper only),
               'cited' (cited reference papers only), or 'global' (entire corpus).
        top_k: Number of passages to return (default: 5).
        expand_window: Number of adjacent chunks (+/-) to include around each hit.
        search_resources: Optional pre-loaded (collection, bm25, texts, metadatas).

    Returns:
        Dictionary containing query metadata and ranked, neighbor-expanded passages.
    """
    seed_doc_key = ""
    cited_doc_keys: List[str] = []
    subgraph_keys: List[str] = []
    task_obj: Optional[Task] = None

    if isinstance(graph_or_subgraph, TaskGraph):
        sub = (graph_or_subgraph.metadata or {}).get("subgraph") or {}
        seed_doc_key = sub.get("seed_doc_key") or pdf_key(graph_or_subgraph.paper_id)
        cited_doc_keys = list(sub.get("cited_doc_keys") or [])
        subgraph_keys = graph_or_subgraph.subgraph_doc_keys() or ([seed_doc_key] if seed_doc_key else [])
        if task_id:
            task_obj = graph_or_subgraph.get_task(task_id)
    elif isinstance(graph_or_subgraph, dict):
        if "tasks" in graph_or_subgraph and "paper_id" in graph_or_subgraph:
            tg = TaskGraph.from_dict(graph_or_subgraph)
            return query_task_database(
                query=query,
                graph_or_subgraph=tg,
                task_id=task_id,
                scope=scope,
                top_k=top_k,
                expand_window=expand_window,
                search_resources=search_resources,
            )
        seed_doc_key = str(graph_or_subgraph.get("seed_doc_key") or "")
        cited_doc_keys = list(graph_or_subgraph.get("cited_doc_keys") or [])
        subgraph_keys = list(graph_or_subgraph.get("all_subgraph_keys") or ([seed_doc_key] if seed_doc_key else []))

    effective_query = query.strip()
    if task_obj and not effective_query:
        effective_query = f"{task_obj.title} {task_obj.description}".strip()

    scope_clean = (scope or "subgraph").strip().lower()
    doc_filter: Optional[set] = None
    if scope_clean == "seed" and seed_doc_key:
        doc_filter = {seed_doc_key}
    elif scope_clean == "cited" and cited_doc_keys:
        doc_filter = set(cited_doc_keys)
    elif scope_clean == "subgraph" and subgraph_keys:
        doc_filter = set(subgraph_keys)
    elif scope_clean == "global":
        doc_filter = None

    if search_resources is None:
        try:
            search_resources = load_search_resources()
        except Exception as e:
            logger.debug("Could not load search resources for query_task_database: %s", e)
            search_resources = None

    hits: List[Dict[str, Any]] = []
    if search_resources is not None and effective_query:
        try:
            collection, bm25, texts, metadatas = search_resources
            hits = hybrid_search(
                query=effective_query,
                collection=collection,
                bm25=bm25,
                texts=texts,
                metadatas=metadatas,
                top_k=top_k,
                doc_filter=doc_filter,
            )
            # If scoped search returned nothing, fall back to global corpus search
            if not hits and doc_filter is not None:
                hits = hybrid_search(
                    query=effective_query,
                    collection=collection,
                    bm25=bm25,
                    texts=texts,
                    metadatas=metadatas,
                    top_k=top_k,
                    doc_filter=None,
                )
            if hits and expand_window > 0 and texts and metadatas:
                expand_neighbours(hits, texts, metadatas, window=expand_window)
        except Exception as e:
            logger.warning("Database query failed for '%s': %s", effective_query, e)
            hits = []

    formatted_hits: List[Dict[str, Any]] = []
    for h in hits:
        meta = h.get("metadata") or {}
        doc = meta.get("document") or ""
        source_role = "seed_paper" if (seed_doc_key and doc == seed_doc_key) else (
            "cited_reference" if doc in cited_doc_keys else "corpus"
        )
        formatted_hits.append({
            "document": doc,
            "source_role": source_role,
            "citation": meta.get("citation_source") or doc,
            "section": meta.get("section") or "text",
            "page": meta.get("page_first", meta.get("page", "?")),
            "chunk_index": h.get("chunk_index", -1),
            "rrf_score": h.get("rrf_score", 0.0),
            "context_before": (h.get("context_before") or "").strip(),
            "text": (h.get("text") or "").strip(),
            "context_after": (h.get("context_after") or "").strip(),
        })

    # If no live vector DB hits were found (e.g., offline mode) and task has a context_pack, surface cached passages
    if not formatted_hits and task_obj and task_obj.context_pack:
        for p in task_obj.context_pack.seed_passages[:top_k]:
            formatted_hits.append({
                "document": p.get("document") or seed_doc_key,
                "source_role": "seed_paper_cached",
                "citation": p.get("document") or seed_doc_key,
                "section": p.get("section") or "text",
                "page": p.get("page", "?"),
                "chunk_index": p.get("chunk_index", -1),
                "rrf_score": p.get("rrf_score", 0.0),
                "context_before": p.get("context_before", ""),
                "text": p.get("text", ""),
                "context_after": p.get("context_after", ""),
            })
        for c in task_obj.context_pack.cited_protocols[:top_k]:
            formatted_hits.append({
                "document": c.get("cited_doc_key") or "cited_ref",
                "source_role": "cited_reference_cached",
                "citation": c.get("cited_title") or c.get("cite_text") or "",
                "section": c.get("section") or "text",
                "page": c.get("page", "?"),
                "chunk_index": c.get("chunk_index", -1),
                "rrf_score": c.get("rrf_score", 0.0),
                "context_before": "",
                "text": c.get("text", ""),
                "context_after": "",
            })

    return {
        "query": effective_query,
        "scope": scope_clean,
        "task_id": task_id,
        "seed_doc_key": seed_doc_key,
        "subgraph_size": len(subgraph_keys),
        "results": formatted_hits,
    }


def format_query_results_markdown(result: Dict[str, Any]) -> str:
    """Formats the output of `query_task_database()` as a clean Markdown report for agents."""
    if not result or not isinstance(result, dict):
        return "No results found."
    lines = [
        f"# Database Query Results",
        f"- **Query:** `{result.get('query', '')}`",
        f"- **Scope:** `{result.get('scope', 'subgraph')}` (Subgraph size: {result.get('subgraph_size', 0)} docs)",
    ]
    if result.get("task_id"):
        lines.append(f"- **Task ID:** `{result['task_id']}`")
    lines.append("")

    hits = result.get("results") or []
    if not hits:
        lines.append("_No matching passages found in the database._")
        return "\n".join(lines)

    for idx, h in enumerate(hits, 1):
        doc = h.get("document") or "unknown"
        role = h.get("source_role") or "corpus"
        sec = h.get("section") or "text"
        page = h.get("page", "?")
        score = h.get("rrf_score", 0.0)
        lines.append(f"## [{idx}] {doc} (`{role}`) — Page {page}, Section: {sec} (RRF: {score:.4f})")
        if h.get("citation") and h.get("citation") != doc:
            lines.append(f"- **Citation:** {h['citation']}")
        if h.get("context_before"):
            lines.append(f"> _[Preceding Context]_ {h['context_before']}")
        lines.append(f"\n{h.get('text', '')}\n")
        if h.get("context_after"):
            lines.append(f"> _[Following Context]_ {h['context_after']}")
        lines.append("")

    return "\n".join(lines)
