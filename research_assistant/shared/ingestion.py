"""
Shared ingestion pipeline — the single implementation used by Agent 3, Agent 6
and the orchestrator.

Provides:
    - ingest_pdfs()     — Batch ingest: process → upsert → mark → index.
                          Callers should use this rather than composing the
                          steps below themselves, so the check/mark bookkeeping
                          stays consistent across entry points.
    - process_pdf()     — GROBID full text → chunks, captions and figure crops
    - upsert_corpus()   — Embed (text and figure images) + ChromaDB upsert
    - rebuild_bm25()    — Full BM25 index rebuild from the collection
    - pdf_key()         — Normalised document identity used by both stores
"""

import contextlib
import json
import os
import re
import pickle
import threading
import time
import warnings
import concurrent.futures

try:
    import fcntl
except ImportError:      # non-POSIX: fall back to the in-process lock alone
    fcntl = None

warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# PyMuPDF, ChromaDB and the embedding stack are imported inside the functions
# that use them, not here.
#
# This is deliberate and load-bearing, not a style preference: CI installs only
# requirements-test.txt, and test_ingestion.py imports this module. Moving these
# to module scope makes the whole suite uncollectable there.
#
# The bookkeeping in ingest_pdfs() and the text helpers are ordinary Python;
# keeping them importable on their own is what makes them testable anywhere.

from research_assistant.config import (
    DATA_DIR,
    COLLECTION_NAME,
    SUMMARY_COLLECTION_NAME,
    INDEX_VERSION,
    CHUNK_MIN_LENGTH,
    EMBED_BATCH_SIZE,
    EMBED_MAX_CHARS,
    SUMMARY_MODEL,
    SUMMARY_FALLBACK_MODEL,
    SUMMARY_FALLBACK_BACKEND,
    LLM_MODEL,
    SUMMARY_MAX_CHARS,
    PULLED_PDFS_DIR,
    RAW_DIR,
    PAPER_INDEX_PATH,
    DOWNLOADED_JSON_PATH,
)
from research_assistant import config
from research_assistant.prompts import DOCUMENT_SUMMARY
from research_assistant.shared.atomic import atomic_write
from research_assistant.shared.log import get_logger
from research_assistant.shared.db import check_embedder, embedder_record, get_max_chunk_index, open_store
from research_assistant.shared import manifest
from research_assistant.shared.tokenize import tokenize, TOKENIZER_VERSION
from research_assistant.shared import pipeline_status
from research_assistant.shared.paper_header import read_header, PaperHeader
from research_assistant.shared.dedupe_sweep import run_sweep
from research_assistant.shared.paper_index import (
    PaperEntry,
    PaperIndex,
    edit,
    normalise_arxiv,
    repoint_manifest,
    surname_key,
)
from research_assistant.shared.source_key import fold_title, normalise_doi
from research_assistant.shared.grobid_manager import is_grobid_alive
from research_assistant.shared import docling_extract

logger = get_logger("ingestion")

# Serialises ingestion across threads and processes — see _ingest_lock().
INGEST_LOCK_PATH = os.path.join(DATA_DIR, "ingest.lock")

# Re-entrant so a caller already inside ingest_pdfs() does not deadlock itself.
_ingest_thread_lock = threading.RLock()


@contextlib.contextmanager
def _ingest_lock():
    """Hold exclusive access to the corpus for the duration of one ingest.

    upsert_corpus() allocates chunk ids by reading the collection's current
    maximum and counting up from it, and rebuild_bm25() rewrites a single
    pickle. Both are read-modify-write against shared state, so two ingests
    running at once mint colliding ids and race on the index file.

    They really do run at once by default: watch.py drives the raw/ pipeline on
    one thread while its pulled_pdfs/ watcher drains on another, and Agent 2
    downloads *into* pulled_pdfs/ — so fetching references arms the second
    thread every time. app.py and watch.py can also be running side by side.

    The threading lock covers threads in this process; flock covers separate
    processes sharing a DATA_DIR. On a platform without fcntl the in-process
    lock still applies.
    """
    with _ingest_thread_lock:
        if fcntl is None:
            yield
            return
        os.makedirs(os.path.dirname(INGEST_LOCK_PATH) or ".", exist_ok=True)
        with open(INGEST_LOCK_PATH, "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                # Parsing a batch takes minutes, so say why we are stopped
                # rather than letting the caller sit on a silent block.
                logger.info(
                    "Another ingest holds %s — waiting for it to finish.",
                    INGEST_LOCK_PATH,
                )
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


# ─── Core PDF processing ─────────────────────────────────────────────────────

# Entries arrive already chunked from ingest_v2; these pass straight to the store.
# "figure_image" carries a precomputed image embedding (see ingest_v2).
PRECHUNKED_TYPES = ("text_chunk", "caption", "figure_image", "formula")


def process_pdf(
    pdf_path: str,
    citation_string: str,
    images_dir=None,
    enable_ocr: bool | None = None,
    ocr_engine: str | None = None,
):
    """Extract a single PDF into corpus entries (shared.ingest_v2): GROBID full
    text with the PyMuPDF and OCR fallback, sentence-window chunks, captions, figure
    crops and their image embeddings.

    Every entry is tagged with the extraction mode that produced it so
    ingest_pdfs() can report how the batch was actually extracted.
    """
    from research_assistant.shared.ingest_v2 import process_pdf_v2

    return process_pdf_v2(
        pdf_path,
        citation_string,
        images_dir=images_dir,
        enable_ocr=enable_ocr,
        ocr_engine=ocr_engine,
    )


# ─── Document summaries (stage-1 index) ─────────────────────────────────────


# What the most recent upsert_summaries() call attempted and lost. upsert_corpus()
# returns a bare chunk count, so this is how ingest_pdfs() learns that summaries
# failed without changing that return type. Reset at the start of every call.
last_summary_run: dict = {"attempted": 0, "made": 0, "failed": 0, "error": "", "failed_docs": []}


def upsert_summaries(per_doc_text: dict, per_doc_citation: dict) -> int:
    """One LLM summary per new document, into SUMMARY_COLLECTION_NAME.

    This is the coarse "is this paper even relevant" index that retrieval
    consults before touching the detail chunks. Cost is one model call per
    document, once.
    """
    from research_assistant.shared.llm import chat, get_embeddings

    embeddings = get_embeddings()
    client = open_store()
    col = client.get_or_create_collection(
        name=SUMMARY_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", **embedder_record(embeddings)},
    )
    check_embedder(col)

    have = set()
    got = col.get(include=["metadatas"])
    for m in got.get("metadatas") or []:
        if m and m.get("document"):
            have.add(m["document"])

    made = 0
    failed = 0
    failed_docs = set()
    last_error = ""
    last_summary_run.update(attempted=0, made=0, failed=0, error="", failed_docs=[])
    to_summarize = [doc for doc, text in per_doc_text.items() if doc not in have and text.strip()]
    total_to_sum = len(to_summarize)
    model_name = SUMMARY_MODEL or LLM_MODEL
    if total_to_sum:
        pipeline_status.add_event(f"📝 Generating summaries for {total_to_sum} paper(s) with {model_name}…")

    def _summarize_doc(doc):
        if pipeline_status.is_cancel_requested():
            return doc, None, "cancelled"
        text = per_doc_text[doc]
        msg = [{"role": "user", "content": DOCUMENT_SUMMARY.format(text=text[:SUMMARY_MAX_CHARS])}]
        try:
            summary = chat(
                msg,
                model=SUMMARY_MODEL,
            ).content.strip()
            if not summary:
                raise ValueError("the model returned an empty summary")
            return doc, summary, None
        except Exception as e:
            # Fall back to the local Ollama model if the primary summary model fails
            fb_model = getattr(config, "SUMMARY_FALLBACK_MODEL", config.DEFAULT_OLLAMA_MODEL)
            fb_backend = getattr(config, "SUMMARY_FALLBACK_BACKEND", "ollama")
            is_same = (SUMMARY_MODEL or LLM_MODEL) == fb_model and (config.LLM_BACKEND == fb_backend)
            if not is_same and not pipeline_status.is_cancel_requested():
                logger.warning(
                    "Primary summary model (%s) failed on %s: %s. Falling back to %s (%s)...",
                    model_name, doc, e, fb_backend, fb_model,
                )
                try:
                    fb_summary = chat(
                        msg,
                        model=fb_model,
                        backend=fb_backend,
                    ).content.strip()
                    if fb_summary:
                        logger.info("Summary generated with fallback %s for %s", fb_model, doc)
                        return doc, fb_summary, None
                    logger.warning("Fallback model %s returned an empty summary for %s", fb_model, doc)
                except Exception as fb_err:
                    logger.error("Summary fallback model (%s) also failed on %s: %s", fb_model, doc, fb_err)
                    return doc, None, f"primary ({model_name}): {e} | fallback ({fb_model}): {fb_err}"
            return doc, None, str(e)

    workers = min(4, total_to_sum) if total_to_sum else 1
    summaries = []
    if workers <= 1:
        for idx, doc in enumerate(to_summarize, 1):
            if pipeline_status.is_cancel_requested():
                logger.info("Summary generation stopped early by user request at %d/%d", idx, total_to_sum)
                for unfinished_doc in to_summarize[idx - 1:]:
                    failed_docs.add(unfinished_doc)
                break
            pipeline_status.update_progress(
                item_current=idx,
                item_total=total_to_sum,
                current_item_name=doc,
                detail=f"Generating summary with {model_name} ({idx}/{total_to_sum})",
            )
            doc_id, summary_text, err = _summarize_doc(doc)
            if err:
                failed += 1
                failed_docs.add(doc)
                last_error = err
            else:
                summaries.append((doc_id, summary_text))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            fut_to_doc = {executor.submit(_summarize_doc, doc): doc for doc in to_summarize}
            done_count = 0
            for fut in concurrent.futures.as_completed(fut_to_doc):
                done_count += 1
                doc_id, summary_text, err = fut.result()
                pipeline_status.update_progress(
                    item_current=done_count,
                    item_total=total_to_sum,
                    current_item_name=doc_id,
                    detail=f"Generating summary with {model_name} ({done_count}/{total_to_sum})",
                )
                if err:
                    failed += 1
                    failed_docs.add(doc_id)
                    last_error = err
                else:
                    summaries.append((doc_id, summary_text))

    if summaries:
        try:
            docs_list = [d for d, _ in summaries]
            sums_list = [s for _, s in summaries]
            embs = _embed_documents_safe(embeddings, sums_list, sub_batch_size=16)
            col.add(
                ids=[f"sum::{d}" for d in docs_list],
                documents=sums_list,
                embeddings=embs,
                metadatas=[{"document": d, "citation_source": per_doc_citation.get(d, "")} for d in docs_list],
            )
            made += len(summaries)
        except Exception as e:
            logger.error("Failed to add summaries to store: %s", e)
            for d, _ in summaries:
                failed += 1
                failed_docs.add(d)
            last_error = str(e)

    last_summary_run.update(attempted=total_to_sum, made=made, failed=failed, error=last_error, failed_docs=list(failed_docs))
    if made:
        pipeline_status.add_event(f"✓ Generated {made} paper summaries with {model_name}")
        logger.info("✓ Wrote %d document summary/-ies.", made)
    if failed:
        # Without a summary a paper is invisible to stage-1 retrieval, so a run
        # that lost them is not a clean success, and must not look like one.
        msg = f"{failed} of {total_to_sum} summaries failed with {model_name}: {last_error}"
        logger.warning(msg)
        pipeline_status.add_event(f"⚠️ {msg}")
    return made


# ─── ChromaDB upsert ─────────────────────────────────────────────────────────


def _embed_documents_safe(embeddings, texts: list[str], sub_batch_size: int = 32) -> list[list[float]]:
    """Embed texts safely in sub-batches, with single-item fallback if a batch fails."""
    if not texts:
        return []
    results = []
    for j in range(0, len(texts), sub_batch_size):
        sub_chunk = texts[j : j + sub_batch_size]
        try:
            embs = embeddings.embed_documents(sub_chunk)
            results.extend(embs)
        except Exception as e:
            logger.warning(
                "Sub-batch embedding of %d texts failed (%s); trying one by one...",
                len(sub_chunk),
                e,
            )
            for text in sub_chunk:
                safe_text = (text or "")[:EMBED_MAX_CHARS]
                try:
                    embs = embeddings.embed_documents([safe_text])
                    results.extend(embs)
                except Exception as inner_e:
                    logger.error(
                        "Failed to embed single text chunk: %s. Using query fallback or zero vector.",
                        inner_e,
                    )
                    try:
                        emb = embeddings.embed_query(safe_text[:1000])
                        results.append(emb)
                    except Exception:
                        dim = len(results[0]) if results else 768
                        results.append([0.0] * dim)
    return results


def upsert_corpus(corpus: list[dict]):
    """Embed the corpus entries and upsert them into the vector store (LanceDB or ChromaDB).

    Entries arrive already chunked (ingest_v2). A text entry is embedded here
    from its ``embed_text``; a ``figure_image`` entry already carries its image
    embedding in ``embedding`` and is stored as it is.

    Returns the number of new chunks inserted.
    """
    if not corpus:
        logger.info("Empty corpus — nothing to ingest.")
        return 0


    from research_assistant.shared.llm import get_embeddings

    t0 = time.perf_counter()
    embeddings = get_embeddings()
    store_client = open_store()
    collection = store_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", **embedder_record(embeddings)},
    )
    check_embedder(collection)

    # One summary_source entry per document feeds the stage-1 summary index.
    total_corpus = []
    per_doc_text, per_doc_citation = {}, {}
    for entry in corpus:
        if entry["type"] in PRECHUNKED_TYPES:
            total_corpus.append(entry)
        elif entry["type"] == "summary_source":
            per_doc_text[entry["document"]] = entry.get("content") or ""
            per_doc_citation.setdefault(entry["document"], entry["citation"])

    chunk_time = time.perf_counter() - t0
    logger.info("Prepared %d entries in %.1fs", len(total_corpus), chunk_time)

    # ── Deduplicate against existing DB content ──────────────────────────
    # Keyed on (document, text), not text alone. Two papers legitimately share
    # sentences — boilerplate, standard equations, common method descriptions —
    # and a global key silently dropped the second paper's copy, leaving that
    # passage attributed to whichever paper was ingested first. Within one
    # document a repeat is still a duplicate and is still skipped.
    from research_assistant.shared.lance_store import compute_chunk_id

    is_lance = getattr(collection, "is_lancedb", False)
    existing_chunks = set()
    incoming_docs = {entry["document"] for entry in total_corpus if entry.get("document")}
    if incoming_docs:
        doc_list = sorted(incoming_docs)
        # Targeted per-document deduplication checks instead of O(N) full-table RAM scan
        batch_size = 50
        for i in range(0, len(doc_list), batch_size):
            sub_docs = doc_list[i : i + batch_size]
            try:
                where_clause = {"document": {"$in": sub_docs}} if len(sub_docs) > 1 else {"document": sub_docs[0]}
                batch = collection.get(
                    where=where_clause,
                    include=["documents", "metadatas"],
                )
                if batch and batch.get("documents"):
                    for stored, meta in zip(batch["documents"], batch.get("metadatas") or []):
                        doc_name = (meta or {}).get("document")
                        if doc_name in incoming_docs:
                            existing_chunks.add((doc_name, stored))
            except Exception as exc:
                logger.warning("Targeted deduplication check error: %s", exc)

    current_index = get_max_chunk_index(collection) if not is_lance else 0
    documents, embed_texts, metadatas, ids, seen = [], [], [], [], set()
    precomputed = []                    # per entry: a ready vector (figure images) or None

    for entry in total_corpus:
        content = entry["content"]
        # Compare on the same form that reaches the store: add() truncates to
        # EMBED_MAX_CHARS, so comparing full text against stored text would
        # never match for a long chunk and would re-insert it on every run.
        stored_form = content[:EMBED_MAX_CHARS]
        identity = (entry["document"], stored_form)
        if identity in seen or identity in existing_chunks or len(content.strip()) < CHUNK_MIN_LENGTH:
            continue
        seen.add(identity)
        meta = {
            "document": entry["document"],
            "page": entry["page"],
            "type": entry["type"],
            "citation_source": entry["citation"],
        }
        if "metadata" in entry:
            for k, v in entry["metadata"].items():
                meta[f"extra_{k}"] = str(v)
        # v2: flat metadata under its own names (section, seq, figure_id, …).
        # Chroma accepts str / int / float / bool values only.
        for k, v in (entry.get("meta") or {}).items():
            meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        documents.append(stored_form)
        # A contextual header + text is embedded; a figure image arrives embedded.
        embed_texts.append((entry.get("embed_text") or content)[:EMBED_MAX_CHARS])
        precomputed.append(entry.get("embedding"))
        metadatas.append(meta)
        if is_lance:
            seq = int(meta.get("seq", 0))
            ids.append(compute_chunk_id(entry["document"], stored_form, seq, content))
        else:
            ids.append(f"chunk_{current_index}")
            current_index += 1

    if documents:
        db_name = "LanceDB" if is_lance else "ChromaDB"
        logger.info("Embedding and ingesting %d chunks into %s…", len(documents), db_name)
        pipeline_status.update_progress(
            item_current=0,
            item_total=len(documents),
            current_item_name=f"{db_name} Vectors",
            detail=f"Embedding {len(documents)} chunks",
        )
        pipeline_status.add_event(f"🧠 Embedding {len(documents)} chunks into {db_name}…")
        embed_t0 = time.perf_counter()
        for i in range(0, len(documents), EMBED_BATCH_SIZE):
            if pipeline_status.is_cancel_requested():
                logger.info("Embedding stopped early by user request at chunk %d/%d", i, len(documents))
                break
            batch_embed_texts = embed_texts[i : i + EMBED_BATCH_SIZE]
            batch_ready = precomputed[i : i + EMBED_BATCH_SIZE]
            todo = [t for t, v in zip(batch_embed_texts, batch_ready) if v is None]
            fresh = iter(_embed_documents_safe(embeddings, todo))
            batch_embs = [v if v is not None else next(fresh) for v in batch_ready]
            collection.add(
                embeddings=batch_embs,
                documents=documents[i : i + EMBED_BATCH_SIZE],
                metadatas=metadatas[i : i + EMBED_BATCH_SIZE],
                ids=ids[i : i + EMBED_BATCH_SIZE],
            )
            batch_end = min(i + EMBED_BATCH_SIZE, len(documents))
            pipeline_status.update_progress(
                item_current=batch_end,
                item_total=len(documents),
                detail=f"Embedding {len(documents)} chunks ({batch_end}/{len(documents)})",
            )
        elapsed = time.perf_counter() - t0
        embed_elapsed = time.perf_counter() - embed_t0
        pipeline_status.add_event(f"✅ Embedded and indexed {len(documents)} chunks")
        logger.info(
            "✓ Ingested %d chunks into %s in %.1fs (embed: %.1fs, total: %.1fs)",
            len(documents), db_name, elapsed, embed_elapsed, elapsed,
        )
        if hasattr(collection, "optimize"):
            try:
                collection.optimize()
            except Exception as exc:
                logger.warning("Post-ingest collection optimization skipped: %s", exc)
    else:
        logger.info("No new chunks to insert (all duplicates or empty).")

    # Stage-1 summary index — one model call per new document.
    try:
        upsert_summaries(per_doc_text, per_doc_citation)
    except Exception as e:
        logger.error("Summary index update failed: %s", e)
        pipeline_status.add_event(f"⚠️ Summary index update failed: {e}")
        failed_doc_ids = set(last_summary_run.get("failed_docs", []))
        if per_doc_text:
            failed_doc_ids.update(per_doc_text.keys())
        last_summary_run.update(
            failed=max(last_summary_run.get("failed", 0), len(failed_doc_ids), 1),
            error=str(e),
            failed_docs=list(failed_doc_ids),
        )

    return len(documents)


# ─── Already-ingested check ──────────────────────────────────────────────────


def get_ingested_documents() -> set[str]:
    """Documents already parsed, per the manifest.

    The collection scan this used to do on every call is now
    shared.manifest.rebuild_from_collection(), a repair path — reading every
    metadata record in the corpus before ingesting one file was the single
    most expensive thing about a warm run.
    """
    return set(manifest.load())


def pdf_key(pdf_path: str) -> str:
    """Return the normalised document key for *pdf_path*.

    This is the identity a PDF has everywhere in the system: the ``document``
    field written into ChromaDB metadata, and the key written to the ingestion
    manifest. It was previously re-derived inline in four separate places, so a
    change in one would silently stop matching the others.
    """
    return os.path.basename(pdf_path).strip().replace(" ", "_").lower()


# ─── BM25 rebuild ─────────────────────────────────────────────────────────────


def rebuild_bm25():
    """Refresh the LanceDB full-text search (Tantivy FTS) index."""
    if pipeline_status.is_cancel_requested():
        logger.info("rebuild_bm25 skipped because cancellation was requested")
        return

    t0 = time.perf_counter()
    logger.info("Refreshing LanceDB Tantivy FTS index…")
    pipeline_status.update_progress(
        current_item_name="FTS Index",
        detail="Refreshing Tantivy full-text search index…",
    )
    pipeline_status.add_event("📚 Refreshing Tantivy FTS index…")
    store_client = open_store()
    collection = store_client.get_collection(name=COLLECTION_NAME)

    if hasattr(collection, "create_fts_index"):
        collection.create_fts_index("text", replace=True)
        pipeline_status.add_event("📚 Refreshed LanceDB Tantivy FTS index")

    from research_assistant.shared.db import clear_search_cache
    clear_search_cache()
    elapsed = time.perf_counter() - t0
    logger.info("✓ Tantivy FTS index refreshed in %.1fs.", elapsed)



# ─── Unified batch ingestion ──────────────────────────────────────────────────


def ingest_pdfs(
    pdfs,
    workers: int = 1,
    skip_ingested: bool = True,
    rebuild_index: bool = True,
    log_prefix: str = "",
    require_grobid: bool = False,
    enable_ocr: bool | None = None,
    ocr_engine: str | None = None,
) -> dict:
    """Ingest a batch of PDFs: process → upsert → mark → rebuild the BM25 index.

    This is the single ingestion path shared by Agent 3 (batch from
    downloaded.json), Agent 6 (one manually placed file) and the orchestrator
    (startup sync and the pulled_pdfs/ watcher).

    Having one implementation matters because the check/mark bookkeeping has to
    agree across callers. Processing is by far the most expensive step in the
    pipeline — GROBID full text plus an image embedding per figure — so a
    PDF that gets processed but never marked is re-processed in full on every
    subsequent run, and one that gets marked but never checked is processed
    twice in the same run.

    Args:
        pdfs:          Mapping of ``{pdf_path: citation_label}``, or an iterable
                       of paths (the filename stem is then used as the label).
        workers:       Worker processes for the parsing stage. 1 runs in-process.
        skip_ingested: Skip PDFs already recorded as ingested.
        rebuild_index: Rebuild the BM25 index once at the end. Pass False when
                       ingesting several batches and rebuild once yourself.
        log_prefix:    Prefix for log lines, e.g. ``"[Sync] "``.
        enable_ocr:    Enable OCR for scanned/image-only PDFs (defaults to config.OCR_ENABLED).
        ocr_engine:    OCR engine to use ("easyocr" or "vlm").

    Returns:
        dict: ``{"processed", "skipped", "inserted", "failed"}`` — ``failed``
        is the list of paths that could not be read.
    """
    with _ingest_lock():
        res = _ingest_pdfs_locked(
            pdfs,
            workers,
            skip_ingested,
            rebuild_index,
            log_prefix,
            require_grobid,
            enable_ocr=enable_ocr,
            ocr_engine=ocr_engine,
        )
    try:
        run_sweep()
    except Exception as exc:
        logger.warning("%sEnd-of-ingest dedupe sweep failed: %s", log_prefix, exc)
    return res


def _read_header_for_dedupe(path: str) -> PaperHeader | None:
    header = read_header(path)
    if header is not None:
        return header
    cache_path = docling_extract._docling_cache_path(path)
    if not os.path.exists(cache_path):
        return read_header(path, convert_if_missing=True)
    return None


def _keeper_rank(document: str) -> tuple[int, str]:
    if document.startswith("doi_"):
        return (0, document)
    if document.startswith("arxiv_"):
        return (1, document)
    return (2, document)


def _plan_drops(
    candidates: dict[str, str],
    headers: dict[str, PaperHeader | None],
    index: PaperIndex,
    ingested: set[str],
) -> dict[str, PaperEntry | str]:
    """Plan duplicate drops against store and within batch.

    Returns mapping of dropped path -> PaperEntry (store match) or keeper path (batch match).
    """
    drops: dict[str, PaperEntry | str] = {}
    accepted_batch: list[tuple[str, PaperHeader]] = []
    candidate_paths = {os.path.abspath(p) for p in candidates}

    sorted_paths = sorted(candidates.keys(), key=lambda p: _keeper_rank(pdf_key(p)))

    for path in sorted_paths:
        doc_key = pdf_key(path)
        header = headers.get(path)
        if not header or not header.title or not header.first_authors:
            continue

        f_title = fold_title(header.title)
        if not f_title:
            continue

        s_keys = {k for a in header.first_authors if (k := surname_key(a)) is not None}
        if not s_keys:
            continue

        # 1. Store match
        store_match = None
        for entry in index.find_all_by_title_author(header.title, header.first_authors, exclude=doc_key):
            entry_abspath = os.path.abspath(entry.path)
            if (
                entry.document in ingested
                and os.path.isfile(entry.path)
                and entry_abspath != os.path.abspath(path)
                and entry_abspath not in candidate_paths
            ):
                store_match = entry
                break

        if store_match is not None:
            drops[path] = store_match
            continue

        # 2. Batch match among already-accepted candidates
        batch_match_path = None
        for acc_path, acc_header in accepted_batch:
            if fold_title(acc_header.title) == f_title:
                acc_keys = {k for a in acc_header.first_authors if (k := surname_key(a)) is not None}
                if s_keys & acc_keys:
                    batch_match_path = acc_path
                    break

        if batch_match_path is not None:
            drops[path] = batch_match_path
        else:
            accepted_batch.append((path, header))

    return drops


def _execute_drops(
    drops: dict[str, PaperEntry | str],
    index: PaperIndex,
    log_prefix: str = "",
) -> list[dict]:
    """Execute duplicate drops: delete if in pulled_pdfs, alias, repoint, log, and event."""
    duplicates: list[dict] = []
    real_pulled = os.path.realpath(PULLED_PDFS_DIR)

    for path, target in drops.items():
        doc_key = pdf_key(path)
        if isinstance(target, PaperEntry):
            keeper_doc = target.document
            keeper_path = target.path
        else:
            keeper_path = target
            keeper_doc = pdf_key(target)

        real_path = os.path.realpath(path)
        is_in_pulled = (
            os.path.commonpath([real_path, real_pulled]) == real_pulled
            and real_path != real_pulled
        )
        # Record where the duplicate went before deleting it: if this raises, the PDF stays.
        index.add_alias(doc_key, keeper_doc)
        repoint_manifest(DOWNLOADED_JSON_PATH, path, keeper_path)

        deleted = False
        if is_in_pulled:
            try:
                os.unlink(path)
                deleted = True
            except OSError as exc:
                logger.warning("%sFailed to delete duplicate PDF %s: %s", log_prefix, path, exc)

        dup_rec = {
            "path": path,
            "document": doc_key,
            "kept": keeper_doc,
            "deleted": deleted,
        }
        duplicates.append(dup_rec)
        logger.info(
            "%sDuplicate dropped: %s (matches %s, deleted=%s)",
            log_prefix, doc_key, keeper_doc, deleted,
        )
        try:
            pipeline_status.add_event(
                f"🗑️ Dropped duplicate {doc_key} (matches {keeper_doc})"
            )
        except Exception:
            pass

    return duplicates


def _drop_duplicates_before_parse(
    candidates: dict[str, str],
    log_prefix: str = "",
) -> tuple[dict[str, str], list[dict], dict[str, PaperHeader | None]] | None:
    """Pre-parse duplicate check. Read headers and drop duplicates before parsing.

    Returns (kept_candidates, duplicate_records, headers), or None if cancelled.
    """
    headers: dict[str, PaperHeader | None] = {}
    n = len(candidates)

    for i, (path, label) in enumerate(candidates.items(), 1):
        if pipeline_status.is_cancel_requested():
            return None
        name = os.path.basename(path)
        pipeline_status.update_progress(
            item_current=i,
            item_total=n,
            detail=f"Checking for duplicates [{i}/{n}]: {name}",
        )
        headers[path] = _read_header_for_dedupe(path)

    with edit(PAPER_INDEX_PATH) as index:
        ingested = set(manifest.load())
        drops = _plan_drops(candidates, headers, index, ingested)
        duplicate_records = _execute_drops(drops, index, log_prefix=log_prefix)

    kept_candidates = {p: l for p, l in candidates.items() if p not in drops}
    return kept_candidates, duplicate_records, headers


def _drop_title_duplicates(
    candidates: dict[str, str],
    corpus: list[dict],
    log_prefix: str = "",
    headers: dict[str, PaperHeader | None] | None = None,
) -> tuple[dict[str, str], list[dict], list[dict], dict[str, PaperHeader]]:
    """Drop title+author duplicates before store upsert.

    Returns (kept candidates, kept corpus entries, duplicates, headers of kept paths).
    """
    merged_headers: dict[str, PaperHeader | None] = {}
    for path in candidates:
        if headers is not None and headers.get(path) is not None:
            merged_headers[path] = headers[path]
        else:
            merged_headers[path] = read_header(path)

    with edit(PAPER_INDEX_PATH) as index:
        ingested = set(manifest.load())
        drops = _plan_drops(candidates, merged_headers, index, ingested)
        duplicates = _execute_drops(drops, index, log_prefix=log_prefix)

    dropped_documents = {pdf_key(p) for p in drops}
    kept_candidates = {p: l for p, l in candidates.items() if p not in drops}
    kept_corpus = [
        entry for entry in corpus
        if entry.get("document") not in dropped_documents
    ]
    kept_headers = {
        p: merged_headers[p]
        for p in kept_candidates
        if merged_headers.get(p) is not None
    }

    return kept_candidates, kept_corpus, duplicates, kept_headers


def _add_ingested_to_paper_index(
    marked_paths: list[str],
    kept_headers: dict[str, PaperHeader],
) -> None:
    """Add successfully marked paths to paper index."""
    downloaded_records = {}
    if os.path.isfile(DOWNLOADED_JSON_PATH):
        try:
            with open(DOWNLOADED_JSON_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    downloaded_records = data
        except Exception as exc:
            logger.debug("Failed reading downloaded.json: %s", exc)

    by_basename = {}
    for rec in downloaded_records.values():
        if isinstance(rec, dict) and rec.get("path"):
            by_basename[os.path.basename(rec["path"])] = rec

    with edit(PAPER_INDEX_PATH) as index:
        for path in marked_paths:
            doc_key = pdf_key(path)
            header = kept_headers.get(path)
            rec = by_basename.get(os.path.basename(path), {})

            dois = set()
            arxiv_ids = set()

            if header:
                if header.doi:
                    dois.add(header.doi)
                if header.arxiv_id:
                    arxiv_ids.add(header.arxiv_id)

            if rec.get("doi"):
                norm_d = normalise_doi(rec["doi"])
                if norm_d:
                    dois.add(norm_d)
            if rec.get("arxiv_id"):
                norm_a = normalise_arxiv(rec["arxiv_id"])
                if norm_a:
                    arxiv_ids.add(norm_a)

            entry = PaperEntry(
                document=doc_key,
                path=path,
                title=header.title if header else None,
                first_authors=header.first_authors if header else (),
                dois=tuple(sorted(dois)),
                arxiv_ids=tuple(sorted(arxiv_ids)),
            )
            index.add(entry)


def _ingest_pdfs_locked(
    pdfs,
    workers,
    skip_ingested,
    rebuild_index,
    log_prefix,
    require_grobid=False,
    enable_ocr=None,
    ocr_engine=None,
) -> dict:
    """ingest_pdfs()'s body. Call it through ingest_pdfs(), which holds the lock."""
    if not isinstance(pdfs, dict):
        pdfs = {p: os.path.splitext(os.path.basename(p))[0] for p in pdfs}

    result = {"processed": 0, "skipped": 0, "inserted": 0, "failed": [], "duplicates": []}
    logger.info("%sIndex v%d.", log_prefix, INDEX_VERSION)

    candidates = {}
    for path, label in pdfs.items():
        resolved_path = path
        if not os.path.exists(resolved_path):
            cand = os.path.join(PULLED_PDFS_DIR, os.path.basename(path))
            if os.path.exists(cand):
                resolved_path = cand
            else:
                raw_cand = os.path.join(RAW_DIR, os.path.basename(path))
                if os.path.exists(raw_cand):
                    resolved_path = raw_cand
        if os.path.exists(resolved_path):
            candidates[resolved_path] = label
        else:
            logger.warning("%sFile not found, skipping: %s", log_prefix, path)
            result["failed"].append(path)

    if skip_ingested and candidates:
        already = get_ingested_documents()
        remaining = {p: l for p, l in candidates.items() if pdf_key(p) not in already}
        result["skipped"] = len(candidates) - len(remaining)
        if result["skipped"]:
            logger.info(
                "%sSkipping %d already-ingested PDF(s); %d to process.",
                log_prefix, result["skipped"], len(remaining),
            )
        candidates = remaining

    if not candidates:
        logger.info("%sNothing to ingest.", log_prefix)
        if result["skipped"]:
            pipeline_status.add_event(f"ℹ️ All {result['skipped']} PDF(s) already indexed into corpus")
            pipeline_status.update_progress(
                item_current=0,
                item_total=0,
                detail=f"All {result['skipped']} PDF(s) already ingested, skipping",
            )
        return result

    standalone = not pipeline_status.get_status().get("active")
    if standalone:
        pipeline_status.set_status(
            active=True,
            stage="ingest_refs",
            stage_label="Ingesting and summarizing papers",
            current_step=5,
            total_steps=5,
            item_total=len(candidates),
            item_current=0,
            detail=f"Starting ingestion of {len(candidates)} PDF(s)",
        )

    try:
        pre_headers: dict[str, PaperHeader | None] = {}
        pre_duplicates: list[dict] = []
        if not is_grobid_alive():
            logger.warning("%sGROBID is not reachable; skipping the pre-parse duplicate check.", log_prefix)
        else:
            pre_res = _drop_duplicates_before_parse(candidates, log_prefix=log_prefix)
            if pre_res is None:
                pipeline_status.add_event("⏹️ Ingestion stopped by user")
                logger.info("%sIngestion stopped early by user request.", log_prefix)
                return result
            kept_cands, pre_duplicates, pre_headers = pre_res
            n_dropped = len(pre_duplicates)
            n_total = len(candidates)
            cand_keys = {pdf_key(p) for p in candidates}
            n_batch = sum(1 for d in pre_duplicates if d["kept"] in cand_keys)
            n_store = n_dropped - n_batch
            n_unreadable = sum(
                1 for p in kept_cands
                if not pre_headers.get(p) or not pre_headers[p].title or not pre_headers[p].first_authors
            )
            logger.info(
                "%sPre-parse duplicate check: %d of %d PDF(s) dropped (%d match the store, %d within this batch); %d without a readable header will be parsed.",
                log_prefix,
                n_dropped,
                n_total,
                n_store,
                n_batch,
                n_unreadable,
            )
            candidates = kept_cands

        corpus = []
        scanned_or_empty = []
        if workers <= 1:
            logger.info("%sProcessing %d PDF(s) sequentially…", log_prefix, len(candidates))
            for i, (path, label) in enumerate(candidates.items(), 1):
                if pipeline_status.is_cancel_requested():
                    logger.info("%sIngestion stopped early by user request at PDF %d/%d", log_prefix, i, len(candidates))
                    pipeline_status.add_event(f"⏹️ Ingestion stopped by user ({i-1}/{len(candidates)} processed)")
                    break
                name = os.path.basename(path)
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(candidates),
                    current_item_name=label or name,
                    detail=f"Parsing PDF [{i}/{len(candidates)}]: {name}",
                )
                logger.info("%s[%d/%d] %s", log_prefix, i, len(candidates), name)
                entries = process_pdf(
                    path,
                    label,
                    enable_ocr=enable_ocr,
                    ocr_engine=ocr_engine,
                )
                if not entries:
                    scanned_or_empty.append(name)
                corpus.extend(entries)
        else:
            logger.info(
                "%sProcessing %d PDF(s) with %d workers…", log_prefix, len(candidates), workers
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_pdf,
                        path,
                        label,
                        None,
                        enable_ocr,
                        ocr_engine,
                    ): (path, label)
                    for path, label in candidates.items()
                }
                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    if pipeline_status.is_cancel_requested():
                        logger.info("%sIngestion stopped early by user request (cancelling pending jobs)", log_prefix)
                        for f in futures:
                            f.cancel()
                        pipeline_status.add_event(f"⏹️ Ingestion stopped by user ({completed_count}/{len(candidates)} processed)")
                        break
                    path, label = futures[future]
                    name = os.path.basename(path)
                    completed_count += 1
                    pipeline_status.update_progress(
                        item_current=completed_count,
                        item_total=len(candidates),
                        current_item_name=label or name,
                        detail=f"Parsed [{completed_count}/{len(candidates)}]: {name}",
                    )
                    try:
                        entries = future.result()
                        if not entries:
                            scanned_or_empty.append(name)
                        corpus.extend(entries)
                    except Exception as e:
                        logger.error("%sWorker failed on %s: %s", log_prefix, path, e)
                        result["failed"].append(path)

        result["processed"] = len(candidates)
        result["scanned_or_empty"] = scanned_or_empty

        if require_grobid:
            # A migration must not keep a paper GROBID could not read: its PyMuPDF text would sit in
            # the index beside the GROBID chunks a later run adds. Drop it, report it as failed so it
            # is not marked, and let the next run retry it.
            fell_back = {e["document"] for e in corpus if e.get("extraction") == "pymupdf" and e.get("document")}
            corpus = [e for e in corpus if e.get("document") not in fell_back]
            result["fell_back"] = sorted(fell_back)
            result["failed"].extend(p for p in candidates if pdf_key(p) in fell_back and p not in result["failed"])
            result["processed"] = len(candidates) - len(fell_back)
            if fell_back:
                logger.warning("%s%d document(s) could not be read by GROBID and were not stored: %s",
                               log_prefix, len(fell_back), ", ".join(sorted(fell_back)[:5]))

        # How the batch was actually extracted, counted per document rather than
        # per entry. Agent 1 records its GROBID-vs-regex fallback the same way, for
        # the same reason: a run that silently degraded should say so in its result,
        # not only in a warning somewhere up the log.
        from research_assistant.shared.ingest_v2 import EXTRACTION_MODES_V2 as modes

        by_mode = {mode: set() for mode in modes}
        for entry in corpus:
            docs = by_mode.get(entry.get("extraction"))
            if docs is not None and entry.get("document"):
                docs.add(entry["document"])
        result["extraction"] = {mode: len(docs) for mode, docs in by_mode.items()}
        result["figure_images"] = sum(1 for e in corpus if e.get("type") == "figure_image")

        if by_mode.get("ocr"):
            logger.info(
                "%s%d of %d document(s) extracted via OCR (%s).",
                log_prefix, len(by_mode["ocr"]), result["processed"], ocr_engine or "easyocr",
            )

        if by_mode["pymupdf"]:
            logger.warning(
                "%s%d of %d document(s) fell back to PyMuPDF extraction — no sections, "
                "captions or figures for them, and the bibliography may be in the text. "
                "Check GROBID at the configured GROBID_SERVER.",
                log_prefix, len(by_mode["pymupdf"]), result["processed"],
            )

        kept_candidates, corpus, duplicates, kept_headers = _drop_title_duplicates(
            candidates, corpus, log_prefix=log_prefix, headers=pre_headers
        )
        candidates = kept_candidates
        result["duplicates"] = pre_duplicates + duplicates

        if corpus:
            last_summary_run.update(attempted=0, made=0, failed=0, error="", failed_docs=[])
            result["inserted"] = upsert_corpus(corpus)
            result["summaries"] = dict(last_summary_run)
            logger.info("%sInserted %d new chunk(s).", log_prefix, result["inserted"])
        else:
            logger.info("%sNo content extracted from the processed PDF(s).", log_prefix)

        # Mark every PDF that was attempted and did not fail, including ones that
        # yielded nothing. A corrupt, empty, or duplicate paper produces no new
        # chunks, and without a mark it would be re-parsed on every single run for
        # the rest of time — that part is deliberate. But a path a worker crashed
        # on is also in candidates, and marking it too would hide the crash: only
        # --force (which reprocesses everything) would ever recover it.
        failed_paths = set(result["failed"])
        failed_summary_docs = set(last_summary_run.get("failed_docs", []))
        if last_summary_run.get("failed", 0) > 0 and not failed_summary_docs:
            failed_summary_docs = {entry.get("document") for entry in corpus if entry.get("document")}
        if failed_summary_docs:
            logger.warning("Excluding %d paper(s) from manifest because summary generation failed or was cancelled.", len(failed_summary_docs))
        if pipeline_status.is_cancel_requested():
            docs_in_corpus = {entry.get("document") for entry in corpus if entry.get("document")}
            marked = [
                path for path in candidates
                if path not in failed_paths
                and pdf_key(path) in docs_in_corpus
                and pdf_key(path) not in failed_summary_docs
            ]
        else:
            marked = [
                path for path in candidates
                if path not in failed_paths
                and pdf_key(path) not in failed_summary_docs
            ]
        if marked:
            manifest.add_many(pdf_key(path) for path in marked)
            _add_ingested_to_paper_index(marked, kept_headers)

        if pipeline_status.is_cancel_requested():
            logger.info("%sIngestion stopped early by user request.", log_prefix)
            return result

        if rebuild_index and result["inserted"]:
            rebuild_bm25()

        try:
            store_client = open_store()
            col = store_client.get_collection(COLLECTION_NAME)
            if hasattr(col, "optimize"):
                col.optimize()
        except Exception as exc:
            logger.debug("Post-batch optimization notice: %s", exc)

        pipeline_status.add_event(
            f"✅ Ingested {result['processed']} PDFs ({result['inserted']} chunks)"
        )
        pipeline_status.update_progress(
            item_current=len(candidates),
            item_total=len(candidates),
            detail=f"Ingestion batch complete: {result['processed']} processed, {result['inserted']} chunks inserted",
        )
        if standalone:
            pipeline_status.set_status(
                active=False,
                stage="idle",
                stage_label="Idle",
                detail="Ingestion complete",
                last_completed_at=pipeline_status._now_iso(),
                last_summary=f"+{result['processed']} papers ingested",
            )
        return result
    except BaseException as exc:
        if standalone:
            if pipeline_status.is_cancellation(exc):
                pipeline_status.add_event("⚠️ Ingestion cancelled (session reloaded or stopped)")
                pipeline_status.set_status(
                    active=False,
                    stage="idle",
                    stage_label="Idle",
                    detail="Ingestion cancelled",
                )
            else:
                detail_str = pipeline_status.format_exception_detail(exc)
                logger.exception("Ingestion failed: %s", exc)
                pipeline_status.add_event(f"❌ Ingestion failed: {detail_str}")
                pipeline_status.set_status(
                    active=False,
                    stage="idle",
                    stage_label="Idle",
                    detail=f"Ingestion failed: {detail_str}",
                )
        raise


# Backward compatibility alias
extract_pdf = process_pdf

