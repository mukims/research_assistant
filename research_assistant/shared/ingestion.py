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
    VECTORDB_PATH,
    COLLECTION_NAME,
    SUMMARY_COLLECTION_NAME,
    EMBED_MODEL,
    BM25_INDEX_PATH,
    INDEX_VERSION,
    CHUNK_MIN_LENGTH,
    EMBED_BATCH_SIZE,
    EMBED_MAX_CHARS,
    SUMMARY_MODEL,
    LLM_MODEL,
    SUMMARY_MAX_CHARS,
)
from research_assistant.prompts import DOCUMENT_SUMMARY
from research_assistant.shared.atomic import atomic_write
from research_assistant.shared.log import get_logger
from research_assistant.shared.db import check_embedder, get_max_chunk_index
from research_assistant.shared import manifest
from research_assistant.shared.tokenize import tokenize, TOKENIZER_VERSION
from research_assistant.shared import pipeline_status

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
PRECHUNKED_TYPES = ("text_chunk", "caption", "figure_image")


def process_pdf(pdf_path: str, citation_string: str, images_dir=None):
    """Extract a single PDF into corpus entries (shared.ingest_v2): GROBID full
    text with the PyMuPDF fallback, sentence-window chunks, captions, figure
    crops and their image embeddings.

    Every entry is tagged with the extraction mode that produced it so
    ingest_pdfs() can report how the batch was actually extracted.
    """
    from research_assistant.shared.ingest_v2 import process_pdf_v2

    return process_pdf_v2(pdf_path, citation_string, images_dir=images_dir)


# ─── Document summaries (stage-1 index) ─────────────────────────────────────


# What the most recent upsert_summaries() call attempted and lost. upsert_corpus()
# returns a bare chunk count, so this is how ingest_pdfs() learns that summaries
# failed without changing that return type. Reset at the start of every call.
last_summary_run: dict = {"attempted": 0, "made": 0, "failed": 0, "error": ""}


def upsert_summaries(per_doc_text: dict, per_doc_citation: dict) -> int:
    """One LLM summary per new document, into SUMMARY_COLLECTION_NAME.

    This is the coarse "is this paper even relevant" index that retrieval
    consults before touching the detail chunks. Cost is one model call per
    document, once.
    """
    import chromadb
    from research_assistant.shared.llm import chat, get_embeddings

    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    col = client.get_or_create_collection(
        name=SUMMARY_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedder": EMBED_MODEL},
    )
    check_embedder(col)

    have = set()
    got = col.get(include=["metadatas"])
    for m in got.get("metadatas") or []:
        if m and m.get("document"):
            have.add(m["document"])

    embeddings = get_embeddings()
    made = 0
    failed = 0
    last_error = ""
    last_summary_run.update(attempted=0, made=0, failed=0, error="")
    to_summarize = [doc for doc, text in per_doc_text.items() if doc not in have and text.strip()]
    total_to_sum = len(to_summarize)
    model_name = SUMMARY_MODEL or LLM_MODEL
    if total_to_sum:
        pipeline_status.add_event(f"📝 Generating summaries for {total_to_sum} paper(s) with {model_name}…")

    for idx, doc in enumerate(to_summarize, 1):
        if pipeline_status.is_cancel_requested():
            logger.info("Summary generation stopped early by user request at %d/%d", idx, total_to_sum)
            break
        pipeline_status.update_progress(
            item_current=idx,
            item_total=total_to_sum,
            current_item_name=doc,
            detail=f"Generating summary with {model_name} ({idx}/{total_to_sum})",
        )
        text = per_doc_text[doc]
        try:
            summary = chat(
                [{"role": "user", "content": DOCUMENT_SUMMARY.format(text=text[:SUMMARY_MAX_CHARS])}],
                model=SUMMARY_MODEL,
            ).content.strip()
        except Exception as e:
            logger.error("Summary failed for %s: %s", doc, e)
            failed += 1
            last_error = str(e)
            continue
        if not summary:
            logger.error("Summary for %s came back empty", doc)
            failed += 1
            last_error = last_error or "the model returned an empty summary"
            continue
        col.add(
            ids=[f"sum::{doc}"],
            documents=[summary],
            # A stored summary is a document, not a query: under the v2 nomic
            # prefixes the two are embedded differently. Identical for v1.
            embeddings=embeddings.embed_documents([summary]),
            metadatas=[{"document": doc, "citation_source": per_doc_citation.get(doc, "")}],
        )
        made += 1

    last_summary_run.update(attempted=total_to_sum, made=made, failed=failed, error=last_error)
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
    """Embed the corpus entries and upsert them into ChromaDB.

    Entries arrive already chunked (ingest_v2). A text entry is embedded here
    from its ``embed_text``; a ``figure_image`` entry already carries its image
    embedding in ``embedding`` and is stored as it is.

    Returns the number of new chunks inserted.
    """
    if not corpus:
        logger.info("Empty corpus — nothing to ingest.")
        return 0

    import chromadb

    from research_assistant.shared.llm import get_embeddings

    t0 = time.perf_counter()
    embeddings = get_embeddings()
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedder": EMBED_MODEL},
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
    existing_chunks = set()
    limit, offset = 5000, 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"], limit=limit, offset=offset
        )
        if not batch or not batch["documents"]:
            break
        for stored, meta in zip(batch["documents"], batch["metadatas"] or []):
            existing_chunks.add(((meta or {}).get("document"), stored))
        offset += limit

    current_index = get_max_chunk_index(collection)
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
        ids.append(f"chunk_{current_index}")
        current_index += 1

    if documents:
        logger.info("Embedding and ingesting %d chunks…", len(documents))
        pipeline_status.update_progress(
            item_current=0,
            item_total=len(documents),
            current_item_name="ChromaDB Vectors",
            detail=f"Embedding {len(documents)} chunks",
        )
        pipeline_status.add_event(f"🧠 Embedding {len(documents)} chunks into ChromaDB…")
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
            "✓ Ingested %d chunks into ChromaDB in %.1fs (embed: %.1fs, total: %.1fs)",
            len(documents), elapsed, embed_elapsed, elapsed,
        )
    else:
        logger.info("No new chunks to insert (all duplicates or empty).")

    # Stage-1 summary index — one model call per new document.
    try:
        upsert_summaries(per_doc_text, per_doc_citation)
    except Exception as e:
        logger.error("Summary index update failed: %s", e)
        pipeline_status.add_event(f"⚠️ Summary index update failed: {e}")
        last_summary_run.update(failed=max(last_summary_run["failed"], 1), error=str(e))

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
    """Rebuild the BM25 index from the entire ChromaDB collection."""
    if pipeline_status.is_cancel_requested():
        logger.info("rebuild_bm25 skipped because cancellation was requested")
        return

    import chromadb
    from rank_bm25 import BM25Okapi

    t0 = time.perf_counter()
    logger.info("Rebuilding BM25 index…")
    pipeline_status.update_progress(
        current_item_name="BM25 Index",
        detail="Rebuilding BM25 keyword index…",
    )
    pipeline_status.add_event("📚 Rebuilding BM25 index…")
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Use larger page size to reduce round-trips
    paired, limit, offset = [], 5000, 0
    while True:
        batch = collection.get(include=["documents", "metadatas"], limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for doc, meta, cid in zip(batch["documents"], batch["metadatas"] or [], batch["ids"]):
            try:
                idx = int(cid.split("_")[1])
            except (ValueError, IndexError):
                continue
            # The paper title rides along with every chunk (citation_source),
            # so a query that names the paper keyword-matches its chunks.
            title = (meta or {}).get("citation_source") or ""
            paired.append((idx, doc, title))
        offset += limit

    paired.sort(key=lambda x: x[0])
    bm25 = BM25Okapi([tokenize(f"{title} {text}") for _, text, title in paired])

    # Atomic: load_search_resources() unpickles this on every query, so a
    # partial write is not a degraded index but an exception on the next
    # search, recoverable only by re-ingesting. The header records which
    # tokenizer built it so search tokenises queries the same way.
    payload = {
        "tokenizer": TOKENIZER_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bm25": bm25,
    }
    with atomic_write(BM25_INDEX_PATH, binary=True) as f:
        pickle.dump(payload, f)
    # The cached (collection, bm25, texts, metadatas) in this process is now
    # stale; the mtime key would catch it, but say so explicitly.
    from research_assistant.shared.db import clear_search_cache
    clear_search_cache()
    elapsed = time.perf_counter() - t0
    pipeline_status.add_event(f"✅ BM25 index rebuilt ({len(paired)} chunks)")
    pipeline_status.update_progress(
        current_item_name="",
        detail=f"BM25 index rebuilt ({len(paired)} chunks)",
    )
    logger.info("✓ BM25 index rebuilt (%d documents, tokenizer %s) in %.1fs.",
                len(paired), TOKENIZER_VERSION, elapsed)



# ─── Unified batch ingestion ──────────────────────────────────────────────────


def ingest_pdfs(
    pdfs,
    workers: int = 1,
    skip_ingested: bool = True,
    rebuild_index: bool = True,
    log_prefix: str = "",
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

    Returns:
        dict: ``{"processed", "skipped", "inserted", "failed"}`` — ``failed``
        is the list of paths that could not be read.
    """
    with _ingest_lock():
        return _ingest_pdfs_locked(pdfs, workers, skip_ingested, rebuild_index, log_prefix)


def _ingest_pdfs_locked(pdfs, workers, skip_ingested, rebuild_index, log_prefix) -> dict:
    """ingest_pdfs()'s body. Call it through ingest_pdfs(), which holds the lock."""
    if not isinstance(pdfs, dict):
        pdfs = {p: os.path.splitext(os.path.basename(p))[0] for p in pdfs}

    result = {"processed": 0, "skipped": 0, "inserted": 0, "failed": []}
    logger.info("%sIndex v%d.", log_prefix, INDEX_VERSION)

    candidates = {}
    for path, label in pdfs.items():
        if os.path.exists(path):
            candidates[path] = label
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
                entries = process_pdf(path, label)
                if not entries:
                    scanned_or_empty.append(name)
                corpus.extend(entries)
        else:
            logger.info(
                "%sProcessing %d PDF(s) with %d workers…", log_prefix, len(candidates), workers
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_pdf, path, label): (path, label)
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

        if by_mode["pymupdf"]:
            logger.warning(
                "%s%d of %d document(s) fell back to PyMuPDF extraction — no sections, "
                "captions or figures for them, and the bibliography may be in the text. "
                "Check GROBID at the configured GROBID_SERVER.",
                log_prefix, len(by_mode["pymupdf"]), result["processed"],
            )

        if corpus:
            last_summary_run.update(attempted=0, made=0, failed=0, error="")
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
        if pipeline_status.is_cancel_requested():
            docs_in_corpus = {entry.get("document") for entry in corpus if entry.get("document")}
            marked = [path for path in candidates if path not in failed_paths and pdf_key(path) in docs_in_corpus]
        else:
            marked = [path for path in candidates if path not in failed_paths]
        if marked:
            manifest.add_many(pdf_key(path) for path in marked)

        if pipeline_status.is_cancel_requested():
            logger.info("%sIngestion stopped early by user request.", log_prefix)
            return result

        # Rebuilding is only worthwhile when the collection actually changed, but
        # the index must also exist for search to work at all.
        if rebuild_index and (result["inserted"] or not os.path.exists(BM25_INDEX_PATH)):
            rebuild_bm25()

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
