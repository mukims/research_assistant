#!/usr/bin/env python3
"""Rebuild the corpus into LanceDB using Docling extraction.

- Replaces GROBID / legacy extraction with Docling.
- Extracts clean vector plots and visual figures into data/images/ (no caption crops).
- Formats tables as Markdown and extracts mathematical formulas.
- Automatically detects scanned/image PDFs and routes to OCR.
- Pre-seeds existing summaries from data/migration_snapshot.json.
- Ingests abstract_only records from data/migration_snapshot.json.
- Embeds text and figure images using embeddinggemma-2 via Ollama into LanceDB.
- Rebuilds Tantivy full-text search indexes.
- Supports --limit, --offset, --batch-size, --force, and --dry-run.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

# Ensure repository root and package root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "research_assistant")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research_assistant.config import (
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_BATCH_SIZE,
    FIGURE_CROP_DPI,
    IMAGES_DIR,
    LANCEDB_PATH,
    PULLED_PDFS_DIR,
    RAW_DIR,
    SUMMARY_COLLECTION_NAME,
)
from research_assistant.shared import manifest
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.db import (
    check_embedder,
    embedder_record,
    open_store,
)
from research_assistant.shared.docling_extract import extract_docling, needs_ocr
from research_assistant.shared.ingest_v2 import process_pdf_v2
from research_assistant.shared.ingestion import (
    pdf_key,
    rebuild_bm25,
    upsert_corpus,
)
from research_assistant.shared.llm import get_embeddings
from research_assistant.shared.log import get_logger

logger = get_logger("rebuild_docling")


def load_snapshot(snapshot_path: str) -> dict:
    if not os.path.isfile(snapshot_path):
        logger.warning("Migration snapshot not found at %s", snapshot_path)
        return {"labels": {}, "existing_summaries": {}, "abstract_only_records": []}
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Loaded migration snapshot: %d labels, %d summaries, %d abstract-only records",
            len(data.get("labels", {})),
            len(data.get("existing_summaries", {})),
            len(data.get("abstract_only_records", [])),
        )
        return data
    except Exception as exc:
        logger.error("Failed to read migration snapshot from %s: %s", snapshot_path, exc)
        return {"labels": {}, "existing_summaries": {}, "abstract_only_records": []}


def preseed_summaries(
    store_client,
    embeddings,
    existing_summaries: dict,
    kept_keys: set[str] | None = None,
    batch_size: int = 32,
) -> int:
    """Pre-seed physics_summaries_v3 with existing summaries from migration snapshot for kept keys."""
    if not existing_summaries:
        return 0

    sum_col = store_client.get_or_create_collection(
        name=SUMMARY_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", **embedder_record(embeddings)},
    )
    check_embedder(sum_col)

    existing_docs = set()
    try:
        got = sum_col.get(include=["metadatas"])
        for m in got.get("metadatas") or []:
            if m and m.get("document"):
                existing_docs.add(m["document"])
    except Exception as exc:
        logger.debug("Error checking existing summaries: %s", exc)

    to_add = [
        (doc, data["summary"], data.get("citation_source") or doc)
        for doc, data in existing_summaries.items()
        if (kept_keys is None or doc in kept_keys) and doc not in existing_docs and data.get("summary", "").strip()
    ]

    if not to_add:
        logger.info("All %d snapshot summaries already present in %s.", len(existing_summaries), SUMMARY_COLLECTION_NAME)
        return 0

    logger.info("Pre-seeding %d existing summaries into %s...", len(to_add), SUMMARY_COLLECTION_NAME)
    inserted = 0
    t0 = time.perf_counter()

    for i in range(0, len(to_add), batch_size):
        chunk = to_add[i : i + batch_size]
        docs = [item[0] for item in chunk]
        texts = [item[1] for item in chunk]
        cits = [item[2] for item in chunk]
        ids = [f"sum::{d}" for d in docs]
        metadatas = [{"document": d, "citation_source": c} for d, c in zip(docs, cits)]

        try:
            vectors = embeddings.embed_documents(texts)
            sum_col.add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            inserted += len(chunk)
            if inserted % 200 == 0 or inserted == len(to_add):
                logger.info("Pre-seeded %d / %d summaries...", inserted, len(to_add))
        except Exception as exc:
            logger.error("Failed to embed/insert summary batch [%d:%d]: %s", i, i + len(chunk), exc)

    logger.info("✓ Pre-seeded %d summaries in %.1fs", inserted, time.perf_counter() - t0)
    return inserted


def compute_abstract_id(document: str) -> str:
    """Stable, process-independent chunk ID for abstract_only records."""
    return "chk_abs_" + hashlib.sha1(document.encode("utf-8")).hexdigest()[:16]


def delete_document_rows(store_client, doc_keys: list[str], preserve_summaries_for: set[str] | None = None):
    """Delete all rows for the given document keys from papers table, and from summaries table (except preserved)."""
    if not doc_keys or store_client is None:
        return

    # Delete from papers collection
    try:
        col = store_client.get_collection(COLLECTION_NAME)
        if hasattr(col, "delete"):
            if len(doc_keys) == 1:
                col.delete(where={"document": doc_keys[0]})
            else:
                col.delete(where={"document": {"$in": doc_keys}})
        elif hasattr(col, "table") and hasattr(col.table, "delete"):
            quoted = ", ".join(f"'{k.replace(chr(39), chr(39)*2)}'" for k in doc_keys)
            col.table.delete(f"document IN ({quoted})")
    except Exception as exc:
        logger.debug("Notice deleting rows for %s in %s: %s", doc_keys, COLLECTION_NAME, exc)

    # Delete from summaries collection (skipping preserved summaries)
    preserve = preserve_summaries_for or set()
    sum_keys = [k for k in doc_keys if k not in preserve]
    if sum_keys:
        try:
            col = store_client.get_collection(SUMMARY_COLLECTION_NAME)
            if hasattr(col, "delete"):
                if len(sum_keys) == 1:
                    col.delete(where={"document": sum_keys[0]})
                else:
                    col.delete(where={"document": {"$in": sum_keys}})
            elif hasattr(col, "table") and hasattr(col.table, "delete"):
                quoted = ", ".join(f"'{k.replace(chr(39), chr(39)*2)}'" for k in sum_keys)
                col.table.delete(f"document IN ({quoted})")
        except Exception as exc:
            logger.debug("Notice deleting rows for %s in %s: %s", sum_keys, SUMMARY_COLLECTION_NAME, exc)


def ingest_abstract_only(
    store_client,
    embeddings,
    abstract_records: list[dict],
    batch_size: int = 32,
) -> int:
    """Ingest abstract_only records into physics_papers_v3."""
    if not abstract_records:
        return 0

    papers_col = store_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", **embedder_record(embeddings)},
    )
    check_embedder(papers_col)

    existing_abstract_docs = set()
    try:
        got = papers_col.get(
            where={"stored_form": "abstract_only"},
            include=["metadatas"],
        )
        for m in got.get("metadatas") or []:
            if m and m.get("document"):
                existing_abstract_docs.add(m["document"])
    except Exception as exc:
        logger.debug("Error checking existing abstract_only records: %s", exc)

    to_add = [
        r for r in abstract_records
        if r.get("document") not in existing_abstract_docs and (r.get("text") or "").strip()
    ]

    if not to_add:
        logger.info("All %d abstract_only records already present in %s.", len(abstract_records), COLLECTION_NAME)
        return 0

    logger.info("Ingesting %d abstract_only records into %s...", len(to_add), COLLECTION_NAME)
    inserted = 0
    t0 = time.perf_counter()

    for i in range(0, len(to_add), batch_size):
        chunk = to_add[i : i + batch_size]
        docs = [r.get("document", "") for r in chunk]
        texts = [r.get("text", "") for r in chunk]
        cits = [r.get("citation_source") or r.get("document", "") for r in chunk]
        ids = [compute_abstract_id(d) for d in docs]
        metadatas = [
            {
                "document": d,
                "citation_source": c,
                "type": "abstract_only",
                "stored_form": "abstract_only",
                "tier": "abstract_only",
                "evidence_tier": "abstract_only",
                "page": 0,
                "page_first": 0,
                "page_last": 0,
                "section": "abstract",
                "seq": 0,
            }
            for d, c in zip(docs, cits)
        ]

        try:
            delete_document_rows(store_client, docs)
            vectors = embeddings.embed_documents(texts)
            papers_col.add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            sum_col = store_client.get_or_create_collection(
                name=SUMMARY_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", **embedder_record(embeddings)},
            )
            sum_metas = [{"document": d, "citation_source": c} for d, c in zip(docs, cits)]
            sum_ids = [f"sum::{d}" for d in docs]
            sum_col.add(
                embeddings=vectors,
                documents=texts,
                metadatas=sum_metas,
                ids=sum_ids,
            )
            manifest.add_many(docs)
            inserted += len(chunk)
        except Exception as exc:
            logger.error("Failed to embed/insert abstract_only batch [%d:%d]: %s", i, i + len(chunk), exc)

    logger.info("✓ Ingested %d abstract_only records in %.1fs", inserted, time.perf_counter() - t0)
    return inserted


def discover_pdfs() -> dict[str, str]:
    """Find all candidate PDFs in DATA_DIR (excluding quarantine, archive, and caches)."""
    candidates: dict[str, str] = {}
    for root, dirs, fnames in os.walk(DATA_DIR):
        if any(ignored in root for ignored in ("quarantine", "_archive", "docling_cache", "images", "lancedb")):
            continue
        for f in sorted(fnames):
            if f.lower().endswith(".pdf"):
                p = os.path.join(root, f)
                k = pdf_key(p)
                if k not in candidates:
                    candidates[k] = p
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Rebuild corpus using Docling and LanceDB")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of PDFs to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start processing from")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size of PDFs per LanceDB commit")
    parser.add_argument("--force", action="store_true", help="Reprocess PDFs even if in manifest")
    parser.add_argument("--dry-run", action="store_true", help="Extract and report without writing to LanceDB")
    parser.add_argument("--pdf", type=str, default=None, help="Process a single specific PDF path or key")
    parser.add_argument("--keys", type=str, default=None, help="Comma-separated list of PDF keys to process")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip pre-seeding summaries")
    parser.add_argument("--skip-abstracts", action="store_true", help="Skip ingesting abstract_only records")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR to prevent memory spikes on scanned PDFs")
    args = parser.parse_args()

    if args.no_ocr:
        os.environ["DOCLING_DISABLE_OCR"] = "1"

    if args.dry_run:
        from research_assistant import config
        from research_assistant.shared import ingest_v2
        config.EMBED_IMAGES = False
        ingest_v2.EMBED_IMAGES = False

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LANCEDB_PATH, exist_ok=True)

    snapshot_path = os.path.join(DATA_DIR, "migration_snapshot.json")
    snapshot = load_snapshot(snapshot_path)
    labels = snapshot.get("labels", {})
    existing_summaries = snapshot.get("existing_summaries", {})
    abstract_records = snapshot.get("abstract_only_records", [])

    dedupe_path = os.path.join(DATA_DIR, "corpus_dedupe.json")
    if not os.path.isfile(dedupe_path):
        logger.error("Required deduplication manifest '%s' is missing. Run dedupe_corpus_pdfs.py first.", dedupe_path)
        sys.exit(1)

    try:
        with open(dedupe_path, "r", encoding="utf-8") as f:
            dedupe_data = json.load(f)
        kept_keys = set(dedupe_data.get("kept", {}).values())
    except Exception as exc:
        logger.error("Failed to read %s: %s", dedupe_path, exc)
        sys.exit(1)

    if not kept_keys:
        logger.error("No kept keys found in %s.", dedupe_path)
        sys.exit(1)

    candidates = discover_pdfs()
    total_found = len(candidates)
    candidates = {k: p for k, p in candidates.items() if k in kept_keys}
    logger.info("Found %d candidate PDFs matching kept keys in corpus_dedupe.json (out of %d on disk).", len(candidates), total_found)

    if args.pdf:
        target_key = pdf_key(args.pdf)
        if target_key not in kept_keys:
            logger.error("Target PDF %s is not in kept keys of corpus_dedupe.json.", target_key)
            sys.exit(1)
        matched_path = candidates.get(target_key) or (args.pdf if os.path.isfile(args.pdf) else None)
        if not matched_path:
            logger.error("Target PDF %s not found in candidates or on disk.", args.pdf)
            sys.exit(1)
        candidates = {target_key: matched_path}
    elif args.keys:
        target_keys = set(k.strip() for k in args.keys.split(","))
        unknown = target_keys - kept_keys
        if unknown:
            logger.error("The following keys are not in kept keys: %s", unknown)
            sys.exit(1)
        candidates = {k: p for k, p in candidates.items() if k in target_keys}
    elif not args.force:
        already_ingested = manifest.load()
        candidates = {k: p for k, p in candidates.items() if k not in already_ingested}
        logger.info("%d PDFs remain to be ingested (excluding %d in manifest).", len(candidates), len(already_ingested))

    items = sorted(candidates.items())
    if args.offset:
        items = items[args.offset :]
    if args.limit:
        items = items[: args.limit]

    logger.info("Target batch to process: %d PDF(s).", len(items))
    if not items:
        logger.info("No PDFs to process. Exiting.")
        return

    if args.dry_run:
        logger.info("=== DRY RUN PREFLIGHT REPORT ===")
        logger.info("Kept keys count in corpus_dedupe.json: %d", len(kept_keys))
        logger.info("Candidate PDFs found on disk matching kept keys: %d", len(candidates))

        cache_path = os.path.join(DATA_DIR, "ocr_preflight.json")
        ocr_keys = set()
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                ocr_keys = set(cdata.get("ocr_keys", []))
            except Exception:
                pass

        if ocr_keys:
            digital_count = sum(1 for k in candidates if k not in ocr_keys)
            ocr_count = sum(1 for k in candidates if k in ocr_keys)
        else:
            digital_count = 0
            ocr_count = 0
            for k, p in sorted(candidates.items()):
                if needs_ocr(p):
                    ocr_count += 1
                else:
                    digital_count += 1

        logger.info("Digital PDFs (do_ocr=False): %d", digital_count)
        logger.info("Scanned / garbled PDFs flagged for OCR (do_ocr=True): %d", ocr_count)
        logger.info("=================================")
        return

    embeddings = get_embeddings()
    store_client = open_store()

    if not args.skip_summaries and existing_summaries:
        preseed_summaries(store_client, embeddings, existing_summaries, kept_keys=kept_keys)

    if not args.skip_abstracts and abstract_records:
        ingest_abstract_only(store_client, embeddings, abstract_records)

    total_chunks = 0
    total_captions = 0
    total_images = 0
    total_formulas = 0
    succeeded_keys = []
    t_start = time.perf_counter()

    # Process in batches of PDFs
    for batch_idx in range(0, len(items), args.batch_size):
        batch_slice = items[batch_idx : batch_idx + args.batch_size]
        batch_corpus = []
        batch_keys = []

        logger.info(
            "--- Processing batch %d/%d (%d PDFs) ---",
            (batch_idx // args.batch_size) + 1,
            (len(items) + args.batch_size - 1) // args.batch_size,
            len(batch_slice),
        )

        for i, (key, path) in enumerate(batch_slice, 1):
            abs_idx = batch_idx + i
            citation = labels.get(key) or key
            t_pdf0 = time.perf_counter()
            try:
                entries = process_pdf_v2(
                    path,
                    citation_string=citation,
                    images_dir=IMAGES_DIR,
                    extractor="docling",
                )
                pdf_elapsed = time.perf_counter() - t_pdf0
                n_text = sum(1 for e in entries if e["type"] == "text_chunk")
                n_cap = sum(1 for e in entries if e["type"] == "caption")
                n_img = sum(1 for e in entries if e["type"] == "figure_image")
                n_form = sum(1 for e in entries if e["type"] == "formula")
                total_chunks += n_text
                total_captions += n_cap
                total_images += n_img
                total_formulas += n_form
                batch_corpus.extend(entries)
                batch_keys.append(key)
                logger.info(
                    "[%d/%d] %s: %d text, %d caps, %d figs, %d forms (%.2fs)",
                    abs_idx,
                    len(items),
                    key,
                    n_text,
                    n_cap,
                    n_img,
                    n_form,
                    pdf_elapsed,
                )
            except Exception as exc:
                logger.error("Failed to process PDF %s: %s", key, exc, exc_info=True)

        if batch_corpus and not args.dry_run:
            logger.info("Committing batch of %d entries from %d PDFs to LanceDB...", len(batch_corpus), len(batch_keys))
            delete_document_rows(store_client, batch_keys, preserve_summaries_for=set(existing_summaries.keys()))
            upsert_corpus(batch_corpus)
            manifest.add_many(batch_keys)
            succeeded_keys.extend(batch_keys)

            # Periodic compaction to reclaim intermediate MVCC fragment files
            batch_num = (batch_idx // args.batch_size) + 1
            if batch_num % 10 == 0:
                try:
                    from datetime import timedelta
                    col = store_client.get_collection(COLLECTION_NAME)
                    if hasattr(col, "table") and hasattr(col.table, "optimize"):
                        col.table.optimize(cleanup_older_than=timedelta(seconds=0))
                        logger.info("Periodic table compaction complete at batch %d", batch_num)
                except Exception as exc:
                    logger.debug("Periodic table optimize notice: %s", exc)
        elif args.dry_run:
            logger.info("[DRY RUN] Would commit %d entries for %d PDFs.", len(batch_corpus), len(batch_keys))
            succeeded_keys.extend(batch_keys)

    total_time = time.perf_counter() - t_start

    # Final table compaction and cleanup of old versions
    if not args.dry_run and succeeded_keys:
        logger.info("Compacting LanceDB tables and pruning obsolete MVCC versions...")
        try:
            from datetime import timedelta
            for tname in (COLLECTION_NAME, SUMMARY_COLLECTION_NAME):
                col = store_client.get_collection(tname)
                if hasattr(col, "table") and hasattr(col.table, "optimize"):
                    col.table.optimize(cleanup_older_than=timedelta(seconds=0))
                    logger.info("✓ Final optimize complete for table %s", tname)
        except Exception as exc:
            logger.warning("Final table optimize notice: %s", exc)

    # Rebuild Tantivy index
    if not args.dry_run and succeeded_keys:
        logger.info("Refreshing LanceDB Tantivy FTS index...")
        try:
            rebuild_bm25()
        except Exception as exc:
            logger.warning("FTS index rebuild notice: %s", exc)

    logger.info("==================================================")
    logger.info("REBUILD COMPLETE")
    logger.info("PDFs processed: %d / %d", len(succeeded_keys), len(items))
    logger.info("Text chunks: %d", total_chunks)
    logger.info("Captions: %d", total_captions)
    logger.info("Figure images: %d", total_images)
    logger.info("Formulas: %d", total_formulas)
    logger.info("Total elapsed: %.1fs", total_time)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
