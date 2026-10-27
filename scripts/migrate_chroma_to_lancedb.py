#!/usr/bin/env python3
"""
Migrate existing ChromaDB vectors, text chunks, and summaries to LanceDB.

Reads existing embeddings, documents, and metadata directly from ChromaDB
and inserts them into LanceDB with:
- Zero re-embedding calls (preserves exact existing vectors)
- Computed content-addressed chunk IDs and hashes
- Tantivy FTS full-text index creation
- Document summaries migration
"""

import argparse
import hashlib
import json
import os
import sys
import time

# Ensure package root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import chromadb
from research_assistant.config import (
    COLLECTION_NAME,
    LANCE_SUMMARY_TABLE_NAME,
    LANCE_TABLE_NAME,
    LANCEDB_PATH,
    SUMMARY_COLLECTION_NAME,
    VECTORDB_PATH,
)
from research_assistant.shared.lance_store import (
    LanceStoreClient,
    LanceTableAdapter,
    compute_chunk_id,
    compute_summary_id,
    get_lance_connection,
    get_papers_schema,
    get_summaries_schema,
    list_table_names,
)
from research_assistant.shared.log import get_logger

logger = get_logger("migrate_chroma_to_lance")


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB corpus to LanceDB")
    parser.add_argument("--dry-run", action="store_true", help="Only count records and check schemas")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for paging from ChromaDB")
    parser.add_argument("--force", action="store_true", help="Overwrite existing LanceDB tables")
    return parser.parse_args()


def migrate():
    args = parse_args()
    logger.info("ChromaDB path: %s", VECTORDB_PATH)
    logger.info("LanceDB path:  %s", LANCEDB_PATH)

    if not os.path.exists(VECTORDB_PATH):
        logger.error("ChromaDB path '%s' does not exist.", VECTORDB_PATH)
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)

    # 1. Inspect main chunks collection
    try:
        chroma_papers = chroma_client.get_collection(COLLECTION_NAME)
        total_chunks = chroma_papers.count()
        logger.info("Chroma collection '%s' has %d chunks.", COLLECTION_NAME, total_chunks)
    except Exception as e:
        logger.error("Could not find collection '%s' in ChromaDB: %s", COLLECTION_NAME, e)
        total_chunks = 0
        chroma_papers = None

    # 2. Inspect summaries collection
    try:
        chroma_summaries = chroma_client.get_collection(SUMMARY_COLLECTION_NAME)
        total_summaries = chroma_summaries.count()
        logger.info("Chroma collection '%s' has %d summaries.", SUMMARY_COLLECTION_NAME, total_summaries)
    except Exception as e:
        logger.warning("Could not find summary collection '%s': %s", SUMMARY_COLLECTION_NAME, e)
        total_summaries = 0
        chroma_summaries = None

    if args.dry_run:
        logger.info("DRY RUN complete. Total records to migrate: %d chunks, %d summaries.", total_chunks, total_summaries)
        return

    # Check embedding dimension
    sample = chroma_papers.get(limit=1, include=["embeddings"]) if chroma_papers and total_chunks > 0 else None
    sample_embs = sample.get("embeddings") if sample else None
    if sample_embs is not None and len(sample_embs) > 0:
        dim = len(sample_embs[0])
    else:
        dim = 768
    logger.info("Detected embedding dimension: %d", dim)

    lance_client = LanceStoreClient(LANCEDB_PATH)
    conn = get_lance_connection(LANCEDB_PATH)

    existing_tables = list_table_names(conn)
    if LANCE_TABLE_NAME in existing_tables:
        if args.force:
            logger.info("Dropping existing table '%s'...", LANCE_TABLE_NAME)
            conn.drop_table(LANCE_TABLE_NAME)
            existing_tables.remove(LANCE_TABLE_NAME)
        else:
            logger.info("Table '%s' already exists in LanceDB. Skipping chunks or use --force.", LANCE_TABLE_NAME)
            chroma_papers = None

    # Migrate Main Papers Table
    if chroma_papers and total_chunks > 0:
        logger.info("Creating LanceDB table '%s' with dimension %d...", LANCE_TABLE_NAME, dim)
        papers_table = conn.create_table(LANCE_TABLE_NAME, schema=get_papers_schema(dim))
        adapter = LanceTableAdapter(papers_table)

        offset = 0
        migrated_chunks = 0
        t0 = time.time()

        while offset < total_chunks:
            batch = chroma_papers.get(
                limit=args.batch_size,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            docs = batch.get("documents")
            if docs is None or len(docs) == 0:
                break

            metas = batch.get("metadatas")
            if metas is None:
                metas = []
            embs = batch.get("embeddings")
            if embs is None:
                embs = []
            cids = batch.get("ids")
            if cids is None:
                cids = []

            records = []
            for doc_text, meta, emb, raw_cid in zip(docs, metas, embs, cids):
                meta = meta or {}
                doc_name = meta.get("document", "")
                stored_form = meta.get("type", "text_chunk")
                seq = int(meta.get("seq", 0))

                content_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
                cid = compute_chunk_id(doc_name, stored_form, seq, doc_text)

                # Skip any malformed zero-vectors
                vec = list(emb) if hasattr(emb, "__iter__") else [0.0] * dim
                if len(vec) != dim:
                    continue

                records.append({
                    "chunk_id": cid,
                    "document": doc_name,
                    "citation_source": str(meta.get("citation_source", "")),
                    "stored_form": str(stored_form),
                    "section": str(meta.get("section", "")),
                    "seq": seq,
                    "page": int(meta.get("page", 1)),
                    "page_first": int(meta.get("page_first", meta.get("page", 1))),
                    "page_last": int(meta.get("page_last", meta.get("page", 1))),
                    "tier": str(meta.get("tier", "full_text")),
                    "text": doc_text,
                    "content_hash": content_hash,
                    "vector": vec,
                    "metadata_json": json.dumps(meta),
                })

            if records:
                papers_table.add(records)
                migrated_chunks += len(records)

            offset += len(docs)
            logger.info("Migrated %d / %d chunks (%.1f%%)...", migrated_chunks, total_chunks, (migrated_chunks/total_chunks)*100)

        elapsed = time.time() - t0
        logger.info("Finished migrating %d chunks in %.1fs. Creating Tantivy FTS index...", migrated_chunks, elapsed)
        adapter.create_fts_index("text")
        logger.info("FTS index built successfully.")

    # Migrate Summaries Table
    if chroma_summaries and total_summaries > 0:
        if LANCE_SUMMARY_TABLE_NAME in existing_tables:
            if args.force:
                conn.drop_table(LANCE_SUMMARY_TABLE_NAME)
                existing_tables.remove(LANCE_SUMMARY_TABLE_NAME)
            else:
                logger.info("Summary table '%s' already exists in LanceDB. Skipping.", LANCE_SUMMARY_TABLE_NAME)
                chroma_summaries = None

        if chroma_summaries:
            logger.info("Migrating %d summaries to '%s'...", total_summaries, LANCE_SUMMARY_TABLE_NAME)
            sum_table = conn.create_table(LANCE_SUMMARY_TABLE_NAME, schema=get_summaries_schema(dim))
            s_batch = chroma_summaries.get(include=["documents", "metadatas", "embeddings"])
            s_docs = s_batch.get("documents")
            if s_docs is None:
                s_docs = []
            s_metas = s_batch.get("metadatas")
            if s_metas is None:
                s_metas = []
            s_embs = s_batch.get("embeddings")
            if s_embs is None:
                s_embs = []

            s_records = []
            for s_text, s_meta, s_emb in zip(s_docs, s_metas, s_embs):
                s_meta = s_meta or {}
                s_doc = s_meta.get("document", "")
                vec = list(s_emb) if hasattr(s_emb, "__iter__") else [0.0] * dim
                s_records.append({
                    "summary_id": compute_summary_id(s_doc),
                    "document": s_doc,
                    "citation_source": str(s_meta.get("citation_source", "")),
                    "summary": s_text,
                    "vector": vec,
                    "metadata_json": json.dumps(s_meta),
                })
            if s_records:
                sum_table.add(s_records)
            logger.info("Migrated %d summaries successfully.", len(s_records))

    logger.info("Migration to LanceDB complete!")


if __name__ == "__main__":
    migrate()
