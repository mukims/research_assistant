"""
Unified LanceDB storage and search adapter.

Replaces ChromaDB + monolithic BM25Okapi pickle with an embedded LanceDB
columnar store supporting:
- Deterministic content-addressed chunk IDs (no global sequential chunk_N)
- Fast zero-copy Arrow memory mapping
- Native Tantivy full-text search (FTS) + vector search
- Reciprocal Rank Fusion (RRF)
- Direct SQL filter pushdown (doc_filter, exclude_docs, exclude_types)
- Direct sequence-based neighbour lookup without loading the corpus into RAM
"""

import hashlib
import json
import os
from typing import Any, Optional

import numpy as np

try:
    import lancedb
    import lancedb.index
    import pyarrow as pa
except ImportError:
    lancedb = None
    pa = None

from research_assistant.config import (
    DEFAULT_TOP_K,
    EXCLUDED_FROM_EVIDENCE,
    LANCE_SUMMARY_TABLE_NAME,
    LANCE_TABLE_NAME,
    LANCEDB_PATH,
    QUARANTINE_TABLE_NAME,
    RRF_K,
)
from research_assistant.shared.log import get_logger

logger = get_logger("lance_store")


def compute_chunk_id(document: str, stored_form: str, seq: int, text: str) -> str:
    """Generate a deterministic content-addressed chunk identifier."""
    raw = f"{document}\x00{stored_form}\x00{seq}\x00{text}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"chk_{digest[:24]}"


def compute_summary_id(document: str) -> str:
    """Deterministic ID for document summary."""
    return f"sum::{document}"


def get_papers_schema(dim: int = 768) -> "pa.Schema":
    """Schema for the main research papers chunk table."""
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("document", pa.string()),
        pa.field("citation_source", pa.string()),
        pa.field("stored_form", pa.string()),
        pa.field("section", pa.string()),
        pa.field("seq", pa.int32()),
        pa.field("page", pa.int32()),
        pa.field("page_first", pa.int32()),
        pa.field("page_last", pa.int32()),
        pa.field("tier", pa.string()),
        pa.field("text", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("metadata_json", pa.string()),
    ])


def get_summaries_schema(dim: int = 768) -> "pa.Schema":
    """Schema for document-level summaries (stage-1 retrieval)."""
    return pa.schema([
        pa.field("summary_id", pa.string()),
        pa.field("document", pa.string()),
        pa.field("citation_source", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("metadata_json", pa.string()),
    ])


def get_quarantine_schema() -> "pa.Schema":
    """Schema for dead-letter quarantined chunks (e.g. failed embeddings)."""
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("document", pa.string()),
        pa.field("stored_form", pa.string()),
        pa.field("seq", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("error_reason", pa.string()),
        pa.field("created_at", pa.string()),
    ])


def get_lance_connection(path: Optional[str] = None):
    """Open or create a connection to the LanceDB database."""
    if lancedb is None:
        raise RuntimeError("lancedb is not installed. Please install lancedb and pyarrow.")
    target_path = path or LANCEDB_PATH
    os.makedirs(target_path, exist_ok=True)
    return lancedb.connect(target_path)


def list_table_names(conn) -> list[str]:
    """Return a list of table names across LanceDB API versions."""
    try:
        res = conn.list_tables()
        if hasattr(res, "tables"):
            return list(res.tables)
        if hasattr(conn, "table_names"):
            return list(conn.table_names())
        return [str(x) for x in res]
    except Exception:
        return []


class LanceTableAdapter:
    """
    Adapter wrapping a LanceDB table to provide both modern native search
    and a Chroma-compatible collection facade.
    """

    def __init__(self, table, is_summary: bool = False):
        self.table = table
        self.name = table.name
        self.is_lancedb = True
        self.is_summary = is_summary
        self.metadata = {"hnsw:space": "cosine", "backend": "lancedb"}

    def count(self) -> int:
        try:
            return len(self.table)
        except Exception:
            return 0

    def __len__(self) -> int:
        return self.count()

    def create_fts_index(self, column: str = "text"):
        """Create or replace Tantivy FTS index on the target text column."""
        try:
            self.table.create_index(column, config=lancedb.index.FTS(), replace=True)
            logger.info("Created Tantivy FTS index on table %s (column=%s)", self.name, column)
        except Exception as e:
            logger.warning("FTS index creation warning for %s: %s", self.name, e)

    def _build_where_sql(
        self,
        doc_filter: Optional[set] = None,
        exclude_types: Optional[set] = None,
        exclude_docs: Optional[set] = None,
    ) -> Optional[str]:
        clauses = []
        if doc_filter:
            quoted = ", ".join(f"'{d.replace(chr(39), chr(39)*2)}'" for d in sorted(doc_filter))
            clauses.append(f"document IN ({quoted})")
        if exclude_docs:
            quoted = ", ".join(f"'{d.replace(chr(39), chr(39)*2)}'" for d in sorted(exclude_docs))
            clauses.append(f"document NOT IN ({quoted})")
        if exclude_types:
            quoted = ", ".join(f"'{t.replace(chr(39), chr(39)*2)}'" for t in sorted(exclude_types))
            clauses.append(f"stored_form NOT IN ({quoted})")
        return " AND ".join(clauses) if clauses else None

    def search_hybrid(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = RRF_K,
        doc_filter: Optional[set] = None,
        exclude_types: Optional[set] = None,
        exclude_docs: Optional[set] = None,
    ) -> list[dict]:
        """
        Executes hybrid search combining dense vector search and Tantivy FTS,
        fused via Reciprocal Rank Fusion (RRF).
        """
        where_sql = self._build_where_sql(doc_filter, exclude_types, exclude_docs)
        k_cand = max(60, top_k * 10) if (doc_filter or exclude_docs) else max(15, top_k * 3)

        # 1. Vector Search
        try:
            vec_q = self.table.search(query_vector, query_type="vector")
            if where_sql:
                vec_q = vec_q.where(where_sql)
            vec_hits = vec_q.limit(k_cand).to_list()
        except Exception as e:
            logger.warning("LanceDB vector search failed: %s", e)
            vec_hits = []

        # 2. Tantivy Full-Text Search
        try:
            fts_q = self.table.search(query, query_type="fts")
            if where_sql:
                fts_q = fts_q.where(where_sql)
            fts_hits = fts_q.limit(k_cand).to_list()
        except Exception as e:
            logger.warning("LanceDB FTS search failed: %s", e)
            fts_hits = []

        # 3. Reciprocal Rank Fusion
        fused_scores = {}
        lookup = {}

        for rank, hit in enumerate(fts_hits):
            cid = hit.get("chunk_id")
            if not cid:
                continue
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rank + 1 + rrf_k))
            lookup[cid] = hit

        for rank, hit in enumerate(vec_hits):
            cid = hit.get("chunk_id")
            if not cid:
                continue
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rank + 1 + rrf_k))
            lookup[cid] = hit

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, rrf_score in ranked:
            h = lookup[cid]
            meta = {
                "document": h.get("document", ""),
                "citation_source": h.get("citation_source", ""),
                "stored_form": h.get("stored_form", "text_chunk"),
                "section": h.get("section", ""),
                "seq": h.get("seq", 0),
                "page": h.get("page", 1),
                "page_first": h.get("page_first", h.get("page", 1)),
                "page_last": h.get("page_last", h.get("page", 1)),
                "type": h.get("stored_form", "text_chunk"),
                "tier": h.get("tier", "full_text"),
                "content_hash": h.get("content_hash", ""),
            }
            if h.get("metadata_json"):
                try:
                    extra = json.loads(h["metadata_json"])
                    if isinstance(extra, dict):
                        meta.update(extra)
                except Exception:
                    pass

            results.append({
                "chunk_id": cid,
                "chunk_index": int(h.get("seq", 0)),
                "text": h.get("text", ""),
                "metadata": meta,
                "rrf_score": round(rrf_score, 5),
            })

        return results

    def get_neighbours(
        self,
        document: str,
        stored_form: str,
        seq: int,
        window: int = 1,
        exclude_types: Optional[set] = None,
    ) -> tuple[str, str]:
        """
        Fetch adjacent chunks belonging to the same document and stored_form.
        Returns (context_before, context_after).
        """
        if window <= 0:
            return "", ""

        conds = [
            f"document = '{document.replace(chr(39), chr(39)*2)}'",
            f"stored_form = '{stored_form.replace(chr(39), chr(39)*2)}'",
            f"seq >= {seq - window}",
            f"seq <= {seq + window}",
        ]
        if exclude_types:
            quoted = ", ".join(f"'{t.replace(chr(39), chr(39)*2)}'" for t in sorted(exclude_types))
            conds.append(f"stored_form NOT IN ({quoted})")

        where_sql = " AND ".join(conds)
        try:
            rows = self.table.search().where(where_sql).to_list()
        except Exception as e:
            logger.warning("Neighbour fetch failed: %s", e)
            return "", ""

        rows.sort(key=lambda r: r.get("seq", 0))
        before = [r.get("text", "") for r in rows if r.get("seq", 0) < seq]
        after = [r.get("text", "") for r in rows if r.get("seq", 0) > seq]
        return "\n".join(before), "\n".join(after)

    # ─── ChromaDB Compatibility Facade ────────────────────────────────────────

    def add(
        self,
        embeddings: Optional[list] = None,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs,
    ):
        """Chroma-compatible add interface for LanceDB."""
        ids = ids or kwargs.get("ids", [])
        documents = documents or kwargs.get("documents", [])
        embeddings = embeddings or kwargs.get("embeddings", [])
        metadatas = metadatas or kwargs.get("metadatas", [])

        if not ids:
            return

        records = []
        for i, cid in enumerate(ids):
            doc_text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            vec = embeddings[i] if i < len(embeddings) and embeddings[i] is not None else []
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            elif hasattr(vec, "__iter__"):
                vec = [float(x) for x in vec]
            else:
                vec = []

            extra_meta = {
                k: v for k, v in meta.items()
                if k not in ("document", "citation_source", "type", "stored_form", "section", "seq", "page", "page_first", "page_last", "tier")
            }
            extra_json = json.dumps(extra_meta) if extra_meta else ""

            if self.is_summary:
                rec = {
                    "summary_id": cid,
                    "document": str(meta.get("document", "")),
                    "citation_source": str(meta.get("citation_source", "")),
                    "summary": str(doc_text),
                    "vector": vec,
                    "metadata_json": extra_json,
                }
            else:
                stored_form = str(meta.get("stored_form") or meta.get("type") or "text_chunk")
                seq = int(meta.get("seq", 0))
                content_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
                page = int(meta.get("page", 1))
                rec = {
                    "chunk_id": cid,
                    "document": str(meta.get("document", "")),
                    "citation_source": str(meta.get("citation_source", "")),
                    "stored_form": stored_form,
                    "section": str(meta.get("section", "")),
                    "seq": seq,
                    "page": page,
                    "page_first": int(meta.get("page_first", page)),
                    "page_last": int(meta.get("page_last", page)),
                    "tier": str(meta.get("tier") or meta.get("evidence_tier") or "full_text"),
                    "text": str(doc_text),
                    "content_hash": content_hash,
                    "vector": vec,
                    "metadata_json": extra_json,
                }
            records.append(rec)

        if records:
            self.table.add(records)

    def upsert(
        self,
        embeddings: Optional[list] = None,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs,
    ):
        """Chroma-compatible upsert interface for LanceDB."""
        ids = ids or kwargs.get("ids", [])
        if not ids:
            return
        id_field = "summary_id" if self.is_summary else "chunk_id"
        try:
            quoted = ", ".join(f"'{c.replace(chr(39), chr(39)*2)}'" for c in ids)
            self.table.delete(f"{id_field} IN ({quoted})")
        except Exception:
            pass
        self.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids, **kwargs)

    def delete(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        **kwargs,
    ):
        """Chroma-compatible delete interface for LanceDB."""
        id_field = "summary_id" if self.is_summary else "chunk_id"
        conds = []
        if ids:
            quoted = ", ".join(f"'{c.replace(chr(39), chr(39)*2)}'" for c in ids)
            conds.append(f"{id_field} IN ({quoted})")
        if where:
            w_sql = self._parse_chroma_where(where)
            if w_sql:
                conds.append(w_sql)
        if not conds:
            raise ValueError("delete() requires either ids or where clause")
        where_sql = " AND ".join(conds)
        self.table.delete(where_sql)

    def rebuild_fts_index(self, column: str = "text"):
        """Rebuild FTS index on target text/summary column."""
        target_col = "summary" if self.is_summary else column
        self.create_fts_index(target_col)

    def query(
        self,
        query_embeddings: Optional[list] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        include: Optional[list] = None,
    ) -> dict:
        """Chroma-compatible query interface."""
        if not query_embeddings or not query_embeddings[0]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        vec = query_embeddings[0]
        q = self.table.search(vec, query_type="vector")

        if where:
            where_sql = self._parse_chroma_where(where)
            if where_sql:
                q = q.where(where_sql)

        hits = q.limit(n_results).to_list()
        ids = [h.get("chunk_id") or h.get("summary_id", "") for h in hits]
        texts = [h.get("text") or h.get("summary", "") for h in hits]
        metadatas = []
        distances = [float(h.get("_distance", 0.0)) for h in hits]

        for h in hits:
            meta = {
                "document": h.get("document", ""),
                "citation_source": h.get("citation_source", ""),
                "stored_form": h.get("stored_form", ""),
                "section": h.get("section", ""),
                "seq": h.get("seq", 0),
                "page": h.get("page", 1),
                "type": h.get("stored_form", ""),
            }
            if h.get("metadata_json"):
                try:
                    meta.update(json.loads(h["metadata_json"]))
                except Exception:
                    pass
            metadatas.append(meta)

        return {
            "ids": [ids],
            "documents": [texts],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list] = None,
    ) -> dict:
        """Chroma-compatible get interface."""
        conds = []
        id_col = "summary_id" if self.is_summary else "chunk_id"
        if ids:
            quoted = ", ".join(f"'{cid.replace(chr(39), chr(39)*2)}'" for cid in ids)
            conds.append(f"{id_col} IN ({quoted})")
        if where:
            w_sql = self._parse_chroma_where(where)
            if w_sql:
                conds.append(w_sql)

        q = self.table.search()
        if conds:
            q = q.where(" AND ".join(conds))
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)

        try:
            hits = q.to_list()
        except Exception:
            hits = []

        ret_ids = [h.get(id_col, "") for h in hits]
        ret_docs = [h.get("text") or h.get("summary", "") for h in hits]
        ret_metas = []
        for h in hits:
            m = {
                "document": h.get("document", ""),
                "citation_source": h.get("citation_source", ""),
                "type": h.get("stored_form", ""),
                "section": h.get("section", ""),
                "seq": h.get("seq", 0),
            }
            if h.get("metadata_json"):
                try:
                    m.update(json.loads(h["metadata_json"]))
                except Exception:
                    pass
            ret_metas.append(m)

        return {"ids": ret_ids, "documents": ret_docs, "metadatas": ret_metas}

    def _parse_chroma_where(self, where: dict) -> Optional[str]:
        if not where:
            return None
        parts = []
        if "$and" in where:
            sub = [self._parse_chroma_where(w) for w in where["$and"]]
            return " AND ".join(s for s in sub if s)
        for k, v in where.items():
            if isinstance(v, dict):
                if "$in" in v:
                    quoted = ", ".join(f"'{x.replace(chr(39), chr(39)*2)}'" for x in v["$in"])
                    col = "stored_form" if k == "type" else k
                    parts.append(f"{col} IN ({quoted})")
                elif "$nin" in v:
                    quoted = ", ".join(f"'{x.replace(chr(39), chr(39)*2)}'" for x in v["$nin"])
                    col = "stored_form" if k == "type" else k
                    parts.append(f"{col} NOT IN ({quoted})")
            else:
                col = "stored_form" if k == "type" else k
                parts.append(f"{col} = '{str(v).replace(chr(39), chr(39)*2)}'")
        return " AND ".join(parts) if parts else None


class LanceStoreClient:
    """Chroma-like client interface for LanceDB."""

    def __init__(self, path: Optional[str] = None):
        self.db = get_lance_connection(path)

    def heartbeat(self):
        return 1

    def list_tables(self) -> list[str]:
        return list_table_names(self.db)

    def list_collections(self) -> list[str]:
        return self.list_tables()

    def get_or_create_collection(self, name: str, metadata: Optional[dict] = None) -> LanceTableAdapter:
        dim = 768
        name_lower = name.lower()
        is_summary = "summar" in name_lower
        tables = self.list_tables()
        if name in tables:
            return LanceTableAdapter(self.db.open_table(name), is_summary=is_summary)

        if is_summary:
            table = self.db.create_table(name, schema=get_summaries_schema(dim))
        elif "quarantine" in name_lower:
            table = self.db.create_table(name, schema=get_quarantine_schema())
        else:
            table = self.db.create_table(name, schema=get_papers_schema(dim))
        return LanceTableAdapter(table, is_summary=is_summary)

    def get_collection(self, name: str) -> LanceTableAdapter:
        tables = self.list_tables()
        if name not in tables:
            raise KeyError(f"Table {name} not found in LanceDB (existing: {tables})")
        return LanceTableAdapter(self.db.open_table(name), is_summary=("summar" in name.lower()))
