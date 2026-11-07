"""Store duplicate detection, keeper selection, and cleanup planning."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from research_assistant.shared.source_key import fold_title

TITLE_MIN = 15


def title_key(title: str | None) -> str | None:
    """Fold title if length >= TITLE_MIN, else None."""
    if not title:
        return None
    folded = fold_title(title)
    if not folded or len(folded) < TITLE_MIN:
        return None
    return folded


@dataclass(frozen=True)
class StoreDoc:
    document: str
    kind: Literal["pdf", "stub"] | None = None
    title: str | None = None
    surnames: frozenset[str] = frozenset()
    chunks: int = 1
    has_doi: bool = False

    def __post_init__(self):
        if self.kind is None:
            object.__setattr__(
                self, "kind", "pdf" if self.document.endswith(".pdf") else "stub"
            )


@dataclass(frozen=True)
class Group:
    title_key: str
    keep: str
    remove: tuple[str, ...]
    reason: str  # "pdf+pdf" / "pdf+stub" / "stub+stub" etc: sorted kinds joined by "+"


def plan_cleanup(
    docs: Iterable[StoreDoc],
    keep: Iterable[str] = (),
    skip: Iterable[str] = (),
    drop: Iterable[str] = (),
) -> list[Group]:
    """Group store duplicates and select keepers.

    Groups by title_key. Inside each bucket, links docs unless both are PDFs with
    disjoint surnames. Connected components of size >= 2 form duplicate groups.
    keep forces the keeper; skip drops the whole group; drop creates single-member
    groups with keep="" and reason="drop".
    """
    keep_set = set(keep)
    skip_set = set(skip)
    drop_set = set(drop)

    docs_list = list(docs)
    doc_map = {d.document: d for d in docs_list}

    # Bucket non-dropped docs by title_key
    buckets: dict[str, list[StoreDoc]] = defaultdict(list)
    for d in docs_list:
        if d.document in drop_set:
            continue
        tk = title_key(d.title)
        if tk is not None:
            buckets[tk].append(d)

    groups: list[Group] = []

    for tk, bdocs in buckets.items():
        if len(bdocs) < 2:
            continue

        # Build adjacency graph
        adj: dict[str, set[str]] = {d.document: set() for d in bdocs}
        for i in range(len(bdocs)):
            d1 = bdocs[i]
            for j in range(i + 1, len(bdocs)):
                d2 = bdocs[j]
                if d1.kind == "pdf" and d2.kind == "pdf":
                    if d1.surnames & d2.surnames:
                        adj[d1.document].add(d2.document)
                        adj[d2.document].add(d1.document)
                else:
                    # At least one is a stub
                    adj[d1.document].add(d2.document)
                    adj[d2.document].add(d1.document)

        visited = set()
        for d in bdocs:
            doc_id = d.document
            if doc_id in visited:
                continue

            # Connected component via BFS
            comp_ids = []
            queue = [doc_id]
            visited.add(doc_id)
            while queue:
                curr = queue.pop(0)
                comp_ids.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(comp_ids) < 2:
                continue

            # Check skip override
            if any(cid in skip_set for cid in comp_ids):
                continue

            comp_docs = [doc_map[cid] for cid in comp_ids]

            # Determine keeper
            # 1. Manual keep override
            manual_keeps = [cd for cd in comp_docs if cd.document in keep_set]
            if manual_keeps:
                keeper_doc = manual_keeps[0]
            else:
                pdfs = [cd for cd in comp_docs if cd.kind == "pdf"]
                if pdfs:
                    # Sort PDFs: most chunks, then has_doi, then smallest key
                    pdfs.sort(
                        key=lambda x: (
                            -x.chunks,
                            0 if x.has_doi else 1,
                            x.document,
                        )
                    )
                    keeper_doc = pdfs[0]
                else:
                    # All stubs: doi: key, then has_doi, then smallest key
                    stubs = list(comp_docs)
                    stubs.sort(
                        key=lambda x: (
                            0 if x.document.startswith("doi:") else 1,
                            0 if x.has_doi else 1,
                            x.document,
                        )
                    )
                    keeper_doc = stubs[0]

            keep_id = keeper_doc.document
            remove_ids = tuple(sorted(cid for cid in comp_ids if cid != keep_id))
            reason = "+".join(sorted(cd.kind for cd in comp_docs))
            groups.append(
                Group(
                    title_key=tk,
                    keep=keep_id,
                    remove=remove_ids,
                    reason=reason,
                )
            )

    # Process drop documents
    for drop_id in sorted(drop_set):
        matching = doc_map.get(drop_id)
        tk = title_key(matching.title) if matching else ""
        groups.append(
            Group(
                title_key=tk or "",
                keep="",
                remove=(drop_id,),
                reason="drop",
            )
        )

    groups.sort(key=lambda g: (g.title_key, g.keep))
    return groups


def ingested_titles(
    collection,
    index,
    ingested: Iterable[str],
) -> dict[str, str]:
    """Map title_key -> document for every document in `ingested`.

    PDFs: index title.
    Stubs: citation_source of collection.get(where={"type": "abstract_only"}, include=["metadatas"]) rows (paged, limit 5000).
    """
    ingested_set = set(ingested)
    mapping: dict[str, str] = {}

    # PDFs from PaperIndex
    for doc in ingested_set:
        if doc.endswith(".pdf"):
            entry = index.get(doc)
            if entry and entry.title:
                tk = title_key(entry.title)
                if tk and tk not in mapping:
                    mapping[tk] = doc

    # Stubs from collection
    if collection is not None:
        limit = 5000
        offset = 0
        while True:
            batch = collection.get(
                where={"type": "abstract_only"},
                include=["metadatas"],
                limit=limit,
                offset=offset,
            )
            metadatas = batch.get("metadatas") if batch else None
            if not metadatas:
                break
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            if not metadatas:
                break
            for meta in metadatas:
                if not meta:
                    continue
                doc = meta.get("document")
                if doc and doc in ingested_set:
                    cit = meta.get("citation_source")
                    if cit:
                        tk = title_key(cit)
                        if tk and tk not in mapping:
                            mapping[tk] = doc
            if len(metadatas) < limit:
                break
            offset += limit

    return mapping


def scan_store(collection, index, downloaded: dict) -> list[StoreDoc]:
    """Scan collection to build StoreDoc entries.

    One paged pass over collection.get(include=["metadatas"], limit=5000, offset=…):
    counts chunks per document, takes citation_source of the first row seen.
    PDFs look up title, surnames, and has_doi in index.
    Stubs use citation_source as title; has_doi is True if key starts with 'doi:' or
    matches a downloaded record with 'doi'.
    """
    from research_assistant.shared.paper_index import surname_key

    downloaded_has_doi = set()
    for k, v in (downloaded or {}).items():
        if isinstance(v, dict) and v.get("doi"):
            downloaded_has_doi.add(k)
            if v.get("key"):
                downloaded_has_doi.add(v["key"])

    chunk_counts: dict[str, int] = defaultdict(int)
    first_citations: dict[str, str] = {}

    limit = 5000
    offset = 0
    while True:
        batch = collection.get(include=["metadatas"], limit=limit, offset=offset)
        metadatas = batch.get("metadatas") if batch else None
        if not metadatas:
            break
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if not metadatas:
            break
        for meta in metadatas:
            if not meta:
                continue
            doc = meta.get("document")
            if not doc:
                continue
            chunk_counts[doc] += 1
            if doc not in first_citations:
                first_citations[doc] = meta.get("citation_source") or ""
        if len(metadatas) < limit:
            break
        offset += limit

    docs: list[StoreDoc] = []
    for doc, chunks in chunk_counts.items():
        if doc.endswith(".pdf"):
            entry = index.get(doc)
            title = (entry.title if entry and entry.title else None) or first_citations.get(doc)
            surnames = (
                frozenset(k for a in entry.first_authors if (k := surname_key(a)) is not None)
                if entry
                else frozenset()
            )
            has_doi = bool(entry.dois) if entry else False
            docs.append(
                StoreDoc(
                    document=doc,
                    kind="pdf",
                    title=title,
                    surnames=surnames,
                    chunks=chunks,
                    has_doi=has_doi,
                )
            )
        else:
            title = first_citations.get(doc)
            has_doi = doc.startswith("doi:") or (doc in downloaded_has_doi)
            docs.append(
                StoreDoc(
                    document=doc,
                    kind="stub",
                    title=title,
                    surnames=frozenset(),
                    chunks=chunks,
                    has_doi=has_doi,
                )
            )

    return docs


def apply_cleanup(
    groups: list[Group],
    papers,
    summaries,
    index_path: str,
    quarantine_dir: str,
) -> dict:
    """Apply cleanup by deleting duplicate records from store and quarantine PDFs.

    Order per removed document:
    1. papers.delete(where={"document": doc}), then summaries.delete(where={"document": doc}).
    2. Only if both deletes succeeded: manifest.discard_many([doc]). For a PDF, inside
       with paper_index.edit(index_path) as idx: read entry's path, then idx.retire(doc, keep).
       Then, if the path exists, move the file into quarantine_dir with a unique name.
    3. If a delete raises, record the error, leave the manifest and index alone for that document,
       and continue with the next one.
    Record papers.table.version and summaries.table.version before the first delete.
    """
    import os
    import shutil
    from research_assistant.shared import manifest
    from research_assistant.shared.paper_index import edit

    def _get_version(tbl) -> int:
        t = getattr(tbl, "table", tbl)
        v = getattr(t, "version", 0)
        if callable(v):
            v = v()
        return int(v) if v is not None else 0

    versions_before = {
        "papers": _get_version(papers),
        "summaries": _get_version(summaries),
    }

    removed: list[str] = []
    moved: list[list[str]] = []
    errors: list[dict[str, str]] = []

    for g in groups:
        keep = g.keep
        for doc in g.remove:
            try:
                papers.delete(where={"document": doc})
                summaries.delete(where={"document": doc})
            except Exception as exc:
                errors.append({"document": doc, "stage": "delete", "error": str(exc)})
                continue

            removed.append(doc)

            try:
                manifest.discard_many([doc])
            except Exception as exc:
                errors.append({"document": doc, "stage": "manifest", "error": str(exc)})

            if doc.endswith(".pdf"):
                pdf_path = None
                try:
                    with edit(index_path) as idx:
                        entry = idx.get(doc)
                        if entry and entry.path:
                            pdf_path = entry.path
                        idx.retire(doc, keep)
                except Exception as exc:
                    errors.append({"document": doc, "stage": "index", "error": str(exc)})

                if pdf_path and os.path.exists(pdf_path):
                    try:
                        os.makedirs(quarantine_dir, exist_ok=True)
                        base_name = os.path.basename(pdf_path)
                        dest_path = os.path.join(quarantine_dir, base_name)
                        if os.path.exists(dest_path):
                            stem, ext = os.path.splitext(base_name)
                            counter = 1
                            while os.path.exists(dest_path):
                                dest_path = os.path.join(quarantine_dir, f"{stem}_{counter}{ext}")
                                counter += 1
                        shutil.move(pdf_path, dest_path)
                        moved.append([pdf_path, dest_path])
                    except Exception as exc:
                        errors.append({"document": doc, "stage": "move", "error": str(exc)})

    return {
        "versions_before": versions_before,
        "removed": removed,
        "moved": moved,
        "errors": errors,
    }
