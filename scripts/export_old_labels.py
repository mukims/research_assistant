"""Export the citation label each paper has in the old (v1 and v2) indexes.

    PYTHONPATH=. python scripts/export_old_labels.py                  # writes <data>/migration_labels.json
    PYTHONPATH=. python scripts/export_old_labels.py --data-dir D --out FILE

A paper's label is stored as `citation_source` on every one of its chunks and is the `\\cite{}` key
the corpus has handed out. The migration to index v3 should give each paper the same label, but only
the old Chroma store knows it for papers that never went through downloaded.json (seeds, uploads).

The store belongs to the running app, so this never opens it with Chroma (which writes bookkeeping
rows on open). It reads chroma.sqlite3 with plain sqlite3, read-only and immutable, and refuses a
store that has a live write-ahead log (stop the app first). Nothing in the store changes.

migrate_to_v3.py reads <source>/migration_labels.json when it exists, ahead of downloaded.json.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from urllib.parse import quote

# Newest first: where both collections hold a paper, v2's label wins.
COLLECTIONS = ["physics_papers_v2", "physics_papers"]

_QUERY = """
    select c.name, d.string_value, l.string_value
    from embeddings e
    join segments s on e.segment_id = s.id
    join collections c on s.collection = c.id
    join embedding_metadata d on d.id = e.id and d.key = 'document'
    left join embedding_metadata l on l.id = e.id and l.key = 'citation_source'
    where c.name in ({marks})
    group by 1, 2, 3
"""


def read_labels(vectordb_dir: str, collections=COLLECTIONS) -> list[tuple[str, str, str]]:
    """(collection, document, citation_source) for every distinct label, without writing to the store."""
    path = os.path.join(vectordb_dir, "chroma.sqlite3")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no chroma.sqlite3 in {vectordb_dir}")
    wal = path + "-wal"
    if os.path.exists(wal) and os.path.getsize(wal) > 0:
        raise RuntimeError(f"{wal} is a live write-ahead log: the store is in use. Stop the app, then re-run.")
    uri = f"file:{quote(os.path.abspath(path))}?mode=ro&immutable=1"
    db = sqlite3.connect(uri, uri=True)
    try:
        return db.execute(_QUERY.format(marks=",".join("?" * len(collections))), list(collections)).fetchall()
    finally:
        db.close()


def merge_labels(rows, order=COLLECTIONS):
    """({document: label}, {document: {collection: label}} where the collections disagree).

    The first collection in `order` that has a non-blank label for a document wins."""
    rank = {name: i for i, name in enumerate(order)}
    per_doc: dict[str, dict[str, str]] = {}
    for coll, document, label in rows:
        if document and label and label.strip():
            per_doc.setdefault(document, {})[coll] = label
    labels, conflicts = {}, {}
    for document, by_coll in per_doc.items():
        ordered = sorted(by_coll.items(), key=lambda kv: rank.get(kv[0], len(rank)))
        labels[document] = ordered[0][1]
        if len({v for _, v in ordered}) > 1:
            conflicts[document] = dict(ordered)
    return labels, conflicts


def main(argv=None) -> int:
    from research_assistant import config

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="the data directory holding physics_vectordb/")
    parser.add_argument("--out", default=None, help="default: <data-dir>/migration_labels.json")
    args = parser.parse_args(argv)

    store = os.path.join(args.data_dir, "physics_vectordb")
    out = args.out or os.path.join(args.data_dir, "migration_labels.json")
    rows = read_labels(store)
    labels, conflicts = merge_labels(rows)
    payload = {"version": 1, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "collections": COLLECTIONS, "labels": dict(sorted(labels.items())), "conflicts": conflicts}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"{len(labels)} labels from {len(rows)} (collection, document) rows; "
          f"{len(conflicts)} documents where v1 and v2 disagree (v2 wins) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
