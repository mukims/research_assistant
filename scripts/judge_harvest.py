"""Harvest labelling candidates for the judge evaluation set.

    PYTHONPATH=. CITATION_INDEX_VERSION=2 python scripts/judge_harvest.py abstract --papers 15 --per-paper 1 --seed 7
    PYTHONPATH=. python scripts/judge_harvest.py verification path/to/x_verification.json [more.json]

Appends to data/eval/judge/candidates.jsonl (ids never duplicated). The
operator then runs scripts/judge_label.py.
"""

import argparse
import json
import os
import random

from research_assistant.config import DATA_DIR
from research_assistant.judgement import harvest as hv
from research_assistant.shared.atomic import atomic_write

DEFAULT_OUT = os.path.join(DATA_DIR, "eval", "judge", "candidates.jsonl")


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _save(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with atomic_write(path) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def harvest_abstract(papers: int, per_paper: int, seed: int) -> list[dict]:
    from research_assistant.shared.db import load_search_resources
    from research_assistant.shared.search import hybrid_search

    try:
        from research_assistant.shared.search import expand_neighbours
    except ImportError:  # not on the hardening branch yet
        expand_neighbours = None

    collection, bm25, texts, metadatas = load_search_resources()
    abstracts = collection.get(where={"section": "abstract"}, include=["documents", "metadatas"])
    by_doc = {}
    for text, meta in zip(abstracts["documents"], abstracts["metadatas"]):
        by_doc.setdefault(meta["document"], {"text": "", "title": meta.get("citation_source", "")})
        by_doc[meta["document"]]["text"] += " " + text

    rng = random.Random(seed)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    out, n = [], 0
    for doc in docs:
        if len(out) >= papers * per_paper:
            break
        claims = hv.abstract_sentences(by_doc[doc]["text"])
        rng.shuffle(claims)
        taken = 0
        for claim in claims:
            if taken >= per_paper:
                break
            hits = hybrid_search(
                claim,
                collection,
                bm25,
                texts,
                metadatas,
                top_k=4,
                doc_filter={doc},
                exclude_types={"figure_description", "caption"},
            )
            hits = [h for h in hits if (h.get("metadata") or {}).get("section") != "abstract"]
            if not hits:
                continue
            if expand_neighbours:
                expand_neighbours(hits[:1], texts, metadatas, window=1)
            n += 1
            out.append(hv.make_candidate(f"a_{n:04d}", claim, hits[0], doc, by_doc[doc]["title"], "abstract"))
            taken += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    # --out belongs to both subcommands (a top-level option would have to
    # precede the subcommand, which reads unnaturally on the command line).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("abstract", parents=[common])
    a.add_argument("--papers", type=int, default=15)
    a.add_argument("--per-paper", type=int, default=1)
    a.add_argument("--seed", type=int, default=7)
    v = sub.add_parser("verification", parents=[common])
    v.add_argument("files", nargs="+")
    args = ap.parse_args()

    existing = _load(args.out)
    ids = {c["id"] for c in existing}
    if args.mode == "abstract":
        new = harvest_abstract(args.papers, args.per_paper, args.seed)
    else:
        new = []
        for i, path in enumerate(args.files, 1):
            with open(path, encoding="utf-8") as fh:
                new.extend(hv.candidates_from_verification(json.load(fh), prefix=f"v{i}"))
    added = [c for c in new if c["id"] not in ids]
    _save(args.out, existing + added)
    print(f"{len(added)} candidate(s) added → {args.out} ({len(existing) + len(added)} total)")


if __name__ == "__main__":
    main()
