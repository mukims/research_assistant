# scripts/index_stats.py
"""Chunk-length distribution and type counts for the active index.

    CITATION_INDEX_VERSION=2 python scripts/index_stats.py
"""
import collections
import statistics as st

import chromadb

from research_assistant.config import VECTORDB_PATH, COLLECTION_NAME, INDEX_VERSION

col = chromadb.PersistentClient(path=VECTORDB_PATH).get_collection(COLLECTION_NAME)
lens, types, docs, modes, described = [], collections.Counter(), set(), collections.Counter(), 0
off = 0
while True:
    b = col.get(include=["documents", "metadatas"], limit=5000, offset=off)
    if not b["ids"]:
        break
    for d, m in zip(b["documents"], b["metadatas"]):
        lens.append(len(d)); types[m.get("type")] += 1; docs.add(m.get("document"))
        modes[m.get("extraction", "?")] += 1
        if m.get("type") == "caption" and m.get("described"):
            described += 1
    off += 5000
lens.sort()
q = lambda p: lens[int(p * (len(lens) - 1))] if lens else 0
print(f"index v{INDEX_VERSION} ({COLLECTION_NAME}): {len(lens):,} chunks over {len(docs)} docs")
print(f"chars  p10 {q(.1)}  p25 {q(.25)}  median {q(.5)}  p75 {q(.75)}  p90 {q(.9)}  max {lens[-1] if lens else 0}")
print(f"< 200 chars: {sum(l < 200 for l in lens) / max(1, len(lens)):.0%}")
print("types:", dict(types))
print("extraction:", dict(modes), "| captions described:", described)
