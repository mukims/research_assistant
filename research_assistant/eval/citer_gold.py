"""The citer's ground truth, from a seed paper's own citations.

A published paper is a labelled dataset for citation: its author decided
which sentence cites which work. For every body sentence whose author-cited
reference is in the corpus, a record — the clean sentence, its three-sentence
window, and the documents the author cited that we hold. A matched sample of
the paper's uncited sentences is the negative class for citation need.

Author citations are a floor, not truth: a correct paper the author did not
cite scores as wrong, and an uncited sentence may merely be under-cited. The
metrics print both caveats.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from bs4 import BeautifulSoup

from research_assistant.shared.claim_text import paragraph_sentences, sentence_context
from research_assistant.shared.seed_audit import (
    _load_downloaded_manifest,
    _match_downloaded_paper,
    _paragraph_section,
    extract_seed_citation_claims,
    find_tei_for_seed,
)
from research_assistant.shared.tei_structure import section_breadcrumb

CASES_DIR = Path(__file__).parent / "cases"
MIN_WORDS = 8
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
REQUIRED = ("id", "seed", "kind", "sentence", "context", "author_documents")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _short(stem: str) -> str:
    """An id prefix: the full arXiv id when there is one, else the stem.
    Ids must be unique across every gold file load_cases() reads, so the
    prefix keeps whatever tells two seeds apart."""
    m = re.search(r"\d{4}\.\d{4,5}", stem)
    return m.group(0) if m else stem


def cited_records(claims: list[dict], seed_name: str, manifest: dict) -> list[dict]:
    """One record per sentence with at least one author citation whose
    reference PDF is in the corpus. References not in the corpus are dropped
    from the record: the citer cannot be expected to find them."""
    by_sent: dict[tuple, dict] = {}
    for c in claims:
        key = (c.get("paragraph_index", 0), c.get("sentence_index", 0))
        rec = by_sent.setdefault(key, {
            "sentence": c["claim"],
            "context": c.get("context") or "",
            "section": c.get("section") or "other",
            "section_heading": c.get("section_heading") or "",
            "paragraph_id": c.get("paragraph_id"),
            "sentence_index": c.get("sentence_index", 0),
            "author_documents": [], "author_refs": [], "roles": [],
        })
        if not c.get("resolved"):
            continue
        ref = c.get("ref") or {}
        dl = _match_downloaded_paper(ref, seed_name, manifest)
        if not (dl and dl.get("path") and os.path.exists(dl["path"])):
            continue
        doc = os.path.basename(dl["path"])
        if doc in rec["author_documents"]:
            continue
        rec["author_documents"].append(doc)
        rec["author_refs"].append({"index": ref.get("index"), "title": ref.get("title")})
        rec["roles"].append(c.get("role") or "evidential")
    return [rec for _, rec in sorted(by_sent.items()) if rec["author_documents"]]


def uncited_pool(tei_path: str) -> list[dict]:
    """Body sentences from paragraphs that cite nothing, MIN_WORDS alphabetic
    words or more; footnotes and page furniture (<note>) are skipped.
    Paragraph-level on purpose: a paragraph with no <ref type="bibr">
    has clean sentences and nothing to strip."""
    with open(tei_path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "xml")
    body = soup.find("body")
    pool = []
    for p_idx, p in enumerate(body.find_all("p") if body else []):
        if p.find_parent("note") is not None:
            continue
        if p.find_all("ref", type="bibr"):
            continue
        sents = paragraph_sentences(p)
        for s_idx, sent in enumerate(sents):
            if len(_WORD_RE.findall(sent)) < MIN_WORDS:
                continue
            pool.append({
                "sentence": sent,
                "context": sentence_context(sents, s_idx),
                "section": _paragraph_section(p),
                "section_heading": section_breadcrumb(p),
                "paragraph_id": p.get("xml:id") or p.get("id") or f"p_{p_idx}",
                "sentence_index": s_idx,
                "author_documents": [], "author_refs": [], "roles": [],
            })
    return pool


def build(seed_pdf_path: str, *, negatives_seed: int = 7) -> list[dict]:
    """Gold records for one seed: every cited sentence with an in-corpus
    reference, then as many uncited sentences, sampled with a fixed seed so
    two builds agree."""
    tei = find_tei_for_seed(seed_pdf_path)
    if not tei:
        raise FileNotFoundError(f"no GROBID TEI for {seed_pdf_path}")
    seed_name = os.path.basename(seed_pdf_path)
    stem = _stem(seed_pdf_path)
    short = _short(stem)
    manifest = _load_downloaded_manifest()
    cited = cited_records(extract_seed_citation_claims(tei), seed_name, manifest)
    pool = uncited_pool(tei)
    random.Random(negatives_seed).shuffle(pool)
    negatives = pool[: len(cited)]
    out = []
    for n, rec in enumerate(cited, 1):
        out.append({"id": f"c_{short}_{n:04d}", "seed": stem, "kind": "cited", **rec})
    for n, rec in enumerate(negatives, 1):
        out.append({"id": f"u_{short}_{n:04d}", "seed": stem, "kind": "uncited", **rec})
    return out


def write_cases(records: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_cases(paths=None) -> list[dict]:
    """Every citer_*.jsonl in CASES_DIR unless paths are given. A record
    missing a required field, or a duplicate id, raises — a silent gap here
    is a wrong score later."""
    paths = [Path(p) for p in paths] if paths else sorted(CASES_DIR.glob("citer_*.jsonl"))
    out, seen = [], set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                missing = [k for k in REQUIRED if k not in rec]
                if missing:
                    raise ValueError(f"{path.name}: record {rec.get('id', '?')!r} missing {missing}")
                if rec["id"] in seen:
                    raise ValueError(f"duplicate case id {rec['id']!r} in {path.name}")
                seen.add(rec["id"])
                out.append(rec)
    return out
