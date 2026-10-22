#!/usr/bin/env python3
"""Build and populate isolated case corpora for technique transfer retrodiction cases."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Callable

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.agents import agent0_discoverer
from research_assistant.eval.transfer_cases import case_data_dir, load_cases
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.fetch import download_pdf
from research_assistant.shared.ingestion import ingest_pdfs, pdf_key


def guard_data_dir(case_id: str) -> int | None:
    """Ensure current CITATION_DATA_DIR matches the case data directory, exiting 4 otherwise."""
    target = case_data_dir(case_id)
    if os.path.realpath(config.DATA_DIR) != os.path.realpath(target):
        sys.stderr.write(f"set CITATION_DATA_DIR={target} to build this case\n")
        return 4
    return None


def pdf_filename(title: str) -> str:
    """A file name for a case paper: its title's start plus a hash of the whole title.

    Titles can share their first 40 characters ("The density-matrix
    renormalization group" and its "... in the age of matrix product states").
    """
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:40] + f"_{digest}.pdf"


def _normalize_title(t: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", t.lower()))


def _jaccard(t1: str, t2: str) -> float:
    s1 = set(_normalize_title(t1).split())
    s2 = set(_normalize_title(t2).split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def default_search(query: str, limit: int = 5) -> list[dict]:
    """Default live search: arXiv and OpenAlex together.

    arXiv's relevance search often returns near-titles only, which would hide
    the exact match OpenAlex has, so both are always asked.
    """
    return agent0_discoverer._search_arxiv(query, limit) + agent0_discoverer._search_openalex(query, limit)


def _preference(hit: dict) -> tuple:
    """Order equally close hits: an arXiv PDF first, then the earliest year."""
    url = hit.get("pdf_url") or ""
    try:
        year = int(hit.get("year"))
    except (TypeError, ValueError):
        year = 9999
    return ("arxiv.org" not in url, year)


def resolve_titles(titles: list[str], search: Callable = default_search) -> list[dict]:
    """Resolve target titles against preprint and open-access search indices."""
    results = []
    for title in titles:
        hits = search(title, limit=5)
        best_hit = None
        best_score = 0.0
        for h in hits:
            score = _jaccard(title, h.get("title") or "")
            if score > best_score:
                best_score, best_hit = score, h
        # Among hits as close as the best, take one with a PDF, preferring arXiv.
        with_pdf = [h for h in hits if h.get("pdf_url") and _jaccard(title, h.get("title") or "") == best_score]
        if with_pdf:
            best_hit = min(with_pdf, key=_preference)
        if best_hit is not None and best_score >= 0.8 and best_hit.get("pdf_url"):
            results.append({
                "title": title,
                "match_title": best_hit["title"],
                "year": best_hit.get("year"),
                "pdf_url": best_hit.get("pdf_url"),
                "score": best_score,
            })
        else:
            results.append({
                "title": title,
                "match_title": None,
                "year": None,
                "pdf_url": None,
                "score": best_score,
            })
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build and populate an isolated case corpus.")
    parser.add_argument("--case", type=str, help="Case identifier (e.g. nqs, kpm).")
    parser.add_argument("--resolve", action="store_true", help="Resolve paper titles to search hits and PDF links.")
    parser.add_argument("--confirm", action="store_true", help="Download resolved PDFs and ingest into case index.")
    parser.add_argument("--write-key-template", action="store_true", help="Write TEMPLATE.json for private answer keys.")

    args = parser.parse_args(argv)

    if args.write_key_template:
        out_dir = os.path.join(config.DATA_DIR, "eval", "transfer", "answer_keys")
        os.makedirs(out_dir, exist_ok=True)
        template_path = os.path.join(out_dir, "TEMPLATE.json")
        template_data = {
            "id": "template_id",
            "kind": "positive",
            "query": "query placeholder",
            "generic": "generic bottleneck placeholder",
            "cutoff_year": None,
            "corpus": "main",
            "papers": {
                "target": [],
                "source": [],
                "distractor": [],
                "hidden": [],
            },
            "exclude_documents": [],
            "expected": {
                "source_titles": [],
                "technique_terms": [],
                "mapping_terms": [],
                "critic_terms": [],
                "set_aside": None,
            },
        }
        atomic_write_json(template_path, template_data)
        sys.stdout.write(f"Wrote answer key template: {template_path}\n")
        return 0

    if not args.case:
        parser.error("--case is required unless --write-key-template is given")

    # Guard directory
    rc = guard_data_dir(args.case)
    if rc is not None and rc != 0:
        return rc

    cases = load_cases()
    case = next((c for c in cases if c.id == args.case), None)
    if case is None:
        sys.stderr.write(f"Unknown case '{args.case}'\n")
        return 1

    case_dir = case_data_dir(args.case)
    os.makedirs(case_dir, exist_ok=True)

    # Collect titles to resolve/download (hidden papers are strictly excluded)
    titles_to_process = (
        case.papers.get("target", [])
        + case.papers.get("source", [])
        + case.papers.get("distractor", [])
    )

    if args.resolve:
        resolved = resolve_titles(titles_to_process)
        out_path = os.path.join(case_dir, "resolution.json")
        atomic_write_json(out_path, resolved)
        sys.stdout.write(f"Wrote {out_path}\n")
        for r in resolved:
            status = "MATCH" if r["match_title"] else "NO MATCH"
            sys.stdout.write(f"[{status}] score={r['score']:.2f}: {r['title']} -> {r.get('match_title')}\n")
        return 0

    if args.confirm:
        resolution_file = os.path.join(case_dir, "resolution.json")
        if not os.path.exists(resolution_file):
            sys.stderr.write(f"Run --resolve first ({resolution_file} not found)\n")
            return 1
        import json
        with open(resolution_file, "r", encoding="utf-8") as fh:
            resolved_list = json.load(fh)

        pulled_dir = os.path.join(case_dir, "pulled_pdfs")
        os.makedirs(pulled_dir, exist_ok=True)

        downloaded_paths = []
        resolved_map = {}

        for item in resolved_list:
            pdf_url = item.get("pdf_url")
            orig_title = item.get("title")
            if not pdf_url:
                continue
            safe_name = pdf_filename(orig_title)
            dest = os.path.join(pulled_dir, safe_name)
            success, err = download_pdf(pdf_url, dest)
            if success:
                downloaded_paths.append(dest)
                # The id the ingest stores, so F2/F3 can find the paper's cards.
                resolved_map[orig_title] = pdf_key(dest)

        if downloaded_paths:
            ingest_pdfs(downloaded_paths, workers=1, rebuild_index=True, require_grobid=True)

        atomic_write_json(os.path.join(case_dir, "resolved.json"), resolved_map)
        sys.stdout.write(f"Ingested {len(downloaded_paths)} PDFs into {case_dir}\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
