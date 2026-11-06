"""Backfill/build the corpus paper index from PDFs on disk and manifests."""

import argparse
from collections import defaultdict
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from research_assistant import config
from research_assistant.shared.dedupe_sweep import run_sweep
from research_assistant.shared.ingestion import pdf_key
from research_assistant.shared.paper_header import read_header
from research_assistant.shared.paper_index import (
    PaperEntry,
    PaperIndex,
    edit,
    normalise_arxiv,
    surname_key,
)
from research_assistant.shared.source_key import normalise_doi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_paper_index")


def grobid_alive(server: str = config.GROBID_SERVER, timeout: float = 10) -> bool:
    """Probe GROBID /api/isalive."""
    try:
        resp = requests.get(f"{server.rstrip('/')}/api/isalive", timeout=timeout)
        return resp.ok and "true" in resp.text.lower()
    except Exception as exc:
        logger.warning("GROBID probe failed: %s", exc)
        return False


def _find_pdfs(directories: list[str]) -> list[str]:
    """Scan directories for .pdf files, excluding quarantine or _archive."""
    found = []
    seen_paths = set()
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for root, dirs, files in os.walk(directory):
            if "quarantine" in root or "_archive" in root:
                continue
            for file in files:
                if file.lower().endswith(".pdf"):
                    full_path = os.path.abspath(os.path.join(root, file))
                    if "quarantine" in full_path or "_archive" in full_path:
                        continue
                    if full_path not in seen_paths:
                        seen_paths.add(full_path)
                        found.append(full_path)
    return sorted(found)


def _load_manifest_metadata(manifest_paths: list[str]) -> dict[str, dict]:
    """Map basename -> dict of {'dois': set, 'arxiv_ids': set}."""
    by_basename: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"dois": set(), "arxiv_ids": set()})

    for m_path in manifest_paths:
        if not m_path or not os.path.isfile(m_path):
            continue
        try:
            with open(m_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.values() if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for rec in records:
                if isinstance(rec, dict) and rec.get("path"):
                    base = os.path.basename(rec["path"])
                    if rec.get("doi"):
                        norm_d = normalise_doi(rec["doi"])
                        if norm_d:
                            by_basename[base]["dois"].add(norm_d)
                    if rec.get("arxiv_id"):
                        norm_a = normalise_arxiv(rec["arxiv_id"])
                        if norm_a:
                            by_basename[base]["arxiv_ids"].add(norm_a)
        except Exception as exc:
            logger.warning("Could not read manifest %s: %s", m_path, exc)

    return by_basename


def build_index(
    index_path: str | None = None,
    report_path: str | None = None,
    pulled_dir: str | None = None,
    raw_dir: str | None = None,
    manifest_paths: list[str] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    target_index = index_path if index_path is not None else config.PAPER_INDEX_PATH
    target_report = report_path if report_path is not None else config.DEDUPE_REPORT_PATH
    target_pulled = pulled_dir if pulled_dir is not None else config.PULLED_PDFS_DIR
    target_raw = raw_dir if raw_dir is not None else config.RAW_DIR

    all_manifests = [config.DOWNLOADED_JSON_PATH]
    if manifest_paths:
        all_manifests.extend(manifest_paths)

    all_pdfs = _find_pdfs([target_pulled, target_raw])
    idx = PaperIndex.load(target_index)

    pending_pdfs = [p for p in all_pdfs if idx.get(pdf_key(p)) is None]

    if limit is not None:
        pending_pdfs = pending_pdfs[:limit]

    logger.info(
        "Found %d PDFs total across pulled and raw. %d already indexed, %d pending.",
        len(all_pdfs),
        len(all_pdfs) - len(pending_pdfs),
        len(pending_pdfs),
    )

    if dry_run:
        print(
            f"Dry run complete. Total PDFs: {len(all_pdfs)}, already indexed: {len(all_pdfs) - len(pending_pdfs)}, pending: {len(pending_pdfs)}"
        )
        return {
            "total": len(all_pdfs),
            "already_indexed": len(all_pdfs) - len(pending_pdfs),
            "pending": len(pending_pdfs),
        }

    if not grobid_alive():
        msg = "GROBID is down — rerun to resume"
        logger.error(msg)
        return {"error": msg, "indexed": 0}

    manifest_meta = _load_manifest_metadata(all_manifests)

    indexed_count = 0
    unreadable_headers = 0
    start_time = time.time()
    batch: list[PaperEntry] = []

    for i, pdf_path in enumerate(pending_pdfs, 1):
        doc_key = pdf_key(pdf_path)
        base = os.path.basename(pdf_path)
        t0 = time.time()

        try:
            header = read_header(pdf_path, convert_if_missing=True)
        except Exception as exc:
            logger.warning("Header extraction exception on %s: %s", base, exc)
            header = None

        if not header or not header.title or not header.first_authors:
            unreadable_headers += 1

        dois = set()
        arxiv_ids = set()

        if header:
            if header.doi:
                dois.add(header.doi)
            if header.arxiv_id:
                arxiv_ids.add(header.arxiv_id)

        meta = manifest_meta.get(base, {})
        dois.update(meta.get("dois", set()))
        arxiv_ids.update(meta.get("arxiv_ids", set()))

        entry = PaperEntry(
            document=doc_key,
            path=pdf_path,
            title=header.title if header else None,
            first_authors=header.first_authors if header else (),
            dois=tuple(sorted(dois)),
            arxiv_ids=tuple(sorted(arxiv_ids)),
        )
        batch.append(entry)
        indexed_count += 1

        dt = time.time() - t0
        if i % 10 == 0 or i == len(pending_pdfs):
            elapsed = time.time() - start_time
            rate = elapsed / i if i > 0 else 0.0
            logger.info(
                "[%d/%d] Processed %s (%.2fs) | avg %.2fs/paper",
                i, len(pending_pdfs), base, dt, rate,
            )

        if len(batch) >= 25:
            if not grobid_alive():
                with edit(target_index) as fresh:
                    for e in batch:
                        fresh.add(e)
                batch.clear()
                msg = "GROBID is down — rerun to resume"
                logger.error(msg)
                return {"error": msg, "indexed": indexed_count}

            with edit(target_index) as fresh:
                for e in batch:
                    fresh.add(e)
            batch.clear()
            logger.info("Checkpoint: saved %d indexed papers to %s", indexed_count, target_index)

    if batch:
        with edit(target_index) as fresh:
            for e in batch:
                fresh.add(e)
        batch.clear()

    final_idx = PaperIndex.load(target_index)
    logger.info("Saved final index with %d total papers to %s", len(final_idx), target_index)

    sweep_res = run_sweep(index_path=target_index, report_path=target_report)
    groups = sweep_res.get("groups", [])

    print(
        f"Summary: {indexed_count} papers indexed, {unreadable_headers} headers unreadable, {len(groups)} duplicate group(s) found."
    )

    return {
        "indexed": indexed_count,
        "unreadable_headers": unreadable_headers,
        "groups": groups,
        "total_papers": len(final_idx),
    }


def main():
    parser = argparse.ArgumentParser(description="Build/backfill the corpus paper index.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and exit without modifying index.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of papers to index.")
    parser.add_argument("--extra-manifest", action="append", default=[], dest="extra_manifests", help="Extra manifest paths to scan for DOIs/arXiv IDs.")
    args = parser.parse_args()

    res = build_index(
        dry_run=args.dry_run,
        limit=args.limit,
        manifest_paths=args.extra_manifests,
    )
    if isinstance(res, dict) and "error" in res:
        sys.exit(1)


if __name__ == "__main__":
    main()
