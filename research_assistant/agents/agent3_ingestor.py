"""
Agent 3 — ingestor.

Feeds what Agent 2 downloaded into the shared ingestion path
(research_assistant.shared.ingestion.ingest_pdfs).
"""

import os
import json
import argparse
import multiprocessing

from research_assistant.config import DOWNLOADED_JSON_PATH
from research_assistant.schemas import DownloadedPaper, SchemaError
from research_assistant.shared.log import get_logger
from research_assistant.shared.ingestion import ingest_pdfs

logger = get_logger("agent3")


def _pdfs_from_manifest(downloaded: dict) -> dict[str, str]:
    """Turn Agent 2's manifest into the {pdf_path: citation_label} ingest wants.

    The citation label becomes citation_source on every chunk, and from there the
    key in a cited draft's mapping — so prefer the parsed title over the raw
    reference string, and never fall back to the opaque source key.
    """
    pdfs = {}
    for key, record in downloaded.items():
        try:
            paper = DownloadedPaper.from_dict(record)
        except SchemaError as exc:
            logger.warning("Skipping malformed manifest entry %s: %s", key, exc)
            continue
        pdfs[paper.path] = paper.title or paper.raw_reference or paper.key
    return pdfs


def run_ingestor(workers=1, force=False):
    """Ingest every PDF Agent 2 successfully downloaded.

    Args:
        workers: Parallel worker processes for the parsing stage.
        force:   Re-process PDFs even if they are already recorded as ingested.
    """
    if not os.path.exists(DOWNLOADED_JSON_PATH):
        logger.info(
            "No download manifest at %s — run Agent 2 (python agent2_fetcher.py) "
            "first to fetch papers.", DOWNLOADED_JSON_PATH,
        )
        return

    with open(DOWNLOADED_JSON_PATH, "r") as f:
        downloaded = json.load(f)

    if not downloaded:
        logger.info("Download manifest is empty — nothing to ingest.")
        return

    if force:
        logger.info("--force: re-processing all PDFs regardless of ingestion status.")

    pdfs = _pdfs_from_manifest(downloaded)
    if not pdfs:
        logger.info("No usable PDF paths in the download manifest — nothing to ingest.")
        return

    result = ingest_pdfs(pdfs, workers=workers, skip_ingested=not force)

    logger.info(
        "Done. Processed %d, skipped %d, inserted %d chunk(s), %d unreadable.",
        result["processed"], result["skipped"], result["inserted"], len(result["failed"]),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 3 — Ingestor with parallelization")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Use 1 to disable parallelization, 2+ for multiprocessing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all PDFs, even if already ingested.",
    )
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    run_ingestor(workers=args.workers, force=args.force)
