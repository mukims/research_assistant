"""
Agent 3 — ingestor.

Feeds what Agent 2 downloaded into the shared ingestion path
(research_assistant.shared.ingestion.ingest_pdfs).
"""

import os
import json
import argparse
import multiprocessing

from research_assistant.config import DOWNLOADED_JSON_PATH, PULLED_PDFS_DIR
from research_assistant.schemas import DownloadedPaper, SchemaError
from research_assistant.shared.log import get_logger
from research_assistant.shared.ingestion import ingest_pdfs
from research_assistant.shared import pipeline_status

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
        path = paper.path
        label = paper.title or paper.raw_reference or paper.key
        pdfs[path] = label
        if not os.path.exists(path):
            candidate = os.path.join(PULLED_PDFS_DIR, os.path.basename(path))
            if os.path.exists(candidate):
                pdfs[candidate] = label
    return pdfs


def resolve_labels_for_pdfs(
    pdf_paths: list[str],
    manifest_path: str = DOWNLOADED_JSON_PATH,
) -> dict[str, str]:
    """Turn a list of PDF paths into a {pdf_path: citation_label} mapping.

    Prefers the parsed title or reference from the download manifest,
    matching on exact path or basename. Falls back to filename stem.
    """
    downloaded = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                downloaded = json.load(fh)
        except Exception as exc:
            logger.warning("Could not read manifest %s: %s", manifest_path, exc)

    by_path = _pdfs_from_manifest(downloaded) if downloaded else {}
    by_base = {os.path.basename(p): label for p, label in by_path.items()}
    return {
        path: (
            by_path.get(path)
            or by_base.get(os.path.basename(path))
            or os.path.splitext(os.path.basename(path))[0]
        )
        for path in pdf_paths
    }


def run_ingestor(workers: int = 1, force: bool = False):
    """Ingest every PDF Agent 2 successfully downloaded.

    Args:
        workers: Parallel worker processes for the parsing stage.
        force:   Re-process PDFs even if they are already recorded as ingested.
    """
    standalone = not pipeline_status.get_status().get("active")
    with pipeline_status.track_stage(
        "ingest_refs",
        "Ingesting and summarizing papers",
        current_step=5,
        total_steps=5,
        detail="Ingesting referenced papers into corpus",
        mark_idle_on_exit=standalone,
    ):
        if not os.path.exists(DOWNLOADED_JSON_PATH):
            logger.info(
                "No download manifest at %s — run Agent 2 (python agent2_fetcher.py) "
                "first to fetch papers.", DOWNLOADED_JSON_PATH,
            )
            pipeline_status.update_progress(detail="No download manifest found")
            return

        with open(DOWNLOADED_JSON_PATH, "r") as f:
            downloaded = json.load(f)

        if not downloaded:
            logger.info("Download manifest is empty — nothing to ingest.")
            pipeline_status.update_progress(detail="Download manifest is empty")
            return

        if force:
            logger.info("--force: re-processing all PDFs regardless of ingestion status.")

        pdfs = _pdfs_from_manifest(downloaded)
        if not pdfs:
            logger.info("No usable PDF paths in the download manifest — nothing to ingest.")
            pipeline_status.update_progress(detail="No usable PDF paths in download manifest")
            return

        result = ingest_pdfs(pdfs, workers=workers, skip_ingested=not force)

        logger.info(
            "Done. Processed %d, skipped %d, inserted %d chunk(s), %d unreadable.",
            result["processed"], result["skipped"], result["inserted"], len(result["failed"]),
        )
        summ = result.get("summaries") or {}
        if summ.get("failed"):
            logger.warning(
                "%d of %d paper summaries failed (%s); those papers will not be found by topic search.",
                summ["failed"], summ.get("attempted", summ["failed"]), summ.get("error"),
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
