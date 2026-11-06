"""End-of-run deduplication sweep and report generator."""

from datetime import datetime, timezone
import logging

from research_assistant import config
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.paper_index import PaperIndex
from research_assistant.shared import pipeline_status

logger = logging.getLogger(__name__)


def run_sweep(index_path: str | None = None, report_path: str | None = None) -> dict:
    """Scan the corpus paper index for duplicates, write dedupe_report.json atomically.

    Never raises (logs and returns {"error": str, "groups": []} on failure);
    deletes nothing; adds one pipeline_status event when groups is non-empty.
    """
    target_index = index_path if index_path is not None else config.PAPER_INDEX_PATH
    target_report = report_path if report_path is not None else config.DEDUPE_REPORT_PATH

    try:
        idx = PaperIndex.load(target_index)
        groups = idx.duplicate_groups()
        now_str = datetime.now(timezone.utc).isoformat()
        report = {
            "generated_at": now_str,
            "papers": len(idx),
            "groups": groups,
        }
        atomic_write_json(target_report, report)

        if groups:
            try:
                pipeline_status.add_event(f"⚠️ Dedupe sweep found {len(groups)} duplicate group(s)")
            except Exception:
                pass

        return report
    except Exception as exc:
        logger.warning("Dedupe sweep failed: %s", exc)
        return {"error": str(exc), "groups": []}
