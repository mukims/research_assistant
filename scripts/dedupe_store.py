"""CLI script to detect and clean up store duplicates."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.shared import db
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.paper_index import PaperIndex
from research_assistant.shared.store_dedupe import apply_cleanup, plan_cleanup, scan_store

logger = logging.getLogger("dedupe_store")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Deduplicate store and clean up chunks.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions to LanceDB tables, quarantine PDFs, and update manifest/index.",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Force keep specific document in its duplicate group (repeatable).",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip cleaning up duplicate group containing specific document (repeatable).",
    )
    parser.add_argument(
        "--drop",
        action="append",
        default=[],
        help="Force drop specific document (repeatable).",
    )
    parser.add_argument(
        "--plan",
        default=os.path.join(config.DATA_DIR, "store_dedupe_plan.json"),
        help="Path to store dedupe plan JSON file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    col = db.open_store().get_collection(name=db.COLLECTION_NAME)
    idx = PaperIndex.load(config.PAPER_INDEX_PATH)

    downloaded = {}
    if os.path.exists(config.DOWNLOADED_JSON_PATH):
        try:
            with open(config.DOWNLOADED_JSON_PATH, "r", encoding="utf-8") as f:
                downloaded = json.load(f)
        except Exception as exc:
            logger.warning("Could not read downloaded manifest %s: %s", config.DOWNLOADED_JSON_PATH, exc)

    docs = scan_store(col, idx, downloaded)
    doc_map = {d.document: d for d in docs}

    groups = plan_cleanup(
        docs,
        keep=args.keep or [],
        skip=args.skip or [],
        drop=args.drop or [],
    )

    if not args.apply:
        remove_docs = sum(len(g.remove) for g in groups)
        remove_pdf_docs = sum(sum(1 for doc in g.remove if doc.endswith(".pdf")) for g in groups)
        remove_stub_docs = sum(sum(1 for doc in g.remove if not doc.endswith(".pdf")) for g in groups)
        remove_chunks = sum(
            sum(doc_map[doc].chunks for doc in g.remove if doc in doc_map)
            for g in groups
        )

        totals = {
            "groups": len(groups),
            "remove_docs": remove_docs,
            "remove_pdf_docs": remove_pdf_docs,
            "remove_stub_docs": remove_stub_docs,
            "remove_chunks": remove_chunks,
        }

        plan_data = {
            "created": datetime.now(timezone.utc).isoformat(),
            "groups": [asdict(g) for g in groups],
            "totals": totals,
            "args": {
                "keep": list(args.keep or []),
                "skip": list(args.skip or []),
                "drop": list(args.drop or []),
                "plan": args.plan,
            },
        }

        atomic_write_json(args.plan, plan_data)

        print(
            f"Totals: {totals['groups']} groups, {totals['remove_docs']} remove docs "
            f"({totals['remove_pdf_docs']} PDFs, {totals['remove_stub_docs']} stubs), "
            f"{totals['remove_chunks']} chunks"
        )
        for g in groups:
            removed_str = ", ".join(g.remove)
            print(f"{g.reason}  {g.keep}  ←  {removed_str}")

        return 0

    # --apply mode
    if not os.path.exists(args.plan):
        print(f"Plan file {args.plan} does not exist. Run dry run first.")
        return 1

    try:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
    except Exception as exc:
        print(f"Could not read plan file {args.plan}: {exc}")
        return 1

    plan_args = plan_data.get("args", {})
    if (
        list(args.keep or []) != list(plan_args.get("keep", []))
        or list(args.skip or []) != list(plan_args.get("skip", []))
        or list(args.drop or []) != list(plan_args.get("drop", []))
    ):
        print("Plan arguments differ from command line options. Rerun dry run with desired options.")
        return 1

    current_group_dicts = [asdict(g) for g in groups]
    if current_group_dicts != plan_data.get("groups"):
        print("store changed since the dry run — rerun the dry run")
        return 1

    from research_assistant.shared import ingestion, manifest
    from research_assistant.shared.dedupe_sweep import run_sweep

    with ingestion._ingest_lock():
        manifest_before_count = len(manifest.load())
        summaries_col = db.open_store().get_collection(name=config.SUMMARY_COLLECTION_NAME)
        quarantine_dir = os.path.join(config.DATA_DIR, "quarantine", "store_dedupe")
        res = apply_cleanup(groups, col, summaries_col, config.PAPER_INDEX_PATH, quarantine_dir)

        db.clear_search_cache()
        rebuild_count = manifest.rebuild_from_collection()
        expected_count = manifest_before_count - len(res["removed"])
        print(f"Rebuilt manifest count: {rebuild_count} (expected {expected_count})")

        run_sweep()

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        applied_path = os.path.join(config.DATA_DIR, f"store_dedupe_applied-{now_str}.json")
        atomic_write_json(applied_path, res)

        if res.get("errors"):
            print(f"Apply completed with {len(res['errors'])} error(s): {res['errors']}")
            return 1

        return 0


if __name__ == "__main__":
    sys.exit(main())
