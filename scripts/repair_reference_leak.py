"""Repair stored papers whose text chunks hold reference lists.

    python scripts/repair_reference_leak.py            # dry run: write the plan, change nothing
    python scripts/repair_reference_leak.py --apply    # re-scan, check it matches the plan, apply

Each candidate is re-chunked from its Docling cache only. --apply needs Marvin
stopped and a backup of data/lancedb: a paper whose old chunks were deleted but
whose new ones failed to insert is listed under "errors" in the applied report
as deleted_not_reinserted, to be restored from that backup.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.shared import db
from research_assistant.shared import paper_index
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.reference_repair import apply_reference_repair, scan_reference_leaks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Repair stored papers whose chunks hold reference lists.")
    parser.add_argument("--apply", action="store_true",
                        help="Replace the planned papers' text chunks (Marvin stopped, data/lancedb backed up).")
    parser.add_argument("--plan", default=os.path.join(config.DATA_DIR, "reference_repair_plan.json"),
                        help="Path of the dry-run plan JSON.")
    return parser.parse_args(argv)


def _totals(repairs) -> dict:
    return {
        "papers": len(repairs),
        "stored_chunks": sum(r.stored_chunks for r in repairs),
        "fresh_chunks": sum(r.fresh_chunks for r in repairs),
        "heavy_removed": sum(r.stored_heavy - r.fresh_heavy for r in repairs),
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    col = db.open_store().get_collection(name=db.COLLECTION_NAME)
    index = paper_index.PaperIndex.load(config.PAPER_INDEX_PATH)

    if args.apply:
        return _apply(args, col, index)

    repairs, skips, fresh = scan_reference_leaks(col, index)
    totals = _totals(repairs)

    skip_counts: dict[str, int] = {}
    for s in skips:
        reason = s.reason.split(":", 1)[0]
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
    atomic_write_json(args.plan, {
        "created": datetime.now(timezone.utc).isoformat(),
        "repairs": [asdict(r) for r in repairs],
        "skips": [asdict(s) for s in skips],
        "totals": {**totals, "skips": skip_counts},
    })
    for r in repairs:
        print(f"{r.document}  {r.stored_chunks}→{r.fresh_chunks} chunks, "
              f"reference-heavy {r.stored_heavy}→{r.fresh_heavy}")
    for reason, n in sorted(skip_counts.items()):
        print(f"skipped {n}: {reason}")
    print(f"{totals['papers']} papers, {totals['stored_chunks']} stored chunks → "
          f"{totals['fresh_chunks']} fresh chunks, {totals['heavy_removed']} reference-heavy chunks removed")
    print(f"Plan written to {args.plan}")
    return 0


def _apply(args, col, index) -> int:
    if not os.path.exists(args.plan):
        print(f"Plan file {args.plan} does not exist. Run the dry run first.")
        return 1
    try:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
    except Exception as exc:
        print(f"Could not read plan file {args.plan}: {exc}")
        return 1

    from research_assistant.shared import ingestion, pipeline_status

    # upsert_corpus stops embedding on a pending cancel, and every paper's rows are
    # deleted before its upsert: refuse rather than strand papers without chunks.
    if pipeline_status.is_cancel_requested():
        print("A pipeline cancel is pending; clear it (or restart Marvin) before applying.")
        return 1
    try:
        from research_assistant.shared.llm import get_embeddings
        get_embeddings().embed_query("reference repair preflight")
    except Exception as exc:
        print(f"The embedder is not answering ({exc}); nothing was changed.")
        return 1

    res = {"versions_before": {}, "repaired": [], "errors": []}
    with ingestion._ingest_lock():
        repairs, _skips, fresh = scan_reference_leaks(col, index)
        if json.loads(json.dumps([asdict(r) for r in repairs])) != plan_data.get("repairs"):
            print("store changed since the dry run — rerun the dry run")
            return 1
        try:
            res = apply_reference_repair(repairs, fresh, col)
            print(f"Store version before repair: papers={res['versions_before'].get('papers')}")
            if res["repaired"]:
                ingestion.rebuild_bm25()
            db.clear_search_cache()
        finally:
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
            applied_path = os.path.join(config.DATA_DIR, f"reference_repair_applied-{now_str}.json")
            atomic_write_json(applied_path, res)
            print(f"Applied report written to {applied_path}")

    print(f"Repaired {len(res['repaired'])} of {len(repairs)} papers")
    if res.get("errors"):
        print(f"{len(res['errors'])} error(s): {res['errors']}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
