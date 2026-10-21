#!/usr/bin/env python3
"""
CLI runner for Gate F1 card review sheet generation and scoring.

Usage:
    python scripts/review_cards.py --sheet
    python scripts/review_cards.py --score data/eval/transfer/f1/20261021-120000-cards.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.config import DATA_DIR, TRANSFER_DIR
from research_assistant.transfer.cards import CardStore
from research_assistant.transfer.review import review_sheet, score_review


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate review sheet or score Gate F1 review.")
    parser.add_argument("--sheet", action="store_true", help="Generate review sheet and CSV template.")
    parser.add_argument("--documents", nargs="*", default=None, help="Specific documents to include in review sheet.")
    parser.add_argument("--score", type=str, default=None, help="Path to completed CSV file to score.")

    args = parser.parse_args(argv)

    if args.score:
        csv_path = Path(args.score)
        if not csv_path.exists():
            sys.stderr.write(f"File not found: {csv_path}\n")
            return 1

        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        summary = score_review(rows)
        print(f"Gate F1 Evaluation: {summary}")
        if summary.get("passed"):
            print("F1 PASS")
            return 0
        else:
            print("F1 FAIL")
            return 1

    if args.sheet:
        store = CardStore(TRANSFER_DIR)
        index = store.load()

        if args.documents:
            docs = args.documents
        else:
            manifest_path = os.path.join(TRANSFER_DIR, "manifest.json")
            docs = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    docs = manifest.get("_last_sample", [])
                except Exception:
                    docs = []
            if not docs:
                docs = list(store.done_documents())

        if not docs:
            sys.stderr.write("No cards found to generate review sheet.\n")
            return 1

        md_text, csv_text = review_sheet(index, docs)

        out_dir = Path(DATA_DIR) / "eval" / "transfer" / "f1"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        md_path = out_dir / f"{stamp}-cards.md"
        csv_path = out_dir / f"{stamp}-cards.csv"

        md_path.write_text(md_text, encoding="utf-8")
        csv_path.write_text(csv_text, encoding="utf-8")

        print(f"Wrote review markdown to: {md_path}")
        print(f"Wrote review CSV template to: {csv_path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
