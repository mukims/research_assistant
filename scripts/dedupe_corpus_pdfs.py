#!/usr/bin/env python3
"""Deduplicate and quarantine corpus PDFs.

Scans all PDFs in data/, computes SHA-256 hashes, validates member labels against
pages 1–2 text, selects the best representative per duplicate group, and identifies
non-matching members for quarantine.

By default, only reports statistics and outputs data/corpus_dedupe.json.
With --apply, moves duplicate aliases and quarantined files to data/quarantine/ and
rewrites data/migration_snapshot.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pymupdf

from research_assistant.config import DATA_DIR
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.shared.tokenize import tokenize

logger = get_logger("dedupe_corpus")


def get_page12_text(pdf_path: str) -> str:
    """Extract concatenated digital text from pages 1 and 2."""
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for i in range(min(2, len(doc))):
            text += " " + (doc[i].get_text() or "")
        return text
    except Exception as exc:
        logger.debug("Failed to read text from %s: %s", pdf_path, exc)
        return ""


def member_matches(key: str, label: str, page_text: str) -> bool:
    """Check if a PDF member matches its assigned key/label on pages 1–2."""
    if key.startswith("doi_"):
        doi_raw = key[4:].removesuffix(".pdf")
        doi_slash = doi_raw.replace("_", "/", 1).lower()
        doi_compact = re.sub(r"[^a-z0-9]", "", doi_raw.lower())
        page_compact = re.sub(r"[^a-z0-9]", "", page_text.lower())
        return (doi_slash in page_text.lower()) or (doi_compact in page_compact)

    if not label or not label.strip():
        return False

    tokens = tokenize(label)
    if not tokens:
        tokens = [t for t in re.findall(r"[a-z0-9]+", label.lower()) if len(t) > 2]
    if not tokens:
        return False

    page_tokens = set(tokenize(page_text)) | set(re.findall(r"[a-z0-9]+", page_text.lower()))
    matched = [t for t in tokens if t in page_tokens]
    return (len(matched) / len(tokens)) >= 0.8


def prefix_rank(key: str) -> int:
    """Key prefix preference: doi_ > arxiv_ > title_ > other."""
    if key.startswith("doi_"):
        return 0
    if key.startswith("arxiv_"):
        return 1
    if key.startswith("title_"):
        return 2
    return 3


def collect_pdfs(data_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Scan all PDFs in data_dir (excluding quarantine and archive) and group by SHA-256."""
    shas: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for root, dirs, fnames in os.walk(data_dir):
        if "quarantine" in root or "_archive" in root:
            continue
        for f in sorted(fnames):
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                try:
                    with open(path, "rb") as fh:
                        sha = hashlib.sha256(fh.read()).hexdigest()
                    key = os.path.basename(f).strip().replace(" ", "_").lower()
                    shas[sha].append((key, path))
                except Exception as exc:
                    logger.warning("Could not read %s: %s", path, exc)
    return shas


def main():
    parser = argparse.ArgumentParser(description="Deduplicate and quarantine corpus PDFs.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move duplicate aliases and quarantined files to data/quarantine/ and rewrite migration_snapshot.json",
    )
    args = parser.parse_args()

    snapshot_path = os.path.join(DATA_DIR, "migration_snapshot.json")
    if not os.path.isfile(snapshot_path):
        logger.error("Snapshot not found at %s", snapshot_path)
        sys.exit(1)

    with open(snapshot_path, "r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    labels = snapshot.get("labels", {})
    existing_summaries = snapshot.get("existing_summaries", {})
    abstract_only_records = snapshot.get("abstract_only_records", [])

    logger.info("Scanning PDFs under %s...", DATA_DIR)
    shas = collect_pdfs(DATA_DIR)

    total_files = sum(len(v) for v in shas.values())
    distinct_shas = len(shas)
    dup_groups = {s: items for s, items in shas.items() if len(items) > 1}
    single_groups = {s: items[0] for s, items in shas.items() if len(items) == 1}
    extra_copies = sum(len(items) - 1 for items in dup_groups.values())

    logger.info("Total PDF files: %d", total_files)
    logger.info("Distinct contents (SHA-256): %d", distinct_shas)
    logger.info("Duplicate groups: %d", len(dup_groups))
    logger.info("Extra duplicate copies: %d", extra_copies)

    kept: dict[str, str] = {}
    aliases: dict[str, str] = {}
    quarantined: list[dict] = []
    quarantine_paths: list[str] = []
    alias_paths: list[str] = []

    # Process duplicate groups
    for sha, items in sorted(dup_groups.items()):
        # Sample text from first copy
        sample_path = items[0][1]
        page_text = get_page12_text(sample_path)

        matching_members: list[tuple[str, str]] = []
        non_matching_members: list[tuple[str, str]] = []

        for key, path in items:
            lbl = labels.get(key, "")
            if member_matches(key, lbl, page_text):
                matching_members.append((key, path))
            else:
                non_matching_members.append((key, path))

        if not matching_members:
            # Whole group fails to match: quarantine all members
            for key, path in items:
                quarantined.append({
                    "key": key,
                    "sha256": sha,
                    "label": labels.get(key, ""),
                    "reason": "no_member_matches",
                    "path": path,
                })
                quarantine_paths.append(path)
        else:
            # Sort matching members by prefix rank (doi > arxiv > title > other),
            # then descending by label length, then key
            matching_members.sort(
                key=lambda x: (prefix_rank(x[0]), -len(labels.get(x[0], "")), x[0])
            )
            kept_key, kept_path = matching_members[0]
            kept[sha] = kept_key

            # Extra matching members become aliases
            for key, path in matching_members[1:]:
                aliases[key] = kept_key
                alias_paths.append(path)

            # Non-matching members are quarantined
            for key, path in non_matching_members:
                quarantined.append({
                    "key": key,
                    "sha256": sha,
                    "label": labels.get(key, ""),
                    "reason": "label_mismatch",
                    "path": path,
                })
                quarantine_paths.append(path)

    # For unique/single-copy groups, record as kept
    for sha, (key, path) in single_groups.items():
        kept[sha] = key

    # Single-copy title_* check (for report only)
    single_title_members = [
        (k, p) for s, (k, p) in single_groups.items() if k.startswith("title_")
    ]
    logger.info("Evaluating %d single-copy title_* files against page 1-2 text...", len(single_title_members))
    failed_single_title = 0
    for k, p in single_title_members:
        txt = get_page12_text(p)
        lbl = labels.get(k, "")
        if not member_matches(k, lbl, txt):
            failed_single_title += 1

    # Prepare corpus_dedupe.json structure
    # Clean quarantined list to match schema
    quarantined_clean = [
        {
            "key": q["key"],
            "sha256": q["sha256"],
            "label": q["label"],
            "reason": q["reason"],
        }
        for q in quarantined
    ]

    dedupe_payload = {
        "kept": kept,
        "aliases": aliases,
        "quarantined": quarantined_clean,
    }

    dedupe_json_path = os.path.join(DATA_DIR, "corpus_dedupe.json")
    atomic_write_json(dedupe_json_path, dedupe_payload, indent=2)
    logger.info("Wrote %s", dedupe_json_path)

    print("\n" + "=" * 60)
    print("DEDUPLICATION & QUARANTINE REPORT")
    print("=" * 60)
    print(f"Total PDF file count:             {total_files}")
    print(f"Distinct contents (SHA-256):       {distinct_shas}")
    print(f"Duplicate groups:                 {len(dup_groups)}")
    print(f"Extra duplicate copies:           {extra_copies}")
    print(f"Kept keys (unique SHAs):          {len(kept)}")
    print(f"Aliases (extra matching copies):  {len(aliases)}")
    print(f"Quarantined files:                {len(quarantined)}")
    print(f"Single-copy title_* label fails:  {failed_single_title}")
    print("=" * 60)

    print("\nQuarantined Files List:")
    for idx, q in enumerate(quarantined, 1):
        print(f"  {idx:3d}. [{q['reason']}] {q['key']} (SHA: {q['sha256'][:10]}...) — Label: {q['label']!r}")

    if args.apply:
        quarantine_dir = os.path.join(DATA_DIR, "quarantine")
        os.makedirs(quarantine_dir, exist_ok=True)
        logger.info("Moving %d quarantined and %d alias files to %s...", len(quarantine_paths), len(alias_paths), quarantine_dir)

        moved_count = 0
        for p in quarantine_paths + alias_paths:
            if os.path.isfile(p):
                dest = os.path.join(quarantine_dir, os.path.basename(p))
                if os.path.abspath(p) != os.path.abspath(dest):
                    shutil.move(p, dest)
                    moved_count += 1

        logger.info("Moved %d files to %s", moved_count, quarantine_dir)

        # Rewrite migration_snapshot.json
        logger.info("Rewriting %s...", snapshot_path)
        new_labels = {}
        for k, v in labels.items():
            if k in aliases:
                continue
            new_labels[k] = v

        new_summaries = {}
        for k, v in existing_summaries.items():
            if k in aliases:
                continue
            new_summaries[k] = v

        # Drop abstract-only records whose document is now a kept full-text key
        kept_keys_set = set(kept.values())
        new_abstracts = []
        for r in abstract_only_records:
            doc_id = r.get("document", "")
            norm_key = doc_id.lower().replace("doi:", "doi_").replace("/", "_")
            if not norm_key.endswith(".pdf"):
                norm_key += ".pdf"
            if norm_key not in kept_keys_set and doc_id not in kept_keys_set:
                new_abstracts.append(r)

        new_snapshot = {
            "labels": new_labels,
            "existing_summaries": new_summaries,
            "abstract_only_records": new_abstracts,
        }
        atomic_write_json(snapshot_path, new_snapshot, indent=2)
        logger.info(
            "Snapshot rewritten: %d labels (was %d), %d summaries (was %d), %d abstracts (was %d)",
            len(new_labels), len(labels),
            len(new_summaries), len(existing_summaries),
            len(new_abstracts), len(abstract_only_records),
        )


if __name__ == "__main__":
    main()
