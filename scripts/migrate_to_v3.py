"""Re-ingest every paper the older indexes hold into index v3.

    PYTHONPATH=. python scripts/migrate_to_v3.py --dry-run
    PYTHONPATH=. python scripts/migrate_to_v3.py --limit 3          # trial run
    CITATION_DATA_DIR=<scratch> PYTHONPATH=. python scripts/migrate_to_v3.py --limit 3 --from-data-dir data
    PYTHONPATH=. python scripts/migrate_to_v3.py                    # everything

Index v3 (embeddinggemma-2, figure images) has its own collections and manifest,
so this never touches the v1 or v2 data. It collects the document keys recorded
in ingested.json and ingested_v2.json, finds each PDF on disk, and hands them to
ingest_pdfs() in batches. ingest_pdfs() marks a paper in the v3 manifest only
once it is stored, so an interrupted run resumes where it stopped and a finished
paper is never redone. The BM25 index is rebuilt once at the end, or on Ctrl-C.

Needs: GROBID answering (the figures come from its coordinates), Ollama with
embeddinggemma-2, a summary model that loads, and disk. CITATION_EMBED_MODEL in
.env or data/.env overrides the default, so remove a leftover nomic value there.
"""

import argparse
import json
import os
import shutil
import sys
import time

from research_assistant import config
from research_assistant.shared.ingestion import pdf_key

MIN_FREE_GIB = 2.0
BATCH_SIZE = 10
OLD_MANIFESTS = ("ingested.json", "ingested_v2.json")


def old_document_keys(data_dir: str) -> tuple[list[str], list[str]]:
    """(pdf keys, other keys) recorded by the v1 and v2 manifests, sorted."""
    keys = set()
    for name in OLD_MANIFESTS:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                keys.update(json.load(fh))
    pdfs = sorted(k for k in keys if k.endswith(".pdf"))
    return pdfs, sorted(keys - set(pdfs))


def find_pdfs(keys: list[str], search_dirs: list[str]) -> tuple[dict[str, str], list[str]]:
    """({key: path}, [keys with no PDF on disk]). A key is a PDF's pdf_key()."""
    wanted = set(keys)
    found: dict[str, str] = {}
    for root_dir in search_dirs:
        for root, _, files in sorted(os.walk(root_dir)):
            for name in sorted(files):
                key = pdf_key(name)
                if key in wanted and key not in found:
                    found[key] = os.path.join(root, name)
    return found, sorted(wanted - set(found))


def grobid_alive() -> bool:
    import requests

    try:
        requests.get(f"{config.GROBID_SERVER}/api/isalive", timeout=5).raise_for_status()
        return True
    except Exception:                                          # noqa: BLE001
        return False


def run_batches(todo, found, ingest, alive, batch_size, say=print):
    """Ingest `todo` keys batch by batch. Returns (totals, stop reason or "").

    A paper GROBID could not read falls back to PyMuPDF and is still marked
    done, with no figures, and a re-run skips it. So GROBID is checked before
    every batch, and the run stops at the first batch that fell back.
    """
    totals = {"processed": 0, "inserted": 0, "failed": 0, "summaries_failed": 0, "figure_images": 0,
              "grobid": 0, "pymupdf": 0}
    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        keys = todo[i : i + batch_size]
        if not alive():
            return totals, f"GROBID stopped answering before the batch starting at {keys[0]}"
        result = ingest([found[k] for k in keys])
        totals["processed"] += result["processed"]
        totals["inserted"] += result["inserted"]
        totals["failed"] += len(result["failed"])
        totals["summaries_failed"] += (result.get("summaries") or {}).get("failed", 0)
        totals["figure_images"] += result.get("figure_images", 0)
        for mode, n in (result.get("extraction") or {}).items():
            totals[mode] = totals.get(mode, 0) + n
        say(f"[{min(i + batch_size, len(todo))}/{len(todo)}] {time.time() - t0:.0f}s  {totals}")
        if (result.get("extraction") or {}).get("pymupdf"):
            return totals, (f"{result['extraction']['pymupdf']} paper(s) in the batch {keys[0]} .. {keys[-1]} "
                            "fell back to PyMuPDF (no figures)")
    return totals, ""


def preflight() -> list[str]:
    """Reasons this run cannot succeed, empty when it can."""
    import requests

    problems = []
    if config.INDEX_VERSION != 3:
        problems.append(f"CITATION_INDEX_VERSION is {config.INDEX_VERSION}, not 3")
    if not config.EMBED_MODEL.startswith("embeddinggemma-2"):
        problems.append(
            f"the embedder is {config.EMBED_MODEL!r}: remove CITATION_EMBED_MODEL from .env and "
            "data/.env (or set it to embeddinggemma-2)"
        )
    if not grobid_alive():
        problems.append(f"GROBID is not answering at {config.GROBID_SERVER} (docker start grobid)")
    try:
        from research_assistant.shared.llm import _ollama_host

        tags = requests.get(f"{_ollama_host()}/api/tags", timeout=5).json()
        names = {m["name"] for m in tags.get("models", [])}
        if not any(n == config.EMBED_MODEL or n.startswith(config.EMBED_MODEL + ":") for n in names):
            problems.append(f"Ollama has no {config.EMBED_MODEL} (ollama pull {config.EMBED_MODEL})")
    except Exception as exc:                                   # noqa: BLE001
        problems.append(f"Ollama is not answering ({exc})")
    free = shutil.disk_usage(config.DATA_DIR).free / 2**30
    if free < MIN_FREE_GIB:
        problems.append(f"only {free:.1f} GiB free where {config.DATA_DIR} lives (need {MIN_FREE_GIB})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list what would be ingested and stop")
    parser.add_argument("--limit", type=int, default=None, help="ingest at most this many papers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--from-data-dir", default=config.DATA_DIR,
                        help="where the old manifests and PDFs are read from (read-only); defaults to the data "
                             "directory v3 is written to. Point CITATION_DATA_DIR at a scratch directory and this "
                             "at the real one for a trial that leaves the real data alone.")
    args = parser.parse_args(argv)

    source = args.from_data_dir
    keys, other = old_document_keys(source)
    search = [d for d in (os.path.join(source, "pulled_pdfs"), os.path.join(source, "raw"),
                          os.path.join(source, "raw_pdfs")) if os.path.isdir(d)]
    found, missing = find_pdfs(keys, search)
    from research_assistant.shared import manifest

    done = set(manifest.load())
    todo = [k for k in sorted(found) if k not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"older indexes record {len(keys)} PDFs ({len(other)} other entries ignored: {other[:3]})")
    print(f"  on disk: {len(found)}   missing: {len(missing)}   already in v3: {len(done & set(found))}   to ingest: {len(todo)}")
    for key in missing[:10]:
        print(f"  missing: {key}")
    if args.dry_run or not todo:
        return 0

    problems = preflight()
    if problems:
        print("\nNot starting:")
        for p in problems:
            print(f"  - {p}")
        return 2

    from research_assistant.shared.ingestion import ingest_pdfs, rebuild_bm25
    from watch import _labeled_pdfs

    try:
        totals, stopped = run_batches(
            todo, found,
            lambda paths: ingest_pdfs(_labeled_pdfs(paths), workers=1, rebuild_index=False),
            grobid_alive, args.batch_size,
        )
    finally:
        print("rebuilding the BM25 index…", flush=True)
        rebuild_bm25()
    print(f"\ndone: {totals}")
    if stopped:
        print(f"\nSTOPPED: {stopped}.\nThe papers before that batch are complete. Papers the batch lost to PyMuPDF "
              "are marked done with text only: fix GROBID, then remove those keys from "
              f"{os.path.basename(config.INGESTED_MANIFEST_PATH)} (and tell me, a repair option is a small addition).")
        return 3
    if totals["failed"] or totals["summaries_failed"]:
        print("Papers that failed outright are not marked, and a re-run retries them. A paper that lost its summary "
              "IS marked done and a re-run will not redo it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
