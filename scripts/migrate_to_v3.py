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

Needs: GROBID answering (the figures come from its coordinates), Ollama 0.36 or
newer with embeddinggemma-2 (ollama pull embeddinggemma-2), a summary model that
answers (a Gemini key in .env, or a local model that loads), and disk.
CITATION_EMBED_MODEL in .env or data/.env overrides the default, so remove a
leftover nomic value there.

On another machine
  1. Install: pip install -e . && pip install -r requirements.txt, then docker run
     -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
  2. Copy the old data over, read-only inputs: ingested.json, ingested_v2.json,
     downloaded.json, the PDF folders (pulled_pdfs/, raw/, raw_pdfs/) and, to skip
     GROBID work where possible, raw/grobid_output/ (119 MB; only TEI that has figure
     boxes is reused, the rest is read again). The Chroma stores are not needed.
  3. Point --from-data-dir at that copy and CITATION_DATA_DIR at where v3 should live.
     Labels come from downloaded.json in the source; the GROBID cache is copied from it.
  4. --dry-run first, then --limit 3, then everything, then --verify.
  (On the OLD machine, once: scripts/export_old_labels.py writes <data>/migration_labels.json, the
   label each paper has in the old index, and lists papers in the index but in no manifest.
   Copy it with the rest; without it, labels fall back to downloaded.json and then file names.)
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
EXPORTED_LABELS = "migration_labels.json"      # written by scripts/export_old_labels.py


def exported_labels(data_dir: str) -> dict[str, str]:
    """{pdf key: label} exported from the old Chroma store, or {} when there is no export."""
    path = os.path.join(data_dir, EXPORTED_LABELS)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("labels", {})


def manifest_keys(data_dir: str) -> set[str]:
    """Every document key the v1 and v2 manifests record."""
    keys: set[str] = set()
    for name in OLD_MANIFESTS:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                keys.update(json.load(fh))
    return keys


def old_document_keys(data_dir: str) -> tuple[list[str], list[str]]:
    """(pdf keys, other keys) the old indexes hold, sorted.

    The manifests, plus the papers the exported labels show are in the old store: hand-added
    papers were in the store but in no manifest, and a manifest-only list would drop them."""
    keys = set(exported_labels(data_dir)) | manifest_keys(data_dir)
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


def tei_for(cache_dir: str, pdf_path: str):
    """A cached TEI for this PDF that can be used (it has figure boxes, or no figures), else None."""
    from research_assistant.shared.extract import find_cached_tei

    return find_cached_tei(cache_dir, pdf_path)


def seed_tei_cache(paths, source_dir: str, target_dir: str) -> tuple[int, int, int]:
    """Copy usable cached TEI the target lacks from the source (never overwriting).

    Returns (copied, already in the target, in neither: GROBID must read them).
    Agent 1's TEI has figures but no boxes for them, so it does not count."""
    copied = had = uncached = 0
    for path in paths:
        if tei_for(target_dir, path):
            had += 1
            continue
        found = tei_for(source_dir, path)
        if found is None:
            uncached += 1
            continue
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(found, os.path.join(target_dir, os.path.basename(found)))
        copied += 1
    return copied, had, uncached


def label_map(source_dir: str) -> dict[str, str]:
    """{pdf key: citation label}. The label the old index holds wins, then downloaded.json.

    A label becomes citation_source on every chunk and so the cite key. It is read from the
    source, not from DATA_DIR: a fresh target has neither file, and every paper would be
    labelled by its file name. Where the two disagree the old index is right: downloaded.json
    sometimes holds a journal abbreviation ("Nat. Phys") where the index has the title."""
    from research_assistant.agents.agent3_ingestor import _pdfs_from_manifest

    labels: dict[str, str] = {}
    path = os.path.join(source_dir, "downloaded.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            downloaded = json.load(fh)
        labels.update({pdf_key(p): label for p, label in _pdfs_from_manifest(downloaded).items()})
    labels.update(exported_labels(source_dir))
    return labels


def labeled(paths, by_key: dict[str, str]) -> dict[str, str]:
    """{path: label}: the known label, else the file name without its extension."""
    return {p: by_key.get(pdf_key(p)) or os.path.splitext(os.path.basename(p))[0] for p in paths}


def bm25_is_stale(bm25_path: str, manifest_path: str) -> bool:
    """True when papers are recorded but the keyword index is missing or older than
    the record, which is what a hard kill (power, kill -9) between batches leaves."""
    if not os.path.exists(manifest_path):
        return False
    return not os.path.exists(bm25_path) or os.path.getmtime(bm25_path) < os.path.getmtime(manifest_path)


MIN_OLLAMA = (0, 36, 0)


def ollama_version_problem(version: str):
    import re

    m = re.search(r"(\d+)\.(\d+)\.(\d+)", version or "")
    if not m:
        return f"Ollama reported an unreadable version {version!r}"
    if tuple(int(x) for x in m.groups()) < MIN_OLLAMA:
        return f"Ollama {version} is older than {'.'.join(map(str, MIN_OLLAMA))}, which embeddinggemma-2 needs (update Ollama)"
    return None


def summary_model_problem(chat_fn):
    """Why the summary model cannot answer, or None. A paper whose summary fails is
    still marked done, so this has to be known before the first batch."""
    name = config.SUMMARY_MODEL or config.LLM_MODEL
    try:
        reply = chat_fn([{"role": "user", "content": "Reply with the single word: ok"}], model=config.SUMMARY_MODEL)
    except Exception as exc:                                   # noqa: BLE001
        return f"the summary model ({name}) failed: {exc}"
    if not (getattr(reply, "content", "") or "").strip():
        return f"the summary model ({name}) returned an empty reply"
    return None


def verify_counts(metadatas, manifest_keys, summary_docs, expected_keys) -> dict:
    """What a finished migration should show, from the stored metadata alone.

    not_migrated: named by the old manifests, absent from the v3 manifest.
    no_summary / no_text: in the v3 manifest but lacking a summary, or any text chunk.
    figures_without_image: captions beyond the figure images (a figure with no box)."""
    import collections

    by_doc = collections.defaultdict(collections.Counter)
    for m in metadatas:
        by_doc[m["document"]][m["type"]] += 1
    types = collections.Counter()
    for counts in by_doc.values():
        types.update(counts)
    manifest_keys = set(manifest_keys)
    return {
        "papers_in_manifest": len(manifest_keys),
        "papers_with_chunks": len(by_doc),
        "types": dict(types),
        "not_migrated": sorted(set(expected_keys) - manifest_keys),
        "no_summary": sorted(manifest_keys - set(summary_docs)),
        "no_text": sorted(k for k in manifest_keys if not by_doc.get(k, {}).get("text_chunk")),
        "figures_without_image": max(0, types["caption"] - types["figure_image"]),
    }


def verify(expected_keys) -> int:
    """Open the v3 store (read-only use) and print whether the migration is complete."""
    import chromadb

    from research_assistant.shared import manifest

    client = chromadb.PersistentClient(path=config.VECTORDB_PATH)
    col = client.get_collection(config.COLLECTION_NAME)
    summaries = client.get_collection(config.SUMMARY_COLLECTION_NAME)

    def all_metadatas(c):
        out, offset = [], 0
        while True:
            batch = c.get(include=["metadatas"], limit=5000, offset=offset)
            if not batch["ids"]:
                return out
            out.extend(batch["metadatas"])
            offset += 5000

    r = verify_counts(all_metadatas(col), manifest.load(), {m["document"] for m in all_metadatas(summaries)}, expected_keys)
    print(f"index v{config.INDEX_VERSION} ({config.COLLECTION_NAME}), embedder record: {col.metadata}")
    print(f"  papers recorded: {r['papers_in_manifest']}   with chunks: {r['papers_with_chunks']}   entries by type: {r['types']}")
    print(f"  figures with no image (no box in GROBID's output): {r['figures_without_image']}")
    for label, key in (("NOT MIGRATED", "not_migrated"), ("NO SUMMARY", "no_summary"), ("NO TEXT CHUNK (blank or scanned?)", "no_text")):
        if r[key]:
            print(f"  {label}: {len(r[key])}  e.g. {r[key][:5]}")
    bad = bool(r["not_migrated"] or r["no_summary"])
    print("  RESULT:", "INCOMPLETE" if bad else "complete" + (" (see notes above)" if r["no_text"] else ""))
    return 4 if bad else 0


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

        problem = ollama_version_problem(requests.get(f"{_ollama_host()}/api/version", timeout=5).json().get("version"))
        if problem:
            problems.append(problem)
        tags = requests.get(f"{_ollama_host()}/api/tags", timeout=5).json()
        names = {m["name"] for m in tags.get("models", [])}
        if not any(n == config.EMBED_MODEL or n.startswith(config.EMBED_MODEL + ":") for n in names):
            problems.append(f"Ollama has no {config.EMBED_MODEL} (ollama pull {config.EMBED_MODEL})")
    except Exception as exc:                                   # noqa: BLE001
        problems.append(f"Ollama is not answering ({exc})")
    from research_assistant.shared.llm import chat

    problem = summary_model_problem(chat)
    if problem:
        problems.append(problem)
    free = shutil.disk_usage(config.DATA_DIR).free / 2**30
    if free < MIN_FREE_GIB:
        problems.append(f"only {free:.1f} GiB free where {config.DATA_DIR} lives (need {MIN_FREE_GIB})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list what would be ingested and stop")
    parser.add_argument("--limit", type=int, default=None, help="ingest at most this many papers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--verify", action="store_true",
                        help="check the finished v3 index against the old manifests and stop")
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

    by_base = label_map(source)
    exported = exported_labels(source)
    from_index = sum(1 for k in todo if k in exported)
    from_manifest = sum(1 for k in todo if k not in exported and pdf_key(found[k]) in by_base)
    store_only = len(set(exported) - manifest_keys(source))
    source_cache = os.path.join(source, "raw", "grobid_output")
    cached = sum(1 for k in todo if tei_for(config.GROBID_TEI_DIR, found[k]) or tei_for(source_cache, found[k]))
    print(f"  labels: {from_index} from the old index, {from_manifest} from downloaded.json, "
          f"{len(todo) - from_index - from_manifest} will be labelled by file name"
          + ("" if exported else "   (no migration_labels.json: run scripts/export_old_labels.py on the old machine)"))
    if store_only:
        print(f"  {store_only} papers are in the old index but in no manifest; they are included")
    print(f"  GROBID cache: {cached} of {len(todo)} already have a TEI; GROBID reads the other {len(todo) - cached}")
    if args.dry_run:
        return 0
    if args.verify:
        return verify(set(found))
    if not todo:
        if bm25_is_stale(config.BM25_INDEX_PATH, config.INGESTED_MANIFEST_PATH):
            from research_assistant.shared.ingestion import rebuild_bm25

            print("nothing to ingest, but the keyword index is missing or older than the record: rebuilding it")
            rebuild_bm25()
        return 0

    problems = preflight()
    if problems:
        print("\nNot starting:")
        for p in problems:
            print(f"  - {p}")
        return 2

    from research_assistant.shared.ingestion import ingest_pdfs, rebuild_bm25

    copied, had, uncached = seed_tei_cache([found[k] for k in todo], source_cache, config.GROBID_TEI_DIR)
    print(f"GROBID cache: {copied} TEI copied from the source, {had} already here, {uncached} need GROBID")
    try:
        totals, stopped = run_batches(
            todo, found,
            lambda paths: ingest_pdfs(labeled(paths, by_base), workers=1, rebuild_index=False),
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
