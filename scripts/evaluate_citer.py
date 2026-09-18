"""Evaluate Agent 5's citer against the seed papers' own citations.

    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/evaluate_citer.py --build data/raw/processed/arxiv_2108.10114v3.pdf
    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/evaluate_citer.py --score --runs 2
    PYTHONPATH=. python scripts/evaluate_citer.py --compare data/eval/citer/results/A.json data/eval/citer/results/B.json

--build writes research_assistant/eval/cases/citer_<stem>.jsonl from a seed's
TEI and the download manifest. --score runs the real citer on every case —
the batched citation-need check, then cite_sentence with the seed paper
excluded from retrieval — and writes data/eval/citer/results/<stamp>-citer.json.
Author citations are a floor; every report says so.
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime

from research_assistant.agents import agent5_batch_citer as citer
from research_assistant.config import DATA_DIR
from research_assistant.eval.citer_gold import CASES_DIR, _stem, build, load_cases, write_cases
from research_assistant.eval.citer_metrics import cited_documents, compare, render, score
from research_assistant.shared.atomic import atomic_write_json

RESULTS_DIR = os.path.join(DATA_DIR, "eval", "citer", "results")


def _norm(name: str) -> str:
    return re.sub(r"v\d+$", "", os.path.splitext(os.path.basename(name))[0].lower())


def seed_documents(seed_stem: str, metadatas) -> set:
    """Index documents that are the seed paper itself. Names differ by
    source — 'nature12952.pdf' was ingested as 'doi_10.1038_nature12952.pdf',
    'arxiv_2007.12504v1.pdf' as 'arxiv_2007.12504.pdf' — so a document
    matches when its normalised stem equals the seed's or contains it. Only
    that direction: the seed's stem is the long, specific one, and a short
    document name ('a.pdf') is a substring of almost anything."""
    stem = _norm(seed_stem)
    out = set()
    for m in metadatas:
        doc = (m or {}).get("document")
        if not doc:
            continue
        d = _norm(doc)
        if d == stem or (len(stem) >= 6 and stem in d):
            out.add(doc)
    return out


def run(cases, resources, runs=1, *, judge_gate=None):
    """The citer on every case: need check batched per seed file, as
    run_batch_citer batches a draft, then cite_sentence with the seed
    excluded. One sentence's failure is recorded, never fatal."""
    _, _, _, metadatas = resources
    by_seed = {}
    for c in cases:
        by_seed.setdefault(c["seed"], []).append(c)
    excluded = {seed: sorted(seed_documents(seed, metadatas)) for seed in by_seed}

    results = []
    for run_no in range(1, runs + 1):
        for seed, seed_cases in by_seed.items():
            eligible = [c for c in seed_cases if len(c["sentence"].split()) >= 4]
            need_by_id, need_error = {}, None
            if eligible:
                try:
                    verdicts = citer._batch_needs_citation([c["sentence"] for c in eligible])
                    need_by_id = {c["id"]: v for c, v in zip(eligible, verdicts)}
                except Exception as exc:  # noqa: BLE001 — recorded on every case of the seed
                    need_error = f"{type(exc).__name__}: {exc}"

            key_registry = {}
            for c in seed_cases:
                t0 = time.perf_counter()
                rec = {"case": c, "needs_cite": bool(need_by_id.get(c["id"], False)),
                       "cite": None, "run": run_no, "error": need_error}
                if rec["needs_cite"] and not need_error:
                    try:
                        res = citer.cite_sentence(
                            c["sentence"], resources, key_registry,
                            context=c.get("context"), exclude_docs=set(excluded[seed]),
                            paragraph_id=c.get("paragraph_id"),
                            judge_gate=judge_gate,
                        )
                        rec["cite"] = asdict(res)
                    except Exception as exc:  # noqa: BLE001 — recorded, never fatal
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["seconds"] = round(time.perf_counter() - t0, 2)
                results.append(rec)
                got = sorted(cited_documents(rec["cite"]))
                print(f"run {run_no} {c['id']:16s} need={str(rec['needs_cite']):5s} cited={got} "
                      f"author={c['author_documents']}{'  ERROR ' + rec['error'] if rec['error'] else ''}")

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "judge_gate": judge_gate,
        "excluded": excluded,
        "summary": score(results),
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="*", metavar="SEED_PDF", help="write gold for these seed PDFs")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--judge", choices=["on", "off", "config"], default="config",
                    help="force the judge acceptance test on or off; config reads CITATION_CITER_JUDGE")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as fa, open(args.compare[1]) as fb:
            print(compare(json.load(fa), json.load(fb)))
        return

    if args.build:
        for pdf in args.build:
            records = build(pdf)
            path = CASES_DIR / f"citer_{_stem(pdf)}.jsonl"
            write_cases(records, path)
            n_c = sum(1 for r in records if r["kind"] == "cited")
            print(f"{path}: {n_c} cited + {len(records) - n_c} uncited")
        return

    if args.score:
        from research_assistant.shared.db import load_search_resources

        cases = load_cases(args.cases)
        judge_gate = {"on": True, "off": False}.get(args.judge)
        if judge_gate is None:
            from research_assistant.config import CITATION_CITER_JUDGE
            judge_gate = CITATION_CITER_JUDGE
        out = run(cases, load_search_resources(), runs=args.runs, judge_gate=judge_gate)
        print("\n" + render(out["summary"]))
        for seed, docs in out["excluded"].items():
            print(f"(excluded from search for {seed}: {docs or 'nothing — seed not in the index'})")
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-judge{'on' if judge_gate else 'off'}.json")
        atomic_write_json(path, out)
        print(f"\nwritten {path}")
        return

    ap.error("one of --build, --score, --compare is required")


if __name__ == "__main__":
    main()
