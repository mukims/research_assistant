#!/usr/bin/env python3
"""Retrodiction evaluation harness and CLI for technique transfer (Gates F2 and F3)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.eval import transfer_metrics
from research_assistant.eval.transfer_cases import Case, case_data_dir, load_cases
from research_assistant.shared.atomic import atomic_write, atomic_write_json
from research_assistant.transfer import engine
from research_assistant.transfer.deps import Deps, TransferSettings
from research_assistant.transfer.models import result_from_dict, result_to_dict
from research_assistant.transfer.render import premise_badge
from research_assistant.transfer.scoring import final_rows


def get_case_resolved(case: Case) -> dict[str, str]:
    """Retrieve resolved mapping {title -> document} for a case."""
    candidates = [
        os.path.join(case_data_dir(case.id), "resolved.json"),
        os.path.join(config.DATA_DIR, "resolved.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass

    # Fallback default: map each title to itself if no explicit resolution file exists
    all_titles = (
        case.papers.get("target", [])
        + case.papers.get("source", [])
        + case.papers.get("distractor", [])
        + case.papers.get("hidden", [])
    )
    return {t: t for t in all_titles}


def get_corpus_documents(case: Case, data_dir: str | None = None) -> set[str]:
    """Return set of document filenames present in the active corpus."""
    if data_dir is None:
        data_dir = case_data_dir(case.id) if case.corpus == "case" else config.DATA_DIR

    docs: set[str] = set()

    # 1. From metadata.json
    meta_p = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_p):
        try:
            with open(meta_p, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                docs.update(meta.keys())
        except Exception:
            pass

    # 2. From cards.jsonl
    cards_p = os.path.join(data_dir, "transfer", "cards.jsonl")
    if os.path.exists(cards_p):
        try:
            with open(cards_p, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        card = json.loads(line)
                        if "document" in card:
                            docs.add(card["document"])
        except Exception:
            pass

    # 3. From filesystem PDFs
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".pdf"):
                docs.add(f)

    return docs


def run_retrieval_case(case: Case, deps: Deps) -> dict[str, Any]:
    """Run F2 retrieval evaluation for a single case using card embeddings and search."""
    query_vec = deps.embed_query(case.generic)
    exclude = set(case.exclude_documents)
    results = deps.cards.search(query_vec, k=50, exclude_documents=exclude, verified_only=True)
    seeds = [
        {
            "card_document": c.document,
            "score": sim,
            "card_id": c.card_id,
            "name": c.name,
        }
        for c, sim in results
    ]

    resolved = get_case_resolved(case)
    expected_docs = transfer_metrics.expected_documents(case, resolved)
    rank = transfer_metrics.retrieval_rank(seeds, expected_docs)

    return {
        "case_id": case.id,
        "retrieval_rank": rank,
        "expected_documents": sorted(list(expected_docs)),
        "seeds_count": len(seeds),
    }


def run_score_case(case: Case, deps: Deps, run_idx: int = 0) -> dict[str, Any]:
    """Run full F3 retrodiction evaluation for a single case."""
    settings = TransferSettings(
        exclude_documents=frozenset(case.exclude_documents),
        prior_art_before_year=case.cutoff_year,
    )
    result = engine.run_best_of_n(case.query, deps, settings)
    resolved = get_case_resolved(case)
    expected_docs = transfer_metrics.expected_documents(case, resolved)
    eval_res = transfer_metrics.case_passed(result, case, expected_docs)

    return {
        "case_id": case.id,
        "run": run_idx + 1,
        "passed": eval_res["passed"],
        "found_rank": eval_res["found_rank"],
        "mapped": eval_res["mapped"],
        "critiqued": eval_res["critiqued"],
        "reason": eval_res["reason"],
        "transfers": [
            {
                "id": t.id,
                "card_id": t.card_id,
                "card_document": t.card_document,
                "title": t.title,
                "scores": (
                    {"reward": t.scores.reward, "depth": t.scores.depth, "set_aside": t.scores.set_aside}
                    if t.scores
                    else None
                ),
            }
            for t in result.transfers[:3]
        ],
        "result_dict": result_to_dict(result),
    }


def run_case_in_process(
    case: Case,
    mode: str = "retrieval",
    runs: int = 1,
    deps: Deps | None = None,
) -> list[dict[str, Any]]:
    """Execute case evaluation in-process."""
    if deps is None:
        deps = engine.live_deps()

    if mode == "retrieval":
        return [run_retrieval_case(case, deps)]
    elif mode == "score":
        return [run_score_case(case, deps, run_idx=i) for i in range(runs)]
    return []


def case_runner(
    case: Case,
    mode: str = "retrieval",
    runs: int = 1,
    deps: Deps | None = None,
) -> list[dict[str, Any]]:
    """
    Execute case evaluation, delegating to child subprocess if case has its own corpus.
    """
    if deps is not None:
        return run_case_in_process(case, mode=mode, runs=runs, deps=deps)

    target_dir = case_data_dir(case.id)
    if case.corpus == "case" and os.path.realpath(config.DATA_DIR) != os.path.realpath(target_dir):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                sys.executable,
                __file__,
                "--case-single",
                case.id,
                "--record-to",
                tmp_path,
            ]
            if mode == "retrieval":
                cmd.append("--retrieval-only")
            else:
                cmd.extend(["--score", "--runs", str(runs)])

            subprocess.run(
                cmd,
                env={**os.environ, "CITATION_DATA_DIR": target_dir},
                check=True,
            )
            with open(tmp_path, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            return records if isinstance(records, list) else [records]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return run_case_in_process(case, mode=mode, runs=runs, deps=None)


def write_review_md(
    review_path: str,
    cases: list[Case],
    records: list[dict[str, Any]],
) -> None:
    """Write human-readable review markdown of the top 3 transfers per case."""
    lines = ["# Retrodiction Run Review\n\n"]
    by_case: dict[str, list[dict]] = {}
    for r in records:
        cid = r.get("case_id", "unknown")
        by_case.setdefault(cid, []).append(r)

    case_map = {c.id: c for c in cases}

    for cid, runs in sorted(by_case.items()):
        c = case_map.get(cid)
        query = c.query if c else cid
        lines.append(f"## Case: {cid}\n\n")
        lines.append(f"**Query:** {query}\n\n")

        for run_info in runs:
            run_num = run_info.get("run", 1)
            passed_str = "PASS" if run_info.get("passed") else "FAIL"
            rank_str = run_info.get("found_rank") or "-"
            lines.append(f"### Run {run_num} [{passed_str}] (rank: {rank_str})\n\n")
            lines.append(f"Reason: {run_info.get('reason', '')}\n\n")

            res_dict = run_info.get("result_dict")
            if res_dict:
                try:
                    result = result_from_dict(res_dict)
                    for idx, t in enumerate(result.transfers[:3], start=1):
                        lines.append(f"#### Top {idx}: {t.title} (<- {t.card_document})\n\n")
                        card_obj = result.cards.get(t.card_id)
                        card_name = card_obj.name if card_obj else t.card_document
                        lines.append(f"**Technique:** {card_name}\n\n")
                        lines.append(f"**Proposal:** {t.summary}\n\n")
                        lines.append(f"- **Baseline:** {t.baseline}\n")
                        lines.append(f"- **Metric:** {t.metric}\n")
                        lines.append(f"- **Prediction:** {t.prediction}\n")
                        lines.append(f"- **Refuting result:** {t.refuting_result}\n\n")
                        lines.append("---\n\n")
                except Exception as exc:
                    lines.append(f"(Could not render transfers: {exc})\n\n")

    with atomic_write(review_path) as fh:
        fh.write("".join(lines))


def compare_results(file_a: str, file_b: str) -> None:
    """Print per-case differences between two evaluation results JSON files."""
    with open(file_a, "r", encoding="utf-8") as fh:
        data_a = json.load(fh)
    with open(file_b, "r", encoding="utf-8") as fh:
        data_b = json.load(fh)

    cases_a = data_a.get("cases", {})
    cases_b = data_b.get("cases", {})
    all_cases = sorted(set(cases_a.keys()) | set(cases_b.keys()))

    sys.stdout.write(f"\nComparison: {file_a} vs {file_b}\n")
    sys.stdout.write(f"{'Case':<12} {'A Pass':<8} {'A Rank':<8} {'B Pass':<8} {'B Rank':<8}\n")
    sys.stdout.write("-" * 50 + "\n")

    for cid in all_cases:
        runs_a = cases_a.get(cid, [])
        runs_b = cases_b.get(cid, [])
        pass_a = all(r.get("passed", False) for r in runs_a) if runs_a else False
        pass_b = all(r.get("passed", False) for r in runs_b) if runs_b else False
        rank_a = str(runs_a[0].get("found_rank")) if runs_a else "-"
        rank_b = str(runs_b[0].get("found_rank")) if runs_b else "-"
        sys.stdout.write(f"{cid:<12} {str(pass_a):<8} {rank_a:<8} {str(pass_b):<8} {rank_b:<8}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrodiction evaluation harness.")
    parser.add_argument("--retrieval-only", action="store_true", help="Gate F2: Card retrieval search only.")
    parser.add_argument("--score", action="store_true", help="Gate F3: Full transfer retrodiction evaluation.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per case (e.g. --runs 2 for F3).")
    parser.add_argument("--cases", nargs="*", default=None, help="Specific case IDs to evaluate.")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("FILE_A", "FILE_B"), help="Compare two results JSON files.")
    parser.add_argument("--case-single", type=str, default=None, help="Execute a single case (child process).")
    parser.add_argument("--record-to", type=str, default=None, help="File to write single case JSON record.")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save eval results.")

    args = parser.parse_args(argv)

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return 0

    all_cases = load_cases()
    if args.cases:
        selected_cases = [c for c in all_cases if c.id in args.cases]
    else:
        selected_cases = all_cases

    # Internal child process mode
    if args.case_single:
        case = next((c for c in all_cases if c.id == args.case_single), None)
        if case is None:
            sys.stderr.write(f"Unknown case: {args.case_single}\n")
            return 1
        mode = "retrieval" if args.retrieval_only else "score"
        records = run_case_in_process(case, mode=mode, runs=args.runs)
        if args.record_to:
            atomic_write_json(args.record_to, records)
        return 0

    if args.score:
        # 1. Guard check: ensure_idle before anything else
        try:
            engine.ensure_idle()
        except engine.BusyError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2

        # 2. Leak guard across all cases before running any case
        for case in selected_cases:
            resolved = get_case_resolved(case)
            corpus_docs = get_corpus_documents(case)
            try:
                transfer_metrics.leak_guard(case, resolved, corpus_docs)
            except RuntimeError as exc:
                sys.stderr.write(f"Leak guard error: {exc}\n")
                return 1

        all_records = []
        for case in selected_cases:
            records = case_runner(case, mode="score", runs=args.runs)
            all_records.extend(records)

        summary = transfer_metrics.score(all_records)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or os.path.join(config.DATA_DIR, "eval", "transfer", "results")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{stamp}.json")
        review_path = os.path.join(out_dir, f"{stamp}-review.md")

        atomic_write_json(
            json_path,
            {"summary": summary, "records": all_records, "created_at": stamp},
        )
        write_review_md(review_path, selected_cases, all_records)

        sys.stdout.write(f"JSON: {json_path}\n")
        sys.stdout.write(f"Review: {review_path}\n\n")

        f3 = summary["f3"]
        status = "PASS" if f3["all_passed"] else "FAIL"
        sys.stdout.write(f"F3 {status}: {f3['passed']} of {f3['of']} passed (spread={f3['spread']:.2f})\n")
        return 0 if f3["all_passed"] else 1

    if args.retrieval_only:
        all_records = []
        for case in selected_cases:
            records = case_runner(case, mode="retrieval")
            all_records.extend(records)

        summary = transfer_metrics.score(all_records)
        f2 = summary["f2"]
        sys.stdout.write("\nF2 Retrieval Ranks:\n")
        for cid, rank in f2["ranks"].items():
            sys.stdout.write(f"  {cid:<12} rank={rank}\n")

        status = "PASS" if f2["passed"] else "FAIL"
        sys.stdout.write(f"F2 {status}: {f2['top5_count']} of {f2['of']} cases in top 5\n")
        return 0 if f2["passed"] else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
