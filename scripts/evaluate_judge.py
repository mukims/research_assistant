"""Evaluate Agent 8's judge on the held-out set.

    PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/evaluate_judge.py --mode judge --runs 1
    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/evaluate_judge.py --mode verifier
    PYTHONPATH=. python scripts/evaluate_judge.py --compare data/eval/judge/results/A.json data/eval/judge/results/B.json

judge mode: judge(claim, evidence, context) on each case's stored evidence;
--context picks what the judge sees besides them: 'window' (default — the
sentence window, as the audit sends it), 'full' (adds the section breadcrumb
and figure captions; lost its gate in 25d6fd9), or 'none'.
verifier mode: a one-sentence draft per case through verify_draft() against
the live index (needs `document` and `citation_source` on the case).
In-prompt cases (cases.jsonl) are reported separately and never in the
headline; any case whose text appears in prompt.md is refused.
"""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime

from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.config import DATA_DIR
from research_assistant.judgement.evalset import DEFAULT_CASE_FILES, load_cases, split_held_out
from research_assistant.judgement.judge import PROMPT_TEMPLATE, compose_context, judge
from research_assistant.judgement.metrics import compare, render, score
from research_assistant.shared.atomic import atomic_write_json

RESULTS_DIR = os.path.join(DATA_DIR, "eval", "judge", "results")


def _context_for(case, context_mode):
    """What the judge is shown besides claim and evidence: nothing, the
    sentence window the case was harvested with, or the full block the
    audit builds (section, captions, window)."""
    if context_mode == "none":
        return None
    if context_mode == "window":
        return case.get("context")
    return compose_context(case.get("context"), section=case.get("section_heading"), artifacts=case.get("artifacts"))


def _judge_one(case, mode, context_mode="window"):
    if mode == "judge":
        verdict = judge(case["claim"], case["citation_evidence"], context=_context_for(case, context_mode))
        return verdict["judgement"], verdict
    with tempfile.TemporaryDirectory() as d:
        draft = os.path.join(d, "draft.txt")
        with open(draft, "w", encoding="utf-8") as fh:
            fh.write(f"{case['claim']} \\cite{{cite_1}}.")
        with open(os.path.join(d, "draft_citations.json"), "w", encoding="utf-8") as fh:
            json.dump({case["citation_source"]: "cite_1"}, fh)
        report = verify_draft(draft)
    entry = report["results"][0]
    if entry.get("outcome") != "judged":
        raise RuntimeError(f"verifier outcome {entry.get('outcome')}: {entry.get('raw', '')}")
    return entry["judgement"], entry


def run(cases, mode="judge", runs=1, limit=None, context_mode="window"):
    held, refused = split_held_out(cases, PROMPT_TEMPLATE)
    in_prompt = [c for c in refused if c["source"] == "prompt_example"]
    refused = [c for c in refused if c["source"] != "prompt_example"]
    skipped_no_paper = 0
    if mode == "verifier":
        keep = [c for c in held if c.get("document") and c.get("citation_source")]
        skipped_no_paper = len(held) - len(keep)
        held = keep
    if limit:
        held = held[:limit]

    results, prompt_results = [], []
    for run_no in range(1, runs + 1):
        for case in held + in_prompt:
            t0 = time.perf_counter()
            got, verdict, error = None, None, None
            try:
                got, verdict = _judge_one(case, mode, context_mode)
            except Exception as exc:  # noqa: BLE001 — recorded, never fatal
                error = f"{type(exc).__name__}: {exc}"
            rec = {
                "case": case,
                "got": got,
                "verdict": verdict,
                "seconds": time.perf_counter() - t0,
                "run": run_no,
                "error": error,
            }
            (prompt_results if case["source"] == "prompt_example" else results).append(rec)
            print(
                f"run {run_no} {case['id']:8s} {rec['seconds']:6.1f}s  expected={case['expected_judgement']!r} got={got!r}{'  ERROR ' + error if error else ''}"
            )

    summary = score(results)
    in_prompt_summary = score(prompt_results) if prompt_results else None
    return {
        "mode": mode,
        "context_mode": context_mode,
        "runs": runs,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "in_prompt": in_prompt_summary,
        "refused": refused,
        "skipped_no_paper": skipped_no_paper,
        "results": [
            {
                "id": r["case"]["id"],
                "source": r["case"]["source"],
                "transform": r["case"].get("transform"),
                "expected": r["case"]["expected_judgement"],
                "got": r["got"],
                "run": r["run"],
                "seconds": round(r["seconds"], 2),
                "error": r["error"],
                "verdict": r["verdict"],
            }
            for r in results + prompt_results
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["judge", "verifier"], default="judge")
    ap.add_argument(
        "--context",
        choices=["none", "window", "full"],
        default="window",
        help="what the judge sees besides claim and evidence (judge mode); "
             "'window' is what the audit sends, 'full' adds the section breadcrumb and figure captions",
    )
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", action="append", default=None, help="restrict to these sources (repeatable)")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as fa, open(args.compare[1]) as fb:
            print(compare(json.load(fa), json.load(fb)))
        return

    cases = load_cases(args.cases or DEFAULT_CASE_FILES)
    if args.source:
        cases = [c for c in cases if c["source"] in set(args.source) or c["source"] == "prompt_example"]
    out = run(cases, mode=args.mode, runs=args.runs, limit=args.limit, context_mode=args.context)
    print("\n" + render(out["summary"], out["refused"], out["in_prompt"]))
    if out["skipped_no_paper"]:
        print(f"(verifier mode skipped {out['skipped_no_paper']} case(s) with no paper reference)")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.mode}-{args.context}.json")
    atomic_write_json(path, out)
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
