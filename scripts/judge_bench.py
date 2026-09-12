# scripts/judge_bench.py
"""Time and check the judge on the regression cases.

    PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/judge_bench.py [--runs N]

Prints per-case seconds and pass/fail against expected_judgement, then the
median over calls 2..end (call 1 is a cold prompt cache). Live model call;
not a unit test.
"""
import argparse, json, statistics as st, time
from pathlib import Path

from research_assistant.judgement.judge import judge, JudgementParseError

CASES = Path("research_assistant/judgement/cases/cases.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    times, passes = [], 0
    for run in range(args.runs):
        for c in cases:
            t0 = time.perf_counter()
            try:
                v = judge(c["claim"], c["citation_evidence"])
                got = v["judgement"]
            except JudgementParseError as exc:
                got = f"PARSE_FAIL ({exc})"
            secs = time.perf_counter() - t0
            times.append(secs)
            ok = got == c["expected_judgement"]
            passes += ok
            print(f"run {run+1} {c['id']:8s} {secs:6.1f}s  {'PASS' if ok else 'FAIL'}  expected={c['expected_judgement']!r} got={got!r}")
    warm = times[1:] or times
    print(f"\npass {passes}/{len(times)}   first call {times[0]:.1f}s   median of the rest {st.median(warm):.1f}s   max {max(warm):.1f}s")


if __name__ == "__main__":
    main()
