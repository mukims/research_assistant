"""Label harvested candidates. The operator's judgement, one keystroke each.

    PYTHONPATH=. python scripts/judge_label.py

Shows claim and evidence; asks for s/p/c/d/u (x skips, q quits); for a
Supports, offers to record a negated claim (becomes a Contradicts case);
then reveals what the judge said at harvest time. Every label is written
immediately to research_assistant/judgement/cases/human.jsonl.
"""

import argparse
import json
import os
import textwrap

from research_assistant.config import DATA_DIR
from research_assistant.judgement import labeling as lb
from research_assistant.judgement.evalset import CASES_DIR, load_cases
from research_assistant.shared.atomic import atomic_write

DEFAULT_CANDIDATES = os.path.join(DATA_DIR, "eval", "judge", "candidates.jsonl")
DEFAULT_OUT = str(CASES_DIR / "human.jsonl")


def _write_cases(path, cases):
    with atomic_write(path) as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.candidates, encoding="utf-8") as fh:
        candidates = [json.loads(l) for l in fh if l.strip()]
    cases = load_cases([args.out])
    todo = lb.unlabelled(candidates, cases)
    print(f"{len(todo)} candidate(s) to label, {len(cases)} already in {args.out}\n")

    for i, cand in enumerate(todo, 1):
        print("=" * 78)
        print(f"[{i}/{len(todo)}] {cand['id']}  ({cand['source']}; {cand.get('citation_source') or '?'})\n")
        print("CLAIM:\n" + textwrap.fill(cand["claim"], 78) + "\n")
        print("EVIDENCE:\n" + textwrap.fill(cand["citation_evidence"], 78) + "\n")
        key = input(
            "verdict [s=Supports p=Partially c=Contradicts d=Does not support u=Unclear | x=skip q=quit]: "
        ).strip().lower()
        if key == "q":
            break
        if key == "x" or key not in lb.KEYS:
            print("skipped\n")
            continue
        negation = input("negated claim for a Contradicts case (Enter to skip): ").strip() if key == "s" else ""
        note = input("note (Enter to skip): ").strip()
        new = lb.apply_label(cand, key, {c["id"] for c in cases}, note=note, negation=negation)
        cases.extend(new)
        _write_cases(args.out, cases)
        hidden = (cand.get("hidden") or {}).get("model_judgement")
        print(f"saved {', '.join(c['id'] for c in new)}   (the judge said at harvest: {hidden or 'n/a'})\n")

    print(f"{len(cases)} case(s) in {args.out}")


if __name__ == "__main__":
    main()
