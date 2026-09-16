"""Label harvested candidates. The operator's judgement, one keystroke each.

    PYTHONPATH=. python scripts/judge_label.py

Shows claim and evidence; asks for s/p/c/d/u (x skips, q quits); for a
Supports, offers to record a negated claim (becomes a Contradicts case);
then reveals what the judge said at harvest time. Every label is written
immediately to research_assistant/judgement/cases/human.jsonl.

The label is a verdict on THESE PASSAGES, not on the cited paper. A pass on
2026-09-16 lost that distinction: six of nineteen labels overrode the judge
and every one turned "Unclear" into a decisive verdict, three of them into
Supports against passages that never mentioned the claim's subject. A set
built that way trains and scores the judge to stop hedging when it should.
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


RULE = """\
Label the EVIDENCE, not the paper.
  The question is whether the passages shown support the claim — not whether
  the cited paper, somewhere, does. If the passages are off-topic or too
  fragmentary to tell, that is u (Unclear), even when you are confident the
  paper would back the claim. "Does not support" says these passages report
  the opposite or are silent on a matter they clearly cover; it does not mean
  "the claim's subject is missing here" — that is u as well.
  When in doubt, u. x skips.
"""


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
    print(RULE)

    for i, cand in enumerate(todo, 1):
        print("=" * 78)
        print(f"[{i}/{len(todo)}] {cand['id']}  ({cand['source']}; {cand.get('citation_source') or '?'})\n")
        print("CLAIM:\n" + textwrap.fill(cand["claim"], 78) + "\n")
        print("EVIDENCE:\n" + textwrap.fill(cand["citation_evidence"], 78) + "\n")
        if cand.get("context"):
            print("CONTEXT (what the judge saw around the claim):\n" + textwrap.fill(cand["context"], 78) + "\n")
        key = input(
            "Does THIS EVIDENCE support the claim? (not whether the paper does)\n"
            "verdict [s=Supports p=Partially c=Contradicts d=Does not support u=Unclear | x=skip q=quit]: "
        ).strip().lower()
        if key == "q":
            break
        if key == "x" or key not in lb.KEYS:
            print("skipped\n")
            continue
        negation = input("negated claim for a Contradicts case (Enter to skip): ").strip() if key == "s" else ""
        note = input("note (Enter to skip): ").strip()
        tags = [t.strip() for t in input("tags, comma-separated — figure_ref, method_transfer, … (Enter to skip): ").split(",") if t.strip()]
        new = lb.apply_label(cand, key, {c["id"] for c in cases}, note=note, negation=negation, tags=tags)
        cases.extend(new)
        _write_cases(args.out, cases)
        hidden = (cand.get("hidden") or {}).get("model_judgement")
        print(f"saved {', '.join(c['id'] for c in new)}   (the judge said at harvest: {hidden or 'n/a'})\n")

    print(f"{len(cases)} case(s) in {args.out}")


if __name__ == "__main__":
    main()
