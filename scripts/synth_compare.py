# scripts/synth_compare.py
"""Run research_answer in both modes on the same ideas and report the
code-checkable depth criteria (spec §6). Live; not a unit test.

    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/synth_compare.py "idea one" "idea two"
"""
import json
import os
import re
import sys
import time

from research_assistant.config import DATA_DIR
from research_assistant.shared import retrieve
from research_assistant.shared.atomic import atomic_write_json

OUT_DIR = os.path.join(DATA_DIR, "eval", "synthesis")
HEADINGS = ("### What is established", "### Where the papers differ", "### The gap")


def criteria(out):
    text = out["suggestion"]
    used, _ = retrieve.check_citation_keys(text, set(out["keys"]))
    return {
        "mode": out["mode"], "papers_shortlisted": len(out["selected"]), "papers_cited": len([k for k in used if k in out["keys"]]),
        "words": len(text.split()), "headings_present": all(h in text for h in HEADINGS),
        "unverified": out["unverified_citations"], "irrelevant_cited": out["irrelevant_cited"],
        "not_relevant_notes": sum(1 for n in out["notes"] if not n["relevant"]), "timings": out["timings"],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idea in sys.argv[1:]:
        for mode in ("single", "map_reduce"):
            out = retrieve.research_answer(idea, mode=mode)
            c = criteria(out)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(OUT_DIR, f"{stamp}-{mode}-{re.sub(r'[^a-z0-9]+', '-', idea.lower())[:40]}.json")
            atomic_write_json(path, {"idea": idea, "criteria": c, "answer": out}, default=str)
            print(f"\n=== {mode} · {idea}\n{json.dumps(c, indent=1)}\n--- synthesis ---\n{out['suggestion']}\n(saved {path})")


if __name__ == "__main__":
    main()
