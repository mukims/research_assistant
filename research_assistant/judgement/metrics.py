"""Scoring for the judge evaluation. Pure; the harness collects results."""

from __future__ import annotations

import statistics as st
from collections import defaultdict

JUDGEMENTS = [
    "Supports",
    "Partially supports",
    "Contradicts",
    "Does not support",
    "Unclear / insufficient evidence",
]


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


def score(results: list[dict]) -> dict:
    errors = [r for r in results if r.get("error")]
    scored = [r for r in results if not r.get("error") and r.get("got") is not None]
    n = len(scored)
    strict = sum(1 for r in scored if r["got"] == r["case"]["expected_judgement"])
    lenient = sum(1 for r in scored if r["got"] in r["case"]["accept"])

    confusion = {e: {g: 0 for g in JUDGEMENTS} for e in JUDGEMENTS}
    for r in scored:
        confusion[r["case"]["expected_judgement"]][r["got"]] += 1
    per_class = {}
    for j in JUDGEMENTS:
        tp = confusion[j][j]
        expected_n = sum(confusion[j].values())
        predicted_n = sum(confusion[e][j] for e in JUDGEMENTS)
        per_class[j] = {
            "n": expected_n,
            "recall": (tp / expected_n) if expected_n else None,
            "precision": (tp / predicted_n) if predicted_n else None,
        }
    cd = {
        "contradicts_as_dns": confusion["Contradicts"]["Does not support"],
        "dns_as_contradicts": confusion["Does not support"]["Contradicts"],
    }

    def _group(key):
        groups = defaultdict(list)
        for r in scored:
            k = key(r)
            if k:
                groups[k].append(r)
        return {
            k: {
                "n": len(v),
                "strict": sum(1 for r in v if r["got"] == r["case"]["expected_judgement"]) / len(v),
                "lenient": sum(1 for r in v if r["got"] in r["case"]["accept"]) / len(v),
            }
            for k, v in groups.items()
        }

    verdicts = [r.get("verdict") or {} for r in scored]
    esc = [v.get("escalated") for v in verdicts]
    changed = [v.get("first_judgement") != v.get("judgement") for v in verdicts if v.get("escalated")]
    signals = {
        "rubric_mismatch": _rate([v.get("rubric_mismatch") for v in verdicts]),
        "span_unverified": _rate(
            [(v["span_verified"] is False) if "span_verified" in v else None for v in verdicts]
        ),
        "escalated": _rate(esc),
        "escalation_changed": (sum(changed) / len(changed)) if changed else None,
    }

    runs = {r["run"] for r in scored}
    stability = None
    if len(runs) > 1:
        by_id = defaultdict(set)
        for r in scored:
            by_id[r["case"]["id"]].add(r["got"])
        complete = [i for i in by_id if sum(1 for r in scored if r["case"]["id"] == i) == len(runs)]
        stability = (sum(1 for i in complete if len(by_id[i]) == 1) / len(complete)) if complete else None

    secs = sorted(r["seconds"] for r in scored) or [0.0]
    latency = {
        "median": st.median(secs),
        "p90": secs[min(len(secs) - 1, int(0.9 * (len(secs) - 1)))],
    }

    return {
        "n": n,
        "strict_acc": (strict / n) if n else None,
        "lenient_acc": (lenient / n) if n else None,
        "confusion": confusion,
        "per_class": per_class,
        "cd_confusion": cd,
        "by_source": _group(lambda r: r["case"]["source"]),
        "by_transform": _group(lambda r: r["case"].get("transform")),
        "signals": signals,
        "stability": stability,
        "latency": latency,
        "errors": len(errors),
    }


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render(summary: dict, refused: list[dict], in_prompt: dict | None = None) -> str:
    lines = [
        f"**Held-out:** n={summary['n']}  strict {_pct(summary['strict_acc'])}  lenient {_pct(summary['lenient_acc'])}"
        f"  errors {summary['errors']}  stability {_pct(summary['stability'])}"
        f"  latency median {summary['latency']['median']:.1f}s p90 {summary['latency']['p90']:.1f}s"
    ]
    if in_prompt:
        lines.append(f"**In-prompt examples (not held out):** n={in_prompt['n']} strict {_pct(in_prompt['strict_acc'])}")
    if refused:
        lines.append(f"**Refused (text appears in prompt.md):** {', '.join(c['id'] for c in refused)}")
    lines += ["", "| source / transform | n | strict | lenient |", "|---|---|---|---|"]
    for k, v in sorted(summary["by_source"].items()):
        lines.append(f"| {k} | {v['n']} | {_pct(v['strict'])} | {_pct(v['lenient'])} |")
    for k, v in sorted(summary["by_transform"].items()):
        lines.append(f"| ↳ {k} | {v['n']} | {_pct(v['strict'])} | {_pct(v['lenient'])} |")
    lines += [
        "",
        "| expected \\ got | " + " | ".join(j.split(" /")[0] for j in JUDGEMENTS) + " |",
        "|---|" + "---|" * len(JUDGEMENTS),
    ]
    for e in JUDGEMENTS:
        lines.append(f"| {e.split(' /')[0]} | " + " | ".join(str(summary["confusion"][e][g]) for g in JUDGEMENTS) + " |")
    cd = summary["cd_confusion"]
    lines += [
        "",
        f"Contradicts→Does-not-support: {cd['contradicts_as_dns']}   Does-not-support→Contradicts: {cd['dns_as_contradicts']}",
    ]
    s = summary["signals"]
    lines.append(
        f"rubric_mismatch {_pct(s['rubric_mismatch'])}  span_unverified {_pct(s['span_unverified'])}  "
        f"escalated {_pct(s['escalated'])}  escalation changed verdict {_pct(s['escalation_changed'])}"
    )
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    sa, sb = a["summary"], b["summary"]
    lines = [
        f"strict {_pct(sa['strict_acc'])} → {_pct(sb['strict_acc'])}   lenient {_pct(sa['lenient_acc'])} → {_pct(sb['lenient_acc'])}"
        f"   median latency {sa['latency']['median']:.1f}s → {sb['latency']['median']:.1f}s",
        "",
        "| case | A | B |",
        "|---|---|---|",
    ]
    ga = {r["id"]: r["got"] for r in a["results"]}
    for r in b["results"]:
        if r["id"] in ga and ga[r["id"]] != r["got"]:
            lines.append(f"| {r['id']} | {ga[r['id']]} | {r['got']} |")
    return "\n".join(lines)
