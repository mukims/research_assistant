"""Scoring for the citer evaluation. Pure; the harness collects results.

A result is {case, needs_cite, cite, run, error}; ``cite`` is the
dataclasses.asdict of a CiteResult, or None when nothing was attempted.
"""

from __future__ import annotations

from collections import Counter, defaultdict

FLOOR_NOTE = (
    "Author citations are a floor, not truth: a correct paper the author did not cite scores as "
    "wrong, and an uncited sentence may merely be under-cited. Compare runs against each other; "
    "do not read an absolute number as the citer's accuracy."
)

HEADLINE = ("need_recall", "need_specificity", "target_precision", "target_recall", "sentence_hit", "end_to_end")


def cited_documents(cite: dict | None) -> set:
    """The documents behind the keys that actually made it into the sentence."""
    if not cite:
        return set()
    keys = set(cite.get("keys") or [])
    return {c["document"] for c in (cite.get("candidates") or [])
            if c.get("key") in keys and c.get("document")}


def _rate(num, den):
    return (num / den) if den else None


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def score(results: list[dict]) -> dict:
    scored = [r for r in results if not r.get("error")]
    cited = [r for r in scored if r["case"]["kind"] == "cited"]
    uncited = [r for r in scored if r["case"]["kind"] == "uncited"]

    made = [r for r in cited if r["needs_cite"] and cited_documents(r.get("cite"))]
    hit_ids = {id(r) for r in made if cited_documents(r["cite"]) & set(r["case"]["author_documents"])}
    inter = sum(len(cited_documents(r["cite"]) & set(r["case"]["author_documents"])) for r in made)
    n_cited_docs = sum(len(cited_documents(r["cite"])) for r in made)
    n_author_docs = sum(len(set(r["case"]["author_documents"])) for r in cited)
    declined = [r for r in cited if r["needs_cite"] and not cited_documents(r.get("cite"))]

    def _group(keys):
        groups = defaultdict(list)
        for r in cited:
            for k in keys(r):
                if k:
                    groups[k].append(r)
        return {k: {"n": len(v), "end_to_end": _rate(sum(1 for r in v if id(r) in hit_ids), len(v))}
                for k, v in groups.items()}

    runs = {r["run"] for r in cited}
    stability = None
    if len(runs) > 1:
        outcome = defaultdict(dict)
        for r in cited:
            outcome[r["case"]["id"]][r["run"]] = id(r) in hit_ids
        complete = [o for o in outcome.values() if len(o) == len(runs)]
        stability = _rate(sum(1 for o in complete if len(set(o.values())) == 1), len(complete))

    return {
        "n_cited": len(cited),
        "n_uncited": len(uncited),
        "need_recall": _rate(sum(1 for r in cited if r["needs_cite"]), len(cited)),
        "need_specificity": _rate(sum(1 for r in uncited if not r["needs_cite"]), len(uncited)),
        "target_precision": _rate(inter, n_cited_docs),
        "target_recall": _rate(inter, n_author_docs),
        "sentence_hit": _rate(len(hit_ids), len(made)),
        "end_to_end": _rate(len(hit_ids), len(cited)),
        "declined": {
            "n": len(declined),
            "reasons": dict(Counter((r.get("cite") or {}).get("skip_reason") or "no attempt" for r in declined)),
        },
        "by_role": _group(lambda r: set(r["case"].get("roles") or [])),
        "by_section": _group(lambda r: [r["case"].get("section")]),
        "stability": stability,
        "errors": len(results) - len(scored),
    }


def render(summary: dict) -> str:
    lines = [
        f"_{FLOOR_NOTE}_",
        "",
        f"**Cited sentences:** n={summary['n_cited']}   **uncited (negatives):** n={summary['n_uncited']}"
        f"   errors {summary['errors']}   stability {_pct(summary['stability'])}",
        "",
        "| metric | value |", "|---|---|",
    ]
    for k in HEADLINE:
        lines.append(f"| {k} | {_pct(summary[k])} |")
    d = summary["declined"]
    lines += ["", f"**Needed a citation, none made:** {d['n']}"]
    for reason, n in sorted(d["reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"- {n} × {reason}")
    lines += ["", "| role / section | n | end_to_end |", "|---|---|---|"]
    for k, v in sorted(summary["by_role"].items()):
        lines.append(f"| {k} | {v['n']} | {_pct(v['end_to_end'])} |")
    for k, v in sorted(summary["by_section"].items()):
        lines.append(f"| § {k} | {v['n']} | {_pct(v['end_to_end'])} |")
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    sa, sb = a["summary"], b["summary"]
    lines = [f"_{FLOOR_NOTE}_", ""]
    for k in HEADLINE:
        lines.append(f"{k:18} {_pct(sa[k]):>5} → {_pct(sb[k]):>5}")
    lines.append(f"{'declined':18} {sa['declined']['n']:>5} → {sb['declined']['n']:>5}")
    for tag in sorted(set(sa["by_role"]) | set(sb["by_role"])):
        ta, tb = sa["by_role"].get(tag), sb["by_role"].get(tag)
        lines.append(f"# {tag}: end_to_end {_pct(ta['end_to_end'] if ta else None)} → "
                     f"{_pct(tb['end_to_end'] if tb else None)}  (n {ta['n'] if ta else 0} → {tb['n'] if tb else 0})")
    return "\n".join(lines)
