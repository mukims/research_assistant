"""Pure evaluation metrics and scoring functions for technique transfer retrodiction."""

from __future__ import annotations

from typing import Any

CAVEAT_NOTE = (
    "A positive case counts only when the transfer's technique card came from an expected "
    "source paper; naming the technique is not enough. Prior art is limited to before the case's cutoff."
)


def _get_case_field(case: Any, field_name: str, default: Any = None) -> Any:
    """Retrieve field from Case instance, dict, or nested expected/papers mapping."""
    if hasattr(case, field_name):
        val = getattr(case, field_name)
        if val is not None:
            return val
    if isinstance(case, dict) and field_name in case:
        return case[field_name]
    expected = (
        getattr(case, "expected", None)
        if hasattr(case, "expected")
        else (case.get("expected") if isinstance(case, dict) else None)
    )
    if isinstance(expected, dict) and field_name in expected:
        return expected[field_name]
    papers = (
        getattr(case, "papers", None)
        if hasattr(case, "papers")
        else (case.get("papers") if isinstance(case, dict) else None)
    )
    if isinstance(papers, dict) and field_name in papers:
        return papers[field_name]
    return default


def expected_documents(case: Any, resolved: dict[str, str]) -> set[str]:
    """Return set of resolved document filenames for the case's expected source papers."""
    source_titles = _get_case_field(case, "source_titles") or _get_case_field(case, "source") or []
    return {resolved[t] for t in source_titles if t in resolved}


def matches(transfer: Any, cards: dict, case: Any, expected_docs: set[str]) -> bool:
    """
    Check if a transfer matches the expected source documents and technique terms.

    Matches only when the transfer's card_document is in expected_docs and any of
    technique_terms appears, case-insensitively, in the card's name or problem_solved.
    The transfer's own prose is not consulted.
    """
    doc = (
        transfer.get("card_document")
        if isinstance(transfer, dict)
        else getattr(transfer, "card_document", "")
    )
    if doc not in expected_docs:
        return False

    card_name = (
        transfer.get("card_name", "")
        if isinstance(transfer, dict)
        else getattr(transfer, "card_name", "")
    )
    prob_solved = (
        transfer.get("problem_solved", "")
        if isinstance(transfer, dict)
        else getattr(transfer, "problem_solved", "")
    )
    card_id = (
        transfer.get("card_id")
        if isinstance(transfer, dict)
        else getattr(transfer, "card_id", None)
    )

    if cards and card_id in cards:
        c = cards[card_id]
        c_name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else "")
        c_prob = getattr(c, "problem_solved", None) or (c.get("problem_solved") if isinstance(c, dict) else "")
        card_name = card_name or c_name
        prob_solved = prob_solved or c_prob

    combined = f"{card_name} {prob_solved}".lower()
    technique_terms = _get_case_field(case, "technique_terms", [])
    if not technique_terms:
        return True
    return any(term.lower() in combined for term in technique_terms)


def found_rank(result: Any, case: Any, expected_docs: set[str]) -> int | None:
    """
    Return 1-based rank among transfers that were not set aside.

    Transfers set aside are ignored during ranking.
    """
    transfers = (
        result.get("transfers", [])
        if isinstance(result, dict)
        else getattr(result, "transfers", [])
    )
    cards = (
        result.get("cards", {})
        if isinstance(result, dict)
        else getattr(result, "cards", {})
    )

    rank = 0
    for t in transfers:
        sa = None
        if isinstance(t, dict):
            sa = t.get("set_aside")
            if not sa and isinstance(t.get("scores"), dict):
                sa = t["scores"].get("set_aside")
        else:
            sa = getattr(t, "set_aside", None)
            if not sa and getattr(t, "scores", None):
                sa = getattr(t.scores, "set_aside", None)

        if sa:
            continue

        rank += 1
        if matches(t, cards, case, expected_docs):
            return rank

    return None


def mapped_share(transfer: Any, case: Any) -> float:
    """Return share of mapping_terms found in the text of the mapping rows."""
    mapping_terms = _get_case_field(case, "mapping_terms", [])
    if not mapping_terms:
        return 1.0

    mapping = (
        transfer.get("mapping", [])
        if isinstance(transfer, dict)
        else getattr(transfer, "mapping", [])
    )
    row_texts: list[str] = []
    for row in mapping:
        if isinstance(row, dict):
            for k in ("source_role", "target_role", "assumption", "note", "status"):
                v = row.get(k)
                if isinstance(v, str):
                    row_texts.append(v)
        else:
            for k in ("source_role", "target_role", "assumption", "note", "status"):
                v = getattr(row, k, None)
                if isinstance(v, str):
                    row_texts.append(v)

    combined = " ".join(row_texts).lower()
    found = sum(1 for term in mapping_terms if term.lower() in combined)
    return found / len(mapping_terms)


def critiqued(transfer: Any, case: Any) -> bool:
    """Return whether any of critic_terms appears in the critique's objections or row reasons."""
    critic_terms = _get_case_field(case, "critic_terms", [])
    if not critic_terms:
        return True

    critique = (
        transfer.get("critique", {})
        if isinstance(transfer, dict)
        else getattr(transfer, "critique", None)
    )
    if not critique:
        return False

    critic_texts: list[str] = []
    objections = (
        critique.get("objections", [])
        if isinstance(critique, dict)
        else getattr(critique, "objections", [])
    )
    for obj in objections:
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str):
                    critic_texts.append(v)
        elif isinstance(obj, str):
            critic_texts.append(obj)

    rows = (
        critique.get("rows", [])
        if isinstance(critique, dict)
        else getattr(critique, "rows", [])
    )
    for r in rows:
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, str):
                    critic_texts.append(v)
        elif isinstance(r, str):
            critic_texts.append(r)

    ident = (
        critique.get("identifiability", "")
        if isinstance(critique, dict)
        else getattr(critique, "identifiability", "")
    )
    if isinstance(ident, str):
        critic_texts.append(ident)

    combined = " ".join(critic_texts).lower()
    return any(term.lower() in combined for term in critic_terms)


def case_passed(result: Any, case: Any, expected_docs: set[str]) -> dict[str, Any]:
    """
    Evaluate whether a retrodiction case passed per spec §6 rules.

    Returns dict with keys: passed, found_rank, mapped, critiqued, reason.
    """
    kind = _get_case_field(case, "kind", "positive")
    technique_terms = _get_case_field(case, "technique_terms", [])
    transfers = (
        result.get("transfers", [])
        if isinstance(result, dict)
        else getattr(result, "transfers", [])
    )
    cards = (
        result.get("cards", {})
        if isinstance(result, dict)
        else getattr(result, "cards", {})
    )

    if kind == "negative":
        expected_set_aside = _get_case_field(case, "set_aside")
        matching_transfers = []
        for t in transfers:
            card_name = (
                t.get("card_name", "")
                if isinstance(t, dict)
                else getattr(t, "card_name", "")
            )
            prob = (
                t.get("problem_solved", "")
                if isinstance(t, dict)
                else getattr(t, "problem_solved", "")
            )
            card_id = (
                t.get("card_id")
                if isinstance(t, dict)
                else getattr(t, "card_id", None)
            )
            if cards and card_id in cards:
                c = cards[card_id]
                c_name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else "")
                c_prob = getattr(c, "problem_solved", None) or (c.get("problem_solved") if isinstance(c, dict) else "")
                card_name = card_name or c_name
                prob = prob or c_prob

            text = f"{card_name} {prob}".lower()
            if any(term.lower() in text for term in technique_terms):
                matching_transfers.append(t)

        if not matching_transfers:
            return {
                "passed": False,
                "found_rank": None,
                "mapped": 0.0,
                "critiqued": False,
                "reason": "no transfer matched technique terms",
            }

        reasons_found = []
        all_set_aside = True
        for t in matching_transfers:
            sa = None
            if isinstance(t, dict):
                sa = t.get("set_aside") or (
                    t.get("scores", {}).get("set_aside")
                    if isinstance(t.get("scores"), dict)
                    else None
                )
            else:
                sa = getattr(t, "set_aside", None) or (
                    t.scores.set_aside if getattr(t, "scores", None) else None
                )
            if not sa:
                all_set_aside = False
            else:
                reasons_found.append(sa)

        if not all_set_aside:
            return {
                "passed": False,
                "found_rank": None,
                "mapped": 0.0,
                "critiqued": False,
                "reason": "at least one matching transfer was not set aside",
            }

        if expected_set_aside and expected_set_aside not in reasons_found:
            return {
                "passed": False,
                "found_rank": None,
                "mapped": 0.0,
                "critiqued": False,
                "reason": f"expected set_aside '{expected_set_aside}' not among reasons: {reasons_found}",
            }

        return {
            "passed": True,
            "found_rank": None,
            "mapped": 0.0,
            "critiqued": False,
            "reason": f"all matching transfers set aside with expected reason '{expected_set_aside}'",
        }

    # Positive case
    rank = found_rank(result, case, expected_docs)
    if rank is None:
        return {
            "passed": False,
            "found_rank": None,
            "mapped": 0.0,
            "critiqued": False,
            "reason": "expected transfer not found or set aside",
        }

    # Locate the matching non-set-aside transfer at rank
    matching_t = None
    curr_rank = 0
    for t in transfers:
        sa = None
        if isinstance(t, dict):
            sa = t.get("set_aside") or (
                t.get("scores", {}).get("set_aside")
                if isinstance(t.get("scores"), dict)
                else None
            )
        else:
            sa = getattr(t, "set_aside", None) or (
                t.scores.set_aside if getattr(t, "scores", None) else None
            )
        if sa:
            continue
        curr_rank += 1
        if curr_rank == rank:
            matching_t = t
            break

    mapped = mapped_share(matching_t, case)
    crit = critiqued(matching_t, case)
    passed = (rank <= 3) and (mapped >= 0.5) and crit

    reasons = []
    if rank > 3:
        reasons.append(f"rank {rank} > 3")
    if mapped < 0.5:
        reasons.append(f"mapped {mapped:.1%} < 50%")
    if not crit:
        reasons.append("critic terms not hit")

    return {
        "passed": passed,
        "found_rank": rank,
        "mapped": mapped,
        "critiqued": crit,
        "reason": "top 3, mapped >= 50%, critiqued" if passed else ", ".join(reasons),
    }


def retrieval_rank(seeds: list[dict], expected_docs: set[str]) -> int | None:
    """Return 1-based rank of the first seed matching expected_docs, or None."""
    # Ensure seeds sorted descending by score if present
    sorted_seeds = sorted(seeds, key=lambda s: s.get("score", 0.0), reverse=True)
    for idx, s in enumerate(sorted_seeds, start=1):
        doc = (
            s.get("card_document")
            if isinstance(s, dict)
            else getattr(s, "card_document", None)
        )
        if doc in expected_docs:
            return idx
    return None


def leak_guard(case: Any, resolved: dict[str, str], corpus_documents: set[str]) -> None:
    """
    Ensure no hidden benchmark paper is present in the active corpus.

    Raises RuntimeError naming the paper if any hidden title resolves to a corpus document.
    """
    hidden_titles = _get_case_field(case, "hidden") or []
    for title in hidden_titles:
        if title in resolved:
            doc = resolved[title]
            if doc in corpus_documents:
                raise RuntimeError(
                    f"Leak guard: hidden paper '{title}' ({doc}) is present in the corpus!"
                )


def score(records: list[dict] | dict) -> dict[str, Any]:
    """
    Compute aggregate evaluation summary across retrodiction cases and runs.

    Returns dict with keys: cases, f2, f3, note.
    """
    grouped: dict[str, list[dict]] = {}
    if isinstance(records, list):
        for rec in records:
            cid = rec.get("case_id") or rec.get("id") or "default"
            grouped.setdefault(cid, []).append(rec)
    elif isinstance(records, dict):
        for k, v in records.items():
            if isinstance(v, list):
                grouped[k] = v
            elif isinstance(v, dict):
                grouped[k] = [v]
            elif isinstance(v, (int, float)) or v is None:
                grouped[k] = [{"case_id": k, "retrieval_rank": v}]

    # Gate F2 summary
    ranks_f2 = {}
    for cid, runs in grouped.items():
        for r in runs:
            if "retrieval_rank" in r:
                ranks_f2[cid] = r["retrieval_rank"]
                break

    if ranks_f2:
        top5_count = sum(1 for r in ranks_f2.values() if r is not None and 1 <= r <= 5)
        total_f2 = len(ranks_f2)
        # F2 passes when top 5 for >= 3 of 4 cases (or all cases if < 4)
        f2_passed = (top5_count >= 3) if total_f2 >= 4 else (top5_count == total_f2 and total_f2 > 0)
        f2_summary = {
            "passed": f2_passed,
            "top5_count": top5_count,
            "of": total_f2,
            "ranks": ranks_f2,
        }
    else:
        f2_summary = {"passed": False, "top5_count": 0, "of": 0, "ranks": {}}

    # Gate F3 summary
    differences = []
    passed_cases_count = 0
    total_f3_cases = 0

    for cid, runs in grouped.items():
        f3_runs = [r for r in runs if "passed" in r or "found_rank" in r]
        if not f3_runs:
            continue
        total_f3_cases += 1
        if all(r.get("passed", False) for r in f3_runs):
            passed_cases_count += 1

        if len(f3_runs) >= 2:
            r1 = f3_runs[0].get("found_rank")
            r2 = f3_runs[1].get("found_rank")
            v1 = 10 if r1 is None else r1
            v2 = 10 if r2 is None else r2
            differences.append(abs(v1 - v2))

    spread = (sum(differences) / len(differences)) if differences else 0.0
    f3_summary = {
        "passed": passed_cases_count,
        "of": total_f3_cases,
        "spread": spread,
        "all_passed": (passed_cases_count == total_f3_cases and total_f3_cases > 0),
    }

    return {
        "cases": grouped,
        "f2": f2_summary,
        "f3": f3_summary,
        "note": CAVEAT_NOTE,
    }
