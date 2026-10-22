"""Rendering utilities for technique transfer reports and search trees."""

from __future__ import annotations

from research_assistant.transfer.models import Premise, RunResult, Transfer
from research_assistant.transfer.scoring import final_rows


def premise_badge(p: Premise) -> str:
    """Return badge for premise verification status in order of precedence."""
    if p.accepted:
        return "✅ Supports"
    if p.partial:
        return "🟡 Partially supports"
    if p.judgement == "Contradicts" or getattr(p, "contradicted", False):
        return "❌ Contradicts"
    if bool(p.error):
        return "⛔ Judge error"
    return "⚠️ Not accepted"


def mermaid_label(text: str, limit: int = 60) -> str:
    """Sanitize and truncate text for Mermaid flowchart labels."""
    cleaned = " ".join(text.split()).replace('"', "'")
    for ch in '#<>&`{}[]|':
        cleaned = cleaned.replace(ch, "")
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned


def render_tree(result: RunResult) -> str:
    """Render the technique transfer search tree as a Mermaid diagram."""
    lines = [
        "flowchart TD",
        '    classDef best fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    classDef aside fill:#f8d7da,stroke:#dc3545,stroke-width:1px,stroke-dasharray: 5 5;',
        '    R["Technique transfer"]',
    ]

    top_valid = next(
        (t for t in result.transfers if t.scores and t.scores.set_aside is None),
        None,
    )

    for b in result.bottlenecks:
        b_node = b.id
        b_label = mermaid_label(f"{b.id} · {b.statement}", 50)
        lines.append(f'    R --> {b_node}["{b_label}"]')

        b_transfers = [t for t in result.transfers if t.bottleneck_id == b.id]
        for t in b_transfers:
            t_node = f"T_{t.id.replace('.', '_')}"
            card_obj = result.cards.get(t.card_id)
            card_name = card_obj.name if card_obj else t.title
            rew_str = f"{t.scores.reward:.2f}" if t.scores else "0.00"
            t_label = mermaid_label(f"{card_name} · {rew_str}", 50)
            lines.append(f'    {b_node} --> {t_node}["{t_label}"]')

            if top_valid and t.id == top_valid.id:
                lines.append(f"    class {t_node} best")
            elif t.scores and t.scores.set_aside is not None:
                lines.append(f"    class {t_node} aside")

    return "\n".join(lines)


def render_markdown(result: RunResult) -> str:
    """Render full Markdown report for a technique transfer run."""
    lines = []

    # 1. Title
    lines.append(f"# Technique transfer: {result.query}\n")

    # 2. Models and calls
    gen_m = result.models.get("generator", "unknown")
    jud_m = result.models.get("judge", "unknown")
    crit_m = result.models.get("critic", "unknown")
    lines.append(
        f"**Models:** generator: `{gen_m}` | judge: `{jud_m}` | critic: `{crit_m}` | "
        f"**Calls:** {result.calls} | **Time:** {result.seconds}s\n"
    )

    # 3. Caveat
    lines.append("> Premises are checked against their papers; the mapping, steps and prediction are the tool's proposal.\n")

    # 4. Bottlenecks
    lines.append("## Bottlenecks\n")
    for b in result.bottlenecks:
        lines.append(f"### {b.id}: {b.statement}\n")
        lines.append(f"- **Current method:** {b.current_method}")
        lines.append(f"- **Quantities:** {', '.join(b.quantities)}")
        lines.append("- **Premises:**")
        for p in b.premises:
            lines.append(f"  - {premise_badge(p)} {p.claim} ({p.key}, {p.document})")
        lines.append("")

    # 5. Transfers in ranked order
    lines.append("## Proposed Transfers\n")
    for t in result.transfers:
        card_obj = result.cards.get(t.card_id)
        card_name = card_obj.name if card_obj else "unknown technique"
        card_cit = card_obj.citation if card_obj else t.card_document
        lines.append(f"### {t.id}: {t.title}\n")
        lines.append(f"**Technique:** {card_name} ({card_cit})\n")
        lines.append(f"**Proposal (not stated by any source):** {t.summary}\n")

        # Mapping table
        f_rows = final_rows(t)
        lines.append("| Source role | Target role | Assumption | Status | Core | Note |")
        lines.append("|---|---|---|---|---|---|")
        for r in f_rows:
            note_str = r.note or ""
            core_str = "core" if r.core else ""
            lines.append(f"| {r.source_role} | {r.target_role} | {r.assumption} | {r.status} | {core_str} | {note_str} |")
        lines.append("")

        # Identifiability
        lines.append("#### Identifiability")
        lines.append(f"- **Measurements:** {t.identifiability.measurements}")
        lines.append(f"- **Unknowns:** {t.identifiability.unknowns}")
        sym_str = ", ".join(t.identifiability.symmetries) if t.identifiability.symmetries else "none"
        lines.append(f"- **Symmetries:** {sym_str}")
        lines.append(f"- **Verdict:** {t.identifiability.verdict}")
        lines.append(f"- **Argument:** {t.identifiability.argument}\n")

        # Steps
        lines.append("#### Steps")
        for idx, s in enumerate(t.steps, start=1):
            lines.append(f"{idx}. **{s.action}:** input `{s.input}` → output `{s.output}` using `{s.tool}`")
        lines.append("")

        lines.append(f"- **Data requirement:** {t.data_requirement}")
        lines.append(f"- **Baseline:** {t.baseline}")
        lines.append(f"- **Metric:** {t.metric}")
        lines.append(f"- **Prediction:** {t.prediction}")
        lines.append(f"- **Refuting result:** {t.refuting_result}\n")

        # Premises
        lines.append("#### Premises")
        for p in t.premises:
            span_str = f' — *"{p.supporting_span}"*' if p.supporting_span else ""
            lines.append(f"- {premise_badge(p)} {p.claim} ({p.key}, {p.document}){span_str}")
        lines.append("")

        # Critique
        lines.append("#### Critique & Revision")
        lines.append(f"- **Problem shape match:** {t.critique.shape_match}")
        lines.append(f"- **Kind (vs the current method):** {t.critique.kind}")
        if t.critique.identifiability:
            lines.append(f"- **Critic identifiability override:** {t.critique.identifiability}")
        for o in t.critique.objections:
            ans_str = "answered" if o.get("answered") else "unanswered"
            conc_str = ", conceded" if o.get("conceded") else ""
            lines.append(f"- **Objection:** {o['text']}")
            lines.append(f"  - **Response ({ans_str}{conc_str}):** {o.get('response', '')}")
        lines.append("")

        # Prior art
        lines.append("#### Prior Art")
        lines.append(f"- **Status:** {t.prior_art.status}")
        for h in t.prior_art.hits:
            sim_str = f" (similarity: {h.get('similarity', 0.0):.2f})" if "similarity" in h else ""
            lines.append(f"  - [{h.get('title')}]({h.get('url')}){sim_str}")
        lines.append("")

    # 6. Set aside
    lines.append("## Set aside\n")
    aside_transfers = [t for t in result.transfers if t.scores and t.scores.set_aside is not None]
    if aside_transfers:
        for t in aside_transfers:
            lines.append(f"- **{t.id} ({t.title}):** set aside because `{t.scores.set_aside}`")
    else:
        lines.append("None.")
    lines.append("")

    # 7. Already done
    lines.append("## Already done\n")
    done_transfers = [t for t in result.transfers if t.prior_art and t.prior_art.status == "done_before"]
    if done_transfers:
        for t in done_transfers:
            lines.append(f"- **{t.id}:** {t.title}")
    else:
        lines.append("None.")
    lines.append("")

    # 8. Search tree
    lines.append("## Search tree\n")
    lines.append("```mermaid")
    lines.append(render_tree(result))
    lines.append("```\n")

    return "\n".join(lines)
