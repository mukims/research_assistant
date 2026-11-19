#!/usr/bin/env python3
"""CLI for unattended technique transfer and candidate card inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.shared import pipeline_status
from research_assistant.transfer import engine, scout, stages, store
from research_assistant.transfer.deps import TransferSettings
from research_assistant.transfer.premises import PremiseChecker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Find cross-disciplinary technique transfers for a research question.")
    parser.add_argument("--query", "-q", required=True, help="Research question or problem statement.")
    parser.add_argument("--source-topic", "-s", default="", help="Optional source discipline or second topic to bridge techniques from.")
    parser.add_argument("--scout", action="store_true", help="Enable active LLM-guided discovery (propose external techniques & fetch missing OA papers/cards).")
    parser.add_argument("--scout-max-papers", type=int, default=config.TRANSFER_SCOUT_MAX_PAPERS, help="Max open-access papers to fetch per scout phase.")
    parser.add_argument("--bottlenecks", type=int, default=config.TRANSFER_BOTTLENECKS, help="Max bottlenecks to identify.")
    parser.add_argument("--techniques", type=int, default=config.TRANSFER_TECHNIQUES_PER_BOTTLENECK, help="Max technique cards per bottleneck.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Documents to exclude from transfer sources and near set.")
    parser.add_argument("--before-year", type=int, default=None, help="Prior art cutoff year.")
    parser.add_argument("--show-cards", action="store_true", help="Show candidate cards and generic bottlenecks without writing transfers.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save runs.")

    args = parser.parse_args(argv)

    # 1. Guard check: exit 2 if busy before loading anything
    try:
        engine.ensure_idle()
    except engine.BusyError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    # Only past the busy guard: a cancel pending against another live run is not stale,
    # and clearing it here would un-stop that run. With no run active, it is.
    pipeline_status.clear_cancel_request()

    settings = TransferSettings(
        bottlenecks=args.bottlenecks,
        techniques_per_bottleneck=args.techniques,
        exclude_documents=frozenset(args.exclude),
        prior_art_before_year=args.before_year,
        source_topic=args.source_topic.strip(),
        enable_scout=bool(args.scout),
        scout_max_papers=args.scout_max_papers,
    )

    deps = engine.live_deps()

    if args.show_cards:
        checker = PremiseChecker(deps.judge, deps.texts, deps.metadatas, chunk_store=deps.chunk_store)
        try:
            grounding = stages.ground(args.query, deps, settings, checker)
        except stages.CorpusCoverageError as exc:
            if not settings.enable_scout:
                sys.stderr.write(f"{exc}\n")
                return 3
            t_docs, fb_grounding, _ = scout.scout_target_topic(
                args.query, deps, settings, existing_docs=scout.known_corpus_documents(deps)
            )
            grounding = None
            if t_docs:
                deps = scout.reload_live_deps(deps)
                checker = PremiseChecker(deps.judge, deps.texts, deps.metadatas, chunk_store=deps.chunk_store)
                try:
                    grounding = stages.ground(args.query, deps, settings, checker)
                except stages.CorpusCoverageError:
                    grounding = None
            if grounding is None:
                if fb_grounding is not None:
                    grounding = fb_grounding
                else:
                    sys.stderr.write(f"{exc}\n")
                    return 3

        leads = []
        if settings.enable_scout or settings.source_topic:
            leads, _ = scout.propose_technique_leads(args.query, grounding, deps, settings)
            if settings.enable_scout and leads:
                deps, _, _, _ = scout.materialize_leads_into_cards(
                    leads, deps, settings, existing_docs=scout.known_corpus_documents(deps)
                )

        seeds, warnings = stages.candidate_cards(grounding, deps, settings, leads=leads)

        sys.stdout.write(f"Query: {args.query}\n")
        if settings.source_topic:
            sys.stdout.write(f"Source topic: {settings.source_topic}\n")
        for b in grounding.bottlenecks:
            sys.stdout.write(f"\nBottleneck {b.id}: {b.statement}\n")
            sys.stdout.write(f"  specific: {b.generic.get('specific')}\n")
            sys.stdout.write(f"  mid:      {b.generic.get('mid')}\n")
            sys.stdout.write(f"  broad:    {b.generic.get('broad')}\n")
            for ld in [x for x in leads if x.bottleneck_id == b.id]:
                sys.stdout.write(f"  * Guidance [{ld.technique_name} / {ld.source_discipline}]: {ld.resolution_guidance}\n")
            b_seeds = [s for s in seeds if s.bottleneck_id == b.id]
            for s in b_seeds:
                sys.stdout.write(f"  - [{s.level} #{s.rank}] score={s.score:.3f}: {s.card.name} ({s.card.document})\n")
        return 0

    try:
        result = engine.run_best_of_n(args.query, deps, settings)
    except stages.CorpusCoverageError as exc:
        sys.stderr.write(f"{exc}\n")
        return 3

    json_path = store.save_result(result, directory=args.out)
    md_path = json_path[:-5] + ".md"

    sys.stdout.write(f"JSON: {json_path}\n")
    sys.stdout.write(f"Report: {md_path}\n\n")

    for rank_idx, t in enumerate(result.transfers, start=1):
        aside = t.scores.set_aside if t.scores and t.scores.set_aside else "-"
        rew = f"{t.scores.reward:.2f}" if t.scores else "0.00"
        dep = f"{t.scores.depth:.2f}" if t.scores else "0.00"
        card_name = result.cards.get(t.card_id).name if t.card_id in result.cards else t.title
        sys.stdout.write(f"{rank_idx:<4} {rew:<7} {dep:<6} {aside:<24} {card_name:<30} <- {t.card_document}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
