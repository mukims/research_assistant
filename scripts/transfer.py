#!/usr/bin/env python3
"""CLI for unattended technique transfer and candidate card inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import config
from research_assistant.transfer import engine, stages, store
from research_assistant.transfer.deps import TransferSettings
from research_assistant.transfer.premises import PremiseChecker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Find cross-disciplinary technique transfers for a research question.")
    parser.add_argument("--query", "-q", required=True, help="Research question or problem statement.")
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

    settings = TransferSettings(
        bottlenecks=args.bottlenecks,
        techniques_per_bottleneck=args.techniques,
        exclude_documents=frozenset(args.exclude),
        prior_art_before_year=args.before_year,
    )

    deps = engine.live_deps()

    if args.show_cards:
        checker = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        try:
            grounding = stages.ground(args.query, deps, settings, checker)
            seeds, warnings = stages.candidate_cards(grounding, deps, settings)
        except stages.CorpusCoverageError as exc:
            sys.stderr.write(f"{exc}\n")
            return 3

        sys.stdout.write(f"Query: {args.query}\n")
        for b in grounding.bottlenecks:
            sys.stdout.write(f"\nBottleneck {b.id}: {b.statement}\n")
            sys.stdout.write(f"  specific: {b.generic.get('specific')}\n")
            sys.stdout.write(f"  mid:      {b.generic.get('mid')}\n")
            sys.stdout.write(f"  broad:    {b.generic.get('broad')}\n")
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
