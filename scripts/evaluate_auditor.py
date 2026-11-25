"""End-to-end benchmark for the Seed Paper Citation Auditor pipeline.

    PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/evaluate_auditor.py
    PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/evaluate_auditor.py --live-judge

Evaluates the six deterministic & retrieval stages of `seed_audit.py` that sit
around `judge()` — non-citation noise rejection, numeric & author-year
bibliography resolution, multi-citation clause attribution, bibliometric/retraction
source grading, intra-paper retrieval + reranking, and disaggregated scorecard
accounting — plus real-TEI resolution telemetry when `data/raw/grobid_output`
is present, and optional live held-out LLM judging (`--live-judge`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

from bs4 import BeautifulSoup

from research_assistant.audit.resolver import _resolve_ref, enrich_item_bibliometrics
from research_assistant.config import DATA_DIR
from research_assistant.judgement.evalset import load_cases, split_held_out
from research_assistant.judgement.judge import PROMPT_TEMPLATE, filter_relevant_artifacts
from research_assistant.judgement.policy import ReliabilityRating, evaluate_reliability
from research_assistant.judgement.source_assessor import SourceGrade, assess_source
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.claim_text import (
    classify_citation_role,
    describe_citation_locus,
    is_non_citation_ref,
)
from research_assistant.shared.search import rerank_candidates, supplement_intra_doc_lexical_hits
from research_assistant.shared.seed_audit import compute_totals, extract_seed_citation_claims

RESULTS_DIR = os.path.join(DATA_DIR, "eval", "auditor", "results")


_BENCHMARK_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>1. Introduction</head>
        <p><s>Substituting Eq. <ref type="bibr">(1.3)</ref> and <ref type="bibr">(1.191</ref><ref type="bibr">.198)</ref> into <ref type="bibr">(Section B.6)</ref> gives the bound.</s>
           <s>Pioneered by Ronald <ref type="bibr">Fisher (1890</ref><ref type="bibr">Fisher ( -1962) )</ref> in statistical inference.</s>
           <s>Variational autoencoders were introduced simultaneously <ref type="bibr">(Kingma and</ref> <ref type="bibr">Welling, 2014, Rezende et al., 2014)</ref> for latent inference.</s>
           <s>Functional analysis foundations follow <ref type="bibr">Bollobas (1999)</ref> throughout this work.</s>
           <s>Inverse problems are used in medical imaging <ref type="bibr" target="#b3">[4]</ref> and seismology <ref type="bibr" target="#b4">[5]</ref>, while quantum device mapping remains challenging.</s>
           <s>The solver is implemented in the Wannier90 package <ref type="bibr" target="#b5">[6]</ref> and run on CPUs.</s></p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct xml:id="b0">
          <analytic>
            <title level="a">Auto-Encoding Variational Bayes</title>
            <author><persName><surname>Kingma</surname></persName></author>
            <author><persName><surname>Welling</surname></persName></author>
          </analytic>
          <monogr><imprint><date when="2014" /></imprint></monogr>
        </biblStruct>
        <biblStruct xml:id="b1">
          <analytic>
            <title level="a">Stochastic Backpropagation and Approximate Inference</title>
            <author><persName><surname>Rezende</surname></persName></author>
            <author><persName><surname>Mohamed</surname></persName></author>
            <author><persName><surname>Wierstra</surname></persName></author>
          </analytic>
          <monogr><imprint><date when="2014" /></imprint></monogr>
        </biblStruct>
        <biblStruct xml:id="b2">
          <monogr>
            <title level="m">Linear Analysis</title>
            <author><persName><surname>Bollobás</surname></persName></author>
            <imprint><date when="1999" /></imprint>
          </monogr>
        </biblStruct>
        <biblStruct xml:id="b3">
          <analytic>
            <title level="a">Image reconstruction in medical tomography</title>
            <author><persName><surname>Natterer</surname></persName></author>
          </analytic>
          <monogr><title level="j">Inverse Problems</title><imprint><date when="2001" /></imprint></monogr>
        </biblStruct>
        <biblStruct xml:id="b4">
          <analytic>
            <title level="a">Spectral-element and adjoint methods in seismology</title>
            <author><persName><surname>Tromp</surname></persName></author>
          </analytic>
          <monogr><title level="j">Commun. Comput. Phys.</title><imprint><date when="2008" /></imprint></monogr>
        </biblStruct>
        <biblStruct xml:id="b5">
          <analytic>
            <title level="a">Wannier90 as a community code: new features and applications</title>
            <author><persName><surname>Pizzi</surname></persName></author>
          </analytic>
          <monogr><title level="j">J. Phys. Condens. Matter</title><imprint><date when="2020" /></imprint></monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>
"""


def run_deterministic_suite() -> dict:
    # 1. Non-citation noise rejection
    noisy_inputs = [
        "", "(1.3)", "(1.198)", "(1.191", ".198)", "1.3.6",
        "[__name__]", "Fisher (1890-1962)", "Chapter 14)",
        "(Section B.6)", "[1:3]", "[0,", "(3, 4) X tensor([[ 0., 1., 2., 3.",
    ]
    valid_inputs = ["[12]", "[13–20]", "Bollobás (1999)", "(Kingma and Welling, 2014)"]
    noise_hits = sum(1 for s in noisy_inputs if is_non_citation_ref(s, has_valid_target=False))
    valid_kept = sum(1 for s in valid_inputs if not is_non_citation_ref(s, has_valid_target=False))
    noise_acc = (noise_hits + valid_kept) / (len(noisy_inputs) + len(valid_inputs))

    # 2. End-to-end TEI extraction, stitching, resolution & role classification
    claims = extract_seed_citation_claims(BeautifulSoup(_BENCHMARK_TEI, "xml"))
    expected_ids = {"b0", "b1", "b2", "b3", "b4", "b5"}
    got_ids = {c["ref"]["xml_id"] for c in claims if c.get("resolved")}
    resolution_acc = len(expected_ids & got_ids) / len(expected_ids) if len(claims) == 6 else 0.0

    by_id = {c["ref"]["xml_id"]: c for c in claims if c.get("resolved")}
    # Multi-citation clause attribution check on b3 vs b4
    clause_ok = int(
        "medical imaging" in (by_id.get("b3", {}).get("target_clause") or "")
        and "seismology" not in (by_id.get("b3", {}).get("target_clause") or "")
        and "seismology" in (by_id.get("b4", {}).get("target_clause") or "")
        and by_id.get("b5", {}).get("role") == "software"
    )

    # 3. Source grading & retraction propagation
    retracted_item = {"ref": {"title": "Retracted Paper", "venue": "Nature"}}
    enrich_item_bibliometrics(retracted_item, dl_entry={"is_retracted": True, "citation_count": 500})
    g_ret = assess_source(retracted_item["metadata"], retracted_item["ref"])
    rel_ret = evaluate_reliability("Supports", g_ret["grade"], confidence="High", span_verified=True)

    rigorous_item = {"ref": {"title": "Well-Cited Primary", "venue": "Journal of Alloys and Compounds"}}
    enrich_item_bibliometrics(rigorous_item, dl_entry={"is_retracted": False, "citation_count": 120})
    g_rig = assess_source(rigorous_item["metadata"], rigorous_item["ref"])

    preprint_item = {"ref": {"title": "Preprint"}}
    enrich_item_bibliometrics(preprint_item, dl_entry={"doi": "https://doi.org/10.1101/2024.01.01.111111"})
    g_pre = assess_source(preprint_item["metadata"], preprint_item["ref"])

    grading_checks = [
        g_ret["grade"] == SourceGrade.RETRACTED_OR_FLAWED.value,
        rel_ret["rating"] == ReliabilityRating.LOW.value,
        g_rig["grade"] == SourceGrade.RIGOROUS_PRIMARY.value,
        g_pre["grade"] == SourceGrade.PREPRINT_UNREVIEWED.value,
    ]
    grading_acc = sum(grading_checks) / len(grading_checks)

    # 4. Intra-paper lexical rescue & section-diverse reranking
    texts = [
        "Introduction: Broad survey of supercapacitors [1-15].",
        "Adjacent intro paragraph on conducting polymers.",
        "Results: Symmetric MnFe2O4@PANI cells deliver 623 F/g in 1M H2SO4.",
    ]
    metas = [
        {"document": "p.pdf", "type": "text_chunk", "section": "introduction"},
        {"document": "p.pdf", "type": "text_chunk", "section": "introduction"},
        {"document": "p.pdf", "type": "text_chunk", "section": "results"},
    ]
    merged = supplement_intra_doc_lexical_hits(
        ["MnFe2O4@PANI reaches 623 F/g in 1M H2SO4"],
        {"p.pdf"},
        None,
        texts,
        metas,
        [{"chunk_index": 0, "text": texts[0], "metadata": metas[0], "rrf_score": 0.03}],
    )
    ranked = rerank_candidates(
        "supercapacitor performance",
        merged,
        top_k=2,
        claim_text="MnFe2O4@PANI reaches 623 F/g in 1M H2SO4",
        primary_k=1,
    )
    retrieval_acc = 1.0 if (ranked and ranked[0]["chunk_index"] == 2) else 0.0

    # 5. Selective context & scorecard partition invariant
    arts = [
        {"label": "Fig. 1", "caption": "Device schematic."},
        {"label": "Fig. 2", "caption": "Nyquist plot."},
    ]
    sel = filter_relevant_artifacts("Fig. 2b shows reduced impedance.", arts)
    sel_empty = filter_relevant_artifacts("Impedance drops at 300 K.", arts)
    totals = compute_totals([
        {"outcome": "judged", "judgement": "Supports", "downloaded": True, "reliability": "HIGH"},
        {"outcome": "judged", "judgement": "Unclear / insufficient evidence", "downloaded": True, "reliability": "UNRESOLVED"},
        {"outcome": "not_a_claim", "downloaded": False, "reliability": "UNRESOLVED"},
        {"outcome": "not_downloaded", "downloaded": False, "reliability": "UNRESOLVED"},
    ])
    rb = totals["reliability_breakdown"]
    context_and_accounting_ok = float(
        [a["label"] for a in sel] == ["Fig. 2"]
        and sel_empty == []
        and sum(rb.values()) == totals["total"]
        and rb["inconclusive"] == 1
        and rb["non_evidential_skipped"] == 1
        and rb["unavailable_paywalled"] == 1
    )

    # 6. Evalset integrity (zero prompt leakage refusals on held-out transforms)
    all_cases = load_cases()
    held, refused = split_held_out(all_cases, PROMPT_TEMPLATE)
    leaked_non_prompt = [c["id"] for c in refused if c["source"] != "prompt_example"]

    # Optional real-TEI resolution telemetry if local GROBID TEIs exist
    tei_telemetry = []
    for tei_path in sorted(glob.glob(os.path.join(DATA_DIR, "raw", "grobid_output", "*.tei.xml")))[:10]:
        c_list = extract_seed_citation_claims(tei_path)
        n_tot = len(c_list)
        n_res = sum(1 for x in c_list if x.get("resolved"))
        tei_telemetry.append({
            "tei": os.path.basename(tei_path),
            "total_claims": n_tot,
            "resolved": n_res,
            "resolution_rate": round(n_res / n_tot, 4) if n_tot else 1.0,
        })

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "noise_filter_accuracy": round(noise_acc, 4),
            "bibliography_resolution_accuracy": round(resolution_acc, 4),
            "multi_citation_clause_attribution": float(clause_ok),
            "source_bibliometric_grading_accuracy": round(grading_acc, 4),
            "intra_doc_retrieval_top1_recall": retrieval_acc,
            "selective_context_and_scorecard_invariant": context_and_accounting_ok,
        },
        "evalset_health": {
            "held_out_cases": len(held),
            "leaked_non_prompt_cases": leaked_non_prompt,
        },
        "corpus_tei_telemetry": tei_telemetry,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-judge", action="store_true", help="Also run live held-out LLM judge benchmark (selective context)")
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()

    summary = run_deterministic_suite()
    print("=== Citation Auditor Deterministic Pipeline Benchmark ===")
    for k, v in summary["metrics"].items():
        print(f"  {k:44s}: {v * 100:6.1f}%")
    eh = summary["evalset_health"]
    print(f"  held_out_eval_cases                         : {eh['held_out_cases']} (leaked={len(eh['leaked_non_prompt_cases'])})")
    if summary["corpus_tei_telemetry"]:
        print("\n=== Local Corpus TEI Resolution Telemetry ===")
        for row in summary["corpus_tei_telemetry"]:
            print(f"  {row['tei']:44s}: {row['resolved']:4d}/{row['total_claims']:4d} ({row['resolution_rate'] * 100:5.1f}%)")

    if args.live_judge:
        from scripts import evaluate_judge as ej
        from research_assistant.judgement.metrics import render as render_judge

        print("\n=== Live Held-Out Judge Benchmark (context=selective) ===")
        live = ej.run(load_cases(), mode="judge", runs=1, context_mode="selective")
        print("\n" + render_judge(live["summary"], in_prompt=live["in_prompt"], refused=live["refused"]))
        summary["live_judge"] = live["summary"]

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = os.path.join(args.out, f"{stamp}-auditor-eval.json")
    atomic_write_json(out_file, summary)
    print(f"\nwritten {out_file}")


if __name__ == "__main__":
    main()
