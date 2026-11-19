"""Unit tests for LLM-guided cross-disciplinary scouting, two-topic bridging, and dynamic card expansion."""

from __future__ import annotations

from contextlib import redirect_stdout
import dataclasses
import io
import os
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_assistant.transfer import cards as cards_mod
from research_assistant.transfer import engine, render, scout, stages, store
from research_assistant.transfer.deps import TransferSettings
from research_assistant.transfer.models import Card, Premise, TechniqueLead
from scripts import transfer as cli
from tests.transfer_fixtures import sample_result, world


def _public_addrinfo(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]


class TestTransferScout(unittest.TestCase):
    def test_sparse_target_corpus_raises_without_scout_and_recovers_with_scout(self):
        deps_no_scout, _ = world(gate_keeps=0)
        with self.assertRaises(stages.CorpusCoverageError):
            engine.run_best_of_n("recovering parameters from sparse noisy spectra", deps_no_scout)

        deps_scout, _ = world(gate_keeps=0)
        settings = TransferSettings(
            bottlenecks=1,
            techniques_per_bottleneck=2,
            enable_scout=True,
            scout_max_papers=2,
        )
        result = engine.run_best_of_n(
            "recovering parameters from sparse noisy spectra",
            deps_scout,
            settings,
            scout_hooks={
                "search_fn": lambda q, k: [],
            },
        )
        self.assertEqual(len(result.bottlenecks), 1)
        self.assertEqual(result.bottlenecks[0].premises[0].key, "C_QUERY")
        self.assertGreaterEqual(len(result.leads), 1)
        self.assertEqual(result.leads[0].technique_name, "Sinkhorn entropic optimal transport")
        self.assertEqual(len(result.transfers), 2)
        self.assertTrue(result.discovery.get("enabled"))

    def test_materialize_leads_fetches_builds_verified_cards_and_blocks_unsafe_urls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = os.path.join(tmp_dir, "pdfs")
            card_dir = os.path.join(tmp_dir, "cards")
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(card_dir, exist_ok=True)

            deps, _ = world()
            lead = TechniqueLead(
                bottleneck_id="B1",
                technique_name="Sinkhorn entropic optimal transport",
                source_discipline="computational optimal transport",
                resolution_guidance="Use entropic Bregman projections instead of cubic matrix inversion.",
                search_queries=["Sinkhorn entropic optimal transport inverse problems"],
            )

            def fake_search(q: str, limit: int):
                return [
                    {
                        "key": "ssrf:bad",
                        "title": "Internal SSRF Probe",
                        "pdf_url": "http://127.0.0.1:8080/secret.pdf",
                    },
                    {
                        "key": "arxiv:2401.99999",
                        "arxiv_id": "2401.99999",
                        "title": "Sinkhorn Entropic Scaling for Ill-Posed Operators",
                        "authors": ["Cuturi"],
                        "year": 2024,
                        "pdf_url": "https://arxiv.org/pdf/2401.99999.pdf",
                    },
                ]

            downloaded_urls: list[str] = []

            def fake_download(url: str, dest_path: str):
                downloaded_urls.append(url)
                with open(dest_path, "wb") as fh:
                    fh.write(b"%PDF-1.4\n" + b"0" * 2048 + b"\n%%EOF\n")
                return True, None

            ingested_docs: list[str] = []

            def fake_ingest(pdf_map: dict[str, str]):
                for path in pdf_map:
                    ingested_docs.append(os.path.basename(path))
                return {"ingested": len(pdf_map), "failed": []}

            new_doc_id = "arxiv_2401.99999.pdf"
            augmented_texts = list(deps.texts) + [
                "Sinkhorn scaling converges linearly on positive Gibbs kernels K = exp(-C / eps) "
                "without dense matrix inversion (sample supporting span)."
            ]
            augmented_metas = list(deps.metadatas) + [
                {
                    "document": new_doc_id,
                    "seq": 0,
                    "type": "text_chunk",
                    "section": "methods",
                    "citation": "Cuturi (2024) — Sinkhorn Entropic Scaling",
                }
            ]

            def fake_reload_resources():
                return ("col", "bm25", augmented_texts, augmented_metas)

            settings = TransferSettings(enable_scout=True, scout_max_papers=2)

            with patch.object(scout.config, "PULLED_PDFS_DIR", pdf_dir), \
                 patch("research_assistant.shared.batch_uploader.socket.getaddrinfo", side_effect=_public_addrinfo), \
                 patch.object(engine, "hybrid_search", return_value=[]):
                refreshed_deps, fetched_docs, built_count, warnings = scout.materialize_leads_into_cards(
                    [lead],
                    deps,
                    settings,
                    card_dir=card_dir,
                    existing_docs=scout.known_corpus_documents(deps),
                    search_fn=fake_search,
                    download_fn=fake_download,
                    ingest_fn=fake_ingest,
                    reload_resources_fn=fake_reload_resources,
                )

            # Unsafe localhost URL was rejected; only public OA PDF was downloaded
            self.assertEqual(downloaded_urls, ["https://arxiv.org/pdf/2401.99999.pdf"])
            self.assertEqual(fetched_docs, [new_doc_id])
            self.assertEqual(ingested_docs, [new_doc_id])
            self.assertEqual(built_count, 1)
            self.assertEqual(lead.fetched_documents, [new_doc_id])
            self.assertEqual(len(lead.built_card_ids), 1)
            self.assertTrue(any(c.document == new_doc_id for c in refreshed_deps.cards.cards))

            store_check = cards_mod.CardStore(card_dir).load(check_model=False)
            self.assertEqual(len(store_check.cards), 1)
            self.assertEqual(store_check.cards[0].name, "Sinkhorn entropic optimal transport")

    def test_two_topic_bridge_passes_constraint_boosts_topic_cards_and_injects_guidance(self):
        deps, _ = world()
        prompts_seen: list[str] = []
        orig_chat = deps.chat

        def capturing_chat(msgs, temperature=0.7):
            prompts_seen.append(msgs[-1]["content"])
            return orig_chat(msgs, temperature)

        # Add a card explicitly matching the requested second topic ("optimal transport")
        ot_ev = Premise(
            claim="Claim for s1",
            key="C1",
            document="s1",
            chunk_index=6,
            judgement="Supports",
            span_verified=True,
            supporting_span="sample span",
        )
        ot_card = Card(
            card_id="s1#ot",
            document="s1",
            citation="Villani 2009 Optimal Transport",
            name="Sinkhorn entropic optimal transport",
            problem_solved="couples noisy distributions via optimal transport",
            context="computational optimal transport",
            inputs="marginals",
            outputs="coupling",
            assumptions=["positive kernel"],
            evidence=ot_ev,
            mechanism=["compute Gibbs kernel", "iterate Bregman projections"],
        )
        new_cards = list(deps.cards.cards) + [ot_card]
        new_vecs = np.vstack([deps.cards.vectors, deps.cards.vectors[0:1]])
        deps = dataclasses.replace(
            deps,
            chat=capturing_chat,
            cards=cards_mod.CardIndex(new_cards, new_vecs),
        )

        settings = TransferSettings(
            bottlenecks=1,
            techniques_per_bottleneck=1,
            source_topic="optimal transport",
            enable_scout=False,
        )
        result = engine.run_best_of_n("matrix inversion bottleneck", deps, settings)

        self.assertEqual(result.source_topic, "optimal transport")
        self.assertEqual(len(result.leads), 1)
        # Verify SCOUT_TECHNIQUES_USER received the source_topic constraint
        scout_prompts = [p for p in prompts_seen if "TASK: scout_techniques" in p]
        self.assertEqual(len(scout_prompts), 1)
        self.assertIn("optimal transport", scout_prompts[0])

        # Verify TRANSFER_USER received the LLM Cross-Disciplinary Guidance block
        transfer_prompts = [p for p in prompts_seen if "TASK: transfer" in p]
        self.assertGreaterEqual(len(transfer_prompts), 1)
        self.assertTrue(any("LLM Cross-Disciplinary Guidance:" in p for p in transfer_prompts))
        self.assertTrue(any("Bregman scaling iterations" in p for p in transfer_prompts))

    def test_render_and_store_round_trip_with_leads_and_discovery(self):
        r = sample_result()
        r.source_topic = "geophysical seismic tomography"
        r.discovery = {
            "enabled": True,
            "target_fetched": ["arxiv_2402.00001.pdf"],
            "source_fetched": ["arxiv_2402.00002.pdf"],
            "cards_built": 2,
        }
        r.leads = [
            TechniqueLead(
                bottleneck_id="B1",
                technique_name="Adjoint-state full-waveform inversion",
                source_discipline="geophysical seismic tomography",
                resolution_guidance="Computes exact Frechet sensitivity kernels via one forward and one adjoint wavefield solve.",
                search_queries=["adjoint state method seismic inversion"],
                fetched_documents=["arxiv_2402.00002.pdf"],
                built_card_ids=["arxiv_2402.00002.pdf#0"],
            )
        ]

        md = render.render_markdown(r)
        self.assertIn("geophysical seismic tomography", md)
        self.assertIn("## LLM Cross-Disciplinary Guidance & Discovery", md)
        self.assertIn("Adjoint-state full-waveform inversion", md)
        self.assertIn("Frechet sensitivity kernels", md)

        with tempfile.TemporaryDirectory() as d:
            saved_path = store.save_result(r, directory=d)
            loaded = store.load_result(saved_path)
        self.assertEqual(loaded, r)
        self.assertEqual(loaded.source_topic, "geophysical seismic tomography")
        self.assertEqual(len(loaded.leads), 1)
        self.assertEqual(loaded.leads[0].built_card_ids, ["arxiv_2402.00002.pdf#0"])

    def test_cli_supports_source_topic_in_show_cards_mode(self):
        deps, _ = world()
        out = io.StringIO()
        with patch.object(cli.engine, "ensure_idle"), \
             patch.object(cli.engine, "live_deps", return_value=deps), \
             redirect_stdout(out):
            rc = cli.main(["-q", "quantum transport bottleneck", "--source-topic", "optimal transport", "--show-cards"])
        self.assertEqual(rc, 0)
        rendered = out.getvalue()
        self.assertIn("Source topic: optimal transport", rendered)
        self.assertIn("Guidance [Sinkhorn entropic optimal transport", rendered)


if __name__ == "__main__":
    unittest.main()
