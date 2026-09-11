# tests/test_ingest_v2.py
"""process_pdf_v2: Document → corpus entries.

What matters is the contract with upsert_corpus (Task 10): pre-chunked
entries carry the stored text in `content`, the text to embed in
`embed_text`, and flat metadata in `meta`; captions are always emitted; a
description entry appears only when the run's switch is on; one
summary_source entry per document feeds the stage-1 index.
"""

import unittest
from unittest.mock import patch

from research_assistant.shared import ingest_v2 as iv
from research_assistant.shared.extract import Document, Section, Paragraph, Figure


def _sentence(i):
    return f"Sentence number {i} has about sixty characters of text in it, yes."


def _doc():
    return Document("paper.pdf", "A Title", [Paragraph(" ".join(_sentence(i) for i in range(4)), 0)], [
        Section("1. Introduction", "introduction", [Paragraph(" ".join(_sentence(i) for i in range(12)), 0)]),
        Section("4. Conclusion", "conclusion", [Paragraph(" ".join(_sentence(i) for i in range(6)), 5)]),
    ], [
        Figure("fig_0", "figure", "Figure 1", "Density of states.", 1, (0, 0, 10, 10), ["Seen in Figure 1."]),
    ], "grobid", n_bib=12)


class V2TestCase(unittest.TestCase):
    def setUp(self):
        p = patch.object(iv, "extract", return_value=_doc()); p.start(); self.addCleanup(p.stop)
        p = patch.object(iv, "crop_figure", return_value=True); p.start(); self.addCleanup(p.stop)
        describe = patch.object(iv, "describe_crop", return_value="Two sharp peaks.")
        self.describe_mock = describe.start(); self.addCleanup(describe.stop)


class TestTextEntries(V2TestCase):
    def test_shape_and_header(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=False, images_dir="/tmp/img")
        text = [e for e in entries if e["type"] == "text_chunk"]
        self.assertGreater(len(text), 1)
        e = text[0]
        self.assertEqual(e["document"], "paper.pdf")
        self.assertEqual(e["citation"], "A Title")
        self.assertEqual(e["extraction"], "grobid")
        self.assertEqual(e["page"], e["meta"]["page_first"])
        self.assertTrue(e["embed_text"].startswith("Title: A Title. Section: "))
        self.assertTrue(e["embed_text"].endswith(e["content"]))
        self.assertEqual(set(e["meta"]), {"section", "section_raw", "page_first", "page_last", "seq", "extraction"})
        self.assertEqual(e["meta"]["extraction"], "grobid")

    def test_seq_runs_over_text_then_captions_and_summary_source_is_last(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        seqs = [e["meta"]["seq"] for e in entries if "meta" in e]
        self.assertEqual(seqs, list(range(len(seqs))))
        self.assertEqual(entries[-1]["type"], "summary_source")
        self.assertEqual(entries[-2]["type"], "caption")


class TestCaptionsAndDescriptions(V2TestCase):
    def test_caption_always_present_with_crop_and_described_false(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=False, images_dir="/tmp/img")
        caps = [e for e in entries if e["type"] == "caption"]
        self.assertEqual(len(caps), 1)
        c = caps[0]
        self.assertEqual(c["content"], "Figure 1: Density of states.")
        self.assertEqual(c["meta"]["figure_id"], "fig_0")
        self.assertEqual(c["meta"]["figure_kind"], "figure")
        self.assertEqual(c["meta"]["image_path"], "paper.pdf_p1_f0.png")
        self.assertFalse(c["meta"]["described"])
        self.assertEqual([e for e in entries if e["type"] == "figure_description"], [])
        self.describe_mock.assert_not_called()

    def test_switch_on_describes_every_figure_inline(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        desc = [e for e in entries if e["type"] == "figure_description"]
        self.assertEqual(len(desc), 1)
        d = desc[0]
        self.assertEqual(d["content"], "Auto-generated description of Figure 1 — verify values against the figure: Two sharp peaks.")
        self.assertEqual(d["meta"]["extraction"], "vlm")
        self.assertEqual(d["meta"]["figure_id"], "fig_0")
        self.assertEqual(d["extraction"], "grobid")          # per-document mode for the run report
        self.assertTrue(d["embed_text"].startswith("Title: A Title. Section: Figure 1."))
        cap = [e for e in entries if e["type"] == "caption"][0]
        self.assertTrue(cap["meta"]["described"])

    def test_description_context_is_caption_plus_refs(self):
        with patch.object(iv, "describe_crop", return_value="x") as d:
            iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        ctx = d.call_args.args[2]
        self.assertIn("Caption: Figure 1: Density of states.", ctx)
        self.assertIn("- Seen in Figure 1.", ctx)

    def test_description_failure_leaves_caption_undescribed_and_continues(self):
        with patch.object(iv, "describe_crop", side_effect=RuntimeError("model down")):
            entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        self.assertEqual([e for e in entries if e["type"] == "figure_description"], [])
        self.assertFalse([e for e in entries if e["type"] == "caption"][0]["meta"]["described"])
        self.assertTrue(any(e["type"] == "text_chunk" for e in entries))

    def test_no_crop_means_no_description_but_caption_stays(self):
        with patch.object(iv, "crop_figure", return_value=False):
            entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        cap = [e for e in entries if e["type"] == "caption"][0]
        self.assertEqual(cap["meta"]["image_path"], "")
        self.assertEqual([e for e in entries if e["type"] == "figure_description"], [])


class TestSummarySource(V2TestCase):
    def test_abstract_intro_conclusion_in_that_order(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        src = [e for e in entries if e["type"] == "summary_source"]
        self.assertEqual(len(src), 1)
        text = src[0]["content"]
        self.assertLess(text.index("Sentence number 0"), text.index("Sentence number 11"))
        self.assertIn("Sentence number 5 has", text)              # conclusion included
        self.assertEqual(src[0]["document"], "paper.pdf")


class TestPyMuPDFDocument(V2TestCase):
    def test_no_figures_no_descriptions_mode_recorded(self):
        doc = Document("paper.pdf", "paper.pdf", [], [Section("", "other", [Paragraph(" ".join(_sentence(i) for i in range(8)), 0)])], [], "pymupdf")
        with patch.object(iv, "extract", return_value=doc):
            entries = iv.process_pdf_v2("/x/paper.pdf", "L", describe_figures=True, images_dir="/tmp/img")
        self.assertTrue(all(e["extraction"] == "pymupdf" for e in entries))
        self.assertEqual([e["type"] for e in entries if e["type"] != "text_chunk"], ["summary_source"])
