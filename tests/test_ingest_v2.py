# tests/test_ingest_v2.py
"""process_pdf_v2: Document → corpus entries.

What matters is the contract with upsert_corpus (Task 10): pre-chunked
entries carry the stored text in `content`, the text to embed in
`embed_text`, and flat metadata in `meta`; captions are always emitted; a
figure_image entry (the crop, already embedded) follows each caption that has
a crop; one summary_source entry per document feeds the stage-1 index.
"""

import types
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
        p = patch.object(iv, "EMBED_IMAGES", True); p.start(); self.addCleanup(p.stop)
        self.embedder = types.SimpleNamespace(embed_image=lambda path: [0.25, 0.75])
        import research_assistant.shared.llm as llm_mod
        p = patch.object(llm_mod, "get_embeddings", lambda *a, **k: self.embedder)
        p.start(); self.addCleanup(p.stop)


class TestTextEntries(V2TestCase):
    def test_shape_and_header(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
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
        seqs = [e["meta"]["seq"] for e in entries if "meta" in e and e["type"] != "figure_image"]
        self.assertEqual(seqs, list(range(len(seqs))))
        # a figure's image is the same figure as its caption, so it shares the caption's seq
        cap = [e for e in entries if e["type"] == "caption"][0]
        img = [e for e in entries if e["type"] == "figure_image"][0]
        self.assertEqual(img["meta"]["seq"], cap["meta"]["seq"])
        self.assertEqual(entries[-1]["type"], "summary_source")
        self.assertEqual(entries[-2]["type"], "figure_image")


class TestCaptionsAndImages(V2TestCase):
    def test_caption_present_with_its_crop(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        caps = [e for e in entries if e["type"] == "caption"]
        self.assertEqual(len(caps), 1)
        c = caps[0]
        self.assertEqual(c["content"], "Figure 1: Density of states.")
        self.assertEqual(c["meta"]["figure_id"], "fig_0")
        self.assertEqual(c["meta"]["figure_kind"], "figure")
        self.assertEqual(c["meta"]["image_path"], "paper.pdf_p1_f0.png")
        self.assertNotIn("described", c["meta"])

    def test_each_cropped_figure_gets_one_embedded_image_entry(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        imgs = [e for e in entries if e["type"] == "figure_image"]
        self.assertEqual(len(imgs), 1)
        e = imgs[0]
        self.assertEqual(e["embedding"], [0.25, 0.75])
        self.assertEqual(e["content"], "Image of Figure 1: Density of states.")
        self.assertEqual(e["meta"]["figure_id"], "fig_0")
        self.assertEqual(e["meta"]["image_path"], "paper.pdf_p1_f0.png")
        self.assertEqual(e["meta"]["extraction"], "image")
        self.assertEqual(e["extraction"], "grobid")          # per-document mode for the run report

    def test_image_text_differs_from_the_caption_so_dedupe_keeps_both(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        cap = [e for e in entries if e["type"] == "caption"][0]
        img = [e for e in entries if e["type"] == "figure_image"][0]
        self.assertNotEqual(cap["content"], img["content"])

    def test_the_crop_is_embedded_from_the_file_that_was_written(self):
        seen = []
        self.embedder.embed_image = lambda path: seen.append(path) or [1.0]
        iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        self.assertEqual(seen, ["/tmp/img/paper.pdf_p1_f0.png"])

    def test_embedding_failure_keeps_the_caption_and_continues(self):
        def boom(path):
            raise RuntimeError("ollama down")
        self.embedder.embed_image = boom
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        self.assertEqual([e for e in entries if e["type"] == "figure_image"], [])
        self.assertEqual(len([e for e in entries if e["type"] == "caption"]), 1)
        self.assertTrue(any(e["type"] == "text_chunk" for e in entries))
        self.assertEqual(entries[-1]["type"], "summary_source")

    def test_no_crop_means_no_image_entry_but_caption_stays(self):
        with patch.object(iv, "crop_figure", return_value=False):
            entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        cap = [e for e in entries if e["type"] == "caption"][0]
        self.assertEqual(cap["meta"]["image_path"], "")
        self.assertEqual([e for e in entries if e["type"] == "figure_image"], [])

    def test_an_embedder_without_images_leaves_captions_and_crops_only(self):
        with patch.object(iv, "EMBED_IMAGES", False):
            entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        self.assertEqual([e for e in entries if e["type"] == "figure_image"], [])
        self.assertEqual([e for e in entries if e["type"] == "caption"][0]["meta"]["image_path"],
                         "paper.pdf_p1_f0.png")


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
    def test_no_figures_no_images_mode_recorded(self):
        doc = Document("paper.pdf", "paper.pdf", [], [Section("", "other", [Paragraph(" ".join(_sentence(i) for i in range(8)), 0)])], [], "pymupdf")
        with patch.object(iv, "extract", return_value=doc):
            entries = iv.process_pdf_v2("/x/paper.pdf", "L", images_dir="/tmp/img")
        self.assertTrue(all(e["extraction"] == "pymupdf" for e in entries))
        self.assertEqual([e["type"] for e in entries if e["type"] != "text_chunk"], ["summary_source"])
