# tests/test_chunking.py
"""Sentence windows: what replaces per-page SemanticChunker.

The v1 index's median chunk was 242 chars and 45% were under 200 — bibliography
fragments and clauses cut at an embedding-distance breakpoint. These tests pin
the four rules that stop that: pack to a target, never exceed the max, overlap
one sentence, and don't cut a sentence in half.
"""

import unittest

from research_assistant.shared import chunking as ck
from research_assistant.shared.extract import Paragraph


class TestSentences(unittest.TestCase):
    def test_splits_plain_prose(self):
        self.assertEqual(ck.sentences("One here. Two here. Three here."), ["One here.", "Two here.", "Three here."])

    def test_physics_abbreviations_do_not_split(self):
        text = "We follow Ref. [3] and Fig. 2b. Phys. Rev. B 70, 241403 (2004) reports it. The gap is 0.5 eV (see Sec. 3). Done."
        out = ck.sentences(text)
        self.assertEqual(out, [
            "We follow Ref. [3] and Fig. 2b.",
            "Phys. Rev. B 70, 241403 (2004) reports it.",
            "The gap is 0.5 eV (see Sec. 3).",
            "Done.",
        ])

    def test_empty(self):
        self.assertEqual(ck.sentences("   "), [])


class TestPackWindows(unittest.TestCase):
    def _sent(self, i, page=0):
        return f"Sentence number {i} has about sixty characters of text in it, yes."   # ~66 chars

    def _paras(self, n, per_para=5, page=0):
        return [Paragraph(" ".join(self._sent(i * per_para + j) for j in range(per_para)), page) for i in range(n)]

    def test_windows_reach_target_and_never_exceed_max(self):
        out = ck.pack_windows(self._paras(6), target=300, hard_max=450)
        self.assertGreater(len(out), 1)
        for text, _, _ in out:
            self.assertLessEqual(len(text), 450)
        # every window but the last is at least the target or a paragraph close
        for text, _, _ in out[:-1]:
            self.assertGreaterEqual(len(text), 0.7 * 300)

    def test_one_sentence_overlap(self):
        out = ck.pack_windows(self._paras(2, per_para=6), target=300, hard_max=450)
        first, second = out[0][0], out[1][0]
        last_sentence_of_first = ck.sentences(first)[-1]
        self.assertTrue(second.startswith(last_sentence_of_first))

    def test_no_sentence_is_cut(self):
        out = ck.pack_windows(self._paras(4), target=300, hard_max=450)
        for text, _, _ in out:
            for s in ck.sentences(text):
                self.assertTrue(s.endswith("."), s)

    def test_a_sentence_longer_than_max_stands_alone(self):
        giant = "X" + "x" * 600 + "."
        paras = [Paragraph(f"Short one. {giant} Short two.", 0)]
        out = ck.pack_windows(paras, target=300, hard_max=450)
        self.assertEqual([t for t, _, _ in out], ["Short one.", giant, "Short two."])

    def test_prefers_closing_at_a_paragraph_boundary(self):
        # Two paragraphs of ~260 chars: a 300 target must not glue them if the
        # first is already ≥ 70% of target.
        paras = self._paras(2, per_para=4)
        out = ck.pack_windows(paras, target=300, hard_max=450)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], paras[0].text)

    def test_pages_are_tracked_across_the_window(self):
        paras = [Paragraph(self._sent(0) + " " + self._sent(1), 3), Paragraph(self._sent(2) + " " + self._sent(3), 4)]
        out = ck.pack_windows(paras, target=1000, hard_max=1500)
        self.assertEqual(out, [(paras[0].text + " " + paras[1].text, 3, 4)])

    def test_empty_input(self):
        self.assertEqual(ck.pack_windows([], 300, 450), [])

    def test_overlap_alone_is_never_a_second_window(self):
        # 5 sentences ≈ 334 chars close one window at the target; the carried
        # overlap sentence must not be flushed as a window of its own.
        out = ck.pack_windows(self._paras(1, per_para=5), target=300, hard_max=450)
        self.assertEqual(len(out), 1)


from research_assistant.shared.extract import Document, Section, Figure


def _sentence(i):
    return f"Sentence number {i} has about sixty characters of text in it, yes."


def _section(heading, kind, n_sentences, page=0):
    return Section(heading, kind, [Paragraph(" ".join(_sentence(i) for i in range(n_sentences)), page)])


class TestFilters(unittest.TestCase):
    def test_alpha_ratio(self):
        self.assertGreater(ck.alpha_ratio("plain words here"), 0.9)
        self.assertLess(ck.alpha_ratio("2D! Q82A'C#+,!#>! #6!"), 0.6)

    def test_garbage_paragraphs_and_acknowledgements_are_dropped(self):
        doc = Document("k.pdf", "T", [], [
            _section("Introduction", "introduction", 6),
            Section("Results", "results", [Paragraph("2D! Q82A'C#+,!#>! #6! S%86'6! F?;! 5%.6'8%5%+>6", 3),
                                            Paragraph(" ".join(_sentence(i) for i in range(6)), 3)]),
            _section("Acknowledgements", "acknowledgements", 3),
        ], [], "grobid")
        out = ck.filter_document(doc)
        self.assertEqual([s.kind for s in out.sections], ["introduction", "results"])
        self.assertEqual(len(out.sections[1].paragraphs), 1)

    def test_short_sections_merge_forward(self):
        doc = Document("k.pdf", "T", [], [
            Section("Nomenclature", "other", [Paragraph("Ten chars.", 0)]),
            _section("Introduction", "introduction", 6, page=1),
        ], [], "grobid")
        out = ck.filter_document(doc)
        self.assertEqual(len(out.sections), 1)
        self.assertEqual(out.sections[0].kind, "introduction")
        self.assertEqual(out.sections[0].paragraphs[0].text, "Ten chars.")

    def test_trailing_short_section_merges_backward(self):
        doc = Document("k.pdf", "T", [], [
            _section("Introduction", "introduction", 6),
            Section("Note", "other", [Paragraph("Ten chars.", 5)]),
        ], [], "grobid")
        out = ck.filter_document(doc)
        self.assertEqual(len(out.sections), 1)
        self.assertEqual(out.sections[0].paragraphs[-1].text, "Ten chars.")


class TestChunkDocument(unittest.TestCase):
    def _doc(self):
        return Document("k.pdf", "A title", [Paragraph(" ".join(_sentence(i) for i in range(4)), 0)], [
            _section("1. Introduction", "introduction", 10, page=0),
            _section("2. Methods", "methods", 10, page=2),
        ], [
            Figure("fig_0", "figure", "Figure 1", "Density of states of the first site.", 1, (0, 0, 1, 1), ["Seen in Figure 1."]),
            Figure("tab_0", "table", "Table 1", "Mobilities.", 3, None, []),
        ], "grobid")

    def test_chunks_never_cross_a_section(self):
        chunks = ck.chunk_document(self._doc(), target=300, hard_max=450)
        for c in chunks:
            if c.type == "text":
                self.assertIn(c.section, ("abstract", "introduction", "methods"))
        sections_in_order = [c.section for c in chunks if c.type == "text"]
        self.assertEqual(sections_in_order, sorted(sections_in_order, key=["abstract", "introduction", "methods"].index))

    def test_seq_is_contiguous_and_pages_carried(self):
        chunks = ck.chunk_document(self._doc(), target=300, hard_max=450)
        self.assertEqual([c.seq for c in chunks], list(range(len(chunks))))
        methods = [c for c in chunks if c.section == "methods"]
        self.assertTrue(all(c.page_first == 2 for c in methods))
        self.assertEqual(chunks[0].section_raw, "")            # abstract has no heading
        self.assertEqual([c.section_raw for c in chunks if c.section == "methods"][0], "2. Methods")

    def test_captions_are_chunks_exempt_from_the_length_floor(self):
        chunks = ck.chunk_document(self._doc(), target=300, hard_max=450)
        caps = [c for c in chunks if c.type == "caption"]
        self.assertEqual([(c.figure_id, c.figure_kind, c.figure_label) for c in caps],
                         [("fig_0", "figure", "Figure 1"), ("tab_0", "table", "Table 1")])
        self.assertEqual(caps[0].text, "Figure 1: Density of states of the first site.")
        self.assertEqual(caps[1].page_first, 3)
        self.assertLess(len(caps[1].text), 200)

    def test_short_text_windows_are_dropped(self):
        doc = Document("k.pdf", "T", [], [Section("Intro", "introduction", [Paragraph("Tiny.", 0)])], [], "grobid")
        self.assertEqual(ck.chunk_document(doc, target=300, hard_max=450), [])

    def test_figure_without_caption_is_still_a_chunk(self):
        doc = Document("k.pdf", "T", [], [], [Figure("fig_0", "figure", "Figure 2", "", 0, None, [])], "grobid")
        chunks = ck.chunk_document(doc, target=300, hard_max=450)
        self.assertEqual(chunks[0].text, "Figure 2")
