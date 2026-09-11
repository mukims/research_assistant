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
