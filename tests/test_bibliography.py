# tests/test_bibliography.py
"""Reference-list detection: marks, entries, runs and bibliography headings."""

import unittest

from research_assistant.shared.bibliography import (
    is_reference_entry,
    is_reference_heavy_chunk,
    reference_runs,
)

APS = "F. Bloch, Phys. Rev. A 7, 2187 (1973)."
APS2 = "A. Sartori, J. Marino, S. Stringari, and A. Recati, New J. Phys. 17, 093036 (2015)."
ELS = "S. Torquato, F. H. Stillinger, Tunneled crystals, J. Appl. Phys. 102 (2007) 093511."
ARX = "A. Gallemí, A. Muñoz Mateo, R. Mayol, and M. Guilleumas, arXiv:1509.04418."
DOI = "Cai, J. et al. Graphene nanoribbon heterojunctions. doi:10.1038/nnano.2014.184"
NUM = "[12] E. Sandier and S. Serfaty, 2D Coulomb gases and the renormalized energy, Ann. Probab., 2015."
SPLIT = "M. Benakli and S. Raghavan, Europhys. Lett. (1999)"     # a continuation, no mark
PROSE = "We grew the samples by chemical vapour deposition and measured them at 4.2 K."


class TestReferenceEntries(unittest.TestCase):
    def test_each_reference_style_is_an_entry(self):
        for text in (APS, ELS, ARX, DOI, NUM):
            self.assertTrue(is_reference_entry(text), text)

    def test_prose_and_bare_years_are_not_entries(self):
        for text in (PROSE, "In 2019 the group repeated the measurement.", SPLIT,
                     "The gap closes at 4.2 K, as shown in earlier work [12]."):
            self.assertFalse(is_reference_entry(text), text)

    def test_merged_entries_count_at_any_length(self):
        merged = " ".join([APS, APS2, ELS] * 4)        # > 600 chars, 12 marks
        self.assertGreater(len(merged), 600)
        self.assertTrue(is_reference_entry(merged))

    def test_reference_heavy_chunk_threshold(self):
        self.assertTrue(is_reference_heavy_chunk(" ".join([APS, APS2, ELS])))
        self.assertFalse(is_reference_heavy_chunk(" ".join([APS, APS2])))


class TestReferenceRuns(unittest.TestCase):
    def test_a_run_needs_three_entries(self):
        self.assertEqual(reference_runs([PROSE, APS, APS2, PROSE]), [])
        self.assertEqual(reference_runs([PROSE, APS, APS2, ELS, PROSE]), [(1, 4)])

    def test_a_split_entry_stays_inside_the_run(self):
        self.assertEqual(reference_runs([APS, SPLIT, APS2, ARX]), [(0, 4)])

    def test_a_continuation_cannot_start_a_run(self):
        self.assertEqual(reference_runs([SPLIT, APS, APS2, ARX]), [(1, 4)])

    def test_a_long_paragraph_ends_the_run(self):
        long_prose = PROSE * 9 + " (2010)"            # > 600 chars, one year, no mark
        self.assertGreater(len(long_prose), 600)
        self.assertEqual(reference_runs([APS, APS2, ARX, long_prose]), [(0, 3)])

    def test_long_prose_paragraphs_with_inline_citations_survive(self):
        p = PROSE * 8 + " This agrees with Phys. Rev. B 82, 165109 (2010)."   # > 600 chars, 1 mark
        self.assertGreater(len(p), 600)
        self.assertEqual(reference_runs([p, p, p]), [])

    def test_numbered_procedure_without_years_survives(self):
        steps = ["1. Anneal at 300 C for 2 h.", "2. Cool to room temperature.", "3. Etch in HF for 30 s."]
        self.assertEqual(reference_runs(steps), [])


class TestBibliographyHeading(unittest.TestCase):
    def test_bibliography_heading_variants(self):
        from research_assistant.shared.extract import _BIB_HEAD
        for h in ("References", "REFERENCES", "VI. References", "5 References", "Supplementary references",
                  "References and Notes", "Notes and references", "Bibliography", "Literature cited",
                  "Literature", "Литература", "Works cited", "References:"):
            self.assertTrue(_BIB_HEAD.match(h), h)
        for h in ("Literature review", "Reference frames", "Notes", "Methods", "Referenced work"):
            self.assertFalse(_BIB_HEAD.match(h), h)


if __name__ == "__main__":
    unittest.main()
