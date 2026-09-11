# tests/test_extract.py
"""TEI → Document. What the chunker receives from GROBID.

The fixture exercises the things PyMuPDF got wrong on the real corpus:
sections with headings, a running header GROBID mislabels as a section,
formulas as siblings of paragraphs, a figure with a graphic box, a table
with no graphic, in-text figure references, and a bibliography kept apart.
"""

import os
import unittest

from research_assistant.shared import extract as ex

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.grobid.tei.xml")


def _doc():
    with open(FIXTURE, "rb") as f:
        return ex.parse_tei(f.read(), key="sample.pdf")


class TestHeaderAndAbstract(unittest.TestCase):
    def test_title_and_key(self):
        d = _doc()
        self.assertEqual(d.key, "sample.pdf")
        self.assertEqual(d.title, "Charge transport in covalent MoS2 networks")
        self.assertEqual(d.extraction, "grobid")

    def test_abstract_is_paragraphs_with_zero_based_page(self):
        d = _doc()
        self.assertEqual(len(d.abstract), 1)
        self.assertTrue(d.abstract[0].text.startswith("We study charge transport"))
        self.assertEqual(d.abstract[0].page, 0)

    def test_bibliography_is_counted_not_included(self):
        d = _doc()
        self.assertEqual(d.n_bib, 2)
        body = " ".join(p.text for s in d.sections for p in s.paragraphs)
        self.assertNotIn("Ref one", body)


class TestSections(unittest.TestCase):
    def test_headings_kinds_and_running_header_folded_into_previous(self):
        d = _doc()
        self.assertEqual([s.heading for s in d.sections], ["Introduction", "Methods", "Acknowledgements"])
        self.assertEqual([s.kind for s in d.sections], ["introduction", "methods", "acknowledgements"])
        intro = d.sections[0]
        self.assertEqual(len(intro.paragraphs), 3)            # 2 own + 1 folded from "(3 of 11)"
        self.assertEqual([p.page for p in intro.paragraphs], [0, 1, 2])

    def test_paragraph_text_includes_ref_text_and_is_single_spaced(self):
        p = _doc().sections[0].paragraphs[0]
        self.assertIn("(Figure 1b)", p.text)
        self.assertNotIn("\n", p.text)

    def test_formulas_are_not_paragraphs(self):
        methods = _doc().sections[1]
        self.assertEqual(len(methods.paragraphs), 2)
        self.assertNotIn("σ = n e μ", " ".join(p.text for p in methods.paragraphs))


class TestFigures(unittest.TestCase):
    def test_figure_with_graphic_box(self):
        fig = _doc().figures[0]
        self.assertEqual((fig.id, fig.kind, fig.label), ("fig_0", "figure", "Figure 1"))
        self.assertTrue(fig.caption.startswith("Figure 1. THz spectral analysis"))
        self.assertEqual(fig.page, 2)
        self.assertEqual(fig.bbox, (50.0, 379.0, 542.0, 643.0))   # x0, y0, x0+w, y0+h

    def test_table_uses_union_of_figure_coords(self):
        tab = _doc().figures[1]
        self.assertEqual((tab.id, tab.kind, tab.label), ("tab_0", "table", "Table 1"))
        self.assertEqual(tab.page, 4)
        self.assertEqual(tab.bbox, (48.0, 100.0, 544.0, 300.0))

    def test_in_text_references_are_attached_as_sentences(self):
        fig = _doc().figures[0]
        self.assertEqual(len(fig.refs), 1)
        self.assertIn("photoconductivity increases by 10%", fig.refs[0])
        self.assertEqual(_doc().figures[1].refs, [])


class TestSectionKinds(unittest.TestCase):
    def test_keyword_mapping(self):
        cases = {
            "1. Introduction": "introduction", "Related work": "background",
            "Experimental methods": "methods", "Results and discussion": "results",
            "IV. DISCUSSION": "discussion", "Summary and outlook": "conclusion",
            "Conclusions": "conclusion", "Acknowledgments": "acknowledgements",
            "Funding": "acknowledgements", "Author contributions": "acknowledgements",
            "Terahertz spectral analysis": "other", "": "other",
        }
        for heading, kind in cases.items():
            self.assertEqual(ex.normalise_section_kind(heading), kind, heading)

    def test_running_header_detection(self):
        for h in ["(3 of 11)", "Odashima et al.", "12", "Adv. Mater. 2023, 35, 2211157"]:
            self.assertTrue(ex.is_running_header(h), h)
        for h in ["Introduction", "2.1 Green's functions", "Results"]:
            self.assertFalse(ex.is_running_header(h), h)
