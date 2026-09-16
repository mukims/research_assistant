"""Where a paragraph sits in the citing paper's TEI and what it points at.

GROBID's sections are flat siblings ("2." and "2.1." side by side) and its
figure list carries page-header fragments; these pin the walks that make a
breadcrumb and an artifact list out of that.
"""

import unittest

from bs4 import BeautifulSoup

from research_assistant.shared import tei_structure as ts

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><p xml:id="p0">A paragraph before any heading.</p></div>
<div><head n="1.">Introduction</head><p xml:id="p1">Intro.</p></div>
<div><head n="2.">Results and Discussion</head><p xml:id="p2">Top of results.</p></div>
<div><head n="2.1.">Terahertz Spectral Analysis</head>
  <p xml:id="p3">See <ref type="figure" target="#fig_0">1b</ref> and Table <ref type="table">2</ref> and Fig. 3 and Fig. 1 again.</p>
</div>
<div><head n="2211157">(3 of 11)</head><p xml:id="p4">Under a page header.</p></div>
<div><head n="2.2.">Charge Transport</head><p xml:id="p5">More.</p></div>
<div><p xml:id="p6">A headless continuation.</p></div>
<div><head>Conclusions</head><p xml:id="p7">Done.</p></div>
<figure xml:id="fig_0"><head>Figure 1 .</head><label>1</label><figDesc>Figure 1. THz spectra of films. """ + "x" * 400 + """</figDesc></figure>
<figure xml:id="fig_1"><figDesc>Adv. Mater. 2023, 2211157</figDesc></figure>
<figure xml:id="fig_2"><head>Figure 4 .</head><figDesc>Figure 4. Headed but unlabelled.</figDesc></figure>
<figure type="table" xml:id="tab_0"><head>Table 2 .</head><label>2</label><figDesc>Fit parameters.</figDesc></figure>
</body></text></TEI>"""


def _soup():
    return BeautifulSoup(TEI, "xml")


def _p(soup, xml_id):
    return soup.find("p", attrs={"xml:id": xml_id})


class TestSectionBreadcrumb(unittest.TestCase):
    def test_subsection_gets_its_parent(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p3")),
                         "2. Results and Discussion > 2.1. Terahertz Spectral Analysis")

    def test_top_level_is_alone(self):
        s = _soup()
        self.assertEqual(ts.section_breadcrumb(_p(s, "p1")), "1. Introduction")
        self.assertEqual(ts.section_breadcrumb(_p(s, "p2")), "2. Results and Discussion")

    def test_unnumbered_heading_is_alone(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p7")), "Conclusions")

    def test_before_any_heading_is_empty(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p0")), "")

    def test_page_header_and_headless_divs_inherit_the_previous_heading(self):
        s = _soup()
        self.assertEqual(ts.section_breadcrumb(_p(s, "p4")),
                         "2. Results and Discussion > 2.1. Terahertz Spectral Analysis")
        self.assertEqual(ts.section_breadcrumb(_p(s, "p6")),
                         "2. Results and Discussion > 2.2. Charge Transport")

    def test_a_later_sibling_skips_the_page_header(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p5")),
                         "2. Results and Discussion > 2.2. Charge Transport")
