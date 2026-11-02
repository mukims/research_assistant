"""
Adversarial Empirical Stress Test Suite for Milestone 2 (Tier P1).
Authored by Challenger M2 1 (teamwork_challenger_m2_1).

Challenges and empirically verifies:
1. Minus-sign normalization in research_assistant/judgement/judge.py:
   - Negative physics values (-0.5 eV, -120 °C, -50 K, 10^-5, -3.2e-4) preservation.
   - Word hyphen un-breaking (energy-\\n dependent, cross-\\n section).
   - Verbatim span matching with sign differentiation.
2. TEI Table and Formula Extraction in research_assistant/shared/extract.py:
   - Multi-row, multi-column tables with empty cells, ragged rows, pipes, and headers into valid Markdown.
   - <formula> tag extraction into Document.formulas with ID, page, coords, and bounding box.
3. LanceDB retrieval without row caps in research_assistant/shared/lance_store.py:
   - Real on-disk LanceDB table retrieval with limit=None on >10 rows (25 rows).
"""

import tempfile
import unittest
from bs4 import BeautifulSoup
from lxml import etree

from research_assistant.judgement import judge
from research_assistant.shared import extract, lance_store


class TestAdversarialMinusSignAndHyphen(unittest.TestCase):
    """Area 1: Minus-sign normalization and line-hyphenation un-breaking."""

    def test_negative_physics_values_preservation(self):
        """Negative physics quantities must NOT be stripped or collapsed to positive values."""
        cases = [
            ("-0.5 eV", "0.5 eV"),
            ("-120 °C", "120 °C"),
            ("-50 K", "50 K"),
            ("10^-5", "10^5"),
            ("-3.2e-4", "3.2e-4"),
            ("-1.602e-19 J", "1.602e-19 J"),
            ("-40 °F", "40 °F"),
            ("-0.001 s^-1", "0.001 s^-1"),
            ("-273.15 °C", "273.15 °C"),
        ]
        for neg_val, pos_val in cases:
            norm_neg = judge._normalise(neg_val)
            norm_pos = judge._normalise(pos_val)
            self.assertNotEqual(
                norm_neg,
                norm_pos,
                f"Negative value {neg_val!r} collapsed to positive {pos_val!r}: norm={norm_neg!r}",
            )
            if neg_val.startswith("-"):
                self.assertTrue(
                    norm_neg.startswith("-"),
                    f"Negative value {neg_val!r} must preserve leading minus sign: got {norm_neg!r}",
                )
                self.assertFalse(
                    norm_pos.startswith("-"),
                    f"Positive value {pos_val!r} should not have leading minus sign: got {norm_pos!r}",
                )

    def test_word_hyphen_unbreaking(self):
        """Hyphenated words split across lines or within words must be unified cleanly."""
        cases = [
            ("energy-\n dependent", "energydependent"),
            ("cross-\n section", "crosssection"),
            ("non-\n linear", "nonlinear"),
            ("quasi-\n particle", "quasiparticle"),
            ("informa-\n tion", "information"),
            ("energy-\n   dependent", "energydependent"),
            ("cross-\r\n section", "crosssection"),
        ]
        for hyphenated, expected in cases:
            norm = judge._normalise(hyphenated)
            self.assertEqual(
                norm,
                expected,
                f"Hyphenated word {hyphenated!r} failed to un-hyphenate to {expected!r}: got {norm!r}",
            )

    def test_span_is_verbatim_sign_rejection_and_acceptance(self):
        """Verbatim span matching must reject sign-inverted claims and accept exact matches."""
        # Case A: Evidence is positive, claim is negative -> MUST REJECT
        ev_pos = "The measured band gap was 0.5 eV under ambient conditions."
        self.assertFalse(
            judge.span_is_verbatim("-0.5 eV", ev_pos),
            "Negative claim '-0.5 eV' must be rejected against positive evidence '0.5 eV'",
        )
        self.assertTrue(
            judge.span_is_verbatim("0.5 eV", ev_pos),
            "Positive claim '0.5 eV' must match positive evidence",
        )

        # Case B: Evidence is negative, claim matches exact negative -> MUST ACCEPT
        ev_neg = "The thermal drift measured was -0.5 eV in the substrate."
        self.assertTrue(
            judge.span_is_verbatim("-0.5 eV", ev_neg),
            "Negative claim '-0.5 eV' must match negative evidence exactly",
        )

        # Case C: Exponents sign differentiation
        ev_exp = "The relaxation rate was 10^5 s^-1."
        self.assertFalse(
            judge.span_is_verbatim("10^-5", ev_exp),
            "Negative exponent '10^-5' must not match positive exponent '10^5'",
        )

        # Case D: Hyphenated words matching across line breaks
        ev_hyphen = "We observed energy-\n dependent damping in the acoustic phonon mode."
        self.assertTrue(
            judge.span_is_verbatim("energy-dependent damping", ev_hyphen),
            "Span with 'energy-dependent' must match evidence with 'energy-\\n dependent'",
        )

    def test_en_dash_and_em_dash_normalization(self):
        """En-dash and em-dash minus equivalents should normalize to ASCII minus."""
        # Unicode en-dash \u2013 and em-dash \u2014 are in _QUOTES translation table
        norm_en = judge._normalise("–0.5 eV")
        norm_em = judge._normalise("—0.5 eV")
        norm_ascii = judge._normalise("-0.5 eV")
        self.assertEqual(norm_en, norm_ascii, "En-dash minus must normalize to ASCII minus")
        self.assertEqual(norm_em, norm_ascii, "Em-dash minus must normalize to ASCII minus")


class TestAdversarialTEITablesAndFormulas(unittest.TestCase):
    """Area 2: TEI Table and Formula Extraction."""

    def test_multi_row_multi_column_table_with_empty_cells_and_headers(self):
        """TEI table with empty cells, ragged rows, and pipe characters converts to valid Markdown."""
        table_xml = """<table xmlns="http://www.tei-c.org/ns/1.0">
          <row role="label">
            <cell>Sample ID</cell>
            <cell>Composition</cell>
            <cell>T_c (K)</cell>
            <cell>Notes</cell>
          </row>
          <row>
            <cell>S-01</cell>
            <cell></cell>
            <cell>38.5</cell>
            <cell>Underdoped</cell>
          </row>
          <row>
            <cell>S-02</cell>
            <cell>YBa2Cu3O7-x</cell>
            <cell>92.0</cell>
          </row>
          <row>
            <cell>S-03 | pristine</cell>
            <cell>Bi2Sr2CaCu2O8+d</cell>
            <cell>110.2</cell>
            <cell>High purity</cell>
          </row>
        </table>"""
        el = etree.fromstring(table_xml)
        md = extract._format_tei_table(el)

        lines = md.split("\n")
        # Check header
        self.assertEqual(lines[0], "| Sample ID | Composition | T_c (K) | Notes |")
        # Check separator line with 4 columns
        self.assertEqual(lines[1], "| --- | --- | --- | --- |")
        # Check row with empty cell
        self.assertEqual(lines[2], "| S-01 |  | 38.5 | Underdoped |")
        # Check ragged row padded with empty string to 4 columns
        self.assertEqual(lines[3], "| S-02 | YBa2Cu3O7-x | 92.0 |  |")
        # Check row with escaped pipe
        self.assertEqual(lines[4], "| S-03 \\| pristine | Bi2Sr2CaCu2O8+d | 110.2 | High purity |")

        # Verify all lines have equal column delimiters (5 pipes = 4 columns)
        for i, line in enumerate(lines):
            # Escaped pipe does not count as column separator
            unescaped_pipes = line.replace("\\|", "").count("|")
            self.assertEqual(
                unescaped_pipes,
                5,
                f"Line {i} does not have exactly 4 columns (5 pipes): {line!r}",
            )

    def test_tei_table_beautifulsoup_parity(self):
        """BeautifulSoup elements format identically to lxml elements."""
        table_xml = """<table xmlns="http://www.tei-c.org/ns/1.0">
          <row role="label"><cell>A</cell><cell>B</cell></row>
          <row><cell>1</cell><cell>2</cell></row>
        </table>"""
        el_lxml = etree.fromstring(table_xml)
        soup = BeautifulSoup(table_xml, "xml")
        el_bs = soup.find("table")

        md_lxml = extract._format_tei_table(el_lxml)
        md_bs = extract._format_tei_table(el_bs)
        self.assertEqual(md_lxml, md_bs)
        self.assertEqual(md_lxml, "| A | B |\n| --- | --- |\n| 1 | 2 |")

    def test_tei_table_empty_and_none_handled(self):
        """Empty table or None returns empty string without crashing."""
        self.assertEqual(extract._format_tei_table(None), "")
        empty_el = etree.fromstring("<table xmlns='http://www.tei-c.org/ns/1.0'/>")
        self.assertEqual(extract._format_tei_table(empty_el), "")

    def test_formula_extraction_in_parse_tei(self):
        """parse_tei extracts all <formula> elements with ID, page, coords, and bbox."""
        tei_xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:xml="http://www.w3.org/XML/1998/namespace">
          <teiHeader><fileDesc><titleStmt><title>Quantum Formulas</title></titleStmt></fileDesc></teiHeader>
          <text>
            <body>
              <div n="1">
                <head>Methods</head>
                <p coords="1,50.0,100.0,300.0,50.0">
                  The Hamiltonian is:
                  <formula xml:id="eq_ham" coords="1,60.0,120.0,200.0,25.0">H = \\sum_k \\epsilon_k c_k^\\dagger c_k</formula>
                </p>
                <p coords="2,50.0,100.0,300.0,50.0">
                  The gap equation is:
                  <formula coords="2,60.0,130.0,180.0,20.0">\\Delta = 2 \\omega_D e^{-1/N(0)V}</formula>
                </p>
              </div>
            </body>
          </text>
        </TEI>"""
        doc = extract.parse_tei(tei_xml.encode("utf-8"), key="test_key_quant")
        self.assertEqual(doc.key, "test_key_quant")
        self.assertEqual(len(doc.formulas), 2)

        # Formula 1 with explicit xml:id
        f1 = doc.formulas[0]
        self.assertEqual(f1.id, "eq_ham")
        self.assertEqual(f1.text, "H = \\sum_k \\epsilon_k c_k^\\dagger c_k")
        self.assertEqual(f1.page, 0)  # 1-based "1,..." converted to 0-based
        self.assertEqual(f1.coords, "1,60.0,120.0,200.0,25.0")
        self.assertEqual(f1.bbox, (60.0, 120.0, 260.0, 145.0))

        # Formula 2 without xml:id (fallback index)
        f2 = doc.formulas[1]
        self.assertEqual(f2.id, "formula_1")
        self.assertEqual(f2.text, "\\Delta = 2 \\omega_D e^{-1/N(0)V}")
        self.assertEqual(f2.page, 1)  # 1-based "2,..." converted to 0-based
        self.assertEqual(f2.coords, "2,60.0,130.0,180.0,20.0")
        self.assertEqual(f2.bbox, (60.0, 130.0, 240.0, 150.0))


class TestAdversarialLanceDBRowCaps(unittest.TestCase):
    """Area 3: LanceDB retrieval without row caps."""

    def test_lancedb_get_without_limit_returns_all_25_rows(self):
        """LanceTableAdapter.get(limit=None) on a table with 25 rows returns ALL 25 rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = lance_store.LanceStoreClient(path=tmpdir)
            col = client.get_or_create_collection("adversarial_chunks", {"embedding_dim": 768})

            n_rows = 25
            ids = [f"chunk_{i:02d}" for i in range(n_rows)]
            docs = [f"Text passage content for index {i}" for i in range(n_rows)]
            metas = [
                {"document": f"paper_{i % 3}.pdf", "seq": i, "stored_form": "text_chunk"}
                for i in range(n_rows)
            ]
            vecs = [[0.05 * (i % 7)] * 768 for i in range(n_rows)]

            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
            self.assertEqual(col.count(), n_rows)

            # Test 1: get(limit=None) MUST return all 25 rows (not capped at LanceDB default 10)
            res_all = col.get(limit=None)
            self.assertEqual(
                len(res_all["ids"]),
                25,
                f"Expected 25 rows when limit=None, but got {len(res_all['ids'])} (10-row cap bug!)",
            )
            self.assertEqual(set(res_all["ids"]), set(ids))

            # Test 2: explicit limit=5 returns exactly 5 rows
            res_5 = col.get(limit=5)
            self.assertEqual(len(res_5["ids"]), 5)

            # Test 3: explicit limit=15 returns exactly 15 rows (>10, verifying explicit limit)
            res_15 = col.get(limit=15)
            self.assertEqual(len(res_15["ids"]), 15)

            # Test 4: where filter + limit=None returns all matching rows (>10 if matching)
            # paper_0 has indices 0, 3, 6, 9, 12, 15, 18, 21, 24 (9 rows)
            # paper_1 has indices 1, 4, 7, 10, 13, 16, 19, 22 (8 rows)
            # filter for stored_form="text_chunk" matches all 25
            res_filtered = col.get(where={"type": "text_chunk"}, limit=None)
            self.assertEqual(len(res_filtered["ids"]), 25)


if __name__ == "__main__":
    unittest.main()
