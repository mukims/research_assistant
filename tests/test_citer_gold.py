"""The citer's ground truth is a seed paper's own citations: which sentence
the author had cite which work. These pin how a record is built — an
in-corpus reference keeps the sentence, an unavailable one is dropped from
it, and the negatives are sampled from paragraphs that cite nothing."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.eval import citer_gold as cg

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>Seed</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
  <text><body>
    <div><head n="2.">Results</head>
      <p xml:id="p_cited">Graphene shows ballistic transport at low temperature <ref type="bibr" target="#b0">[1]</ref>. A second sentence cites something we do not hold <ref type="bibr" target="#b1">[2]</ref>.</p>
    </div>
    <div><head n="3.">Discussion</head>
      <p xml:id="p_plain">This paragraph makes no citation at all and has enough words to count. Another sentence with plenty of words that also cites nothing at all here.</p>
      <p xml:id="p_short">Too short to count.</p>
      <note place="foot"><p>0 F E B R U A R Y 2 0 1 4 | V O L 5 0 6 | N A T U R E | one two three four five six seven eight</p></note>
      <p xml:id="p_symbols">E F 5 0 V g 5 0 G 5 0.95 G 0 T 5 4.2 K L 1,3 L 4,6.</p>
    </div>
  </body>
  <back><div type="references"><listBibl>
    <biblStruct xml:id="b0"><analytic><title level="a" type="main">Held paper</title>
      <author><persName><forename>A.</forename><surname>Held</surname></persName></author>
      <idno type="DOI">10.1/held</idno></analytic></biblStruct>
    <biblStruct xml:id="b1"><analytic><title level="a" type="main">Missing paper</title>
      <author><persName><forename>B.</forename><surname>Missing</surname></persName></author>
      <idno type="DOI">10.1/missing</idno></analytic></biblStruct>
  </listBibl></div></back></text>
</TEI>"""


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tei = os.path.join(self.tmp.name, "seed.grobid.tei.xml")
        with open(self.tei, "w", encoding="utf-8") as fh:
            fh.write(TEI)
        self.held_pdf = os.path.join(self.tmp.name, "doi_10.1_held.pdf")
        with open(self.held_pdf, "wb") as fh:
            fh.write(b"%PDF-1.4 stub")
        self.manifest = {"doi:10.1/held": {
            "key": "doi:10.1/held", "path": self.held_pdf, "doi": "10.1/held",
            "title": "Held paper", "cited_by": "seed.pdf", "xml_id": "b0",
        }}

    def _build(self, **kw):
        with patch.object(cg, "find_tei_for_seed", return_value=self.tei), \
             patch.object(cg, "_load_downloaded_manifest", return_value=self.manifest):
            return cg.build(os.path.join(self.tmp.name, "seed.pdf"), **kw)

    def test_a_cited_sentence_with_an_in_corpus_reference_is_a_record(self):
        cited = [r for r in self._build() if r["kind"] == "cited"]
        self.assertEqual(len(cited), 1)
        rec = cited[0]
        self.assertEqual(rec["sentence"], "Graphene shows ballistic transport at low temperature.")
        self.assertEqual(rec["author_documents"], ["doi_10.1_held.pdf"])
        self.assertEqual(rec["author_refs"], [{"index": 1, "title": "Held paper"}])
        self.assertEqual(rec["roles"], ["evidential"])
        self.assertEqual(rec["section"], "results")
        self.assertEqual(rec["section_heading"], "2. Results")
        self.assertEqual(rec["paragraph_id"], "p_cited")
        # The window is the audit's display form: the marked sentence keeps its
        # rendered citation marker, exactly as the judge sees it today.
        self.assertIn("«Graphene shows ballistic transport at low temperature", rec["context"])
        self.assertIn("A second sentence cites something we do not hold", rec["context"])
        self.assertEqual(rec["seed"], "seed")
        self.assertEqual(rec["id"], "c_seed_0001")

    def test_a_sentence_whose_only_reference_is_not_in_the_corpus_is_dropped(self):
        sentences = [r["sentence"] for r in self._build() if r["kind"] == "cited"]
        self.assertNotIn("A second sentence cites something we do not hold.", sentences)

    def test_negatives_come_from_uncited_paragraphs_and_match_the_cited_count(self):
        out = self._build()
        neg = [r for r in out if r["kind"] == "uncited"]
        self.assertEqual(len(neg), 1)                      # as many as cited records
        self.assertEqual(neg[0]["paragraph_id"], "p_plain")
        self.assertEqual(neg[0]["author_documents"], [])
        self.assertGreaterEqual(len(neg[0]["sentence"].split()), cg.MIN_WORDS)
        self.assertEqual(neg[0]["id"], "u_seed_0001")
        self.assertEqual(neg[0]["section"], "discussion")

    def test_the_short_paragraph_is_never_a_negative(self):
        pool = cg.uncited_pool(self.tei)
        self.assertEqual({p["paragraph_id"] for p in pool}, {"p_plain"})
        self.assertEqual(len(pool), 2)

    def test_page_furniture_and_symbol_soup_are_never_negatives(self):
        pool = cg.uncited_pool(self.tei)
        texts = [r["sentence"] for r in pool]
        self.assertFalse(any("N A T U R E" in t for t in texts))          # <note> paragraphs are skipped
        self.assertFalse(any(t.startswith("E F 5 0") for t in texts))      # 8 whitespace tokens, 0 words
        self.assertNotIn("p_symbols", {r["paragraph_id"] for r in pool})

    def test_negatives_are_sampled_deterministically(self):
        self.assertEqual(self._build(), self._build())
        a = [r["sentence"] for r in self._build(negatives_seed=1) if r["kind"] == "uncited"]
        b = [r["sentence"] for r in self._build(negatives_seed=1) if r["kind"] == "uncited"]
        self.assertEqual(a, b)

    def test_id_prefix_tells_same_month_seeds_apart(self):
        self.assertEqual(cg._short("arxiv_2108.10114v3"), "2108.10114")
        self.assertNotEqual(cg._short("arxiv_2108.10114v3"), cg._short("arxiv_2108.99999v1"))
        self.assertEqual(cg._short("doi_10.1038_nature12952"), "doi_10.1038_nature12952")

    def test_no_tei_raises(self):
        with patch.object(cg, "find_tei_for_seed", return_value=None):
            with self.assertRaises(FileNotFoundError):
                cg.build("/nowhere/seed.pdf")


class TestLoadCases(unittest.TestCase):
    def _rec(self, id, **over):
        d = {"id": id, "seed": "s", "kind": "cited", "sentence": "S.", "context": "«S.»",
             "section": "other", "section_heading": "", "paragraph_id": "p_0", "sentence_index": 0,
             "author_documents": ["a.pdf"], "author_refs": [], "roles": ["evidential"]}
        d.update(over)
        return d

    def test_round_trip_and_duplicate_ids_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "citer_s.jsonl")
            cg.write_cases([self._rec("c_1"), self._rec("u_1", kind="uncited", author_documents=[])], path)
            cases = cg.load_cases([path])
            self.assertEqual([c["id"] for c in cases], ["c_1", "u_1"])
            cg.write_cases([self._rec("c_1"), self._rec("c_1")], path)
            with self.assertRaises(ValueError):
                cg.load_cases([path])

    def test_a_record_missing_a_required_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "citer_s.jsonl")
            rec = self._rec("c_1")
            del rec["author_documents"]
            with open(path, "w") as fh:
                fh.write(json.dumps(rec) + "\n")
            with self.assertRaises(ValueError):
                cg.load_cases([path])
