import io
import json
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
import unittest
from unittest.mock import patch

from research_assistant.eval import transfer_cases
from research_assistant.shared.ingestion import pdf_key
from scripts import build_case_corpus as bcc


class TestBuildCaseCorpus(unittest.TestCase):
    def test_it_refuses_to_build_into_any_directory_but_the_cases_own(self):
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(bcc.main(["--case", "nqs", "--resolve"]), 4)      # DATA_DIR is the main data directory in tests
        self.assertIn("CITATION_DATA_DIR", err.getvalue())

    def test_titles_resolve_only_on_a_close_match_with_a_pdf(self):
        hits = [
            {"title": "Training Products of Experts by Minimizing Contrastive Divergence", "year": "2002", "pdf_url": "u"},
            {"title": "Something unrelated", "year": "2002", "pdf_url": "v"},
        ]
        got = bcc.resolve_titles(
            ["Training products of experts by minimizing contrastive divergence", "Nothing like it"],
            search=lambda q, limit: hits,
        )
        self.assertEqual([g["match_title"] is not None for g in got], [True, False])

    def test_among_equal_titles_an_arxiv_pdf_then_the_earliest_year_wins(self):
        hits = [
            {"title": "Auto-Encoding Variational Bayes", "year": 2024, "pdf_url": "https://junk.example/33.pdf"},
            {"title": "Auto-Encoding Variational Bayes", "year": 2013, "pdf_url": "https://uni.example/page.html"},
            {"title": "Auto-Encoding Variational Bayes", "year": 2014, "pdf_url": "https://arxiv.org/pdf/1312.6114"},
            {"title": "Auto-Encoding Variational Bayes", "year": 2012, "pdf_url": None},
        ]
        got = bcc.resolve_titles(["Auto-encoding variational Bayes"], search=lambda q, limit: hits)
        self.assertEqual(got[0]["pdf_url"], "https://arxiv.org/pdf/1312.6114")

    def test_the_default_search_asks_arxiv_and_openalex_even_when_arxiv_answers(self):
        arxiv = [{"title": "Information Constraints on Auto-Encoding Variational Bayes", "pdf_url": "a"}]
        openalex = [{"title": "Auto-Encoding Variational Bayes", "pdf_url": "b"}]
        with patch.object(bcc.agent0_discoverer, "_search_arxiv", return_value=arxiv), \
                patch.object(bcc.agent0_discoverer, "_search_openalex", return_value=openalex):
            self.assertEqual(bcc.default_search("Auto-encoding variational Bayes"), arxiv + openalex)

    def test_titles_that_share_their_first_40_characters_get_different_files(self):
        a = bcc.pdf_filename("The density-matrix renormalization group")
        b = bcc.pdf_filename("The density-matrix renormalization group in the age of matrix product states")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("The_density-matrix_renormalization_group") and a.endswith(".pdf"))
        self.assertEqual(a, bcc.pdf_filename("The density-matrix renormalization group"))

    def test_resolved_names_match_the_document_ids_the_ingest_stores(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "resolution.json"), "w", encoding="utf-8") as fh:
                json.dump([{"title": "The Kernel Polynomial Method", "pdf_url": "u"}], fh)
            written = {}
            with patch.object(bcc, "guard_data_dir", return_value=None), \
                    patch.object(bcc, "case_data_dir", return_value=d), \
                    patch.object(bcc, "download_pdf", return_value=(True, None)), \
                    patch.object(bcc, "ingest_pdfs") as ingest, \
                    patch.object(bcc, "atomic_write_json", side_effect=lambda p, data: written.update({p: data})), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(bcc.main(["--case", "kpm", "--confirm"]), 0)
            stored = {pdf_key(p) for p in ingest.call_args.args[0]}
            self.assertEqual(set(written[os.path.join(d, "resolved.json")].values()), stored)

    def test_hidden_papers_are_never_resolved(self):
        recorded = []

        def fake_resolve(titles, search=None):
            recorded.extend(titles)
            return []

        with patch.object(bcc, "guard_data_dir"), patch.object(bcc, "resolve_titles", side_effect=fake_resolve):
            with patch.object(bcc, "atomic_write_json"):
                bcc.main(["--case", "nqs", "--resolve"])

        case_nqs = next(c for c in transfer_cases.load_cases() if c.id == "nqs")
        for h in case_nqs.papers.get("hidden", []):
            self.assertNotIn(h, recorded)
