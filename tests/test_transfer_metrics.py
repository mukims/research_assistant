import unittest

from research_assistant.eval import transfer_metrics as m


def _case(
    id="test_case",
    kind="positive",
    query="query",
    generic="generic",
    cutoff_year=2020,
    corpus="main",
    source_titles=None,
    technique_terms=None,
    mapping_terms=None,
    critic_terms=None,
    set_aside=None,
    hidden=None,
):
    source_titles = source_titles or []
    technique_terms = technique_terms or []
    mapping_terms = mapping_terms or []
    critic_terms = critic_terms or []
    hidden = hidden or []
    return {
        "id": id,
        "kind": kind,
        "query": query,
        "generic": generic,
        "cutoff_year": cutoff_year,
        "corpus": corpus,
        "papers": {
            "source": source_titles,
            "hidden": hidden,
        },
        "hidden": hidden,
        "technique_terms": technique_terms,
        "mapping_terms": mapping_terms,
        "critic_terms": critic_terms,
        "set_aside": set_aside,
        "exclude_documents": [],
        "expected": {
            "source_titles": source_titles,
            "technique_terms": technique_terms,
            "mapping_terms": mapping_terms,
            "critic_terms": critic_terms,
            "set_aside": set_aside,
        },
    }


def _result_with(transfers):
    t_list = []
    for t in transfers:
        d = dict(t)
        if "mapping" not in d:
            d["mapping"] = []
        if "critique" not in d:
            d["critique"] = {"objections": [], "rows": []}
        t_list.append(d)
    return {"transfers": t_list, "cards": {}}


class TestTransferMetrics(unittest.TestCase):
    def test_a_technique_named_in_prose_but_not_retrieved_from_an_expected_paper_does_not_count(self):
        r = _result_with([{"card_document": "other.pdf", "card_name": "restricted Boltzmann machine", "summary": "RBM ansatz"}])
        self.assertIsNone(m.found_rank(r, _case(technique_terms=["boltzmann"]), {"hinton2002.pdf"}))

    def test_found_rank_counts_only_transfers_not_set_aside(self):
        r = _result_with([
            {"card_document": "x.pdf", "card_name": "kernel method", "set_aside": None},
            {"card_document": "hinton2002.pdf", "card_name": "restricted boltzmann machine", "set_aside": "shape_mismatch"},
            {"card_document": "hinton2002.pdf", "card_name": "restricted boltzmann machine", "set_aside": None},
        ])
        self.assertEqual(m.found_rank(r, _case(technique_terms=["boltzmann"]), {"hinton2002.pdf"}), 2)

    def test_a_negative_case_passes_only_if_every_matching_transfer_is_set_aside_as_expected(self):
        case = _case(kind="negative", technique_terms=["autoencoder"], set_aside="not_identifiable")
        good = _result_with([{"card_document": "a.pdf", "card_name": "autoencoder", "set_aside": "not_identifiable"}])
        bad = _result_with([{"card_document": "a.pdf", "card_name": "autoencoder", "set_aside": None}])
        self.assertTrue(m.case_passed(good, case, set())["passed"])
        self.assertFalse(m.case_passed(bad, case, set())["passed"])

    def test_a_negative_run_with_no_matching_transfer_is_not_tested_rather_than_failed(self):
        case = _case(kind="negative", technique_terms=["autoencoder"])
        res = m.case_passed(_result_with([{"card_document": "a.pdf", "card_name": "genetic search", "set_aside": None}]), case, set())
        self.assertEqual((res["passed"], res["tested"]), (False, False))
        self.assertTrue(m.case_passed(_result_with([]), _case(), set())["tested"])

    def test_f3_judges_a_case_on_its_tested_runs_and_cannot_pass_with_an_untested_case(self):
        records = [
            {"case_id": "neg", "passed": True, "tested": True},
            {"case_id": "neg", "passed": False, "tested": False},
            {"case_id": "pos", "passed": True},
            {"case_id": "never", "passed": False, "tested": False},
        ]
        f3 = m.score(records)["f3"]
        self.assertEqual((f3["passed"], f3["of"], f3["not_tested"]), (2, 2, 1))
        self.assertFalse(f3["all_passed"])
        self.assertTrue(m.score(records[:3])["f3"]["all_passed"])

    def test_a_mapping_term_may_be_a_group_of_synonyms(self):
        case = _case(mapping_terms=[["spin", "quantum state"], ["amplitude", "wave function"], ["energy"]])
        transfer = {"mapping": [{"source_role": "hidden units", "target_role": "quantum state and wave function"}]}
        self.assertAlmostEqual(m.mapped_share(transfer, case), 2 / 3)

    def test_the_leak_guard_stops_a_run_whose_hidden_paper_is_in_the_corpus(self):
        case = _case(hidden=["Solving the quantum many-body problem with artificial neural networks"])
        with self.assertRaisesRegex(RuntimeError, "many-body"):
            m.leak_guard(case, {"Solving the quantum many-body problem with artificial neural networks": "carleo.pdf"}, {"carleo.pdf"})

    def test_retrieval_rank_and_the_f2_rule(self):
        self.assertEqual(m.retrieval_rank([{"card_document": "x", "score": .9}, {"card_document": "k.pdf", "score": .8}], {"k.pdf"}), 2)
        rec_fail = [
            {"case_id": "c1", "retrieval_rank": 1},
            {"case_id": "c2", "retrieval_rank": 3},
            {"case_id": "c3", "retrieval_rank": 7},
            {"case_id": "c4", "retrieval_rank": None},
        ]
        s_fail = m.score(rec_fail)
        self.assertFalse(s_fail["f2"]["passed"])
        self.assertEqual(s_fail["f2"]["top5_count"], 2)

        rec_pass = [
            {"case_id": "c1", "retrieval_rank": 1},
            {"case_id": "c2", "retrieval_rank": 3},
            {"case_id": "c3", "retrieval_rank": 5},
            {"case_id": "c4", "retrieval_rank": None},
        ]
        s_pass = m.score(rec_pass)
        self.assertTrue(s_pass["f2"]["passed"])
        self.assertEqual(s_pass["f2"]["top5_count"], 3)

    def test_expected_documents_resolves_only_sources(self):
        case = _case(source_titles=["Paper A", "Paper B"])
        resolved = {"Paper A": "a.pdf", "Paper C": "c.pdf"}
        self.assertEqual(m.expected_documents(case, resolved), {"a.pdf"})

    def test_mapped_share_and_critiqued(self):
        case = _case(mapping_terms=["spin", "energy"], critic_terms=["sign", "phase"])
        transfer = {
            "mapping": [
                {"source_role": "spin configuration", "target_role": "state"},
                {"source_role": "energy function", "target_role": "cost"},
            ],
            "critique": {
                "objections": [{"objection": "The sign problem prevents convergence"}],
                "rows": [],
            },
        }
        self.assertEqual(m.mapped_share(transfer, case), 1.0)
        self.assertTrue(m.critiqued(transfer, case))

    def test_positive_case_passed_checks_rank_mapping_and_critic(self):
        case = _case(
            kind="positive",
            source_titles=["Paper A"],
            technique_terms=["boltzmann"],
            mapping_terms=["spin", "energy"],
            critic_terms=["phase"],
        )
        res_pass = _result_with([{
            "card_document": "a.pdf",
            "card_name": "Boltzmann machine",
            "set_aside": None,
            "mapping": [{"source_role": "spin energy"}],
            "critique": {"objections": [{"objection": "negative phase"}]},
        }])
        p = m.case_passed(res_pass, case, {"a.pdf"})
        self.assertTrue(p["passed"])
        self.assertEqual(p["found_rank"], 1)

        # Fails if rank > 3
        res_low_rank = _result_with([
            {"card_document": "other1.pdf", "card_name": "x", "set_aside": None},
            {"card_document": "other2.pdf", "card_name": "y", "set_aside": None},
            {"card_document": "other3.pdf", "card_name": "z", "set_aside": None},
            {
                "card_document": "a.pdf",
                "card_name": "Boltzmann machine",
                "set_aside": None,
                "mapping": [{"source_role": "spin energy"}],
                "critique": {"objections": [{"objection": "negative phase"}]},
            },
        ])
        p_low = m.case_passed(res_low_rank, case, {"a.pdf"})
        self.assertFalse(p_low["passed"])
        self.assertEqual(p_low["found_rank"], 4)

    def test_f3_score_computes_spread_with_missing_counted_as_10(self):
        records = [
            {"case_id": "c1", "found_rank": 1, "passed": True},
            {"case_id": "c1", "found_rank": 3, "passed": True},
            {"case_id": "c2", "found_rank": 2, "passed": True},
            {"case_id": "c2", "found_rank": None, "passed": False},
        ]
        s = m.score(records)
        # c1 diff: |1 - 3| = 2
        # c2 diff: |2 - 10| = 8
        # mean spread = (2 + 8) / 2 = 5.0
        self.assertEqual(s["f3"]["spread"], 5.0)
        self.assertEqual(s["f3"]["passed"], 1)  # only c1 passed both runs
        self.assertEqual(s["f3"]["of"], 2)
        self.assertFalse(s["f3"]["all_passed"])
