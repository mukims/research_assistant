import json
import unittest

from research_assistant.transfer import stages
from research_assistant.transfer.deps import TransferSettings
from research_assistant.transfer.premises import PremiseChecker
from tests.transfer_fixtures import world


class TestGround(unittest.TestCase):
    def test_a_question_the_corpus_does_not_cover_is_refused_before_anything_else(self):
        deps, log = world(gate_keeps=1)
        with self.assertRaises(stages.CorpusCoverageError) as cm:
            stages.ground(
                "lithium dendrites with MXenes",
                deps,
                TransferSettings(),
                PremiseChecker(deps.judge, deps.texts, deps.metadatas),
            )
        self.assertEqual((log.chat_calls, log.searches, log.judged), (0, [], []))
        self.assertIn("Research idea", str(cm.exception))

    def test_bottlenecks_are_grounded_in_the_kept_papers_and_shared_premises_judged_once(self):
        deps, log = world()
        g = stages.ground("q", deps, TransferSettings(), PremiseChecker(deps.judge, deps.texts, deps.metadatas))
        self.assertEqual([b.id for b in g.bottlenecks], ["B1", "B2"])
        self.assertEqual(log.searches[0][0], {"n0", "n1", "n2"})
        self.assertEqual(log.judged.count("bottleneck fact two"), 1)

    def test_a_bottleneck_says_whether_it_infers_unknowns_from_measurements(self):
        reply = json.dumps({"bottlenecks": [
            {"statement": "Representation", "quantities": ["energy"], "current_method": "DMRG",
             "inference": False, "generic": {"specific": "a", "mid": "b", "broad": "c"},
             "premises": [{"claim": "bottleneck fact one", "key": "C1"}]},
            {"statement": "Inversion", "quantities": ["conductance"], "current_method": "misfit",
             "generic": {"specific": "a", "mid": "b", "broad": "c"},
             "premises": [{"claim": "bottleneck fact two", "key": "C2"}]},
        ]})
        deps, _ = world(replies={("bottlenecks", 1): reply})
        g = stages.ground("q", deps, TransferSettings(), PremiseChecker(deps.judge, deps.texts, deps.metadatas))
        self.assertEqual([b.inference for b in g.bottlenecks], [False, True])     # missing means inference

    def test_missing_generic_levels_are_repaired_not_dropped(self):
        reply = json.dumps({"bottlenecks": [
            {"statement": "Only mid", "quantities": ["q"], "current_method": "m",
             "generic": {"mid": "the mid form"}, "premises": [{"claim": "bottleneck fact one", "key": "C1"}]},
            {"statement": "No generic forms", "quantities": ["q"], "current_method": "m",
             "premises": [{"claim": "bottleneck fact two", "key": "C2"}]},
        ]})
        deps, _ = world(replies={("bottlenecks", 1): reply})
        g = stages.ground("q", deps, TransferSettings(), PremiseChecker(deps.judge, deps.texts, deps.metadatas))
        self.assertEqual([b.generic for b in g.bottlenecks], [
            {"specific": "the mid form", "mid": "the mid form", "broad": "the mid form"},
            {"specific": "No generic forms", "mid": "No generic forms", "broad": "No generic forms"},
        ])
        self.assertTrue(any(w.startswith("Repaired generic levels") for w in g.warnings))

    def test_excluded_documents_never_reach_the_gate_or_the_search(self):
        deps, log = world()
        stages.ground(
            "q",
            deps,
            TransferSettings(exclude_documents=frozenset({"n0"})),
            PremiseChecker(deps.judge, deps.texts, deps.metadatas),
        )
        self.assertEqual(log.searches[0], ({"n1", "n2", "n3"}, {"n0"}))


class TestCandidates(unittest.TestCase):
    def setUp(self):
        self.deps, self.log = world()
        self.g = stages.ground("q", self.deps, TransferSettings(), PremiseChecker(self.deps.judge, self.deps.texts, self.deps.metadatas))

    def test_each_bottleneck_gets_its_nearest_verified_cards_and_never_a_near_paper_or_unverified_card(self):
        seeds, _ = stages.candidate_cards(self.g, self.deps, TransferSettings())
        by_b = {b: sorted(s.card.document for s in seeds if s.bottleneck_id == b) for b in ("B1", "B2")}
        self.assertEqual(by_b, {"B1": ["s1", "s2", "s3"], "B2": ["s4", "s5", "s6"]})

    def test_another_copy_of_a_near_paper_is_never_a_source(self):
        copy = next(c for c in self.deps.cards.cards if c.document == "s1")
        copy.citation = "CITATION n-0"          # the same title as near paper n0, filed under another name
        seeds, _ = stages.candidate_cards(self.g, self.deps, TransferSettings())
        self.assertNotIn("s1", {s.card.document for s in seeds})

    def test_every_seed_records_its_level_and_rank(self):
        seeds, _ = stages.candidate_cards(self.g, self.deps, TransferSettings())
        s1 = next(s for s in seeds if s.card.document == "s1")
        self.assertEqual((s1.level, s1.rank), ("specific", 1))

    def test_an_empty_card_index_reports_a_corpus_gap(self):
        import numpy as np
        from research_assistant.transfer.cards import CardIndex
        self.deps.cards = CardIndex([], np.zeros((0, 2)))
        seeds, warnings = stages.candidate_cards(self.g, self.deps, TransferSettings())
        self.assertEqual(seeds, [])
        self.assertTrue(any(w.startswith("corpus gap for B1") for w in warnings))


from research_assistant.shared import pipeline_status
from tests.transfer_fixtures import good_transfer_reply


class TestDevelop(unittest.TestCase):
    def _go(self, **kw):
        deps, log = world(**kw)
        s = TransferSettings()
        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        g = stages.ground("q", deps, s, ch)
        seeds, _ = stages.candidate_cards(g, deps, s)
        return deps, log, g, seeds, s, ch

    def test_a_complete_transfer_has_source_premises_critique_answers_and_prior_art(self):
        deps, log, g, seeds, s, ch = self._go()
        t = stages.develop(seeds[0], 1, g, deps, s, ch)
        self.assertEqual(t.id, "B1.T1")
        self.assertEqual(t.card_document, "s1")
        self.assertTrue(any(p.document == "s1" and p.accepted for p in t.premises))
        self.assertTrue(all(o["answered"] for o in t.critique.objections))
        self.assertEqual(t.prior_art.status, "not_found")
        self.assertEqual(log.critic_calls, 1)

    def test_a_premise_citing_a_passage_from_another_prompt_is_dropped(self):
        bad = good_transfer_reply(premises=[{"claim": "x", "key": "C3"}, {"claim": "source fact", "key": "C4"}])  # fixture helper
        deps, log, g, seeds, s, ch = self._go(replies={("transfer", 1): [bad]})
        t = stages.develop(seeds[0], 1, g, deps, s, ch)          # B1's prompt shows C1, C2 and the source passage C4
        self.assertEqual([p.key for p in t.premises], ["C4"])
        self.assertTrue(any("C3" in w for w in t.warnings))

    def test_a_reply_that_never_parses_is_skipped_with_a_warning(self):
        deps, log, g, seeds, s, ch = self._go(replies={("transfer", 1): ["nope", "still nope"]})
        self.assertIsNone(stages.develop(seeds[0], 1, g, deps, s, ch))
        self.assertTrue(any(w.startswith("B1.T1:") for w in g.warnings))

    def test_a_row_the_critic_breaks_goes_to_the_reviser_as_an_objection_on_that_row(self):
        revise = json.dumps({"responses": [
            {"objection": 0, "response": "Addressed objection with concrete argument.", "concede": False},
            {"objection": 1, "response": "The surrogate never assumes long-range order.", "concede": False},
        ]})
        deps, log, g, seeds, s, ch = self._go(replies={("revise", 1): revise})
        deps.critic = lambda m: json.dumps({
            "shape_match": "yes",
            "rows": [{"row": 0, "status": "breaks", "reason": "2D has no order"}],
            "objections": ["Sample critic objection"],
        })
        t = stages.develop(seeds[0], 1, g, deps, s, ch)
        row_objection = t.critique.objections[1]
        self.assertEqual(row_objection["row"], 0)
        self.assertIn("2D has no order", row_objection["text"])
        self.assertEqual((row_objection["answered"], row_objection["conceded"]), (True, False))
        self.assertIsNone(t.critique.objections[0]["row"])

    def test_a_failing_critic_leaves_an_empty_critique_not_a_crash(self):
        deps, log, g, seeds, s, ch = self._go()
        deps.critic = lambda m: (_ for _ in ()).throw(RuntimeError("down"))
        t = stages.develop(seeds[0], 1, g, deps, s, ch)
        self.assertEqual(t.critique.objections, [])
        self.assertTrue(any("critic" in w for w in t.warnings))

    def test_cancelling_stops_before_the_transfer(self):
        deps, log, g, seeds, s, ch = self._go()
        deps.cancelled = lambda: True
        with self.assertRaises(pipeline_status.PipelineCancelledError):
            stages.develop(seeds[0], 1, g, deps, s, ch)


