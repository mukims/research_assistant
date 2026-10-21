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

