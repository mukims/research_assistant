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
