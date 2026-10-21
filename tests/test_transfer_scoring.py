import unittest

from research_assistant.transfer import scoring
from tests.transfer_fixtures import good_transfer, premise


class TestScoring(unittest.TestCase):
    def test_the_fixture_transfer_passes_every_gate_and_every_depth_check(self):
        t, b, c = good_transfer()
        s = scoring.score_transfer(t, b, c)
        self.assertEqual((s.set_aside, s.depth, s.reward), (None, 1.0, 1.25))

    def test_each_gate_sets_the_transfer_aside_with_its_reason(self):
        cases = {
            "contradicted": lambda t, b, c: t.premises.append(premise("Contradicts", "C9", "src.pdf", 9)),
            "ungrounded_bottleneck": lambda t, b, c: setattr(b, "premises", [premise("Unclear / insufficient evidence")]),
            "unverified_technique": lambda t, b, c: setattr(c, "evidence", premise("Does not support")),
            "shape_mismatch": lambda t, b, c: setattr(t.critique, "shape_match", "no"),
            "core_assumption_breaks": lambda t, b, c: t.critique.rows.append({"row": 0, "status": "breaks", "reason": "2D has no order"}),
            "not_identifiable": lambda t, b, c: setattr(t.critique, "identifiability", "underdetermined"),
            "off_target": lambda t, b, c: setattr(t, "metric", "wall-clock time"),
            "done_before": lambda t, b, c: setattr(t.prior_art, "status", "done_before"),
        }
        for reason, spoil in cases.items():
            t, b, c = good_transfer()
            spoil(t, b, c)
            s = scoring.score_transfer(t, b, c)
            self.assertEqual((s.set_aside, s.reward), (reason, 0.0), reason)

    def test_each_depth_item_fails_on_its_own(self):
        spoils = {
            "data_quantified": ("data_requirement", "a lot of samples"),
            "baseline_is_current": ("baseline", "random search"),
            "prediction_directional": ("prediction", "the method will behave differently overall"),
            "refutable": ("refuting_result", "the conductance error will decrease twofold at equal cost"),
        }
        for name, (field, value) in spoils.items():
            t, b, c = good_transfer()
            setattr(t, field, value)
            checks = scoring.depth_checks(t, b, c)
            self.assertEqual([k for k, ok in checks.items() if not ok], [name], name)

    def test_without_novelty_the_reward_is_the_depth(self):
        t, b, c = good_transfer()
        t.prior_art.status = "partial"
        self.assertEqual(scoring.score_transfer(t, b, c).reward, 1.0)

    def test_rank_puts_set_aside_transfers_last(self):
        t1, b, c = good_transfer()
        t1.id = "T1"
        t2, _, _ = good_transfer()
        t2.id = "T2"
        t3, _, _ = good_transfer()
        t3.id = "T3"
        scoring.score_transfer(t1, b, c)  # reward 1.25
        t2.prior_art.status = "partial"
        scoring.score_transfer(t2, b, c)
        t2.scores.reward = 0.5
        t3.critique.shape_match = "no"
        scoring.score_transfer(t3, b, c)  # set_aside = "shape_mismatch", reward 0.0
        ranked = scoring.rank([t2, t3, t1])
        self.assertEqual([t.id for t in ranked], ["T1", "T2", "T3"])
