import unittest

from research_assistant.judgement import metrics as mx
from research_assistant.judgement.evalset import validate_case


def _r(
    id,
    expected,
    got,
    source="human",
    transform=None,
    accept=None,
    run=1,
    seconds=1.0,
    verdict=None,
    error=None,
):
    case = validate_case({
        "id": id,
        "source": source,
        "transform": transform,
        "claim": "c",
        "citation_evidence": "e",
        "expected_judgement": expected,
        "accept": accept,
    })
    return {"case": case, "got": got, "verdict": verdict, "seconds": seconds, "run": run, "error": error}


class TestScore(unittest.TestCase):
    def test_accuracies_confusion_and_cd_cells(self):
        rs = [
            _r("1", "Supports", "Supports"),
            _r("2", "Contradicts", "Does not support"),
            _r("3", "Does not support", "Contradicts", source="transform", transform="scope_swap"),
            _r(
                "4",
                "Does not support",
                "Unclear / insufficient evidence",
                accept=["Unclear / insufficient evidence"],
                source="transform",
                transform="delete_key_sentence",
            ),
        ]
        s = mx.score(rs)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["strict_acc"], 0.25)
        self.assertAlmostEqual(s["lenient_acc"], 0.5)
        self.assertEqual(s["confusion"]["Contradicts"]["Does not support"], 1)
        self.assertEqual(s["cd_confusion"], {"contradicts_as_dns": 1, "dns_as_contradicts": 1})
        self.assertEqual(s["by_source"]["transform"]["n"], 2)
        self.assertEqual(s["by_transform"]["delete_key_sentence"]["lenient"], 1.0)
        self.assertAlmostEqual(s["per_class"]["Supports"]["recall"], 1.0)

    def test_signals_come_from_verdicts_and_are_none_when_absent(self):
        rs = [
            _r(
                "1",
                "Supports",
                "Supports",
                verdict={
                    "rubric_mismatch": True,
                    "span_verified": False,
                    "escalated": True,
                    "first_judgement": "Unclear / insufficient evidence",
                    "judgement": "Supports",
                },
            ),
            _r(
                "2",
                "Supports",
                "Supports",
                verdict={
                    "rubric_mismatch": False,
                    "span_verified": True,
                    "escalated": False,
                    "judgement": "Supports",
                },
            ),
        ]
        s = mx.score(rs)["signals"]
        self.assertEqual(
            (s["rubric_mismatch"], s["span_unverified"], s["escalated"], s["escalation_changed"]),
            (0.5, 0.5, 0.5, 1.0),
        )
        self.assertIsNone(
            mx.score([_r("1", "Supports", "Supports", verdict={"judgement": "Supports"})])["signals"][
                "rubric_mismatch"
            ]
        )

    def test_stability_over_runs_and_errors(self):
        rs = [
            _r("1", "Supports", "Supports", run=1),
            _r("1", "Supports", "Supports", run=2),
            _r("2", "Supports", "Supports", run=1),
            _r("2", "Supports", "Contradicts", run=2),
            _r("3", "Supports", None, run=1, error="parse"),
            _r("3", "Supports", None, run=2, error="parse"),
        ]
        s = mx.score(rs)
        self.assertAlmostEqual(s["stability"], 0.5)  # case 1 stable, case 2 not; errored cases excluded
        self.assertEqual(s["errors"], 2)
        self.assertEqual(s["n"], 4)  # scored results exclude errors

    def test_render_and_compare_do_not_crash(self):
        a = mx.score([_r("1", "Supports", "Supports")])
        b = mx.score([_r("1", "Supports", "Contradicts")])
        self.assertIn("strict", mx.render(a, refused=[]))
        self.assertIn(
            "1",
            mx.compare(
                {"summary": a, "results": [{"id": "1", "got": "Supports"}]},
                {"summary": b, "results": [{"id": "1", "got": "Contradicts"}]},
            ),
        )


if __name__ == "__main__":
    unittest.main()
