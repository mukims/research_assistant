import unittest

from research_assistant.transfer import models, replies

P = {
    "C1": models.Passage("C1", 4, "a.pdf", "A", "methods", "t", "near"),
    "C2": models.Passage("C2", 9, "b.pdf", "B", "other", "t", "source"),
}


class TestReplies(unittest.TestCase):
    def test_fenced_bare_and_prose_wrapped_replies_parse(self):
        for raw in ['```json\n{"a": 1}\n```', '{"a": 1}', 'Sure: {"a": 1} done']:
            self.assertEqual(replies.parse_reply(raw, ("a",)), {"a": 1})

    def test_a_missing_required_key_is_rejected(self):
        with self.assertRaises(replies.ReplyParseError):
            replies.parse_reply('{"a": 1}', ("a", "b"))

    def test_one_retry_with_the_nudge_then_give_up(self):
        sent = []

        def call(m):
            sent.append(m)
            return "no json"

        with self.assertRaises(replies.ReplyParseError):
            replies.ask_json(call, [{"role": "user", "content": "q"}], ("a",))
        self.assertEqual(
            sent[1][-2:],
            [{"role": "assistant", "content": "no json"}, {"role": "user", "content": replies.JSON_NUDGE}],
        )

    def test_a_premise_citing_a_passage_not_shown_is_dropped_with_a_warning(self):
        got, w = replies.resolve_premises([{"claim": "X.", "key": "C99"}], P, limit=4)
        self.assertEqual(got, [])
        self.assertIn("C99", w[0])

    def test_keys_are_normalised_and_carry_provenance_and_extras_are_dropped(self):
        items = [
            {"claim": "A.", "key": "[c2]"},
            {"claim": "A.", "key": "C2"},
            {"claim": "", "key": "C1"},
            {"claim": "B.", "key": "C1"},
        ]
        got, w = replies.resolve_premises(items, P, limit=1)
        self.assertEqual([(p.key, p.document, p.chunk_index) for p in got], [("C2", "b.pdf", 9)])
        self.assertEqual(len(w), 3)

    def test_validation_error_retries_with_schema_hint_and_succeeds(self):
        sent = []

        def call(msgs):
            sent.append(msgs)
            if len(sent) == 1:
                return (
                    '{"title": "x", "summary": "s", "mapping": "bad", '
                    '"identifiability": {}, "steps": [], "premises": [], "prediction": "p"}'
                )
            return (
                '{"title": "Valid", "summary": "S", "mapping": [{"source_role": "a", "target_role": "b"}], '
                '"identifiability": {"verdict": "determined"}, "steps": [{"action": "s"}], '
                '"premises": [], "prediction": "decreases by 2x"}'
            )

        out = replies.ask_json(
            call,
            [{"role": "user", "content": "draft"}],
            ("title", "summary", "mapping", "identifiability", "steps", "premises", "prediction"),
            validator=replies.validate_transfer_reply,
        )
        self.assertEqual(out["title"], "Valid")
        self.assertEqual(len(sent), 2)
        self.assertIn("Fix schema error(s):", sent[1][-1]["content"])

    def test_validate_critique_revise_and_verify_rebuttal(self):
        self.assertEqual(
            replies.validate_critique_reply(
                {"shape_match": "yes", "rows": [{"row": 0, "status": "holds"}], "objections": []}
            ),
            [],
        )
        self.assertTrue(
            replies.validate_critique_reply({"shape_match": "maybe", "rows": [], "objections": []})
        )
        self.assertEqual(
            replies.validate_revise_reply(
                {"responses": [{"objection": 0, "response": "Fixed", "concede": False}]}
            ),
            [],
        )
        self.assertTrue(replies.validate_revise_reply({"responses": ["bad"]}))
        self.assertEqual(
            replies.validate_verify_rebuttal_reply({"verdicts": [{"objection": 0, "resolved": True}]}),
            [],
        )
        self.assertTrue(
            replies.validate_verify_rebuttal_reply({"verdicts": "not-a-list"})
        )
