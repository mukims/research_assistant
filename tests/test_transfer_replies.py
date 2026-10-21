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
