# tests/test_agent7_chat.py
"""Comprehensive tests for Agent 7 ResearchChat.

Validates multi-turn state preservation, query condensation, focus paper search,
token budget fitting, memory folding, structured turn tracking, brainstorming lenses,
and conversation export with scratchpad notes.
"""

import os
import tempfile
import types
import unittest
from unittest.mock import patch

from research_assistant.agents import agent7_research_chat as a7


def _hit(i, doc, text, page=1):
    return {
        "chunk_index": i,
        "text": text,
        "metadata": {
            "document": doc,
            "citation_source": f"Title {doc}",
            "page": page,
        },
        "rrf_score": 0.5,
    }


class ChatTestCase(unittest.TestCase):
    def setUp(self):
        self.searches = []

        def fake_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=None, exclude_types=None, **kw):
            self.searches.append({
                "query": query,
                "top_k": top_k,
                "doc_filter": doc_filter,
                "exclude_types": exclude_types,
            })
            if doc_filter:
                return [_hit(90 + i, next(iter(doc_filter)), f"focus chunk {i}") for i in range(top_k)]
            return [
                _hit(1, "a.pdf", "A one"),
                _hit(2, "a.pdf", "A two"),
                _hit(3, "a.pdf", "A three"),
                _hit(4, "a.pdf", "A four"),
                _hit(5, "b.pdf", "B one"),
                _hit(6, "c.pdf", "C one"),
            ][:top_k]

        p = patch.object(a7, "hybrid_search", side_effect=fake_search)
        p.start()
        self.addCleanup(p.stop)

        self.chats = []

        def fake_chat(messages, model=None, temperature=None, options=None, **kw):
            content = messages[-1]["content"]
            self.chats.append({"content": content, "temperature": temperature, "options": options})
            if "Rewrite the new question" in content:
                return types.SimpleNamespace(content="limitations of hopping study")
            return types.SimpleNamespace(content="MEMORY: earlier turns summarised")

        p = patch.object(a7, "chat", side_effect=fake_chat)
        p.start()
        self.addCleanup(p.stop)

        self.streams = []
        self.answer = (
            "Hopping dominates [S1]. See also [S2]. Unknown [S9].\n\n"
            "### 💡 Suggested Next Questions:\n"
            "- What are the high temperature limits?\n"
            "- How does gating modulate mobility?\n"
        )

        def fake_stream(messages, model=None, options=None, temperature=None):
            self.streams.append({"messages": messages, "temperature": temperature, "options": options})
            yield from [self.answer[:20], self.answer[20:]]

        p = patch.object(a7, "chat_stream", side_effect=fake_stream)
        p.start()
        self.addCleanup(p.stop)

        self.agent = a7.ResearchChat(top_k=5, search_resources=(None, None, [], []))

    def _turn(self, q, mode=None):
        return "".join(self.agent.stream_turn(q, mode=mode))


class TestFirstTurn(ChatTestCase):
    def test_no_condense_on_first_turn(self):
        self._turn("What does the MoS2 paper say?")
        self.assertEqual([c for c in self.chats if "Rewrite" in c["content"]], [])
        self.assertEqual(self.searches[0]["top_k"], 10)
        self.assertEqual(self.searches[0]["exclude_types"], {"figure_description"})
        docs = [s["document"] for s in self.agent.last_sources]
        self.assertEqual(docs, ["a.pdf", "a.pdf", "a.pdf", "b.pdf", "c.pdf"])

    def test_keys_checked_and_focus_set(self):
        self._turn("q")
        self.assertEqual([s["cited"] for s in self.agent.last_sources], [True, True, False, False, False])
        self.assertEqual(self.agent.last_warnings, ["cited [S9], which is not among this turn's sources"])
        self.assertEqual(self.agent.focus_documents, {"a.pdf"})

    def test_suggestions_parsed(self):
        self._turn("q")
        self.assertEqual(len(self.agent.last_suggestions), 2)
        self.assertIn("What are the high temperature limits?", self.agent.last_suggestions[0])
        self.assertIn("How does gating modulate mobility?", self.agent.last_suggestions[1])

    def test_structured_turns_recorded(self):
        self._turn("What is hopping?")
        self.assertEqual(len(self.agent.turns), 1)
        turn = self.agent.turns[0]
        self.assertEqual(turn["turn_index"], 1)
        self.assertEqual(turn["user_message"], "What is hopping?")
        self.assertEqual(len(turn["sources"]), 5)
        self.assertEqual(len(turn["suggestions"]), 2)
        self.assertEqual(turn["warnings"], ["cited [S9], which is not among this turn's sources"])


class TestFollowUp(ChatTestCase):
    def test_condensed_query_is_searched_and_focus_runs(self):
        self._turn("What does the MoS2 paper say?")
        self.searches.clear()
        self._turn("What about its limitations?")
        condense = [c for c in self.chats if "Rewrite the new question" in c["content"]]
        self.assertEqual(len(condense), 1)
        self.assertEqual(self.agent.last_query, "limitations of hopping study")
        self.assertTrue(all(s["query"] == self.agent.last_query for s in self.searches))
        focus = [s for s in self.searches if s["doc_filter"]]
        self.assertEqual(focus[0]["doc_filter"], {"a.pdf"})

    def test_turns_retain_independent_sources(self):
        self._turn("Turn 1 question")
        sources_turn1 = list(self.agent.turns[0]["sources"])
        self._turn("Turn 2 question")
        sources_turn2 = list(self.agent.turns[1]["sources"])
        self.assertEqual(len(self.agent.turns), 2)
        self.assertEqual(self.agent.turns[0]["user_message"], "Turn 1 question")
        self.assertEqual(self.agent.turns[1]["user_message"], "Turn 2 question")
        self.assertTrue(len(sources_turn1) > 0)
        self.assertTrue(len(sources_turn2) > 0)


class TestBrainstormingModes(ChatTestCase):
    def test_mode_selection_adapts_system_prompt(self):
        self._turn("Find gaps", mode="gaps")
        sys_msg = self.streams[-1]["messages"][0]["content"]
        self.assertIn("Literature Gaps & Open Challenges", sys_msg)
        self.assertEqual(self.agent.mode, "gaps")
        self.assertEqual(self.agent.turns[-1]["mode"], "gaps")

        self._turn("Hypotheses", mode="hypotheses")
        sys_msg2 = self.streams[-1]["messages"][0]["content"]
        self.assertIn("Novel Hypothesis Formulation", sys_msg2)
        self.assertEqual(self.agent.mode, "hypotheses")


class TestBudgetAndMemory(ChatTestCase):
    def test_older_turns_fold_into_memory(self):
        self.answer = "answer " * 120 + "[S1]."
        with patch.object(a7, "CHAT_OLLAMA_OPTIONS", {"num_ctx": 1400}), patch.object(a7, "CHAT_CONTEXT_MAX_CHARS", 600):
            for i in range(5):
                self._turn(f"question number {i}")
        self.assertIn("MEMORY", self.agent.memory)
        self.assertTrue(any("Summarise these earlier turns" in c["content"] for c in self.chats))
        last = self.streams[-1]
        self.assertIn("Conversation so far", last["messages"][0]["content"])
        self.assertLessEqual(self.agent.last_budget["prompt_tokens_est"], 1400 - a7.CHAT_ANSWER_RESERVE_TOKENS)


class TestClearAndExport(ChatTestCase):
    def test_clear_resets_everything(self):
        self._turn("q")
        self.agent.memory = "m"
        self.agent.clear_history()
        self.assertEqual(
            (self.agent.history, self.agent.turns, self.agent.memory, self.agent.focus_documents, self.agent.last_sources),
            ([], [], "", set(), []),
        )

    def test_export_includes_turns_sources_and_notes(self):
        self._turn("q")
        self.agent.memory = "established ideas"
        notes = ["Important idea: hopping exponent is 0.5", "Hypothesis: test with chemical gate"]
        with tempfile.TemporaryDirectory() as d, patch.object(a7, "DRAFTS_DIR", d):
            path = self.agent.export_conversation(scratchpad_notes=notes)
            text = open(path, encoding="utf-8").read()
            self.assertIn("established ideas", text)
            self.assertIn("Pinned Scratchpad Notes", text)
            self.assertIn("hopping exponent is 0.5", text)
            self.assertIn("Turn 1: 🧑‍🔬 Researcher", text)
            self.assertIn("Sources Used", text)

        json_export = self.agent.export_conversation_json(scratchpad_notes=notes)
        self.assertEqual(json_export["scratchpad_notes"], notes)
        self.assertEqual(len(json_export["turns"]), 1)


if __name__ == "__main__":
    unittest.main()
