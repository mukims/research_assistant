"""
Agent 7 — Interactive Research Assistant & Brainstorming Studio (agent7_research_chat.py)

A multi-turn conversational RAG agent that lets researchers brainstorm,
explore hypotheses, identify literature gaps, and debate conflicting evidence
against their ingested paper database.

Features:
- Standalone follow-up condensation so pronoun references ("it", "that paper") resolve
- Dual retrieval: global search capped per paper + targeted focus search on recently cited papers
- Explicit token budgeting with rolling memory folding (no silent truncation)
- Keyed [S#] citations with validation and resolution to paper titles
- Brainstorming lenses (explore, gaps, contradictions, hypotheses, methodology)
- Automated follow-up question suggestions per turn
- Session export including memory, turns, sources, and pinned scratchpad notes

Commands inside the CLI chat:
    /clear     — Reset conversation history
    /sources   — Show the sources used in the last response
    /memory    — Show the rolling conversation memory
    /mode      — View or change brainstorming lens (explore, gaps, contradictions, hypotheses, methodology)
    /export    — Save the full conversation to a timestamped markdown file
    /help      — Show available commands
    quit/exit  — Exit the session
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

from research_assistant.config import (
    CHAT_ANSWER_RESERVE_TOKENS,
    CHAT_CONDENSE,
    CHAT_CONTEXT_MAX_CHARS,
    CHAT_FOCUS_TOP_K,
    CHAT_MODEL,
    CHAT_OLLAMA_OPTIONS,
    CHAT_PER_DOC_CAP,
    CHAT_TEMPERATURE,
    DRAFTS_DIR,
)
from research_assistant.prompts import (
    BRAIN_LENS_INSTRUCTIONS,
    CHAT_CONDENSE_USER,
    CHAT_MEMORY_USER,
    CHAT_TURN_NO_CONTEXT,
    CHAT_TURN_USER,
    RESEARCH_CHAT_SYSTEM,
)
from research_assistant.shared.chat_context import (
    cap_per_document,
    cited_keys,
    estimate_tokens,
    fit_history,
    format_context,
    merge_results,
    messages_tokens,
    parse_brainstorm_suggestions,
    resolve_keys,
)
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.llm import chat, chat_stream
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import hybrid_search

logger = get_logger("agent7")

SYSTEM_PROMPT = RESEARCH_CHAT_SYSTEM

HELP_TEXT = """
╔══════════════════════════════════════════════════════════╗
║               Available Commands                         ║
╠══════════════════════════════════════════════════════════╣
║  /clear    — Reset conversation history                  ║
║  /sources  — Show sources from last response             ║
║  /memory   — Show rolling conversation memory            ║
║  /mode     — View or switch brainstorming lens           ║
║  /export   — Save conversation & sources to markdown     ║
║  /help     — Show this help message                      ║
║  quit/exit — End the session                             ║
╚══════════════════════════════════════════════════════════╝
"""


class ResearchChat:
    """Multi-turn conversational RAG agent backed by the paper database.

    Each turn: condense follow-up query → search global + focus papers → fit context
    and history within token budget (folding older turns into rolling memory) → stream
    grounded answer with [S#] keys → validate citations and extract suggestions.
    """

    def __init__(self, top_k: int = 5, search_resources: tuple | None = None, mode: str = "explore"):
        self.top_k = top_k
        self.mode = mode if mode in BRAIN_LENS_INSTRUCTIONS else "explore"
        self.history: list[dict[str, str]] = []       # list of {"role": ..., "content": ...}
        self.turns: list[dict[str, Any]] = []          # structured per-turn history with sources
        self.memory = ""                               # rolling summary of turns that no longer fit
        self.focus_documents: set[str] = set()
        self.last_sources: list[dict[str, Any]] = []  # this turn's keyed sources, with "cited"
        self.last_query = ""                          # query actually used for retrieval
        self.last_warnings: list[str] = []
        self.last_budget: dict[str, Any] = {}
        self.last_suggestions: list[str] = []         # suggested next research questions
        self.turn_count = 0

        if search_resources:
            self.collection, self.bm25, self.texts, self.metadatas = search_resources
        else:
            logger.info("Loading search resources…")
            self.collection, self.bm25, self.texts, self.metadatas = load_search_resources()
        logger.info("Ready. %d chunks in database.", len(self.texts))

    # ── Retrieval ────────────────────────────────────────────────────────

    @staticmethod
    def _transcript(messages: list[dict], limit: int) -> str:
        return "\n".join(
            f"{'Researcher' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:limit]}"
            for m in messages
        )

    def _condense(self, question: str) -> str:
        """Rewrite follow-up into a standalone query using recent history."""
        if not CHAT_CONDENSE or not self.history:
            return question
        try:
            recent = self._transcript(self.history[-4:], 600)
            prompt = CHAT_CONDENSE_USER.format(history=recent, question=question)
            out = chat(
                [{"role": "user", "content": prompt}],
                model=CHAT_MODEL,
                temperature=0.0,
                options=CHAT_OLLAMA_OPTIONS,
            ).content.strip().strip('"')
        except Exception as exc:  # noqa: BLE001
            logger.warning("Condense call failed (%s) — searching question as typed.", exc)
            return question

        words = len(out.split())
        if not (3 <= words <= 60):
            return question
        return out

    def _search(self, query: str) -> list[dict]:
        """Perform diverse global search plus targeted focus search on cited papers."""
        common = dict(top_k=self.top_k * 2, exclude_types={"figure_description"})
        global_hits = hybrid_search(query, self.collection, self.bm25, self.texts, self.metadatas, **common)
        global_hits = cap_per_document(global_hits, CHAT_PER_DOC_CAP)[: self.top_k]

        focus_hits = []
        if self.focus_documents and CHAT_FOCUS_TOP_K > 0:
            focus_hits = hybrid_search(
                query,
                self.collection,
                self.bm25,
                self.texts,
                self.metadatas,
                top_k=CHAT_FOCUS_TOP_K,
                doc_filter=set(self.focus_documents),
                exclude_types={"figure_description"},
            )
        return merge_results(focus_hits, global_hits)

    # ── Window budget ────────────────────────────────────────────────────

    def _system_prompt(self, mode: str | None = None) -> str:
        current_mode = mode or self.mode
        lens_instruction = BRAIN_LENS_INSTRUCTIONS.get(current_mode, "")
        prompt = RESEARCH_CHAT_SYSTEM
        if lens_instruction:
            prompt = f"{prompt}\n\n[Active Brainstorming Lens]\n{lens_instruction}"
        if self.memory:
            prompt = f"{prompt}\n\nConversation so far: {self.memory}"
        return prompt

    def _fold_memory(self, dropped: list[dict]) -> None:
        """Summarize older turns that leave the window into rolling memory."""
        try:
            turns_text = self._transcript(dropped, 800)
            out = chat(
                [{"role": "user", "content": CHAT_MEMORY_USER.format(
                    memory=self.memory or "(none)", turns=turns_text)}],
                model=CHAT_MODEL,
                temperature=0.0,
                options=CHAT_OLLAMA_OPTIONS,
            ).content.strip()
            if out:
                self.memory = out
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Memory fold failed (%s) — %d earlier message(s) leave window unsummarised.",
                exc, len(dropped),
            )

    def _fit(self, user_content: str, mode: str | None = None) -> tuple[list[dict], int]:
        """Fit history within window, folding older turns into memory if needed."""
        num_ctx = CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096)
        folded = 0
        for _ in range(2):
            fixed = (
                estimate_tokens(self._system_prompt(mode))
                + estimate_tokens(user_content)
                + CHAT_ANSWER_RESERVE_TOKENS
            )
            kept, dropped = fit_history(self.history, max(0, num_ctx - fixed))
            if not dropped:
                break
            logger.info("Folding %d earlier message(s) into memory to fit num_ctx=%d.", len(dropped), num_ctx)
            self._fold_memory(dropped)
            self.history = kept
            folded += 1
        return kept, folded

    # ── Public API ───────────────────────────────────────────────────────

    def stream_turn(self, user_message: str, mode: str | None = None):
        """Process one user turn: condense, search, fit window, stream response, and update history."""
        self.turn_count += 1
        active_mode = mode or self.mode
        self.mode = active_mode

        query = self._condense(user_message)
        self.last_query = query

        results = self._search(query)
        context, sources = format_context(results, CHAT_CONTEXT_MAX_CHARS)
        if context:
            user_content = CHAT_TURN_USER.format(question=user_message, context=context)
        else:
            user_content = CHAT_TURN_NO_CONTEXT.format(question=user_message)

        kept, folded = self._fit(user_content, mode=active_mode)
        messages = [{"role": "system", "content": self._system_prompt(active_mode)}] + kept + [
            {"role": "user", "content": user_content}
        ]
        num_ctx = CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096)
        self.last_budget = {
            "num_ctx": num_ctx,
            "prompt_tokens_est": messages_tokens(messages),
            "reserve": CHAT_ANSWER_RESERVE_TOKENS,
            "context_chars": len(context),
            "history_messages": len(kept),
            "folded": folded,
        }

        full_answer = ""
        for content in chat_stream(
            messages,
            model=CHAT_MODEL,
            options=CHAT_OLLAMA_OPTIONS,
            temperature=CHAT_TEMPERATURE,
        ):
            full_answer += content
            yield content

        used = cited_keys(full_answer)
        valid = {s["key"] for s in sources}
        for s in sources:
            s["cited"] = s["key"] in used
        self.last_sources = sources
        self.last_warnings = [
            f"cited [{k}], which is not among this turn's sources"
            for k in used
            if k not in valid
        ]
        cited_docs = {s["document"] for s in sources if s["cited"] and s["document"]}
        self.focus_documents = cited_docs or {s["document"] for s in sources[:2] if s["document"]}

        # Parse follow-up brainstorming suggestions
        self.last_suggestions = parse_brainstorm_suggestions(full_answer)

        # Store resolved text in history so future turns reference real paper titles
        resolved_answer = resolve_keys(full_answer, sources)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": resolved_answer})

        # Store rich structured turn record for persistent UI rendering & export
        self.turns.append({
            "turn_index": self.turn_count,
            "user_message": user_message,
            "condensed_query": query,
            "raw_answer": full_answer,
            "resolved_answer": resolved_answer,
            "content": full_answer,
            "sources": sources,
            "warnings": list(self.last_warnings),
            "suggestions": list(self.last_suggestions),
            "mode": active_mode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def clear_history(self):
        """Reset the conversation, memory, and turn records."""
        self.history.clear()
        self.turns.clear()
        self.memory = ""
        self.focus_documents = set()
        self.last_sources = []
        self.last_query = ""
        self.last_warnings = []
        self.last_budget = {}
        self.last_suggestions = []
        self.turn_count = 0

    def export_conversation(self, scratchpad_notes: list[str] | None = None) -> str:
        """Save the conversation, sources, memory, and pinned notes to markdown."""
        os.makedirs(DRAFTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(DRAFTS_DIR, f"research_chat_{timestamp}.md")

        lines = [
            f"# 🔬 Research Chat & Brainstorming Session",
            f"*Date:* {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"*Active Lens:* {self.mode.capitalize()}",
            "",
        ]

        if self.memory:
            lines.extend([
                "## 🧠 Rolling Memory",
                self.memory,
                "",
            ])

        if scratchpad_notes:
            lines.extend([
                "## 📝 Pinned Scratchpad Notes & Key Takeaways",
                *[f"- {note}" for note in scratchpad_notes],
                "",
            ])

        lines.append("## 💬 Conversation Transcript\n")

        # Use structured turns if available, else fallback to raw history
        if self.turns:
            for turn in self.turns:
                lines.append(f"### Turn {turn['turn_index']}: 🧑‍🔬 Researcher")
                lines.append(f"{turn['user_message']}\n")
                if turn.get("condensed_query") and turn["condensed_query"] != turn["user_message"]:
                    lines.append(f"*🔍 Literature Query:* `{turn['condensed_query']}`\n")
                lines.append(f"### Turn {turn['turn_index']}: 🤖 Assistant ({turn.get('mode', 'explore')})")
                lines.append(f"{turn['raw_answer']}\n")

                if turn.get("sources"):
                    lines.append("<details><summary>📚 Sources Used</summary>\n")
                    for s in turn["sources"]:
                        status = "✅ Cited" if s.get("cited") else "⚪ Referenced"
                        lines.append(f"- **[{s['key']}]** `{s.get('document', 'Unknown')}` — {s.get('citation', '')} (p.{s.get('page', '?')}) [{status}]")
                    lines.append("\n</details>\n")

                if turn.get("suggestions"):
                    lines.append("**💡 Suggested Follow-ups:**")
                    for sug in turn["suggestions"]:
                        lines.append(f"- {sug}")
                    lines.append("")
                lines.append("---\n")
        else:
            for msg in self.history:
                role = "🧑‍🔬 **Researcher**" if msg["role"] == "user" else "🤖 **Assistant**"
                lines.append(f"\n{role}\n\n{msg['content']}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath

    def export_conversation_json(self, scratchpad_notes: list[str] | None = None) -> dict[str, Any]:
        """Return structured session JSON for programmatic consumption."""
        return {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "memory": self.memory,
            "scratchpad_notes": scratchpad_notes or [],
            "turns": self.turns,
            "history": self.history,
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def run_repl(agent: ResearchChat, banner: str | None = None):
    """Run interactive REPL against ResearchChat."""
    if banner:
        print(banner)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/clear":
            agent.clear_history()
            print("✓ Conversation history and memory cleared.\n")
            continue

        if user_input == "/memory":
            if not agent.memory:
                print("No memory formed yet (all recent turns currently fit in prompt window).\n")
            else:
                print("\n🧠 Rolling Conversation Memory:")
                print(agent.memory)
                print()
            continue

        if user_input.startswith("/mode"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip().lower() in BRAIN_LENS_INSTRUCTIONS:
                agent.mode = parts[1].strip().lower()
                print(f"✓ Brainstorming lens set to: {agent.mode}\n")
            else:
                print(f"\nCurrent mode: {agent.mode}")
                print("Available lenses:", ", ".join(BRAIN_LENS_INSTRUCTIONS.keys()))
                print("Use: /mode <lens>\n")
            continue

        if user_input == "/sources":
            if not agent.last_sources:
                print("No sources from the last response.\n")
            else:
                print("\n📚 Sources used in the last response:")
                for s in agent.last_sources:
                    mark = "✓ cited" if s.get("cited") else "  ref"
                    print(f"  [{s['key']}] ({mark}) [{s.get('document', '?')}] {s.get('citation', '')} (p.{s.get('page', '?')})")
                if agent.last_warnings:
                    print("\n⚠️ Citation Warnings:")
                    for w in agent.last_warnings:
                        print(f"  - {w}")
                print()
            continue

        if user_input == "/export":
            path = agent.export_conversation()
            print(f"✓ Conversation exported to {path}\n")
            continue

        try:
            if agent.history and CHAT_CONDENSE:
                condensed = agent._condense(user_input)
                if condensed != user_input:
                    print(f"🔍 Searching: {condensed}")

            print("\nAssistant: ", end="", flush=True)
            for chunk in agent.stream_turn(user_input):
                print(chunk, end="", flush=True)
            print("\n")

            if agent.last_sources:
                cited = [s["citation"][:45] for s in agent.last_sources if s.get("cited")]
                if cited:
                    print(f"  📚 Cited: {', '.join(cited)}")
                print("  (type /sources for full details)\n")

            if agent.last_suggestions:
                print("  💡 Suggested follow-ups:")
                for sug in agent.last_suggestions[:3]:
                    print(f"    • {sug}")
                print()

        except Exception as e:
            logger.error("Error during chat: %s", e)
            print(f"\n⚠ Error: {e}. Please try again.\n")


def main():
    parser = argparse.ArgumentParser(description="Agent 7 — Interactive Research Assistant")
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of context chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--mode", type=str, default="explore",
        choices=list(BRAIN_LENS_INSTRUCTIONS.keys()),
        help="Brainstorming lens (explore, gaps, contradictions, hypotheses, methodology)",
    )
    args = parser.parse_args()

    agent = ResearchChat(top_k=args.top_k, mode=args.mode)

    run_repl(agent, banner=(
        "\n" + "=" * 60 + "\n"
        "  🔬  Research Assistant — Interactive Brainstorming Studio\n"
        f"  Active Lens: {args.mode.capitalize()} | Powered by ingested paper database\n"
        "  Type /help for commands, /mode to switch lens, 'quit' to exit\n"
        + "=" * 60 + "\n"
    ))


if __name__ == "__main__":
    main()
