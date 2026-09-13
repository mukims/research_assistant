# Brainstorm Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the research-chat studio into a brainstorming workspace: sessions on disk; a **board** (idea, established, objections, open questions, experiments, with stable paper keys) that rides in every prompt and is maintained by a structured call enforced in code; **moves** (sharpen, what's been done, argue against, test it, compare, dig in) beside the lenses; a `Position:` line in every answer; **Go and read** (the map-reduce synthesis folded into the conversation and the board); *Brainstorm this* from a Tab 1 synthesis; a window budget that knows the backend.

**Architecture:** A delta on `6b9ca58`'s `ResearchChat` — nothing it does is removed. `research_assistant/shared/brainstorm.py` (new) owns the board and the session, pure and tested. `ResearchChat` gains `attach(session)`, a board block in its system prompt, `move=`/`papers=` on `stream_turn`, an `S# → P#` mapping after each answer, `_update_board` (one call, `parse_delta` + `apply_delta`, retried once), `pin` / `edit_section` / `set_idea`, `dive()` and `seed_from_synthesis()` over `retrieve.research_answer(mode="map_reduce")`. Tab 4 becomes a session bar and two columns; Tab 1's synthesis gets one button.

**Tech Stack:** Python 3.12, Streamlit 1.53.1, the existing Ollama / OpenAI-compatible (Gemini) backends. No new dependencies.

**Spec:** `notimportant/superpowers/specs/2026-09-13-brainstorm-workspace-design.md` — read it first.

**Base commit:** `6b9ca58` (branch `synthesis-depth`). Every diff below is against that tree.

**Validated:** every block was assembled into a scratch copy at the base commit; the suite there passes (620 passed, 1 skipped, `tests/test_ingestion.py` deselected because the ingest lock was held); and the page was exercised live on a copy of the v2 index with the Gemini backend — a free turn filled the board (idea + three established items keyed `[P1]`), *Argue against* added an objection and hit the open-questions cap with a warning instead of overflowing, *What's been done* keyed five summaries and registered five papers with stable keys, a full server restart resumed the session with its turns and board, and *Go and read* ran in 9 s, rewrote the synthesis's local `P1/P2/P3` to the session's `P2/P4/P5` and added two established items. Two page bugs found live are already fixed in these blocks (a keyed selectbox that would not follow a programmatic session switch; a stale prefill in the dive question).

## Global Constraints

- **Nothing the studio does is removed**: condense, focus search, memory fold, `[S#]` keys, lenses, suggestion chips, starters, evidence cards, exports, the CLI REPL and `tests/test_agent7_chat.py` all stay. One line of that test file changes (Task 3) because the window is now `CHAT_WINDOW_TOKENS`.
- **The board is enforced in code.** Every delta goes through `bs.apply_delta`; nothing the model says lands on the board unchecked. Keep `parse_delta` and `apply_delta` the only way in.
- **Board items cite `P#` (session keys); a turn's sources are `[S#]`.** The mapping happens once per turn in `_register_cited`; the update call sees the `P#`-keyed answer; history keeps the studio's title-resolved text.
- **Keyed Streamlit widgets keep their own state**: to change a keyed selectbox programmatically, set its key in `st.session_state` before the rerun (`_switch_to` in Task 4); never prefill a keyed text input with `value=` that you expect to change later.
- **Patch real modules in tests**, not `sys.modules` — `from research_assistant.shared import retrieve` resolves the package attribute and ignores a `sys.modules` stand-in once the real module was imported earlier in the run (found the hard way; Task 3's tests do it right).
- **Test conventions:** `unittest.TestCase`; `CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider`; if `data/ingest.lock` is held, `--deselect tests/test_ingestion.py`. Interpreter on this machine: `/home/shardul/miniconda3/envs/ml/bin/python`; run scripts with `PYTHONPATH=.`.
- **Commit per task** with the message given; end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File map

| file | task | responsibility |
|---|---|---|
| `research_assistant/config.py` | 1 | `CHAT_WINDOW_TOKENS`, `CHAT_BOARD_UPDATE`, `CHAT_MOVE_PER_PAPER_CHUNKS`, `BRAINSTORMS_DIR` |
| `research_assistant/shared/brainstorm.py` (new) | 1 | board (render, deltas, caps, ids, edit round trip, markdown), papers (`register_paper`, `rewrite_keys`, `resolve_paper_keys`), `Session`, `list_sessions`, `delete_session` |
| `research_assistant/prompts.py`, `research_assistant/shared/chat_context.py` | 2 | the `Position:` rule, `CHAT_BOARD_BLOCK`, `CHAT_MOVE_USER`, `CHAT_BOARD_UPDATE_USER`; `parse_position` |
| `research_assistant/agents/agent7_research_chat.py` | 3 | sessions, the board in the prompt, the window, moves, `S# → P#`, the board update, pin/edit/idea, `dive`, `seed_from_synthesis`, export with the board |
| `app.py` | 4 | Tab 4: session bar, two columns, moves, the board panel, the dive; Tab 1: *Brainstorm this* |
| `HOW_TO_USE.md`, `PIPELINE.md`, `ARCHITECTURE.md` | 5 | docs |
| — | 6 | the live check |

---

### Task 1: `shared/brainstorm.py` — the board and the session

**Files:**
- Modify: `research_assistant/config.py`
- Create: `research_assistant/shared/brainstorm.py`
- Test: `tests/test_brainstorm.py`

**Interfaces:**
- Consumes: `shared.atomic.atomic_write_json`, `shared.chat_context.short_title`, `config.BRAINSTORMS_DIR`.
- Produces: constants `SECTIONS`, `CAPS`, `ITEM_MAX_CHARS`, `PREFIX`, `TITLES`, `LABELS`, `PAPER_KEY_RE`; `empty_board()`, `clip(text, limit)`, `paper_keys_in(text)`, `is_empty(board)`, `board_text(board, papers)` (the prompt form), `board_markdown(board, papers)` (the export/page form), `add_item(board, section, text, papers, refs=None) -> warnings`, `parse_delta(text) -> dict|None`, `apply_delta(board, delta, papers) -> warnings`, `section_edit_text(board, section)`, `section_from_edit_text(board, section, text, papers) -> warnings`, `register_paper(papers, document, citation) -> "P#"`, `rewrite_keys(text, mapping)` (one pass; `[S1, S2]` and `[P1]` groups), `resolve_paper_keys(text, papers)`, `Session` (`new(title, origin)`, `load(id)`, `save()`, `rename()`, `id/title/board/papers/turns/dir()/board_text()/board_markdown()`), `list_sessions(limit=50)`, `delete_session(id)`, `set_brainstorms_dir(path|None)`. Board item ids are `E/O/Q/X` + a counter (`board["next_id"]`) that is never reused.

- [ ] **Step 1: Config**

**Apply this change to `research_assistant/config.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/config.py
+++ b/research_assistant/config.py
@@ -104,6 +104,16 @@
 CHAT_PER_DOC_CAP            = _env_int("CITATION_CHAT_PER_DOC_CAP", 3)
 CHAT_FOCUS_TOP_K            = _env_int("CITATION_CHAT_FOCUS_TOP_K", 2)
 CHAT_TEMPERATURE            = float(os.environ.get("CITATION_CHAT_TEMPERATURE", "0.3"))
+# The window the chat budgets its prompt against (spec 2026-09-13-brainstorm
+# §2I). num_ctx is an Ollama runtime option; a hosted OpenAI-compatible model
+# has its own, much larger window, and folding history into memory at 4k
+# there throws away turns for nothing.
+CHAT_WINDOW_TOKENS          = _env_int("CITATION_CHAT_WINDOW_TOKENS",
+                                       32768 if LLM_BACKEND == "openai" else CHAT_OLLAMA_OPTIONS["num_ctx"])
+# The brainstorm board (spec §2B–C): one structured call after each turn
+# proposes a delta; code enforces it. 0 → the board is only what you pin and edit.
+CHAT_BOARD_UPDATE           = _env_bool("CITATION_CHAT_BOARD_UPDATE", True)
+CHAT_MOVE_PER_PAPER_CHUNKS  = _env_int("CITATION_CHAT_MOVE_PER_PAPER_CHUNKS", 4)
 
 
 # ─── Index version ───────────────────────────────────────────────────────────
@@ -144,6 +154,9 @@
 INGESTED_MANIFEST_PATH   = os.path.join(DATA_DIR, f"ingested{_INDEX_SUFFIX}.json")
 PIPELINE_STATUS_PATH     = os.path.join(DATA_DIR, "pipeline_status.json")
 INGEST_LOCK_PATH         = os.path.join(DATA_DIR, "ingest.lock")
+# Brainstorm sessions (spec 2026-09-13-brainstorm §2A): one directory per
+# session holding session.json — the board, the papers seen, the turns.
+BRAINSTORMS_DIR          = os.path.join(DATA_DIR, "brainstorms")
 
 # ─── Detectron2 / layout detection ──────────────────────────────────────────
 # Layout detection (figure/table crops + a VLM description of each) is the
```

- [ ] **Step 2: Write the failing tests**

**Create `tests/test_brainstorm.py` with exactly this content:**

````python
"""brainstorm — the board, deltas, editing, papers, sessions
(spec 2026-09-13-brainstorm-workspace §2A–C)."""

import json
import os
import tempfile
import unittest

from research_assistant.shared import brainstorm as bs

PAPERS = {"P1": {"document": "a.pdf", "citation": "Anderson 1958 Absence of diffusion in certain random lattices"},
          "P2": {"document": "b.pdf", "citation": "Mott 1969 Conduction in non-crystalline materials"}}


class BoardCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        bs.set_brainstorms_dir(self.tmp.name)
        self.addCleanup(bs.set_brainstorms_dir, None)
        self.board = bs.empty_board()
        self.papers = json.loads(json.dumps(PAPERS))


class TestRender(BoardCase):
    def test_empty_board_text_says_so(self):
        text = bs.board_text(self.board, {})
        self.assertIn("IDEA: (not stated yet)", text)
        self.assertIn("ESTABLISHED:\n- (none yet)", text)
        self.assertNotIn("PAPERS:", text)
        self.assertTrue(bs.is_empty(self.board))

    def test_items_render_with_ids_and_keys(self):
        bs.apply_delta(self.board, {"idea": "Disorder localises every state in 1D.",
                                     "add": {"established": [{"text": "Localisation length ~ mean free path", "papers": ["P1"]}],
                                             "open": ["What about interactions?"]}}, self.papers)
        text = bs.board_text(self.board, self.papers)
        self.assertIn("IDEA: Disorder localises every state in 1D.", text)
        self.assertIn("- (E1) Localisation length ~ mean free path [P1]", text)
        self.assertIn("- (Q2) What about interactions?", text)
        self.assertIn("PAPERS: [P1] Anderson 1958 Absence of diffusion in certain random; [P2]", text)
        self.assertFalse(bs.is_empty(self.board))
        md = bs.board_markdown(self.board, self.papers)
        self.assertIn("**Idea.** Disorder", md)
        self.assertIn("- Localisation length ~ mean free path [Anderson 1958 Absence of diffusion in]", md)
        self.assertIn("- **P1** — Anderson 1958", md)


class TestApplyDelta(BoardCase):
    def test_add_replace_remove_and_ids_never_reused(self):
        w = bs.apply_delta(self.board, {"add": {"established": ["one [P1]", "two"], "objections": [{"text": "obj", "papers": "P2"}]}}, self.papers)
        self.assertEqual(w, [])
        self.assertEqual([i["id"] for i in self.board["established"]], ["E1", "E2"])
        self.assertEqual(self.board["established"][0], {"id": "E1", "text": "one", "papers": ["P1"]})
        self.assertEqual(self.board["objections"][0], {"id": "O3", "text": "obj", "papers": ["P2"]})
        w = bs.apply_delta(self.board, {"replace": {"E1": {"text": "one, sharper", "papers": ["P1", "P2"]}},
                                         "remove": ["E2"]}, self.papers)
        self.assertEqual(w, [])
        self.assertEqual(self.board["established"], [{"id": "E1", "text": "one, sharper", "papers": ["P1", "P2"]}])
        bs.apply_delta(self.board, {"add": {"established": ["three"]}}, self.papers)
        self.assertEqual(self.board["established"][-1]["id"], "E4")     # E2 is not reused

    def test_every_rejection_is_a_warning_not_a_change(self):
        w = bs.apply_delta(self.board, {
            "idea": 42, "add": {"nope": ["x"], "established": [{"text": "", "papers": []}, 7, {"text": "ok", "papers": ["P9"]}]},
            "replace": {"E99": "x", "bad": "y"}, "remove": ["O1"], "extra": 1,
        }, self.papers)
        self.assertEqual(self.board["idea"], "")
        self.assertEqual([i["text"] for i in self.board["established"]], ["ok"])
        self.assertEqual(self.board["established"][0]["papers"], [])
        joined = "\n".join(w)
        for needle in ("idea was not text", "unknown section 'nope'", "empty item", "not text", "unknown paper key(s) P9",
                       "replace E99: no such item", "replace bad: no such item", "remove O1: no such item", "unknown key 'extra'"):
            self.assertIn(needle, joined)

    def test_caps_and_clipping(self):
        bs.apply_delta(self.board, {"add": {"open": [f"q{i}" for i in range(7)]}}, self.papers)
        self.assertEqual(len(self.board["open"]), 5)
        w = bs.apply_delta(self.board, {"idea": "x" * 700, "add": {"open": ["one more"], "established": ["y" * 300]}}, self.papers)
        self.assertEqual(len(self.board["idea"]), 600)
        self.assertTrue(self.board["idea"].endswith("…"))
        self.assertEqual(len(self.board["established"][0]["text"]), 240)
        self.assertTrue(any("at its cap of 5" in x for x in w))
        self.assertEqual(len(self.board["open"]), 5)

    def test_not_an_object(self):
        self.assertEqual(bs.apply_delta(self.board, ["x"], self.papers), ["board update was not an object — ignored"])
        w = bs.apply_delta(self.board, {"add": ["x"], "replace": ["y"], "remove": "E1"}, self.papers)
        self.assertEqual(len(w), 3)

    def test_add_item_pin_and_note(self):
        self.assertEqual(bs.add_item(self.board, "established", "pinned [P1]", self.papers, refs=["P1", "P7"]), [])
        self.assertEqual(self.board["established"][0]["papers"], ["P1"])
        self.assertEqual(bs.add_item(self.board, "nope", "x", self.papers), ["unknown section 'nope'"])
        self.assertEqual(bs.add_item(self.board, "open", "  ", self.papers), ["empty item — nothing added"])
        for i in range(5):
            bs.add_item(self.board, "open", f"q{i}", self.papers)
        self.assertEqual(bs.add_item(self.board, "open", "q6", self.papers), ["Open questions is at its cap of 5 — remove one first"])


class TestParseDelta(BoardCase):
    def test_bare_fenced_and_embedded(self):
        obj = {"idea": "x", "add": {"open": ["q"]}}
        self.assertEqual(bs.parse_delta(json.dumps(obj)), obj)
        self.assertEqual(bs.parse_delta("Here you go:\n```json\n" + json.dumps(obj) + "\n```\nDone."), obj)
        self.assertEqual(bs.parse_delta("Sure. " + json.dumps(obj) + " That's it."), obj)

    def test_broken_or_missing(self):
        self.assertIsNone(bs.parse_delta(""))
        self.assertIsNone(bs.parse_delta("no json here"))
        self.assertIsNone(bs.parse_delta("{not: valid}"))
        self.assertIsNone(bs.parse_delta("[1, 2]"))


class TestEdit(BoardCase):
    def test_round_trip_keeps_ids_and_assigns_new_ones(self):
        bs.apply_delta(self.board, {"add": {"established": ["one [P1]", "two"]}}, self.papers)
        text = bs.section_edit_text(self.board, "established")
        self.assertEqual(text, "one [P1]\ntwo")
        w = bs.section_from_edit_text(self.board, "established", "two\n- three [P2, P9]\n\none [P1]", self.papers)
        self.assertEqual([i["id"] for i in self.board["established"]], ["E2", "E3", "E1"])
        self.assertEqual(self.board["established"][1], {"id": "E3", "text": "three", "papers": ["P2"]})
        self.assertTrue(any("P9" in x for x in w))

    def test_edit_cap(self):
        w = bs.section_from_edit_text(self.board, "open", "\n".join(f"q{i}" for i in range(7)), self.papers)
        self.assertEqual(len(self.board["open"]), 5)
        self.assertTrue(any("more than 5" in x for x in w))
        self.assertNotIn("papers", self.board["open"][0])


class TestPapers(BoardCase):
    def test_register_is_stable_by_document(self):
        papers = {}
        self.assertEqual(bs.register_paper(papers, "a.pdf", "A title"), "P1")
        self.assertEqual(bs.register_paper(papers, "b.pdf", "B title"), "P2")
        self.assertEqual(bs.register_paper(papers, "a.pdf", "A title again"), "P1")
        self.assertEqual(papers["P1"]["citation"], "A title")
        self.assertEqual(bs.register_paper(papers, "c.pdf", ""), "P3")
        self.assertEqual(papers["P3"]["citation"], "c.pdf")

    def test_rewrite_keys_one_pass(self):
        self.assertEqual(bs.rewrite_keys("A [S1]. B [S2, S1]. C [S9].", {"S1": "P3", "S2": "P1"}), "A [P3]. B [P1, P3]. C [S9].")
        # a synthesis's local P1/P2 → the session's P2/P1: no double substitution
        self.assertEqual(bs.rewrite_keys("x [P1] y [P2] z [P1, P2]", {"P1": "P2", "P2": "P1"}), "x [P2] y [P1] z [P2, P1]")

    def test_resolve_paper_keys(self):
        self.assertEqual(bs.resolve_paper_keys("see [P1, P2] and [P7]", PAPERS),
                         "see [Anderson 1958 Absence of diffusion in; Mott 1969 Conduction in non-crystalline materials] and [P7]")

    def test_paper_keys_in(self):
        self.assertEqual(bs.paper_keys_in("a [P2] b [P1, P2]"), ["P2", "P1"])


class TestSession(BoardCase):
    def test_new_save_load_list_rename_delete(self):
        s = bs.Session.new("Disordered wires", origin={"query": "disordered wires"})
        self.assertTrue(os.path.exists(os.path.join(s.dir(), "session.json")))
        bs.apply_delta(s.board, {"idea": "x"}, s.papers)
        s.data["turns"].append({"user_message": "q", "resolved_answer": "a"})
        s.data["memory"] = "m"
        s.save()
        t = bs.Session.load(s.id)
        self.assertEqual(t.title, "Disordered wires")
        self.assertEqual(t.board["idea"], "x")
        self.assertEqual(t.data["memory"], "m")
        self.assertEqual(t.data["origin"], {"query": "disordered wires"})
        self.assertEqual(len(t.turns), 1)
        t.rename("Wires, disordered")
        listed = bs.list_sessions()
        self.assertEqual(listed[0]["title"], "Wires, disordered")
        self.assertEqual(listed[0]["turns"], 1)
        bs.delete_session(s.id)
        self.assertEqual(bs.list_sessions(), [])

    def test_unreadable_session_is_listed_as_such(self):
        os.makedirs(os.path.join(self.tmp.name, "junk"))
        open(os.path.join(self.tmp.name, "junk", "session.json"), "w").write("{oops")
        listed = bs.list_sessions()
        self.assertTrue(listed[0]["error"])
        with self.assertRaises(ValueError):
            bs.Session.load("junk")
        bs.set_brainstorms_dir(os.path.join(self.tmp.name, "nowhere"))
        self.assertEqual(bs.list_sessions(), [])

    def test_load_fills_defaults(self):
        os.makedirs(os.path.join(self.tmp.name, "old"))
        with open(os.path.join(self.tmp.name, "old", "session.json"), "w") as fh:
            json.dump({"id": "old", "title": "t", "board": {"idea": "", "established": [], "objections": [], "open": [], "experiments": []}}, fh)
        s = bs.Session.load("old")
        self.assertEqual(s.board["next_id"], 1)
        self.assertEqual(s.data["mode"], "explore")
        self.assertEqual(s.papers, {})


if __name__ == "__main__":
    unittest.main()
````

- [ ] **Step 3: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_brainstorm.py
```

Expected: FAIL — `ModuleNotFoundError: research_assistant.shared.brainstorm`

- [ ] **Step 4: Write the module**

**Create `research_assistant/shared/brainstorm.py` with exactly this content:**

````python
"""
brainstorm — the board and the session (spec 2026-09-13-brainstorm-workspace §2A–C).

The board is the model's working memory: five small sections that ride in
every turn's prompt, maintained by a structured call after each turn and
enforced here in code — a delta the model proposes is parsed, validated
against the schema, capped and applied; anything that does not fit is
dropped with a warning. The board never changes silently.

A session is a directory under config.BRAINSTORMS_DIR holding session.json:
the board, the registry of papers cited (stable keys P1…), the studio's
turns, memory, lens and focus. Pure: no model, no index, no Streamlit.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.chat_context import short_title

SECTIONS = ("established", "objections", "open", "experiments")
CAPS = {"idea": 600, "established": 8, "objections": 5, "open": 5, "experiments": 5}
ITEM_MAX_CHARS = 240
PREFIX = {"established": "E", "objections": "O", "open": "Q", "experiments": "X"}
SECTION_OF_PREFIX = {v: k for k, v in PREFIX.items()}
TITLES = {"established": "ESTABLISHED", "objections": "OBJECTIONS", "open": "OPEN QUESTIONS",
          "experiments": "EXPERIMENTS"}
LABELS = {"established": "Established", "objections": "Objections", "open": "Open questions",
          "experiments": "Experiments"}

BRAINSTORMS_DIR_OVERRIDE: Optional[str] = None

PAPER_KEY_RE = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")
_ITEM_ID_RE = re.compile(r"^([EOQX])(\d+)$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def brainstorms_dir() -> str:
    if BRAINSTORMS_DIR_OVERRIDE is not None:
        return BRAINSTORMS_DIR_OVERRIDE
    from research_assistant import config

    return config.BRAINSTORMS_DIR


def set_brainstorms_dir(path: Optional[str]) -> None:
    global BRAINSTORMS_DIR_OVERRIDE
    BRAINSTORMS_DIR_OVERRIDE = path


# ─── The board ───────────────────────────────────────────────────────────────


def empty_board() -> dict[str, Any]:
    return {"idea": "", "established": [], "objections": [], "open": [], "experiments": [], "next_id": 1}


def clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def paper_keys_in(text: str) -> list[str]:
    """[P2], [P1, P3] → P-keys in first-use order."""
    out = []
    for m in PAPER_KEY_RE.finditer(text or ""):
        for k in re.split(r"\s*,\s*", m.group(1)):
            if k not in out:
                out.append(k)
    return out


def is_empty(board: dict[str, Any]) -> bool:
    return not (board.get("idea") or any(board.get(s) for s in SECTIONS))


def board_text(board: dict[str, Any], papers: dict[str, dict[str, Any]]) -> str:
    """The board as the prompt sees it."""
    lines = [f"IDEA: {board.get('idea') or '(not stated yet)'}"]
    for section in SECTIONS:
        lines.append(f"{TITLES[section]}:")
        items = board.get(section) or []
        if not items:
            lines.append("- (none yet)")
        for item in items:
            refs = f" [{', '.join(item['papers'])}]" if item.get("papers") else ""
            lines.append(f"- ({item['id']}) {item['text']}{refs}")
    if papers:
        lines.append("PAPERS: " + "; ".join(f"[{k}] {short_title(v.get('citation', ''), 8)}"
                                            for k, v in papers.items()))
    return "\n".join(lines)


def board_markdown(board: dict[str, Any], papers: dict[str, dict[str, Any]]) -> str:
    """The board for the export and the page."""
    lines = [f"**Idea.** {board.get('idea') or '—'}", ""]
    for section in SECTIONS:
        lines.append(f"### {LABELS[section]}")
        items = board.get(section) or []
        if not items:
            lines.append("—")
        for item in items:
            refs = ""
            if item.get("papers"):
                refs = " [" + "; ".join(short_title(papers[k].get("citation", ""), 6)
                                        for k in item["papers"] if k in papers) + "]"
            lines.append(f"- {item['text']}{refs}")
        lines.append("")
    if papers:
        lines.append("### Papers")
        for k, rec in papers.items():
            lines.append(f"- **{k}** — {rec.get('citation') or rec.get('document')}")
        lines.append("")
    return "\n".join(lines)


def _new_item(board: dict[str, Any], section: str, text: str, papers: list[str]) -> dict[str, Any]:
    item = {"id": f"{PREFIX[section]}{board['next_id']}", "text": clip(text, ITEM_MAX_CHARS)}
    board["next_id"] += 1
    if section != "open":
        item["papers"] = list(papers)
    return item


def _find(board: dict[str, Any], item_id: str):
    m = _ITEM_ID_RE.match(item_id or "")
    if not m:
        return None, None
    section = SECTION_OF_PREFIX[m.group(1)]
    for i, item in enumerate(board.get(section) or []):
        if item["id"] == item_id:
            return section, i
    return section, None


def add_item(board: dict[str, Any], section: str, text: str, papers: dict[str, Any],
             refs: Optional[list[str]] = None) -> list[str]:
    """A pin or a typed note. Same caps and checks as a delta's add."""
    warnings: list[str] = []
    if section not in SECTIONS:
        return [f"unknown section {section!r}"]
    text = " ".join(str(text or "").split())
    if not text:
        return ["empty item — nothing added"]
    if len(board[section]) >= CAPS[section]:
        return [f"{LABELS[section]} is at its cap of {CAPS[section]} — remove one first"]
    refs = [r for r in (refs or []) if r in papers]
    board[section].append(_new_item(board, section, text, refs))
    return warnings


# ─── Deltas ──────────────────────────────────────────────────────────────────


def parse_delta(text: str) -> Optional[dict[str, Any]]:
    """A JSON object from a model reply: bare, in a ```json fence, or the
    outermost {...} in the text. None when there is none."""
    if not text:
        return None
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        candidates.append(fence.group(1))
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _item_from(raw: Any, papers: dict[str, Any], warnings: list[str], where: str) -> Optional[dict[str, Any]]:
    """(text, papers) out of a delta item — a dict with `text` (+ `papers`)
    or a bare string with [P#] in it; unknown P-keys dropped with a warning."""
    if isinstance(raw, str):
        refs = paper_keys_in(raw)
        text = PAPER_KEY_RE.sub("", raw).strip()
    elif isinstance(raw, dict):
        text = str(raw.get("text") or "").strip()
        refs = raw.get("papers") or []
        if isinstance(refs, str):
            refs = paper_keys_in(refs) or [refs]
        refs = [str(r).strip() for r in refs]
        refs = refs + [k for k in paper_keys_in(text) if k not in refs]
        text = PAPER_KEY_RE.sub("", text).strip()
    else:
        warnings.append(f"{where}: item is not text — dropped")
        return None
    if not text:
        warnings.append(f"{where}: empty item — dropped")
        return None
    unknown = [r for r in refs if r not in papers]
    if unknown:
        warnings.append(f"{where}: unknown paper key(s) {', '.join(unknown)} — dropped from the item")
    return {"text": text, "papers": [r for r in refs if r in papers]}


def apply_delta(board: dict[str, Any], delta: dict[str, Any], papers: dict[str, Any]) -> list[str]:
    """Apply a delta in place, enforcing the schema. Returns the warnings —
    everything the delta asked for that the schema or the caps refused."""
    warnings: list[str] = []
    if not isinstance(delta, dict):
        return ["board update was not an object — ignored"]

    idea = delta.get("idea")
    if isinstance(idea, str) and idea.strip():
        board["idea"] = clip(idea, CAPS["idea"])
    elif idea not in (None, ""):
        warnings.append("idea was not text — ignored")

    removals = delta.get("remove") or []
    if not isinstance(removals, list):
        warnings.append("remove was not a list — ignored")
        removals = []
    for item_id in removals:
        section, i = _find(board, str(item_id))
        if section is None or i is None:
            warnings.append(f"remove {item_id}: no such item")
            continue
        del board[section][i]

    replace = delta.get("replace") or {}
    if isinstance(replace, dict):
        for item_id, raw in replace.items():
            section, i = _find(board, str(item_id))
            if section is None or i is None:
                warnings.append(f"replace {item_id}: no such item")
                continue
            parsed = _item_from(raw, papers, warnings, f"replace {item_id}")
            if parsed is None:
                continue
            item = {"id": item_id, "text": clip(parsed["text"], ITEM_MAX_CHARS)}
            if section != "open":
                item["papers"] = parsed["papers"]
            board[section][i] = item
    else:
        warnings.append("replace was not an object — ignored")

    add = delta.get("add") or {}
    if isinstance(add, dict):
        for section, items in add.items():
            if section not in SECTIONS:
                warnings.append(f"add: unknown section {section!r} — ignored")
                continue
            if not isinstance(items, list):
                items = [items]
            for raw in items:
                parsed = _item_from(raw, papers, warnings, f"add {section}")
                if parsed is None:
                    continue
                if len(board[section]) >= CAPS[section]:
                    warnings.append(f"add {section}: at its cap of {CAPS[section]} — dropped: {clip(parsed['text'], 60)}")
                    continue
                board[section].append(_new_item(board, section, parsed["text"], parsed["papers"]))
    else:
        warnings.append("add was not an object — ignored")

    for key in delta:
        if key not in ("idea", "add", "replace", "remove"):
            warnings.append(f"unknown key {key!r} in the board update — ignored")
    return warnings


# ─── Editing by hand ─────────────────────────────────────────────────────────


def section_edit_text(board: dict[str, Any], section: str) -> str:
    """One item per line, papers in brackets at the end — what the Edit
    expander shows."""
    lines = []
    for item in board.get(section) or []:
        refs = f" [{', '.join(item['papers'])}]" if item.get("papers") else ""
        lines.append(f"{item['text']}{refs}")
    return "\n".join(lines)


def section_from_edit_text(board: dict[str, Any], section: str, text: str, papers: dict[str, Any]) -> list[str]:
    """Replace a section from edited lines. A line equal to an existing
    item keeps its id; a new line gets a new id; unknown P-keys are dropped;
    lines beyond the cap are dropped. Returns warnings."""
    warnings: list[str] = []
    existing = {}
    for it in board.get(section) or []:
        refs = f" [{', '.join(it['papers'])}]" if it.get("papers") else ""
        existing[f"{it['text']}{refs}"] = it
    new_items = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-•").strip()
        if not line:
            continue
        if len(new_items) >= CAPS[section]:
            warnings.append(f"{LABELS[section]}: more than {CAPS[section]} items — the rest dropped")
            break
        if line in existing:
            new_items.append(existing[line])
            continue
        refs = paper_keys_in(line)
        body = PAPER_KEY_RE.sub("", line).strip()
        unknown = [r for r in refs if r not in papers]
        if unknown:
            warnings.append(f"{LABELS[section]}: unknown paper key(s) {', '.join(unknown)} dropped")
        new_items.append(_new_item(board, section, body, [r for r in refs if r in papers]))
    board[section] = new_items
    return warnings


# ─── Papers ──────────────────────────────────────────────────────────────────


def register_paper(papers: dict[str, dict[str, Any]], document: str, citation: str) -> str:
    """The session-stable key for a document, assigned on first sight."""
    for key, rec in papers.items():
        if rec.get("document") == document:
            if citation and not rec.get("citation"):
                rec["citation"] = citation
            return key
    key = f"P{len(papers) + 1}"
    papers[key] = {"document": document, "citation": citation or document}
    return key


def rewrite_keys(text: str, mapping: dict[str, str]) -> str:
    """[S2] → [P5], [P1, P3] → [P7, P2] in one pass, per `mapping`; keys not
    in it are left as they are. Used for S# → P# after a turn and for a
    synthesis's own P# (local to it) → the session's."""
    group_re = re.compile(r"\[((?:S|P)\d+(?:\s*,\s*(?:S|P)\d+)*)\]")

    def _sub(m):
        keys = re.split(r"\s*,\s*", m.group(1))
        return "[" + ", ".join(mapping.get(k, k) for k in keys) + "]"

    return group_re.sub(_sub, text or "")


def resolve_paper_keys(text: str, papers: dict[str, dict[str, Any]], words: int = 6) -> str:
    """[P2] → [<short title>] for the export; unknown keys left as-is."""
    def _sub(m):
        keys = re.split(r"\s*,\s*", m.group(1))
        if not all(k in papers for k in keys):
            return m.group(0)
        return "[" + "; ".join(short_title(papers[k].get("citation", ""), words) for k in keys) + "]"
    return PAPER_KEY_RE.sub(_sub, text or "")


# ─── The session ─────────────────────────────────────────────────────────────


class Session:
    """A brainstorm on disk. `data` mirrors session.json."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    @classmethod
    def new(cls, title: str, origin: Optional[dict[str, Any]] = None) -> "Session":
        now = _now_iso()
        sid = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(2)}"
        s = cls({
            "id": sid, "title": (title or "Untitled brainstorm").strip()[:120], "created_at": now,
            "updated_at": now, "origin": origin, "mode": "explore", "memory": "", "focus_documents": [],
            "board": empty_board(), "papers": {}, "turns": [],
        })
        s.save()
        return s

    @classmethod
    def load(cls, session_id: str) -> "Session":
        path = os.path.join(brainstorms_dir(), session_id, "session.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "board" not in data:
            raise ValueError(f"not a brainstorm session: {path}")
        for key, default in (("papers", {}), ("turns", []), ("focus_documents", []), ("memory", ""),
                             ("mode", "explore"), ("origin", None)):
            data.setdefault(key, default)
        data["board"].setdefault("next_id", 1)
        return cls(data)

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def title(self) -> str:
        return self.data["title"]

    @property
    def board(self) -> dict[str, Any]:
        return self.data["board"]

    @property
    def papers(self) -> dict[str, dict[str, Any]]:
        return self.data["papers"]

    @property
    def turns(self) -> list[dict[str, Any]]:
        return self.data["turns"]

    def dir(self) -> str:
        return os.path.join(brainstorms_dir(), self.id)

    def save(self) -> None:
        self.data["updated_at"] = _now_iso()
        os.makedirs(self.dir(), exist_ok=True)
        atomic_write_json(os.path.join(self.dir(), "session.json"), self.data, ensure_ascii=False)

    def rename(self, title: str) -> None:
        self.data["title"] = (title or self.title).strip()[:120]
        self.save()

    def board_text(self) -> str:
        return board_text(self.board, self.papers)

    def board_markdown(self) -> str:
        return board_markdown(self.board, self.papers)


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """{id, title, created_at, updated_at, turns} per session, newest first."""
    base = brainstorms_dir()
    try:
        names = os.listdir(base)
    except OSError:
        return []
    out = []
    for name in names:
        path = os.path.join(base, name, "session.json")
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            out.append({"id": d["id"], "title": d.get("title", name), "created_at": d.get("created_at", ""),
                        "updated_at": d.get("updated_at", ""), "turns": len(d.get("turns") or [])})
        except (OSError, ValueError, KeyError):
            out.append({"id": name, "title": f"(unreadable) {name}", "created_at": "", "updated_at": "",
                        "turns": 0, "error": True})
    out.sort(key=lambda m: (m.get("updated_at") or "", m["id"]), reverse=True)
    return out[:limit]


def delete_session(session_id: str) -> None:
    shutil.rmtree(os.path.join(brainstorms_dir(), session_id), ignore_errors=True)
````

- [ ] **Step 5: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_brainstorm.py tests/test_prompts_chat.py
```

Expected: 18 passed in the new file; test_prompts_chat still passes (its config assertions are extended in Task 2)

- [ ] **Step 6: Commit**

```bash
git add research_assistant/config.py research_assistant/shared/brainstorm.py tests/test_brainstorm.py
git commit -m "feat(brainstorm): the board and the session — capped sections with stable ids, deltas enforced in code, papers P#, session.json

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Prompts — the position, the board block, the moves, the update call; `parse_position`

**Files:**
- Modify: `research_assistant/prompts.py`, `research_assistant/shared/chat_context.py`
- Test: `tests/test_prompts_chat.py`, `tests/test_chat_context.py`

**Interfaces:**
- Produces: `RESEARCH_CHAT_SYSTEM` with the `Position:` requirement placed before the suggested-questions block; `CHAT_BOARD_BLOCK` (`{board}`); `CHAT_MOVE_USER` — a dict with keys `sharpen, prior, against, test, compare, dig`, each with `{idea}`, `{question}`, `{context}`; `CHAT_BOARD_UPDATE_USER` (`{board}`, `{move}`, `{question}`, `{answer}`; JSON braces doubled); `chat_context.parse_position(answer) -> str`.

- [ ] **Step 1: Add the failing tests**

**Apply this change to `tests/test_prompts_chat.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_prompts_chat.py
+++ b/tests/test_prompts_chat.py
@@ -16,6 +16,18 @@
         self.assertIn("{turns}", p.CHAT_MEMORY_USER)
         self.assertIn("[S1]", p.RESEARCH_CHAT_SYSTEM)
         self.assertIn("💡 Suggested Next Questions", p.RESEARCH_CHAT_SYSTEM)
+        # The brainstorm delta (spec 2026-09-13 §2B–E): a position line, the
+        # board block, one template per move, the board-update call.
+        self.assertIn("`Position:`", p.RESEARCH_CHAT_SYSTEM)
+        self.assertLess(p.RESEARCH_CHAT_SYSTEM.index("Position:"), p.RESEARCH_CHAT_SYSTEM.index("Suggested Next Questions"))
+        self.assertIn("{board}", p.CHAT_BOARD_BLOCK)
+        self.assertEqual(set(p.CHAT_MOVE_USER), {"sharpen", "prior", "against", "test", "compare", "dig"})
+        for tmpl in p.CHAT_MOVE_USER.values():
+            for ph in ("{idea}", "{question}", "{context}"):
+                self.assertIn(ph, tmpl)
+        for ph in ("{board}", "{move}", "{question}", "{answer}"):
+            self.assertIn(ph, p.CHAT_BOARD_UPDATE_USER)
+        self.assertIn('"add"', p.CHAT_BOARD_UPDATE_USER.format(board="", move="", question="", answer=""))
 
     def test_brain_lens_instructions(self):
         self.assertIn("explore", p.BRAIN_LENS_INSTRUCTIONS)
@@ -32,6 +44,11 @@
         )
         self.assertAlmostEqual(c.CHAT_TEMPERATURE, 0.3)
         self.assertEqual(c.CHAT_OLLAMA_OPTIONS["num_ctx"], 4096)
+        self.assertTrue(c.CHAT_BOARD_UPDATE)
+        self.assertEqual(c.CHAT_MOVE_PER_PAPER_CHUNKS, 4)
+        expected_window = 32768 if c.LLM_BACKEND == "openai" else c.CHAT_OLLAMA_OPTIONS["num_ctx"]
+        self.assertEqual(c.CHAT_WINDOW_TOKENS, expected_window)
+        self.assertTrue(c.BRAINSTORMS_DIR.endswith("brainstorms"))
 
 
 if __name__ == "__main__":
```

**Apply this change to `tests/test_chat_context.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_chat_context.py
+++ b/tests/test_chat_context.py
@@ -131,3 +131,12 @@
 
 if __name__ == "__main__":
     unittest.main()
+
+
+class TestParsePosition(unittest.TestCase):
+    def test_plain_bold_and_last_wins(self):
+        self.assertEqual(cc.parse_position("Analysis.\nPosition: the gap closes at 2 T [S1].\n\n### 💡 Suggested"), "the gap closes at 2 T [S1].")
+        self.assertEqual(cc.parse_position("**Position:** objection — Mott [S2] shows no gap."), "objection — Mott [S2] shows no gap.")
+        self.assertEqual(cc.parse_position("**Position**: first.\nPosition: second."), "second.")
+        self.assertEqual(cc.parse_position("No line here."), "")
+        self.assertEqual(cc.parse_position(""), "")
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_prompts_chat.py tests/test_chat_context.py
```

Expected: fail — `AttributeError` on the new names

- [ ] **Step 3: Apply the prompts and the parser**

**Apply this change to `research_assistant/prompts.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/prompts.py
+++ b/research_assistant/prompts.py
@@ -73,6 +73,12 @@
 - Say plainly what the sources do not cover.
 - Keep your answers concise, structured, and scientifically rigorous.
 
+Take a position. End your analysis — before the suggested questions — with one
+line that begins `Position:` and is one of: a sharper statement of the idea
+than the one on the board; the strongest objection to it, with the paper that
+raises it; or the single question that would settle the most. Never all
+three, never a hedge.
+
 Conclude every response with 2 to 3 concrete, testable next research directions formatted as:
 ### 💡 Suggested Next Questions
 - <First sharp follow-up question or exploration angle>
@@ -125,6 +131,80 @@
     "Return only the query."
 )
 
+# ─── The brainstorm board (spec 2026-09-13-brainstorm-workspace §2B–D) ─────
+# Appended to the system prompt when a session is attached. The board is
+# what we think so far; the memory (below it) is what was said.
+CHAT_BOARD_BLOCK = (
+    "The board — what we think so far. Do not repeat it; build on it, correct "
+    "it, or push it further. Board items cite papers by session key [P#]; "
+    "this turn's sources are keyed [S#] and you cite those.\n{board}"
+)
+
+# One user template per move. {question} is the researcher's steer (may be
+# empty), {idea} the board's idea, {context} the keyed sources.
+CHAT_MOVE_USER = {
+    "sharpen": (
+        "Sharpen the idea.\nIdea as it stands: {idea}\n{question}\n\n"
+        "Restate it as ONE falsifiable claim: the system, the quantity, the "
+        "condition, and the expected sign or magnitude. Then list the "
+        "assumptions it rests on, and say which of them the sources support "
+        "and which they do not.\n\nSources retrieved for this turn:\n{context}"
+    ),
+    "prior": (
+        "What has been done on this idea?\nIdea: {idea}\n{question}\n\n"
+        "The sources are one summary per shortlisted paper. For each paper, "
+        "ONE line: what it did relative to the idea (system, method, result). "
+        "Then: which paper is closest to the idea, and what none of them did.\n\n"
+        "Sources retrieved for this turn:\n{context}"
+    ),
+    "against": (
+        "Argue against the idea.\nIdea: {idea}\n{question}\n\n"
+        "Give the strongest objection the sources support, naming the paper "
+        "and the specific result. Then say what would have to be wrong, or "
+        "different, for the idea to survive it.\n\nSources retrieved for this turn:\n{context}"
+    ),
+    "test": (
+        "How would we test the idea?\nIdea: {idea}\n{question}\n\n"
+        "Propose the measurement or simulation that would decide it: the "
+        "observable, the expected signature if the idea is right, the closest "
+        "method among the sources (say which), and what would count as a "
+        "negative result.\n\nSources retrieved for this turn:\n{context}"
+    ),
+    "compare": (
+        "Compare two papers against the idea.\nIdea: {idea}\n{question}\n\n"
+        "The sources come from exactly two papers. Where do they agree, where "
+        "do they differ (systems, conditions, numbers, interpretation), and "
+        "which is closer to the idea, and why?\n\nSources retrieved for this turn:\n{context}"
+    ),
+    "dig": (
+        "Dig into one paper.\nIdea: {idea}\n{question}\n\n"
+        "The sources come from one paper. What does it establish (with the "
+        "numbers it gives), by what method, and what are its limits relative "
+        "to the idea?\n\nSources retrieved for this turn:\n{context}"
+    ),
+}
+
+# The structured call after each turn. Code enforces the schema (brainstorm.apply_delta).
+CHAT_BOARD_UPDATE_USER = (
+    "You maintain the board of a research brainstorm. Given the board, the "
+    "move, the researcher's question and the answer just given, return the "
+    "changes to the board as ONE JSON object and nothing else:\n"
+    '{{"idea": <new one-paragraph idea, or null if unchanged>,\n'
+    ' "add": {{"established": [<item>...], "objections": [<item>...], '
+    '"open": [<item>...], "experiments": [<item>...]}},\n'
+    ' "replace": {{"<item id>": <item>}},\n'
+    ' "remove": ["<item id>", ...]}}\n'
+    'where <item> is {{"text": "<one sentence, at most 240 characters>", '
+    '"papers": ["P#", ...]}} and "papers" may only use keys listed under PAPERS '
+    "on the board. Add only what the answer established, objected, asked or "
+    "proposed that is NOT already on the board; replace an item when the "
+    "answer sharpened it; remove one the answer refuted. Change the idea only "
+    "on a free or sharpen move, and only when the answer's Position line "
+    "sharpened it — an objection or a test never rewrites the idea. Prefer "
+    'few changes. Empty sections may be omitted.\n\nBOARD:\n{board}\n\nMOVE: {move}\n'
+    "QUESTION: {question}\n\nANSWER:\n{answer}"
+)
+
 # Older turns leave the prompt only by being folded into this.
 CHAT_MEMORY_USER = (
     "Summarise these earlier turns of a research conversation in at most 120 "
```

**Apply this change to `research_assistant/shared/chat_context.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/shared/chat_context.py
+++ b/research_assistant/shared/chat_context.py
@@ -180,3 +180,14 @@
                     break
 
     return suggestions[:4]
+
+
+_POSITION_RE = re.compile(r"^\s*(?:\*\*|__)?\s*Position\s*:?\s*(?:\*\*|__)?\s*:?\s*(.+?)\s*$", re.I | re.M)
+
+
+def parse_position(answer: str) -> str:
+    """The `Position:` line a brainstorm answer ends its analysis with
+    (spec 2026-09-13-brainstorm §2E) — bold or plain, colon inside or outside
+    the bold; the last one if the model wrote several. Empty when absent."""
+    found = _POSITION_RE.findall(answer or "")
+    return found[-1].strip().strip("*_ ") if found else ""
```

- [ ] **Step 4: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_prompts_chat.py tests/test_chat_context.py tests/test_agent7_chat.py
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add research_assistant/prompts.py research_assistant/shared/chat_context.py tests/test_prompts_chat.py tests/test_chat_context.py
git commit -m "feat(chat): a Position line in every answer; the board block, one template per move, the board-update call

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `ResearchChat` — sessions, the board, moves, the update, dives

**Files:**
- Modify: `research_assistant/agents/agent7_research_chat.py`
- Test: `tests/test_agent7_brainstorm.py` (new), `tests/test_agent7_chat.py` (one line)

**Interfaces:**
- Consumes: Tasks 1–2; `retrieve.rank_documents`, `retrieve.gate_documents`, `retrieve.research_answer(query, mode=, on_progress=)` (synthesis-depth, on the base commit).
- Produces: module constants `MOVES`, `IDEA_MOVES`, `PAPER_MOVES`, `MOVE_LABELS`; `ResearchChat(top_k, search_resources, mode, session=None)`; `attach(session)`; `session`, `board`, `papers`, `last_move`, `last_position`, `last_board_warnings`; `stream_turn(user_message, mode=None, move="free", papers=())`; `pin(text, section="established", refs=()) -> warnings`; `edit_section(section, text) -> warnings`; `set_idea(text)`; `dive(question, on_progress=None) -> turn`; `seed_from_synthesis(query, answer) -> turn`. Turn records gain `move`, `position`, `paper_keys`, `board_delta`, `board_warnings` (dives also `notes`, `timings`). `last_budget["num_ctx"]` becomes `last_budget["window"]`. `_fit` budgets against `CHAT_WINDOW_TOKENS`.

Why the moves live inside `stream_turn` rather than separate methods: one code path for budget, streaming, key checks, registration, the update and the save — a move changes only `_plan_move`'s retrieval and template.

- [ ] **Step 1: Write the failing tests** (they build on the studio's `ChatTestCase` fixture and `_hit`)

**Create `tests/test_agent7_brainstorm.py` with exactly this content:**

````python
"""Agent 7 with a session attached: the board in the prompt, the moves, the
S# → P# mapping, the board update and its failure paths, dives and seeding
(spec 2026-09-13-brainstorm-workspace §2A–G). Builds on the studio's fakes."""

import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from research_assistant.agents import agent7_research_chat as a7
from research_assistant.shared import brainstorm as bs
from tests.test_agent7_chat import ChatTestCase, _hit


class BrainstormCase(ChatTestCase):
    """The studio's fixture plus: a session on a temp dir, a chat fake that
    answers the board-update prompt with JSON, and a `Position:` line."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        bs.set_brainstorms_dir(self.tmp.name)
        self.addCleanup(bs.set_brainstorms_dir, None)
        self.session = bs.Session.new("Hopping in MoS2")
        self.agent.attach(self.session)

        self.delta = {"idea": "Hopping dominates below 100 K.",
                      "add": {"established": [{"text": "Hopping dominates [P1]", "papers": ["P1"]}],
                              "open": ["Does gating change the exponent?"]}}
        self.update_prompts = []

        def fake_chat(messages, model=None, temperature=None, options=None, **kw):
            content = messages[-1]["content"]
            self.chats.append({"content": content, "temperature": temperature, "options": options})
            if "Rewrite the new question" in content:
                return types.SimpleNamespace(content="limitations of hopping study")
            if "You maintain the board" in messages[0]["content"]:
                self.update_prompts.append(messages)
                return types.SimpleNamespace(content=self.update_reply(messages))
            return types.SimpleNamespace(content="MEMORY: earlier turns summarised")

        p = patch.object(a7, "chat", side_effect=fake_chat)
        p.start()
        self.addCleanup(p.stop)
        self.answer = ("Hopping dominates [S1]. See also [S2]. Unknown [S9].\n\n"
                       "Position: the exponent is 1/2 in every sample [S1].\n\n"
                       "### 💡 Suggested Next Questions:\n- What are the high temperature limits?\n")

    def update_reply(self, messages):
        return json.dumps(self.delta)


class TestBoardInPrompt(BrainstormCase):
    def test_system_prompt_carries_the_board_and_the_position_is_parsed(self):
        self._turn("What does the MoS2 paper say?")
        sys_msg = self.streams[-1]["messages"][0]["content"]
        self.assertIn("The board — what we think so far", sys_msg)
        self.assertIn("IDEA: (not stated yet)", sys_msg)         # the board as it was when the turn started
        self.assertEqual(self.agent.last_position, "the exponent is 1/2 in every sample [S1].")
        self.assertEqual(self.agent.turns[-1]["position"], self.agent.last_position)
        self.assertEqual(self.agent.turns[-1]["move"], "free")

    def test_cited_sources_become_session_papers_and_the_update_sees_p_keys(self):
        self._turn("q")
        self.assertEqual(self.agent.turns[-1]["paper_keys"], {"S1": "P1", "S2": "P1"})   # both hits are a.pdf
        self.assertEqual(self.session.papers, {"P1": {"document": "a.pdf", "citation": "Title a.pdf"}})
        prompt = self.update_prompts[-1][-1]["content"]
        self.assertIn("Hopping dominates [P1]. See also [P1]. Unknown [S9].", prompt)
        self.assertIn("MOVE: free", prompt)
        self.assertIn("PAPERS: [P1] Title a.pdf", prompt)

    def test_delta_applied_and_saved(self):
        self._turn("q")
        board = self.session.board
        self.assertEqual(board["idea"], "Hopping dominates below 100 K.")
        self.assertEqual(board["established"][0]["papers"], ["P1"])
        self.assertEqual(board["open"][0]["text"], "Does gating change the exponent?")
        self.assertEqual(self.agent.last_board_warnings, [])
        reloaded = bs.Session.load(self.session.id)
        self.assertEqual(reloaded.board["idea"], "Hopping dominates below 100 K.")
        self.assertEqual(len(reloaded.turns), 1)
        self.assertEqual(reloaded.turns[0]["board_delta"], self.delta)
        # The next turn's prompt shows the updated board.
        self._turn("and then?")
        self.assertIn("IDEA: Hopping dominates below 100 K.", self.streams[-1]["messages"][0]["content"])
        self.assertIn("- (E1) Hopping dominates [P1]", self.streams[-1]["messages"][0]["content"])


class TestBoardUpdateFailures(BrainstormCase):
    def test_unparseable_twice_leaves_the_board_and_warns(self):
        self.update_reply = lambda messages: "I'd rather not."
        self._turn("q")
        self.assertEqual(len(self.update_prompts), 2)
        self.assertIn("Return only the JSON object", self.update_prompts[1][-1]["content"])
        self.assertTrue(bs.is_empty(self.session.board))
        self.assertEqual(self.agent.last_board_warnings, ["board update was not valid JSON twice — board unchanged"])
        self.assertIsNone(self.agent.turns[-1]["board_delta"])

    def test_second_attempt_can_succeed(self):
        replies = iter(["no", "```json\n" + json.dumps({"add": {"open": ["q?"]}}) + "\n```"])
        self.update_reply = lambda messages: next(replies)
        self._turn("q")
        self.assertEqual(self.session.board["open"][0]["text"], "q?")

    def test_rejections_are_warnings(self):
        self.delta = {"add": {"established": [{"text": "x", "papers": ["P9"]}], "nope": ["y"]}}
        self._turn("q")
        self.assertEqual(self.session.board["established"][0]["papers"], [])
        joined = "\n".join(self.agent.last_board_warnings)
        self.assertIn("unknown paper key(s) P9", joined)
        self.assertIn("unknown section 'nope'", joined)
        self.assertEqual(self.agent.turns[-1]["board_warnings"], self.agent.last_board_warnings)

    def test_update_off(self):
        with patch.object(a7, "CHAT_BOARD_UPDATE", False):
            self._turn("q")
        self.assertEqual(self.update_prompts, [])
        self.assertTrue(bs.is_empty(self.session.board))
        self.assertEqual(self.session.papers["P1"]["document"], "a.pdf")   # papers still registered

    def test_no_session_no_update(self):
        agent = a7.ResearchChat(top_k=5, search_resources=(None, None, [], []))
        "".join(agent.stream_turn("q"))
        self.assertEqual(self.update_prompts, [])
        self.assertEqual(agent.turns[-1]["paper_keys"], {})
        self.assertNotIn("The board", self.streams[-1]["messages"][0]["content"])


class TestMoves(BrainstormCase):
    def setUp(self):
        super().setUp()
        self.session.board["idea"] = "Variable-range hopping sets the low-T conductance of MoS2 networks."
        self.session.save()

    def test_idea_moves_search_the_idea_and_use_their_template(self):
        for move, needle in (("sharpen", "Restate it as ONE falsifiable claim"),
                             ("against", "strongest objection"),
                             ("test", "negative result")):
            self.searches.clear()
            idea_before = self.session.board["idea"]          # the update after each turn may sharpen it
            "".join(self.agent.stream_turn("", move=move))
            self.assertEqual(self.searches[0]["query"], idea_before)
            user = self.streams[-1]["messages"][-1]["content"]
            self.assertIn(needle, user)
            self.assertIn("Idea as it stands" if move == "sharpen" else "Idea:", user)
            self.assertNotIn("Researcher's steer", user)
            self.assertEqual(self.agent.turns[-1]["move"], move)
            self.assertEqual(self.agent.turns[-1]["user_message"], a7.MOVE_LABELS[move])
        self.assertFalse(any("Rewrite the new question" in c["content"] for c in self.chats))   # no condense

    def test_steer_is_appended(self):
        "".join(self.agent.stream_turn("focus on the 10-50 K range", move="against"))
        user = self.streams[-1]["messages"][-1]["content"]
        self.assertIn("Researcher's steer: focus on the 10-50 K range", user)
        self.assertEqual(self.agent.turns[-1]["user_message"], "Argue against: focus on the 10-50 K range")

    def test_prior_uses_the_summary_shortlist(self):
        ranked = [{"document": "x.pdf", "citation": "X paper", "summary": "X did this.", "score": 0.9},
                  {"document": "y.pdf", "citation": "Y paper", "summary": "Y did that.", "score": 0.8}]
        # Patched on the real module: `from research_assistant.shared import retrieve`
        # resolves the package attribute, so a sys.modules stand-in would be
        # ignored whenever an earlier test already imported the real thing.
        with patch("research_assistant.shared.retrieve.rank_documents", lambda q, k=6: ranked), \
             patch("research_assistant.shared.retrieve.gate_documents", lambda q, r: r[:1]):
            "".join(self.agent.stream_turn("", move="prior"))
        self.assertEqual(self.searches, [])                                # no chunk search
        user = self.streams[-1]["messages"][-1]["content"]
        self.assertIn("[S1] X paper — p.summary, summary\nX did this.", user)
        self.assertNotIn("Y did that", user)                               # gated out
        self.assertIn("ONE line", user)
        self.assertEqual(self.agent.last_sources[0]["document"], "x.pdf")

    def test_compare_and_dig_search_per_paper(self):
        bs.register_paper(self.session.papers, "a.pdf", "Title a.pdf")
        bs.register_paper(self.session.papers, "b.pdf", "Title b.pdf")
        "".join(self.agent.stream_turn("", move="compare", papers=("P1", "P2")))
        self.assertEqual([s["doc_filter"] for s in self.searches], [{"a.pdf"}, {"b.pdf"}])
        self.assertEqual([s["top_k"] for s in self.searches], [4, 4])
        self.assertEqual(self.agent.turns[-1]["user_message"], "Compare P1 vs P2")
        self.assertIn("exactly two papers", self.streams[-1]["messages"][-1]["content"])
        # every source of a paper move is registered, cited or not
        self.assertEqual(set(self.agent.turns[-1]["paper_keys"].values()), {"P1", "P2"})
        self.searches.clear()
        "".join(self.agent.stream_turn("its limits", move="dig", papers=("P2",)))
        self.assertEqual([s["doc_filter"] for s in self.searches], [{"b.pdf"}])
        self.assertEqual(self.agent.turns[-1]["user_message"], "Dig in P2: its limits")

    def test_unknown_move(self):
        with self.assertRaises(ValueError):
            "".join(self.agent.stream_turn("q", move="dance"))

    def test_idea_move_without_an_idea_uses_the_text(self):
        self.session.board["idea"] = ""
        "".join(self.agent.stream_turn("gating changes the exponent", move="sharpen"))
        self.assertEqual(self.searches[0]["query"], "gating changes the exponent")
        self.assertIn("Idea as it stands: gating changes the exponent", self.streams[-1]["messages"][-1]["content"])


class TestPinEditAttach(BrainstormCase):
    def test_pin_note_edit_and_idea(self):
        bs.register_paper(self.session.papers, "a.pdf", "Title a.pdf")
        self.assertEqual(self.agent.pin("Exponent is 1/2 [P1]", refs=("P1",)), [])
        self.assertEqual(self.session.board["established"][0], {"id": "E1", "text": "Exponent is 1/2 [P1]", "papers": ["P1"]})
        self.assertEqual(self.agent.pin("Try a gate", section="experiments"), [])
        self.assertEqual(self.agent.edit_section("open", "one?\ntwo?"), [])
        self.assertEqual([i["text"] for i in self.session.board["open"]], ["one?", "two?"])
        self.agent.set_idea("  a sharper   idea ")
        self.assertEqual(bs.Session.load(self.session.id).board["idea"], "a sharper idea")

    def test_attach_restores_the_conversation(self):
        self._turn("first")
        self._turn("second")
        self.agent.memory = "what was said"
        self.agent._save()
        fresh = a7.ResearchChat(top_k=5, search_resources=(None, None, [], []), session=bs.Session.load(self.session.id))
        self.assertEqual(len(fresh.turns), 2)
        self.assertEqual(fresh.turn_count, 2)
        self.assertEqual(fresh.history[0], {"role": "user", "content": "first"})
        self.assertEqual(fresh.history[3]["role"], "assistant")
        self.assertEqual(fresh.memory, "what was said")
        self.assertEqual(fresh.focus_documents, {"a.pdf"})
        self.assertEqual(fresh.last_position, "the exponent is 1/2 in every sample [S1].")
        self.assertEqual(fresh.last_suggestions, self.agent.last_suggestions)
        fresh.clear_history()
        self.assertEqual(bs.Session.load(self.session.id).turns, [])
        self.assertEqual(bs.Session.load(self.session.id).board["idea"], "Hopping dominates below 100 K.")   # kept

    def test_export_starts_with_the_board(self):
        self._turn("q")
        with tempfile.TemporaryDirectory() as d, patch.object(a7, "DRAFTS_DIR", d):
            text = open(self.agent.export_conversation(), encoding="utf-8").read()
        self.assertIn("*Session:* Hopping in MoS2", text)
        self.assertIn("## 🧭 The board", text)
        self.assertIn("**Idea.** Hopping dominates below 100 K.", text)
        self.assertLess(text.index("The board"), text.index("Conversation Transcript"))
        self.assertEqual(self.agent.export_conversation_json()["session"]["id"], self.session.id)


RESULT = {
    "suggestion": "Both show hopping [P1]; the review disagrees [P2].\n\n### 💡 Suggested Next Questions:\n- Why?",
    "citations": ["Paper one", "Paper two"], "passages": [],
    "selected": [{"key": "P1", "document": "x.pdf", "citation": "Paper one", "score": 0.9, "summary": "s1"},
                 {"key": "P2", "document": "y.pdf", "citation": "Paper two", "score": 0.8, "summary": "s2"}],
    "keys": {"P1": "Paper one", "P2": "Paper two"}, "unverified_citations": [], "irrelevant_cited": [],
    "notes": [{"key": "P1", "notes": "n1", "relevant": True}], "timings": {"total": 3.0}, "mode": "map_reduce",
}


class TestDiveAndSeed(BrainstormCase):
    def _install_retrieve(self, fn):
        p = patch("research_assistant.shared.retrieve.research_answer", fn)
        p.start()
        self.addCleanup(p.stop)

    def test_dive_rewrites_the_synthesis_keys_to_session_keys(self):
        # a.pdf is already P1 in the session; the synthesis's own P1 is x.pdf.
        bs.register_paper(self.session.papers, "a.pdf", "Title a.pdf")
        calls = []
        self._install_retrieve(lambda q, mode=None, on_progress=None, **k: calls.append((q, mode)) or RESULT)
        turn = self.agent.dive("what has been measured?", on_progress=lambda *a: None)
        self.assertEqual(calls, [("what has been measured?", "map_reduce")])
        self.assertEqual(self.session.papers["P2"]["document"], "x.pdf")
        self.assertEqual(self.session.papers["P3"]["document"], "y.pdf")
        self.assertEqual(turn["move"], "dive")
        self.assertEqual(turn["user_message"], "Go and read: what has been measured?")
        self.assertTrue(turn["content"].startswith("Both show hopping [P2]; the review disagrees [P3]."))
        self.assertEqual(turn["paper_keys"], {"P1": "P2", "P2": "P3"})
        self.assertEqual([(s["key"], s["cited"]) for s in turn["sources"]], [("P2", True), ("P3", True)])
        self.assertEqual(turn["position"], "")                 # a synthesis takes no position line
        self.assertEqual(turn["notes"][0]["key"], "P1")
        self.assertIn("MOVE: dive", self.update_prompts[-1][-1]["content"])
        self.assertEqual(self.agent.history[-1]["content"], "Both show hopping [Paper one]; the review disagrees [Paper two].\n\n### 💡 Suggested Next Questions:\n- Why?")
        self.assertEqual(len(bs.Session.load(self.session.id).turns), 1)

    def test_dive_with_nothing_raises_and_adds_no_turn(self):
        self._install_retrieve(lambda q, **k: None)
        with self.assertRaises(RuntimeError):
            self.agent.dive("q")
        self.assertEqual(self.agent.turns, [])

    def test_seed_sets_the_idea_and_the_first_turn(self):
        turn = self.agent.seed_from_synthesis("hopping in MoS2 networks", RESULT)
        self.assertEqual(self.session.board["idea"], "Hopping dominates below 100 K.")   # the update's idea wins
        self.assertEqual(turn["move"], "seed")
        self.assertEqual(turn["user_message"], "Brainstorm this: hopping in MoS2 networks")
        self.assertEqual(self.session.papers["P1"]["document"], "x.pdf")
        # without an update the query itself is the idea
        with patch.object(a7, "CHAT_BOARD_UPDATE", False):
            s2 = bs.Session.new("other")
            agent2 = a7.ResearchChat(top_k=5, search_resources=(None, None, [], []), session=s2)
            agent2.seed_from_synthesis("the query", RESULT)
        self.assertEqual(s2.board["idea"], "the query")


if __name__ == "__main__":
    unittest.main()
````

- [ ] **Step 2: The one-line change in the studio's own tests** — the window is no longer `num_ctx`:

**Apply this change to `tests/test_agent7_chat.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_agent7_chat.py
+++ b/tests/test_agent7_chat.py
@@ -162,7 +162,7 @@
 class TestBudgetAndMemory(ChatTestCase):
     def test_older_turns_fold_into_memory(self):
         self.answer = "answer " * 120 + "[S1]."
-        with patch.object(a7, "CHAT_OLLAMA_OPTIONS", {"num_ctx": 1400}), patch.object(a7, "CHAT_CONTEXT_MAX_CHARS", 600):
+        with patch.object(a7, "CHAT_WINDOW_TOKENS", 1400), patch.object(a7, "CHAT_CONTEXT_MAX_CHARS", 600):
             for i in range(5):
                 self._turn(f"question number {i}")
         self.assertIn("MEMORY", self.agent.memory)
```

- [ ] **Step 3: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_agent7_brainstorm.py
```

Expected: fail — `attach` / `move=` unknown

- [ ] **Step 4: Change the agent**

**Apply this change to `research_assistant/agents/agent7_research_chat.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/agents/agent7_research_chat.py
+++ b/research_assistant/agents/agent7_research_chat.py
@@ -34,23 +34,30 @@
 
 from research_assistant.config import (
     CHAT_ANSWER_RESERVE_TOKENS,
+    CHAT_BOARD_UPDATE,
     CHAT_CONDENSE,
     CHAT_CONTEXT_MAX_CHARS,
     CHAT_FOCUS_TOP_K,
     CHAT_MODEL,
+    CHAT_MOVE_PER_PAPER_CHUNKS,
     CHAT_OLLAMA_OPTIONS,
     CHAT_PER_DOC_CAP,
     CHAT_TEMPERATURE,
+    CHAT_WINDOW_TOKENS,
     DRAFTS_DIR,
 )
 from research_assistant.prompts import (
     BRAIN_LENS_INSTRUCTIONS,
+    CHAT_BOARD_BLOCK,
+    CHAT_BOARD_UPDATE_USER,
     CHAT_CONDENSE_USER,
     CHAT_MEMORY_USER,
+    CHAT_MOVE_USER,
     CHAT_TURN_NO_CONTEXT,
     CHAT_TURN_USER,
     RESEARCH_CHAT_SYSTEM,
 )
+from research_assistant.shared import brainstorm as bs
 from research_assistant.shared.chat_context import (
     cap_per_document,
     cited_keys,
@@ -60,6 +67,7 @@
     merge_results,
     messages_tokens,
     parse_brainstorm_suggestions,
+    parse_position,
     resolve_keys,
 )
 from research_assistant.shared.db import load_search_resources
@@ -71,6 +79,17 @@
 
 SYSTEM_PROMPT = RESEARCH_CHAT_SYSTEM
 
+# The moves (spec 2026-09-13-brainstorm §2D). A lens sets the tone; a move
+# changes what is retrieved and what is asked. IDEA_MOVES work from the
+# board's idea (or the typed text when the board has none yet).
+MOVES = ("free", "sharpen", "prior", "against", "test", "compare", "dig")
+IDEA_MOVES = ("sharpen", "prior", "against", "test")
+PAPER_MOVES = ("compare", "dig")
+MOVE_LABELS = {
+    "free": "", "sharpen": "Sharpen", "prior": "What's been done", "against": "Argue against",
+    "test": "Test it", "compare": "Compare", "dig": "Dig in", "dive": "Go and read", "seed": "Brainstorm this",
+}
+
 HELP_TEXT = """
 ╔══════════════════════════════════════════════════════════╗
 ║               Available Commands                         ║
@@ -94,9 +113,14 @@
     grounded answer with [S#] keys → validate citations and extract suggestions.
     """
 
-    def __init__(self, top_k: int = 5, search_resources: tuple | None = None, mode: str = "explore"):
+    def __init__(self, top_k: int = 5, search_resources: tuple | None = None, mode: str = "explore",
+                 session: "bs.Session | None" = None):
         self.top_k = top_k
         self.mode = mode if mode in BRAIN_LENS_INSTRUCTIONS else "explore"
+        self.session: bs.Session | None = None         # the brainstorm on disk, when attached
+        self.last_move = "free"
+        self.last_position = ""                        # the answer's `Position:` line
+        self.last_board_warnings: list[str] = []       # what the board update refused
         self.history: list[dict[str, str]] = []       # list of {"role": ..., "content": ...}
         self.turns: list[dict[str, Any]] = []          # structured per-turn history with sources
         self.memory = ""                               # rolling summary of turns that no longer fit
@@ -114,6 +138,52 @@
             logger.info("Loading search resources…")
             self.collection, self.bm25, self.texts, self.metadatas = load_search_resources()
         logger.info("Ready. %d chunks in database.", len(self.texts))
+        if session is not None:
+            self.attach(session)
+
+    # ── Session (spec §2A) ───────────────────────────────────────────────
+
+    def attach(self, session: "bs.Session") -> None:
+        """Resume a brainstorm: its turns, memory, lens and focus become this
+        agent's; history is rebuilt from the turns (the studio stores the
+        title-resolved answer per turn)."""
+        self.session = session
+        d = session.data
+        self.turns = list(d.get("turns") or [])
+        self.memory = d.get("memory") or ""
+        self.mode = d.get("mode") if d.get("mode") in BRAIN_LENS_INSTRUCTIONS else "explore"
+        self.focus_documents = set(d.get("focus_documents") or [])
+        self.history = []
+        for t in self.turns:
+            self.history.append({"role": "user", "content": t.get("user_message", "")})
+            self.history.append({"role": "assistant", "content": t.get("resolved_answer") or t.get("content", "")})
+        self.turn_count = len(self.turns)
+        last = self.turns[-1] if self.turns else {}
+        self.last_sources = list(last.get("sources") or [])
+        self.last_suggestions = list(last.get("suggestions") or [])
+        self.last_warnings = list(last.get("warnings") or [])
+        self.last_position = last.get("position", "")
+        self.last_query = last.get("condensed_query", "")
+        self.last_move = last.get("move", "free")
+        self.last_board_warnings = list(last.get("board_warnings") or [])
+
+    def _save(self) -> None:
+        if self.session is None:
+            return
+        d = self.session.data
+        d["turns"] = self.turns
+        d["memory"] = self.memory
+        d["mode"] = self.mode
+        d["focus_documents"] = sorted(self.focus_documents)
+        self.session.save()
+
+    @property
+    def board(self) -> dict | None:
+        return self.session.board if self.session is not None else None
+
+    @property
+    def papers(self) -> dict:
+        return self.session.papers if self.session is not None else {}
 
     # ── Retrieval ────────────────────────────────────────────────────────
 
@@ -174,6 +244,9 @@
         prompt = RESEARCH_CHAT_SYSTEM
         if lens_instruction:
             prompt = f"{prompt}\n\n[Active Brainstorming Lens]\n{lens_instruction}"
+        if self.session is not None:
+            # The board is what we think; the memory below it is what was said.
+            prompt = f"{prompt}\n\n{CHAT_BOARD_BLOCK.format(board=self.session.board_text())}"
         if self.memory:
             prompt = f"{prompt}\n\nConversation so far: {self.memory}"
         return prompt
@@ -198,8 +271,12 @@
             )
 
     def _fit(self, user_content: str, mode: str | None = None) -> tuple[list[dict], int]:
-        """Fit history within window, folding older turns into memory if needed."""
-        num_ctx = CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096)
+        """Fit history within the window, folding older turns into memory if needed.
+
+        The window is CHAT_WINDOW_TOKENS — Ollama's num_ctx, or a hosted
+        model's own (much larger) window — not num_ctx unconditionally.
+        """
+        num_ctx = CHAT_WINDOW_TOKENS
         folded = 0
         for _ in range(2):
             fixed = (
@@ -218,29 +295,152 @@
 
     # ── Public API ───────────────────────────────────────────────────────
 
-    def stream_turn(self, user_message: str, mode: str | None = None):
-        """Process one user turn: condense, search, fit window, stream response, and update history."""
+    # ── Moves (spec §2D) ─────────────────────────────────────────────────
+
+    def _idea(self, text: str) -> str:
+        board = self.board or {}
+        return (board.get("idea") or "").strip() or text.strip()
+
+    def _search_summaries(self, query: str) -> list[dict]:
+        """`prior`: the stage-1 shortlist — one summary per gated paper — as
+        hit-shaped results so format_context keys them like chunks."""
+        from research_assistant.shared import retrieve
+
+        ranked = retrieve.rank_documents(query)
+        if not ranked:
+            return []
+        kept = retrieve.gate_documents(query, ranked)
+        return [{
+            "text": r.get("summary", ""), "chunk_index": None, "rrf_score": r.get("score"),
+            "metadata": {"document": r.get("document"), "citation_source": r.get("citation") or r.get("document"),
+                         "page": "summary", "section": "summary"},
+        } for r in kept]
+
+    def _search_papers(self, query: str, keys: tuple) -> list[dict]:
+        """`compare` / `dig`: a restricted search per chosen paper."""
+        out = []
+        for key in keys:
+            rec = self.papers.get(key)
+            if not rec:
+                continue
+            out += hybrid_search(
+                query, self.collection, self.bm25, self.texts, self.metadatas,
+                top_k=CHAT_MOVE_PER_PAPER_CHUNKS, doc_filter={rec["document"]},
+                exclude_types={"figure_description"},
+            )
+        return out
+
+    def _plan_move(self, user_message: str, move: str, papers: tuple) -> tuple[str, list[dict], str, str]:
+        """(query, results, user_content, shown_message) for a move."""
+        text = (user_message or "").strip()
+        if move == "free":
+            query = self._condense(text)
+            results = self._search(query)
+            context, _ = format_context(results, CHAT_CONTEXT_MAX_CHARS)
+            content = (CHAT_TURN_USER.format(question=text, context=context) if context
+                       else CHAT_TURN_NO_CONTEXT.format(question=text))
+            return query, results, content, text
+
+        idea = self._idea(text)
+        if move in IDEA_MOVES:
+            query = idea
+            results = self._search_summaries(query) if move == "prior" else self._search(query)
+            shown = MOVE_LABELS[move] + (f": {text}" if text and text != idea else "")
+        elif move in PAPER_MOVES:
+            query = idea or text
+            results = self._search_papers(query, papers)
+            shown = f"{MOVE_LABELS[move]} {' vs '.join(papers)}" + (f": {text}" if text else "")
+        else:
+            raise ValueError(f"unknown move {move!r}")
+        context, _ = format_context(results, CHAT_CONTEXT_MAX_CHARS)
+        steer = f"Researcher's steer: {text}" if text and text != idea else ""
+        content = CHAT_MOVE_USER[move].format(idea=idea or "(none on the board yet)", question=steer,
+                                              context=context or "(nothing retrieved)")
+        return query, results, content, shown
+
+    # ── The board (spec §2C) ─────────────────────────────────────────────
+
+    def _register_cited(self, sources: list[dict], always: bool = False) -> dict[str, str]:
+        """S# → P# for this turn's cited sources (all of them for a paper
+        move); registers new papers in the session."""
+        if self.session is None:
+            return {}
+        mapping = {}
+        for s in sources:
+            if (s.get("cited") or always) and s.get("document"):
+                mapping[s["key"]] = bs.register_paper(self.papers, s["document"], s.get("citation") or "")
+        return mapping
+
+    def _update_board(self, move: str, question: str, answer_p: str) -> tuple[dict | None, list[str]]:
+        """One structured call; the schema is enforced by bs.apply_delta.
+        Returns (delta applied or None, warnings)."""
+        if self.session is None or not CHAT_BOARD_UPDATE:
+            return None, []
+        prompt = CHAT_BOARD_UPDATE_USER.format(board=self.session.board_text(), move=move,
+                                               question=question[:800], answer=answer_p[:6000])
+        messages = [{"role": "user", "content": prompt}]
+        delta = None
+        for attempt in range(2):
+            try:
+                reply = chat(messages, model=CHAT_MODEL, temperature=0.0, options=CHAT_OLLAMA_OPTIONS).content
+            except Exception as exc:  # noqa: BLE001
+                logger.warning("Board update call failed: %s", exc)
+                return None, [f"board update failed: {exc}"]
+            delta = bs.parse_delta(reply)
+            if delta is not None:
+                break
+            messages = messages + [{"role": "assistant", "content": reply or ""},
+                                   {"role": "user", "content": "Return only the JSON object, nothing else."}]
+        if delta is None:
+            return None, ["board update was not valid JSON twice — board unchanged"]
+        warnings = bs.apply_delta(self.session.board, delta, self.papers)
+        return delta, warnings
+
+    def pin(self, text: str, section: str = "established", refs: tuple = ()) -> list[str]:
+        """A pinned takeaway or a typed note onto the board (spec §2H)."""
+        if self.session is None:
+            return ["no session attached"]
+        warnings = bs.add_item(self.session.board, section, text, self.papers, refs=list(refs))
+        self._save()
+        return warnings
+
+    def edit_section(self, section: str, text: str) -> list[str]:
+        if self.session is None:
+            return ["no session attached"]
+        warnings = bs.section_from_edit_text(self.session.board, section, text, self.papers)
+        self._save()
+        return warnings
+
+    def set_idea(self, text: str) -> None:
+        if self.session is None:
+            return
+        self.session.board["idea"] = bs.clip(text, bs.CAPS["idea"])
+        self._save()
+
+    # ── Public API ───────────────────────────────────────────────────────
+
+    def stream_turn(self, user_message: str, mode: str | None = None, move: str = "free", papers: tuple = ()):
+        """Process one turn: plan the move (condense / idea / shortlist / per
+        paper), fit the window, stream the answer, check keys, register the
+        papers cited, update the board, save."""
+        if move not in MOVES:
+            raise ValueError(f"unknown move {move!r}")
         self.turn_count += 1
         active_mode = mode or self.mode
         self.mode = active_mode
+        self.last_move = move
 
-        query = self._condense(user_message)
+        query, results, user_content, shown_message = self._plan_move(user_message, move, tuple(papers))
         self.last_query = query
-
-        results = self._search(query)
         context, sources = format_context(results, CHAT_CONTEXT_MAX_CHARS)
-        if context:
-            user_content = CHAT_TURN_USER.format(question=user_message, context=context)
-        else:
-            user_content = CHAT_TURN_NO_CONTEXT.format(question=user_message)
 
         kept, folded = self._fit(user_content, mode=active_mode)
         messages = [{"role": "system", "content": self._system_prompt(active_mode)}] + kept + [
             {"role": "user", "content": user_content}
         ]
-        num_ctx = CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096)
+        num_ctx = CHAT_WINDOW_TOKENS
         self.last_budget = {
-            "num_ctx": num_ctx,
+            "window": num_ctx,
             "prompt_tokens_est": messages_tokens(messages),
             "reserve": CHAT_ANSWER_RESERVE_TOKENS,
             "context_chars": len(context),
@@ -271,18 +471,26 @@
         cited_docs = {s["document"] for s in sources if s["cited"] and s["document"]}
         self.focus_documents = cited_docs or {s["document"] for s in sources[:2] if s["document"]}
 
-        # Parse follow-up brainstorming suggestions
+        # Parse follow-up brainstorming suggestions, and the position taken
         self.last_suggestions = parse_brainstorm_suggestions(full_answer)
+        self.last_position = parse_position(full_answer)
 
         # Store resolved text in history so future turns reference real paper titles
         resolved_answer = resolve_keys(full_answer, sources)
-        self.history.append({"role": "user", "content": user_message})
+        self.history.append({"role": "user", "content": shown_message})
         self.history.append({"role": "assistant", "content": resolved_answer})
 
+        # The board: S# → session P# for what was cited (every source of a
+        # paper move), then the structured update over the P#-keyed answer.
+        paper_keys = self._register_cited(sources, always=move in PAPER_MOVES)
+        answer_p = bs.rewrite_keys(full_answer, paper_keys)
+        delta, board_warnings = self._update_board(move, shown_message, answer_p)
+        self.last_board_warnings = board_warnings
+
         # Store rich structured turn record for persistent UI rendering & export
         self.turns.append({
             "turn_index": self.turn_count,
-            "user_message": user_message,
+            "user_message": shown_message,
             "condensed_query": query,
             "raw_answer": full_answer,
             "resolved_answer": resolved_answer,
@@ -291,11 +499,79 @@
             "warnings": list(self.last_warnings),
             "suggestions": list(self.last_suggestions),
             "mode": active_mode,
+            "move": move,
+            "position": self.last_position,
+            "paper_keys": paper_keys,
+            "board_delta": delta,
+            "board_warnings": list(board_warnings),
             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         })
+        self._save()
+
+    # ── Going to read, and seeding (spec §2F–G) ──────────────────────────
+
+    def _fold_synthesis(self, shown_message: str, question: str, result: dict, move: str) -> dict:
+        """A research_answer() result becomes a turn: its papers registered,
+        its own P# (local to that synthesis) rewritten to the session's, the
+        board updated over it."""
+        selected = result.get("selected") or []
+        mapping = {}
+        for r in selected:
+            if r.get("key") and r.get("document") and self.session is not None:
+                mapping[r["key"]] = bs.register_paper(self.papers, r["document"], r.get("citation") or "")
+        text = result.get("suggestion") or ""
+        used = set(cited_keys(text) + bs.paper_keys_in(text))
+        answer_p = bs.rewrite_keys(text, mapping)
+        sources = [{
+            "key": mapping.get(r.get("key"), r.get("key")), "document": r.get("document"),
+            "citation": r.get("citation") or r.get("document"), "page": "synthesis", "section": "notes",
+            "chunk_index": None, "cited": r.get("key") in used,
+        } for r in selected]
+        self.turn_count += 1
+        self.last_query = question
+        self.last_sources = sources
+        self.last_warnings = [f"cited [{k}], which is not on the shortlist" for k in result.get("unverified_citations") or []]
+        self.last_suggestions = parse_brainstorm_suggestions(text)
+        self.last_position = parse_position(text)
+        self.last_move = move
+        self.focus_documents = {s["document"] for s in sources if s["cited"] and s["document"]} or self.focus_documents
+        self.history.append({"role": "user", "content": shown_message})
+        self.history.append({"role": "assistant", "content": bs.resolve_paper_keys(answer_p, self.papers)})
+        delta, board_warnings = self._update_board(move, shown_message, answer_p)
+        self.last_board_warnings = board_warnings
+        turn = {
+            "turn_index": self.turn_count, "user_message": shown_message, "condensed_query": question,
+            "raw_answer": answer_p, "resolved_answer": bs.resolve_paper_keys(answer_p, self.papers),
+            "content": answer_p, "sources": sources, "warnings": list(self.last_warnings),
+            "suggestions": list(self.last_suggestions), "mode": self.mode, "move": move,
+            "position": self.last_position, "paper_keys": mapping, "board_delta": delta,
+            "board_warnings": list(board_warnings), "notes": result.get("notes") or [],
+            "timings": result.get("timings") or {}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
+        }
+        self.turns.append(turn)
+        self._save()
+        return turn
+
+    def dive(self, question: str, on_progress=None) -> dict:
+        """Go and read: shortlist, per-paper notes, synthesis — the
+        synthesis-depth machinery — folded into the conversation and the board."""
+        from research_assistant.shared import retrieve
+
+        question = (question or "").strip() or self._idea("")
+        result = retrieve.research_answer(question, mode="map_reduce", on_progress=on_progress)
+        if not result:
+            raise RuntimeError("nothing in the corpus matched that question")
+        return self._fold_synthesis(f"{MOVE_LABELS['dive']}: {question}", question, result, "dive")
+
+    def seed_from_synthesis(self, query: str, answer: dict) -> dict:
+        """Brainstorm this: a Tab 1 synthesis becomes the idea and the first turn."""
+        if self.session is not None and not self.session.board.get("idea"):
+            self.session.board["idea"] = bs.clip(query, bs.CAPS["idea"])
+        return self._fold_synthesis(f"{MOVE_LABELS['seed']}: {query}", query, answer, "seed")
 
     def clear_history(self):
-        """Reset the conversation, memory, and turn records."""
+        """Reset the conversation, memory, and turn records. The board and
+        the papers of an attached session stay — they are the point."""
         self.history.clear()
         self.turns.clear()
         self.memory = ""
@@ -305,7 +581,11 @@
         self.last_warnings = []
         self.last_budget = {}
         self.last_suggestions = []
+        self.last_position = ""
+        self.last_board_warnings = []
+        self.last_move = "free"
         self.turn_count = 0
+        self._save()
 
     def export_conversation(self, scratchpad_notes: list[str] | None = None) -> str:
         """Save the conversation, sources, memory, and pinned notes to markdown."""
@@ -320,6 +600,10 @@
             "",
         ]
 
+        if self.session is not None:
+            lines.extend([f"*Session:* {self.session.title}", "", "## 🧭 The board", "",
+                          self.session.board_markdown(), ""])
+
         if self.memory:
             lines.extend([
                 "## 🧠 Rolling Memory",
@@ -378,6 +662,7 @@
             "scratchpad_notes": scratchpad_notes or [],
             "turns": self.turns,
             "history": self.history,
+            "session": self.session.data if self.session is not None else None,
         }
 
 
```

- [ ] **Step 5: Run the chat tests, then the whole suite**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_agent7_brainstorm.py tests/test_agent7_chat.py tests/test_brainstorm.py
```

Expected: 48 passed

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/ 2>&1 | tail -2
```

Expected: green (620 passed, 1 skipped in the validation, with test_ingestion deselected)

- [ ] **Step 6: Commit**

```bash
git add research_assistant/agents/agent7_research_chat.py tests/test_agent7_brainstorm.py tests/test_agent7_chat.py
git commit -m "feat(chat): sessions on disk, the board in every prompt and maintained by an enforced update, moves, S#→P#, dives and seeding

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Tab 4 — the workspace; Tab 1 — *Brainstorm this*

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_names.py` (unchanged; must pass)

**Interfaces:**
- Consumes: `bs.Session`, `bs.list_sessions`, `bs.delete_session`, `bs.SECTIONS/LABELS/CAPS`, `bs.section_edit_text`, `bs.short_title`; `ResearchChat(session=)`, `.session`, `.pin`, `.edit_section`, `.set_idea`, `.dive`, `.seed_from_synthesis`, `.stream_turn(move=, papers=)`, `MOVE_LABELS`.
- Produces the page. What changes:
  - **Tab 4** — a session bar (`＋ New session` or a session; the selectbox is keyed `brainstorm_session_select`, and `_switch_to()` sets both `brainstorm_session_id` and that key so a programmatic switch sticks); one agent per session in `st.session_state[f"chat_agent:{id}"]`, resumed with `ResearchChat(..., session=bs.Session.load(id))`; rename (popover), `.md` / `.json` downloads, delete (popover). Two columns [3, 2]. **Right**: the board — idea, the four sections with `P#` → short titles, the last update's 📋 warnings, *Edit the board* (idea + one text area per section, *Save*), *Add a note* (text + section), *Papers in this session*. **Left**: the lens, *Reset chat* (turns and memory; the board stays), the memory expander, starters, the turns (with `Move:` in the caption, the 🎯 **Position** box, a dive's notes expander, *Pin to the board*, the sources expander showing `[S#] → P#`, warnings, 📋 board warnings), the suggestion chips, the **Moves** row (Sharpen · What's been done · Argue against · Test it — the last three disabled until the board has an idea), three expanders (*Compare two papers* with a two-paper multiselect, *Dig into a paper*, *Go and read* with a question whose placeholder is the idea — empty means the idea), and the chat input. A pending action is one dict in `st.session_state["chat_pending"]` (`text`, `move`, `papers`) consumed on the next run; a dive runs under `st.status` with the same per-paper progress lines Tab 1 uses. The scratchpad is gone — pins go to the board.
  - **Tab 1** — `_render_brainstorm_this(answer)` under `_render_synthesis`: a button that creates a session named after the query, seeds it (`seed_from_synthesis`), stores the agent, sets `st.session_state["brainstorm_open"]` and reruns; Tab 4 opens on it.

- [ ] **Step 1: Apply the change**

**Apply this change to `app.py`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/app.py
+++ b/app.py
@@ -460,6 +460,33 @@
     t = answer.get("timings")
     if t:
         st.caption(f"{answer.get('mode', '')}: gate {t['gate']}s · notes {t['map']}s · synthesis {t['reduce']}s · total {t['total']}s")
+    _render_brainstorm_this(answer)
+
+
+def _render_brainstorm_this(answer):
+    """One button under a finished synthesis: a new brainstorm session seeded
+    with it (spec 2026-09-13-brainstorm §2G), then Tab 4 opens on it."""
+    query = st.session_state.get("build_query") or ""
+    if not query or not answer.get("suggestion") or answer.get("ungrounded"):
+        return
+    if st.button("🧠 Brainstorm this", key=f"brainstorm_this_{hashlib.sha1(query.encode()).hexdigest()[:8]}",
+                 help="Open a brainstorm session with this synthesis as its first turn and the idea on the board"):
+        from research_assistant.agents.agent7_research_chat import ResearchChat
+        from research_assistant.shared import brainstorm as bs
+
+        session = bs.Session.new(query[:120], origin={"query": query})
+        try:
+            cached_res = get_cached_search_resources(_get_resources_mtime())
+            agent = ResearchChat(top_k=5, search_resources=cached_res, session=session)
+            with st.spinner("Seeding the board…"):
+                agent.seed_from_synthesis(query, answer)
+            st.session_state[f"chat_agent:{session.id}"] = agent
+        except Exception as e:  # noqa: BLE001
+            st.exception(e)
+            return
+        st.session_state["brainstorm_open"] = session.id
+        st.toast("Session created — open Research chat")
+        st.rerun()
 
 
 def _render_seed_citation_audit(final):
@@ -1601,223 +1628,366 @@
 
 with tab_chat:
     st.markdown("### 🔬 Interactive Research Brainstorming Studio")
-    st.caption("Multi-turn research collaboration grounded in your ingested literature. Condenses follow-up queries, retains evidence cards per turn, folds historical context into rolling memory, and suggests next exploration directions.")
+    st.caption("Multi-turn research collaboration grounded in your ingested literature. A board of what we think so far rides in every turn; moves change what he retrieves and what he is asked; sessions persist and resume.")
 
     if chunks == 0:
         st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")
     else:
-        if "chat_agent" not in st.session_state or st.session_state["chat_agent"] is None:
+        from research_assistant.shared import brainstorm as bs
+
+        # ── Sessions (spec §2A): one agent per session id ─────────────────
+        sessions = bs.list_sessions()
+        NEW = "__new__"
+        options = [NEW] + [s["id"] for s in sessions]
+        by_id = {s["id"]: s for s in sessions}
+
+        def _session_label(sid):
+            if sid == NEW:
+                return "＋ New session"
+            s = by_id[sid]
+            return f"{s['title']} · {s['turns']} turns · {(s['updated_at'] or '')[:10]}"
+
+        def _switch_to(session_id):
+            """Programmatic session change: the selectbox is keyed, so its
+            stored value must move too or it wins on the next run."""
+            st.session_state["brainstorm_session_id"] = session_id
+            st.session_state["brainstorm_session_select"] = session_id
+
+        pending_open = st.session_state.pop("brainstorm_open", None)
+        if pending_open and pending_open in by_id:
+            _switch_to(pending_open)
+        current = st.session_state.get("brainstorm_session_id")
+        if current not in by_id:
+            current = sessions[0]["id"] if sessions else NEW
+            _switch_to(current)
+        if st.session_state.get("brainstorm_session_select") not in options:
+            st.session_state["brainstorm_session_select"] = current
+
+        sb1, sb2, sb3, sb4, sb5 = st.columns([4, 1, 1, 1, 1])
+        chosen = sb1.selectbox("Session", options, format_func=_session_label,
+                               key="brainstorm_session_select", label_visibility="collapsed")
+        if chosen != current:
+            st.session_state["brainstorm_session_id"] = chosen
+            st.rerun()
+
+        if chosen == NEW:
+            with st.form("new_brainstorm_form"):
+                title = st.text_input("Name the session", placeholder="e.g. hopping transport in MoS2 networks")
+                if st.form_submit_button("Start", type="primary"):
+                    s = bs.Session.new(title or "Untitled brainstorm")
+                    _switch_to(s.id)
+                    st.rerun()
+            st.stop()
+
+        session_meta = by_id[chosen]
+        if session_meta.get("error"):
+            st.error(f"That session's file cannot be read: `{chosen}`. Pick another or delete it.")
+            if sb5.button("🗑️ Delete", key="delete_bad_session"):
+                bs.delete_session(chosen)
+                st.session_state.pop("brainstorm_session_id", None)
+                st.rerun()
+            st.stop()
+
+        agent_key = f"chat_agent:{chosen}"
+        if agent_key not in st.session_state:
             from research_assistant.agents.agent7_research_chat import ResearchChat
 
             try:
                 cached_res = get_cached_search_resources(_get_resources_mtime())
-                st.session_state["chat_agent"] = ResearchChat(top_k=5, search_resources=cached_res)
-            except Exception as e:
-                st.warning(f"Could not load search index: {e}")
-                st.session_state["chat_agent"] = None
-
-        agent = st.session_state.get("chat_agent")
+                st.session_state[agent_key] = ResearchChat(top_k=5, search_resources=cached_res,
+                                                           session=bs.Session.load(chosen))
+            except Exception as e:  # noqa: BLE001
+                st.warning(f"Could not open the session: {e}")
+                st.session_state[agent_key] = None
+        agent = st.session_state.get(agent_key)
         if agent is None:
-            st.info("Ingest papers to enable research chat.", icon="📭")
-        else:
-            # Session state defaults
-            st.session_state.setdefault("chat_scratchpad", [])
-            st.session_state.setdefault("chat_pending_query", None)
-
-            # Brainstorming modes & definitions
-            LENS_LABELS = {
-                "explore": "🧭 Explore (Broad connections)",
-                "gaps": "💡 Gaps (Literature blind spots)",
-                "contradictions": "⚔️ Contradictions (Competing theories)",
-                "hypotheses": "🧪 Hypotheses (Novel proposals)",
-                "methodology": "🔬 Methodology (Protocols & measurement)",
-            }
+            st.stop()
 
-            STARTERS_BY_MODE = {
-                "explore": [
-                    "What are the central findings and overarching themes across these papers?",
-                    "How is the primary physical phenomenon modelled across the corpus?",
-                    "What are the latest discoveries and their broader implications?",
-                ],
-                "gaps": [
-                    "What are the major open questions or unaddressed gaps across these papers?",
-                    "What experimental conditions or control parameters remain untested?",
-                    "Where do the authors identify the need for future theoretical development?",
-                ],
-                "contradictions": [
-                    "Where do the findings or interpretations in the corpus directly disagree?",
-                    "Compare differing theoretical assumptions made by different authors.",
-                    "Are there conflicting experimental measurements across these studies?",
-                ],
-                "hypotheses": [
-                    "Propose a novel testable hypothesis connecting two or more papers in the corpus.",
-                    "What new experiment could resolve the conflicting findings reported here?",
-                    "How could the existing model be extended to account for unexplained observations?",
-                ],
-                "methodology": [
-                    "Compare the experimental techniques and measurement setups used across papers.",
-                    "What are the sample preparation conditions and measurement limitations?",
-                    "Compare the numerical simulation methods and boundary conditions applied.",
-                ],
-            }
+        with sb2.popover("✏️"):
+            new_title = st.text_input("Title", value=agent.session.title, key=f"rename_{chosen}")
+            if st.button("Save title", key=f"rename_btn_{chosen}"):
+                agent.session.rename(new_title)
+                st.rerun()
+        export_path = agent.export_conversation(scratchpad_notes=None)
+        try:
+            with open(export_path, encoding="utf-8") as _f:
+                export_md_text = _f.read()
+        except Exception:  # noqa: BLE001
+            export_md_text = "# Research Chat Transcript\n"
+        sb3.download_button("📥", data=export_md_text, file_name=f"brainstorm_{chosen}.md", mime="text/markdown",
+                            use_container_width=True, help="The board, then the conversation")
+        sb4.download_button("💾", data=json.dumps(agent.export_conversation_json(), indent=2),
+                            file_name=f"brainstorm_{chosen}.json", mime="application/json", use_container_width=True)
+        with sb5.popover("🗑️"):
+            st.caption("Delete this session and its board.")
+            if st.button("Delete", key=f"delete_{chosen}", type="primary"):
+                bs.delete_session(chosen)
+                st.session_state.pop(agent_key, None)
+                st.session_state.pop("brainstorm_session_id", None)
+                st.session_state.pop("brainstorm_session_select", None)
+                st.rerun()
+
+        # Brainstorming modes & definitions
+        LENS_LABELS = {
+            "explore": "🧭 Explore (Broad connections)",
+            "gaps": "💡 Gaps (Literature blind spots)",
+            "contradictions": "⚔️ Contradictions (Competing theories)",
+            "hypotheses": "🧪 Hypotheses (Novel proposals)",
+            "methodology": "🔬 Methodology (Protocols & measurement)",
+        }
 
-            # Top controls toolbar
-            col_mode, col_clear, col_exp_md, col_exp_json = st.columns([3, 1, 1.2, 1.2])
+        STARTERS_BY_MODE = {
+            "explore": [
+                "What are the central findings and overarching themes across these papers?",
+                "How is the primary physical phenomenon modelled across the corpus?",
+                "What are the latest discoveries and their broader implications?",
+            ],
+            "gaps": [
+                "What are the major open questions or unaddressed gaps across these papers?",
+                "What experimental conditions or control parameters remain untested?",
+                "Where do the authors identify the need for future theoretical development?",
+            ],
+            "contradictions": [
+                "Where do the findings or interpretations in the corpus directly disagree?",
+                "Compare differing theoretical assumptions made by different authors.",
+                "Are there conflicting experimental measurements across these studies?",
+            ],
+            "hypotheses": [
+                "Propose a novel testable hypothesis connecting two or more papers in the corpus.",
+                "What new experiment could resolve the conflicting findings reported here?",
+                "How could the existing model be extended to account for unexplained observations?",
+            ],
+            "methodology": [
+                "Compare the experimental techniques and measurement setups used across papers.",
+                "What are the sample preparation conditions and measurement limitations?",
+                "Compare the numerical simulation methods and boundary conditions applied.",
+            ],
+        }
+
+        st.session_state.setdefault("chat_pending", None)      # {"text", "move", "papers"} to run on this rerun
+
+        col_chat, col_board = st.columns([3, 2])
+
+        # ── Right: the board (spec §2H) ───────────────────────────────────
+        with col_board:
+            board, papers = agent.session.board, agent.session.papers
+            st.markdown("#### 🧭 The board")
+            with st.container(border=True):
+                st.markdown("**Idea**")
+                st.markdown(board.get("idea") or "*not stated yet — say it, or press Sharpen*")
+            for section in bs.SECTIONS:
+                items = board.get(section) or []
+                st.markdown(f"**{bs.LABELS[section]}** · {len(items)}/{bs.CAPS[section]}")
+                if not items:
+                    st.caption("—")
+                for item in items:
+                    refs = ""
+                    if item.get("papers"):
+                        refs = " · " + ", ".join(f"[{k}] {bs.short_title(papers[k]['citation'], 4)}"
+                                                 for k in item["papers"] if k in papers)
+                    st.markdown(f"- {item['text']}<sub>{refs}</sub>", unsafe_allow_html=True)
+            if agent.last_board_warnings:
+                for w in agent.last_board_warnings:
+                    st.warning(w, icon="📋")
+            with st.expander("✏️ Edit the board", expanded=False):
+                new_idea = st.text_area("Idea", value=board.get("idea") or "", key=f"idea_{chosen}", height=80)
+                edits = {}
+                for section in bs.SECTIONS:
+                    edits[section] = st.text_area(f"{bs.LABELS[section]} — one per line, papers as [P1, P3]",
+                                                  value=bs.section_edit_text(board, section),
+                                                  key=f"edit_{section}_{chosen}", height=100)
+                if st.button("Save the board", key=f"save_board_{chosen}", type="primary"):
+                    agent.set_idea(new_idea)
+                    warns = []
+                    for section in bs.SECTIONS:
+                        warns += agent.edit_section(section, edits[section])
+                    for w in warns:
+                        st.warning(w, icon="📋")
+                    st.rerun()
+            with st.expander("➕ Add a note", expanded=False):
+                n1, n2 = st.columns([3, 1])
+                note = n1.text_input("Note", key=f"note_{chosen}", placeholder="e.g. Test temperature dependence of VRH")
+                section = n2.selectbox("Into", list(bs.SECTIONS), format_func=lambda s: bs.LABELS[s], key=f"note_section_{chosen}")
+                if st.button("Add", key=f"add_note_{chosen}") and note.strip():
+                    for w in agent.pin(note, section=section):
+                        st.warning(w, icon="📋")
+                    st.rerun()
+            if papers:
+                with st.expander(f"📚 Papers in this session · {len(papers)}", expanded=False):
+                    for k, rec in papers.items():
+                        st.caption(f"**{k}** — {rec.get('citation') or rec.get('document')}")
+
+        # ── Left: the conversation ────────────────────────────────────────
+        with col_chat:
+            col_mode, col_clear = st.columns([3, 1])
             with col_mode:
+                lens_values = list(LENS_LABELS.values())
                 selected_label = st.selectbox(
-                    "Brainstorming Lens",
-                    options=list(LENS_LABELS.values()),
-                    index=0,
-                    key="chat_lens_selector",
-                    label_visibility="collapsed",
+                    "Brainstorming Lens", options=lens_values,
+                    index=lens_values.index(LENS_LABELS.get(agent.mode, lens_values[0])),
+                    key=f"chat_lens_{chosen}", label_visibility="collapsed",
                     help="Adjust the analytical lens and guidance used to evaluate the papers.",
                 )
                 active_mode = next((k for k, v in LENS_LABELS.items() if v == selected_label), "explore")
                 agent.mode = active_mode
-
             with col_clear:
-                if st.button("🗑️ Reset Chat", use_container_width=True, help="Clear conversation turns and memory"):
+                if st.button("🗑️ Reset chat", use_container_width=True, help="Clear the turns and memory; the board stays"):
                     agent.clear_history()
-                    st.session_state["chat_pending_query"] = None
+                    st.session_state["chat_pending"] = None
                     st.rerun()
 
-            with col_exp_md:
-                export_path = agent.export_conversation(scratchpad_notes=st.session_state.get("chat_scratchpad"))
-                try:
-                    with open(export_path, encoding="utf-8") as _f:
-                        export_md_text = _f.read()
-                except Exception:
-                    export_md_text = "# Research Chat Transcript\n"
-                st.download_button(
-                    "📥 Export (.md)",
-                    data=export_md_text,
-                    file_name=f"research_brainstorm_{int(time.time())}.md",
-                    mime="text/markdown",
-                    use_container_width=True,
-                    help="Download complete brainstorming session with sources and notes",
-                )
-
-            with col_exp_json:
-                export_json_dict = agent.export_conversation_json(scratchpad_notes=st.session_state.get("chat_scratchpad"))
-                st.download_button(
-                    "💾 Export JSON",
-                    data=json.dumps(export_json_dict, indent=2),
-                    file_name=f"research_brainstorm_{int(time.time())}.json",
-                    mime="application/json",
-                    use_container_width=True,
-                    help="Download structured JSON session data",
-                )
-
-            # Interactive Scratchpad & Pinned Ideas Drawer
-            scratchpad = st.session_state.get("chat_scratchpad", [])
-            with st.expander(f"📝 Brainstorm Scratchpad & Pinned Ideas ({len(scratchpad)})", expanded=False):
-                st.caption("Pin important takeaways, hypotheses, or paper quotes as you brainstorm. These are included when exporting your session.")
-                if scratchpad:
-                    for idx, note in enumerate(scratchpad):
-                        sc1, sc2 = st.columns([9, 1])
-                        sc1.markdown(f"- {note}")
-                        if sc2.button("✕", key=f"del_note_{idx}", help="Remove this note"):
-                            st.session_state["chat_scratchpad"].pop(idx)
-                            st.rerun()
-                else:
-                    st.info("No pinned ideas yet. Click '📌 Pin to Scratchpad' on any response below to save it here.")
-
-                new_note = st.text_input("Add a manual research idea or hypothesis:", key="manual_note_input", placeholder="e.g. Test temperature dependence of variable-range hopping...")
-                if st.button("➕ Add Note", key="add_manual_note_btn"):
-                    if new_note.strip():
-                        st.session_state["chat_scratchpad"].append(new_note.strip())
-                        st.rerun()
-
-            # Rolling Memory Display (if memory exists)
             if agent.memory:
                 with st.expander("🧠 Rolling Conversation Memory (Context Summary)", expanded=False):
                     st.info(agent.memory)
 
-            # Quick Starter Prompts (when no turns yet)
-            if not agent.turns and not agent.history:
+            if not agent.turns:
                 with st.container(border=True):
                     st.markdown(f"##### 💡 Quick-start your brainstorming session ({selected_label}):")
                     starters = STARTERS_BY_MODE.get(active_mode, STARTERS_BY_MODE["explore"])
                     scols = st.columns(len(starters))
                     for i, s in enumerate(starters):
                         if scols[i].button(f"✨ {s}", key=f"starter_btn_{i}", use_container_width=True):
-                            st.session_state["chat_pending_query"] = s
+                            st.session_state["chat_pending"] = {"text": s, "move": "free", "papers": ()}
                             st.rerun()
 
-            # Turn-by-Turn History Rendering
-            if agent.turns:
-                for turn in agent.turns:
-                    with st.chat_message("user"):
-                        st.markdown(turn["user_message"])
-                        if turn.get("condensed_query") and turn["condensed_query"] != turn["user_message"]:
-                            st.caption(f"🔍 *Searched literature for:* `{turn['condensed_query']}`")
-
-                    with st.chat_message("assistant"):
-                        mode_tag = turn.get("mode", "explore").capitalize()
-                        st.caption(f"**Lens:** {mode_tag}")
-                        st.markdown(turn["content"])
-
-                        # Pin action
-                        col_pin, col_empty = st.columns([2, 5])
-                        if col_pin.button("📌 Pin to Scratchpad", key=f"pin_btn_{turn['turn_index']}"):
-                            summary_snippet = turn["content"][:250].replace("\n", " ") + ("..." if len(turn["content"]) > 250 else "")
-                            st.session_state["chat_scratchpad"].append(f"[Turn {turn['turn_index']}] {summary_snippet}")
-                            st.toast("Saved to Scratchpad!")
-
-                        # Per-turn Sources Expander
-                        if turn.get("sources"):
-                            with st.expander(f"📚 Sources & Evidence ({len(turn['sources'])}) · Turn {turn['turn_index']}"):
-                                for s in turn["sources"]:
-                                    status_badge = "✅ Cited" if s.get("cited") else "⚪ Referenced"
-                                    score_text = f" · RRF: {s['rrf_score']:.3f}" if s.get("rrf_score") else ""
-                                    st.markdown(f"**[{s['key']}]** `{s.get('document', 'Unknown')}` — {s.get('citation', '')} *(p.{s.get('page', '?')}, {s.get('section', 'text')})* `[{status_badge}{score_text}]`")
-                                    if s.get("text"):
-                                        with st.container(border=True):
-                                            st.caption(f"Snippet: {s['text'][:300]}...")
-
-                        if turn.get("warnings"):
-                            for w in turn["warnings"]:
-                                st.warning(f"⚠️ {w}")
-            else:
-                for msg in agent.history:
-                    with st.chat_message(msg["role"]):
-                        st.markdown(msg["content"])
+            for turn in agent.turns:
+                with st.chat_message("user"):
+                    st.markdown(turn["user_message"])
+                    if turn.get("condensed_query") and turn["condensed_query"] != turn["user_message"] and turn.get("move", "free") == "free":
+                        st.caption(f"🔍 *Searched literature for:* `{turn['condensed_query']}`")
+                with st.chat_message("assistant"):
+                    mode_tag = turn.get("mode", "explore").capitalize()
+                    move_tag = f" · **Move:** {turn['move']}" if turn.get("move", "free") != "free" else ""
+                    st.caption(f"**Lens:** {mode_tag}{move_tag}")
+                    st.markdown(turn["content"])
+                    if turn.get("position"):
+                        st.info(f"**Position:** {turn['position']}", icon="🎯")
+                    if turn.get("notes"):
+                        with st.expander(f"📝 His notes on each paper · {len(turn['notes'])}"):
+                            for n in turn["notes"]:
+                                mark = "📝" if n.get("relevant", True) else "➖"
+                                st.markdown(f"{mark} **{n.get('key')}** — {n.get('citation', '')}: {n.get('notes', '')}")
+                    col_pin, _ = st.columns([2, 5])
+                    if col_pin.button("📌 Pin to the board", key=f"pin_btn_{turn['turn_index']}"):
+                        text = turn.get("position") or turn["content"][:240].replace("\n", " ")
+                        refs = tuple(sorted(set((turn.get("paper_keys") or {}).values())))
+                        for w in agent.pin(text, section="established", refs=refs):
+                            st.warning(w, icon="📋")
+                        st.toast("Pinned to Established")
+                        st.rerun()
+                    if turn.get("sources"):
+                        with st.expander(f"📚 Sources & Evidence ({len(turn['sources'])}) · Turn {turn['turn_index']}"):
+                            pk = turn.get("paper_keys") or {}
+                            for s in turn["sources"]:
+                                status_badge = "✅ Cited" if s.get("cited") else "⚪ Referenced"
+                                session_key = f" → **{pk[s['key']]}**" if s["key"] in pk else ""
+                                st.markdown(f"**[{s['key']}]**{session_key} `{s.get('document', 'Unknown')}` — {s.get('citation', '')} *(p.{s.get('page', '?')}, {s.get('section', 'text')})* `[{status_badge}]`")
+                    for w in turn.get("warnings") or []:
+                        st.warning(f"⚠️ {w}")
+                    for w in turn.get("board_warnings") or []:
+                        st.caption(f"📋 {w}")
 
-            # Clickable Follow-up Suggestion Chips Under Latest Turn
             if agent.last_suggestions and agent.turns:
                 st.markdown("##### 💡 Next Exploration Directions (Click to brainstorm):")
                 sug_cols = st.columns(min(len(agent.last_suggestions), 3))
                 for idx, sug in enumerate(agent.last_suggestions[:3]):
                     if sug_cols[idx].button(f"➡️ {sug}", key=f"sug_btn_{idx}_{agent.turn_count}", use_container_width=True):
-                        st.session_state["chat_pending_query"] = sug
+                        st.session_state["chat_pending"] = {"text": sug, "move": "free", "papers": ()}
                         st.rerun()
 
-            # Chat Input & Processing
-            chat_input_text = st.chat_input("Ask about the literature or brainstorm a hypothesis…")
-            pending_query = st.session_state.pop("chat_pending_query", None)
-            query_to_run = pending_query or chat_input_text
+            # ── The moves (spec §2D, §2F) ─────────────────────────────────
+            st.markdown("###### Moves")
+            m1, m2, m3, m4 = st.columns(4)
+            has_idea = bool(agent.session.board.get("idea"))
+            idea_help = "" if has_idea else " — state the idea first (type it, or Sharpen with your text)"
+            if m1.button("✂️ Sharpen", use_container_width=True, help="Restate the idea as one falsifiable claim" + idea_help):
+                st.session_state["chat_pending"] = {"text": "", "move": "sharpen", "papers": ()}
+                st.rerun()
+            if m2.button("📚 What's been done", use_container_width=True, help="One line per shortlisted paper on what it did relative to the idea", disabled=not has_idea):
+                st.session_state["chat_pending"] = {"text": "", "move": "prior", "papers": ()}
+                st.rerun()
+            if m3.button("⚔️ Argue against", use_container_width=True, help="The strongest objection the sources support", disabled=not has_idea):
+                st.session_state["chat_pending"] = {"text": "", "move": "against", "papers": ()}
+                st.rerun()
+            if m4.button("🧪 Test it", use_container_width=True, help="The measurement or simulation that would decide it", disabled=not has_idea):
+                st.session_state["chat_pending"] = {"text": "", "move": "test", "papers": ()}
+                st.rerun()
+            paper_keys = list(agent.session.papers)
+            p1, p2, p3 = st.columns(3)
+            with p1.expander("🔀 Compare two papers", expanded=False):
+                if len(paper_keys) < 2:
+                    st.caption("Needs two papers in this session — cite a second one first.")
+                else:
+                    pair = st.multiselect("Two papers", paper_keys, max_selections=2, key=f"compare_{chosen}",
+                                          format_func=lambda k: f"{k} {bs.short_title(agent.session.papers[k]['citation'], 5)}")
+                    steer = st.text_input("Steer (optional)", key=f"compare_steer_{chosen}")
+                    if st.button("Compare them", key=f"compare_go_{chosen}", disabled=len(pair) != 2):
+                        st.session_state["chat_pending"] = {"text": steer, "move": "compare", "papers": tuple(pair)}
+                        st.rerun()
+            with p2.expander("🔎 Dig into a paper", expanded=False):
+                if not paper_keys:
+                    st.caption("Needs a paper in this session — ask something first.")
+                else:
+                    one = st.selectbox("Paper", paper_keys, key=f"dig_{chosen}",
+                                       format_func=lambda k: f"{k} {bs.short_title(agent.session.papers[k]['citation'], 5)}")
+                    steer = st.text_input("Steer (optional)", key=f"dig_steer_{chosen}")
+                    if st.button("Dig in", key=f"dig_go_{chosen}"):
+                        st.session_state["chat_pending"] = {"text": steer, "move": "dig", "papers": (one,)}
+                        st.rerun()
+            with p3.expander("📖 Go and read", expanded=False):
+                st.caption("He shortlists papers, reads each, and writes a synthesis — folded into the board. "
+                           "Leave the question empty to read on the idea itself.")
+                dq = st.text_input("Question", key=f"dive_q_{chosen}",
+                                   placeholder=(agent.session.board.get("idea") or "a question for the corpus")[:80])
+                if st.button("Go", key=f"dive_go_{chosen}", type="primary", disabled=not (dq.strip() or has_idea)):
+                    st.session_state["chat_pending"] = {"text": dq.strip(), "move": "dive", "papers": ()}
+                    st.rerun()
 
-            if query_to_run:
+            # ── Input & processing ────────────────────────────────────────
+            chat_input_text = st.chat_input("Ask about the literature or brainstorm a hypothesis…")
+            pending = st.session_state.pop("chat_pending", None)
+            if chat_input_text and not pending:
+                pending = {"text": chat_input_text, "move": "free", "papers": ()}
+
+            if pending:
+                move, text, picked = pending["move"], pending["text"], tuple(pending.get("papers") or ())
+                from research_assistant.agents.agent7_research_chat import MOVE_LABELS
+
+                if move == "dive" and not text:
+                    text = agent.session.board.get("idea") or ""
+                shown = text if move == "free" else (MOVE_LABELS[move] + (f" {' vs '.join(picked)}" if picked else "") + (f": {text}" if text else ""))
                 with st.chat_message("user"):
-                    st.markdown(query_to_run)
+                    st.markdown(shown)
                 with st.chat_message("assistant"):
                     try:
-                        with st.spinner("Searching literature, condensing context, and formulating response…"):
-                            stream = agent.stream_turn(query_to_run, mode=active_mode)
-                            first_chunk = next(stream, None)
-                        if first_chunk is not None:
-                            def _generator():
-                                yield first_chunk
-                                yield from stream
-                            st.write_stream(_generator())
+                        if move == "dive":
+                            with st.status("Going to read…", expanded=True) as status:
+                                def _progress(stage, payload):
+                                    if stage == "shortlist":
+                                        st.write("📚 Reading " + ", ".join(f"{p['key']} {p['citation'][:50]}" for p in payload["papers"]))
+                                    elif stage == "notes":
+                                        mark = "📝" if payload.get("relevant", True) else "➖"
+                                        st.write(f"{mark} {payload['key']} · {payload['citation'][:60]} ({payload.get('seconds', '?')}s)")
+                                    elif stage == "synthesis":
+                                        st.write(f"🧠 Synthesis written ({payload.get('seconds', '?')}s)")
+                                agent.dive(text, on_progress=_progress)
+                                status.update(label="Read and folded into the board", state="complete")
                         else:
-                            st.info("No response generated.")
+                            with st.spinner("Searching literature, condensing context, and formulating response…"):
+                                stream = agent.stream_turn(text, mode=active_mode, move=move, papers=picked)
+                                first_chunk = next(stream, None)
+                            if first_chunk is not None:
+                                def _generator():
+                                    yield first_chunk
+                                    yield from stream
+                                st.write_stream(_generator())
+                            else:
+                                st.info("No response generated.")
                     except Exception as e:  # noqa: BLE001
                         st.exception(e)
                 st.rerun()
 
 
-
 # ─── Tab 5: how to use ─────────────────────────────────────────────────────
 
 with tab_help:
```

- [ ] **Step 2: Guards**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_app_names.py && CITATION_LOG_FILE=0 python -c "import ast; ast.parse(open('app.py').read()); print('ok')"
```

Expected: 1 passed; ok

- [ ] **Step 3: Start the page and click through Tab 4 with no session** — the `＋ New session` form renders; create one; the two columns render with an empty board; *What's been done* / *Argue against* / *Test it* are disabled until the board has an idea; the three expanders open.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): the brainstorm workspace — sessions, the board beside the chat, moves, Go and read, Brainstorm this

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Docs

**Files:**
- Modify: `HOW_TO_USE.md` (Tab 4 section), `PIPELINE.md` (Agent 7 row, layout tree, `data/brainstorms/`), `ARCHITECTURE.md` (new §2.4)

- [ ] **Step 1: Apply the three diffs**

**Apply this change to `HOW_TO_USE.md`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/HOW_TO_USE.md
+++ b/HOW_TO_USE.md
@@ -105,14 +105,26 @@
 
 ---
 
-### Tab 4 — Research chat (Conversational Exploration)
+### Tab 4 — Research chat (Brainstorming Studio)
 
-Use this tab to conduct multi-turn literature question-answering grounded in your vector library.
+A brainstorm is a **session**: named, saved on disk, listed newest first, resumable after a refresh or a restart. Pick one in the bar at the top, or start a new one; rename, export (`.md` — the board first, then the conversation; or `.json`) and delete are beside it.
 
-1. Type any research question into the chat input.
-2. The response streams token-by-token in real time.
-3. Under each response, expand **Sources** to view the exact text passages that informed the answer.
-4. Export the conversation to Markdown or clear the history at any time.
+**Left — the conversation.**
+1. Pick a **lens** (explore, gaps, contradictions, hypotheses, methodology) — it sets the tone.
+2. Type a question, or press a starter. Follow-ups are condensed into standalone searches (shown as *Searched literature for*); the papers cited last turn get their own search.
+3. Every answer cites its sources as `[S#]`, ends with a **Position** — a sharper idea, an objection with its paper, or the one question to settle — and 2–3 suggested next questions you can click.
+4. **Moves** change what he retrieves and what he is asked:
+   - **Sharpen** — the idea as one falsifiable claim, with its assumptions and which the sources support.
+   - **What's been done** — one line per shortlisted paper on what it did relative to the idea (from the paper summaries).
+   - **Argue against** — the strongest objection the sources support, with the paper.
+   - **Test it** — the measurement or simulation that would decide it.
+   - **Compare two papers** / **Dig into a paper** — pick from the papers this session has cited.
+   - **Go and read** — he shortlists papers, reads each (per-paper notes), writes a synthesis, and folds it into the board. About 10 s on Gemini; minutes on a CPU-only Ollama.
+5. **Pin to the board** on any answer puts its Position (or its first lines) under *Established*, with the papers it cited.
+
+**Right — the board.** What we think so far: the **idea** as it currently stands, **established** points, **objections**, **open questions**, **experiments** — each item with the papers behind it (`P#`, stable within the session). The board rides in every turn's prompt; after each turn one structured call proposes changes, which the code checks (unknown papers, unknown items and anything over a section's cap are dropped and shown as 📋 warnings). Edit any section by hand under **Edit the board** (one item per line, `[P1, P3]` at the end keeps the papers); **Add a note** puts a line of yours into a section.
+
+**Brainstorm this**, under a finished Tab 1 synthesis, starts a session with that synthesis as its first turn and the idea on the board.
 
 ---
 
```

**Apply this change to `PIPELINE.md`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/PIPELINE.md
+++ b/PIPELINE.md
@@ -46,7 +46,7 @@
 
 | **Agent 5 — Batch citer** | [agent5_batch_citer.py](research_assistant/agents/agent5_batch_citer.py) | The same job as Agent 4, over a whole draft file at once. Splits the text into sentences, batches an LLM citation-need check across them, then cites every sentence that needs it and the corpus supports. Writes the cited draft, a `_citations.json` key → source mapping (BibTeX input), and a `_report.md` explaining every decision sentence by sentence. Aborts and writes nothing if a batched verdict list cannot be aligned back to its input sentences, rather than guessing which verdict belongs to which sentence. |
 | **Agent 6 — Manual ingestor** | [agent6_manual_ingestor.py](research_assistant/agents/agent6_manual_ingestor.py) | Watches `data/pulled_pdfs/` for PDFs dropped in by hand — papers Agent 2 couldn't find automatically (e.g. paywalled, so downloaded manually) — and ingests each one through the same pipeline as Agent 3, with the same already-ingested check. `--once <file>` ingests a single PDF immediately instead of watching. |
-| **Agent 7 — Research chat** | [agent7_research_chat.py](research_assistant/agents/agent7_research_chat.py) | A multi-turn conversational agent over the ingested corpus, with `/sources` and `/export` (to a timestamped markdown file) commands. Streams its answers token-by-token via `shared.llm.chat_stream()` instead of returning one block. |
+| **Agent 7 — Research chat** | [agent7_research_chat.py](research_assistant/agents/agent7_research_chat.py) | A multi-turn brainstorming partner over the ingested corpus. Follow-ups are condensed into standalone searches; the papers cited last turn get a focus search; sources are keyed `[S#]` and checked; older turns fold into a rolling memory; lenses set the tone. With a **session** attached (`shared/brainstorm.py`, `data/brainstorms/<id>/session.json`) the **board** — idea, established, objections, open questions, experiments, each with stable paper keys `P#` — rides in every prompt and is maintained by one structured call after each turn, enforced in code. **Moves** (`stream_turn(..., move=)`: sharpen, prior, against, test, compare, dig) change the retrieval and the question; `dive()` runs the map-reduce synthesis and folds it in. Every answer ends with a `Position:` line. Streams via `shared.llm.chat_stream()`. |
 | **Agent 8 — Verifier** | [agent8_verifier.py](research_assistant/agents/agent8_verifier.py) | Audits an already-cited draft. For every `\cite{key}` Agent 5 inserted, it re-retrieves the best-matching chunk from that source and asks a claim–evidence rubric whether the evidence actually supports the claim — decomposing the claim into `finding` / `scope` / `strength`, verdicting each, and aggregating into `Supports` / `Partially supports` / `Contradicts` / `Does not support` / `Unclear`. It reports rather than gates: a wrong citation stays in the draft and is flagged, with a slot table and a verbatim supporting span. Separately rates `evidence_sufficiency`, which distinguishes "the paper does not support this" from "retrieval did not return enough to tell". Writes `_verification.json` and `_verification.md`. Claims keep their sentence boundary (Agent 5 restores the full stop after `\cite{}`); the aggregate is derived from the slots in code and disagreement is recorded; the supporting span is verified verbatim; the top hit is judged with its neighbours and, when the verdict says it saw too little, re-judged once on the top 3. |
 
 Agent 0 just leaves a PDF in `data/raw/`, which is exactly what Agent 1 already
@@ -137,6 +137,8 @@
 │       ├── figures.py      figure cropping, context formatting and VLM description
 │       ├── ingest_v2.py    PDF → corpus entries: extract, chunk, caption, crop, describe
 │       ├── batch_uploader.py  multi-PDF / ZIP upload → direct ingestion (UI tab 1)
+│       ├── brainstorm.py   the brainstorm board and session: deltas enforced in code, papers P#, session.json
+│       ├── chat_context.py the mechanics of a chat turn: budget, keys, context blocks, position, suggestions
 │       ├── grobid_manager.py  GROBID lifecycle: start / stop / health (Docker, JAR or remote)
 │       ├── manifest.py     what has been ingested (data/ingested.json)
 │       ├── search.py       hybrid BM25 + dense with Reciprocal Rank Fusion
@@ -152,6 +154,7 @@
     ├── raw/                        PDFs to process; raw/grobid_output/ caches GROBID's TEI
     ├── pulled_pdfs/                PDFs from Agent 2, or dropped by hand for Agent 6
     ├── drafts/                     drop a .txt here and watch.py runs Agent 5 on it
+    ├── brainstorms/                one directory per brainstorm session (session.json: board, papers, turns)
     ├── images/                     figure/table crops from ingestion (debug artefact)
     ├── logs/
     ├── physics_vectordb/           persistent ChromaDB store
```

**Apply this change to `ARCHITECTURE.md`** (a unified diff against the file at the base commit; `git apply` it from a temp file, or apply hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/ARCHITECTURE.md
+++ b/ARCHITECTURE.md
@@ -147,6 +147,45 @@
 
 ---
 
+### 2.4 The brainstorm board: the model's working memory, enforced in code
+
+Agent 7's studio keeps the thread (condensed follow-ups, focus search, a
+rolling memory, keyed sources). A brainstorm needs more than a thread: a
+place where what we *think* accumulates and that the model sees every turn.
+That is the board (`shared/brainstorm.py`): the idea as it stands and four
+capped sections — established, objections, open questions, experiments —
+each item citing papers by a session-stable key `P#`. The board is
+rendered into the system prompt after the lens and before the memory.
+
+**Why a structured call, not the answer.** After each turn one call asks
+the model for a JSON delta (`idea`, `add`, `replace`, `remove`) over the
+board it was shown and the answer it gave with `P#` keys. The reply is
+parsed (bare or fenced JSON, retried once with a nudge) and *validated*:
+unknown sections, ids and paper keys are dropped, texts clipped, adds past
+a cap refused — each a warning on the turn. A model's format discipline is
+never trusted, and the board never changes silently. Putting the delta
+into the answer itself would pollute the stream and lose the update on any
+parse failure; regenerating the board on demand from the whole
+conversation would leave the model without it during the turn.
+
+**Why sessions on disk.** The board is what you take away; it must outlive
+the browser session. `data/brainstorms/<id>/session.json` holds the board,
+the paper registry and the studio's turns; the agent attaches to it and
+saves after every turn; the page keeps one agent per session id.
+
+**Moves.** `stream_turn(..., move=)` keeps one code path and changes two
+things per move: the retrieval (the idea; the stage-1 summary shortlist;
+per-paper restricted searches) and the user template. `dive()` is the
+synthesis-depth `research_answer(mode="map_reduce")` folded into a turn —
+the synthesis's own `P#` keys, local to it, are rewritten to the session's
+in one pass.
+
+**The window.** `_fit` budgets against `CHAT_WINDOW_TOKENS` — Ollama's
+`num_ctx`, or 32k on an OpenAI-compatible backend — rather than `num_ctx`
+unconditionally, so a hosted model does not fold history after two turns.
+
+---
+
 ## 3. Discovery (Agent 0)
 
 ### 3.1 Provider order: arXiv → OpenAlex → Semantic Scholar
```

- [ ] **Step 2: Commit**

```bash
git add HOW_TO_USE.md PIPELINE.md ARCHITECTURE.md
git commit -m "docs: the brainstorm workspace — sessions, the board, moves, going to read

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The live check (spec §6)

No code. Run the page against the v2 index (`CITATION_INDEX_VERSION=2`; with `GEMINI_API_KEY` in `.env` the chat is Gemini and each turn is seconds; on Ollama each is a minute or two). Record the outcome in `notimportant/superpowers/plans/2026-09-13-brainstorm-workspace-verification.md`.

If you would rather not write into the real `data/`, copy `data/physics_vectordb`, `data/bm25_index_v2.pkl`, `data/ingested_v2.json`, `data/downloaded.json`, `data/seed_papers.json` and `.env` into a scratch directory and set `CITATION_DATA_DIR` to it — that is how this plan was validated.

- [ ] **Gate 1 — a session and a free turn.** Tab 4 → `＋ New session`, name it, *Start* → the bar shows it. Ask a specific question. Expected: the answer streams with `[S#]` keys; a 🎯 **Position** box under it; suggestion chips; the sources expander shows `[S1] → P1 …`; the right column's board now has an idea and one or more *Established* items with `[P1] <title>`; `data/brainstorms/<id>/session.json` exists with one turn whose `board_delta` is the JSON applied.

- [ ] **Gate 2 — the moves.** *Argue against* → an assistant turn captioned `Move: against`, an *Objections* item with its paper. *What's been done* → sources are summaries (`p.summary`), several papers register under *Papers in this session*, and their keys are stable across turns (a paper seen before keeps its `P#`). *Sharpen* → the idea changes. *Test it* → an *Experiments* item. When a section is at its cap the turn shows a 📋 warning and the board does not overflow. *Compare two papers* and *Dig into a paper* → `doc_filter` searches (the sources are all from the chosen papers).

- [ ] **Gate 3 — the board by hand.** *Edit the board* → change a line, add one, *Save* → the list updates and the ids of unchanged lines are preserved in `session.json`. *Add a note* into *Open questions*. *Pin to the board* on a turn → its position lands in *Established* with the turn's papers.

- [ ] **Gate 4 — Go and read.** With the question empty (→ the idea). Expected: the status box lists the shortlist and one line per paper as its notes land, then the synthesis; a new assistant turn captioned `Move: dive` whose keys are session `P#` (no key that also names a different paper in *Papers in this session*), with *His notes on each paper*; the board gains items citing those papers.

- [ ] **Gate 5 — resume.** Reload the browser mid-session and, separately, restart the server. Expected: the session bar lists the session with its turn count; selecting it shows every turn and the board; a new turn continues the thread (the follow-up is condensed against the restored history).

- [ ] **Gate 6 — Brainstorm this.** Tab 1: any run that ends in a synthesis → the button under it → Tab 4 opens on a new session named after the query, with the synthesis as its first turn (`Move: seed`) and the idea on the board.

- [ ] **Gate 7 — export.** `.md` starts with *Session:* and the board, then the conversation with `[P#]` resolved to titles; `.json` carries `session`.

- [ ] **Record** the gates, the session id, and what was seen; commit the record.

```bash
git add notimportant/superpowers/plans/2026-09-13-brainstorm-workspace-verification.md
git commit -m "chore(chat): brainstorm workspace — live check on the v2 index

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** §2A sessions → Task 1 (`Session`, `list_sessions`, `delete_session`) + Task 3 (`attach`, `_save`) + Task 4 (the bar, one agent per id). §2B the board → Task 1 (sections, caps, ids, `board_text`) + Task 3 (`_system_prompt`). §2C the update enforced in code → Task 1 (`parse_delta`, `apply_delta`) + Task 3 (`_update_board`: retry once, off switch, warnings on the turn) + Task 2 (`CHAT_BOARD_UPDATE_USER`). §2D moves → Task 2 (`CHAT_MOVE_USER`) + Task 3 (`_plan_move`, `_search_summaries`, `_search_papers`) + Task 4 (buttons, pickers). §2E the position → Task 2 (prompt + `parse_position`) + Task 3 (`last_position`, turn record) + Task 4 (the box, the pin). §2F the dive → Task 3 (`dive`, `_fold_synthesis`, key rewriting) + Task 4 (the status box). §2G seeding → Task 3 (`seed_from_synthesis`) + Task 4 (`_render_brainstorm_this`, `brainstorm_open`). §2H Tab 4 → Task 4. §2I the window → Task 1 (`CHAT_WINDOW_TOKENS`) + Task 3 (`_fit`). §3 honoured — nothing of the studio removed; the scratchpad's job moves to the board. §4 config → Task 1. §5 errors → `apply_delta` warnings, the retry, `list_sessions` flagging unreadable files, `dive` raising with no turn added, the disabled buttons. §6 → the unit tests per task and Task 6.

**Placeholder scan.** Every step carries its code or its diff and its expected result.

**Type consistency.** `bs.Session` / `.board` / `.papers` / `.board_text()` / `.board_markdown()` used by the agent (Task 3) and the page (Task 4) are defined in Task 1. `stream_turn(user_message, mode, move, papers)` and `MOVE_LABELS` are what the page's `chat_pending` handler calls. `parse_position` (Task 2) is what the agent stores as `position` and the page shows. The turn keys the page reads — `move`, `position`, `paper_keys`, `board_warnings`, `notes`, `sources[].key/cited/page/section` — are the ones `stream_turn` and `_fold_synthesis` write. `retrieve.research_answer`'s result keys the agent reads (`suggestion`, `selected[].key/document/citation`, `unverified_citations`, `notes`, `timings`) are the ones the base commit returns.
