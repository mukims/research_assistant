# Brainstorm workspace — a delta on the studio: a board the model sees, moves, and going to read

*2026-09-13 · base `6b9ca58` (branch `synthesis-depth`)*

## 1. What exists, and what is missing

Commit `6b9ca58` turned Agent 7 into a "brainstorming studio". It already
does what the chat-depth plan asked for and more: follow-ups condensed into
standalone queries; global retrieval capped per paper plus a focus search
on the papers cited last turn; a token budget with older turns folded into
a rolling memory; keyed `[S#]` sources checked and resolved to titles;
five **lenses** (explore, gaps, contradictions, hypotheses, methodology) as
a system-prompt addendum; every answer ending in a `### 💡 Suggested Next
Questions` list parsed into clickable chips; starter prompts; per-turn
evidence cards; a scratchpad of pinned snippets; md/json export; tests.

Two things landed with it that change the ground: the **Gemini backend**
(`b6ebf7b`) — with `GEMINI_API_KEY` set, chat calls take seconds and the
real window is far larger than the 4,096 `_fit` still budgets against —
and cooperative **cancellation** in `pipeline_status`.

Against what the operator asked for on 2026-09-13, the studio lacks:

| asked for | studio | missing |
|---|---|---|
| he takes a position | suggests 2–3 *questions*; lenses set a focus | never commits to a sharper idea or an objection with its paper |
| an idea board | pinned snippets in `st.session_state`, invisible to the model, lost on refresh | structured, persistent, *in the prompt* |
| named moves | lenses (tone), starters, chips | idea-centric moves (sharpen, test it), paper-centric (compare, dig in), *what's been done* over the shortlist |
| he goes and reads | — | a dive: `research_answer(mode="map_reduce", on_progress=)` exists and is ~30 s on Gemini |
| sessions | one in-memory conversation; export | named, listed, resumable; seeded from a Tab 1 run |

This spec is the delta. `ResearchChat`, its prompts, lenses, chips,
condense, focus, memory and tests stay; everything below is added to them.

## 2. Decisions

### A. Sessions on disk

`data/brainstorms/<session_id>/session.json`:

```
{id, title, created_at, updated_at, origin: null | {"query": …},
 mode, memory, focus_documents[],
 board: {idea, established[], objections[], open[], experiments[], next_id},
 papers: {"P1": {document, citation}, …},
 turns: [ …the studio's turn dicts, plus paper_keys, board_delta, board_warnings ]}
```

`research_assistant/shared/brainstorm.py` owns it: `Session.new(title,
origin)`, `Session.load(id)`, `session.save()` (atomic),
`list_sessions()` (newest first, unreadable ones flagged), `delete_session(id)`,
`session.rename(title)`. `ResearchChat.attach(session)` restores `turns`,
`memory`, `mode`, `focus_documents` and rebuilds `history` from the turns
(`user_message` / `resolved_answer`); after every turn the agent saves. The
page keeps one agent per session id. The scratchpad becomes the board.

### B. The board

Five sections, small by construction:

| section | item | cap |
|---|---|---|
| `idea` | one paragraph — the idea as it currently stands | 600 chars |
| `established` | `{id: "E3", text, papers: ["P1", "P4"]}` | 8 |
| `objections` | `{id: "O2", text, papers}` | 5 |
| `open` | `{id: "Q1", text}` | 5 |
| `experiments` | `{id: "X2", text, papers}` | 5 |

Item texts ≤ 240 chars; ids `E/O/Q/X` + a counter never reused, so a delta
can replace or remove by id. `papers` is a registry of every paper cited in
the session with stable keys `P1…`, assigned by document the first time a
cited source appears; board items cite `P#`, never a turn's `S#`.

`board_text(board, papers)` renders it for the prompt (IDEA / ESTABLISHED /
OBJECTIONS / OPEN QUESTIONS / EXPERIMENTS / PAPERS with short titles). When
a session is attached, `_system_prompt` appends `CHAT_BOARD_BLOCK` with it,
after the lens and before the memory. The board is what we think; the
memory is what was said.

### C. The board is maintained by a structured call, enforced in code

After each turn (and after a dive, a seed, a pin), one call with
`CHAT_BOARD_UPDATE_USER` — the board text, the move, the question, the
answer with `[P#]` keys — must return JSON:
`{"idea": str|null, "add": {section: [item…]}, "replace": {id: item},
"remove": [id…]}`. `parse_delta` accepts a bare or fenced object;
`apply_delta` enforces the schema: unknown sections/ids dropped, `P#` not in
the registry dropped from the item, texts clipped, adds beyond a cap
dropped — each a warning on the turn. An unparseable reply is retried once
with a "return only the JSON object" nudge; a second failure leaves the
board unchanged, with the warning. `CITATION_CHAT_BOARD_UPDATE=0` turns
the call off (the board is then only what you pin and edit).

Mapping: every `[S#]` the answer cites is registered (`register_paper` by
document) and the answer the update call sees has `[P#]` keys
(`turn["paper_keys"]` records `S# → P#`). History keeps the studio's
title-resolved text.

### D. Moves

`stream_turn(user_message, mode=None, move="free", papers=())`. Lenses keep
setting the tone; a move changes the retrieval and the question put to the
model:

| move | button | retrieval | user template says |
|---|---|---|---|
| `free` | (typed) | condensed query + focus | the studio's `CHAT_TURN_USER` |
| `sharpen` | Sharpen | the board's idea | restate the idea as one falsifiable claim — system, quantity, condition, expected sign or magnitude; the assumptions it rests on; which the sources support |
| `prior` | What's been done | stage 1: `rank_documents(idea)` → `gate_documents` → the shortlisted papers' summaries as the sources | for each paper one line on what it did relative to the idea; which is closest; what none of them did |
| `against` | Argue against | the idea | the strongest objection the sources support, naming the paper; what would have to be wrong for the idea to survive |
| `test` | Test it | the idea | the measurement or simulation that would decide it: observable, expected signature, the closest method among the sources, what counts as a negative result |
| `compare` | Compare | 4 chunks per chosen paper (`doc_filter`) on the idea | where the two agree, where they differ, which is closer to the idea and why |
| `dig` | Dig in | 4 chunks of the chosen paper on the idea | what it establishes, its method, its limits relative to the idea |

When the board has no idea yet, `sharpen`/`against`/`test`/`prior` use the
typed text (the button then prompts for it). The typed text, when given
with a move, is appended as the researcher's steer.

### E. He takes a position

`RESEARCH_CHAT_SYSTEM` gains one requirement, placed before the suggested
questions: **every reply ends its analysis with one line beginning
`Position:`** — a sharper statement of the idea, or the strongest objection
with its paper, or the single question to settle next. The suggestions
block stays (the chips depend on it). `chat_context.parse_position(answer)`
returns that line; the page shows it as the turn's caption and the board
update sees it.

### F. He goes and reads

`ResearchChat.dive(question, on_progress=None) -> dict` calls
`retrieve.research_answer(question, mode="map_reduce", on_progress=…)`,
registers the result's `selected` papers, rewrites its `[P#]` keys — which
are local to that synthesis — to the session's keys, adds a user turn (*Go
and read: …*) and an assistant turn (the synthesis text; sources = the
shortlist with `cited` marks; `move="dive"`), then runs the board update
over it. On the page the dive runs inline under `st.status` with the same
per-paper progress lines Tab 1 shows; on Gemini ~30 s, on Ollama minutes.

### G. Seeding from Tab 1

Under a finished Tab 1 synthesis, **Brainstorm this**: `Session.new(title=query,
origin={"query": query})`, `agent.seed_from_synthesis(query, answer)` — idea
= the query, the synthesis as the first assistant turn (its keys rewritten
to session keys, its papers registered), a board update over it — then
Tab 4 opens on that session.

### H. Tab 4

A session bar (selectbox of sessions newest first, *New*, *Rename*,
*Export*, *Delete*), then two columns [3, 2]:

- **Left — the conversation**, as the studio has it (lens, starters, turns
  with evidence cards, chips, input), plus the move buttons row (*Sharpen ·
  What's been done · Argue against · Test it · Compare · Dig in · Go and
  read*), pickers over the registry for *Compare* (two) and *Dig in*
  (one), a question box for *Go and read*, and each turn's `Position:` as
  its caption. *Pin* on a turn adds its position (or first 240 chars) to
  `established` with the turn's papers.
- **Right — the board**: the idea in a bordered container; each section as
  a list with `P#` → short titles; an *Edit* expander with a text area per
  section (one item per line, `[P1, P3]` at the end keeps the papers) and
  *Save*; a note box with a section selector; the papers registry.

Export (md) starts with the board; the json export carries the session.

### I. The window knows the backend

`CHAT_WINDOW_TOKENS = _env_int("CITATION_CHAT_WINDOW_TOKENS", 32768 if
LLM_BACKEND == "openai" else CHAT_OLLAMA_OPTIONS["num_ctx"])`; `_fit`
budgets against it. On Ollama nothing changes; on Gemini the memory stops
folding after two turns.

## 3. Not changed

Condense, focus, memory, keys, lenses, chips, starters, the CLI REPL,
`hybrid_search`, `research_answer`, Tabs 1–3 beyond one button, the theme.
No new dependencies.

## 4. Configuration

| variable | default | meaning |
|---|---|---|
| `BRAINSTORMS_DIR` (code) | `data/brainstorms` | sessions |
| `CITATION_CHAT_WINDOW_TOKENS` | 32768 on `openai`, else `num_ctx` | the budget `_fit` uses |
| `CITATION_CHAT_BOARD_UPDATE` | `1` | the structured call after each turn |
| `CITATION_CHAT_MOVE_PER_PAPER_CHUNKS` | `4` | chunks per paper for compare / dig |

## 5. Errors

- Delta unparseable twice → board unchanged, warning. Unknown `P#`, section
  or id; a section at its cap → dropped with a warning.
- Session file unreadable → listed as unreadable; others fine.
- Stream raises mid-answer → the studio's behaviour (the exception shows);
  no board update.
- Dive raises → shown; no turns added.
- Compare with fewer than two papers in the registry, dig with none → the
  button is disabled.

## 6. Verification

Unit, without a model: the board (render, ids, caps, registry,
`parse_delta` on fenced / bare / broken JSON, `apply_delta` on every
rejection path, edit round trip, markdown); the session (new / save / load /
list / rename / delete); the agent with the studio's fakes (each move's
query and template, the board in the system prompt, `S# → P#` mapping, the
update applied and its warnings recorded, retry, off switch, attach /
restore, pin, dive key rewriting, seed); `parse_position`; the app-names
guard. Live: one session on the v2 index — free, sharpen, what's been done,
argue against, test it, compare, go and read — reading the board after each,
a refresh mid-session resuming it, and the export.
