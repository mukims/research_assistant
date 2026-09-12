# Research chat depth — remember the thread, retrieve for the follow-up, fit the window

*2026-09-12*

## 1. Purpose

Research chat (Agent 7, Tab 4) is the brainstorming partner Marvin was
started for. Its depth is limited by three things, all shown in one
three-turn run on the v2 index on 2026-09-12:

1. **Retrieval on the raw follow-up.** Turn 1 *"What does the covalent MoS₂
   network paper say about hopping transport?"* retrieved five chunks of that
   paper. Turn 2 *"What about its limitations?"* retrieved from four
   unrelated papers (random-matrix theory, a 1997 Physics Reports…) — the
   MoS₂ paper was not among them — and the model, correctly, said it could
   not answer. The question is embedded as typed; "its" has no referent
   outside the conversation.
2. **The window fills and then truncates silently.** Prompt size grew 1,977
   → 2,273 → 2,789 estimated tokens over three turns, with ~500 tokens of
   answer still to generate, in a `num_ctx` of 4,096. Two more turns and
   Ollama drops tokens from the front of the prompt — the system prompt goes
   first, then the earliest turns — with nothing in the UI to say so. The
   agent's only defence is "keep the last 10 messages", which is not a
   token budget.
3. **Five of five chunks from one paper.** Right when the question is about
   that paper; a keyhole when the question is "what has been done on X".
   And, as with the synthesis, no temperature is passed: the answer samples
   at the model's default 1.0.

Depth in a conversation is continuity: the follow-up must be understood in
context, the context must fit, and what has been established must survive
the window. The model (`gemma4:e2b`, 5.1 B, CPU) and the window (4,096 —
`GEMINI.md`'s rule for the VM) are fixed; the design works inside them and
adapts if the window is raised.

## 2. Decisions

### A. The follow-up is condensed into a standalone query before retrieval

On every turn after the first, one short model call rewrites the question
as a standalone search query: it resolves *it / that paper / the same
system* against the recent turns, keeps the specific papers, materials,
quantities and conditions, drops conversational filler, ≤ 40 words. The
condensed query is what retrieval embeds, and it is shown to the user
("searched for: …") so a bad rewrite is visible rather than mysterious. A
failed or empty rewrite falls back to the raw question. ~5–10 s on CPU
against a 40–80 s answer. `CITATION_CHAT_CONDENSE=0` disables.

### B. Focus papers get a second, restricted search

The papers cited in the previous answer are the conversation's *focus*.
Each turn runs the global hybrid search (top-k×2, then at most
`CHAT_PER_DOC_CAP` = 3 chunks per paper, so a topic question sees more than
one paper) **and**, when there is a focus, a search restricted to the focus
papers (`CHAT_FOCUS_TOP_K` = 2). Focus hits go first; duplicates are
removed. A follow-up about "its limitations" is guaranteed to see the paper
it is about even if the condensed query drifted.

### C. An explicit token budget, and a rolling memory instead of silent loss

Every turn is assembled against a budget derived from `num_ctx`:

```
answer reserve        700 tokens     (what the model needs to write)
system prompt        ~250 tokens     (+ memory, ≤ 200)
context              ≤ 1,750 tokens  (CHAT_CONTEXT_MAX_CHARS = 7,000; chunks dropped from the tail)
history              the rest        (whole user/assistant pairs, most recent first)
```

Tokens are estimated at 4 characters each (this model's tokenizer on
English prose; a 10% margin is built into the reserve). When the recent
pairs that fit leave older pairs out, those older pairs are **summarised
into a rolling memory** by one model call (≤ 120 words: what was asked,
what was established with paper titles, what is open) that merges with the
previous memory; the memory rides in the system message as *"Conversation
so far: …"*. Nothing is dropped without being folded into the memory first;
the log records each fold. If the summariser fails, the previous memory
stands and the pairs are still dropped — the window cannot be exceeded.

Raising `CHAT_OLLAMA_OPTIONS["num_ctx"]` on a machine that can afford it
enlarges every line of the budget automatically.

### D. Answers cite keyed sources, and the keys are checked

Context blocks are labelled `[S1]…[Sn]` with the paper title, section and
page. The system prompt asks for a citation key on every factual sentence,
for the specifics the sources give (numbers, conditions, systems), for an
explicit statement of what the sources do not cover, and — when the corpus
holds a paper that likely does — which one to ask about next. After the
stream completes, the keys used are checked against the context: unknown
keys are reported as warnings, each source is marked *cited* or not, and
the focus for the next turn is the cited papers. In the stored history and
the export, `[S3]` is replaced by the paper's short title so a later turn
(and a reader of the export) does not see keys that meant something only
in one turn.

Temperature is explicit: `CHAT_TEMPERATURE` = 0.3; `chat_stream` gains a
`temperature` argument for both backends.

### E. The UI and the CLI show the machinery

Tab 4's sources expander shows the condensed query, each source with its
key, title, page and cited/not-cited mark, and any warnings; a memory
expander shows the rolling summary once it exists. The CLI's `/sources`
prints the same; `/memory` prints the summary. Export includes the memory.

## 3. Not changed

The index and `hybrid_search`. The synthesis path. `num_ctx` (stays 4,096
by config; the budget scales if it is raised). No reranker, no neighbour
expansion here — the per-turn budget is spent on more distinct chunks.

## 4. Configuration

| variable | default | meaning |
|---|---|---|
| `CITATION_CHAT_CONDENSE` | `1` | rewrite follow-ups into standalone queries |
| `CITATION_CHAT_CONTEXT_MAX_CHARS` | `7000` | cap on retrieved context per turn |
| `CITATION_CHAT_ANSWER_RESERVE_TOKENS` | `700` | kept free for the answer |
| `CITATION_CHAT_PER_DOC_CAP` | `3` | max chunks per paper in the global search |
| `CITATION_CHAT_FOCUS_TOP_K` | `2` | chunks from the focus papers |
| `CITATION_CHAT_TEMPERATURE` | `0.3` | answer sampling |
| `CHAT_OLLAMA_OPTIONS` (code) | `{"num_ctx": 4096}` | unchanged; the budget derives from it |

## 5. Risks

- **The condense call can rewrite a question wrongly.** It is shown; the raw
  question is still what the model answers; `CITATION_CHAT_CONDENSE=0`
  restores today's behaviour.
- **The memory can drop something the user cares about.** It is shown and
  exported; and it is strictly better than silent truncation, which is what
  happens today from turn four or five.
- **Two searches and up to two extra model calls per turn** (condense, and
  the occasional memory fold): ~10–15 s on CPU on top of a 40–80 s answer.
- **Keys checked, content not.** As with the synthesis: an invented key is
  caught; an invented fact under a real key is the judge's problem.

## 6. Verification

No harness. The live check repeats the three-turn conversation above and
extends it to six turns, instrumented: the turn-2 query must be standalone
and retrieve the MoS₂ paper; every turn's assembled prompt must fit
`num_ctx − reserve`; the memory must appear before any pair is dropped;
every `[S#]` in every answer must resolve; and the operator reads the six
answers.
