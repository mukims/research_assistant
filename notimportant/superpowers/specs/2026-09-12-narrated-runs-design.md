# Narrated runs — the UI shows the work as it happens

*2026-09-12*

## 1. Purpose

Marvin's jobs are long: a research-idea run on the CPU VM is a seed search,
a bibliography, thirty fetches, thirty parses with a model summary each and
— with figure analysis on — a minute per figure; then a synthesis. What the
reader sees during that hour is a progress bar, a 40-character `detail`
string and fifteen one-line events. What the pipeline *produces* during
that hour — the seed it chose, the reference list, which papers it could
get and why not the others, every paper's summary, every figure's
description, the shortlist and its notes — is written to disk or to Chroma
and never shown. `upsert_summaries` writes each summary straight into the
summary collection; the reader learns nothing until the synthesis lands.

The operator's ask, verbatim: *"this is information processing. we have to
find a way to give constant updates to the user. for example while
summarising the paper, give out summary for user to read; update when next
document is processed. while searching a research idea, communicate with
the user first and then walk the user with how you are researching the
idea."*

Three things are missing and they are one design:

1. **A content channel.** `pipeline_status` carries status. The run needs a
   second channel that carries *what was produced* — typed events with
   payloads — and a place where they accumulate.
2. **A reader that renders as it goes.** Tab 1 renders the result at the end
   of a blocking script. The feed has to appear while the run is in
   progress, survive a refresh, and be visible from any session.
3. **A voice before the work.** A research idea is typed into a form and the
   form submits into a progress bar. Marvin should say what he understood
   and how he will go about it, then go.

The decisions below were agreed with the operator on 2026-09-12: approach
A (background run + on-disk transcript + polling reader) over an inline
render or a chat-shaped Tab 1; the brief is a preface, not a gate ("brief,
then he runs on his own").

## 2. Decisions

### A. A run is a first-class thing on disk

`data/runs/<run_id>/` holds:

- `run.json` — `{run_id, kind, inputs, status, pid, started_at, finished_at,
  summary, error}`; `status ∈ running | done | stopped | cancelled | failed`.
  `stopped` is the graph's own early exit (no seed, no references); the
  reason is in `summary`.
- `events.jsonl` — one event per line: `{seq, ts, type, stage, payload}`.
  `seq` is monotonic within the run. Appended with one `write` per event
  and flushed, so a reader never sees a half-line; a torn last line (process
  killed mid-write) is skipped by the reader.
- `cancel` — present when a stop has been requested (§C).

`data/runs/current` is a one-line text file naming the run most recently
started on this machine. Kinds: `pipeline` (the LangGraph run: research
idea, seed URL, or single uploaded PDF), `ingest` (multi-PDF upload),
`cite` (Tab 3 batch citer), `verify` (Tab 3 verifier).

`research_assistant/shared/runlog.py` owns all of it:

```
start_run(kind, inputs) -> run_id      creates the directory, writes run.json (running),
                                       writes `current`, binds the run in this process;
                                       raises RunActive if a live run is already current
finish_run(status, summary="", error="")
run(kind, inputs)                      context manager: joins the bound run if there is one,
                                       otherwise start/finish around the body and translates
                                       exceptions to a status (RunCancelled → cancelled,
                                       KeyboardInterrupt → cancelled, Exception → failed)
emit(type, stage=None, **payload)      appends one event; no-op when nothing is bound or the
                                       caller is a forked child of the owning process
check_cancelled()                      raises RunCancelled if the bound run's `cancel` file exists
cancel(run_id)                         touches `cancel`
current_run_id()                       from the `current` file (any process)
read_run(run_id) -> (meta, events)
list_runs(limit) -> [meta]             newest first
prune(keep)                            deletes the oldest finished runs beyond `keep`
```

The bound run is a module-level value, not thread-local — the run thread
(§B) does the work, the script thread never emits, and one run per process
is the rule. `emit` from a `ProcessPoolExecutor` child (ingest with
`workers > 1`, CLI only) is a no-op: the child inherits the binding by fork
but not the file handle or the sequence counter. The app always ingests
with `workers=1`.

`pipeline_status` is unchanged in behaviour and stays the compact status
line (sidebar); it gains a `run_id` field set by `start_run` so the two
records can be joined.

### B. The run executes in a thread the page does not own

`app.py` stops driving `graph.stream` itself and stops holding the batch
ingest, batch cite and verify calls open inside a spinner. Every long
operation is one call to one entry point, and each entry point wraps its
own body in `runlog.run(kind, inputs)`:

| entry point | kind | who else calls it |
|---|---|---|
| `orchestrate.run(...)` | `pipeline` | the CLI, `watch.py` |
| `shared/batch_ingest.ingest_uploaded(staged, query, ask, force, describe_figures, search_resources)` — new; the 120 lines of Tab 1's multi-PDF branch moved out of the page | `ingest` | — |
| `agent5_batch_citer.run_batch_citer(...)` | `cite` | the CLI |
| `agent8_verifier.verify_draft(...)` | `verify` | the CLI |

Because the entry point opens the run, a CLI or `watch.py` run leaves the
same transcript and appears in the app exactly like one started from the
app.

`research_assistant/ui/runs.py` is the launcher the page uses:
`launch(kind, inputs, target, *args, **kwargs) -> run_id`. It refuses when
`current_run_id()` names a run whose status is `running` and whose pid is
alive (the page then shows that run instead of the form — replacing
today's "another process is running" box); otherwise it marks a stale
`running` run `failed: process died`, calls `start_run` synchronously (so
`current` is correct before the page reruns), and starts a
`threading.Thread(daemon=True)` whose body calls the target — the target's
own `runlog.run(...)` joins the bound run — and `finish_run` in `finally`
if the target did not. The thread makes no `st.*` call. Streamlit's script
thread returns immediately; the page reruns and finds the run in progress.

A refresh, a closed tab, a second browser and the operator's terminal all
find the run where it is.

### C. Stopping is cooperative, at item boundaries

The *Stop* button calls `runlog.cancel(run_id)`. `runlog.check_cancelled()`
is called, and raises `RunCancelled`, at:

- the top of every graph node in `orchestrate.py`;
- each iteration of Agent 2's fetch loop;
- each PDF in `_ingest_pdfs_locked`, each figure in `process_pdf_v2`, each
  document in `upsert_summaries`;
- each paper in the synthesis map loop (when the synthesis-depth plan's
  `paper_notes` exists) ;
- each sentence in `run_batch_citer`, each citation in `verify_draft`.

A model call in flight finishes first (≤ a minute or two on CPU). The
run's status becomes `cancelled`; `pipeline_status` goes idle;
`track_stage` treats `RunCancelled` like `KeyboardInterrupt` (an "⚠️
cancelled" event, not "❌ error"). The corpus is left as a crash at the same
boundary would leave it: a batch cancelled between PDFs is not upserted and
those PDFs are parsed again next run — the transcript says so in words.
Saving the partial batch on cancel is a deferred improvement (§7).
Streamlit's own stop button no longer reaches the work; the page does not
rely on it.

### D. The events

Every event carries the content, not a description of it. Payload keys are
fixed here so the reader (§F) and the tests can depend on them.

| stage | type | payload |
|---|---|---|
| `brief` | `brief` | `text`, `alternates: [str]` — Marvin's reading of the idea, three alternative search phrasings, what a good seed looks like. One model call (§E); only when the run searches for a seed. |
| `discover` | `search_tried` | `provider`, `query`, `results`, `with_pdf` — one per provider per phrasing |
| | `seed_found` | `title`, `key`, `doi`, `arxiv_id`, `url`, `source` (`search` / `manual-url` / `upload`), `via_query` |
| | `seed_missing` | `query`, `tried: [str]` |
| `ingest_seed`, `ingest_refs` | `paper_read` | `document`, `title`, `chunks`, `captions`, `described`, `extraction` — after `process_pdf` returns; the card appears now |
| | `figure_described` | `document`, `label`, `kind`, `description`, `image_path` — as each description is written |
| | `paper_empty` | `document`, `title` — no text layer |
| | `papers_skipped` | `n` — already indexed |
| | `paper_summarised` | `document`, `citation`, `summary` — as each summary is written; the card gains its summary |
| | `batch_saved` | `processed`, `inserted`, `described` |
| `extract` | `references_extracted` | `n`, `with_doi`, `by_method`, `sample: [{authors, year, title}]` (first ten) |
| `fetch` | `fetch_started` | `remaining`, `already` |
| | `paper_fetched` | `key`, `title`, `provider`, `doi` |
| | `paper_unavailable` | `key`, `title`, `reason` |
| `respond` | `shortlist`, `notes`, `synthesis` | forwarded from `retrieve.research_answer(on_progress=…)` when that argument exists (synthesis-depth plan); otherwise one `synthesis {text, citations}` from the node's result |
| any | `narration` | `text` — a templated line in Marvin's voice at stage transitions; **no model call** |
| any | `stopped` | `reason` — the graph's early exit |
| end | `finished` | `status`, `elapsed_s`, `summary` — written by `finish_run` |

`cite` runs emit `draft_split {n_sentences, n_eligible}`, then per sentence
`sentence_decided {index, sentence, needs_citation, cited, cited_text,
skip_reason, sources: [{key, citation}]}`, then `draft_written {path,
n_cited}`. `verify` runs emit `verify_started {n}`, per citation
`citation_judged {index, cite_key, citation_source, outcome, judgement,
confidence, escalated, rubric_mismatch}`, then `verify_done {totals}`.

The order inside `ingest_refs` follows the code as it is: every PDF is
parsed (a `paper_read` each, with its `figure_described`s), then the batch
is upserted and summarised (a `paper_summarised` each). A card therefore
appears when its paper is parsed and gains its summary later in the same
stage. Restructuring ingestion to per-paper upsert is out of scope.

### E. The brief

When a run will search for a seed (a query and no seed URL or file), the
`discover` node first makes one model call with `RESEARCH_BRIEF_USER`:
restate the idea in two sentences, propose exactly three alternative search
phrasings as numbered lines, and say what a good seed paper would look like
— ≤ 150 words, plain prose, no headings. `parse_brief(text)` extracts the
numbered lines as `alternates`. The event `brief` is emitted before the
search starts, so the reader sees it while the providers are queried.
`CITATION_BRIEF=0` skips the call. ~20–30 s on CPU.

The alternates are real: `agent0_discoverer.find_and_fetch_seed(query,
force, limit, alternates=())` walks providers for the typed query as today
and, only if nothing downloadable turns up, walks them again for each
alternate in order. `seed_found.via_query` records which phrasing worked.
Today that case ends in "No open-access seed PDF found" and an ungrounded
answer. The seed manifest is still keyed by the typed query.

Uploaded-seed and URL-seed runs get a templated narration instead of a
model brief ("You gave me the paper; I'll read it, chase what it cites, and
answer from those.").

### F. The reader

**Tab 1 is either the form or the run.** With no active run, the form as
today with Marvin's copy and a single *Go* button. Once pressed, the page is
the run: a `st.fragment(run_every="2s")` reads `run.json` and the whole of
`events.jsonl` and re-draws the feed. Re-reading the whole file each tick is
deliberate — a few hundred lines, well under a megabyte — and keeps the
renderer a pure function of the transcript.

The pure part lives in `research_assistant/shared/transcript.py` and is
tested without Streamlit: `sections(events)` groups events by stage in a
fixed order and marks the latest; `paper_cards(events)` merges `paper_read`
+ `figure_described` + `paper_summarised` by `document` into one card each,
in first-seen order; `counts(events)`; `status_line(meta, events)` ("Reading
paper 7 of 31", "Fetching — 12 of 31 checked", "Done", "Stopped: …");
`sentence_rows(events)` for `cite` runs; `verdict_rows(events)` for
`verify`. `research_assistant/ui/feed.py` renders them:

```
Marvin the Citebot                          ● Reading paper 7 of 31 · 12 min    [Stop]
▸ The brief          the brief text
▸ The seed           title · identifiers · numbers line · narration
▸ References         "62 found · 31 fetched · 31 unavailable"; first ten of each, expander for the rest
▾ Reading            one bordered card per paper: title · "41 chunks · 3 captions · 2 described · GROBID"
                     · the summary in full (or "summary pending") · figures behind an expander,
                     crop beside description
▸ The synthesis      shortlist with scores → notes per paper as written → the text with its key legend
```

The latest section is expanded, earlier ones collapsed. A cancelled or
failed run ends with the reason in the feed. When the run finishes the feed
stays; the form returns beneath a divider.

**Any session sees the same run.** `_render_tab1_live_status` (the
"Pipeline Active on Server" box) is deleted; a session that opens Tab 1
during a run sees the run.

**Past runs.** Below the form, an expander *What he has read* lists
`list_runs()` — date, kind, query or label, status, elapsed — and selecting
one renders its transcript with the same reader.

**Tab 3** launches `cite` and `verify` runs the same way and renders
`sentence_rows` / `verdict_rows` as they land: cited sentences with the
inserted `\cite{}` and their sources, skipped ones greyed with the reason,
declined ones marked. The finished state — the text area, download button,
report, metric tiles, verification report — is unchanged.

**The sidebar** is sorted by reader: top, Marvin's status line (from the
current run when there is one, else *Idle* with the last run's summary),
papers and chunks, and the one-line capability note ("No corpus yet — build
one first"); below, behind an *Operator* expander closed by default,
backend names, index version, GROBID controls and diagnostics. Nothing is
removed.

### G. Identity

Page title *Marvin the Citebot*, icon 🤖, the README's one-line description
as the caption. The auth screen, empty states, narration templates, button
labels and "nothing matched" messages are his — dry and specific, never
cute; the strings live in `research_assistant/ui/copy.py`. `HOW_TO_USE.md`
is retitled and its Tab 1 section rewritten for the feed. No theme, no CSS,
no new colours: `.streamlit/config.toml` is untouched.

### H. `app.py` shrinks

The page keeps the tab skeleton and the forms. The multi-PDF branch moves
to `shared/batch_ingest.py`; the `graph.stream` driver, `STEPS`,
`node_weights`, `stage_ranges` and the two progress callbacks are deleted
(`orchestrate.run` does that work); rendering of transcripts is in
`ui/feed.py`; strings in `ui/copy.py`; the launcher in `ui/runs.py`.
`tests/test_app_names.py` (every bare name the page calls must be bound)
continues to guard the page.

## 3. Not changed

`hybrid_search`, the index, ingestion's batch structure (parse all →
upsert → summarise), `pipeline_status`'s behaviour, `num_ctx`. Tab 2 (one
40-second call, nothing to narrate) and Tab 4 (already streams; the
chat-depth plan owns its machinery display) beyond copy. The theme. The
deployed `config.toml`. Saving a partially processed batch on cancel.

## 4. Configuration

| variable | default | meaning |
|---|---|---|
| `RUNS_DIR` (code) | `data/runs` | transcripts |
| `CITATION_RUNS_KEEP` | `50` | finished runs kept; oldest pruned at `start_run` |
| `CITATION_BRIEF` | `1` | model brief before a seed search |
| `FEED_REFRESH_SECONDS` (code) | `2` | reader poll interval |
| `CITATION_BRIEF_TEMPERATURE` | `0.3` | the brief call |

## 5. Errors

- **A second run while one is active** — `launch` refuses; the page shows
  the active run. The CLI's `runlog.run` raises `RunActive` with the run id.
- **Stale `running`** (process died) — detected by pid at `launch` and by
  the reader (status `running`, pid dead → shown as *died*); `start_run`
  marks it `failed` before starting the next.
- **Torn last line** in `events.jsonl` — skipped by the reader.
- **Brief call fails or returns nothing parseable** — no `brief` event
  beyond a narration line "I'll search for it as you wrote it"; no
  alternates; the run proceeds.
- **`research_answer` without `on_progress`** (synthesis-depth not built) —
  the node detects the signature and emits `synthesis` from the result.
- **Any `st.*` call from the run thread** is a bug: the live check greps
  Streamlit's log for `missing ScriptRunContext`.

## 6. Testing

Unit, without models or Streamlit: `runlog` (start/emit/finish/read/list,
join, cancel, stale, prune, torn line, child-pid no-op), `transcript`
(sections, cards, counts, status line, sentence and verdict rows),
`narration` (every template with 0/1/n), `parse_brief`, Agent 0 alternates
(mocked providers), each emitter (orchestrate nodes with mocked agents;
`_ingest_pdfs_locked` with a fake `process_pdf`; the fetch loop; the batch
citer; the verifier) asserting the events by type and payload keys, and
cancel propagation at each boundary; `ui/runs.launch` (thread runs the
target, run finishes, second launch refused). The existing app-names guard.

Live: one research-idea run on the v2 index, watched from two browser
sessions and after a refresh; the brief appears before any search event;
the seed card, the reference list, the fetch outcomes, each paper card and
its summary, and the synthesis appear in that order while the run is in
progress; Stop during `ingest_refs` ends the run as `cancelled` within one
paper; the Streamlit log has no `ScriptRunContext` warning; a `cite` and a
`verify` run on a three-sentence draft show rows as they land.

## 7. Risks

- **Daemon thread and process exit.** A run dies with the Streamlit
  process; the transcript records `running` and the next start marks it
  `failed`. The CLI path is unaffected. Same as today's exposure.
- **Two-second polling re-renders a few hundred elements** per session per
  tick. Fine for one or two readers on the VM; `FEED_REFRESH_SECONDS` is
  one line.
- **Expander state resets on rerun.** Streamlit keeps it when the element's
  position is stable; the latest-open rule makes the default the right one.
- **The brief can misread the idea.** It is shown; the typed query is still
  searched first; `CITATION_BRIEF=0` restores today's behaviour.
- **Cancel between PDFs discards the parsed batch.** Recorded in the
  transcript; saving the partial batch is deferred.
