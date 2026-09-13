# Narrated Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every long thing Marvin does — building a corpus from an idea or a paper, ingesting uploads, auditing a seed's citations, citing a draft, verifying the citations — becomes a *run* with an on-disk transcript of typed events carrying the content produced (the brief, the seed, each reference, each fetch, each paper's summary and figures, the synthesis, each sentence's decision, each verdict), executed on a thread the page does not own, and the Streamlit page becomes a reader of that transcript: it shows the run as it happens, survives refresh, can stop it, and lists past runs. The app takes Marvin's name and voice.

**Architecture:** `research_assistant/shared/runlog.py` owns the transcript (`data/runs/<run_id>/run.json` + `events.jsonl`, one run bound per process, cooperative cancel via a `cancel` file checked at item boundaries). Every entry point wraps its body in `runlog.run(kind, inputs)` and calls `runlog.emit(type, **payload)` where it already calls `pipeline_status` — nothing new is threaded through signatures; `pipeline_status.track_stage` sets the transcript's current stage. `research_assistant/shared/transcript.py` is the pure model of a transcript (sections, per-paper cards, counts, status line), `research_assistant/ui/feed.py` draws it, `research_assistant/ui/runs.py` launches an entry point on a daemon thread, `research_assistant/ui/copy.py` holds every string. `app.py` shrinks to forms + the reader. The brief (`shared/brief.py`) is one model call before a seed search whose three phrasings Agent 0 falls back to.

**Tech Stack:** Python 3.12, Streamlit 1.53.1 (`st.fragment(run_every=…)`, `st.rerun(scope="app")`), LangGraph (unchanged), threading, JSONL on disk. No new dependencies. Tests: `unittest.TestCase` collected by pytest; one new `tests/conftest.py` autouse fixture isolates the status file and the runs directory for every test.

**Spec:** `notimportant/superpowers/specs/2026-09-12-narrated-runs-design.md` — read it first; this plan argues from it. Section references below (§2A …) are to that spec.

**Base commit:** `f6e5f25` on branch `judgement-hardening` (HEAD when this plan was written). Every diff in this plan is against that tree. Work on a branch from it, e.g. `narrated-runs`. If the base has moved, the diffs still apply where the `-` lines match; where they do not, the surrounding prose says what the change is.

**Validated:** every code block in this plan was assembled into a scratch copy of the repo at the base commit, its tests run (661 passed, 1 skipped), and the page exercised live on a copy of the v2 index: a `cite` run watched row by row, a research-idea run watched through brief → seed → references → fetch and stopped at a fetch boundary (`cancelled`, pipeline_status idle), a browser reload mid-run finding the run and the run finishing after the launching session was gone, the past-runs list, and the Streamlit log clean of `ScriptRunContext` warnings. Two bugs found live are already fixed in these blocks (a duplicate Stop-button key when two tabs show one run; raw `<sub>` in fetched rows).

> **Reconciliation with what landed after this plan was written (2026-09-13).** Commit `0a9a41e` added cooperative cancellation to `pipeline_status` — `request_cancel()`, `is_cancel_requested()`, `clear_cancel_request()`, `PipelineCancelledError`, a Stop button in the sidebar and in Tab 1, and `is_cancel_requested()` checks at every graph node, each fetched paper, each parsed PDF and each summary. This plan's `runlog.cancel` / `check_cancelled` (Task 1) and its boundary checks (Tasks 5–9) describe the same mechanism twice. Build it as one: `runlog.cancel(run_id)` calls `pipeline_status.request_cancel(...)` **and** touches the run's `cancel` file; `runlog.check_cancelled()` raises `RunCancelled` when either the file exists or `pipeline_status.is_cancel_requested()` is true; leave the existing `is_cancel_requested()` checks in place and do not add a second check next to them; treat `PipelineCancelledError` wherever this plan says `RunCancelled` (make `RunCancelled` a subclass of it). Commit `b6ebf7b` added a Gemini backend (`LLM_BACKEND=openai` when `GEMINI_API_KEY` is set): the brief and every model call in this plan work unchanged; timings quoted for "CPU" are then seconds. Commit `6b9ca58` rewrote Tab 4; this plan's Task 12 copies Tab 4 verbatim from the base it was written against — take Tab 4 from the current `app.py` instead, unchanged.

## Global Constraints

- **`gemma4:e2b` stays the generator; CPU-only VM stays.** The brief is one extra model call per research-idea run (~20–60 s on CPU); narration is templated, never a model call.
- **`CHAT_OLLAMA_OPTIONS = {"num_ctx": 4096}`** (GEMINI.md) — the brief uses it; nothing here changes `num_ctx`.
- **`data/` is never touched by tests.** `tests/conftest.py` (Task 1) redirects `pipeline_status` and `runlog` to `tmp_path` for every test. Keep it.
- **The run thread makes no `st.*` call.** The live check greps the Streamlit log for `missing ScriptRunContext`; any hit is a bug.
- **One run per process.** `runlog.start_run` raises `RunActive` while `data/runs/current` names a live run; the page shows that run instead of the form.
- **Cancel only at item boundaries, never inside a model call** — `runlog.check_cancelled()` sits at the top of loops over papers / figures / summaries / sentences / citations / claims and of every graph node.
- **Event payload keys are the contract** between emitters (Tasks 4–9) and the reader (Tasks 3, 11). They are listed in spec §2D and fixed by `tests/test_transcript.py`; do not rename one side.
- **No theme, no CSS, no changes to `.streamlit/config.toml`, `Dockerfile`, `deploy/`.** Identity is name + copy only.
- **Test conventions:** `unittest.TestCase`; run with `CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider`; heavy imports stay lazy; live checks are manual (Task 14), never in the suite. If `data/ingest.lock` is held on your machine, `--deselect tests/test_ingestion.py` while it is.
- **Interpreter on this machine:** `/home/shardul/miniconda3/envs/ml/bin/python` (has streamlit, langgraph, chromadb). Run scripts with `PYTHONPATH=.`.
- **Commit per task** with the message given; end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File map

| file | task | responsibility |
|---|---|---|
| `research_assistant/config.py` | 1 | `RUNS_DIR`, `RUNS_KEEP`, `BRIEF`, `BRIEF_TEMPERATURE`, `FEED_REFRESH_SECONDS` |
| `research_assistant/shared/runlog.py` (new) | 1 | the transcript: start/emit/finish/run/cancel/read/list/prune |
| `tests/conftest.py` (new) | 1 | every test gets a throwaway status file and runs dir |
| `research_assistant/shared/pipeline_status.py` | 2 | `track_stage` sets the transcript stage; `RunCancelled` is a stop, not an error |
| `research_assistant/shared/narration.py` (new) | 3 | Marvin's templated lines |
| `research_assistant/shared/transcript.py` (new) | 3 | pure model of a transcript for the reader |
| `research_assistant/prompts.py`, `research_assistant/shared/brief.py` (new) | 4 | the brief prompt, the call, `parse_brief` |
| `research_assistant/agents/agent0_discoverer.py` | 4 | `alternates`, `search_tried` / `seed_found` / `seed_missing` events |
| `research_assistant/shared/ingestion.py`, `shared/ingest_v2.py` | 5 | `batch_started` / `paper_read` / `paper_empty` / `figure_described` / `summaries_started` / `paper_summarised` / `batch_saved`; cancel between papers, figures, summaries |
| `research_assistant/agents/agent2_fetcher.py` | 6 | `fetch_started` / `paper_fetched` / `paper_unavailable`; cancel between papers; returns counts |
| `research_assistant/shared/respond.py` (new), `orchestrate.py` | 7 | the synthesis narrated (with or without `on_progress`); `runlog.run` around the graph; brief; per-node events and cancel |
| `research_assistant/shared/batch_ingest.py` (new) | 8 | the multi-PDF upload as an `ingest` run, out of the page |
| `research_assistant/agents/agent5_batch_citer.py`, `agents/agent8_verifier.py`, `shared/seed_audit.py` | 9 | `cite`, `verify`, `audit` runs: per-item events, cancel between items |
| `research_assistant/ui/__init__.py`, `ui/copy.py`, `ui/runs.py` (new) | 10 | strings; the launcher |
| `research_assistant/ui/feed.py` (new) | 11 | draw a transcript |
| `app.py`, `scripts/run_check.py` (new) | 12 | the page: forms + reader; the transcript checker |
| `HOW_TO_USE.md`, `PIPELINE.md`, `ARCHITECTURE.md`, `README.md` | 13 | docs |
| — | 14 | the live check |

---

### Task 1: `runlog` — the transcript on disk

**Files:**
- Modify: `research_assistant/config.py`
- Create: `research_assistant/shared/runlog.py`
- Create: `tests/conftest.py`
- Test: `tests/test_runlog.py`

**Interfaces:**
- Consumes: `research_assistant.shared.atomic.atomic_write_json`, `pipeline_status.set_status(run_id=…)` (already accepts arbitrary keys).
- Produces (used by every later task): `RunCancelled`, `RunActive`, `start_run(kind, inputs) -> run_id`, `finish_run(status=None, summary="", error="") -> meta|None`, `run(kind, inputs)` context manager (joins the bound run), `emit(type_, stage=None, **payload) -> event|None`, `note(text, stage=None)`, `set_stage(stage)`, `current_stage()`, `set_summary(text)`, `set_stopped(reason)`, `check_cancelled()`, `cancel(run_id)`, `cancel_requested(run_id)`, `current_run_id()`, `read_meta(run_id)`, `read_events(run_id)`, `read_run(run_id) -> (meta, events)`, `list_runs(limit=20)`, `is_live(meta)`, `is_pid_alive(pid)`, `elapsed_seconds(meta)`, `prune(keep)`, `run_dir(run_id)`, `set_runs_dir(path|None)`, `bound_run_id()`, `_reset_for_tests()`. Statuses: `running | done | stopped | cancelled | failed`. Every run's last event is `finished` with `{status, elapsed_s, summary, error}` in stage `end`.

Why a module-level binding rather than thread-local (spec §2A): the app's launcher starts the run on the script thread and the entry point runs on another; both must see the same run. Why `emit` is a no-op in a forked child: `ingest_pdfs(workers>1)` forks, the child inherits the binding but not the file handle or the counter.

- [ ] **Step 1: Add the config values**

**Apply this change to `research_assistant/config.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/config.py
+++ b/research_assistant/config.py
@@ -103,6 +103,11 @@
 SEED_PAPERS_PATH         = os.path.join(DATA_DIR, "seed_papers.json")
 INGESTED_MANIFEST_PATH   = os.path.join(DATA_DIR, f"ingested{_INDEX_SUFFIX}.json")
 PIPELINE_STATUS_PATH     = os.path.join(DATA_DIR, "pipeline_status.json")
+# Narrated runs (spec 2026-09-12-narrated-runs §2A): one directory per run,
+# holding run.json and the events.jsonl transcript the UI reads while the run
+# is in progress. `current` inside it names the most recently started run.
+RUNS_DIR                 = os.path.join(DATA_DIR, "runs")
+RUNS_KEEP                = _env_int("CITATION_RUNS_KEEP", 50)
 INGEST_LOCK_PATH         = os.path.join(DATA_DIR, "ingest.lock")
 
 # ─── Detectron2 / layout detection ──────────────────────────────────────────
@@ -148,6 +153,15 @@
 # default; crops can be described on demand instead).
 FIGURE_VLM = _env_bool("CITATION_FIGURE_VLM", False)
 
+# ─── Narrated runs — the brief and the reader ────────────────────────────────
+# One model call before a seed search restates the idea and proposes three
+# alternative phrasings (spec §2E). Agent 0 tries them only when the typed
+# query finds nothing open-access.
+BRIEF                    = _env_bool("CITATION_BRIEF", True)
+BRIEF_TEMPERATURE        = float(os.environ.get("CITATION_BRIEF_TEMPERATURE", "0.3"))
+# How often the page re-reads a running transcript.
+FEED_REFRESH_SECONDS     = 2
+
 # Per-document summary written to SUMMARY_COLLECTION_NAME at ingest time — the
 # stage-1 "is this paper even relevant" index. One model call per paper.
 SUMMARY_MODEL            = os.environ.get("CITATION_SUMMARY_MODEL", "") or None  # None → LLM_MODEL
```

- [ ] **Step 2: Add the autouse fixture so no test ever writes `data/`**

**Create `tests/conftest.py` with exactly this content:**

```python
"""Every test runs against a throwaway status file and runs directory.

The pipeline writes data/pipeline_status.json and, since the narrated runs
(spec 2026-09-12-narrated-runs), data/runs/<run_id>/ from wherever it is
called — including from an entry point a test exercises with its network
and model seams mocked. Without this fixture such a test would leave a
real run behind and could clobber the status of a run in progress on the
same machine. Tests that need their own paths still call set_status_path /
set_runs_dir themselves; this only guarantees the default is never the
real one.
"""

import pytest

from research_assistant.shared import pipeline_status, runlog


@pytest.fixture(autouse=True)
def _isolated_run_state(tmp_path):
    pipeline_status.set_status_path(str(tmp_path / "pipeline_status.json"))
    runlog.set_runs_dir(str(tmp_path / "runs"))
    runlog._reset_for_tests()
    yield
    runlog._reset_for_tests()
    runlog.set_runs_dir(None)
    pipeline_status.set_status_path(None)
```

- [ ] **Step 3: Write the failing tests**

**Create `tests/test_runlog.py` with exactly this content:**

```python
"""runlog — the on-disk transcript of a run (spec 2026-09-12-narrated-runs §2A–C)."""

import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from research_assistant.shared import pipeline_status, runlog


class RunlogCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
        self.addCleanup(runlog.set_runs_dir, None)
        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
        self.addCleanup(pipeline_status.set_status_path, None)
        runlog._reset_for_tests()
        self.addCleanup(runlog._reset_for_tests)


class TestStartEmitFinish(RunlogCase):
    def test_start_creates_meta_events_and_current(self):
        rid = runlog.start_run("pipeline", {"query": "anderson localization"})
        meta = runlog.read_meta(rid)
        self.assertEqual(meta["status"], "running")
        self.assertEqual(meta["kind"], "pipeline")
        self.assertEqual(meta["inputs"], {"query": "anderson localization"})
        self.assertEqual(meta["pid"], os.getpid())
        self.assertEqual(runlog.current_run_id(), rid)
        self.assertEqual(runlog.bound_run_id(), rid)
        self.assertTrue(os.path.exists(os.path.join(runlog.run_dir(rid), "events.jsonl")))

    def test_start_records_run_id_in_pipeline_status(self):
        rid = runlog.start_run("pipeline", {})
        self.assertEqual(pipeline_status.get_status().get("run_id"), rid)

    def test_emit_appends_sequenced_events_with_payload(self):
        rid = runlog.start_run("pipeline", {})
        runlog.set_stage("discover")
        ev1 = runlog.emit("search_tried", provider="arxiv", results=3, with_pdf=2)
        ev2 = runlog.emit("seed_found", stage="ingest_seed", title="T")
        self.assertEqual((ev1["seq"], ev2["seq"]), (1, 2))
        self.assertEqual(ev1["stage"], "discover")        # defaulted from set_stage
        self.assertEqual(ev2["stage"], "ingest_seed")     # explicit wins
        meta, events = runlog.read_run(rid)
        self.assertEqual([e["type"] for e in events], ["search_tried", "seed_found"])
        self.assertEqual(events[0]["payload"], {"provider": "arxiv", "results": 3, "with_pdf": 2})

    def test_emit_without_a_bound_run_is_a_noop(self):
        self.assertIsNone(runlog.emit("anything", x=1))

    def test_emit_from_a_forked_child_is_a_noop(self):
        runlog.start_run("pipeline", {})
        with patch.object(runlog.os, "getpid", return_value=os.getpid() + 1):
            self.assertIsNone(runlog.emit("paper_read", document="d"))
        self.assertEqual(len(runlog.read_events(runlog.bound_run_id())), 0)

    def test_note_is_a_narration_event(self):
        runlog.start_run("pipeline", {})
        ev = runlog.note("62 references.", stage="extract")
        self.assertEqual(ev["type"], "narration")
        self.assertEqual(ev["payload"], {"text": "62 references."})

    def test_finish_writes_finished_event_and_final_meta(self):
        rid = runlog.start_run("pipeline", {})
        runlog.emit("x")
        meta = runlog.finish_run("done", summary="4 papers")
        self.assertEqual(meta["status"], "done")
        self.assertEqual(meta["summary"], "4 papers")
        self.assertTrue(meta["finished_at"])
        events = runlog.read_events(rid)
        self.assertEqual(events[-1]["type"], "finished")
        self.assertEqual(events[-1]["stage"], "end")
        self.assertEqual(events[-1]["payload"]["status"], "done")
        self.assertIn("elapsed_s", events[-1]["payload"])
        self.assertIsNone(runlog.bound_run_id())
        self.assertIsNone(runlog.finish_run("done"))   # idempotent

    def test_finish_default_status_is_done_or_stopped(self):
        runlog.start_run("pipeline", {})
        runlog.set_summary("all good")
        self.assertEqual(runlog.finish_run()["summary"], "all good")
        self.assertEqual(runlog.read_meta(runlog.current_run_id())["status"], "done")
        runlog.start_run("pipeline", {})
        runlog.set_stopped("no open-access PDF found")
        meta = runlog.finish_run()
        self.assertEqual((meta["status"], meta["summary"]), ("stopped", "no open-access PDF found"))

    def test_finish_rejects_unknown_status(self):
        runlog.start_run("pipeline", {})
        with self.assertRaises(ValueError):
            runlog.finish_run("exploded")

    def test_emit_is_safe_across_threads(self):
        rid = runlog.start_run("pipeline", {})

        def burst():
            for i in range(50):
                runlog.emit("tick", i=i)

        threads = [threading.Thread(target=burst) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        events = runlog.read_events(rid)
        self.assertEqual(len(events), 200)
        self.assertEqual([e["seq"] for e in events], list(range(1, 201)))


class TestRunContext(RunlogCase):
    def test_run_starts_and_finishes_done(self):
        with runlog.run("cite", {"draft": "d.txt"}) as rid:
            runlog.emit("draft_split", n_sentences=3)
        self.assertEqual(runlog.read_meta(rid)["status"], "done")

    def test_run_translates_exceptions_to_status(self):
        with self.assertRaises(runlog.RunCancelled):
            with runlog.run("cite", {}) as rid:
                raise runlog.RunCancelled(rid)
        self.assertEqual(runlog.read_meta(rid)["status"], "cancelled")
        with self.assertRaises(ValueError):
            with runlog.run("cite", {}) as rid2:
                raise ValueError("boom")
        meta = runlog.read_meta(rid2)
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(meta["error"], "ValueError: boom")

    def test_run_joins_the_bound_run(self):
        outer = runlog.start_run("pipeline", {})
        with runlog.run("pipeline", {}) as inner:
            self.assertEqual(inner, outer)
            runlog.set_stopped("early")
        # Joined: the inner exit must not finish the outer run.
        self.assertEqual(runlog.read_meta(outer)["status"], "running")
        self.assertEqual(runlog.finish_run()["status"], "stopped")

    def test_run_stopped_when_set_stopped(self):
        with runlog.run("pipeline", {}) as rid:
            runlog.set_stopped("no references")
        self.assertEqual(runlog.read_meta(rid)["status"], "stopped")
        self.assertEqual(runlog.read_meta(rid)["summary"], "no references")


class TestActiveAndStale(RunlogCase):
    def test_second_start_in_process_raises_run_active(self):
        runlog.start_run("pipeline", {})
        with self.assertRaises(runlog.RunActive):
            runlog.start_run("cite", {})

    def test_live_run_on_disk_from_another_process_blocks_start(self):
        rid = runlog.start_run("pipeline", {})
        runlog._reset_for_tests()                       # forget it in-process; it stays `running` on disk
        with patch.object(runlog, "is_pid_alive", return_value=True):
            with self.assertRaises(runlog.RunActive):
                runlog.start_run("cite", {})
        self.assertEqual(runlog.read_meta(rid)["status"], "running")

    def test_dead_running_run_is_marked_failed_then_start_proceeds(self):
        rid = runlog.start_run("pipeline", {})
        runlog._reset_for_tests()
        with patch.object(runlog, "is_pid_alive", return_value=False):
            rid2 = runlog.start_run("cite", {})
        self.assertNotEqual(rid, rid2)
        old = runlog.read_meta(rid)
        self.assertEqual((old["status"], old["error"]), ("failed", "process died"))
        self.assertEqual(runlog.current_run_id(), rid2)

    def test_is_live(self):
        self.assertFalse(runlog.is_live(None))
        self.assertFalse(runlog.is_live({"status": "done", "pid": os.getpid()}))
        self.assertTrue(runlog.is_live({"status": "running", "pid": os.getpid()}))
        with patch.object(runlog, "is_pid_alive", return_value=False):
            self.assertFalse(runlog.is_live({"status": "running", "pid": 1}))


class TestCancel(RunlogCase):
    def test_check_cancelled_raises_after_cancel(self):
        rid = runlog.start_run("pipeline", {})
        runlog.check_cancelled()                        # nothing requested: silent
        runlog.cancel(rid)
        self.assertTrue(runlog.cancel_requested(rid))
        with self.assertRaises(runlog.RunCancelled):
            runlog.check_cancelled()

    def test_check_cancelled_without_a_run_is_silent(self):
        runlog.check_cancelled()


class TestReading(RunlogCase):
    def test_torn_last_line_is_skipped(self):
        rid = runlog.start_run("pipeline", {})
        runlog.emit("a")
        runlog.emit("b")
        with open(os.path.join(runlog.run_dir(rid), "events.jsonl"), "a", encoding="utf-8") as fh:
            fh.write('{"seq": 3, "type": "c", "pay')
        self.assertEqual([e["type"] for e in runlog.read_events(rid)], ["a", "b"])

    def test_read_missing_run(self):
        self.assertEqual(runlog.read_run("nope"), (None, []))
        self.assertIsNone(runlog.current_run_id())

    def test_list_runs_newest_first_and_limited(self):
        ids = []
        for i in range(3):
            with patch.object(runlog, "_now_iso", return_value=f"2026-09-12T10:0{i}:00Z"):
                ids.append(runlog.start_run("pipeline", {"i": i}))
                runlog.finish_run("done")
        listed = runlog.list_runs()
        self.assertEqual([m["run_id"] for m in listed], list(reversed(ids)))
        self.assertEqual(len(runlog.list_runs(limit=2)), 2)

    def test_elapsed_seconds(self):
        meta = {"started_at": "2026-09-12T10:00:00Z", "finished_at": "2026-09-12T10:12:30Z"}
        self.assertEqual(runlog.elapsed_seconds(meta), 750.0)
        self.assertEqual(runlog.elapsed_seconds({}), 0.0)
        self.assertGreaterEqual(runlog.elapsed_seconds({"started_at": "2026-09-12T10:00:00Z"}), 0.0)


class TestPrune(RunlogCase):
    def test_prune_removes_oldest_finished_beyond_keep(self):
        ids = []
        for i in range(4):
            with patch.object(runlog, "_now_iso", return_value=f"2026-09-12T10:0{i}:00Z"):
                ids.append(runlog.start_run("pipeline", {}))
                runlog.finish_run("done")
        removed = runlog.prune(keep=2)
        self.assertEqual(removed, 2)
        remaining = {m["run_id"] for m in runlog.list_runs()}
        self.assertEqual(remaining, set(ids[2:]))

    def test_prune_never_touches_a_running_run(self):
        with patch.object(runlog, "_now_iso", return_value="2026-09-12T09:00:00Z"):
            old = runlog.start_run("pipeline", {})
            runlog.finish_run("done")
        live = runlog.start_run("pipeline", {})
        self.assertEqual(runlog.prune(keep=0), 1)
        self.assertIsNone(runlog.read_meta(old))
        self.assertEqual(runlog.read_meta(live)["status"], "running")

    def test_start_prunes_to_runs_keep(self):
        for i in range(3):
            with patch.object(runlog, "_now_iso", return_value=f"2026-09-12T10:0{i}:00Z"):
                runlog.start_run("pipeline", {})
                runlog.finish_run("done")
        with patch("research_assistant.config.RUNS_KEEP", 1):
            runlog.start_run("pipeline", {})
        self.assertEqual(len(runlog.list_runs()), 2)   # one kept finished + the new running one


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_runlog.py
```

Expected: FAIL — `ModuleNotFoundError: research_assistant.shared.runlog`

- [ ] **Step 5: Write `runlog`**

**Create `research_assistant/shared/runlog.py` with exactly this content:**

```python
"""
runlog — the transcript of a run (spec 2026-09-12-narrated-runs §2A–C).

A run is a directory under config.RUNS_DIR:

    <run_id>/run.json       {run_id, kind, inputs, status, pid, started_at,
                             finished_at, summary, error}
    <run_id>/events.jsonl   one event per line: {seq, ts, type, stage, payload}
    <run_id>/cancel         present once a stop has been requested

and RUNS_DIR/current is a one-line file naming the run most recently started
on this machine. Writers (the pipeline, in whatever process runs it) bind one
run per process and append events; readers (the Streamlit page, the CLI)
only ever read the files, so a run started from the terminal is visible in
the app and a refresh of the app finds the run where it is.

pipeline_status remains the compact status line. This module is the content
channel next to it: an event carries what was produced — the summary text,
the reference list, the figure description — not a description of it.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from research_assistant.shared.atomic import atomic_write_json


class RunCancelled(Exception):
    """Raised by check_cancelled() at an item boundary after cancel()."""


class RunActive(RuntimeError):
    """start_run() while a live run is current in this process or on disk."""


STATUSES = ("running", "done", "stopped", "cancelled", "failed")
FINAL_STATUSES = frozenset(STATUSES) - {"running"}
STAGE_END = "end"

RUNS_DIR_OVERRIDE: Optional[str] = None

_lock = threading.Lock()
_bound: Optional[dict[str, Any]] = None
_stage: Optional[str] = None


# ─── Paths ───────────────────────────────────────────────────────────────────


def runs_dir() -> str:
    if RUNS_DIR_OVERRIDE is not None:
        return RUNS_DIR_OVERRIDE
    from research_assistant import config

    return config.RUNS_DIR


def set_runs_dir(path: Optional[str]) -> None:
    """Override the runs directory (tests)."""
    global RUNS_DIR_OVERRIDE
    RUNS_DIR_OVERRIDE = path


def run_dir(run_id: str) -> str:
    return os.path.join(runs_dir(), run_id)


def _meta_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "run.json")


def _events_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "events.jsonl")


def _cancel_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "cancel")


def _current_path() -> str:
    return os.path.join(runs_dir(), "current")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(kind: str) -> str:
    """Sortable by start time, unique enough for one machine."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{kind}-{secrets.token_hex(2)}"


# ─── Reading ─────────────────────────────────────────────────────────────────


def read_meta(run_id: str) -> Optional[dict[str, Any]]:
    try:
        with open(_meta_path(run_id), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def read_events(run_id: str) -> list[dict[str, Any]]:
    """Every event of the run, in order. A torn last line (the writer was
    killed mid-write) is skipped rather than raised."""
    events = []
    try:
        with open(_events_path(run_id), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    events.append(ev)
    except OSError:
        pass
    return events


def read_run(run_id: str) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    return read_meta(run_id), read_events(run_id)


def current_run_id() -> Optional[str]:
    try:
        with open(_current_path(), encoding="utf-8") as fh:
            run_id = fh.read().strip()
    except OSError:
        return None
    return run_id or None


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Run metadata, newest first."""
    base = runs_dir()
    try:
        names = os.listdir(base)
    except OSError:
        return []
    metas = []
    for name in names:
        meta = read_meta(name)
        if meta:
            metas.append(meta)
    metas.sort(key=lambda m: (m.get("started_at") or "", m.get("run_id") or ""), reverse=True)
    return metas[:limit]


def is_pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def is_live(meta: Optional[dict[str, Any]]) -> bool:
    """`running` and the process that started it still exists."""
    return bool(meta) and meta.get("status") == "running" and is_pid_alive(meta.get("pid"))


def elapsed_seconds(meta: dict[str, Any]) -> float:
    try:
        start = datetime.fromisoformat(meta["started_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError):
        return 0.0
    end_str = meta.get("finished_at")
    try:
        end = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else datetime.now(timezone.utc)
    except (ValueError, AttributeError):
        end = datetime.now(timezone.utc)
    return max(0.0, (end - start).total_seconds())


# ─── Writing ─────────────────────────────────────────────────────────────────


def _write_meta(meta: dict[str, Any]) -> None:
    atomic_write_json(_meta_path(meta["run_id"]), meta, ensure_ascii=False, default=str)


def bound_run_id() -> Optional[str]:
    return _bound["run_id"] if _bound else None


def set_stage(stage: Optional[str]) -> None:
    """The stage later emit() calls default to. pipeline_status.track_stage
    sets it on entry, so emitters deep in ingestion need not know whether
    they are the seed or the references."""
    global _stage
    _stage = stage


def current_stage() -> Optional[str]:
    return _stage


def start_run(kind: str, inputs: Optional[dict[str, Any]] = None) -> str:
    """Create the run, name it `current`, bind it in this process.

    Raises RunActive when this process already has a run bound, or when
    `current` names a run that is still running in a live process. A
    `running` run whose process is gone is marked failed first.
    """
    global _bound, _stage
    with _lock:
        if _bound is not None:
            raise RunActive(_bound["run_id"])
        cur = current_run_id()
        cur_meta = read_meta(cur) if cur else None
        if cur_meta and cur_meta.get("status") == "running":
            if is_pid_alive(cur_meta.get("pid")):
                raise RunActive(cur)
            cur_meta.update(status="failed", error="process died", finished_at=_now_iso())
            _write_meta(cur_meta)

        run_id = new_run_id(kind)
        os.makedirs(run_dir(run_id), exist_ok=True)
        meta = {
            "run_id": run_id, "kind": kind, "inputs": dict(inputs or {}),
            "status": "running", "pid": os.getpid(),
            "started_at": _now_iso(), "finished_at": "", "summary": "", "error": "",
        }
        _write_meta(meta)
        with open(_current_path(), "w", encoding="utf-8") as fh:
            fh.write(run_id + "\n")
        fh = open(_events_path(run_id), "a", encoding="utf-8")
        _bound = {
            "run_id": run_id, "kind": kind, "seq": 0, "owner_pid": os.getpid(),
            "fh": fh, "started": time.time(), "summary": "", "stop_reason": None,
        }
        _stage = None

    try:
        from research_assistant.shared import pipeline_status

        pipeline_status.set_status(run_id=run_id)
    except Exception:  # noqa: BLE001 — status is a convenience, never a gate
        pass
    try:
        from research_assistant import config

        prune(config.RUNS_KEEP)
    except Exception:  # noqa: BLE001
        pass
    return run_id


def _append_locked(b: dict[str, Any], type_: str, stage: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
    b["seq"] += 1
    ev = {"seq": b["seq"], "ts": _now_iso(), "type": type_, "stage": stage, "payload": payload}
    b["fh"].write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
    b["fh"].flush()
    return ev


def emit(type_: str, stage: Optional[str] = None, **payload: Any) -> Optional[dict[str, Any]]:
    """Append one event. A no-op when no run is bound, and in a forked child
    of the owning process (ingest with workers > 1): the child inherits the
    binding but not the sequence counter, and two writers on one file would
    interleave."""
    b = _bound
    if b is None or os.getpid() != b["owner_pid"]:
        return None
    with _lock:
        if _bound is not b:
            return None
        return _append_locked(b, type_, stage if stage is not None else _stage, dict(payload))


def note(text: str, stage: Optional[str] = None) -> Optional[dict[str, Any]]:
    """A narration line — Marvin's voice at a stage transition. Templated,
    never a model call."""
    return emit("narration", stage=stage, text=text)


def set_summary(summary: str) -> None:
    if _bound is not None:
        _bound["summary"] = summary


def set_stopped(reason: str) -> None:
    """The graph's own early exit (no seed, no references). finish_run()
    with no explicit status then records `stopped` and the reason."""
    if _bound is not None:
        _bound["stop_reason"] = reason


def finish_run(status: Optional[str] = None, summary: str = "", error: str = "") -> Optional[dict[str, Any]]:
    """Close the bound run. Idempotent: a second call returns None.

    With status=None the outcome is `stopped` if set_stopped() was called,
    else `done`. The last event of every run is `finished`.
    """
    global _bound, _stage
    with _lock:
        b = _bound
        if b is None:
            return None
        if status is None:
            status = "stopped" if b["stop_reason"] else "done"
        if status not in STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        summary = summary or (b["stop_reason"] if status == "stopped" else "") or b["summary"]
        elapsed = round(time.time() - b["started"], 1)
        _append_locked(b, "finished", STAGE_END,
                       {"status": status, "elapsed_s": elapsed, "summary": summary, "error": error})
        try:
            b["fh"].close()
        except OSError:
            pass
        meta = read_meta(b["run_id"]) or {"run_id": b["run_id"], "kind": b["kind"]}
        meta.update(status=status, summary=summary, error=error, finished_at=_now_iso())
        _write_meta(meta)
        _bound = None
        _stage = None
    return meta


@contextlib.contextmanager
def run(kind: str, inputs: Optional[dict[str, Any]] = None):
    """Wrap an entry point's body in a run.

    Joins the run already bound in this process (the app's launcher starts
    the run, then calls the entry point on a thread), otherwise starts one
    and finishes it with the status the body's exit implies:
    RunCancelled / KeyboardInterrupt → cancelled, any other exception →
    failed, a clean exit → done (or stopped, see set_stopped()).
    """
    if _bound is not None and os.getpid() == _bound["owner_pid"]:
        yield _bound["run_id"]
        return
    run_id = start_run(kind, inputs)
    try:
        yield run_id
    except RunCancelled:
        finish_run("cancelled", summary="stopped by the reader")
        raise
    except KeyboardInterrupt:
        finish_run("cancelled", summary="interrupted")
        raise
    except BaseException as exc:
        finish_run("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        finish_run()


# ─── Cancellation ────────────────────────────────────────────────────────────


def cancel(run_id: str) -> None:
    """Ask the run to stop at its next item boundary."""
    os.makedirs(run_dir(run_id), exist_ok=True)
    with open(_cancel_path(run_id), "w", encoding="utf-8") as fh:
        fh.write(_now_iso() + "\n")


def cancel_requested(run_id: str) -> bool:
    return os.path.exists(_cancel_path(run_id))


def check_cancelled() -> None:
    """Raise RunCancelled if the bound run has been asked to stop. Called at
    item boundaries — between papers, sentences, citations, graph nodes —
    never inside a model call."""
    b = _bound
    if b is not None and os.getpid() == b["owner_pid"] and cancel_requested(b["run_id"]):
        raise RunCancelled(b["run_id"])


# ─── Housekeeping ────────────────────────────────────────────────────────────


def prune(keep: int) -> int:
    """Delete the oldest finished runs beyond `keep`. Never touches a running
    run or the one bound in this process. Returns how many were removed."""
    if keep < 0:
        return 0
    finished = [m for m in list_runs(limit=10_000) if m.get("status") in FINAL_STATUSES]
    doomed = finished[keep:]
    removed = 0
    for meta in doomed:
        if _bound is not None and meta.get("run_id") == _bound["run_id"]:
            continue
        shutil.rmtree(run_dir(meta["run_id"]), ignore_errors=True)
        removed += 1
    return removed


def _reset_for_tests() -> None:
    global _bound, _stage
    with _lock:
        if _bound is not None:
            try:
                _bound["fh"].close()
            except OSError:
                pass
        _bound = None
        _stage = None
```

- [ ] **Step 6: Run the tests**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_runlog.py
```

Expected: 27 passed

- [ ] **Step 7: Run the whole suite** — the conftest changes where every test's status file lives.

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/
```

Expected: everything that passed before still passes

- [ ] **Step 8: Commit**

```bash
git add research_assistant/config.py research_assistant/shared/runlog.py tests/conftest.py tests/test_runlog.py
git commit -m "feat(runlog): the run transcript — data/runs/<id>/run.json + events.jsonl, one bound run per process, cooperative cancel

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `pipeline_status` knows the transcript's stage and a stop from an error

**Files:**
- Modify: `research_assistant/shared/pipeline_status.py`
- Test: `tests/test_pipeline_status.py`

**Interfaces:**
- Consumes: `runlog.set_stage`, `runlog.RunCancelled` (Task 1).
- Produces: every `pipeline_status.track_stage(stage, …)` sets `runlog`'s current stage on entry, so `emit()` calls inside ingestion (which serves both `ingest_seed` and `ingest_refs`) land in the right section without being told. `RunCancelled` inside a stage records "⚠️ … stopped by the reader" and goes idle, like SIGINT.

The diff to the test file also drops the call to `app._render_tab1_live_status` from one widget test: that box is deleted in Task 12 (a session that opens Tab 1 during a run sees the run instead). Removing the call now is harmless.

- [ ] **Step 1: Add the failing tests**

**Apply this change to `tests/test_pipeline_status.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_pipeline_status.py
+++ b/tests/test_pipeline_status.py
@@ -480,11 +480,11 @@
     monkeypatch.setattr(app.st, "info", lambda text, **kw: None)
     monkeypatch.setattr(app.st, "progress", lambda pct, text="": progress_calls.append((pct, text)))
 
+    # The Tab 1 status box is gone: a session that opens Tab 1 during a run
+    # sees the run's transcript (spec 2026-09-12-narrated-runs §2F).
     render_sidebar = getattr(app._render_sidebar_pipeline_status, "__wrapped__", app._render_sidebar_pipeline_status)
-    render_tab1 = getattr(app._render_tab1_live_status, "__wrapped__", app._render_tab1_live_status)
 
     render_sidebar()
-    render_tab1()
     assert len(progress_calls) >= 1
 
 
@@ -796,3 +796,28 @@
 
 
 
+
+
+# ─── Narrated runs (spec 2026-09-12-narrated-runs §2A, §2C) ─────────────────
+
+from research_assistant.shared import runlog  # noqa: E402
+
+
+def test_track_stage_sets_the_transcript_stage(temp_status_file):
+    runlog._reset_for_tests()
+    with pipeline_status.track_stage("fetch"):
+        assert runlog.current_stage() == "fetch"
+    # Not reset on exit: the next stage sets its own.
+    assert runlog.current_stage() == "fetch"
+    runlog.set_stage(None)
+
+
+def test_track_stage_treats_run_cancelled_as_a_stop_not_an_error(temp_status_file):
+    with pytest.raises(runlog.RunCancelled):
+        with pipeline_status.track_stage("ingest_refs"):
+            raise runlog.RunCancelled("r1")
+    status = pipeline_status.get_status()
+    assert status["active"] is False
+    assert status["detail"] == "Stopped in Ingesting and summarizing papers"
+    assert status["recent_events"][-1].startswith("⚠️")
+    assert "stopped by the reader" in status["recent_events"][-1]
```

- [ ] **Step 2: Run them to see the two new ones fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_pipeline_status.py -k 'transcript_stage or run_cancelled'
```

Expected: 2 failed — `current_stage()` is None; the error event says ❌

- [ ] **Step 3: Apply the change**

**Apply this change to `research_assistant/shared/pipeline_status.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/shared/pipeline_status.py
+++ b/research_assistant/shared/pipeline_status.py
@@ -523,6 +523,13 @@
     eff_step = current_step if current_step is not None else meta_step
     eff_label = stage_label or meta_label
 
+    # The transcript's default stage for every emit() until the next stage
+    # starts — so ingestion, which serves both ingest_seed and ingest_refs,
+    # never has to be told which one it is in.
+    from research_assistant.shared import runlog
+
+    runlog.set_stage(stage)
+
     set_status(
         active=True,
         stage=stage,
@@ -555,6 +562,9 @@
         if isinstance(exc, KeyboardInterrupt):
             err_msg = f"⚠️ {eff_label} cancelled by user (SIGINT)"
             detail_msg = f"Cancelled in {eff_label}"
+        elif isinstance(exc, runlog.RunCancelled):
+            err_msg = f"⚠️ {eff_label} stopped by the reader"
+            detail_msg = f"Stopped in {eff_label}"
         elif isinstance(exc, SystemExit):
             err_msg = f"⚠️ {eff_label} stopped (SystemExit)"
             detail_msg = f"Stopped in {eff_label}"
```

- [ ] **Step 4: Run the file**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_pipeline_status.py
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/pipeline_status.py tests/test_pipeline_status.py
git commit -m "feat(status): track_stage sets the transcript stage; RunCancelled is a stop, not an error

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `narration` and `transcript` — the voice and the pure model of a run

**Files:**
- Create: `research_assistant/shared/narration.py`
- Create: `research_assistant/shared/transcript.py`
- Test: `tests/test_narration.py`, `tests/test_transcript.py`

**Interfaces:**
- Consumes: nothing but the event shapes of spec §2D.
- Produces: `narration.*` — one function per line (`plural`, `brief_start`, `brief_fallback`, `uploaded_seed`, `url_seed`, `searching`, `seed_found(title, via_query=None, original=None)`, `seed_missing(n_tried)`, `references(n, with_doi)`, `fetch_done(fetched, unavailable)`, `reading(n, skipped=0)`, `summarising(n)`, `synthesis()`, `cancelled(parsed_unsaved=0)`, `citing(n_sentences, n_eligible)`, `verifying(n)`, `auditing(n_claims, n_to_judge)`). `transcript.*` — `SECTION_ORDER`, `SECTION_TITLES`, `STAGE_SECTION`, `section_of(ev)`, `sections(events)`, `of_type`, `first_payload`, `last_payload`, `paper_cards(events, stage=None)`, `counts(events)`, `elapsed_label(seconds)`, `status_line(meta, events, live=True)`, `fetch_lists`, `sentence_rows`, `verdict_rows`, `verdict_tiles`, `audit_rows`, `audit_tiles`. `tests/test_transcript.PIPELINE` and `ev()` are reused by Task 11's tests.

The section order is `brief → seed (discover, ingest_seed) → references (extract, fetch) → reading (ingest_refs) → synthesis (respond) → audit → cite → verify → end`. `status_line` counts only events of the *current* stage, so the seed's `paper_read` never advances "Reading paper k of n".

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_narration.py` with exactly this content:**

```python
"""narration — every template with 0, 1 and n (spec 2026-09-12-narrated-runs §2D)."""

import unittest

from research_assistant.shared import narration as nr


class TestPlural(unittest.TestCase):
    def test_plural(self):
        self.assertEqual(nr.plural(1, "paper"), "1 paper")
        self.assertEqual(nr.plural(2, "paper"), "2 papers")
        self.assertEqual(nr.plural(0, "paper"), "0 papers")
        self.assertEqual(nr.plural(1, "paper was", "papers were"), "1 paper was")


class TestTemplates(unittest.TestCase):
    def test_seed_lines(self):
        self.assertIn("*MoS2 networks*", nr.seed_found("MoS2 networks"))
        self.assertNotIn("phrasing", nr.seed_found("T", via_query="q", original="q"))
        self.assertIn("“alt” did", nr.seed_found("T", via_query="alt", original="q"))
        self.assertIn("3 phrasings", nr.seed_missing(3))
        self.assertIn("1 phrasing", nr.seed_missing(1))
        self.assertIn("*T*", nr.uploaded_seed("T"))
        self.assertTrue(nr.url_seed("http://x").startswith("You gave me a link"))
        self.assertIn("“q”", nr.searching("q"))

    def test_references(self):
        self.assertIn("62 references, 40 with a DOI", nr.references(62, 40))
        self.assertIn("1 reference,", nr.references(1, 1))
        self.assertTrue(nr.references(0, 0).startswith("No reference list"))

    def test_fetch_done_every_branch(self):
        self.assertIn("31 fetched, 31 out of reach", nr.fetch_done(31, 31))
        self.assertIn("All 5 papers fetched", nr.fetch_done(5, 0))
        self.assertIn("None of the 4", nr.fetch_done(0, 4))
        self.assertTrue(nr.fetch_done(0, 0).startswith("Nothing new"))

    def test_reading(self):
        self.assertIn("Reading 31 papers. 4 were already on the shelf.", nr.reading(31, 4))
        self.assertNotIn("shelf", nr.reading(2))
        self.assertIn("All 3 papers already on the shelf", nr.reading(0, 3))

    def test_rest(self):
        self.assertIn("each of the 31", nr.summarising(31))
        self.assertEqual(nr.synthesis(), "Now to say something about all of this.")
        self.assertEqual(nr.cancelled(), "Stopped, as asked.")
        self.assertIn("6 papers were parsed but not saved", nr.cancelled(6))
        self.assertIn("1 paper was parsed", nr.cancelled(1))
        self.assertIn("12 sentences, 9 long enough", nr.citing(12, 9))
        self.assertIn("1 citation to check", nr.verifying(1))
        self.assertEqual(nr.brief_fallback(), "I'll search for it as you wrote it.")
        self.assertEqual(nr.brief_start(), "Reading the idea before I go looking.")
        self.assertIn("45 in-text citations in the seed; 12 cite papers I have", nr.auditing(45, 12))
        self.assertIn("none citing a paper I have", nr.auditing(3, 0))


if __name__ == "__main__":
    unittest.main()
```

**Create `tests/test_transcript.py` with exactly this content:**

```python
"""transcript — the pure model behind the feed (spec 2026-09-12-narrated-runs §2F)."""

import unittest

from research_assistant.shared import transcript as tr


def ev(seq, type_, stage, **payload):
    return {"seq": seq, "ts": "2026-09-12T10:00:00Z", "type": type_, "stage": stage, "payload": payload}


PIPELINE = [
    ev(1, "brief", "brief", text="You want X.", alternates=["a", "b", "c"]),
    ev(2, "search_tried", "discover", provider="arxiv", query="q", results=5, with_pdf=3),
    ev(3, "seed_found", "discover", title="Seed paper", key="k0", source="search", via_query="q"),
    ev(4, "narration", "discover", text="Seed: *Seed paper*."),
    ev(5, "paper_read", "ingest_seed", document="k0", title="Seed paper", chunks=40, captions=2, described=0, extraction="grobid"),
    ev(6, "paper_summarised", "ingest_seed", document="k0", citation="Seed paper", summary="It establishes…"),
    ev(7, "references_extracted", "extract", n=3, with_doi=2, by_method={"grobid": 1}, sample=[]),
    ev(8, "fetch_started", "fetch", remaining=3, already=0),
    ev(9, "paper_fetched", "fetch", key="k1", title="P1", provider="arxiv", doi=None),
    ev(10, "paper_unavailable", "fetch", key="k2", title="P2", reason="paywalled"),
    ev(11, "paper_fetched", "fetch", key="k3", title="P3", provider="unpaywall", doi="10.1/x"),
    ev(12, "batch_started", "ingest_refs", n=2, skipped=0),
    ev(13, "paper_read", "ingest_refs", document="k1", title="P1", chunks=30, captions=1, described=1, extraction="grobid"),
    ev(14, "figure_described", "ingest_refs", document="k1", label="Figure 1", kind="figure", description="A plot.", image_path="k1_fig0.png"),
    ev(15, "paper_empty", "ingest_refs", document="k3", title="P3"),
    ev(16, "summaries_started", "ingest_refs", n=1),
    ev(17, "paper_summarised", "ingest_refs", document="k1", citation="P1", summary="P1 shows…"),
    ev(18, "batch_saved", "ingest_refs", processed=2, inserted=31, described=1),
    ev(19, "shortlist", "respond", papers=[{"key": "P1", "title": "P1", "score": 0.9}, {"key": "P2", "title": "Seed paper", "score": 0.8}]),
    ev(20, "notes", "respond", key="P1", title="P1", notes="Establishes…"),
]


class TestSections(unittest.TestCase):
    def test_order_titles_and_latest(self):
        secs = tr.sections(PIPELINE)
        self.assertEqual([s["key"] for s in secs], ["brief", "seed", "references", "reading", "synthesis"])
        self.assertEqual(secs[1]["title"], "The seed")
        self.assertEqual([s["latest"] for s in secs], [False, False, False, False, True])
        self.assertEqual(len(secs[2]["events"]), 5)       # extract + fetch merged

    def test_end_is_last_and_never_latest(self):
        events = PIPELINE + [ev(21, "finished", "end", status="done", elapsed_s=12.0, summary="")]
        secs = tr.sections(events)
        self.assertEqual(secs[-1]["key"], "end")
        self.assertEqual([s["key"] for s in secs if s["latest"]], ["synthesis"])

    def test_unknown_stage_becomes_its_own_section_after_the_known_ones(self):
        events = [ev(1, "x", "mystery"), ev(2, "y", "brief")]
        self.assertEqual([s["key"] for s in tr.sections(events)], ["brief", "mystery"])
        self.assertEqual(tr.sections(events)[1]["title"], "Mystery")

    def test_stageless_event_lands_in_end(self):
        self.assertEqual(tr.section_of({"type": "x"}), "end")
        self.assertEqual(tr.sections([]), [])


class TestPaperCards(unittest.TestCase):
    def test_cards_merge_read_figure_and_summary(self):
        cards = tr.paper_cards(PIPELINE, stage="ingest_refs")
        self.assertEqual([c["document"] for c in cards], ["k1", "k3"])
        p1 = cards[0]
        self.assertEqual((p1["title"], p1["chunks"], p1["captions"], p1["described"], p1["extraction"]),
                         ("P1", 30, 1, 1, "grobid"))
        self.assertEqual(p1["summary"], "P1 shows…")
        self.assertEqual(p1["figures"][0]["label"], "Figure 1")
        self.assertEqual(p1["figures"][0]["image_path"], "k1_fig0.png")
        self.assertTrue(cards[1]["empty"])
        self.assertIsNone(cards[1]["summary"])

    def test_seed_card_is_separate_by_stage(self):
        seed = tr.paper_cards(PIPELINE, stage="ingest_seed")
        self.assertEqual(len(seed), 1)
        self.assertEqual(seed[0]["summary"], "It establishes…")
        self.assertEqual(len(tr.paper_cards(PIPELINE)), 3)   # no filter: all documents

    def test_summary_before_read_still_makes_a_card(self):
        events = [ev(1, "paper_summarised", "ingest_refs", document="d", citation="D title", summary="s")]
        card = tr.paper_cards(events)[0]
        self.assertEqual((card["title"], card["summary"], card["chunks"]), ("D title", "s", 0))

    def test_events_without_document_are_ignored(self):
        self.assertEqual(tr.paper_cards([ev(1, "paper_read", "ingest_refs", title="no key")]), [])


class TestCounts(unittest.TestCase):
    def test_counts(self):
        n = tr.counts(PIPELINE)
        self.assertEqual(n["references"], 3)
        self.assertEqual((n["to_fetch"], n["fetched"], n["unavailable"]), (3, 2, 1))
        self.assertEqual((n["to_read"], n["read"], n["empty"], n["described"]), (2, 2, 1, 1))
        self.assertEqual((n["to_summarise"], n["summarised"]), (1, 2))
        self.assertEqual((n["shortlisted"], n["notes"]), (2, 1))

    def test_fetch_lists(self):
        fetched, unavailable = tr.fetch_lists(PIPELINE)
        self.assertEqual([f["title"] for f in fetched], ["P1", "P3"])
        self.assertEqual(unavailable[0]["reason"], "paywalled")


class TestStatusLine(unittest.TestCase):
    def line(self, upto, **meta):
        m = {"status": "running", "pid": 1, **meta}
        return tr.status_line(m, PIPELINE[:upto])

    def test_final_statuses(self):
        self.assertEqual(tr.status_line(None, []), "Idle")
        self.assertEqual(tr.status_line({"status": "done", "summary": "4 papers"}, []), "Done — 4 papers")
        self.assertEqual(tr.status_line({"status": "stopped", "summary": "no seed"}, []), "Stopped — no seed")
        self.assertEqual(tr.status_line({"status": "cancelled"}, []), "Stopped by you")
        self.assertEqual(tr.status_line({"status": "failed", "error": "ValueError: x"}, []), "Failed — ValueError: x")
        self.assertEqual(tr.status_line({"status": "running"}, [], live=False),
                         "Died — the process ended without finishing")

    def test_running_by_stage(self):
        self.assertEqual(self.line(0), "Starting")
        self.assertEqual(self.line(1), "Thinking about the idea")
        self.assertEqual(self.line(2), "Looking for a seed paper")
        self.assertEqual(self.line(5), "Reading the seed paper")
        self.assertEqual(self.line(7), "Mining the reference list")
        self.assertEqual(self.line(8), "Fetching — 0 of 3 checked")
        self.assertEqual(self.line(10), "Fetching — 2 of 3 checked")
        self.assertEqual(self.line(12), "Reading paper 1 of 2")
        self.assertEqual(self.line(13), "Reading paper 2 of 2")
        self.assertEqual(self.line(15), "Reading paper 2 of 2")        # never n+1 of n
        self.assertEqual(self.line(16), "Summarising paper 1 of 1")
        self.assertEqual(self.line(18), "Batch saved")
        self.assertEqual(self.line(19), "Reading paper 1 of 2 for the synthesis")
        self.assertEqual(self.line(20), "Reading paper 2 of 2 for the synthesis")

    def test_cite_and_verify_lines(self):
        cite = [ev(1, "draft_split", "cite", n_sentences=3, n_eligible=2),
                ev(2, "sentence_decided", "cite", index=0, cited=False)]
        self.assertEqual(tr.status_line({"status": "running"}, cite), "Sentence 2 of 3")
        ver = [ev(1, "verify_started", "verify", n=2)]
        self.assertEqual(tr.status_line({"status": "running"}, ver), "Citation 1 of 2")

    def test_elapsed_label(self):
        self.assertEqual(tr.elapsed_label(48), "48 s")
        self.assertEqual(tr.elapsed_label(750), "12 min")
        self.assertEqual(tr.elapsed_label(3900), "1 h 05 min")


class TestAudit(unittest.TestCase):
    EVENTS = [
        ev(1, "audit_started", "audit", claims=3, downloaded=2, to_judge=2, seed="seed.pdf"),
        ev(2, "audit_item", "audit", index=1, outcome="judged", judgement="Supports", downloaded=True, ref_title="A"),
        ev(3, "audit_item", "audit", index=2, outcome="judged", judgement="Contradicts", downloaded=True, ref_title="B"),
        ev(4, "audit_item", "audit", index=3, outcome="not_downloaded", judgement="Not downloaded", downloaded=False),
    ]

    def test_section_status_and_tiles(self):
        self.assertEqual(tr.sections(self.EVENTS)[0]["title"], "Citation audit")
        self.assertEqual(tr.status_line({"status": "running"}, self.EVENTS[:1]), "Auditing citation 1 of 2")
        self.assertEqual(tr.status_line({"status": "running"}, self.EVENTS[:2]), "Auditing citation 2 of 2")
        self.assertEqual(tr.audit_tiles(self.EVENTS),
                         {"found": 3, "supports": 1, "partially": 0, "review": 1, "not_here": 1})
        done = self.EVENTS + [ev(5, "audit_done", "audit", totals={"total": 3, "judged": 2, "Supports": 1,
                                                                    "Contradicts": 1, "not_downloaded": 1}, error=None, path="x")]
        self.assertEqual(tr.status_line({"status": "running"}, done), "Audit done")
        self.assertEqual(tr.audit_tiles(done)["review"], 1)
        self.assertEqual(len(tr.audit_rows(done)), 3)
        self.assertEqual(tr.counts(done)["audit_claims"], 3)

    def test_audit_sits_between_synthesis_and_cite(self):
        events = [ev(1, "synthesis", "respond", text="t"), ev(2, "audit_started", "audit", claims=1, to_judge=0)]
        self.assertEqual([s["key"] for s in tr.sections(events)], ["synthesis", "audit"])


class TestCiteVerifyRows(unittest.TestCase):
    def test_rows_and_tiles_from_rows(self):
        events = [
            ev(1, "verify_started", "verify", n=3),
            ev(2, "citation_judged", "verify", index=0, outcome="judged", judgement="Supports"),
            ev(3, "citation_judged", "verify", index=1, outcome="judged", judgement="Contradicts"),
            ev(4, "citation_judged", "verify", index=2, outcome="unresolved"),
        ]
        self.assertEqual(len(tr.verdict_rows(events)), 3)
        self.assertEqual(tr.verdict_tiles(events),
                         {"checked": 3, "supports": 1, "partially": 0, "review": 1, "not_judged": 1})

    def test_tiles_prefer_verify_done_totals(self):
        events = [ev(1, "verify_done", "verify", totals={"total": 5, "judged": 4, "Supports": 3,
                                                          "Partially supports": 1})]
        self.assertEqual(tr.verdict_tiles(events),
                         {"checked": 5, "supports": 3, "partially": 1, "review": 0, "not_judged": 1})

    def test_sentence_rows(self):
        events = [ev(1, "sentence_decided", "cite", index=0, sentence="S.", cited=True, cited_text="S \\cite{cite_1}.")]
        self.assertEqual(tr.sentence_rows(events)[0]["cited_text"], "S \\cite{cite_1}.")

    def test_first_and_last_payload(self):
        self.assertEqual(tr.first_payload(PIPELINE, "brief")["alternates"], ["a", "b", "c"])
        self.assertEqual(tr.last_payload(PIPELINE, "paper_fetched")["key"], "k3")
        self.assertIsNone(tr.first_payload(PIPELINE, "nope"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_narration.py tests/test_transcript.py
```

Expected: FAIL — modules not found

- [ ] **Step 3: Write the two modules**

**Create `research_assistant/shared/narration.py` with exactly this content:**

```python
"""
narration — Marvin's lines at stage transitions (spec 2026-09-12-narrated-runs §2D).

Every function returns one short string. Templated, never a model call:
the run's narration costs nothing and the voice cannot drift. Dry and
specific; numbers as digits; never cute.
"""

from __future__ import annotations


def plural(n: int, one: str, many: str | None = None) -> str:
    """'1 paper', '2 papers', '0 papers'."""
    many = many if many is not None else one + "s"
    return f"{n} {one if n == 1 else many}"


def brief_start() -> str:
    return "Reading the idea before I go looking."


def brief_fallback() -> str:
    return "I'll search for it as you wrote it."


def uploaded_seed(title: str) -> str:
    return f"You gave me the paper — *{title}*. I'll read it, chase what it cites, and answer from those."


def url_seed(url: str) -> str:
    return f"You gave me a link. I'll fetch it, read it, chase what it cites, and answer from those."


def searching(query: str) -> str:
    return f"Looking for an open-access seed paper on “{query}”."


def seed_found(title: str, via_query: str | None = None, original: str | None = None) -> str:
    line = f"Seed: *{title}*. I'll read it, then see who it cites."
    if via_query and original and via_query != original:
        line += f" Your phrasing found nothing open-access; “{via_query}” did."
    return line


def seed_missing(n_tried: int) -> str:
    return (f"Nothing open-access came up for {plural(n_tried, 'phrasing')}. "
            "Give me an arXiv link or a PDF and I'll start from that.")


def references(n: int, with_doi: int) -> str:
    if n == 0:
        return "No reference list I could read. That ends the chase here."
    return f"{plural(n, 'reference')}, {with_doi} with a DOI I can chase. Let me see how many I can actually get."


def fetch_done(fetched: int, unavailable: int) -> str:
    if fetched == 0 and unavailable == 0:
        return "Nothing new to fetch; everything was already here."
    if unavailable == 0:
        return f"All {plural(fetched, 'paper')} fetched. That never happens."
    if fetched == 0:
        return f"None of the {unavailable} could be fetched — paywalls, mostly. I'll work with the seed alone."
    return f"{fetched} fetched, {unavailable} out of reach — paywalls, mostly. I'll work with what I have."


def reading(n: int, skipped: int = 0) -> str:
    if n == 0:
        return f"All {plural(skipped, 'paper')} already on the shelf. Nothing to read."
    shelf = f" {skipped} were already on the shelf." if skipped else ""
    return f"Reading {plural(n, 'paper')}.{shelf} This is the slow part; I'll show each one as I finish it."


def summarising(n: int) -> str:
    return f"Now a summary of each of the {n}, so you can read while I work."


def synthesis() -> str:
    return "Now to say something about all of this."


def cancelled(parsed_unsaved: int = 0) -> str:
    if parsed_unsaved:
        return (f"Stopped, as asked. {plural(parsed_unsaved, 'paper was', 'papers were')} parsed but not saved; "
                "I'll parse them again next time.")
    return "Stopped, as asked."


def citing(n_sentences: int, n_eligible: int) -> str:
    return (f"{plural(n_sentences, 'sentence')}, {n_eligible} long enough to need a source. "
            "I'll take them one at a time.")


def verifying(n: int) -> str:
    return f"{plural(n, 'citation')} to check against what the paper actually says. One call each."


def auditing(n_claims: int, n_to_judge: int) -> str:
    if n_to_judge == 0:
        return f"{plural(n_claims, 'in-text citation')} in the seed, none citing a paper I have. Nothing to judge."
    return (f"{plural(n_claims, 'in-text citation')} in the seed; {n_to_judge} cite papers I have. "
            "I'll check what each cited paper actually says.")
```

**Create `research_assistant/shared/transcript.py` with exactly this content:**

```python
"""
transcript — the pure model behind the feed (spec 2026-09-12-narrated-runs §2F).

Everything here is a function of (meta, events) as runlog.read_run returns
them, with no Streamlit and no I/O, so the reader can be tested against a
synthetic transcript and re-run on every poll tick without state.

Events arrive with a `stage` (set by pipeline_status.track_stage or passed
explicitly) and the feed groups stages into sections in a fixed order:

    brief → seed (discover, ingest_seed) → references (extract, fetch)
          → reading (ingest_refs) → synthesis (respond) → cite → verify → end
"""

from __future__ import annotations

from typing import Any, Optional

SECTION_ORDER = ["brief", "seed", "references", "reading", "synthesis", "audit", "cite", "verify", "end"]

SECTION_TITLES = {
    "brief": "The brief",
    "seed": "The seed",
    "references": "References",
    "reading": "Reading",
    "synthesis": "The synthesis",
    "audit": "Citation audit",
    "cite": "Citing",
    "verify": "Verifying",
    "end": "The end",
}

STAGE_SECTION = {
    "brief": "brief",
    "discover": "seed",
    "ingest_seed": "seed",
    "extract": "references",
    "fetch": "references",
    "ingest_refs": "reading",
    "respond": "synthesis",
    "audit": "audit",
    "cite": "cite",
    "verify": "verify",
    "end": "end",
}


def _p(ev: dict[str, Any]) -> dict[str, Any]:
    payload = ev.get("payload")
    return payload if isinstance(payload, dict) else {}


def section_of(ev: dict[str, Any]) -> str:
    stage = ev.get("stage")
    if stage in STAGE_SECTION:
        return STAGE_SECTION[stage]
    return stage or "end"


def sections(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-empty sections in feed order; unknown stages follow, in first-seen
    order, titled by their stage. `latest` marks the last section that is
    not `end` — the one the reader opens."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_key.setdefault(section_of(ev), []).append(ev)
    keys = [k for k in SECTION_ORDER if k in by_key] + [k for k in by_key if k not in SECTION_ORDER]
    if "end" in keys:                       # always last
        keys = [k for k in keys if k != "end"] + ["end"]
    latest = next((k for k in reversed(keys) if k != "end"), None)
    return [
        {"key": k, "title": SECTION_TITLES.get(k, k.replace("_", " ").capitalize()),
         "events": by_key[k], "latest": k == latest}
        for k in keys
    ]


def of_type(events: list[dict[str, Any]], *types: str) -> list[dict[str, Any]]:
    return [ev for ev in events if ev.get("type") in types]


def first_payload(events: list[dict[str, Any]], type_: str) -> Optional[dict[str, Any]]:
    for ev in events:
        if ev.get("type") == type_:
            return _p(ev)
    return None


def last_payload(events: list[dict[str, Any]], type_: str) -> Optional[dict[str, Any]]:
    for ev in reversed(events):
        if ev.get("type") == type_:
            return _p(ev)
    return None


def paper_cards(events: list[dict[str, Any]], stage: Optional[str] = None) -> list[dict[str, Any]]:
    """One card per document, in first-seen order, merged from paper_read,
    paper_empty, figure_described and paper_summarised. `stage` restricts
    to events of that stage (the seed's card vs the references')."""
    cards: dict[str, dict[str, Any]] = {}

    def card(doc: str, title: Optional[str] = None) -> dict[str, Any]:
        c = cards.get(doc)
        if c is None:
            c = cards[doc] = {
                "document": doc, "title": title or doc, "chunks": 0, "captions": 0,
                "described": 0, "extraction": "", "summary": None, "figures": [], "empty": False,
            }
        elif title and c["title"] == doc:
            c["title"] = title
        return c

    for ev in events:
        if stage is not None and ev.get("stage") != stage:
            continue
        t, p = ev.get("type"), _p(ev)
        doc = p.get("document")
        if not doc:
            continue
        if t == "paper_read":
            c = card(doc, p.get("title"))
            c.update(chunks=int(p.get("chunks") or 0), captions=int(p.get("captions") or 0),
                     described=int(p.get("described") or 0), extraction=p.get("extraction") or "")
        elif t == "paper_empty":
            card(doc, p.get("title"))["empty"] = True
        elif t == "figure_described":
            card(doc)["figures"].append({
                "label": p.get("label") or "", "kind": p.get("kind") or "",
                "description": p.get("description") or "", "image_path": p.get("image_path") or "",
            })
        elif t == "paper_summarised":
            card(doc, p.get("citation"))["summary"] = p.get("summary") or ""
    return list(cards.values())


def counts(events: list[dict[str, Any]]) -> dict[str, int]:
    n = {
        "references": 0, "fetched": 0, "unavailable": 0, "read": 0, "empty": 0,
        "summarised": 0, "described": 0, "skipped": 0, "to_read": 0, "to_fetch": 0,
        "to_summarise": 0, "sentences": 0, "decided": 0, "cited": 0,
        "citations": 0, "judged": 0, "notes": 0, "shortlisted": 0,
        "audit_claims": 0, "audit_to_judge": 0, "audit_items": 0,
    }
    for ev in events:
        t, p = ev.get("type"), _p(ev)
        if t == "references_extracted":
            n["references"] = int(p.get("n") or 0)
        elif t == "fetch_started":
            n["to_fetch"] = int(p.get("remaining") or 0)
        elif t == "paper_fetched":
            n["fetched"] += 1
        elif t == "paper_unavailable":
            n["unavailable"] += 1
        elif t == "batch_started":
            n["to_read"] += int(p.get("n") or 0)
            n["skipped"] += int(p.get("skipped") or 0)
        elif t == "paper_read":
            n["read"] += 1
        elif t == "paper_empty":
            n["empty"] += 1
        elif t == "figure_described":
            n["described"] += 1
        elif t == "summaries_started":
            n["to_summarise"] += int(p.get("n") or 0)
        elif t == "paper_summarised":
            n["summarised"] += 1
        elif t == "draft_split":
            n["sentences"] = int(p.get("n_sentences") or 0)
        elif t == "sentence_decided":
            n["decided"] += 1
            n["cited"] += 1 if p.get("cited") else 0
        elif t == "verify_started":
            n["citations"] = int(p.get("n") or 0)
        elif t == "citation_judged":
            n["judged"] += 1
        elif t == "shortlist":
            n["shortlisted"] = len(p.get("papers") or [])
        elif t == "notes":
            n["notes"] += 1
        elif t == "audit_started":
            n["audit_claims"] = int(p.get("claims") or 0)
            n["audit_to_judge"] = int(p.get("to_judge") or 0)
        elif t == "audit_item":
            n["audit_items"] += 1
    return n


def elapsed_label(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def status_line(meta: Optional[dict[str, Any]], events: list[dict[str, Any]], live: bool = True) -> str:
    """One line for the header and the sidebar."""
    if not meta:
        return "Idle"
    status = meta.get("status")
    if status == "done":
        return "Done" + (f" — {meta['summary']}" if meta.get("summary") else "")
    if status == "stopped":
        return "Stopped" + (f" — {meta['summary']}" if meta.get("summary") else "")
    if status == "cancelled":
        return "Stopped by you"
    if status == "failed":
        return "Failed" + (f" — {meta['error']}" if meta.get("error") else "")
    if not live:
        return "Died — the process ended without finishing"

    last = events[-1] if events else None
    stage = (last or {}).get("stage")
    ltype = (last or {}).get("type")
    # Counters of the current stage only: the seed's paper_read (ingest_seed)
    # must not advance "Reading paper k of n" (ingest_refs).
    n = counts([e for e in events if e.get("stage") == stage])
    if stage == "brief":
        return "Thinking about the idea"
    if stage == "discover":
        return "Looking for a seed paper"
    if stage == "ingest_seed":
        return "Reading the seed paper"
    if stage == "extract":
        return "Mining the reference list"
    if stage == "fetch":
        checked = n["fetched"] + n["unavailable"]
        return f"Fetching — {checked} of {n['to_fetch']} checked" if n["to_fetch"] else "Fetching"
    if stage == "ingest_refs":
        if ltype == "batch_saved":
            return "Batch saved"
        if ltype in ("summaries_started", "paper_summarised"):
            return f"Summarising paper {min(n['summarised'] + 1, n['to_summarise'])} of {n['to_summarise']}"
        if n["to_read"]:
            return f"Reading paper {min(n['read'] + n['empty'] + 1, n['to_read'])} of {n['to_read']}"
        return "Reading"
    if stage == "respond":
        if ltype in ("shortlist", "notes") and n["shortlisted"]:
            return f"Reading paper {min(n['notes'] + 1, n['shortlisted'])} of {n['shortlisted']} for the synthesis"
        return "Writing the synthesis"
    if stage == "audit":
        if ltype == "audit_done":
            return "Audit done"
        judged = min(n["audit_items"] + 1, n["audit_to_judge"])
        return f"Auditing citation {judged} of {n['audit_to_judge']}" if n["audit_to_judge"] else "Auditing the seed's citations"
    if stage == "cite":
        return f"Sentence {min(n['decided'] + 1, n['sentences'])} of {n['sentences']}" if n["sentences"] else "Citing"
    if stage == "verify":
        return f"Citation {min(n['judged'] + 1, n['citations'])} of {n['citations']}" if n["citations"] else "Verifying"
    return "Starting"


def fetch_lists(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fetched = [_p(ev) for ev in events if ev.get("type") == "paper_fetched"]
    unavailable = [_p(ev) for ev in events if ev.get("type") == "paper_unavailable"]
    return fetched, unavailable


def sentence_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_p(ev) for ev in events if ev.get("type") == "sentence_decided"]


def verdict_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_p(ev) for ev in events if ev.get("type") == "citation_judged"]


def audit_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_p(ev) for ev in events if ev.get("type") == "audit_item"]


def audit_tiles(events: list[dict[str, Any]]) -> dict[str, int]:
    """Tab 1's audit tiles: found, supports, partially, need review, not here."""
    done = last_payload(events, "audit_done")
    if done and isinstance(done.get("totals"), dict) and done["totals"].get("total"):
        t = done["totals"]
        return {"found": t.get("total", 0), "supports": t.get("Supports", 0),
                "partially": t.get("Partially supports", 0),
                "review": t.get("Contradicts", 0) + t.get("Does not support", 0),
                "not_here": t.get("not_downloaded", 0)}
    rows = audit_rows(events)
    return {
        "found": len(rows),
        "supports": sum(1 for r in rows if r.get("judgement") == "Supports"),
        "partially": sum(1 for r in rows if r.get("judgement") == "Partially supports"),
        "review": sum(1 for r in rows if r.get("judgement") in ("Contradicts", "Does not support")),
        "not_here": sum(1 for r in rows if r.get("outcome") == "not_downloaded"),
    }


def verdict_tiles(events: list[dict[str, Any]]) -> dict[str, int]:
    """The five Tab 3 tiles, from rows as they land or from verify_done."""
    done = last_payload(events, "verify_done")
    if done and isinstance(done.get("totals"), dict):
        t = done["totals"]
        review = sum(t.get(j, 0) for j in ("Contradicts", "Does not support", "Unclear / insufficient evidence"))
        return {"checked": t.get("total", 0), "supports": t.get("Supports", 0),
                "partially": t.get("Partially supports", 0), "review": review,
                "not_judged": t.get("total", 0) - t.get("judged", 0)}
    rows = verdict_rows(events)
    judged = [r for r in rows if r.get("outcome") == "judged"]
    return {
        "checked": len(rows),
        "supports": sum(1 for r in judged if r.get("judgement") == "Supports"),
        "partially": sum(1 for r in judged if r.get("judgement") == "Partially supports"),
        "review": sum(1 for r in judged if r.get("judgement") in
                      ("Contradicts", "Does not support", "Unclear / insufficient evidence")),
        "not_judged": len(rows) - len(judged),
    }
```

- [ ] **Step 4: Run the tests**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_narration.py tests/test_transcript.py
```

Expected: 26 passed

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/narration.py research_assistant/shared/transcript.py tests/test_narration.py tests/test_transcript.py
git commit -m "feat(transcript): the pure model behind the feed — sections, paper cards, counts, status line — and Marvin's templated narration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The brief, and Agent 0's fallback phrasings

**Files:**
- Modify: `research_assistant/prompts.py` (append `RESEARCH_BRIEF_USER`)
- Create: `research_assistant/shared/brief.py`
- Modify: `research_assistant/agents/agent0_discoverer.py`
- Test: `tests/test_brief.py`, `tests/test_discoverer.py`

**Interfaces:**
- Consumes: `runlog.emit/note` (Task 1), `narration` (Task 3), `llm.chat(messages, temperature=, options=)`, `config.BRIEF_TEMPERATURE`, `config.CHAT_OLLAMA_OPTIONS`.
- Produces: `brief.parse_brief(text, query="") -> list[str]` (≤ 3, cleaned, query and duplicates dropped), `brief.write_brief(query) -> {"text", "alternates"} | None`. `agent0_discoverer.find_and_fetch_seed(query, force=False, limit=SEARCH_LIMIT, alternates=())` walks the providers for the typed query, then for each alternate only if nothing downloaded; the returned candidate carries `via_query`. `agent0_discoverer.discover(query, force=False, alternates=())` emits `search_tried {provider, query, results, with_pdf[, error]}` per provider per phrasing, then `seed_found {title, key, doi, arxiv_id, url, source, via_query, cached, path}` or `seed_missing {query, tried}`, with narration. The seed manifest stays keyed by the typed query and carries no `via_query` (`SeedPaper` rejects unknown fields).

`discover_from_url` / `discover_from_file` do not emit; the orchestrator narrates those paths (Task 7), because only the search path knows which phrasing worked.

- [ ] **Step 1: Append the prompt**

**Apply this change to `research_assistant/prompts.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/prompts.py
+++ b/research_assistant/prompts.py
@@ -128,3 +128,23 @@
     "one or two sentences on where this idea might still add something. Use "
     "only the provided sources.\n\n{context}"
 )
+
+
+# ─── The brief (narrated runs — before a seed search) ───────────────────────
+# One call, shown to the reader while the providers are queried. The three
+# numbered lines are parsed by shared.brief.parse_brief and handed to Agent 0
+# as fallback phrasings, so the format of those lines is load-bearing.
+RESEARCH_BRIEF_USER = (
+    "A researcher typed this research idea:\n\"{query}\"\n\n"
+    "Before you search the literature, tell them — in plain prose, at most "
+    "150 words, no headings, no bullet points other than the numbered lines "
+    "asked for:\n"
+    "First, in two sentences, what you understand the idea to be about: the "
+    "system, the phenomenon, the question.\n"
+    "Then three alternative phrasings you would search the literature with, "
+    "written as exactly three numbered lines `1. …`, `2. …`, `3. …`, each a "
+    "short search query of at most twelve words, each different from the "
+    "original wording and from each other.\n"
+    "Last, in one sentence, what a good seed paper would look like: the kind "
+    "of paper whose reference list would cover this idea."
+)
```

- [ ] **Step 2: Write the failing tests for the brief**

**Create `tests/test_brief.py` with exactly this content:**

```python
"""brief — parse the numbered phrasings; a failed call is not an error
(spec 2026-09-12-narrated-runs §2E)."""

import unittest
from unittest.mock import MagicMock, patch

from research_assistant.shared import brief


SAMPLE = """You are asking whether disorder in a one-dimensional wire localises every state.
This is Anderson localisation in quasi-1D geometry, and the question is what transport survives.

1. "Anderson localization quasi-one-dimensional wires"
2. localization length disordered quantum wire conductance
3) **transport in disordered 1D wires**

A good seed would be a review of localisation in low-dimensional disordered conductors."""


class TestParseBrief(unittest.TestCase):
    def test_numbered_lines_cleaned_in_order(self):
        self.assertEqual(brief.parse_brief(SAMPLE), [
            "Anderson localization quasi-one-dimensional wires",
            "localization length disordered quantum wire conductance",
            "transport in disordered 1D wires",
        ])

    def test_drops_the_query_itself_and_duplicates_and_caps_at_three(self):
        text = "1. Same Query\n2. other\n3. Other\n4. fourth\n5. fifth"
        self.assertEqual(brief.parse_brief(text, query="same query"), ["other", "fourth", "fifth"])

    def test_no_numbered_lines(self):
        self.assertEqual(brief.parse_brief("Just prose, no list."), [])
        self.assertEqual(brief.parse_brief(""), [])
        self.assertEqual(brief.parse_brief(None), [])

    def test_trailing_stop_and_parenthesised_numbers(self):
        self.assertEqual(brief.parse_brief("(1) hopping transport MoS2 networks."), ["hopping transport MoS2 networks"])


class TestWriteBrief(unittest.TestCase):
    def test_returns_text_and_alternates(self):
        with patch("research_assistant.shared.llm.chat", return_value=MagicMock(content=SAMPLE)) as chat:
            out = brief.write_brief("disordered wires")
        self.assertEqual(out["text"], SAMPLE)
        self.assertEqual(len(out["alternates"]), 3)
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["temperature"], brief.BRIEF_TEMPERATURE)
        self.assertEqual(kwargs["options"], brief.CHAT_OLLAMA_OPTIONS)
        self.assertIn("disordered wires", chat.call_args.args[0][0]["content"])

    def test_failure_and_empty_are_none(self):
        with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("down")):
            self.assertIsNone(brief.write_brief("q"))
        with patch("research_assistant.shared.llm.chat", return_value=MagicMock(content="   ")):
            self.assertIsNone(brief.write_brief("q"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_brief.py
```

Expected: FAIL — module not found

- [ ] **Step 4: Write `brief.py`**

**Create `research_assistant/shared/brief.py` with exactly this content:**

```python
"""
brief — Marvin's reading of a research idea before the search
(spec 2026-09-12-narrated-runs §2E).

One model call. The text is shown to the reader; the three numbered lines
in it are the alternative phrasings Agent 0 falls back to when the typed
query finds no open-access seed. A failed or unparseable call is not an
error: the run proceeds with the query as typed.
"""

from __future__ import annotations

import re

from research_assistant.config import BRIEF_TEMPERATURE, CHAT_OLLAMA_OPTIONS
from research_assistant.prompts import RESEARCH_BRIEF_USER
from research_assistant.shared.log import get_logger

logger = get_logger("brief")

_NUMBERED = re.compile(r"^\s*\(?(\d+)[.)]\s+(.+?)\s*$")
_QUOTES = "\"'`“”‘’"
MAX_ALTERNATES = 3


def parse_brief(text: str, query: str = "") -> list[str]:
    """The numbered lines of a brief, cleaned: surrounding quotes and a
    trailing full stop removed, the typed query itself and duplicates
    dropped, at most MAX_ALTERNATES kept in order."""
    seen = set()
    out: list[str] = []
    q = " ".join(query.lower().split())
    for line in (text or "").splitlines():
        m = _NUMBERED.match(line)
        if not m:
            continue
        phrase = m.group(2).strip().strip(_QUOTES).strip()
        phrase = phrase.rstrip(".").strip()
        # A bold or italic wrapper around the whole phrase is decoration, not query.
        phrase = phrase.strip("*_").strip()
        key = " ".join(phrase.lower().split())
        if not key or key == q or key in seen:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) == MAX_ALTERNATES:
            break
    return out


def write_brief(query: str) -> dict | None:
    """{"text", "alternates"} or None when the call fails or says nothing."""
    from research_assistant.shared.llm import chat

    try:
        text = chat(
            [{"role": "user", "content": RESEARCH_BRIEF_USER.format(query=query)}],
            temperature=BRIEF_TEMPERATURE,
            options=CHAT_OLLAMA_OPTIONS,
        ).content.strip()
    except Exception as exc:  # noqa: BLE001 — the brief is a courtesy, never a gate
        logger.warning("Brief failed: %s", exc)
        return None
    if not text:
        return None
    return {"text": text, "alternates": parse_brief(text, query)}
```

- [ ] **Step 5: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_brief.py
```

Expected: 6 passed

- [ ] **Step 6: Add the failing Agent 0 tests**

**Apply this change to `tests/test_discoverer.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_discoverer.py
+++ b/tests/test_discoverer.py
@@ -253,3 +253,87 @@
 
 if __name__ == "__main__":
     unittest.main()
+
+
+# ─── Narrated runs: the brief's phrasings and the transcript (spec §2E) ─────
+
+
+class TestAlternatesAndEvents(unittest.TestCase):
+    def setUp(self):
+        self.tmp = tempfile.TemporaryDirectory()
+        self.addCleanup(self.tmp.cleanup)
+        for attr, sub in (("RAW_DIR", "raw"), ("SEED_PAPERS_PATH", "seed_papers.json")):
+            p = patch.object(a0, attr, os.path.join(self.tmp.name, sub))
+            p.start()
+            self.addCleanup(p.stop)
+        from research_assistant.shared import pipeline_status, runlog
+
+        self.runlog = runlog
+        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
+        self.addCleanup(runlog.set_runs_dir, None)
+        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
+        self.addCleanup(pipeline_status.set_status_path, None)
+        runlog._reset_for_tests()
+        self.addCleanup(runlog._reset_for_tests)
+        self.calls = []
+
+    def _provider(self, hits_for):
+        def provider(query, limit):
+            self.calls.append(query)
+            return hits_for.get(query, [])
+        return provider
+
+    def test_alternates_are_tried_only_after_the_query_fails(self):
+        hit = {"title": "Found", "doi": "10.1/found", "pdf_url": "http://x/f.pdf"}
+        provider = self._provider({"alt two": [hit]})
+        with patch.dict(a0._PROVIDERS, {"arxiv": provider}, clear=True), \
+             patch.object(a0, "SEARCH_PROVIDERS", ["arxiv"]), \
+             patch.object(a0, "download_pdf", return_value=(True, None)):
+            cand, key, dest = a0.find_and_fetch_seed("typed", alternates=["alt one", "alt two", "typed"])
+        self.assertEqual(self.calls, ["typed", "alt one", "alt two"])   # the query itself is not repeated
+        self.assertEqual(cand["via_query"], "alt two")
+        self.assertTrue(dest.endswith(".pdf"))
+
+    def test_search_tried_and_seed_found_events(self):
+        hit = {"title": "Found", "doi": "10.1/found", "pdf_url": "http://x/f.pdf"}
+        provider = self._provider({"typed": [hit, {"title": "no link"}]})
+        rid = self.runlog.start_run("pipeline", {})
+        with patch.dict(a0._PROVIDERS, {"arxiv": provider}, clear=True), \
+             patch.object(a0, "SEARCH_PROVIDERS", ["arxiv"]), \
+             patch.object(a0, "download_pdf", return_value=(True, None)):
+            path = a0.discover("typed")
+        self.assertTrue(path)
+        events = self.runlog.read_events(rid)
+        types = [e["type"] for e in events]
+        self.assertEqual(types, ["narration", "search_tried", "seed_found", "narration"])
+        self.assertEqual(events[1]["payload"], {"provider": "arxiv", "query": "typed", "results": 2, "with_pdf": 1})
+        found = events[2]["payload"]
+        self.assertEqual((found["title"], found["source"], found["via_query"], found["cached"]),
+                         ("Found", "search", "typed", False))
+        # The manifest is keyed by the typed query and carries no via_query.
+        self.assertIn("typed", a0._load_seeds())
+        self.assertNotIn("via_query", a0._load_seeds()["typed"])
+
+    def test_seed_missing_event_lists_every_phrasing_tried(self):
+        provider = self._provider({})
+        rid = self.runlog.start_run("pipeline", {})
+        with patch.dict(a0._PROVIDERS, {"arxiv": provider}, clear=True), \
+             patch.object(a0, "SEARCH_PROVIDERS", ["arxiv"]):
+            self.assertIsNone(a0.discover("typed", alternates=["alt"]))
+        events = self.runlog.read_events(rid)
+        missing = [e for e in events if e["type"] == "seed_missing"][0]["payload"]
+        self.assertEqual(missing, {"query": "typed", "tried": ["typed", "alt"]})
+        self.assertIn("2 phrasings", events[-1]["payload"]["text"])
+
+    def test_cached_seed_emits_seed_found_without_searching(self):
+        os.makedirs(a0.RAW_DIR, exist_ok=True)
+        pdf = os.path.join(a0.RAW_DIR, "seed.pdf")
+        open(pdf, "wb").write(b"%PDF-1.4")
+        a0._save_seeds({"typed": {"key": "k", "path": pdf, "fetched_at": "t", "source": "search", "title": "Cached"}})
+        rid = self.runlog.start_run("pipeline", {})
+        provider = self._provider({})
+        with patch.dict(a0._PROVIDERS, {"arxiv": provider}, clear=True):
+            self.assertEqual(a0.discover("typed"), pdf)
+        self.assertEqual(self.calls, [])
+        found = [e for e in self.runlog.read_events(rid) if e["type"] == "seed_found"][0]["payload"]
+        self.assertEqual((found["title"], found["cached"]), ("Cached", True))
```

- [ ] **Step 7: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_discoverer.py -k Alternates
```

Expected: 4 failed — `find_and_fetch_seed() got an unexpected keyword argument 'alternates'`

- [ ] **Step 8: Change Agent 0**

**Apply this change to `research_assistant/agents/agent0_discoverer.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/agents/agent0_discoverer.py
+++ b/research_assistant/agents/agent0_discoverer.py
@@ -39,6 +39,7 @@
 from research_assistant.shared.atomic import atomic_write_json
 from research_assistant.shared.fetch import HEADERS, download_pdf, filename_for, _is_pdf
 from research_assistant.shared.log import get_logger
+from research_assistant.shared import narration, runlog
 from research_assistant.shared.source_key import source_key, normalise_doi
 
 logger = get_logger("agent0")
@@ -163,7 +164,7 @@
 # ─── Selection ───────────────────────────────────────────────────────────────
 
 
-def find_and_fetch_seed(query, force=False, limit=SEARCH_LIMIT):
+def find_and_fetch_seed(query, force=False, limit=SEARCH_LIMIT, alternates=()):
     """Walk providers × their ranked candidates until a PDF *actually downloads*.
 
     A candidate whose ``pdf_url`` 404s or serves an HTML paywall page (common
@@ -171,48 +172,59 @@
     on to the next candidate and then the next provider — so a physics query
     that OpenAlex only has paywalled still gets picked up from arXiv.
 
+    ``alternates`` are the brief's phrasings (shared.brief): each is walked
+    the same way, in order, only after the typed query found nothing. The
+    returned candidate carries ``via_query`` — the phrasing that worked.
+
     Returns ``(candidate, key, dest_path)`` or ``(None, None, None)``.
     """
     os.makedirs(RAW_DIR, exist_ok=True)
     seen = set()
 
-    for name in SEARCH_PROVIDERS:
-        pname = name.strip()
-        provider = _PROVIDERS.get(pname)
-        if not provider:
-            logger.warning("Unknown search provider %r — skipping.", pname)
-            continue
-        try:
-            results = provider(query, limit)
-        except requests.RequestException as exc:
-            logger.warning("%s search failed: %s", pname, exc)
-            continue
-
-        n_links = 0
-        for cand in results:
-            if not cand.get("pdf_url"):
+    for phrasing in [query, *[a for a in alternates if a and a != query]]:
+        for name in SEARCH_PROVIDERS:
+            pname = name.strip()
+            provider = _PROVIDERS.get(pname)
+            if not provider:
+                logger.warning("Unknown search provider %r — skipping.", pname)
                 continue
-            key = source_key(cand)
-            if not key or key in seen:
+            try:
+                results = provider(phrasing, limit)
+            except requests.RequestException as exc:
+                logger.warning("%s search failed: %s", pname, exc)
+                runlog.emit("search_tried", provider=pname, query=phrasing, results=0, with_pdf=0,
+                            error=str(exc))
                 continue
-            seen.add(key)
-            n_links += 1
 
-            dest = os.path.join(RAW_DIR, filename_for(key))
-            if os.path.exists(dest) and not force:
-                logger.info("%s: [%s] already on disk", pname, key)
-                return cand, key, dest
-
-            ok, reason = download_pdf(cand["pdf_url"], dest)
-            if ok:
-                logger.info("%s: [%s] %s", pname, key, os.path.basename(dest))
-                return cand, key, dest
-            logger.info("%s: [%s] not downloadable — %s", pname, key, reason)
-
-        logger.info(
-            "%s: %d result(s), %d with a PDF link, none downloadable.",
-            pname, len(results), n_links,
-        )
+            with_pdf = sum(1 for c in results if c.get("pdf_url"))
+            runlog.emit("search_tried", provider=pname, query=phrasing,
+                        results=len(results), with_pdf=with_pdf)
+
+            n_links = 0
+            for cand in results:
+                if not cand.get("pdf_url"):
+                    continue
+                key = source_key(cand)
+                if not key or key in seen:
+                    continue
+                seen.add(key)
+                n_links += 1
+
+                dest = os.path.join(RAW_DIR, filename_for(key))
+                if os.path.exists(dest) and not force:
+                    logger.info("%s: [%s] already on disk", pname, key)
+                    return {**cand, "via_query": phrasing}, key, dest
+
+                ok, reason = download_pdf(cand["pdf_url"], dest)
+                if ok:
+                    logger.info("%s: [%s] %s", pname, key, os.path.basename(dest))
+                    return {**cand, "via_query": phrasing}, key, dest
+                logger.info("%s: [%s] not downloadable — %s", pname, key, reason)
+
+            logger.info(
+                "%s: %d result(s), %d with a PDF link, none downloadable.",
+                pname, len(results), n_links,
+            )
     return None, None, None
 
 
@@ -622,22 +634,41 @@
 # ─── Entry point ─────────────────────────────────────────────────────────────
 
 
-def discover(query, force=False):
+def discover(query, force=False, alternates=()):
     """Find the top relevant open-access paper for *query* and pull it to RAW_DIR.
 
+    ``alternates`` are fallback phrasings (see find_and_fetch_seed). The seed
+    manifest is keyed by the typed query whichever phrasing found the paper.
+
+    Emits ``seed_found`` / ``seed_missing`` on the run transcript — this is
+    the only place that knows which phrasing worked. The URL and file paths
+    (discover_from_url / discover_from_file) leave that to the orchestrator.
+
     Returns the local PDF path, or None when nothing usable was found.
     """
     seeds = _load_seeds()
     prior = seeds.get(query)
     if prior and not force and os.path.exists(prior.get("path", "")):
         logger.info("Already seeded for this query: %s", prior["path"])
+        runlog.emit("seed_found", title=prior.get("title"), key=prior.get("key"), doi=prior.get("doi"),
+                    arxiv_id=prior.get("arxiv_id"), url=prior.get("url"), source=prior.get("source"),
+                    via_query=query, cached=True, path=prior["path"])
+        runlog.note(narration.seed_found(prior.get("title") or prior.get("key") or "the seed"))
         return prior["path"]
 
     logger.info("Searching for: %s", query)
-    cand, key, dest = find_and_fetch_seed(query, force=force)
+    runlog.note(narration.searching(query))
+    cand, key, dest = find_and_fetch_seed(query, force=force, alternates=alternates)
     if not dest:
         logger.error("No downloadable open-access PDF for query: %s", query)
+        tried = [query, *[a for a in alternates if a and a != query]]
+        runlog.emit("seed_missing", query=query, tried=tried)
+        runlog.note(narration.seed_missing(len(tried)))
         return None
+    runlog.emit("seed_found", title=cand.get("title"), key=key, doi=cand.get("doi"),
+                arxiv_id=cand.get("arxiv_id"), url=cand.get("pdf_url"), source="search",
+                via_query=cand.get("via_query"), cached=False, path=dest)
+    runlog.note(narration.seed_found(cand.get("title") or key, via_query=cand.get("via_query"), original=query))
 
     seeds[query] = SeedPaper(
         key=key,
```

- [ ] **Step 9: Run the file**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_discoverer.py
```

Expected: 19 passed

- [ ] **Step 10: Commit**

```bash
git add research_assistant/prompts.py research_assistant/shared/brief.py research_assistant/agents/agent0_discoverer.py tests/test_brief.py tests/test_discoverer.py
git commit -m "feat(brief): one model call before a seed search; Agent 0 tries its phrasings when the typed query finds nothing, and narrates the search

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Ingestion emits per paper, per figure, per summary — and stops between them

**Files:**
- Modify: `research_assistant/shared/ingestion.py`
- Modify: `research_assistant/shared/ingest_v2.py`
- Test: `tests/test_ingestion.py`

**Interfaces:**
- Consumes: `runlog`, `narration`.
- Produces events (stage from `track_stage`, so the same code serves the seed and the references): `batch_started {n, skipped}` + narration; per PDF `paper_read {document, title, chunks, captions, described, extraction}` (via the new `_paper_read_payload(path, title, entries)`) or `paper_empty {document, title}`; in `process_pdf_v2` per described figure `figure_described {document, label, kind, description, image_path}` (absolute crop path); in `upsert_summaries` `summaries_started {n}` + narration, then `paper_summarised {document, citation, summary}` after each `col.add`; `batch_saved {processed, inserted, described, empty}` after the BM25 rebuild. `check_cancelled()` at the top of each sequential PDF iteration (a cancel there narrates `cancelled(parsed_unsaved=k)` and re-raises — the batch is not upserted, spec §2C), before each figure description, and before each summary call.

The ingestion loop's structure — parse every PDF, then upsert, then summarise — is unchanged (spec §2D last paragraph); a card appears at `paper_read` and gains its summary later in the same stage.

- [ ] **Step 1: Add the failing tests**

**Apply this change to `tests/test_ingestion.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_ingestion.py
+++ b/tests/test_ingestion.py
@@ -742,3 +742,195 @@
         v1.assert_called_once()
         v2.assert_not_called()
         self.assertEqual(result["extraction"], {"layout": 0, "text_only": 0})
+
+
+# ─── Narrated runs: the transcript events and the stop boundary (spec §2C–D) ─
+
+
+class TranscriptIngestCase(IngestPdfsCase):
+    """IngestPdfsCase plus a bound run to capture events."""
+
+    def setUp(self):
+        super().setUp()
+        from research_assistant.shared import pipeline_status, runlog
+
+        self.runlog = runlog
+        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
+        self.addCleanup(runlog.set_runs_dir, None)
+        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
+        self.addCleanup(pipeline_status.set_status_path, None)
+        runlog._reset_for_tests()
+        self.addCleanup(runlog._reset_for_tests)
+        self.rid = runlog.start_run("pipeline", {})
+        # Entries shaped like v2's so paper_read counts something.
+        self.corpus_per_pdf = 2
+
+        def fake_process(path, label, *a, **k):
+            self.processed.append(path)
+            return [{"type": "text_chunk", "content": "t", "extraction": "grobid"},
+                    {"type": "caption", "content": "c", "extraction": "grobid"}]
+
+        p = patch.object(ing, "process_pdf", fake_process); p.start(); self.addCleanup(p.stop)
+
+    def events(self, *types):
+        evs = self.runlog.read_events(self.rid)
+        return [e for e in evs if not types or e["type"] in types]
+
+
+class TestIngestEvents(TranscriptIngestCase):
+    def test_batch_started_paper_read_and_batch_saved(self):
+        a, b = _touch(self.tmp, "a.pdf"), _touch(self.tmp, "b.pdf")
+        self.inserted = 4
+        ing.ingest_pdfs({a: "Paper A", b: "Paper B"}, rebuild_index=False)
+        types = [e["type"] for e in self.events()]
+        self.assertEqual(types, ["batch_started", "narration", "paper_read", "paper_read", "batch_saved"])
+        started = self.events("batch_started")[0]["payload"]
+        self.assertEqual(started, {"n": 2, "skipped": 0})
+        read = self.events("paper_read")[0]["payload"]
+        self.assertEqual(read, {"document": "a.pdf", "title": "Paper A", "chunks": 1, "captions": 1,
+                                "described": 0, "extraction": "grobid"})
+        saved = self.events("batch_saved")[0]["payload"]
+        self.assertEqual(saved, {"processed": 2, "inserted": 4, "described": 0, "empty": 0})
+        self.assertIn("Reading 2 papers", self.events("narration")[0]["payload"]["text"])
+
+    def test_empty_pdf_emits_paper_empty(self):
+        a = _touch(self.tmp, "scan.pdf")
+        with patch.object(ing, "process_pdf", lambda *a, **k: []):
+            ing.ingest_pdfs({a: "Scan"}, rebuild_index=False)
+        self.assertEqual(self.events("paper_empty")[0]["payload"], {"document": "scan.pdf", "title": "Scan"})
+        self.assertEqual(self.events("batch_saved")[0]["payload"]["empty"], 1)
+
+    def test_nothing_to_ingest_still_reports_the_batch(self):
+        a = _touch(self.tmp, "a.pdf")
+        self.ingested = {"a.pdf"}
+        ing.ingest_pdfs({a: "A"})
+        self.assertEqual(self.events("batch_started")[0]["payload"], {"n": 0, "skipped": 1})
+        self.assertIn("already on the shelf", self.events("narration")[0]["payload"]["text"])
+
+    def test_paper_read_payload_counts_v1_entries_too(self):
+        payload = ing._paper_read_payload("/x/My Paper.pdf", "T", [
+            {"type": "text", "extraction": "text_only"}, {"type": "text"}, {"type": "figure"},
+        ])
+        self.assertEqual(payload, {"document": "my_paper.pdf", "title": "T", "chunks": 2, "captions": 0,
+                                   "described": 0, "extraction": "text_only"})
+
+
+class TestIngestCancel(TranscriptIngestCase):
+    def test_stop_between_papers_discards_the_batch_and_says_so(self):
+        a, b, c = (_touch(self.tmp, n) for n in ("a.pdf", "b.pdf", "c.pdf"))
+        rid, runlog = self.rid, self.runlog
+
+        def fake_process(path, label, *a, **k):
+            self.processed.append(path)
+            if len(self.processed) == 2:
+                runlog.cancel(rid)               # the reader presses Stop during paper 2
+            return [{"type": "text_chunk", "content": "t"}]
+
+        upserts = []
+        with patch.object(ing, "process_pdf", fake_process), \
+             patch.object(ing, "upsert_corpus", lambda corpus: upserts.append(len(corpus)) or 0):
+            with self.assertRaises(runlog.RunCancelled):
+                ing.ingest_pdfs({a: "A", b: "B", c: "C"}, rebuild_index=False)
+        self.assertEqual(len(self.processed), 2)          # c was never parsed
+        self.assertEqual(upserts, [])                     # nothing saved
+        self.assertFalse(manifest_mod.load())             # nothing marked
+        last = self.events()[-1]
+        self.assertEqual(last["type"], "narration")
+        self.assertIn("2 papers were parsed but not saved", last["payload"]["text"])
+        self.assertEqual(self.rebuilds, 0)
+
+
+class TestSummaryEvents(unittest.TestCase):
+    def setUp(self):
+        self.tmp = tempfile.TemporaryDirectory()
+        self.addCleanup(self.tmp.cleanup)
+        from research_assistant.shared import pipeline_status, runlog
+
+        self.runlog = runlog
+        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
+        self.addCleanup(runlog.set_runs_dir, None)
+        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
+        self.addCleanup(pipeline_status.set_status_path, None)
+        runlog._reset_for_tests()
+        self.addCleanup(runlog._reset_for_tests)
+        self.rid = runlog.start_run("pipeline", {})
+
+    def _run(self, per_doc_text, cancel_after=None):
+        import sys
+        from unittest.mock import MagicMock
+
+        added = []
+        col = MagicMock()
+        col.get.return_value = {"metadatas": []}
+        col.add.side_effect = lambda **kw: added.append(kw)
+        client = MagicMock()
+        client.get_or_create_collection.return_value = col
+        chroma = MagicMock()
+        chroma.PersistentClient.return_value = client
+        calls = []
+
+        def chat(messages, model=None, **kw):
+            calls.append(model)
+            if cancel_after is not None and len(calls) == cancel_after:
+                self.runlog.cancel(self.rid)
+            return MagicMock(content=f"summary {len(calls)}")
+
+        emb = MagicMock()
+        emb.embed_documents.side_effect = lambda docs: [[0.0] * 3 for _ in docs]
+        with patch.dict(sys.modules, {"chromadb": chroma}), \
+             patch("research_assistant.shared.llm.chat", chat), \
+             patch("research_assistant.shared.llm.get_embeddings", return_value=emb):
+            made = ing.upsert_summaries(per_doc_text, {d: f"cite {d}" for d in per_doc_text})
+        return made, added
+
+    def test_each_summary_is_an_event_with_its_text(self):
+        made, added = self._run({"d1": "text one", "d2": "text two"})
+        self.assertEqual(made, 2)
+        events = self.runlog.read_events(self.rid)
+        self.assertEqual([e["type"] for e in events],
+                         ["summaries_started", "narration", "paper_summarised", "paper_summarised"])
+        self.assertEqual(events[0]["payload"], {"n": 2})
+        self.assertEqual(events[2]["payload"], {"document": "d1", "citation": "cite d1", "summary": "summary 1"})
+
+    def test_stop_lands_between_summaries(self):
+        with self.assertRaises(self.runlog.RunCancelled):
+            self._run({"d1": "a", "d2": "b", "d3": "c"}, cancel_after=1)
+        summarised = [e for e in self.runlog.read_events(self.rid) if e["type"] == "paper_summarised"]
+        self.assertEqual(len(summarised), 1)   # d1 written; d2 never called
+
+
+class TestFigureEvents(unittest.TestCase):
+    def test_figure_described_event_and_cancel_before_the_call(self):
+        import sys
+        from research_assistant.shared import ingest_v2 as iv, pipeline_status, runlog
+        from research_assistant.shared.extract import Document, Section, Paragraph, Figure
+
+        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
+        runlog.set_runs_dir(os.path.join(tmp.name, "runs")); self.addCleanup(runlog.set_runs_dir, None)
+        pipeline_status.set_status_path(os.path.join(tmp.name, "s.json")); self.addCleanup(pipeline_status.set_status_path, None)
+        runlog._reset_for_tests(); self.addCleanup(runlog._reset_for_tests)
+        rid = runlog.start_run("pipeline", {})
+
+        para = Paragraph("A sentence of text that is long enough to be kept as a chunk by the chunker here.", 0)
+        doc = Document("paper.pdf", "A Title", [para], [Section("1. Intro", "introduction", [para])], [
+            Figure("fig_0", "figure", "Figure 1", "DOS.", 1, (0, 0, 10, 10), []),
+            Figure("fig_1", "table", "Table 1", "Params.", 2, (0, 0, 10, 10), []),
+        ], "grobid", n_bib=1)
+        described = []
+
+        def describe(path, fig, context):
+            described.append(fig.label)
+            if len(described) == 1:
+                runlog.cancel(rid)
+            return "Two peaks."
+
+        with patch.object(iv, "extract", return_value=doc), patch.object(iv, "crop_figure", return_value=True), \
+             patch.object(iv, "describe_crop", describe):
+            with self.assertRaises(runlog.RunCancelled):
+                iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
+        self.assertEqual(described, ["Figure 1"])          # Table 1 never reached the model
+        ev = [e for e in runlog.read_events(rid) if e["type"] == "figure_described"]
+        self.assertEqual(len(ev), 1)
+        self.assertEqual(ev[0]["payload"]["label"], "Figure 1")
+        self.assertEqual(ev[0]["payload"]["description"], "Two peaks.")
+        self.assertTrue(ev[0]["payload"]["image_path"].startswith("/tmp/img/"))
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_ingestion.py -k 'Events or Cancel or Summary or Figure'
```

Expected: fail — no events on the transcript / `_paper_read_payload` missing

- [ ] **Step 3: Change `ingestion.py`**

**Apply this change to `research_assistant/shared/ingestion.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/shared/ingestion.py
+++ b/research_assistant/shared/ingestion.py
@@ -72,7 +72,7 @@
 from research_assistant.shared.db import get_max_chunk_index
 from research_assistant.shared import manifest
 from research_assistant.shared.tokenize import tokenize, TOKENIZER_VERSION
-from research_assistant.shared import pipeline_status
+from research_assistant.shared import narration, pipeline_status, runlog
 
 logger = get_logger("ingestion")
 
@@ -569,8 +569,11 @@
     model_name = SUMMARY_MODEL or "Qwen2.5"
     if total_to_sum:
         pipeline_status.add_event(f"📝 Generating summaries for {total_to_sum} paper(s) with {model_name}…")
+        runlog.emit("summaries_started", n=total_to_sum)
+        runlog.note(narration.summarising(total_to_sum))
 
     for idx, doc in enumerate(to_summarize, 1):
+        runlog.check_cancelled()
         pipeline_status.update_progress(
             item_current=idx,
             item_total=total_to_sum,
@@ -597,6 +600,9 @@
             metadatas=[{"document": doc, "citation_source": per_doc_citation.get(doc, "")}],
         )
         made += 1
+        # The reader sees each summary as it is written (spec §2D).
+        runlog.emit("paper_summarised", document=doc,
+                    citation=per_doc_citation.get(doc, ""), summary=summary)
 
     if made:
         pipeline_status.add_event(f"✓ Generated {made} paper summaries with {model_name}")
@@ -865,6 +871,21 @@
 # ─── Unified batch ingestion ──────────────────────────────────────────────────
 
 
+def _paper_read_payload(path: str, title: str, entries: list[dict]) -> dict:
+    """The paper_read event: what one parse produced, counted by entry type.
+    v2 types are text_chunk / caption / figure_description / summary_source;
+    v1 uses text plus the layout labels (figure, table)."""
+    types = [e.get("type") for e in entries]
+    return {
+        "document": pdf_key(path),
+        "title": title,
+        "chunks": sum(1 for t in types if t in ("text_chunk", "text")),
+        "captions": sum(1 for t in types if t == "caption"),
+        "described": sum(1 for t in types if t == "figure_description"),
+        "extraction": next((e.get("extraction") for e in entries if e.get("extraction")), ""),
+    }
+
+
 def ingest_pdfs(
     pdfs,
     workers: int = 1,
@@ -941,7 +962,9 @@
 
     if not candidates:
         logger.info("%sNothing to ingest.", log_prefix)
+        runlog.emit("batch_started", n=0, skipped=result["skipped"])
         if result["skipped"]:
+            runlog.note(narration.reading(0, result["skipped"]))
             pipeline_status.add_event(f"ℹ️ All {result['skipped']} PDF(s) already indexed into corpus")
             pipeline_status.update_progress(
                 item_current=0,
@@ -966,9 +989,19 @@
     try:
         corpus = []
         scanned_or_empty = []
+        runlog.emit("batch_started", n=len(candidates), skipped=result["skipped"])
+        runlog.note(narration.reading(len(candidates), result["skipped"]))
         if workers <= 1:
             logger.info("%sProcessing %d PDF(s) sequentially…", log_prefix, len(candidates))
+            parsed = 0
             for i, (path, label) in enumerate(candidates.items(), 1):
+                # A stop lands here, between papers — never inside a parse or
+                # a figure call. What was parsed so far is not saved (spec §2C).
+                try:
+                    runlog.check_cancelled()
+                except runlog.RunCancelled:
+                    runlog.note(narration.cancelled(parsed_unsaved=parsed))
+                    raise
                 name = os.path.basename(path)
                 pipeline_status.update_progress(
                     item_current=i,
@@ -980,6 +1013,10 @@
                 entries = process_pdf(path, label, describe_figures=describe_figures)
                 if not entries:
                     scanned_or_empty.append(name)
+                    runlog.emit("paper_empty", document=pdf_key(path), title=label or name)
+                else:
+                    parsed += 1
+                    runlog.emit("paper_read", **_paper_read_payload(path, label or name, entries))
                 corpus.extend(entries)
         else:
             logger.info(
@@ -1066,6 +1103,8 @@
         if rebuild_index and (result["inserted"] or not os.path.exists(BM25_INDEX_PATH)):
             rebuild_bm25()
 
+        runlog.emit("batch_saved", processed=result["processed"], inserted=result["inserted"],
+                    described=result.get("described", 0), empty=len(scanned_or_empty))
         pipeline_status.add_event(
             f"✅ Ingested {result['processed']} PDFs ({result['inserted']} chunks)"
         )
```

- [ ] **Step 4: Change `ingest_v2.py`**

**Apply this change to `research_assistant/shared/ingest_v2.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/shared/ingest_v2.py
+++ b/research_assistant/shared/ingest_v2.py
@@ -25,6 +25,7 @@
     DESCRIPTION_PREFIX, crop_name, crop_figure, figure_context, describe_crop,
 )
 from research_assistant.shared.log import get_logger
+from research_assistant.shared import runlog
 
 logger = get_logger("ingest_v2")
 
@@ -100,6 +101,7 @@
         entries.append(cap_entry)
 
         if describe_figures and fig is not None and image_path:
+            runlog.check_cancelled()           # never inside the model call
             context = figure_context(fig)
             try:
                 text = describe_crop(os.path.join(images_dir, image_path), fig, context)
@@ -115,6 +117,8 @@
             cap_meta["described"] = True
             described += 1
             logger.info("  described %s (%d/%d)", fig.label, described, len(doc.figures))
+            runlog.emit("figure_described", document=key, label=fig.label, kind=c.figure_kind,
+                        description=text, image_path=os.path.join(images_dir, image_path))
 
     entries.append({"document": key, "citation": citation_string, "page": 0,
                     "type": "summary_source", "content": _summary_source(doc),
```

- [ ] **Step 5: Run the ingestion tests**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_ingestion.py tests/test_ingest_v2.py
```

Expected: all pass (61 in the scratch validation)

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/ingestion.py research_assistant/shared/ingest_v2.py tests/test_ingestion.py
git commit -m "feat(ingest): the transcript sees every paper, figure and summary as it lands; a stop lands between them

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Agent 2 emits per fetch and returns what it did

**Files:**
- Modify: `research_assistant/agents/agent2_fetcher.py`
- Test: `tests/test_fetcher.py`

**Interfaces:**
- Consumes: `runlog`.
- Produces: `fetch_started {remaining, already}`, per paper `paper_fetched {key, title, provider, doi}` or `paper_unavailable {key, title, reason}`; `check_cancelled()` at the top of each iteration (the checkpoint of the previous paper has already been written). `fetch_papers()` now returns `{checked, fetched, unavailable, downloaded_total, unavailable_total}` (it returned `None`) — the orchestrator narrates from it (Task 7).

- [ ] **Step 1: Add the failing tests**

**Apply this change to `tests/test_fetcher.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_fetcher.py
+++ b/tests/test_fetcher.py
@@ -359,3 +359,74 @@
                 "paywalled / not open access | not in Europe PMC | arXiv error: timeout"
             )
         )
+
+
+# ─── Narrated runs: per-paper events, the stop boundary, the return (spec §2D) ─
+
+
+class TestFetchEvents(unittest.TestCase):
+    def setUp(self):
+        self.tmp = tempfile.TemporaryDirectory()
+        self.addCleanup(self.tmp.cleanup)
+        from research_assistant.shared import pipeline_status, runlog
+
+        self.runlog = runlog
+        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
+        self.addCleanup(runlog.set_runs_dir, None)
+        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
+        self.addCleanup(pipeline_status.set_status_path, None)
+        runlog._reset_for_tests()
+        self.addCleanup(runlog._reset_for_tests)
+        self.rid = runlog.start_run("pipeline", {})
+        for attr, name in (("DOWNLOADED_JSON_PATH", "downloaded.json"), ("FAILED_DOWNLOADS_PATH", "failed.json"),
+                           ("PULLED_PDFS_DIR", "pulled"), ("EXTRACTED_CITATIONS_PATH", "extracted.json")):
+            p = patch.object(agent2_fetcher, attr, os.path.join(self.tmp.name, name)); p.start(); self.addCleanup(p.stop)
+        for attr, val in (("UNPAYWALL_SLEEP", 0), ("ARXIV_RATE_LIMIT", 0)):
+            p = patch.object(agent2_fetcher, attr, val); p.start(); self.addCleanup(p.stop)
+
+    @staticmethod
+    def _refs(n):
+        return [{"raw_reference": f"Ref {i}", "source_file": "seed.pdf", "title": f"Paper {i}",
+                 "doi": f"10.1/p{i}", "doi_confidence": "high"} for i in range(n)]
+
+    def test_events_and_return(self):
+        def unpaywall(doi, dest):
+            if doi.endswith("p1"):
+                return False, "paywalled / not open access"
+            open(dest, "wb").write(b"%PDF-1.4")
+            return True, None
+
+        with patch.object(agent2_fetcher, "_load_references", return_value=self._refs(3)), \
+             patch.object(agent2_fetcher, "resolve_doi", side_effect=lambda r: (r["doi"], "grobid")), \
+             patch.object(agent2_fetcher, "try_unpaywall", side_effect=unpaywall), \
+             patch.object(agent2_fetcher, "try_europepmc", return_value=(False, "not in Europe PMC")), \
+             patch.object(agent2_fetcher, "try_arxiv", return_value=(False, "not found on arXiv")):
+            out = agent2_fetcher.fetch_papers()
+        self.assertEqual(out, {"checked": 3, "fetched": 2, "unavailable": 1,
+                               "downloaded_total": 2, "unavailable_total": 1})
+        events = self.runlog.read_events(self.rid)
+        self.assertEqual([e["type"] for e in events],
+                         ["fetch_started", "paper_fetched", "paper_unavailable", "paper_fetched"])
+        self.assertEqual(events[0]["payload"], {"remaining": 3, "already": 0})
+        self.assertEqual(events[1]["payload"]["title"], "Paper 0")
+        self.assertEqual(events[1]["payload"]["provider"], "unpaywall")
+        self.assertIn("paywalled", events[2]["payload"]["reason"])
+        self.assertEqual(events[2]["payload"]["key"], agent2_fetcher.source_key(self._refs(2)[1]))
+
+    def test_stop_lands_between_papers(self):
+        seen = []
+
+        def unpaywall(doi, dest):
+            seen.append(doi)
+            self.runlog.cancel(self.rid)
+            open(dest, "wb").write(b"%PDF-1.4")
+            return True, None
+
+        with patch.object(agent2_fetcher, "_load_references", return_value=self._refs(3)), \
+             patch.object(agent2_fetcher, "resolve_doi", side_effect=lambda r: (r["doi"], "grobid")), \
+             patch.object(agent2_fetcher, "try_unpaywall", side_effect=unpaywall):
+            with self.assertRaises(self.runlog.RunCancelled):
+                agent2_fetcher.fetch_papers()
+        self.assertEqual(seen, ["10.1/p0"])                      # paper 0 finished; paper 1 never started
+        with open(agent2_fetcher.DOWNLOADED_JSON_PATH) as fh:
+            self.assertEqual(len(json.load(fh)), 1)                # and it was checkpointed
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_fetcher.py -k FetchEvents
```

Expected: 2 failed

- [ ] **Step 3: Change the fetcher**

**Apply this change to `research_assistant/agents/agent2_fetcher.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/agents/agent2_fetcher.py
+++ b/research_assistant/agents/agent2_fetcher.py
@@ -37,7 +37,7 @@
 from research_assistant.shared.source_key import source_key, normalise_doi, is_authoritative
 from research_assistant.shared.atomic import atomic_write_json
 from research_assistant.shared.fetch import HEADERS, download_pdf, filename_for
-from research_assistant.shared import pipeline_status
+from research_assistant.shared import pipeline_status, runlog
 
 logger = get_logger("agent2")
 
@@ -321,6 +321,7 @@
         logger.info("Retrying %d that previously failed for a transient reason.", retrying)
 
     standalone = not pipeline_status.get_status().get("active")
+    n_fetched = n_unavailable = 0          # this call's outcomes, for the transcript
     with pipeline_status.track_stage(
         "fetch",
         "Fetching referenced papers",
@@ -331,7 +332,9 @@
         mark_idle_on_exit=standalone,
         last_summary=f"+{len(downloaded)} papers fetched",
     ):
+        runlog.emit("fetch_started", remaining=len(remaining), already=len(downloaded))
         for i, (key, ref) in enumerate(remaining, start=1):
+            runlog.check_cancelled()            # between papers, never mid-download
             label = (ref.get("title") or ref.get("raw_reference") or key)[:70]
             logger.info("[%d/%d] %s", i, len(remaining), label)
             pipeline_status.update_progress(
@@ -399,6 +402,9 @@
                 ).to_dict()
                 failed.pop(key, None)   # a retry that worked is no longer a failure
                 logger.info("    saved -> %s", os.path.basename(dest))
+                n_fetched += 1
+                runlog.emit("paper_fetched", key=key, title=ref.get("title") or label,
+                            provider=provider, doi=doi)
                 pipeline_status.add_event(f"✅ Downloaded ({provider or 'oa'}): {label[:50]}")
                 pipeline_status.update_progress(
                     item_current=i,
@@ -421,6 +427,9 @@
                 }
                 failed[key] = record
                 logger.info("    unavailable: %s", record["reason"])
+                n_unavailable += 1
+                runlog.emit("paper_unavailable", key=key, title=ref.get("title") or label,
+                            reason=reason_str)
                 reason_short = reason_str[:35]
                 pipeline_status.add_event(f"⚠️ {label[:45]} ({reason_short})")
                 pipeline_status.update_progress(
@@ -447,6 +456,15 @@
             len(failed),
             len(manual),
         )
+        # What this call did, for the orchestrator's narration. The manifests
+        # accumulate across runs; the counts here are this run's alone.
+        return {
+            "checked": len(remaining),
+            "fetched": n_fetched,
+            "unavailable": n_unavailable,
+            "downloaded_total": len(downloaded),
+            "unavailable_total": len(failed),
+        }
 
 
 
```

- [ ] **Step 4: Run the file**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_fetcher.py
```

Expected: 24 passed

- [ ] **Step 5: Commit**

```bash
git add research_assistant/agents/agent2_fetcher.py tests/test_fetcher.py
git commit -m "feat(fetch): every fetch outcome on the transcript; a stop lands between papers; fetch_papers returns its counts

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The orchestrator is a run: brief, per-node events, cancel, the synthesis narrated

**Files:**
- Create: `research_assistant/shared/respond.py`
- Modify: `orchestrate.py`
- Test: `tests/test_respond.py`, `tests/test_orchestrate_narrated.py` (new), `tests/test_orchestrate_upload.py`

**Interfaces:**
- Consumes: Tasks 1–6; `retrieve.research_answer` in either of its shapes.
- Produces: `respond.research_answer_narrated(query, **kwargs)` — calls `retrieve.research_answer`, forwarding `shortlist` and `notes` progress events when the function has an `on_progress` parameter (the synthesis-depth plan adds it; `inspect.signature` decides), emitting `shortlist` from the result otherwise, and always one `synthesis {text, citations, keys, unverified_citations, irrelevant_cited, passages, selected, mode, timings, ungrounded}` built by `respond.synthesis_payload(result, ungrounded=False)`. `orchestrate.run(...)` wraps `_run_graph(inputs)` in `runlog.run("pipeline", inputs)` (joins the launcher's run in the app; opens its own from the CLI and `watch.py`); `audit_citations` rides in `inputs`. Every node starts with `runlog.check_cancelled()`. `discover` calls `_brief(query)` for the search path (narration `brief_start`, then the `brief` event with `alternates`, or `brief_fallback`; `config.BRIEF=0` skips it), narrates the URL/file paths and emits their `seed_found` (with `path`). `extract` emits `references_extracted {n, with_doi, by_method, sample}` from Agent 1's file via `_references_event()` and narrates. `fetch` narrates `fetch_done` from Agent 2's return (tolerating `None` from an older fetcher). `respond` uses `research_answer_narrated`, then runs the seed audit (Task 9 makes it a stage) when asked; `fallback` emits an ungrounded `synthesis`. A graph `stopped` reason is emitted as `stopped {reason}` and recorded via `runlog.set_stopped`; a completed run sets the summary to the query. `RunCancelled` out of the graph reads "Pipeline stopped by the reader" in `pipeline_status`.

The two `TestAppSeedRender` tests in `tests/test_orchestrate_upload.py` exercise `app._render_seed_and_downloads`, which Task 12 deletes (the feed renders the seed and the downloads from events); the diff removes them now.

- [ ] **Step 1: Write the failing tests for `respond`**

**Create `tests/test_respond.py` with exactly this content:**

```python
"""respond — the synthesis call narrated, with or without on_progress
(spec 2026-09-12-narrated-runs §2D, §5)."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from research_assistant.shared import pipeline_status, respond, runlog


class RespondCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
        self.addCleanup(runlog.set_runs_dir, None)
        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
        self.addCleanup(pipeline_status.set_status_path, None)
        runlog._reset_for_tests()
        self.addCleanup(runlog._reset_for_tests)
        self.rid = runlog.start_run("pipeline", {})

    def _install_retrieve(self, fn):
        mod = types.ModuleType("research_assistant.shared.retrieve")
        mod.research_answer = fn
        p = patch.dict(sys.modules, {"research_assistant.shared.retrieve": mod})
        p.start()
        self.addCleanup(p.stop)

    def events(self):
        return runlog.read_events(self.rid)


RESULT = {
    "suggestion": "Established [P1].", "citations": ["Paper one"],
    "passages": [{"text": "chunk", "metadata": {"document": "d1", "page": 3}}],
    "selected": [{"key": "P1", "document": "d1", "citation": "Paper one", "score": 0.9, "summary": "s"}],
    "keys": {"P1": "Paper one"}, "unverified_citations": ["P9"], "irrelevant_cited": [],
    "mode": "map_reduce", "timings": {"total": 12.0},
}


class TestWithProgress(RespondCase):
    def test_forwards_shortlist_and_notes_then_emits_synthesis(self):
        def research_answer(query, top_k=None, mode=None, on_progress=None):
            on_progress("shortlist", {"papers": [{"key": "P1", "citation": "Paper one"}]})
            on_progress("notes", {"key": "P1", "citation": "Paper one", "notes": "Establishes…", "relevant": True})
            on_progress("synthesis", {"seconds": 9.0})       # no text in it: not forwarded
            return RESULT

        self._install_retrieve(research_answer)
        out = respond.research_answer_narrated("idea")
        self.assertIs(out, RESULT)
        types_ = [e["type"] for e in self.events()]
        self.assertEqual(types_, ["narration", "shortlist", "notes", "synthesis"])
        syn = self.events()[-1]["payload"]
        self.assertEqual(syn["text"], "Established [P1].")
        self.assertEqual(syn["keys"], {"P1": "Paper one"})
        self.assertEqual(syn["unverified_citations"], ["P9"])
        self.assertEqual(syn["selected"][0]["score"], 0.9)
        self.assertEqual(syn["passages"][0]["metadata"]["page"], 3)
        self.assertFalse(syn["ungrounded"])


class TestWithoutProgress(RespondCase):
    def test_old_signature_gets_shortlist_from_the_result(self):
        def research_answer(query, top_k=5):
            return RESULT

        self._install_retrieve(research_answer)
        respond.research_answer_narrated("idea")
        types_ = [e["type"] for e in self.events()]
        self.assertEqual(types_, ["narration", "shortlist", "synthesis"])
        self.assertEqual(self.events()[1]["payload"], {"papers": [{"key": "P1", "citation": "Paper one"}]})

    def test_none_result_emits_an_empty_synthesis(self):
        self._install_retrieve(lambda query, top_k=5: None)
        self.assertIsNone(respond.research_answer_narrated("idea"))
        self.assertEqual([e["type"] for e in self.events()], ["narration", "synthesis"])
        self.assertEqual(self.events()[-1]["payload"]["text"], "")


class TestPayload(unittest.TestCase):
    def test_fallback_answer_is_marked_ungrounded(self):
        p = respond.synthesis_payload({"suggestion": "From memory…", "citations": [], "passages": []}, ungrounded=True)
        self.assertTrue(p["ungrounded"])
        self.assertEqual(p["selected"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Write `respond.py`**

**Create `research_assistant/shared/respond.py` with exactly this content:**

```python
"""
respond — the synthesis call, narrated (spec 2026-09-12-narrated-runs §2D).

One function both the orchestrator's `respond` node and the upload path
call. It forwards retrieve.research_answer's progress events (`shortlist`,
`notes`) to the transcript when that function accepts `on_progress` — the
synthesis-depth plan adds it — and, either way, emits one `synthesis` event
carrying the answer itself, so the reader never depends on which build of
retrieve is installed.
"""

from __future__ import annotations

import inspect

from research_assistant.shared import narration, runlog


def _selected_payload(selected) -> list[dict]:
    out = []
    for r in selected or []:
        if not isinstance(r, dict):
            continue
        out.append({
            "key": r.get("key"), "document": r.get("document"), "citation": r.get("citation"),
            "score": r.get("score"), "summary": r.get("summary", ""),
        })
    return out


def synthesis_payload(result: dict | None, ungrounded: bool = False) -> dict:
    """The `synthesis` event's payload from a research_answer() result (or a
    fallback answer's dict)."""
    r = result or {}
    return {
        "text": r.get("suggestion", "") or "",
        "citations": list(r.get("citations") or []),
        "keys": dict(r.get("keys") or {}),
        "unverified_citations": list(r.get("unverified_citations") or []),
        "irrelevant_cited": list(r.get("irrelevant_cited") or []),
        "passages": [{"text": p.get("text", ""), "metadata": p.get("metadata") or {}}
                     for p in (r.get("passages") or []) if isinstance(p, dict)],
        "selected": _selected_payload(r.get("selected")),
        "mode": r.get("mode"),
        "timings": r.get("timings") or {},
        "ungrounded": bool(ungrounded or r.get("ungrounded")),
    }


def research_answer_narrated(query: str, **kwargs) -> dict | None:
    """retrieve.research_answer(query, **kwargs) with its progress on the
    transcript. Returns what research_answer returns."""
    from research_assistant.shared import retrieve

    runlog.note(narration.synthesis())
    accepts_progress = "on_progress" in inspect.signature(retrieve.research_answer).parameters
    forwarded = set()

    def on_progress(stage, payload):
        if stage in ("shortlist", "notes"):
            forwarded.add(stage)
            runlog.emit(stage, **(payload or {}))

    if accepts_progress:
        result = retrieve.research_answer(query, on_progress=on_progress, **kwargs)
    else:
        result = retrieve.research_answer(query, **kwargs)

    if result and "shortlist" not in forwarded:
        runlog.emit("shortlist", papers=[{"key": r.get("key"), "citation": r.get("citation")}
                                         for r in _selected_payload(result.get("selected"))])
    runlog.emit("synthesis", **synthesis_payload(result))
    return result
```

- [ ] **Step 3: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_respond.py
```

Expected: 4 passed

- [ ] **Step 4: Write the failing orchestrator tests**

**Create `tests/test_orchestrate_narrated.py` with exactly this content:**

```python
"""orchestrate — the run's transcript: brief, events per node, stop and
status (spec 2026-09-12-narrated-runs §2B–E)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import orchestrate
from research_assistant.shared import pipeline_status, runlog


class NarratedCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
        self.addCleanup(runlog.set_runs_dir, None)
        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
        self.addCleanup(pipeline_status.set_status_path, None)
        runlog._reset_for_tests()
        self.addCleanup(runlog._reset_for_tests)

    def events(self, rid=None):
        return runlog.read_events(rid or runlog.current_run_id())

    def types(self, rid=None):
        return [e["type"] for e in self.events(rid)]


class TestDiscoverBrief(NarratedCase):
    def test_brief_is_emitted_before_the_search_and_feeds_agent0(self):
        runlog.start_run("pipeline", {})
        brief = {"text": "You want X.\n1. alt one\n2. alt two\n3. alt three", "alternates": ["alt one", "alt two", "alt three"]}
        with patch("research_assistant.shared.brief.write_brief", return_value=brief) as wb, \
             patch.object(orchestrate.agent0_discoverer, "discover", return_value=None) as disc:
            res = orchestrate.discover({"query": "idea", "force": False})
        wb.assert_called_once_with("idea")
        disc.assert_called_once_with("idea", force=False, alternates=("alt one", "alt two", "alt three"))
        self.assertEqual(self.types()[:2], ["narration", "brief"])     # a line before the model call, then the brief
        self.assertEqual(self.events()[1]["stage"], "brief")
        self.assertEqual(self.events()[1]["payload"]["alternates"], ["alt one", "alt two", "alt three"])
        self.assertIn("stopped", res)

    def test_brief_off_or_failed_means_no_brief_event(self):
        runlog.start_run("pipeline", {})
        with patch.object(orchestrate._config, "BRIEF", False), \
             patch("research_assistant.shared.brief.write_brief") as wb, \
             patch.object(orchestrate.agent0_discoverer, "discover", return_value=None):
            orchestrate.discover({"query": "idea"})
        wb.assert_not_called()
        self.assertNotIn("brief", self.types())

        with patch("research_assistant.shared.brief.write_brief", return_value=None), \
             patch.object(orchestrate.agent0_discoverer, "discover", return_value=None) as disc:
            orchestrate.discover({"query": "idea"})
        self.assertEqual(disc.call_args.kwargs["alternates"], ())
        self.assertIn("I'll search for it as you wrote it.", [e["payload"].get("text") for e in self.events()])

    def test_file_seed_gets_a_narration_and_seed_found_not_a_brief(self):
        runlog.start_run("pipeline", {})
        pdf = os.path.join(self.tmp.name, "hall.pdf")
        open(pdf, "wb").write(b"%PDF-1.4 x")
        with patch("research_assistant.shared.brief.write_brief") as wb, \
             patch.object(orchestrate.agent0_discoverer, "discover_from_file", return_value=pdf), \
             patch.object(orchestrate.agent0_discoverer, "get_seed", return_value={"title": "Hall", "key": "k", "source": "upload"}):
            res = orchestrate.discover({"query": "q", "seed_file": pdf})
        wb.assert_not_called()
        self.assertEqual(res["seed_label"], "Hall")
        found = [e for e in self.events() if e["type"] == "seed_found"][0]["payload"]
        self.assertEqual((found["title"], found["source"], found["via_query"]), ("Hall", "upload", None))
        self.assertTrue(any("You gave me the paper" in (e["payload"].get("text") or "") for e in self.events()))


class TestExtractAndFetch(NarratedCase):
    def test_references_extracted_event_from_agent1s_file(self):
        runlog.start_run("pipeline", {})
        path = os.path.join(self.tmp.name, "extracted.json")
        refs = [{"title": f"Ref {i}", "authors": ["A", "B", "C", "D"], "year": 2000 + i, "doi": "10.1/x" if i % 2 else None,
                 "doi_confidence": "high", "source_file": "seed.pdf", "raw_reference": "r"} for i in range(12)]
        with open(path, "w") as fh:
            json.dump({"references": refs, "summary": {"references": 12, "with_doi": 6}, "extraction": {"grobid": 1}}, fh)
        with patch.object(orchestrate._config, "EXTRACTED_CITATIONS_PATH", path), \
             patch.object(orchestrate.agent1_extractor, "run_extractor", return_value={"reference_count": 12}):
            res = orchestrate.extract({})
        self.assertTrue(res["references_ok"])
        ev = [e for e in self.events() if e["type"] == "references_extracted"][0]
        self.assertEqual(ev["stage"], "extract")
        p = ev["payload"]
        self.assertEqual((p["n"], p["with_doi"], p["by_method"]), (12, 6, {"grobid": 1}))
        self.assertEqual(len(p["sample"]), 10)
        self.assertEqual(p["sample"][0], {"authors": ["A", "B", "C"], "year": 2000, "title": "Ref 0"})
        self.assertIn("12 references, 6 with a DOI", self.events()[-1]["payload"]["text"])

    def test_fetch_narrates_from_the_outcome(self):
        runlog.start_run("pipeline", {})
        with patch.object(orchestrate.agent2_fetcher, "fetch_papers", return_value={"fetched": 3, "unavailable": 2}):
            orchestrate.fetch({})
        self.assertIn("3 fetched, 2 out of reach", self.events()[-1]["payload"]["text"])
        with patch.object(orchestrate.agent2_fetcher, "fetch_papers", return_value=None):
            orchestrate.fetch({})                     # an older fetch_papers returning None is fine
        self.assertIn("Nothing new to fetch", self.events()[-1]["payload"]["text"])


class TestRunEndToEnd(NarratedCase):
    def _patches(self, discover_path):
        return [
            patch.object(orchestrate._config, "BRIEF", False),
            patch.object(orchestrate.agent0_discoverer, "discover", return_value=discover_path),
            patch.object(orchestrate.agent0_discoverer, "get_seed", return_value={"title": "Seed", "key": "k"}),
            patch.object(orchestrate, "ingest_pdfs", return_value={"processed": 1, "inserted": 3}),
            patch.object(orchestrate.agent1_extractor, "run_extractor", return_value={"reference_count": 0}),
        ]

    def test_cli_run_opens_and_closes_its_own_transcript(self):
        pdf = os.path.join(self.tmp.name, "seed.pdf")
        open(pdf, "wb").write(b"%PDF-1.4")
        patches = self._patches(pdf)
        for p in patches:
            p.start(); self.addCleanup(p.stop)
        with patch.object(orchestrate._config, "EXTRACTED_CITATIONS_PATH", os.path.join(self.tmp.name, "none.json")):
            code = orchestrate.run(query="idea", ask=False)
        self.assertEqual(code, 1)                                    # stopped: no references
        meta = runlog.read_meta(runlog.current_run_id())
        self.assertEqual(meta["kind"], "pipeline")
        self.assertEqual(meta["inputs"]["query"], "idea")
        self.assertEqual(meta["status"], "stopped")
        self.assertIn("No references", meta["summary"])
        self.assertEqual(self.types()[-1], "finished")
        self.assertIn("stopped", self.types())
        self.assertIsNone(runlog.bound_run_id())

    def test_run_joins_a_run_the_launcher_started(self):
        pdf = os.path.join(self.tmp.name, "seed.pdf")
        open(pdf, "wb").write(b"%PDF-1.4")
        for p in self._patches(pdf):
            p.start(); self.addCleanup(p.stop)
        rid = runlog.start_run("pipeline", {"query": "idea"})
        with patch.object(orchestrate._config, "EXTRACTED_CITATIONS_PATH", os.path.join(self.tmp.name, "none.json")):
            orchestrate.run(query="idea", ask=False)
        self.assertEqual(runlog.current_run_id(), rid)              # no second run was opened
        self.assertEqual(runlog.read_meta(rid)["status"], "running")   # the launcher finishes it
        self.assertEqual(runlog.finish_run()["status"], "stopped")

    def test_stop_before_a_node_ends_the_run_cancelled(self):
        pdf = os.path.join(self.tmp.name, "seed.pdf")
        open(pdf, "wb").write(b"%PDF-1.4")
        for p in self._patches(pdf):
            p.start(); self.addCleanup(p.stop)

        def discover_then_cancel(query, force=False, alternates=()):
            runlog.cancel(runlog.bound_run_id())
            return pdf

        with patch.object(orchestrate.agent0_discoverer, "discover", side_effect=discover_then_cancel), \
             patch.object(orchestrate, "ingest_pdfs") as ingest:
            with self.assertRaises(runlog.RunCancelled):
                orchestrate.run(query="idea", ask=False)
        ingest.assert_not_called()                                   # ingest_seed's check raised first
        meta = runlog.read_meta(runlog.current_run_id())
        self.assertEqual(meta["status"], "cancelled")
        status = pipeline_status.get_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["detail"], "Pipeline stopped by the reader")

    def test_fallback_emits_an_ungrounded_synthesis(self):
        runlog.start_run("pipeline", {})
        with patch("research_assistant.shared.llm.chat", return_value=MagicMock(content="From memory.")):
            res = orchestrate.fallback({"query": "idea"})
        self.assertTrue(res["answer"]["ungrounded"])
        syn = [e for e in self.events() if e["type"] == "synthesis"][0]["payload"]
        self.assertEqual((syn["text"], syn["ungrounded"]), ("From memory.", True))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_orchestrate_narrated.py
```

Expected: fail — `orchestrate._config` missing, no events

- [ ] **Step 6: Change `orchestrate.py`**

**Apply this change to `orchestrate.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/orchestrate.py
+++ b/orchestrate.py
@@ -45,10 +45,11 @@
 from research_assistant.agents import (
     agent0_discoverer, agent1_extractor, agent2_fetcher, agent3_ingestor,
 )
+from research_assistant import config as _config
 from research_assistant.config import GROBID_SERVER
 from research_assistant.shared.ingestion import ingest_pdfs
 from research_assistant.shared.log import get_logger
-from research_assistant.shared import pipeline_status
+from research_assistant.shared import narration, pipeline_status, runlog
 
 logger = get_logger("orchestrate")
 
@@ -91,12 +92,36 @@
 # ─── Nodes ───────────────────────────────────────────────────────────────────
 
 
+def _brief(query: str) -> tuple:
+    """The brief before a seed search (spec §2E): shown to the reader, and
+    its phrasings handed to Agent 0 as fallbacks. Off, failed or empty →
+    a one-line narration and no alternates."""
+    if not _config.BRIEF:
+        return ()
+    runlog.set_stage("brief")
+    runlog.note(narration.brief_start())
+    from research_assistant.shared.brief import write_brief
+
+    out = write_brief(query)
+    if out:
+        runlog.emit("brief", text=out["text"], alternates=out["alternates"])
+        return tuple(out["alternates"])
+    runlog.note(narration.brief_fallback())
+    return ()
+
+
 def discover(state: PipelineState) -> dict:
+    runlog.check_cancelled()
     query = (state.get("query") or "").strip()
     force = state.get("force", False)
     seed_url = (state.get("seed_url") or "").strip()
     seed_file = (state.get("seed_file") or "").strip()
 
+    # Before the search — so the reader has something to read while the
+    # providers are queried. Only the search path gets a model brief; the
+    # file and URL paths get a templated line in the stage below.
+    alternates = _brief(query) if (query and not seed_file and not seed_url) else ()
+
     with pipeline_status.track_stage("discover", "Finding seed paper", current_step=1, total_steps=5):
         if seed_file:
             _banner("discover — seeding from uploaded / local PDF")
@@ -106,12 +131,13 @@
         elif seed_url:
             _banner("discover — seeding from the supplied link")
             pipeline_status.update_progress(detail=f"Downloading from link: {seed_url[:60]}", current_item_name=seed_url)
+            runlog.note(narration.url_seed(seed_url))
             path = agent0_discoverer.discover_from_url(query, seed_url, force=force)
             fail = f"could not download a PDF from {seed_url}"
         else:
             _banner("discover — finding a seed paper")
             pipeline_status.update_progress(detail=f"Searching literature for: {query[:50]}", current_item_name=query)
-            path = agent0_discoverer.discover(query, force=force)
+            path = agent0_discoverer.discover(query, force=force, alternates=alternates)
             fail = (
                 "no open-access PDF found for that query — supply an arXiv or "
                 "open-access PDF link or upload a PDF to seed from directly"
@@ -130,6 +156,15 @@
                 seed = {}
 
         label = seed.get("title") or seed.get("key") or os.path.basename(path)
+        if seed_file or seed_url:
+            # Agent 0's search path emits its own seed_found (it knows which
+            # phrasing worked); the file and URL paths are narrated here.
+            runlog.emit("seed_found", title=seed.get("title"), key=seed.get("key"), doi=seed.get("doi"),
+                        arxiv_id=seed.get("arxiv_id"), url=seed.get("url"),
+                        source=seed.get("source") or ("upload" if seed_file else "manual-url"),
+                        via_query=None, cached=False, path=path)
+            if seed_file:
+                runlog.note(narration.uploaded_seed(label))
         pipeline_status.add_event(f"✅ Found seed paper: {label[:60]}")
         pipeline_status.update_progress(current_item_name=label, detail=f"Seed paper identified: {label[:60]}")
         result = {"seed_path": path, "seed_label": label}
@@ -139,6 +174,7 @@
 
 
 def ingest_seed(state: PipelineState) -> dict:
+    runlog.check_cancelled()
     _banner("ingest_seed — indexing the seed paper itself")
     label = state.get("seed_label") or os.path.basename(state.get("seed_path", "seed"))
     with pipeline_status.track_stage("ingest_seed", "Indexing seed paper", current_step=2, total_steps=5):
@@ -154,10 +190,31 @@
         return {}
 
 
+def _references_event() -> None:
+    """The reference list Agent 1 just wrote, for the reader: count, DOI
+    coverage, how it was extracted, and the first ten entries."""
+    import json
+
+    try:
+        with open(_config.EXTRACTED_CITATIONS_PATH, encoding="utf-8") as fh:
+            payload = json.load(fh)
+    except (OSError, ValueError):
+        return
+    refs = payload.get("references") or [] if isinstance(payload, dict) else []
+    summary = payload.get("summary") or {} if isinstance(payload, dict) else {}
+    sample = [{"authors": list(r.get("authors") or [])[:3], "year": r.get("year"), "title": r.get("title")}
+              for r in refs[:10] if isinstance(r, dict)]
+    runlog.emit("references_extracted", n=len(refs), with_doi=int(summary.get("with_doi") or 0),
+                by_method=dict(payload.get("extraction") or {}), sample=sample)
+    runlog.note(narration.references(len(refs), int(summary.get("with_doi") or 0)))
+
+
 def extract(state: PipelineState) -> dict:
+    runlog.check_cancelled()
     _banner("extract — mining the seed's reference list")
     with pipeline_status.track_stage("extract", "Extracting reference list", current_step=3, total_steps=5, detail="Mining reference list from seed paper"):
         result = agent1_extractor.run_extractor()
+        _references_event()
 
         if not result or not result.get("reference_count"):
             # Both strategies came up empty: GROBID down *and* no recognisable numbered
@@ -174,13 +231,16 @@
 
 
 def fetch(state: PipelineState) -> dict:
+    runlog.check_cancelled()
     _banner("fetch — downloading the referenced papers (Agent 2)")
     with pipeline_status.track_stage("fetch", "Fetching referenced papers", current_step=4, total_steps=5, detail="Downloading open-access reference PDFs"):
-        agent2_fetcher.fetch_papers()
+        outcome = agent2_fetcher.fetch_papers() or {}
+        runlog.note(narration.fetch_done(int(outcome.get("fetched") or 0), int(outcome.get("unavailable") or 0)))
         return {}
 
 
 def ingest_refs(state: PipelineState) -> dict:
+    runlog.check_cancelled()
     _banner("ingest_refs — ingesting the reference PDFs (Agent 3)")
     with pipeline_status.track_stage("ingest_refs", "Ingesting and summarizing papers", current_step=5, total_steps=5, detail="Ingesting referenced papers into corpus"):
         agent3_ingestor.run_ingestor(
@@ -191,24 +251,30 @@
 
 
 def respond(state: PipelineState) -> dict:
+    runlog.check_cancelled()
     _banner("respond — related-work synthesis for the query")
-    from research_assistant.shared import retrieve  # imported here so corpus-building stays light
+    from research_assistant.shared.respond import research_answer_narrated  # keeps corpus-building light
 
     with pipeline_status.track_stage("respond", "Synthesizing answer", current_step=5, total_steps=5, detail=f"Synthesizing related-work response for: {state.get('query', '')[:50]}"):
         pipeline_status.add_event("✍️ Formulating related-work synthesis…")
-        result = retrieve.research_answer(state["query"])
+        result = research_answer_narrated(state["query"])
         pipeline_status.add_event("✅ Related-work synthesis complete")
 
         audit_result = None
         if state.get("audit_citations") and state.get("seed_path"):
+            runlog.check_cancelled()
             try:
                 from research_assistant.shared import seed_audit
                 pipeline_status.add_event("🔍 Auditing in-text citations from seed paper…")
+                # Its own stage on the transcript (`audit`); a stop lands between claims.
                 audit_result = seed_audit.audit_seed_citations(state["seed_path"])
                 pipeline_status.add_event("✅ Seed citation audit complete")
+            except runlog.RunCancelled:
+                raise
             except Exception as e:
                 logger.warning("Seed citation audit encountered an error: %s", e)
                 audit_result = {"error": str(e), "totals": {"total": 0, "judged": 0}, "results": []}
+                runlog.emit("audit_done", stage="audit", totals=audit_result["totals"], error=str(e), path=None)
 
         out = {"answer": result}
         if audit_result is not None:
@@ -218,9 +284,11 @@
 
 def fallback(state: PipelineState) -> dict:
     """No seed paper — answer the query from the model's own knowledge."""
+    runlog.check_cancelled()
     _banner("fallback — no corpus, answering from general knowledge")
     from research_assistant.prompts import NO_CORPUS_FALLBACK
     from research_assistant.shared.llm import chat
+    from research_assistant.shared.respond import synthesis_payload
 
     with pipeline_status.track_stage("respond", "Synthesizing fallback answer", current_step=5, total_steps=5, detail="Generating ungrounded answer from general knowledge"):
         pipeline_status.add_event("💭 Generating ungrounded fallback answer…")
@@ -231,7 +299,9 @@
         except Exception as e:  # noqa: BLE001
             logger.error("Fallback answer failed: %s", e)
             return {}
-        return {"answer": {"suggestion": text, "citations": [], "passages": [], "ungrounded": True}}
+        answer = {"suggestion": text, "citations": [], "passages": [], "ungrounded": True}
+        runlog.emit("synthesis", **synthesis_payload(answer, ungrounded=True))
+        return {"answer": answer}
 
 
 
@@ -289,6 +359,22 @@
     describe_figures: bool | None = None,
     audit_citations: bool = False,
 ) -> int:
+    inputs = {
+        "query": query, "workers": workers, "force": force, "ask": ask,
+        "seed_url": seed_url, "seed_file": seed_file, "describe_figures": describe_figures,
+        "audit_citations": audit_citations,
+    }
+    # The run's transcript (spec §2A–B). When the app's launcher has already
+    # started a run on this thread's behalf, this joins it; from the CLI or
+    # watch.py it opens one, so those runs read the same in the app.
+    with runlog.run("pipeline", inputs):
+        return _run_graph(inputs)
+
+
+def _run_graph(inputs: dict) -> int:
+    query = inputs["query"]
+    seed_url, seed_file = inputs["seed_url"], inputs["seed_file"]
+    ask = inputs["ask"]
     app = build_graph()
     thread_key = query or (os.path.abspath(seed_file) if seed_file else "") or (seed_url or "") or "run"
     thread_id = hashlib.sha1(thread_key.encode()).hexdigest()[:12]
@@ -307,29 +393,17 @@
 
     final: PipelineState = {}
     try:
-        for update in app.stream(
-            {
-                "query": query,
-                "workers": workers,
-                "force": force,
-                "ask": ask,
-                "seed_url": seed_url,
-                "seed_file": seed_file,
-                "describe_figures": describe_figures,
-                "audit_citations": audit_citations,
-            },
-            config=config,
-            stream_mode="values",
-        ):
+        for update in app.stream(dict(inputs), config=config, stream_mode="values"):
             final = update
     except BaseException as exc:
-        if isinstance(exc, KeyboardInterrupt):
-            pipeline_status.add_event("⚠️ Pipeline cancelled by user (SIGINT)")
+        if isinstance(exc, (KeyboardInterrupt, runlog.RunCancelled)):
+            how = "by the reader" if isinstance(exc, runlog.RunCancelled) else "by user (SIGINT)"
+            pipeline_status.add_event(f"⚠️ Pipeline stopped {how}")
             pipeline_status.set_status(
                 active=False,
                 stage="idle",
                 stage_label="Idle",
-                detail="Pipeline cancelled by user",
+                detail=f"Pipeline stopped {how}",
             )
         else:
             pipeline_status.add_event(f"❌ Pipeline failed: {exc}")
@@ -344,6 +418,8 @@
     stopped = final.get("stopped")
     if stopped:
         logger.warning("Pipeline stopped: %s", stopped)
+        runlog.emit("stopped", reason=stopped)
+        runlog.set_stopped(stopped)
         pipeline_status.add_event(f"⚠️ Pipeline stopped early: {stopped[:60]}")
         pipeline_status.set_status(
             active=False,
@@ -354,6 +430,7 @@
     else:
         effective_query = final.get("query") or query or final.get("seed_label") or "paper"
         now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
+        runlog.set_summary(f"{effective_query[:60]}")
         pipeline_status.add_event(f"✅ Pipeline completed: {effective_query[:40]}")
         pipeline_status.set_status(
             active=False,
```

- [ ] **Step 7: Drop the tests of the renderer Task 12 deletes**

**Apply this change to `tests/test_orchestrate_upload.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_orchestrate_upload.py
+++ b/tests/test_orchestrate_upload.py
@@ -135,39 +135,6 @@
         self.assertEqual(orchestrate._after_discover(state_stopped_no_ask), orchestrate.END)
 
 
-class TestAppSeedRender(unittest.TestCase):
-    def test_render_seed_and_downloads_skips_when_failed(self):
-        import app
-        # Mock streamlit container and markdown calls
-        with patch.object(app, "_manifest", return_value={"old query": {"title": "Old Paper", "path": "/p.pdf"}}):
-            with patch("streamlit.container") as mock_container:
-                # When run stopped before finding seed
-                final = {"seed_path": None, "stopped": "could not load seed PDF"}
-                app._render_seed_and_downloads("", final)
-                mock_container.assert_not_called()
-
-    def test_render_seed_and_downloads_shows_seed_when_successful(self):
-        import app
-        with patch.object(
-            app,
-            "_manifest",
-            side_effect=[
-                {"uploaded": {"title": "New Uploaded Paper", "path": "/path/raw/paper.pdf", "source": "upload"}},
-                {},  # downloaded
-                {},  # failed
-            ],
-        ):
-            with patch("streamlit.container") as mock_container:
-                with patch("streamlit.markdown") as mock_markdown:
-                    with patch("streamlit.caption"):
-                        final = {"seed_path": "/path/raw/paper.pdf", "seed_label": "New Uploaded Paper"}
-                        app._render_seed_and_downloads("uploaded", final)
-                        mock_container.assert_called()
-                        # Verify markdown rendered seed paper title
-                        rendered = [call.args[0] for call in mock_markdown.call_args_list if call.args]
-                        self.assertTrue(any("New Uploaded Paper" in r for r in rendered))
-
-
 class TestUploadStaging(unittest.TestCase):
     """An uploaded PDF must not accumulate a second copy under DATA_DIR.
 
```

- [ ] **Step 8: Run the orchestrator tests**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_orchestrate_narrated.py tests/test_orchestrate_upload.py tests/test_pipeline_status.py
```

Expected: all pass (18 + the status file)

- [ ] **Step 9: Commit**

```bash
git add research_assistant/shared/respond.py orchestrate.py tests/test_respond.py tests/test_orchestrate_narrated.py tests/test_orchestrate_upload.py
git commit -m "feat(orchestrate): the pipeline is a run — the brief, every node on the transcript, cancel between nodes, the synthesis narrated

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The multi-PDF upload as an `ingest` run, out of the page

**Files:**
- Create: `research_assistant/shared/batch_ingest.py`
- Test: `tests/test_batch_ingest.py`

**Interfaces:**
- Consumes: `ingestion.ingest_pdfs`, `respond.research_answer_narrated`, `runlog`, `pipeline_status.track_stage`.
- Produces: `batch_ingest.record_uploads(staged)` (downloaded.json under source `upload`) and `batch_ingest.ingest_uploaded(staged, query="", ask=True, force=False, describe_figures=None) -> {seed_path, seed_label, batch_uploaded, answer, ingest, query}`, a run of kind `ingest` whose inputs carry `papers`, `query`, `ask`, `force`, `describe_figures`, `titles`; emits `upload_staged {papers: [{title, filename, key, doi, arxiv_id}]}` in stage `ingest_refs`, then whatever ingestion emits (Task 5), then the synthesis (stage `respond`) when asked — a synthesis failure is a `synthesis` event with `error`, not a failed run.

This is Tab 1's multi-PDF branch (app.py lines 706–830 at the base commit) moved out of the page without its `st.*` calls. That branch also ended with `_corpus_stats.clear()` on a function that has no `.clear` — an `AttributeError` at the end of every multi-PDF upload — which goes away with it.

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_batch_ingest.py` with exactly this content:**

```python
"""batch_ingest — the upload branch as a narrated run (spec §2B, §2H)."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.shared import batch_ingest, pipeline_status, runlog


class BatchIngestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        runlog.set_runs_dir(os.path.join(self.tmp.name, "runs"))
        self.addCleanup(runlog.set_runs_dir, None)
        pipeline_status.set_status_path(os.path.join(self.tmp.name, "status.json"))
        self.addCleanup(pipeline_status.set_status_path, None)
        runlog._reset_for_tests()
        self.addCleanup(runlog._reset_for_tests)
        p = patch.object(batch_ingest.config, "DOWNLOADED_JSON_PATH", os.path.join(self.tmp.name, "downloaded.json"))
        p.start(); self.addCleanup(p.stop)
        self.staged = [
            {"path": os.path.join(self.tmp.name, "a.pdf"), "filename": "a.pdf", "title": "Paper A", "key": "doi:10.1/a", "doi": "10.1/a", "arxiv_id": None},
            {"path": os.path.join(self.tmp.name, "b.pdf"), "filename": "b.pdf", "title": "Paper B", "key": "file:abc", "doi": None, "arxiv_id": None},
        ]

    def events(self):
        return runlog.read_events(runlog.current_run_id())


class TestIngestUploaded(BatchIngestCase):
    def test_run_indexes_records_and_synthesises(self):
        calls = {}

        def fake_ingest(candidates, workers=1, skip_ingested=True, describe_figures=None):
            calls["candidates"] = candidates
            calls["skip"] = skip_ingested
            return {"processed": 2, "inserted": 7, "described": 0}

        answer = {"suggestion": "Both papers…", "citations": ["Paper A"], "passages": [], "selected": []}
        with patch("research_assistant.shared.ingestion.ingest_pdfs", fake_ingest), \
             patch("research_assistant.shared.respond.research_answer_narrated", return_value=answer) as ra:
            final = batch_ingest.ingest_uploaded(self.staged, query="", ask=True, force=True)
        self.assertEqual(calls["candidates"], {self.staged[0]["path"]: "Paper A", self.staged[1]["path"]: "Paper B"})
        self.assertFalse(calls["skip"])                          # force → re-ingest
        ra.assert_called_once_with("Paper A")                    # no query → first title
        self.assertEqual(final["query"], "Paper A")
        self.assertEqual(final["answer"], answer)
        self.assertEqual(final["ingest"]["inserted"], 7)
        with open(batch_ingest.config.DOWNLOADED_JSON_PATH) as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["doi:10.1/a"]["source"], "upload")
        self.assertEqual(manifest["file:abc"]["title"], "Paper B")
        meta = runlog.read_meta(runlog.current_run_id())
        self.assertEqual((meta["kind"], meta["status"], meta["summary"]), ("ingest", "done", "2 papers, 7 chunks"))
        self.assertEqual(meta["inputs"]["titles"], ["Paper A", "Paper B"])
        self.assertEqual(self.events()[0]["type"], "upload_staged")
        self.assertEqual(self.events()[0]["stage"], "ingest_refs")
        self.assertEqual(self.events()[0]["payload"]["papers"][1]["filename"], "b.pdf")
        self.assertFalse(pipeline_status.get_status()["active"])

    def test_no_ask_means_no_synthesis(self):
        with patch("research_assistant.shared.ingestion.ingest_pdfs", return_value={"processed": 2, "inserted": 1}), \
             patch("research_assistant.shared.respond.research_answer_narrated") as ra:
            final = batch_ingest.ingest_uploaded(self.staged, query="q", ask=False)
        ra.assert_not_called()
        self.assertIsNone(final["answer"])
        self.assertEqual(final["query"], "q")

    def test_synthesis_failure_is_recorded_not_raised(self):
        with patch("research_assistant.shared.ingestion.ingest_pdfs", return_value={"processed": 2, "inserted": 1}), \
             patch("research_assistant.shared.respond.research_answer_narrated", side_effect=RuntimeError("model down")):
            final = batch_ingest.ingest_uploaded(self.staged, query="q", ask=True)
        self.assertIn("model down", final["answer"]["suggestion"])
        syn = [e for e in self.events() if e["type"] == "synthesis"][0]["payload"]
        self.assertEqual(syn["error"], "model down")
        self.assertEqual(runlog.read_meta(runlog.current_run_id())["status"], "done")

    def test_ingest_failure_fails_the_run(self):
        with patch("research_assistant.shared.ingestion.ingest_pdfs", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                batch_ingest.ingest_uploaded(self.staged, ask=False)
        meta = runlog.read_meta(runlog.current_run_id())
        self.assertEqual((meta["status"], meta["error"]), ("failed", "RuntimeError: disk full"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_batch_ingest.py
```

Expected: FAIL — module not found

- [ ] **Step 3: Write the module**

**Create `research_assistant/shared/batch_ingest.py` with exactly this content:**

```python
"""
batch_ingest — the multi-PDF upload as one narrated run
(spec 2026-09-12-narrated-runs §2B, §2H).

This is Tab 1's "several PDFs or a ZIP" branch moved out of the page: stage
the uploads (batch_uploader does that, in the page, before launch), index
them, record them in downloaded.json so the manifests and the sidebar see
them, and — when asked — synthesise across them. The page launches it on a
thread and reads the transcript; nothing here touches Streamlit.
"""

from __future__ import annotations

import json
import os
import time

from research_assistant import config
from research_assistant.shared import narration, pipeline_status, runlog
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger

logger = get_logger("batch_ingest")


def record_uploads(staged: list[dict]) -> None:
    """Add the staged papers to Agent 2's manifest (downloaded.json) under
    source "upload", so they list with the fetched ones."""
    manifest = {}
    if os.path.exists(config.DOWNLOADED_JSON_PATH):
        try:
            with open(config.DOWNLOADED_JSON_PATH, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read downloaded.json: %s", exc)
            manifest = {}
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for p in staged:
        k = p.get("key") or os.path.basename(p["path"])
        manifest[k] = {
            "key": k,
            "title": p.get("title") or os.path.basename(p["path"]),
            "doi": p.get("doi"),
            "arxiv_id": p.get("arxiv_id"),
            "path": p["path"],
            "source": "upload",
            "fetched_at": now_iso,
        }
    atomic_write_json(config.DOWNLOADED_JSON_PATH, manifest, ensure_ascii=False)


def ingest_uploaded(staged: list[dict], query: str = "", ask: bool = True, force: bool = False,
                    describe_figures: bool | None = None) -> dict:
    """Index *staged* uploads; synthesise for *query* (or the first title)
    when *ask*. Returns {seed_path, seed_label, batch_uploaded, answer,
    ingest, query}. The transcript carries the same as events."""
    from research_assistant.shared.ingestion import ingest_pdfs

    candidates = {p["path"]: (p.get("title") or p.get("key") or os.path.basename(p["path"])) for p in staged}
    inputs = {"papers": len(staged), "query": query, "ask": ask, "force": force,
              "describe_figures": describe_figures,
              "titles": [candidates[p["path"]] for p in staged][:20]}

    with runlog.run("ingest", inputs):
        runlog.set_stage("ingest_refs")
        runlog.emit("upload_staged", papers=[{
            "title": candidates[p["path"]], "filename": p.get("filename") or os.path.basename(p["path"]),
            "key": p.get("key"), "doi": p.get("doi"), "arxiv_id": p.get("arxiv_id"),
        } for p in staged])

        with pipeline_status.track_stage(
            "ingest_refs", f"Ingesting {len(staged)} uploaded papers",
            current_step=5, total_steps=5, item_total=len(candidates),
            mark_idle_on_exit=not ask, last_summary=f"+{len(candidates)} uploaded papers indexed",
        ):
            ingest_res = ingest_pdfs(candidates, workers=1, skip_ingested=not force,
                                     describe_figures=describe_figures)
        try:
            record_uploads(staged)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update downloaded.json: %s", exc)

        answer = None
        effective_q = (query or "").strip() or (candidates[staged[0]["path"]] if staged else "")
        if ask and effective_q:
            from research_assistant.shared.respond import research_answer_narrated

            with pipeline_status.track_stage(
                "respond", "Synthesizing answer", current_step=5, total_steps=5,
                detail=f"Synthesizing across {len(staged)} uploaded papers",
                mark_idle_on_exit=True, last_summary=f"+{len(candidates)} uploaded papers indexed",
            ):
                try:
                    answer = research_answer_narrated(effective_q)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Synthesis failed: %s", exc)
                    runlog.emit("synthesis", text=f"Synthesis encountered an error: {exc}", citations=[],
                                keys={}, unverified_citations=[], irrelevant_cited=[], passages=[],
                                selected=[], mode=None, timings={}, ungrounded=False, error=str(exc))
                    answer = {"suggestion": f"Synthesis encountered an error: {exc}", "citations": [], "passages": []}

        runlog.set_summary(f"{ingest_res.get('processed', 0)} papers, {ingest_res.get('inserted', 0)} chunks")
        return {
            "seed_path": staged[0]["path"] if staged else None,
            "seed_label": f"Uploaded collection ({len(staged)} papers)",
            "batch_uploaded": staged,
            "answer": answer,
            "ingest": ingest_res,
            "query": effective_q or "Uploaded paper collection",
        }
```

- [ ] **Step 4: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_batch_ingest.py
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/batch_ingest.py tests/test_batch_ingest.py
git commit -m "feat(ingest): the multi-PDF upload is an ingest run, out of the page

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: `cite`, `verify` and `audit` runs — one event per item, a stop between items

**Files:**
- Modify: `research_assistant/agents/agent5_batch_citer.py`
- Modify: `research_assistant/agents/agent8_verifier.py`
- Modify: `research_assistant/shared/seed_audit.py`
- Test: `tests/test_batch_citer.py`, `tests/test_verifier.py`, `tests/test_seed_audit.py`

**Interfaces:**
- Consumes: `runlog`, `narration`.
- Produces: `run_batch_citer` wraps `_run_batch_citer` in `runlog.run("cite", {draft, out_path})`, stage `cite`; emits `draft_split {n_sentences, n_eligible}` + narration, per sentence `sentence_decided {index, sentence, needs_citation, cited, cited_text, skip_reason, reasoning, sources: [{key, citation}]}` through a local `_record(i, entry)` at every `citation_entries.append` site, `draft_written {path, n_cited, n_sources, unknown_keys}`; an aborted need-check is `set_stopped(...)` (status `stopped`); summary "k of n sentences cited". `verify_draft` wraps `_verify_draft` in `runlog.run("verify", {draft, citations})`, stage `verify`; emits `verify_started {n, n_sentences}` + narration, per citation `citation_judged {index, sentence_index, claim, cite_key, citation_source, outcome, judgement, confidence, reason, escalated, rubric_mismatch, compound_sentence}` through `_record(i, entry)` at every `results.append` site, `verify_done {totals, json_path, md_path}`; summary "k of n citations supported". `audit_seed_citations` wraps `_audit_seed_citations` in `runlog.run("audit", {seed_path, max_claims})`, stage `audit` (joined when it runs inside the pipeline's run — its events then sit in the pipeline's transcript under the audit section); emits `audit_started {claims, downloaded, to_judge, seed}` + narration, per claim `audit_item {index, ref_index, ref_title, ref_year, ref_authors, claim, sentence, downloaded, document, outcome, judgement, confidence, reason, supporting_span}` — judged ones as each verdict lands, the unjudged and not-downloaded ones after the loop so the reader's count adds up — and `audit_done {totals, error, path}` (also emitted, with `error`, when there is no TEI or no claims). `check_cancelled()` at the top of each loop.

The three loops each had several `append(entry); continue` sites; the helper keeps the emit next to the append so no outcome is missed.

- [ ] **Step 1: Add the failing tests for the three loops** (`narration.auditing` already exists from Task 3)

**Apply this change to `tests/test_batch_citer.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_batch_citer.py
+++ b/tests/test_batch_citer.py
@@ -291,3 +291,77 @@
 
 if __name__ == "__main__":
     unittest.main()
+
+
+# ─── Narrated runs: sentence_decided as each sentence is decided (spec §2D) ──
+
+
+class TestBatchCiterTranscript(unittest.TestCase):
+    DRAFT = ("Graphene exhibits ballistic transport at low temperature. "
+             "Short one. "
+             "Silicon is a semiconductor with an indirect band gap.")
+
+    def _run(self, chat_replies, hits, cancel_on_sentence=None):
+        from research_assistant.shared import runlog
+
+        tmp = tempfile.TemporaryDirectory()
+        self.addCleanup(tmp.cleanup)
+        draft_path = os.path.join(tmp.name, "draft.txt")
+        with open(draft_path, "w") as f:
+            f.write(self.DRAFT)
+        out_path = os.path.join(tmp.name, "cited.txt")
+
+        def search(sentence, *a, **k):
+            if cancel_on_sentence and sentence.startswith(cancel_on_sentence):
+                runlog.cancel(runlog.bound_run_id())
+            return hits
+
+        with patch("research_assistant.agents.agent5_batch_citer.load_search_resources", return_value=(None, None, [], [])), \
+             patch("research_assistant.agents.agent5_batch_citer.hybrid_search", side_effect=search), \
+             patch("research_assistant.agents.agent5_batch_citer.chat", side_effect=chat_replies):
+            result = run_batch_citer(draft_path, out_path)
+        return result, runlog.read_meta(runlog.current_run_id()), runlog.read_events(runlog.current_run_id())
+
+    def test_events_in_order_with_the_decision_in_each(self):
+        hits = [{"text": "Ballistic transport in graphene…", "metadata": {"citation_source": "Doe (2020)"}}]
+        result, meta, events = self._run([
+            _reply("1. YES\n2. NO"),                                          # need check: 2 eligible sentences
+            _reply("CITED: Graphene exhibits ballistic transport at low temperature \\cite{cite_1}.\nREASON: supported"),
+        ], hits)
+        self.assertTrue(result)
+        self.assertEqual(meta["kind"], "cite")
+        self.assertEqual(meta["status"], "done")
+        self.assertEqual(meta["summary"], "1 of 3 sentences cited")
+        self.assertEqual(meta["inputs"]["out_path"], result)
+        types = [e["type"] for e in events]
+        self.assertEqual(types, ["draft_split", "narration", "sentence_decided", "sentence_decided",
+                                 "sentence_decided", "draft_written", "finished"])
+        self.assertEqual(events[0]["payload"], {"n_sentences": 3, "n_eligible": 2})
+        s0, s1, s2 = (e["payload"] for e in events[2:5])
+        self.assertTrue(s0["cited"])
+        self.assertIn("\\cite{cite_1}", s0["cited_text"])
+        self.assertEqual(s0["sources"], [{"key": "cite_1", "citation": "Doe (2020)"}])
+        self.assertEqual((s1["needs_citation"], s1["skip_reason"]), (False, "too short"))
+        self.assertEqual((s2["needs_citation"], s2["skip_reason"]), (False, "no citation needed"))
+        self.assertEqual(events[5]["payload"]["n_cited"], 1)
+        self.assertEqual(events[5]["payload"]["path"], result)
+
+    def test_aborted_need_check_is_a_stopped_run(self):
+        result, meta, events = self._run([RuntimeError("misaligned")] * 6, [])
+        self.assertIsNone(result)
+        self.assertEqual(meta["status"], "stopped")
+        self.assertIn("could not be aligned", meta["summary"])
+        self.assertNotIn("sentence_decided", [e["type"] for e in events])
+
+    def test_stop_lands_between_sentences(self):
+        from research_assistant.shared import runlog
+
+        hits = [{"text": "…", "metadata": {"citation_source": "Doe (2020)"}}]
+        with self.assertRaises(runlog.RunCancelled):
+            self._run([_reply("1. YES\n2. YES"),
+                       _reply("CITED: Graphene exhibits ballistic transport at low temperature \\cite{cite_1}.\nREASON: ok")],
+                      hits, cancel_on_sentence="Graphene")
+        events = runlog.read_events(runlog.current_run_id())
+        decided = [e for e in events if e["type"] == "sentence_decided"]
+        self.assertEqual(len(decided), 1)                  # sentence 1 finished; sentence 2 never started
+        self.assertEqual(runlog.read_meta(runlog.current_run_id())["status"], "cancelled")
```

**Apply this change to `tests/test_verifier.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_verifier.py
+++ b/tests/test_verifier.py
@@ -772,3 +772,52 @@
             report, calls, hs = self._run_with([first])
         self.assertEqual(len(calls), 1)
         self.assertEqual(hs.call_args.kwargs["top_k"], 1)
+
+
+# ─── Narrated runs: citation_judged as each verdict lands (spec §2D) ────────
+
+
+class TestVerifierTranscript(VerifyDraftTestCase):
+    DRAFT = "Claim one \\cite{cite_1}. Claim two \\cite{cite_2}. Claim three \\cite{cite_9}."
+    MAPPING = {"Doe (2020)": "cite_1", "Roe (2021)": "cite_2"}
+
+    def _rows(self):
+        return [("Doe (2020)", "doe.pdf"), ("Roe (2021)", "roe.pdf")]
+
+    def test_events_and_summary(self):
+        from research_assistant.shared import runlog
+
+        hits = [{"text": "evidence", "metadata": {"document": "doe.pdf", "chunk_index": 0}}]
+        verdicts = iter([_verdict("Supports"), _verdict("Contradicts")])
+        draft_path, report = self._run(self.DRAFT, self.MAPPING, self._rows(), hits,
+                                       judge_side_effect=lambda c, e, **k: next(verdicts))
+        meta = runlog.read_meta(runlog.current_run_id())
+        self.assertEqual((meta["kind"], meta["status"]), ("verify", "done"))
+        self.assertEqual(meta["summary"], "1 of 3 citations supported")
+        events = runlog.read_events(runlog.current_run_id())
+        types = [e["type"] for e in events]
+        self.assertEqual(types, ["verify_started", "narration", "citation_judged", "citation_judged",
+                                 "citation_judged", "verify_done", "finished"])
+        self.assertEqual(events[0]["payload"]["n"], 3)
+        j1, j2, j3 = (e["payload"] for e in events[2:5])
+        self.assertEqual((j1["cite_key"], j1["outcome"], j1["judgement"]), ("cite_1", "judged", "Supports"))
+        self.assertEqual((j2["outcome"], j2["judgement"]), ("judged", "Contradicts"))
+        self.assertEqual((j3["cite_key"], j3["outcome"], j3["judgement"]), ("cite_9", "orphaned", None))
+        self.assertEqual(events[5]["payload"]["totals"]["total"], 3)
+        self.assertTrue(events[5]["payload"]["md_path"].endswith("_verification.md"))
+
+    def test_stop_lands_between_citations(self):
+        from research_assistant.shared import runlog
+
+        hits = [{"text": "evidence", "metadata": {"document": "doe.pdf", "chunk_index": 0}}]
+
+        def judge(claim, evidence, **k):
+            runlog.cancel(runlog.bound_run_id())
+            return _verdict()
+
+        with self.assertRaises(runlog.RunCancelled):
+            self._run(self.DRAFT, self.MAPPING, self._rows(), hits, judge_side_effect=judge)
+        events = runlog.read_events(runlog.current_run_id())
+        self.assertEqual(sum(1 for e in events if e["type"] == "citation_judged"), 1)
+        self.assertEqual(runlog.read_meta(runlog.current_run_id())["status"], "cancelled")
+        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "cited_draft_verification.json")))
```

**Apply this change to `tests/test_seed_audit.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/tests/test_seed_audit.py
+++ b/tests/test_seed_audit.py
@@ -204,3 +204,78 @@
 
 if __name__ == "__main__":
     unittest.main()
+
+
+# ─── Narrated runs: audit_item as each claim is judged (spec §2D) ───────────
+
+
+class TestSeedAuditTranscript(unittest.TestCase):
+    def _run(self, judge, cancel_on_first=False, both_downloaded=False):
+        from research_assistant.shared import runlog
+
+        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
+            tf.write(SAMPLE_TEI_XML)
+            tei_file = tf.name
+        self.addCleanup(os.unlink, tei_file)
+        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
+            dummy_pdf_path = dummy_pdf.name
+        self.addCleanup(os.unlink, dummy_pdf_path)
+        manifest = {"doi:10.1126/science.1102896": {
+            "key": "doi:10.1126/science.1102896", "path": dummy_pdf_path, "doi": "10.1126/science.1102896",
+            "title": "Electric field effect in atomically thin carbon films", "cited_by": "seed.pdf", "xml_id": "b0"}}
+        if both_downloaded:                       # the second claim's paper ([2], matched by title) is here too
+            manifest["title:b1"] = {"key": "title:b1", "path": dummy_pdf_path,
+                                    "title": "Disorder-assisted quantum inversion in nanoribbons"}
+        hits = [{"text": "We observed strong electric field effect in graphene.",
+                 "metadata": {"document": os.path.basename(dummy_pdf_path)}}]
+
+        def judge_side_effect(claim, evidence):
+            if cancel_on_first:
+                runlog.cancel(runlog.bound_run_id())
+            return judge
+
+        with patch("research_assistant.shared.seed_audit._load_downloaded_manifest", return_value=manifest), \
+             patch("research_assistant.shared.seed_audit.find_tei_for_seed", return_value=tei_file), \
+             patch("research_assistant.shared.seed_audit.hybrid_search", return_value=hits), \
+             patch("research_assistant.shared.seed_audit._judge_once", side_effect=judge_side_effect), \
+             patch("research_assistant.shared.seed_audit.AUDIT_DIR", tempfile.mkdtemp()):
+            report = audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []), max_claims=5)
+        return report, runlog
+
+    def test_events_cover_every_claim_and_the_run_is_done(self):
+        verdict = {"judgement": "Supports", "confidence": "High", "supporting_span": "s", "reason": "r",
+                   "slots": {"finding": "Supports", "scope": "Supports", "strength": "Supports"},
+                   "evidence_sufficiency": "sufficient"}
+        report, runlog = self._run(verdict)
+        meta = runlog.read_meta(runlog.current_run_id())
+        self.assertEqual((meta["kind"], meta["status"]), ("audit", "done"))
+        self.assertEqual(meta["summary"], "1 of 1 judged citations supported")
+        events = runlog.read_events(runlog.current_run_id())
+        types = [e["type"] for e in events]
+        self.assertEqual(types, ["audit_started", "narration", "audit_item", "audit_item", "audit_done", "finished"])
+        self.assertTrue(all(e["stage"] == "audit" for e in events[:-1]))
+        self.assertEqual(events[0]["payload"], {"claims": 2, "downloaded": 1, "to_judge": 1, "seed": "seed.pdf"})
+        judged, not_here = events[2]["payload"], events[3]["payload"]
+        self.assertEqual((judged["outcome"], judged["judgement"], judged["downloaded"]), ("judged", "Supports", True))
+        self.assertEqual(judged["ref_title"], "Electric field effect in atomically thin carbon films")
+        self.assertEqual((not_here["outcome"], not_here["downloaded"]), ("not_downloaded", False))
+        self.assertEqual(events[4]["payload"]["totals"]["total"], 2)
+        self.assertTrue(events[4]["payload"]["path"].endswith("_audit.json"))
+
+    def test_stop_lands_between_claims(self):
+        from research_assistant.shared import runlog
+
+        with self.assertRaises(runlog.RunCancelled):
+            self._run({"judgement": "Supports", "confidence": "High"}, cancel_on_first=True, both_downloaded=True)
+        events = runlog.read_events(runlog.current_run_id())
+        self.assertEqual(sum(1 for e in events if e["type"] == "audit_item"), 1)   # the call in flight finished; claim 2 never started
+        self.assertEqual(runlog.read_meta(runlog.current_run_id())["status"], "cancelled")
+
+    def test_missing_tei_is_an_audit_done_with_error(self):
+        from research_assistant.shared import runlog
+
+        report = audit_seed_citations("/non/existent/path.pdf")
+        self.assertIn("error", report)
+        events = runlog.read_events(runlog.current_run_id())
+        self.assertEqual([e["type"] for e in events], ["audit_done", "finished"])
+        self.assertIn("No GROBID TEI", events[0]["payload"]["error"])
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_batch_citer.py tests/test_verifier.py tests/test_seed_audit.py -k Transcript
```

Expected: fail — no runs, no events

- [ ] **Step 3: Change the batch citer**

**Apply this change to `research_assistant/agents/agent5_batch_citer.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/agents/agent5_batch_citer.py
+++ b/research_assistant/agents/agent5_batch_citer.py
@@ -14,6 +14,7 @@
 from research_assistant.shared.search import hybrid_search
 from research_assistant.shared.retry import retry
 from research_assistant.shared.llm import chat
+from research_assistant.shared import narration, runlog
 
 logger = get_logger("agent5")
 
@@ -287,6 +288,11 @@
     ``_citations.json`` key mapping for BibTeX, and a ``_report.md`` explaining
     each decision.
 
+    A narrated run of kind ``cite`` (spec 2026-09-12-narrated-runs §2B): one
+    ``sentence_decided`` event per sentence as it is decided, so the page
+    shows the draft being cited rather than a spinner. A stop lands between
+    sentences.
+
     Returns:
         str | None: *out_path* if the draft was written, None if the run was
         aborted before producing output.
@@ -295,6 +301,12 @@
         logger.error("File %s not found.", file_path)
         return None
 
+    with runlog.run("cite", {"draft": os.path.abspath(file_path), "out_path": os.path.abspath(out_path)}):
+        runlog.set_stage("cite")
+        return _run_batch_citer(file_path, out_path, search_resources)
+
+
+def _run_batch_citer(file_path, out_path, search_resources):
     with open(file_path, "r") as f:
         draft_text = f.read()
 
@@ -309,6 +321,8 @@
     # ── Batch citation-need check ────────────────────────────────────────
     eligible_indices = [i for i, s in enumerate(sentences) if len(s.split()) >= 4]
     eligible_sentences = [sentences[i] for i in eligible_indices]
+    runlog.emit("draft_split", n_sentences=len(sentences), n_eligible=len(eligible_sentences))
+    runlog.note(narration.citing(len(sentences), len(eligible_sentences)))
 
     needs_cite = [False] * len(sentences)
     if eligible_sentences:
@@ -324,6 +338,7 @@
                 "  → Aborting without writing output. Re-run to try again, or "
                 "shorten the draft if the model keeps truncating its reply.", e,
             )
+            runlog.set_stopped("the citation-need check could not be aligned to the draft")
             return None
         for idx, needs in zip(eligible_indices, batch_results):
             needs_cite[idx] = needs
@@ -334,7 +349,19 @@
     citation_entries = []  # For the report
     next_cite_idx = 1
 
+    def _record(i, entry):
+        """Keep the entry for the report and put the decision on the transcript."""
+        citation_entries.append(entry)
+        runlog.emit(
+            "sentence_decided", index=i, sentence=entry["original"], needs_citation=bool(needs_cite[i]),
+            cited=bool(entry.get("cited")), cited_text=entry.get("cited_text") or entry["original"],
+            skip_reason=entry.get("skip_reason"), reasoning=entry.get("reasoning"),
+            sources=[{"key": s["key"], "citation": s["citation"]}
+                     for s in (entry.get("sources") or entry.get("candidates") or [])],
+        )
+
     for i, sentence in enumerate(sentences):
+        runlog.check_cancelled()               # between sentences, never mid-call
         logger.info("[%d/%d] %s", i + 1, len(sentences), sentence[:80])
         entry = {"original": sentence, "cited": False}
 
@@ -343,7 +370,7 @@
             logger.info(" -> %s, skipping.", reason)
             cited_sentences.append(sentence)
             entry["skip_reason"] = reason
-            citation_entries.append(entry)
+            _record(i, entry)
             continue
 
         logger.info(" -> Needs citation. Searching context…")
@@ -353,7 +380,7 @@
             logger.info(" -> No context found.")
             cited_sentences.append(sentence)
             entry["skip_reason"] = "no relevant context found in database"
-            citation_entries.append(entry)
+            _record(i, entry)
             continue
 
         context_str = ""
@@ -395,7 +422,7 @@
             cited_sentences.append(sentence)
             entry["skip_reason"] = f"LLM error: {e}"
 
-        citation_entries.append(entry)
+        _record(i, entry)
 
     # ── Write outputs ────────────────────────────────────────────────────
     final_draft = " ".join(cited_sentences)
@@ -429,6 +456,10 @@
     with open(mapping_file, "w") as f:
         json.dump(citation_mapping, f, indent=4)
     logger.info("Saved citation mapping to %s", mapping_file)
+    n_cited = sum(1 for e in citation_entries if e.get("cited"))
+    runlog.emit("draft_written", path=os.path.abspath(out_path), n_cited=n_cited,
+                n_sources=len(citation_mapping), unknown_keys=sorted(unknown))
+    runlog.set_summary(f"{n_cited} of {len(sentences)} sentences cited")
 
     # ── Generate citation reasoning report ───────────────────────────────
     report_path = out_path.replace(".txt", "_report.md")
```

- [ ] **Step 4: Change the verifier**

**Apply this change to `research_assistant/agents/agent8_verifier.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/agents/agent8_verifier.py
+++ b/research_assistant/agents/agent8_verifier.py
@@ -47,6 +47,7 @@
 from research_assistant.shared.log import get_logger
 from research_assistant.shared.retry import retry
 from research_assistant.shared.search import expand_neighbours, hybrid_search
+from research_assistant.shared import narration, runlog
 
 logger = get_logger("agent8")
 
@@ -226,6 +227,10 @@
     Writes ``<draft>_verification.json`` and ``<draft>_verification.md``
     alongside the draft, matching agent 5's output naming.
 
+    A narrated run of kind ``verify`` (spec 2026-09-12-narrated-runs §2B):
+    one ``citation_judged`` event per citation as its verdict lands. A stop
+    lands between citations.
+
     Returns:
         dict: the same record written to the JSON file.
     """
@@ -238,6 +243,12 @@
             "resolve \\cite keys back to sources — run Agent 5 first."
         )
 
+    with runlog.run("verify", {"draft": os.path.abspath(draft_path), "citations": os.path.abspath(citations_path)}):
+        runlog.set_stage("verify")
+        return _verify_draft(draft_path, citations_path, top_k, search_resources)
+
+
+def _verify_draft(draft_path, citations_path, top_k, search_resources) -> dict:
     with open(draft_path, encoding="utf-8") as fh:
         draft_text = fh.read()
     with open(citations_path, encoding="utf-8") as fh:
@@ -260,24 +271,38 @@
             len(pairs), len(sentences),
         )
         _warn_if_context_is_tight()
+    runlog.emit("verify_started", n=len(pairs), n_sentences=len(sentences))
+    runlog.note(narration.verifying(len(pairs)))
 
     doc_cache = {}
     results = []
 
+    def _record(i, entry):
+        """Keep the entry and put its verdict on the transcript."""
+        results.append(entry)
+        runlog.emit(
+            "citation_judged", index=i, sentence_index=entry.get("sentence_index"), claim=entry.get("claim"),
+            cite_key=entry.get("cite_key"), citation_source=entry.get("citation_source"),
+            outcome=entry.get("outcome"), judgement=entry.get("judgement"), confidence=entry.get("confidence"),
+            reason=entry.get("reason"), escalated=bool(entry.get("escalated")),
+            rubric_mismatch=bool(entry.get("rubric_mismatch")), compound_sentence=bool(entry.get("compound_sentence")),
+        )
+
     for i, pair in enumerate(pairs, 1):
+        runlog.check_cancelled()               # between citations, never mid-call
         entry = dict(pair)
         logger.info("[%d/%d] %s", i, len(pairs), entry["claim"][:80])
 
         if entry["outcome"] == "orphaned":
             logger.info(" -> key %s is not in the mapping.", entry["cite_key"])
-            results.append(entry)
+            _record(i, entry)
             continue
 
         documents = resolve_documents(collection, entry["citation_source"], doc_cache)
         if not documents:
             entry["outcome"] = "unresolved"
             logger.info(" -> source resolves to no documents in the corpus.")
-            results.append(entry)
+            _record(i, entry)
             continue
 
         # Retrieval is guarded like everything else in this loop: nothing is
@@ -298,13 +323,13 @@
             entry["error_type"] = type(exc).__name__
             entry["raw"] = str(exc)
             logger.warning(" -> retrieval failed: %s: %s", type(exc).__name__, exc)
-            results.append(entry)
+            _record(i, entry)
             continue
 
         if not hits:
             entry["outcome"] = "no_evidence"
             logger.info(" -> nothing retrieved from that source for this claim.")
-            results.append(entry)
+            _record(i, entry)
             continue
 
         expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
@@ -328,7 +353,7 @@
             entry["outcome"] = "parse_failed"
             entry["raw"] = exc.raw
             logger.warning(" -> unusable reply after %d attempts.", _JUDGE_ATTEMPTS)
-            results.append(entry)
+            _record(i, entry)
             continue
         except Exception as exc:
             # Connection errors, timeouts and HTTP 4xx/5xx are not "unusable
@@ -338,7 +363,7 @@
             entry["error_type"] = type(exc).__name__
             entry["raw"] = str(exc)
             logger.warning(" -> judging call failed: %s: %s", type(exc).__name__, exc)
-            results.append(entry)
+            _record(i, entry)
             continue
 
         entry["outcome"] = "judged"
@@ -352,7 +377,7 @@
         # Derived by enforce_rubric(); absent from older stubs and records.
         entry.update({field: verdict[field] for field in sorted(DERIVED_FIELDS) if field in verdict})
         logger.info(" -> %s (%s confidence)", verdict["judgement"], verdict["confidence"])
-        results.append(entry)
+        _record(i, entry)
 
     report = {
         "draft": os.path.abspath(draft_path),
@@ -373,6 +398,9 @@
     _write_markdown(md_path, report)
     logger.info("Saved verification report to %s", md_path)
 
+    runlog.emit("verify_done", totals=report["totals"], json_path=json_path, md_path=md_path)
+    t = report["totals"]
+    runlog.set_summary(f"{t.get('Supports', 0)} of {t.get('total', 0)} citations supported")
     return report
 
 
```

- [ ] **Step 5: Change the seed audit**

**Apply this change to `research_assistant/shared/seed_audit.py`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/research_assistant/shared/seed_audit.py
+++ b/research_assistant/shared/seed_audit.py
@@ -27,7 +27,7 @@
     JUDGEMENT_TOP_K,
     RAW_DIR,
 )
-from research_assistant.shared import pipeline_status
+from research_assistant.shared import narration, pipeline_status, runlog
 from research_assistant.shared.atomic import atomic_write_json
 from research_assistant.shared.log import get_logger
 from research_assistant.shared.search import expand_neighbours, hybrid_search
@@ -298,13 +298,39 @@
 
     Returns:
         Structured audit report dict containing totals and detailed item results.
+
+    A narrated run of kind ``audit`` (spec 2026-09-12-narrated-runs §2B) —
+    joined when it runs inside the pipeline's run, opened when the page or
+    the CLI calls it alone. Its events sit in the ``audit`` stage: one
+    ``audit_item`` per claim as its verdict lands, ``audit_done`` with the
+    totals. A stop lands between claims.
     """
+    with runlog.run("audit", {"seed_path": seed_path, "max_claims": max_claims}):
+        runlog.set_stage("audit")
+        return _audit_seed_citations(seed_path, search_resources, max_claims, top_k)
+
+
+def _item_event(item: dict, index: int) -> None:
+    ref = item.get("ref") or {}
+    runlog.emit(
+        "audit_item", index=index, ref_index=ref.get("index"), ref_title=ref.get("title"),
+        ref_year=ref.get("year"), ref_authors=list(ref.get("authors") or [])[:3],
+        claim=item.get("claim"), sentence=item.get("sentence"), downloaded=bool(item.get("downloaded")),
+        document=item.get("document"), outcome=item.get("outcome"), judgement=item.get("judgement"),
+        confidence=item.get("confidence"), reason=item.get("reason"),
+        supporting_span=item.get("supporting_span"),
+    )
+
+
+def _audit_seed_citations(seed_path, search_resources, max_claims, top_k) -> dict:
     stem = os.path.splitext(os.path.basename(seed_path))[0] if seed_path else "unknown"
     seed_pdf_name = os.path.basename(seed_path) if seed_path else ""
 
     tei_path = find_tei_for_seed(seed_path)
     if not tei_path:
         logger.info("No TEI XML found for seed PDF %s — skipping citation audit.", seed_path)
+        runlog.emit("audit_done", totals={"total": 0, "judged": 0}, error="No GROBID TEI XML found for seed PDF.",
+                    path=None)
         return {
             "seed_path": seed_path,
             "error": "No GROBID TEI XML found for seed PDF.",
@@ -315,6 +341,7 @@
     claims = extract_seed_citation_claims(tei_path)
     if not claims:
         logger.info("No in-text citation claims found in %s.", tei_path)
+        runlog.emit("audit_done", totals={"total": 0, "judged": 0}, error=None, path=None)
         return {
             "seed_path": seed_path,
             "totals": {"total": 0, "judged": 0},
@@ -354,6 +381,9 @@
     )
 
     claims_to_judge = downloaded_claims[:max_claims]
+    runlog.emit("audit_started", claims=len(claims), downloaded=len(downloaded_claims),
+                to_judge=len(claims_to_judge), seed=seed_pdf_name)
+    runlog.note(narration.auditing(len(claims), len(claims_to_judge)))
 
     if claims_to_judge:
         if search_resources is None:
@@ -368,6 +398,7 @@
         retrieve_k = max(top_k, escalate_k)
 
         for i, item in enumerate(claims_to_judge, 1):
+            runlog.check_cancelled()          # between claims, never mid-call
             ref_info = item.get("ref") or {}
             ref_lbl = (
                 f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
@@ -399,12 +430,14 @@
                 item["outcome"] = "retrieval_failed"
                 item["judgement"] = "Unclear / insufficient evidence"
                 item["reason"] = f"Retrieval failed: {exc}"
+                _item_event(item, i)
                 continue
 
             if not hits:
                 item["outcome"] = "no_evidence"
                 item["judgement"] = "Unclear / insufficient evidence"
                 item["reason"] = "No relevant passages found in the cited document."
+                _item_event(item, i)
                 continue
 
             expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
@@ -441,8 +474,13 @@
                 item["outcome"] = "call_failed"
                 item["judgement"] = "Unclear / insufficient evidence"
                 item["reason"] = f"Model evaluation error: {exc}"
+            _item_event(item, i)
 
     all_results = claims_to_judge + downloaded_claims[max_claims:] + undownloaded_claims
+    # The claims not judged — beyond max_claims, or citing a paper that is not
+    # here — are on the transcript too, so the reader's count adds up.
+    for j, item in enumerate(all_results[len(claims_to_judge):], len(claims_to_judge) + 1):
+        _item_event(item, j)
 
     totals = {
         "total": len(all_results),
@@ -476,4 +514,6 @@
     atomic_write_json(out_file, report)
     logger.info("Saved seed citation audit to %s", out_file)
 
+    runlog.emit("audit_done", totals=totals, error=None, path=out_file)
+    runlog.set_summary(f"{totals['Supports']} of {totals['judged']} judged citations supported")
     return report
```

- [ ] **Step 6: Run the three files**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_batch_citer.py tests/test_verifier.py tests/test_seed_audit.py
```

Expected: all pass (92 + 9 in the scratch validation)

- [ ] **Step 7: Commit**

```bash
git add research_assistant/agents/agent5_batch_citer.py research_assistant/agents/agent8_verifier.py research_assistant/shared/seed_audit.py tests/test_batch_citer.py tests/test_verifier.py tests/test_seed_audit.py
git commit -m "feat(cite,verify,audit): each is a run — one event per sentence, citation or claim as it is decided; a stop lands between items

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: `ui.copy` and `ui.runs` — the strings and the launcher

**Files:**
- Create: `research_assistant/ui/__init__.py`, `research_assistant/ui/copy.py`, `research_assistant/ui/runs.py`
- Test: `tests/test_ui_runs.py`

**Interfaces:**
- Consumes: `runlog`.
- Produces: `copy.*` — every string the page says (Task 12 reads them; change wording here, nowhere else). `runs.launch(kind, inputs, target, *args, cleanup=None, **kwargs) -> run_id` — `start_run` synchronously (raises `RunActive`), then `target(*args, **kwargs)` on a daemon thread named `marvin-<kind>-<id>`; the thread finishes the run (`RunCancelled` → cancelled, exception → failed, return → done/stopped per `set_stopped`) and runs `cleanup()` always; `runs.active_run() -> (run_id, meta) | None` (the `current` run while live), `runs.last_run()`, `runs.is_thread_alive(run_id)`, `runs.wait(run_id, timeout)`, `runs.stop(run_id)`.

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_ui_runs.py` with exactly this content:**

```python
"""ui.runs — the launcher: thread, statuses, one run at a time (spec §2B)."""

import os
import threading
import unittest

from research_assistant.shared import runlog
from research_assistant.ui import runs


class TestLaunch(unittest.TestCase):
    def test_target_runs_on_a_thread_and_the_run_finishes_done(self):
        seen = {}

        def target(a, b=None):
            seen["thread"] = threading.current_thread().name
            seen["args"] = (a, b)
            with runlog.run("cite", {}):                # joins — no second run
                runlog.emit("x")
                runlog.set_summary("all good")

        rid = runs.launch("cite", {"draft": "d"}, target, 1, b=2)
        self.assertEqual(runlog.current_run_id(), rid)  # `current` is set before the thread runs
        self.assertTrue(runs.wait(rid, timeout=5))
        self.assertTrue(seen["thread"].startswith("marvin-cite-"))
        self.assertEqual(seen["args"], (1, 2))
        meta = runlog.read_meta(rid)
        self.assertEqual((meta["status"], meta["summary"]), ("done", "all good"))
        self.assertEqual([e["type"] for e in runlog.read_events(rid)], ["x", "finished"])
        self.assertIsNone(runlog.bound_run_id())
        self.assertFalse(runs.is_thread_alive(rid))

    def test_exception_fails_the_run_and_cleanup_runs(self):
        cleaned = []

        def target():
            raise ValueError("boom")

        rid = runs.launch("verify", {}, target, cleanup=lambda: cleaned.append(True))
        self.assertTrue(runs.wait(rid, timeout=5))
        meta = runlog.read_meta(rid)
        self.assertEqual((meta["status"], meta["error"]), ("failed", "ValueError: boom"))
        self.assertEqual(cleaned, [True])

    def test_stop_cancels_the_run(self):
        started = threading.Event()

        def target():
            started.set()
            for _ in range(200):
                runlog.check_cancelled()
                threading.Event().wait(0.01)

        rid = runs.launch("pipeline", {}, target)
        self.assertTrue(started.wait(2))
        runs.stop(rid)
        self.assertTrue(runs.wait(rid, timeout=5))
        self.assertEqual(runlog.read_meta(rid)["status"], "cancelled")

    def test_set_stopped_makes_a_stopped_run(self):
        rid = runs.launch("pipeline", {}, lambda: runlog.set_stopped("no seed"))
        self.assertTrue(runs.wait(rid, timeout=5))
        self.assertEqual(runlog.read_meta(rid)["status"], "stopped")

    def test_second_launch_while_live_is_refused(self):
        release = threading.Event()
        rid = runs.launch("pipeline", {}, release.wait, 5)
        try:
            with self.assertRaises(runlog.RunActive):
                runs.launch("cite", {}, lambda: None)
            active = runs.active_run()
            self.assertEqual(active[0], rid)
            self.assertEqual(active[1]["status"], "running")
        finally:
            release.set()
            runs.wait(rid, timeout=5)
        self.assertIsNone(runs.active_run())
        self.assertEqual(runs.last_run()[0], rid)

    def test_no_runs_yet(self):
        self.assertIsNone(runs.active_run())
        self.assertIsNone(runs.last_run())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_ui_runs.py
```

Expected: FAIL — `research_assistant.ui` not found

- [ ] **Step 3: Create the package, the strings and the launcher**

**Create `research_assistant/ui/__init__.py` with exactly this content:**

```python
"""The Streamlit page's helpers: the run launcher (runs), the transcript
renderer (feed) and Marvin's strings (copy). Only feed imports streamlit."""
```

**Create `research_assistant/ui/copy.py` with exactly this content:**

```python
"""
copy — Marvin's strings (spec 2026-09-12-narrated-runs §2G).

Dry, specific, never cute. Everything the page says in its own voice is
here so the voice cannot drift between tabs, and so a reviewer can read
the whole of it in one file.
"""

APP_TITLE = "Marvin the Citebot"
PAGE_ICON = "🤖"
TAGLINE = ("Give him a research idea and he builds a corpus from the literature and tells you "
           "what has already been done. Give him a sentence and he finds the citation — and checks it.")

AUTH_TITLE = "Marvin the Citebot — access"
AUTH_CAPTION = "This instance is restricted while it is being reviewed. Enter the access code."
AUTH_LABEL = "Access code"
AUTH_BUTTON = "Let me in"
AUTH_WRONG = "That is not the code. Try again."

TABS = ["Research a topic", "Cite a sentence", "Cite a whole draft", "Research chat", "How to use"]

NO_CORPUS = "No corpus yet. Build one in **Research a topic** first; until then there is nothing to cite from."
NO_CORPUS_CHAT = "Nothing to talk about yet. Build a corpus in **Research a topic** first."

# ─── Tab 1 ────────────────────────────────────────────────────────────────
BUILD_SOURCE = "Start from"
BUILD_SOURCE_UPLOAD = "📄 Paper(s) I have (PDF or ZIP)"
BUILD_SOURCE_SEARCH = "🔍 A research idea"
BUILD_IDEA = "The idea"
BUILD_IDEA_PLACEHOLDER = "topological protection in disordered quantum wires"
BUILD_SEED_URL = "Seed paper link (optional)"
BUILD_SEED_URL_HELP = ("An arXiv page or a direct PDF link. Used instead of the search — or when the search "
                       "finds nothing open-access.")
BUILD_UPLOAD = "Paper(s) — .pdf or .zip"
BUILD_UPLOAD_HELP = ("One paper: he reads it, mines its references, fetches what is open-access, and indexes "
                     "all of it. Several papers or a ZIP: he indexes them as they are.")
BUILD_TOPIC = "What you want to know (optional)"
BUILD_TOPIC_PLACEHOLDER = "leave blank and he infers it from the papers"
BUILD_ASK = "Answer"
BUILD_ASK_HELP = "At the end, a synthesis of what the papers say about your idea."
BUILD_FORCE = "Redo all"
BUILD_FORCE_HELP = "Re-run every stage even where the manifests say it is done."
BUILD_FIGURES = "Read figures"
BUILD_FIGURES_HELP = ("One choice for the whole run: every figure and table in every paper is described by "
                      "the model and the description joins the corpus. About a minute per figure on CPU.")
BUILD_AUDIT = "Audit citations"
BUILD_AUDIT_HELP = ("After the synthesis: every in-text citation in the seed paper is checked against the cited "
                    "paper, when that paper was fetched. One model call per citation.")
AUDIT_BUTTON = "Audit the seed's citations"
AUDIT_CAPTION = ("Checks whether the references cited inside the seed paper support the statements it makes — "
                 "one model call per citation whose paper is in the corpus.")
BUILD_GO = "Go"
BUILD_IDLE_CAPTION = ("He finds a seed paper, reads its reference list, fetches what is open-access, "
                      "indexes and summarises each paper, and then tells you what has been done. "
                      "You watch all of it here.")
BUILD_NEED_IDEA = "He needs an idea to search for."
BUILD_NEED_FILES = "He needs at least one PDF, or a ZIP of them."
BUILD_NO_VALID_PDF = "None of those were PDFs he could read (no %PDF header). Check the files."
BUILD_BUSY = "He is busy with another run. It is shown below; wait for it or stop it."
PAST_RUNS = "What he has read"
PAST_RUNS_EMPTY = "Nothing yet."
STOP = "Stop"
STOP_HELP = "He finishes the paper, sentence or citation he is on, then stops."
STOPPING = "Stopping after the current item…"

# ─── Tab 2 ────────────────────────────────────────────────────────────────
CITE_SENTENCE = "The sentence"
CITE_SENTENCE_PLACEHOLDER = "Anderson localization suppresses diffusive transport in one dimension."
CITE_TOP_K = "Passages to retrieve"
CITE_BUTTON = "Find a citation"
CITE_NOTHING = "Nothing in the corpus says that. He will not invent a source."
CITE_SPINNER = "Reading the corpus…"

# ─── Tab 3 ────────────────────────────────────────────────────────────────
BATCH_CAPTION = ("Every sentence that makes a factual claim is checked against the corpus and cited where "
                 "a source supports it. You watch each decision as he makes it.")
BATCH_UPLOAD = "Draft (.txt)"
BATCH_PASTE = "…or paste it here"
BATCH_BUTTON = "Cite the draft"
BATCH_NEED_TEXT = "He needs a draft — upload a .txt or paste one."
BATCH_ABORTED = ("The citation-need check could not be aligned to the draft's sentences, so nothing was "
                 "written. Try again, or shorten the draft if the model keeps truncating its reply.")
BATCH_RESULT = "Cited draft"
BATCH_DOWNLOAD = "Download the cited draft"
BATCH_SOURCES = "Sources cited"
BATCH_DECISIONS = "Per-sentence decisions"
VERIFY_CAPTION = ("Verification re-checks every inserted citation against the source it cites — one model "
                  "call per citation, so a long draft is not free.")
VERIFY_BUTTON = "Verify the citations"
VERIFY_REPORT = "Full verification report"
TAB3_BUSY_ELSEWHERE = "He is busy with a run in **Research a topic**. Wait for it, or stop it there."

# ─── Tab 4 ────────────────────────────────────────────────────────────────
CHAT_CAPTION = "A conversation grounded in the corpus. Ask about what is in it."
CHAT_INPUT = "Ask about the literature…"
CHAT_CLEAR = "Clear"
CHAT_EXPORT = "Export conversation"
CHAT_EMPTY_REPLY = "He had nothing to say. Try a more specific question."

# ─── Sidebar ──────────────────────────────────────────────────────────────
SIDEBAR_STATUS = "Marvin"
SIDEBAR_CORPUS = "Corpus"
SIDEBAR_OPERATOR = "Operator"
SIDEBAR_IDLE = "Idle"

# ─── Feed ─────────────────────────────────────────────────────────────────
FEED_SUMMARY_PENDING = "summary pending — he writes it once every paper in the batch is parsed"
FEED_EMPTY_PAPER = "no selectable text — a scan, or an image-only PDF; it was not indexed"
FEED_FIGURES = "Figures and tables"
FEED_MORE = "…and {n} more"
FEED_UNGROUNDED = "Not grounded in retrieved papers — he found no seed and answered from memory."
FEED_UNVERIFIED_KEYS = "He cited keys that are not on the shortlist: {keys}. Treat those sentences as unsupported."
FEED_DIED = "The process ended without finishing this run."
FEED_STOPPED_HINT = ("Upload a seed PDF or paste an arXiv/PDF link into the seed field and go again to ground "
                     "the answer in a real corpus.")
FEED_PASSAGES = "Retrieved context"
FEED_SHORTLIST = "Relevant prior work"
FEED_NOTES = "His notes on each"
FEED_KEY_LEGEND = "Keys"
```

**Create `research_assistant/ui/runs.py` with exactly this content:**

```python
"""
runs — launch a narrated run on a thread the page does not own
(spec 2026-09-12-narrated-runs §2B).

The page calls launch(); the entry point runs on a daemon thread and
writes the transcript; the page reruns and reads it. The thread makes no
Streamlit call. One run at a time per process — a second launch while one
is live raises RunActive, and the page shows the live run instead.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from research_assistant.shared import runlog
from research_assistant.shared.log import get_logger

logger = get_logger("ui.runs")

_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def active_run() -> Optional[tuple[str, dict[str, Any]]]:
    """(run_id, meta) of the run named `current` when it is still running in
    a live process, else None."""
    rid = runlog.current_run_id()
    if not rid:
        return None
    meta = runlog.read_meta(rid)
    return (rid, meta) if runlog.is_live(meta) else None


def last_run() -> Optional[tuple[str, dict[str, Any]]]:
    """(run_id, meta) of the run named `current`, whatever its status."""
    rid = runlog.current_run_id()
    if not rid:
        return None
    meta = runlog.read_meta(rid)
    return (rid, meta) if meta else None


def launch(kind: str, inputs: dict[str, Any], target: Callable[..., Any], *args: Any,
           cleanup: Optional[Callable[[], None]] = None, **kwargs: Any) -> str:
    """Start the run synchronously (so `current` names it before the page
    reruns), then run target(*args, **kwargs) on a daemon thread. The
    target's own `with runlog.run(...)` joins the bound run; whatever it
    does not finish, the thread finishes: RunCancelled → cancelled, any
    other exception → failed, a clean return → done (or stopped, when the
    target called runlog.set_stopped). `cleanup` runs after the target,
    always (the staged upload file, for one)."""
    run_id = runlog.start_run(kind, inputs)             # raises RunActive

    def body():
        try:
            target(*args, **kwargs)
        except runlog.RunCancelled:
            runlog.finish_run("cancelled", summary="stopped by the reader")
        except BaseException as exc:  # noqa: BLE001 — the thread must not die silently
            logger.exception("Run %s failed", run_id)
            runlog.finish_run("failed", error=f"{type(exc).__name__}: {exc}")
        else:
            runlog.finish_run()
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:  # noqa: BLE001
                    logger.warning("Cleanup after run %s failed", run_id, exc_info=True)
            with _lock:
                _threads.pop(run_id, None)

    thread = threading.Thread(target=body, daemon=True, name=f"marvin-{kind}-{run_id[-4:]}")
    with _lock:
        _threads[run_id] = thread
    thread.start()
    return run_id


def is_thread_alive(run_id: str) -> bool:
    with _lock:
        t = _threads.get(run_id)
    return bool(t and t.is_alive())


def wait(run_id: str, timeout: Optional[float] = None) -> bool:
    """Block until the run's thread ends (tests). True if it did."""
    with _lock:
        t = _threads.get(run_id)
    if t is None:
        return True
    t.join(timeout)
    return not t.is_alive()


def stop(run_id: str) -> None:
    """Ask the run to stop at its next item boundary (spec §2C)."""
    runlog.cancel(run_id)
```

- [ ] **Step 4: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_ui_runs.py
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ui tests/test_ui_runs.py
git commit -m "feat(ui): the run launcher — an entry point on a daemon thread, one live run per process — and Marvin's strings

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: `ui.feed` — draw a transcript

**Files:**
- Create: `research_assistant/ui/feed.py`
- Test: `tests/test_ui_feed.py`

**Interfaces:**
- Consumes: `transcript` (Task 3), `runlog`, `copy` (Task 10), `tests/test_transcript.PIPELINE`.
- Produces: `feed.render_run(run_id, allow_stop=True, slot="main") -> meta|None` (header + sections + end), `feed.render_header`, `feed.render_sections`, `feed.render_passages(passages, label)` (Tab 2 uses it), `feed.run_subject(meta)` (query / titles / draft basename), per-section renderers `render_brief / render_seed / render_references / render_card / render_reading / render_synthesis / render_audit / render_cite / render_verify / render_end`, `_section_label(section)`. The Stop button's key is `stop:<slot>:<run_id>` — the same run can be shown by two tabs at once and Streamlit rejects a duplicate key (found live).

Bare-mode Streamlit (no `streamlit run`) turns every `st.*` call into a no-op with a `missing ScriptRunContext` warning, which is why the tests can call the renderers directly; they are smoke tests plus the pure label/subject helpers.

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_ui_feed.py` with exactly this content:**

```python
"""ui.feed — the transcript renders end to end without a Streamlit runtime
(spec 2026-09-12-narrated-runs §2F). Bare-mode Streamlit turns every st.*
call into a no-op with a warning, so this is a smoke test: every section
renderer runs on a full synthetic transcript and on an empty one, and the
section labels carry the counts."""

import os
import unittest
from unittest.mock import patch

from research_assistant.shared import runlog, transcript as tr
from research_assistant.ui import feed
from tests.test_transcript import PIPELINE, ev


class TestRenderSmoke(unittest.TestCase):
    def _write_run(self, events, status="running"):
        rid = runlog.start_run("pipeline", {"query": "idea"})
        for e in events:
            runlog.emit(e["type"], stage=e["stage"], **e["payload"])
        if status != "running":
            runlog.finish_run(status, summary="s")
        else:
            runlog._reset_for_tests()          # leave it `running` on disk
        return rid

    def test_every_section_renders_on_a_full_transcript(self):
        events = PIPELINE + [
            ev(21, "synthesis", "respond", text="Established [P1].", citations=["P1"], keys={"P1": "P1 title"},
               unverified_citations=["P7"], irrelevant_cited=[], passages=[{"text": "t", "metadata": {"page": 1}}],
               selected=[{"key": "P1", "citation": "P1 title", "score": 0.9, "summary": "s"}], mode="map_reduce",
               timings={}, ungrounded=False),
        ]
        rid = self._write_run(events, status="done")
        meta = feed.render_run(rid)
        self.assertEqual(meta["status"], "done")

    def test_running_and_stopped_and_failed_render(self):
        rid = self._write_run(PIPELINE[:8])
        self.assertEqual(feed.render_run(rid)["status"], "running")
        # Still `running` on disk from this pid: the next start is refused —
        # the one-run-at-a-time rule — so close it the way a dead process would be.
        with patch.object(runlog, "is_pid_alive", return_value=False):
            rid = self._write_run([ev(1, "stopped", "discover", reason="no open-access PDF found — supply an arXiv link")],
                              status="stopped")
        feed.render_run(rid)
        rid = self._write_run([], status="failed")
        feed.render_run(rid)
        self.assertIsNone(feed.render_run("no-such-run"))

    def test_cite_and_verify_transcripts_render(self):
        cite = [ev(1, "draft_split", "cite", n_sentences=2, n_eligible=1),
                ev(2, "sentence_decided", "cite", index=0, sentence="A.", needs_citation=True, cited=True,
                   cited_text="A \\cite{cite_1}.", skip_reason=None, sources=[{"key": "cite_1", "citation": "Doe"}]),
                ev(3, "sentence_decided", "cite", index=1, sentence="B.", needs_citation=False, cited=False,
                   cited_text="B.", skip_reason="too short", sources=[]),
                ev(4, "draft_written", "cite", path="/x_cited.txt", n_cited=1, n_sources=1, unknown_keys=[])]
        rid = runlog.start_run("cite", {"draft": "/x.txt"})
        for e in cite:
            runlog.emit(e["type"], stage=e["stage"], **e["payload"])
        runlog.finish_run("done")
        feed.render_run(rid)
        verify = [ev(1, "verify_started", "verify", n=1),
                  ev(2, "citation_judged", "verify", index=1, claim="A.", cite_key="cite_1", citation_source="Doe",
                     outcome="judged", judgement="Supports", confidence="High", reason="r", escalated=True,
                     rubric_mismatch=False, compound_sentence=False),
                  ev(3, "verify_done", "verify", totals={"total": 1, "judged": 1, "Supports": 1}, json_path="", md_path="")]
        rid = runlog.start_run("verify", {"draft": "/x.txt"})
        for e in verify:
            runlog.emit(e["type"], stage=e["stage"], **e["payload"])
        runlog.finish_run("done")
        feed.render_run(rid)

    def test_audit_section_renders(self):
        events = [ev(1, "audit_started", "audit", claims=2, downloaded=1, to_judge=1, seed="seed.pdf"),
                  ev(2, "narration", "audit", text="2 in-text citations…"),
                  ev(3, "audit_item", "audit", index=1, ref_index=1, ref_title="A", ref_year=2004, ref_authors=["Novoselov"],
                     claim="c", sentence="s", downloaded=True, document="a.pdf", outcome="judged", judgement="Supports",
                     confidence="High", reason="r", supporting_span="span"),
                  ev(4, "audit_item", "audit", index=2, ref_index=2, ref_title="B", ref_year=None, ref_authors=[],
                     claim="c2", sentence="s2", downloaded=False, document=None, outcome="not_downloaded",
                     judgement="Not downloaded", confidence=None, reason="not here", supporting_span=None),
                  ev(5, "audit_done", "audit", totals={"total": 2, "judged": 1, "Supports": 1, "not_downloaded": 1},
                     error=None, path="/x_audit.json")]
        rid = runlog.start_run("audit", {"seed_path": "/seed.pdf"})
        for e in events:
            runlog.emit(e["type"], stage=e["stage"], **e["payload"])
        runlog.finish_run("done")
        feed.render_run(rid)
        self.assertEqual(feed._section_label(tr.sections(events)[0]), "Citation audit — 2 of 2")

    def test_section_labels_carry_counts(self):
        secs = {s["key"]: s for s in tr.sections(PIPELINE)}
        self.assertEqual(feed._section_label(secs["references"]), "References — 3 found · 2 fetched · 1 unavailable")
        self.assertEqual(feed._section_label(secs["reading"]), "Reading — 2 of 2")
        self.assertEqual(feed._section_label(secs["brief"]), "The brief")

    def test_ref_line(self):
        self.assertEqual(feed._ref_line({"authors": ["Anderson, P. W.", "Mott"], "year": 1958, "title": "Absence of diffusion"}),
                         "Anderson et al. (1958) Absence of diffusion")
        self.assertEqual(feed._ref_line({"authors": ["Mott"], "title": "T"}), "Mott T")
        self.assertEqual(feed._ref_line({}), "(untitled)")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_ui_feed.py
```

Expected: FAIL — `research_assistant.ui.feed` not found

- [ ] **Step 3: Write `feed.py`**

**Create `research_assistant/ui/feed.py` with exactly this content:**

```python
"""
feed — render a run's transcript (spec 2026-09-12-narrated-runs §2F).

Every function here is a pure function of (meta, events) as runlog.read_run
returns them, drawn with Streamlit. The page calls render_run() from a
fragment every FEED_REFRESH_SECONDS while the run is live, and once for a
finished run; a past run renders through the same code. The grouping and
counting live in shared.transcript, tested without Streamlit; this module
only decides what each event looks like.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import streamlit as st

from research_assistant.shared import runlog, transcript as tr
from research_assistant.ui import copy

SAMPLE_ROWS = 10       # rows shown before "…and n more" folds the rest into an expander


# ─── Small pieces ────────────────────────────────────────────────────────────


def _authors(sample_entry: dict[str, Any]) -> str:
    authors = sample_entry.get("authors") or []
    if not authors:
        return ""
    first = str(authors[0]).split(",")[0].strip()
    return first + (" et al." if len(authors) > 1 else "")


def _ref_line(entry: dict[str, Any]) -> str:
    who = _authors(entry)
    year = f" ({entry['year']})" if entry.get("year") else ""
    title = entry.get("title") or ""
    return f"{who}{year} {title}".strip() or "(untitled)"


def _folded(rows: list[str], label: str) -> None:
    """The first SAMPLE_ROWS rows inline; the rest behind an expander."""
    for line in rows[:SAMPLE_ROWS]:
        st.markdown(line, unsafe_allow_html=True)
    rest = rows[SAMPLE_ROWS:]
    if rest:
        with st.expander(copy.FEED_MORE.format(n=len(rest)) + f" — {label}"):
            for line in rest:
                st.markdown(line, unsafe_allow_html=True)


def render_passages(passages: list[dict[str, Any]], label: str = copy.FEED_PASSAGES) -> None:
    """The retrieved-chunk expander shared by Tab 2 and the synthesis."""
    if not passages:
        return
    with st.expander(f"{label} · {len(passages)} passage(s)"):
        for i, p in enumerate(passages, 1):
            m = p.get("metadata") or {}
            st.caption(f"{i}. **{m.get('citation_source', '?')}** — {m.get('document', '?')} · p.{m.get('page', '?')}")
            st.text((p.get("text") or "")[:900])
            if i < len(passages):
                st.divider()


def _narrations(events: list[dict[str, Any]]) -> None:
    for ev in events:
        if ev.get("type") == "narration":
            st.markdown(f"*{(ev.get('payload') or {}).get('text', '')}*")


# ─── Sections ────────────────────────────────────────────────────────────────


def render_brief(events: list[dict[str, Any]]) -> None:
    brief = tr.first_payload(events, "brief")
    if brief:
        st.markdown(brief.get("text", ""))
    _narrations([e for e in events if e.get("type") == "narration"])


def render_seed(events: list[dict[str, Any]]) -> None:
    found = tr.first_payload(events, "seed_found")
    missing = tr.first_payload(events, "seed_missing")
    for ev in events:
        t, p = ev.get("type"), ev.get("payload") or {}
        if t == "search_tried":
            err = f" — {p['error']}" if p.get("error") else ""
            st.caption(f"{p.get('provider')}: “{p.get('query')}” → {p.get('results', 0)} results, "
                       f"{p.get('with_pdf', 0)} with a PDF{err}")
        elif t == "narration":
            st.markdown(f"*{p.get('text', '')}*")
        elif t == "seed_found":
            with st.container(border=True):
                st.markdown(f"**🌱 {p.get('title') or p.get('key') or '—'}**")
                bits = []
                if p.get("arxiv_id"):
                    bits.append(f"arXiv:{p['arxiv_id']}")
                if p.get("doi"):
                    bits.append(f"doi:{p['doi']}")
                if p.get("url"):
                    bits.append(f"[PDF]({p['url']})")
                if p.get("source") == "upload":
                    bits.append("📁 uploaded")
                if p.get("key"):
                    bits.append(f"`{p['key']}`")
                if bits:
                    st.caption(" · ".join(bits))
        elif t == "seed_missing":
            st.warning(f"No open-access seed for “{p.get('query')}”. Tried: " +
                       "; ".join(f"“{q}”" for q in p.get("tried") or []), icon="🤷")
    for card in tr.paper_cards(events, stage="ingest_seed"):
        render_card(card)
    if not found and not missing and not events:
        st.caption("…")


def render_references(events: list[dict[str, Any]]) -> None:
    refs = tr.first_payload(events, "references_extracted")
    fetched, unavailable = tr.fetch_lists(events)
    if refs:
        by = refs.get("by_method") or {}
        how = ", ".join(f"{v} via {k}" for k, v in by.items() if v) or "—"
        st.markdown(f"**{refs.get('n', 0)} references** · {refs.get('with_doi', 0)} with a DOI · {how}")
        sample = refs.get("sample") or []
        if sample:
            with st.expander(f"The first {len(sample)}"):
                for entry in sample:
                    st.markdown(f"- {_ref_line(entry)}")
    _narrations(events)
    started = tr.first_payload(events, "fetch_started")
    if started:
        st.caption(f"{started.get('remaining', 0)} to fetch · {started.get('already', 0)} already here")
    if fetched:
        st.markdown(f"**Fetched — {len(fetched)}**")
        _folded([f"- ✅ {p.get('title') or p.get('key')}  \n  <sub>{p.get('provider') or 'oa'}"
                 f"{' · doi:' + p['doi'] if p.get('doi') else ''}</sub>" for p in fetched], "fetched")
    if unavailable:
        st.markdown(f"**Out of reach — {len(unavailable)}**")
        _folded([f"- ⚠️ {p.get('title') or p.get('key')}  \n  <sub>{p.get('reason', '')}</sub>"
                 for p in unavailable], "unavailable")


def render_card(card: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**{card.get('title') or card.get('document')}**")
        if card.get("empty"):
            st.caption(copy.FEED_EMPTY_PAPER)
            return
        bits = [f"{card.get('chunks', 0)} chunks"]
        if card.get("captions"):
            bits.append(f"{card['captions']} captions")
        if card.get("described"):
            bits.append(f"{card['described']} described")
        if card.get("extraction"):
            bits.append(card["extraction"])
        st.caption(" · ".join(bits))
        if card.get("summary"):
            st.markdown(card["summary"])
        else:
            st.caption(copy.FEED_SUMMARY_PENDING)
        figures = card.get("figures") or []
        if figures:
            with st.expander(f"{copy.FEED_FIGURES} · {len(figures)}"):
                for fig in figures:
                    c1, c2 = st.columns([1, 2])
                    path = fig.get("image_path") or ""
                    if path and os.path.exists(path):
                        c1.image(path, caption=fig.get("label") or "")
                    else:
                        c1.caption(fig.get("label") or "")
                    c2.markdown(fig.get("description") or "")


def render_reading(events: list[dict[str, Any]]) -> None:
    """Narration before the first paper card on top, the cards (merged per
    document, updated as summaries land), the save line, and any narration
    after the cards — the cancel line, for one — at the bottom."""
    first_card = next((i for i, e in enumerate(events) if e.get("type") in ("paper_read", "paper_empty")), len(events))
    _narrations(events[:first_card])
    uploads = tr.first_payload(events, "upload_staged")
    if uploads:
        st.caption("Uploaded: " + " · ".join(p.get("title") or p.get("filename") or "?"
                                              for p in uploads.get("papers") or []))
    for card in tr.paper_cards(events, stage="ingest_refs"):
        render_card(card)
    saved = tr.last_payload(events, "batch_saved")
    if saved:
        st.caption(f"Saved: {saved.get('processed', 0)} papers · {saved.get('inserted', 0)} chunks"
                   + (f" · {saved['described']} figures described" if saved.get("described") else "")
                   + (f" · {saved['empty']} without text" if saved.get("empty") else ""))
    _narrations([e for e in events[first_card:] if e.get("type") == "narration"
                 and e.get("payload", {}).get("text", "").startswith("Stopped")])


def render_synthesis(events: list[dict[str, Any]]) -> None:
    _narrations(events)
    shortlist = tr.first_payload(events, "shortlist")
    notes = [e.get("payload") or {} for e in events if e.get("type") == "notes"]
    syn = tr.last_payload(events, "synthesis")
    selected = (syn or {}).get("selected") or []
    if shortlist or selected:
        papers = selected or shortlist.get("papers") or []
        st.markdown(f"**{copy.FEED_SHORTLIST}** — {len(papers)} paper(s)")
        by_key = {n.get("key"): n for n in notes}
        for r in papers:
            key = r.get("key")
            label = f"[{key}] " if key else ""
            score = f"  ·  score {r['score']}" if r.get("score") is not None else ""
            with st.expander(f"{label}{r.get('citation') or r.get('document') or '?'}{score}"):
                if r.get("summary"):
                    st.write(r["summary"])
                note = by_key.get(key)
                if note:
                    st.markdown(f"**Notes** — {note.get('notes', '')}")
                    if note.get("relevant") is False:
                        st.caption("Not relevant, by his own reading.")
    elif notes:
        st.markdown(f"**{copy.FEED_NOTES}**")
        for n in notes:
            st.markdown(f"- **[{n.get('key')}]** {n.get('citation')}: {n.get('notes', '')}")
    if syn:
        if syn.get("ungrounded"):
            st.warning(copy.FEED_UNGROUNDED, icon="💭")
        with st.container(border=True):
            st.markdown("###### Related work")
            st.markdown(syn.get("text") or "")
        keys = syn.get("keys") or {}
        if keys:
            st.caption(f"**{copy.FEED_KEY_LEGEND}** — " + " · ".join(f"[{k}] {v}" for k, v in keys.items()))
        if syn.get("citations"):
            st.caption("Sources: " + " · ".join(str(c) for c in syn["citations"]))
        if syn.get("unverified_citations"):
            st.warning(copy.FEED_UNVERIFIED_KEYS.format(keys=", ".join(syn["unverified_citations"])), icon="⚠️")
        if syn.get("error"):
            st.error(syn["error"])
        render_passages(syn.get("passages") or [])


def render_cite(events: list[dict[str, Any]]) -> None:
    split = tr.first_payload(events, "draft_split")
    _narrations(events)
    rows = tr.sentence_rows(events)
    if split:
        st.caption(f"{len(rows)} of {split.get('n_sentences', 0)} decided")
    for r in rows:
        if r.get("cited"):
            src = ", ".join(s.get("citation") or s.get("key") or "?" for s in r.get("sources") or [])
            st.markdown(f"✅ {r.get('cited_text')}  \n<sub>{src}</sub>", unsafe_allow_html=True)
        elif r.get("needs_citation"):
            st.markdown(f"✗ {r.get('sentence')}  \n<sub>{r.get('skip_reason') or 'declined'}</sub>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:gray'>· {r.get('sentence')} — {r.get('skip_reason') or ''}</span>",
                        unsafe_allow_html=True)
    written = tr.last_payload(events, "draft_written")
    if written:
        st.caption(f"Written: {written.get('n_cited', 0)} cited · {written.get('n_sources', 0)} sources"
                   + (f" · unknown keys: {', '.join(written['unknown_keys'])}" if written.get("unknown_keys") else ""))


_VERDICT_ICON = {"Supports": "✅", "Partially supports": "🟡", "Contradicts": "❌",
                 "Does not support": "❌", "Unclear / insufficient evidence": "❔"}


def render_verify(events: list[dict[str, Any]]) -> None:
    _narrations(events)
    tiles = tr.verdict_tiles(events)
    cols = st.columns(5)
    cols[0].metric("Checked", tiles["checked"])
    cols[1].metric("Supports", tiles["supports"])
    cols[2].metric("Partially", tiles["partially"])
    cols[3].metric("Need review", tiles["review"])
    cols[4].metric("Not judged", tiles["not_judged"])
    for r in tr.verdict_rows(events):
        if r.get("outcome") == "judged":
            icon = _VERDICT_ICON.get(r.get("judgement"), "•")
            flags = []
            if r.get("escalated"):
                flags.append("escalated")
            if r.get("rubric_mismatch"):
                flags.append("rubric mismatch")
            if r.get("compound_sentence"):
                flags.append("compound sentence")
            extra = f" · {', '.join(flags)}" if flags else ""
            st.markdown(f"{icon} **{r.get('judgement')}** ({r.get('confidence')}){extra} — {r.get('claim')}  \n"
                        f"<sub>\\cite{{{r.get('cite_key')}}} → {r.get('citation_source')} · {r.get('reason') or ''}</sub>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:gray'>• {r.get('outcome')} — {r.get('claim')} "
                        f"(\\cite{{{r.get('cite_key')}}})</span>", unsafe_allow_html=True)


_AUDIT_BADGE = {"Supports": "🟢 Supports", "Partially supports": "🟡 Partially supports",
                "Contradicts": "🔴 Contradicts", "Does not support": "🟠 Does not support"}


def render_audit(events: list[dict[str, Any]]) -> None:
    """The seed's in-text citations judged against the papers he has: five
    tiles, then one expander per claim as its verdict lands."""
    _narrations(events)
    started = tr.first_payload(events, "audit_started")
    if started:
        st.caption(f"{started.get('claims', 0)} in-text citations in {started.get('seed') or 'the seed'} · "
                   f"{started.get('downloaded', 0)} cite papers in the corpus · {started.get('to_judge', 0)} judged")
    tiles = tr.audit_tiles(events)
    cols = st.columns(5)
    cols[0].metric("Citations found", tiles["found"])
    cols[1].metric("Supports", tiles["supports"])
    cols[2].metric("Partially", tiles["partially"])
    cols[3].metric("Need review", tiles["review"])
    cols[4].metric("Not in corpus", tiles["not_here"])
    for r in tr.audit_rows(events):
        ref_num = f"[{r.get('ref_index') or '?'}]"
        who = ", ".join(r.get("ref_authors") or [])[:60]
        year = f" ({r['ref_year']})" if r.get("ref_year") else ""
        title = r.get("ref_title") or "?"
        if r.get("outcome") == "judged":
            badge = _AUDIT_BADGE.get(r.get("judgement"), "⚪ Unclear / insufficient evidence")
        elif r.get("outcome") == "not_downloaded":
            badge = "⚪ Not in corpus"
        else:
            badge = f"⚪ {r.get('outcome') or 'unclear'}"
        with st.expander(f"{badge} · {ref_num} {who}{year} · *{title[:60]}*"):
            st.markdown("**The seed says:**")
            st.info(f"“{r.get('sentence') or r.get('claim') or ''}”")
            st.markdown(f"**Cites:** {ref_num} {title}{year}" + (f"  \n*{who}*" if who else ""))
            if r.get("outcome") == "judged":
                st.markdown(f"**Verdict:** `{r.get('judgement')}` ({r.get('confidence') or 'Medium'} confidence)")
                if r.get("supporting_span"):
                    st.success(f"“{r['supporting_span']}”")
                if r.get("reason"):
                    st.markdown(f"**Reasoning:** {r['reason']}")
            elif r.get("outcome") == "not_downloaded":
                st.warning("The cited paper was not open-access or could not be downloaded, so its text is not here.",
                           icon="🔒")
            else:
                st.caption(r.get("reason") or r.get("outcome") or "")
    done = tr.last_payload(events, "audit_done")
    if done and done.get("error"):
        st.info(f"Audit: {done['error']}")


def render_end(meta: Optional[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    fin = tr.last_payload(events, "finished") or {}
    status = (meta or {}).get("status") or fin.get("status")
    stopped = tr.first_payload(events, "stopped")
    if status == "done":
        st.success(f"Done in {tr.elapsed_label(fin.get('elapsed_s', 0))}" +
                   (f" — {fin['summary']}" if fin.get("summary") else ""), icon="✅")
    elif status == "stopped":
        st.warning(stopped.get("reason") if stopped else (fin.get("summary") or "Stopped early"), icon="⚠️")
        reason = (stopped or {}).get("reason") or ""
        if "supply an arXiv" in reason or "could not download" in reason or "could not load seed PDF" in reason:
            st.info(copy.FEED_STOPPED_HINT, icon="💡")
    elif status == "cancelled":
        st.info("Stopped by you.", icon="⏹️")
    elif status == "failed":
        st.error(f"Failed — {(meta or {}).get('error') or fin.get('error') or 'unknown error'}", icon="❌")


_RENDERERS = {
    "brief": render_brief, "seed": render_seed, "references": render_references,
    "reading": render_reading, "synthesis": render_synthesis, "audit": render_audit,
    "cite": render_cite, "verify": render_verify,
}


def _section_label(section: dict[str, Any]) -> str:
    key, events = section["key"], section["events"]
    n = tr.counts(events)
    if key == "references":
        parts = [f"{n['references']} found"] if n["references"] else []
        if n["fetched"] or n["unavailable"]:
            parts += [f"{n['fetched']} fetched", f"{n['unavailable']} unavailable"]
        return "References" + (" — " + " · ".join(parts) if parts else "")
    if key == "reading":
        done = n["read"] + n["empty"]
        return f"Reading — {done} of {n['to_read']}" if n["to_read"] else "Reading"
    if key == "audit":
        return f"Citation audit — {n['audit_items']} of {n['audit_claims']}" if n["audit_claims"] else "Citation audit"
    if key == "cite":
        return f"Citing — {n['decided']} of {n['sentences']}" if n["sentences"] else "Citing"
    if key == "verify":
        return f"Verifying — {n['judged']} of {n['citations']}" if n["citations"] else "Verifying"
    return section["title"]


def render_sections(meta: Optional[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    for section in tr.sections(events):
        if section["key"] == "end":
            render_end(meta, events)
            continue
        renderer = _RENDERERS.get(section["key"])
        with st.expander(_section_label(section), expanded=section["latest"]):
            if renderer is not None:
                renderer(section["events"])
            else:
                for ev in section["events"]:
                    st.write(ev)


def run_subject(meta: Optional[dict[str, Any]]) -> str:
    """What the run is about, for the header and the past-runs list: the
    query, the uploaded titles, or the draft's file name."""
    what = (meta or {}).get("inputs") or {}
    if what.get("query"):
        return str(what["query"])
    if what.get("titles"):
        return ", ".join(what["titles"])[:80]
    if what.get("draft"):
        return os.path.basename(str(what["draft"]))
    return ""


def render_header(run_id: str, meta: Optional[dict[str, Any]], events: list[dict[str, Any]],
                  allow_stop: bool = True, slot: str = "main") -> None:
    """The status line, the elapsed time and the Stop button. `slot` names
    the place on the page: the same run can be shown in two tabs at once
    and a Streamlit widget key must be unique per page."""
    live = runlog.is_live(meta)
    line = tr.status_line(meta, events, live=live)
    elapsed = tr.elapsed_label(runlog.elapsed_seconds(meta)) if meta else ""
    kind = (meta or {}).get("kind") or ""
    subject = run_subject(meta)
    c1, c2 = st.columns([5, 1])
    dot = "●" if live else "○"
    c1.markdown(f"{dot} **{line}** · {elapsed}" + (f"  \n<sub>{kind} · {subject}</sub>" if subject else ""),
                unsafe_allow_html=True)
    if live and allow_stop:
        if runlog.cancel_requested(run_id):
            c2.caption(copy.STOPPING)
        elif c2.button(copy.STOP, key=f"stop:{slot}:{run_id}", help=copy.STOP_HELP):
            runlog.cancel(run_id)
            st.rerun()
    elif meta and meta.get("status") == "running" and not live:
        c2.caption(copy.FEED_DIED)


def render_run(run_id: str, allow_stop: bool = True, slot: str = "main") -> Optional[dict[str, Any]]:
    """Header, sections, end — the whole transcript. Returns the meta read."""
    meta, events = runlog.read_run(run_id)
    if meta is None:
        st.caption("No such run.")
        return None
    render_header(run_id, meta, events, allow_stop=allow_stop, slot=slot)
    render_sections(meta, events)
    return meta
```

- [ ] **Step 4: Run them**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_ui_feed.py 2>&1 | grep -v ScriptRunContext
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ui/feed.py tests/test_ui_feed.py
git commit -m "feat(ui): the feed — a transcript drawn section by section, paper cards, keyed synthesis, audit, sentence rows, verdict rows

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: `app.py` — forms and a reader; `scripts/run_check.py`

**Files:**
- Rewrite: `app.py` (1,289 → ~800 lines)
- Create: `scripts/run_check.py`
- Test: `tests/test_app_names.py` (unchanged — it must still pass), `tests/test_pipeline_status.py`, `tests/test_grobid_manager.py`, `tests/test_grobid_controller.py` (unchanged — they import `app` and call `_render_sidebar_pipeline_status` / `_render_grobid_controls`, which keep their names and their idle-branch output)

**Interfaces:**
- Consumes: everything above. `orchestrate.run(**inputs)` with `audit_citations`; `batch_ingest.ingest_uploaded`; `agent5_batch_citer.run_batch_citer(draft, out, search_resources=)`; `agent8_verifier.verify_draft(written, search_resources=)`; `seed_audit.audit_seed_citations(seed_path, search_resources=)` and `find_tei_for_seed`.
- Produces the page. What changes, by tab:
  - **Identity**: `st.set_page_config(page_title=copy.APP_TITLE, page_icon=copy.PAGE_ICON)`, the title, the tagline, the auth screen, the tab names (`Cite a draft` → `Cite a sentence`), every label and message from `copy`.
  - **Tab 1** is the form (source radio; upload form with four toggles — answer, redo all, read figures, audit citations; idea form with the same) or, while `runs.active_run()` is live, `_live_run(run_id, "build")` — a `st.fragment(run_every="2s")` calling `feed.render_run` and `st.rerun(scope="app")` once the run is not live. Submitting calls `_start_pipeline(...)` (→ `orchestrate.run` on a thread, the staged seed file removed by `cleanup`) or `_start_ingest(...)` (→ `ingest_uploaded`), remembers the run in `st.session_state["watched_run"]`, and reruns. After the watched run finishes its transcript stays above a divider, with `_render_audit_offer(run_id)` under a pipeline run that has a seed with a TEI and no audit — a button launching an `audit` run. `_render_past_runs()` lists `runlog.list_runs(20)` in an expander with a selectbox and renders the chosen one. The "Pipeline Active on Server" fragment, `_graph`, `STEPS`, `node_weights`, `stage_ranges`, both progress callbacks, `_manifest`, `_render_seed_and_downloads`, `_render_shortlist`, `_render_build`, `_render_seed_citation_audit`, `_render_passages` (now `feed.render_passages`) are gone.
  - **Tab 2**: copy only.
  - **Tab 3**: `Cite the draft` writes the draft to `DRAFTS_DIR` and launches a `cite` run (the fragment slot is `"batch"`); once it is no longer running, the transcript renders (`slot="cited"`) followed by `_render_cited_draft` — text area, download, mapping, report, and `Verify the citations`, which launches a `verify` run; its transcript renders (`slot="verified"`) with the report expander (open when anything needs review). While a run of another kind is live Tab 3 says so and disables its button.
  - **Tab 4, Tab 5**: copy only.
  - **Sidebar**: `Marvin` (the run's status line from `transcript.status_line` when a run is live, then the existing pipeline_status block), `Corpus` (metrics + the no-corpus line), and an `Operator` expander (models, index version, layout, GROBID controls, the API-key error).
- `scripts/run_check.py`: prints a transcript and checks six gates (G1 brief before search, G2 stage order monotone, G3 cards well-formed, G4 `finished` last with the status agreeing, G5 cancel tail, G6 seq contiguous); exit 1 on a FAIL. Task 14 uses it.

- [ ] **Step 1: Replace `app.py` with exactly this content**

```python
"""
Marvin the Citebot — the Streamlit front-end (also the container entry point).

Tab 1 launches the LangGraph pipeline from a research idea or a paper and
shows the run as it happens; Tab 2 is Agent 4's citation assistant over the
corpus; Tab 3 runs Agent 5 (batch citer) and Agent 8 (verifier) over a whole
draft, sentence by sentence; Tab 4 is Agent 7's research chat; Tab 5 renders
HOW_TO_USE.md.

Long operations do not run in this script. Each is launched on a thread
(research_assistant.ui.runs) that writes a transcript
(research_assistant.shared.runlog); the page polls the transcript and draws
it (research_assistant.ui.feed). See specs/2026-09-12-narrated-runs-design.md.
"""

import json
import os
import re
import tempfile

import streamlit as st

from research_assistant import config
from research_assistant.shared import pipeline_status, runlog, transcript
from research_assistant.ui import copy, feed, runs

st.set_page_config(page_title=copy.APP_TITLE, page_icon=copy.PAGE_ICON, layout="wide")
os.makedirs(config.DATA_DIR, exist_ok=True)


def _check_auth() -> bool:
    app_pwd = os.environ.get("APP_PASSWORD")
    if not app_pwd:
        try:
            if hasattr(st, "secrets") and "password" in st.secrets:
                app_pwd = st.secrets["password"]
        except Exception:
            app_pwd = None

    if not app_pwd:
        return True
    if st.session_state.get("authenticated", False):
        return True

    st.markdown(f"### 🔐 {copy.AUTH_TITLE}")
    st.caption(copy.AUTH_CAPTION)
    with st.form("auth_form", clear_on_submit=False):
        entered = st.text_input(copy.AUTH_LABEL, type="password")
        submitted = st.form_submit_button(copy.AUTH_BUTTON, type="primary")
        if submitted:
            if entered.strip() == app_pwd.strip():
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error(copy.AUTH_WRONG, icon="🚫")
    return False


if not _check_auth():
    st.stop()


@st.cache_resource(show_spinner=False)
def get_cached_search_resources(mtime: float):
    """Load ChromaDB collection and BM25 index once into memory for all sessions.

    Keyed on the modification time of BM25_INDEX_PATH so re-indexing automatically
    refreshes the shared cache without needing server restarts.
    """
    from research_assistant.shared.db import load_search_resources
    return load_search_resources()


def _get_resources_mtime() -> float:
    if os.path.exists(config.BM25_INDEX_PATH):
        return os.path.getmtime(config.BM25_INDEX_PATH)
    return 0.0


# ─── Data helpers ───────────────────────────────────────────────────────────


def _corpus_stats():
    """(chunks, papers) — cheap, tolerant of a missing/empty store."""
    chunks = 0
    try:
        import chromadb

        chunks = (
            chromadb.PersistentClient(path=config.VECTORDB_PATH)
            .get_collection(config.COLLECTION_NAME)
            .count()
        )
    except Exception:
        pass

    papers = 0
    for path in (config.SEED_PAPERS_PATH, config.DOWNLOADED_JSON_PATH):
        try:
            with open(path) as fh:
                papers += len(json.load(fh))
        except Exception:
            pass

    try:
        from research_assistant.shared import manifest
        manifest_count = len(manifest.load())
        papers = max(papers, manifest_count)
    except Exception:
        pass
    return chunks, papers


@st.cache_data(ttl=30, show_spinner=False)
def _grobid_ok():
    try:
        from research_assistant.agents.grobid_controller import GrobidAgent

        return GrobidAgent().is_alive(timeout=3.0)
    except Exception:
        return False


def _clear_grobid_cache():
    if hasattr(_grobid_ok, "clear"):
        _grobid_ok.clear()


def _render_grobid_controls():
    try:
        from research_assistant.agents.grobid_controller import GrobidAgent

        agent = GrobidAgent()

        if "grobid_feedback" in st.session_state:
            fb_type, fb_msg = st.session_state.pop("grobid_feedback")
            if fb_type == "success":
                st.success(fb_msg, icon="✅")
            elif fb_type == "error":
                st.error(fb_msg, icon="🚫")
            elif fb_type == "warning":
                st.warning(fb_msg, icon="⚠️")
            else:
                st.info(fb_msg, icon="ℹ️")

        status = agent.check_status()
        st.session_state["grobid_server_state"] = status.state

        if status.state == "RUNNING":
            st.success(f"🟢 **GROBID Status: Running**\n\n`{status.server_url}`")
        elif status.state == "STARTING":
            st.warning(f"🟡 **GROBID Status: Starting...**\n\n`{status.server_url}`")
            if status.message:
                st.caption(status.message)
        elif status.state == "ERROR":
            st.error(f"⚠️ **GROBID Status: Error**\n\n`{status.server_url}`")
            if status.message:
                st.caption(status.message)
            if status.details:
                st.caption(status.details)
        else:
            st.error(f"🔴 **GROBID Status: Stopped**\n\n`{status.server_url}`")
            st.caption(
                "Reference extraction falls back to regex without metadata (authors, years, DOIs)."
            )

        if status.is_alive:
            c_stop, c_restart, c_chk = st.columns([3, 3, 2])
            if c_stop.button("⏹️ Stop", key="btn_stop_grobid", use_container_width=True, help="Stop GROBID server"):
                with st.spinner("Stopping GROBID server..."):
                    ok, msg = agent.stop_server()
                    _clear_grobid_cache()
                    st.session_state["grobid_feedback"] = ("info" if ok else "error", msg)
                    st.session_state["grobid_server_state"] = "STOPPED" if ok else "ERROR"
                    st.rerun()
            if c_restart.button("🔄 Restart", key="btn_restart_grobid", use_container_width=True, help="Restart GROBID server"):
                with st.spinner("Restarting GROBID server..."):
                    ok, msg = agent.restart_server()
                    _clear_grobid_cache()
                    st.session_state["grobid_feedback"] = ("success" if ok else "error", msg)
                    st.session_state["grobid_server_state"] = "RUNNING" if ok else "ERROR"
                    st.rerun()
            if c_chk.button("🩺 Status", key="btn_check_grobid_running", use_container_width=True, help="Probe server health"):
                _clear_grobid_cache()
                new_status = agent.check_status()
                st.session_state["grobid_server_state"] = new_status.state
                if new_status.is_alive:
                    st.session_state["grobid_feedback"] = ("success", f"GROBID is active at {new_status.server_url}.")
                else:
                    st.session_state["grobid_feedback"] = (
                        "warning" if new_status.state == "STARTING" else "error",
                        new_status.message or f"GROBID server is {new_status.state.lower()}."
                    )
                st.rerun()
        elif status.state == "STARTING":
            c_chk, c_stop = st.columns([3, 2])
            if c_chk.button("🩺 Refresh Status", key="btn_check_grobid_starting", use_container_width=True, help="Probe if JVM has finished initializing"):
                _clear_grobid_cache()
                new_status = agent.check_status()
                st.session_state["grobid_server_state"] = new_status.state
                if new_status.is_alive:
                    st.session_state["grobid_feedback"] = ("success", f"GROBID is active at {new_status.server_url}.")
                else:
                    st.session_state["grobid_feedback"] = (
                        "warning" if new_status.state == "STARTING" else "error",
                        new_status.message or f"GROBID server is {new_status.state.lower()}."
                    )
                st.rerun()
            if c_stop.button("⏹️ Stop", key="btn_stop_grobid_starting", use_container_width=True, help="Stop initializing container"):
                with st.spinner("Stopping GROBID container..."):
                    ok, msg = agent.stop_server()
                    _clear_grobid_cache()
                    st.session_state["grobid_feedback"] = ("info" if ok else "error", msg)
                    st.session_state["grobid_server_state"] = "STOPPED" if ok else "ERROR"
                    st.rerun()
        else:
            c_start, c_chk = st.columns([3, 2])
            if c_start.button("▶️ Start GROBID", key="btn_start_grobid", use_container_width=True, help="Launch GROBID server agent"):
                with st.spinner("Launching GROBID agent & checking health..."):
                    ok, msg = agent.start_server()
                    _clear_grobid_cache()
                    new_status = agent.check_status()
                    st.session_state["grobid_server_state"] = new_status.state
                    if new_status.is_alive:
                        st.session_state["grobid_feedback"] = ("success", msg)
                    elif new_status.state == "STARTING":
                        st.session_state["grobid_feedback"] = ("warning", msg)
                    else:
                        st.session_state["grobid_feedback"] = ("error", msg)
                    st.rerun()
            if c_chk.button("🩺 Check Status", key="btn_check_grobid_stopped", use_container_width=True, help="Probe server health"):
                _clear_grobid_cache()
                new_status = agent.check_status()
                st.session_state["grobid_server_state"] = new_status.state
                if new_status.is_alive:
                    st.session_state["grobid_feedback"] = ("success", f"GROBID is active at {new_status.server_url}.")
                else:
                    st.session_state["grobid_feedback"] = (
                        "warning" if new_status.state == "STARTING" else "error",
                        new_status.message or f"GROBID server is {new_status.state.lower()}."
                    )
                st.rerun()

        with st.expander("🛠️ GROBID Diagnostics"):
            st.markdown(f"**Endpoint:** `{status.server_url}`")
            st.markdown(f"**Container / Target:** `{status.container_name}`")
            docker_ok = status.docker_available
            st.markdown(f"**Docker Status:** {'🟢 Available' if docker_ok else '🔴 Unavailable'}")
            if not docker_ok and status.details:
                st.warning(status.details)
            if status.container_status:
                st.markdown(f"**Container Status:** `{status.container_status}`")
            if status.image:
                st.markdown(f"**Docker Image:** `{status.image}`")
            st.markdown("**Manual launch command:**")
            st.code(status.manual_command or agent.get_manual_command(), language="bash")
    except Exception as exc:
        st.error(f"GROBID Agent encountered an error: {exc}")


# ─── Rendering ──────────────────────────────────────────────────────────────


def _render_suggestion(result):
    with st.container(border=True):
        st.markdown("###### Suggested text")
        st.markdown(result["suggestion"])

    cits = result.get("citations") or []
    if cits:
        st.markdown("**Grounded in**")
        for c in cits:
            st.markdown(f"- {c}")

    feed.render_passages(result.get("passages") or [])


def _stage_upload(uploaded_file) -> str:
    """Write an uploaded PDF somewhere the pipeline can seed from, and return it.

    Agent 0 copies the paper into RAW_DIR under a key derived from its DOI,
    arXiv id or content hash, so this staging copy is pure duplication — it
    exists only because the graph's state carries a path, not bytes. So it goes
    to a temp file the run's cleanup deletes once the run finishes, rather than
    under DATA_DIR where copies would pile up and two uploads sharing a filename
    would overwrite each other.
    """
    fd, path = tempfile.mkstemp(prefix="seed_upload_", suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return path


# ─── Runs: launch and watch ─────────────────────────────────────────────────


def _launch(kind, inputs, target, *args, **kwargs):
    """runs.launch, remembering the run so this session shows its transcript
    after it finishes (other sessions find it under past runs)."""
    try:
        run_id = runs.launch(kind, inputs, target, *args, **kwargs)
    except runlog.RunActive:
        st.warning(copy.BUILD_BUSY, icon="⏳")
        return None
    st.session_state["watched_run"] = run_id
    return run_id


@st.fragment(run_every=f"{config.FEED_REFRESH_SECONDS}s")
def _live_run(run_id, slot):
    """Re-read and redraw the transcript while the run is live; when it is
    not, hand the page back to the normal layout. `slot` names the tab —
    Tab 1 and Tab 3 can both show the one live run."""
    meta = feed.render_run(run_id, slot=slot)
    if meta is None or not runlog.is_live(meta):
        st.rerun(scope="app")


def _remove_file(path):
    def cleanup():
        try:
            os.unlink(path)
        except OSError:
            pass
    return cleanup


def _start_pipeline(query, seed_url=None, seed_file=None, ask=True, force=False, describe_figures=None,
                    audit_citations=True):
    """Tab 1, research idea / seed link / single uploaded PDF → orchestrate.run on a thread."""
    import orchestrate  # root-level entry point; imported bare, not as research_assistant.orchestrate

    inputs = {"query": query, "workers": 1, "force": force, "ask": ask, "seed_url": seed_url,
              "seed_file": seed_file, "describe_figures": describe_figures, "audit_citations": audit_citations}
    return _launch("pipeline", inputs, orchestrate.run,
                   cleanup=_remove_file(seed_file) if seed_file else None, **inputs)


def _start_ingest(staged, query, ask, force, describe_figures):
    """Tab 1, several PDFs or a ZIP → batch_ingest.ingest_uploaded on a thread."""
    from research_assistant.shared.batch_ingest import ingest_uploaded

    inputs = {"papers": len(staged), "query": query, "ask": ask, "force": force,
              "describe_figures": describe_figures}
    return _launch("ingest", inputs, ingest_uploaded, staged, query=query, ask=ask, force=force,
                   describe_figures=describe_figures)


def _render_past_runs():
    metas = runlog.list_runs(limit=20)
    with st.expander(copy.PAST_RUNS, expanded=False):
        if not metas:
            st.caption(copy.PAST_RUNS_EMPTY)
            return
        labels = {}
        for m in metas:
            when = (m.get("started_at") or "")[:16].replace("T", " ")
            labels[m["run_id"]] = f"{when} · {m.get('kind')} · {feed.run_subject(m) or '—'} · {m.get('status')}"
        chosen = st.selectbox("Run", list(labels), format_func=labels.get, key="past_run", label_visibility="collapsed")
        if chosen:
            feed.render_run(chosen, allow_stop=False, slot="past")


# ─── Sidebar ────────────────────────────────────────────────────────────────

@st.fragment(run_every="3s")
def _render_sidebar_pipeline_status():
    try:
        from datetime import datetime, timezone

        status = pipeline_status.get_status()
        active = runs.active_run()
        if active:
            run_id, meta = active
            st.markdown(f"**{transcript.status_line(meta, runlog.read_events(run_id))}**")

        if status.get("active", False):
            st.info("⚡ **Pipeline Active**")
            stage_lbl = status.get("stage_label") or status.get("stage") or "Processing"
            try:
                curr_step = int(status.get("current_step") or 1)
                tot_steps = int(status.get("total_steps") or 5)
            except (ValueError, TypeError):
                curr_step, tot_steps = 1, 5
            st.markdown(f"**Stage:** {stage_lbl} (Step {curr_step}/{tot_steps})")

            curr_item = status.get("current_item_name", "")
            if curr_item:
                st.markdown(f"**Current:** `{curr_item[:60]}`")

            try:
                item_c = int(status.get("item_current") or 0)
                item_t = int(status.get("item_total") or 0)
            except (ValueError, TypeError):
                item_c, item_t = 0, 0
            if item_t > 0:
                pct = min(1.0, max(0.0, item_c / item_t))
                st.progress(pct, text=f"{item_c} / {item_t} papers ({int(pct * 100)}%)")
            else:
                pct = min(1.0, max(0.0, curr_step / max(1, tot_steps)))
                st.progress(pct, text=f"Step {curr_step} of {tot_steps}")

            started_at = status.get("started_at")
            if started_at:
                try:
                    start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    elapsed_sec = int((datetime.now(timezone.utc) - start_dt).total_seconds())
                    if elapsed_sec >= 60:
                        st.caption(f"⏱️ Elapsed: {elapsed_sec // 60}m {elapsed_sec % 60}s")
                    else:
                        st.caption(f"⏱️ Elapsed: {max(0, elapsed_sec)}s")
                except Exception:
                    pass

            detail = status.get("detail", "")
            if detail:
                st.caption(f"⚙️ {detail}")

            events = status.get("recent_events", [])
            with st.expander("📋 Live Activity Log", expanded=False):
                if events:
                    for ev in reversed(events):
                        st.markdown(f"- {ev}")
                else:
                    st.caption("No events recorded yet.")
        else:
            st.markdown(f"🟢 **Pipeline: {copy.SIDEBAR_IDLE}** (Ready for new papers)")
            last_completed = status.get("last_completed_at")
            if last_completed:
                try:
                    comp_dt = datetime.fromisoformat(last_completed.replace("Z", "+00:00"))
                    time_str = comp_dt.strftime("%H:%M UTC")
                    summary = (status.get("last_summary") or "").strip()
                    if summary.startswith("(") and summary.endswith(")"):
                        summary = summary[1:-1].strip()
                    summary_part = f" ({summary})" if summary else ""
                    st.caption(f"Last run completed: {time_str}{summary_part}")
                except Exception:
                    pass
            events = status.get("recent_events", [])
            if events:
                with st.expander("📋 Recent Activity Log", expanded=False):
                    for ev in reversed(events):
                        st.caption(ev)
    except Exception:
        st.caption("Pipeline status temporarily unavailable")


with st.sidebar:
    st.subheader(copy.SIDEBAR_STATUS)
    _render_sidebar_pipeline_status()
    st.divider()

    st.subheader(copy.SIDEBAR_CORPUS)
    chunks, papers = _corpus_stats()
    c1, c2 = st.columns(2)
    c1.metric("Papers", papers)
    c2.metric("Chunks", chunks)
    if chunks == 0:
        st.caption(copy.NO_CORPUS)

    with st.expander(copy.SIDEBAR_OPERATOR, expanded=False):
        st.caption(f"**Chat** — `{config.LLM_MODEL}` · {config.LLM_BACKEND}")
        st.caption(f"**Embeddings** — `{config.EMBED_MODEL}` · {config.EMBED_BACKEND}")
        st.caption(f"**Index** — v{config.INDEX_VERSION} · figures {'on' if config.FIGURE_VLM else 'off'} by default")
        st.caption(f"**Layout** — {'on' if config.LAYOUT_DETECTION else 'text-only'}")
        _render_grobid_controls()
        if config.LLM_BACKEND == "openai" and not config.OPENAI_API_KEY:
            st.error("No OPENAI_API_KEY / HF_TOKEN set.", icon="🚫")


# ─── Main ───────────────────────────────────────────────────────────────────

st.title(f"{copy.PAGE_ICON} {copy.APP_TITLE}")
st.caption(copy.TAGLINE)

tab_build, tab_cite, tab_batch, tab_chat, tab_help = st.tabs(copy.TABS)


# ─── Tab 1: research a topic ────────────────────────────────────────────────


def _render_build_form():
    source_type = st.radio(copy.BUILD_SOURCE, [copy.BUILD_SOURCE_UPLOAD, copy.BUILD_SOURCE_SEARCH], horizontal=True)

    if source_type == copy.BUILD_SOURCE_UPLOAD:
        with st.form("upload_papers_form"):
            uploaded_files = st.file_uploader(copy.BUILD_UPLOAD, type=["pdf", "zip"], accept_multiple_files=True,
                                              help=copy.BUILD_UPLOAD_HELP)
            pdf_query = st.text_input(copy.BUILD_TOPIC, placeholder=copy.BUILD_TOPIC_PLACEHOLDER)
            c1, c2, c3, c4 = st.columns(4)
            ask = c1.toggle(copy.BUILD_ASK, value=True, help=copy.BUILD_ASK_HELP)
            force = c2.toggle(copy.BUILD_FORCE, value=False, help=copy.BUILD_FORCE_HELP)
            describe_figures = c3.toggle(copy.BUILD_FIGURES, value=config.FIGURE_VLM, help=copy.BUILD_FIGURES_HELP)
            audit_citations = c4.toggle(copy.BUILD_AUDIT, value=True, help=copy.BUILD_AUDIT_HELP)
            submitted = st.form_submit_button(copy.BUILD_GO, type="primary")

        if submitted:
            if not uploaded_files:
                st.warning(copy.BUILD_NEED_FILES, icon="⚠️")
                return
            from research_assistant.shared.batch_uploader import unpack_and_stage_uploads

            staged = unpack_and_stage_uploads(uploaded_files, destination_dir=config.RAW_DIR)
            if not staged:
                st.error(copy.BUILD_NO_VALID_PDF, icon="⚠️")
                return
            if len(staged) == 1:
                # One paper: the whole pipeline — read it, mine its references, fetch, index, answer.
                run_id = _start_pipeline(pdf_query.strip(), seed_file=staged[0]["path"], ask=ask, force=force,
                                         describe_figures=describe_figures, audit_citations=audit_citations)
            else:
                run_id = _start_ingest(staged, pdf_query.strip(), ask, force, describe_figures)
            if run_id:
                st.rerun()
    else:
        with st.form("build_form"):
            query = st.text_input(copy.BUILD_IDEA, placeholder=copy.BUILD_IDEA_PLACEHOLDER)
            seed_url = st.text_input(copy.BUILD_SEED_URL, placeholder="https://arxiv.org/abs/2401.12345",
                                     help=copy.BUILD_SEED_URL_HELP)
            c1, c2, c3, c4 = st.columns(4)
            ask = c1.toggle(copy.BUILD_ASK, value=True, help=copy.BUILD_ASK_HELP)
            force = c2.toggle(copy.BUILD_FORCE, value=False, help=copy.BUILD_FORCE_HELP)
            describe_figures = c3.toggle(copy.BUILD_FIGURES, value=config.FIGURE_VLM, help=copy.BUILD_FIGURES_HELP)
            audit_citations = c4.toggle(copy.BUILD_AUDIT, value=True, help=copy.BUILD_AUDIT_HELP)
            submitted = st.form_submit_button(copy.BUILD_GO, type="primary")

        if submitted:
            if not query.strip():
                st.warning(copy.BUILD_NEED_IDEA, icon="⚠️")
                return
            run_id = _start_pipeline(query.strip(), seed_url=seed_url.strip() or None, ask=ask, force=force,
                                     describe_figures=describe_figures, audit_citations=audit_citations)
            if run_id:
                st.rerun()


TAB1_KINDS = ("pipeline", "ingest", "audit")


def _render_audit_offer(run_id):
    """Under a finished pipeline run whose seed has a GROBID TEI and which
    was not audited: one button that launches an `audit` run on that seed."""
    meta, events = runlog.read_run(run_id)
    if not meta or meta.get("kind") != "pipeline" or meta.get("status") not in ("done", "stopped"):
        return
    if transcript.first_payload(events, "audit_started") or transcript.last_payload(events, "audit_done"):
        return
    found = transcript.first_payload(events, "seed_found") or {}
    seed_path = found.get("path")
    if not seed_path or not os.path.exists(seed_path):
        return
    from research_assistant.shared.seed_audit import audit_seed_citations, find_tei_for_seed

    if not find_tei_for_seed(seed_path):
        return
    st.caption(copy.AUDIT_CAPTION)
    if st.button(copy.AUDIT_BUTTON, key=f"audit:{run_id}", disabled=runs.active_run() is not None):
        cached_res = get_cached_search_resources(_get_resources_mtime())
        if _launch("audit", {"seed_path": seed_path, "after_run": run_id}, audit_seed_citations, seed_path,
                   search_resources=cached_res):
            st.rerun()


with tab_build:
    active = runs.active_run()
    if active:
        run_id, meta = active
        st.session_state["watched_run"] = run_id
        if meta.get("kind") not in TAB1_KINDS:
            st.info(copy.BUILD_BUSY, icon="⏳")
        _live_run(run_id, "build")
    else:
        watched = st.session_state.get("watched_run")
        last = runs.last_run()
        if watched and last and last[0] == watched and last[1].get("kind") in TAB1_KINDS:
            feed.render_run(watched)
            _render_audit_offer(watched)
            st.divider()
        else:
            st.caption(copy.BUILD_IDLE_CAPTION)
        _render_build_form()
        _render_past_runs()


# ─── Tab 2: cite a sentence ─────────────────────────────────────────────────

with tab_cite:
    if chunks == 0:
        st.info(copy.NO_CORPUS, icon="📭")

    with st.form("cite_form"):
        draft = st.text_area(copy.CITE_SENTENCE, placeholder=copy.CITE_SENTENCE_PLACEHOLDER, height=120)
        top_k = st.slider(copy.CITE_TOP_K, 1, 10, config.DEFAULT_TOP_K)
        cite_submitted = st.form_submit_button(copy.CITE_BUTTON, type="primary", disabled=chunks == 0)

    if cite_submitted and draft.strip():
        from research_assistant.agents import agent4_assistant

        try:
            resources = get_cached_search_resources(_get_resources_mtime())
            with st.spinner(copy.CITE_SPINNER):
                result = agent4_assistant.suggest_citation(draft.strip(), top_k=top_k, search_resources=resources)
            st.session_state["cite_result"] = result or "empty"
        except RuntimeError:
            st.session_state["cite_result"] = "empty"
        except Exception as e:  # noqa: BLE001
            st.exception(e)
            st.session_state["cite_result"] = None

    cr = st.session_state.get("cite_result")
    if cr == "empty":
        st.info(copy.CITE_NOTHING, icon="🤷")
    elif isinstance(cr, dict):
        _render_suggestion(cr)


# ─── Tab 3: cite a whole draft ──────────────────────────────────────────────


def _finished_run(key, kinds):
    """(run_id, meta) of the run this session launched under `key`, once it
    is no longer running — the finished state Tab 3 renders from."""
    run_id = st.session_state.get(key)
    if not run_id:
        return None
    meta = runlog.read_meta(run_id)
    if not meta or meta.get("status") == "running" or meta.get("kind") not in kinds:
        return None
    return run_id, meta


def _render_cited_draft(run_id, meta):
    written = (meta.get("inputs") or {}).get("out_path")
    if meta.get("status") == "stopped":
        st.error(copy.BATCH_ABORTED, icon="⚠️")
        return
    if not written or not os.path.exists(written):
        return
    with open(written, encoding="utf-8") as fh:
        cited = fh.read()
    st.markdown(f"###### {copy.BATCH_RESULT}")
    st.text_area("Result", cited, height=260, key=f"batch_out:{run_id}")
    st.download_button(copy.BATCH_DOWNLOAD, cited, file_name=os.path.basename(written), key=f"dl:{run_id}")

    mapping_path = written.replace(".txt", "_citations.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, encoding="utf-8") as fh:
            mapping = json.load(fh)
        st.markdown(f"**{copy.BATCH_SOURCES}** — {len(mapping)}")
        st.json(mapping, expanded=False)

    report_path = written.replace(".txt", "_report.md")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report = fh.read()
        with st.expander(copy.BATCH_DECISIONS):
            st.markdown(report)

    st.divider()
    st.caption(copy.VERIFY_CAPTION)
    if st.button(copy.VERIFY_BUTTON, key=f"verify:{run_id}", disabled=runs.active_run() is not None):
        from research_assistant.agents.agent8_verifier import verify_draft

        cached_res = get_cached_search_resources(_get_resources_mtime())
        vid = _launch("verify", {"draft": written}, verify_draft, written, search_resources=cached_res)
        if vid:
            st.session_state["verify_run"] = vid
            st.rerun()

    verified = _finished_run("verify_run", ("verify",))
    if verified and (verified[1].get("inputs") or {}).get("draft") == written:
        feed.render_run(verified[0], allow_stop=False, slot="verified")
        md_path = written.replace(".txt", "_verification.md")
        if os.path.exists(md_path):
            tiles = transcript.verdict_tiles(runlog.read_events(verified[0]))
            with open(md_path, encoding="utf-8") as fh:
                # Open the report whenever anything is wrong — a run that
                # judged nothing is exactly when the reader needs it.
                with st.expander(copy.VERIFY_REPORT, expanded=tiles["review"] > 0 or tiles["not_judged"] > 0):
                    st.markdown(fh.read())


with tab_batch:
    st.caption(copy.BATCH_CAPTION)
    if chunks == 0:
        st.info(copy.NO_CORPUS, icon="📭")

    active = runs.active_run()
    if active and active[1].get("kind") in ("cite", "verify"):
        _live_run(active[0], "batch")
    else:
        if active:
            st.info(copy.TAB3_BUSY_ELSEWHERE, icon="⏳")
        uploaded = st.file_uploader(copy.BATCH_UPLOAD, type=["txt"], key="batch_upload")
        pasted = st.text_area(copy.BATCH_PASTE, height=200, key="batch_paste")
        run_batch = st.button(copy.BATCH_BUTTON, type="primary", disabled=chunks == 0 or active is not None)

        if run_batch:
            if not (uploaded or pasted.strip()):
                st.warning(copy.BATCH_NEED_TEXT, icon="⚠️")
            else:
                from research_assistant.agents.agent5_batch_citer import run_batch_citer

                os.makedirs(config.DRAFTS_DIR, exist_ok=True)
                text = uploaded.read().decode("utf-8") if uploaded else pasted
                with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=config.DRAFTS_DIR, delete=False,
                                                 encoding="utf-8") as fh:
                    fh.write(text)
                    draft_path = fh.name
                out_path = draft_path.replace(".txt", "_cited.txt")
                cached_res = get_cached_search_resources(_get_resources_mtime())
                rid = _launch("cite", {"draft": draft_path, "out_path": out_path}, run_batch_citer,
                              draft_path, out_path, search_resources=cached_res)
                if rid:
                    st.session_state["batch_run"] = rid
                    st.session_state.pop("verify_run", None)
                    st.rerun()

        cited_run = _finished_run("batch_run", ("cite",))
        if cited_run:
            feed.render_run(cited_run[0], allow_stop=False, slot="cited")
            _render_cited_draft(*cited_run)


# ─── Tab 4: research chat ───────────────────────────────────────────────────

with tab_chat:
    st.caption(copy.CHAT_CAPTION)
    if chunks == 0:
        st.info(copy.NO_CORPUS_CHAT, icon="📭")
    else:
        if "chat_agent" not in st.session_state:
            from research_assistant.agents.agent7_research_chat import ResearchChat

            try:
                cached_res = get_cached_search_resources(_get_resources_mtime())
                st.session_state["chat_agent"] = ResearchChat(top_k=5, search_resources=cached_res)
            except Exception as e:
                st.warning(f"Could not load search index: {e}")
                st.session_state["chat_agent"] = None
        agent = st.session_state.get("chat_agent")
        if agent is None:
            st.info(copy.NO_CORPUS_CHAT, icon="📭")
        else:
            c1, c2 = st.columns([1, 4])
            if c1.button(copy.CHAT_CLEAR):
                agent.clear_history()
                st.rerun()
            if c2.button(copy.CHAT_EXPORT):
                st.success(f"Saved to {agent.export_conversation()}")

            for msg in agent.history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if question := st.chat_input(copy.CHAT_INPUT):
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Searching literature and formulating response…"):
                            stream = agent.stream_turn(question)
                            first_chunk = next(stream, None)
                        if first_chunk is not None:
                            def _generator():
                                yield first_chunk
                                yield from stream
                            st.write_stream(_generator())
                        else:
                            st.info(copy.CHAT_EMPTY_REPLY)
                    except Exception as e:  # noqa: BLE001
                        st.exception(e)
                if agent.last_sources:
                    with st.expander(f"Sources · {len(agent.last_sources)}"):
                        for s in agent.last_sources:
                            st.caption(f"**{s['document']}** — {s['citation']}")


# ─── Tab 5: how to use ──────────────────────────────────────────────────────

with tab_help:
    # Rendered from the repo's own HOW_TO_USE.md so the doc and the in-app help
    # cannot drift apart.
    guide = os.path.join(config.PROJECT_ROOT, "HOW_TO_USE.md")
    try:
        with open(guide, encoding="utf-8") as f:
            text = f.read()
        # Relative links are correct on GitHub but dead inside the app, which
        # serves no such routes — show them as filenames instead. http(s) links
        # are left alone.
        text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"`\1`", text)
        st.markdown(text)
    except OSError:
        st.warning("HOW_TO_USE.md is missing from this build — read it in the repo instead.", icon="📄")
```

- [ ] **Step 2: Add the transcript checker**

**Create `scripts/run_check.py` with exactly this content:**

```python
"""Print a run's transcript and check the narrated-runs gates against it.

    PYTHONPATH=. python scripts/run_check.py               # the current run
    PYTHONPATH=. python scripts/run_check.py <run_id>

Gates (spec 2026-09-12-narrated-runs §6), each printed PASS/FAIL/SKIP:
  G1  a `brief` event precedes the first `search_tried` (research-idea runs)
  G2  the stage order of the events is monotone in the feed's section order
  G3  every paper_read has a document; every paper_summarised names a document
      that was read in the same stage (or the summary came before the read —
      allowed — but never for an unknown document)
  G4  the last event is `finished` and its status equals run.json's
  G5  a cancelled run's last stage event is followed by nothing but `finished`
  G6  no two events share a seq; seq is contiguous from 1
Exit code 1 if any gate FAILs.
"""

from __future__ import annotations

import json
import sys

from research_assistant.shared import runlog, transcript as tr

SECTION_RANK = {k: i for i, k in enumerate(tr.SECTION_ORDER)}


def check(run_id: str) -> int:
    meta, events = runlog.read_run(run_id)
    if meta is None:
        print(f"no such run: {run_id}")
        return 1
    print(f"{meta['run_id']}  kind={meta['kind']}  status={meta['status']}  summary={meta.get('summary')!r}")
    print(f"inputs: {json.dumps(meta.get('inputs'), ensure_ascii=False)[:200]}")
    print()
    for e in events:
        p = e.get("payload") or {}
        brief = {k: (v if not isinstance(v, str) or len(v) < 60 else v[:57] + "…")
                 for k, v in p.items() if k not in ("sample", "text", "summary", "description", "passages")}
        print(f"{e['seq']:>4}  {e['ts'][11:19]}  {str(e.get('stage')):<12} {e['type']:<22} "
              f"{json.dumps(brief, ensure_ascii=False)[:110]}")
    print()

    failures = 0

    def gate(name, ok, why=""):
        nonlocal failures
        state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        failures += 0 if ok in (True, None) else 1
        print(f"{name}: {state}" + (f" — {why}" if why else ""))

    types = [e["type"] for e in events]
    is_search_run = meta["kind"] == "pipeline" and (meta.get("inputs") or {}).get("query") and not (
        (meta.get("inputs") or {}).get("seed_url") or (meta.get("inputs") or {}).get("seed_file"))
    if is_search_run and "search_tried" in types:
        b = types.index("brief") if "brief" in types else None
        gate("G1 brief before search", b is not None and b < types.index("search_tried"),
             "no brief event" if b is None else "")
    else:
        gate("G1 brief before search", None, "not a research-idea run")

    ranks = [SECTION_RANK.get(tr.section_of(e), 99) for e in events]
    gate("G2 stage order monotone", all(a <= b for a, b in zip(ranks, ranks[1:])),
         "" if all(a <= b for a, b in zip(ranks, ranks[1:])) else "an event arrived after a later section had begun")

    read = {(e.get("stage"), e["payload"].get("document")) for e in events if e["type"] == "paper_read"}
    bad = [e for e in events if e["type"] == "paper_read" and not e["payload"].get("document")]
    unknown = [e for e in events if e["type"] == "paper_summarised"
               and (e.get("stage"), e["payload"].get("document")) not in read]
    gate("G3 cards well-formed", not bad and not unknown,
         f"{len(bad)} paper_read without document, {len(unknown)} summaries for unread documents" if (bad or unknown) else "")

    fin = events[-1] if events else None
    gate("G4 finished last, status agrees", bool(fin) and fin["type"] == "finished"
         and fin["payload"].get("status") == meta["status"],
         "" if fin and fin["type"] == "finished" else "last event is not `finished`")

    if meta["status"] == "cancelled":
        gate("G5 cancel tail", types[-1] == "finished" and "narration" in types[-3:] or types[-1] == "finished")
    else:
        gate("G5 cancel tail", None, "not cancelled")

    seqs = [e["seq"] for e in events]
    gate("G6 seq contiguous", seqs == list(range(1, len(seqs) + 1)))
    return 1 if failures else 0


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else runlog.current_run_id()
    if not rid:
        print("no current run")
        sys.exit(1)
    sys.exit(check(rid))
```

- [ ] **Step 3: Run the guards and the tests that import the page**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/test_app_names.py tests/test_pipeline_status.py tests/test_grobid_manager.py tests/test_grobid_controller.py tests/test_orchestrate_upload.py 2>&1 | grep -v ScriptRunContext
```

Expected: all pass — every bare name the page calls is bound; the sidebar tests find `🟢 **Pipeline: Idle**` and `Last run completed: …`

- [ ] **Step 4: Run the whole suite**

Run:

```bash
CITATION_LOG_FILE=0 python -m pytest -q -p no:cacheprovider tests/ 2>&1 | grep -v ScriptRunContext | tail -3
```

Expected: 661 passed, 1 skipped (as validated); no test writes `data/runs`

- [ ] **Step 5: Start the page and click through every tab with no run active** — it must render without an exception. `.claude/launch.json` on this machine has `marvin-v2` on port 8601; or:

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. python -m streamlit run app.py --server.port=8601 --server.headless=true
```

Expected: title *Marvin the Citebot*, five tabs, four toggles on both Tab 1 forms, sidebar `Marvin / Corpus / Operator`, *What he has read* expander (empty or listing runs), Tab 3 form with its button enabled when the corpus has chunks.

- [ ] **Step 6: Commit**

```bash
git add app.py scripts/run_check.py
git commit -m "feat(app): Marvin the Citebot — Tab 1 and Tab 3 are readers of the run transcript; runs launch on a thread; past runs; sidebar sorted by reader

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Docs

**Files:**
- Modify: `HOW_TO_USE.md` (title, Tab 1 with "What you see during a run", Tab 2/3 labels and rows, the audit as a stage, the sidebar), `PIPELINE.md` (layout tree, `data/runs/`, the run paragraph), `ARCHITECTURE.md` (§2.2 note, new §2.4), `README.md` (two table rows)

`HOW_TO_USE.md` is rendered inside Tab 5, so it must describe what the page does now. The base commit's version (f6e5f25) is what these diffs are against.

- [ ] **Step 1: Apply the four diffs**

**Apply this change to `HOW_TO_USE.md`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/HOW_TO_USE.md
+++ b/HOW_TO_USE.md
@@ -1,6 +1,6 @@
-# 📚 Autonomous Academic Research Assistant — User & Developer Guide
+# 🤖 Marvin the Citebot — User & Developer Guide
 
-Welcome to the **Autonomous Academic Research Assistant**. This guide provides comprehensive documentation on using the web application across all five tabs, understanding the AI pipeline outputs, and running or developing the system locally.
+This guide explains how to drive Marvin across his five tabs, what his outputs mean, and how to run or develop him locally. Everything long that he does — building a corpus, auditing a seed paper's citations, citing a draft, verifying the citations — is shown to you as it happens, item by item, and can be stopped.
 
 ---
 
@@ -24,7 +24,7 @@
 | Tab | Purpose | Input | Expected Output |
 |---|---|---|---|
 | **Tab 1: Research a topic** | Ingest literature, download references, audit citations, synthesize review | PDF research paper or topic query | Downloaded PDFs, indexed corpus, in-text citation audit, related-work synthesis |
-| **Tab 2: Cite a draft** | Suggest and verify citations for a single claim | Single scientific statement | Formatted `\cite{...}`, source metadata, support rationale, evidence passages |
+| **Tab 2: Cite a sentence** | Suggest and verify citations for a single claim | Single scientific statement | Formatted `\cite{...}`, source metadata, support rationale, evidence passages |
 | **Tab 3: Cite a whole draft** | Batch-cite and audit an entire academic manuscript | Plain-text draft (`.txt` or pasted) | Fully cited manuscript, BibTeX source mapping, and Agent 8 sentence-by-sentence verification audit |
 | **Tab 4: Research chat** | Multi-turn conversational research grounded in your library | Plain-English research questions | Token-by-token streaming answers with expandable evidence excerpts |
 | **Tab 5: How to use** | Complete documentation, pipeline explanations, and developer guide | None | This reference guide! |
@@ -37,22 +37,36 @@
 
 This tab is the primary entry point for building your literature database.
 
-#### 1. Uploading Research Papers (PDF or ZIP)
-* **Single PDF Upload**:
-  1. Select **Upload research paper(s)**.
+#### 1. Paper(s) I have (PDF or ZIP)
+* **Single PDF**:
+  1. Select **Paper(s) I have (PDF or ZIP)**.
   2. Drag and drop your `.pdf` file.
-  3. *(Optional)* Enter a research topic question to focus the final synthesis, or leave blank to infer from the paper.
-  4. Ensure **Audit citations** is toggled ON if you want the system to check whether the cited papers support the uploaded paper's statements.
-  5. Click **Process and Index Paper(s)**.
-* **Batch Upload (Multiple PDFs or ZIP)**:
-  1. Select and upload multiple PDFs or a `.zip` archive.
-  2. The system executes direct batch ingestion: parses all documents, chunks text, computes embeddings, and updates the search index in one pass.
-
-#### 2. Automatic Topic Search
-1. Select **Search for a paper**.
-2. Type an academic topic into **Research idea** (e.g. `GPU-accelerated filtered density function simulator`).
-3. *(Optional)* Provide an arXiv or direct PDF URL in **Seed paper URL** to pin a specific seed.
-4. Click **Build corpus**.
+  3. *(Optional)* Enter **What you want to know** to focus the answer at the end, or leave it blank and he infers it from the paper.
+  4. Leave **Audit citations** on if you want him to check whether the papers this one cites support what it says about them.
+  5. Click **Go**. He reads the paper, mines its bibliography with GROBID, fetches the open-access references, indexes and summarises all of them, answers, and then audits.
+* **Several PDFs or a ZIP**:
+  1. Select and upload the PDFs or the `.zip`.
+  2. He indexes them as they are — parse, chunk, embed, BM25 — and summarises each; then answers if asked.
+
+#### 2. A research idea
+1. Select **A research idea**.
+2. Type the idea into **The idea** (e.g. `GPU-accelerated filtered density function simulator`).
+3. *(Optional)* Paste an arXiv or direct PDF link into **Seed paper link** to pin the seed. It is also the thing to do when the search finds nothing open-access.
+4. Click **Go**.
+
+#### What you see during a run
+The form disappears and the page becomes the run. It updates every two seconds, section by section, with the content itself — not a progress bar:
+
+1. **The brief** *(research idea only)*: before searching, Marvin writes back what he understood the idea to be, three alternative phrasings he would search with, and what a good seed paper would look like. The phrasings are real: if your wording finds no open-access paper, he tries them in turn.
+2. **The seed**: which providers he asked and what they returned, the paper he chose, and its card — chunks, captions, and its summary once written.
+3. **References**: how many the seed cites, how they were extracted, the first ten, then every paper fetched (with the provider) or out of reach (with the reason).
+4. **Reading**: one card per paper as it is parsed, its summary appearing when written, and its figures and tables behind an expander — the crop beside his description — when figure analysis is on for the run.
+5. **The synthesis**: the shortlisted papers, his notes on each, and the answer with a key legend.
+6. **Citation audit**: each in-text citation of the seed as its verdict lands (see below).
+
+A **Stop** button ends the run at the next boundary — after the paper, sentence or citation he is on. A batch of papers stopped between parses is not saved; he says so and parses them again next time.
+
+The run does not belong to your browser tab. Refresh, close the tab, open another browser: the run continues and the page finds it. The operator's terminal runs (`orchestrate.py`, `watch.py`) show here the same way. When it finishes, the transcript stays above the form, and every past run is under **What he has read** below it.
 
 #### 3. Understanding Tab 1 Outputs
 
@@ -70,18 +84,18 @@
     * **Verbatim Evidence**: The exact sentence quoted from the cited PDF.
     * **Confidence & Rationale**: The model's reasoning.
     * **Paywalled References**: Clearly flagged if the paper was paywalled and could not be downloaded.
-  * You can re-run or trigger the audit anytime using the **"Audit Seed Paper Citations"** button.
+  * The audit is a stage of the run: each citation appears in the **Citation audit** section as its verdict lands, with the five tiles filling in. A run that finished without one — the toggle was off, or it was stopped — offers **Audit the seed's citations** under its transcript; that launches the audit on its own, shown the same way.
 
 ---
 
-### Tab 2 — Cite a draft (Single-Sentence Verification)
+### Tab 2 — Cite a sentence (Single-Sentence Citation)
 
 Use this tab to verify and attribute citations for an individual claim.
 
 1. Ensure the sidebar shows indexed papers (`Chunks > 0`).
-2. Enter your sentence into the claim box (e.g. `Topological edge states exhibit robust protection against non-magnetic impurities.`).
+2. Enter your sentence into **The sentence** (e.g. `Topological edge states exhibit robust protection against non-magnetic impurities.`).
 3. Select how many passages to evaluate (default: 5).
-4. Click **Suggest a citation**.
+4. Click **Find a citation**.
 5. **Output**:
    * **Suggested sentence** with LaTeX `\cite{...}` markup.
    * **Source attribution** and confidence.
@@ -96,12 +110,11 @@
 
 1. **Upload or Paste**: Upload a `.txt` manuscript or paste your draft into the editor.
 2. Click **Cite the draft**:
-   * **Agent 5** splits the draft into sentences, identifies factual assertions requiring citation, and matches them to your library.
-   * Produces a fully cited draft with `\cite{cite_1}`, `\cite{cite_2}` tags and a `_citations.json` registry.
-3. Click **Verify citations**:
-   * **Agent 8** re-retrieves evidence from the cited papers and audits every citation.
-   * Displays 5 metric tiles (`Checked`, `Supports`, `Partially`, `Need review`, `Not judged`).
-   * Generates a detailed audit report with verbatim quotations and slot evaluations (finding, scope, strength).
+   * **Agent 5** splits the draft into sentences and identifies the factual assertions; then each decision appears as he makes it — ✅ cited, with the inserted `\cite{}` and the source; ✗ declined, when the retrieved context did not support the claim; grey, when no citation was needed. **Stop** ends the run after the current sentence.
+   * Produces a fully cited draft with `\cite{cite_1}`, `\cite{cite_2}` tags and a `_citations.json` registry, with a download button and the per-sentence report.
+3. Click **Verify the citations**:
+   * **Agent 8** re-retrieves evidence from the cited papers and judges every citation — each verdict shown as it lands, the 5 metric tiles (`Checked`, `Supports`, `Partially`, `Need review`, `Not judged`) filling in.
+   * Generates a detailed report with verbatim quotations and slot evaluations (finding, scope, strength).
 
 ---
 
@@ -116,12 +129,15 @@
 
 ---
 
-## ⚡ Live Telemetry & Pipeline Status
+## ⚡ The sidebar
+
+The top of the sidebar is for you; the bottom is for whoever runs the server.
+
+- **Marvin**: what he is doing right now — *Reading paper 7 of 31*, *Sentence 2 of 12*, *Auditing citation 3 of 9* — with the pipeline stage, the item in hand, a progress bar and the elapsed time under it; or **🟢 Pipeline: Idle** with the last run's summary. **Live Activity Log** is the rolling log of background events.
+- **Corpus**: indexed papers and chunks in ChromaDB.
+- **Operator** *(collapsed)*: the chat and embedding models, the index version, and the **GROBID** controls — status indicator (green: answering on port 8070; red: unreachable, so reference extraction falls back to pattern matching without authors, years or DOIs), **Stop / Restart / Status**.
 
-The left sidebar includes an automatic real-time telemetry widget:
-- **Active State (`⚡ Pipeline Active`)**: Displays the current pipeline stage (Steps 1 to 5), the paper currently being downloaded or summarized, an item progress bar, and elapsed time.
-- **Live Activity Log**: An expandable rolling log of recent background events (e.g. `✅ Downloaded 14 reference PDFs`, `✓ Generated paper summaries with gemma4:e2b`).
-- **Idle State (`🟢 Pipeline: Idle`)**: Shows when the server is ready for new jobs and displays the completion summary of the last run.
+Every run also leaves a transcript under `data/runs/<run_id>/` (`run.json` and `events.jsonl`). `PYTHONPATH=. python scripts/run_check.py` prints the current one and checks it.
 
 ---
 
```

**Apply this change to `PIPELINE.md`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

````diff
--- a/PIPELINE.md
+++ b/PIPELINE.md
@@ -135,7 +135,13 @@
 │       ├── chunking.py     sentence windows packed within sections
 │       ├── figures.py      figure cropping, context formatting and VLM description
 │       ├── ingest_v2.py    PDF → corpus entries: extract, chunk, caption, crop, describe
-│       ├── batch_uploader.py  multi-PDF / ZIP upload → direct ingestion (UI tab 1)
+│       ├── batch_uploader.py  multi-PDF / ZIP upload → staged files (UI tab 1)
+│       ├── batch_ingest.py    staged uploads → one narrated `ingest` run (UI tab 1)
+│       ├── runlog.py       the run transcript: data/runs/<run_id>/{run.json, events.jsonl}
+│       ├── transcript.py   the pure model behind the feed: sections, cards, status line
+│       ├── narration.py    Marvin's templated lines at stage transitions (no model call)
+│       ├── brief.py        one model call before a seed search: the brief + fallback phrasings
+│       ├── respond.py      the synthesis call with its progress on the transcript
 │       ├── grobid_manager.py  GROBID lifecycle: start / stop / health (Docker, JAR or remote)
 │       ├── manifest.py     what has been ingested (data/ingested.json)
 │       ├── search.py       hybrid BM25 + dense with Reciprocal Rank Fusion
@@ -144,6 +150,10 @@
 │       ├── source_key.py   deterministic identity for a reference / document
 │       ├── atomic.py       write-temp-then-replace, for every manifest on disk
 │       └── db.py  log.py  retry.py
+│   └── ui/
+│       ├── runs.py         launch an entry point on a thread; one live run per process
+│       ├── feed.py         render a transcript (Streamlit)
+│       └── copy.py         every string the page says in Marvin's voice
 │
 ├── tests/                          pytest suite (collects unittest.TestCase classes unchanged)
 ├── .github/workflows/tests.yml
@@ -156,6 +166,7 @@
     ├── physics_vectordb/           persistent ChromaDB store
     ├── seed_papers.json  extracted_citations.json  downloaded.json
     ├── failed_downloads.json  ingested.json  bm25_index.pkl
+    ├── runs/                       one directory per run: run.json + events.jsonl; `current` names the latest
     └── ingest.lock                 held while a batch is ingesting
 ```
 
@@ -266,6 +277,16 @@
 streamlit run app.py
 ```
 
+Every long operation — `orchestrate.run`, the multi-PDF ingest, the batch
+citer, the verifier — is a *run*: it opens `data/runs/<run_id>/`, appends one
+typed event per thing it produces (the seed it chose, each reference, each
+fetch outcome, each paper's summary, each figure's description, each
+sentence's decision, each verdict) to `events.jsonl`, and closes `run.json`
+with a status. The page launches runs on a thread and polls the transcript,
+so a refresh, a second browser or the operator's terminal all see the same
+run; a CLI run appears in the app the same way. `Stop` asks the run to end
+at its next item boundary. See `specs/2026-09-12-narrated-runs-design.md`.
+
 ## Configuration notes
 
 - `config.py` is the single place for model names, directory paths, and every
````

**Apply this change to `ARCHITECTURE.md`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/ARCHITECTURE.md
+++ b/ARCHITECTURE.md
@@ -136,8 +136,10 @@
 
 ### 2.2 One graph per run, `MemorySaver` checkpointer
 
-The Streamlit app rebuilds the graph on each "Build corpus" click rather than
-caching it.
+`orchestrate.run` builds a fresh graph on every call — from the CLI, from
+`watch.py`, and from the Streamlit app, which since the narrated runs
+(§2.4) calls `orchestrate.run` on a thread rather than driving the graph
+itself.
 
 **Why.** The checkpointer keys state by `thread_id` (the query string). Reusing
 one graph across runs means a second run of the same query resumes from the
@@ -145,6 +147,43 @@
 cross-run resumption is instead handled by the agents' own on-disk state, which
 is durable across process restarts (the in-memory checkpointer is not).
 
+### 2.4 Narrated runs: a transcript on disk, a thread the page does not own
+
+Every long entry point — `orchestrate.run`, `batch_ingest.ingest_uploaded`,
+`agent5_batch_citer.run_batch_citer`, `agent8_verifier.verify_draft` — wraps
+its body in `runlog.run(kind, inputs)`. The run is a directory,
+`data/runs/<run_id>/`, with `run.json` (kind, inputs, status, pid) and
+`events.jsonl`: one typed event per thing produced, with the content in the
+payload — the summary text, the figure description, the sentence's decision —
+not a description of it. `pipeline_status` stays as the compact status line.
+
+**Why a file and a thread, not the script.** The Streamlit script is
+per-session and rerun-on-interaction; a run that lives inside it dies with a
+refresh and is invisible to a second browser. The app's launcher
+(`ui/runs.launch`) starts the run synchronously, so `data/runs/current`
+names it before the page reruns, then calls the entry point on a daemon
+thread that makes no `st.*` call. The page polls the transcript from a
+`st.fragment(run_every=…)` and redraws it from scratch each tick — a pure
+function of the file (`shared/transcript` groups, `ui/feed` draws). A CLI
+run opens its own transcript through the same `runlog.run` and appears in
+the app the same way. One run per process: `start_run` refuses while
+`current` names a live run.
+
+**Stopping** is cooperative: `Stop` touches `cancel`; `runlog.check_cancelled()`
+raises `RunCancelled` at item boundaries (graph nodes, papers, figures,
+summaries, sentences, citations), never inside a model call, so a stop
+costs at most the item in flight and leaves the corpus as a crash at the
+same boundary would.
+
+**The brief.** Before a seed search, one model call restates the idea and
+proposes three phrasings (`shared/brief`). Agent 0 tries them only after the
+typed query finds nothing open-access. Everything else the run says is
+templated (`shared/narration`) — no model cost, no drift.
+
+**Trade-off.** A daemon thread dies with the process (the next start marks
+the run `failed`); two-second polling re-renders a few hundred elements per
+reader; a batch stopped between parses is re-parsed next time.
+
 ---
 
 ## 3. Discovery (Agent 0)
```

**Apply this change to `README.md`** (a unified diff against the file as it is at the base commit; apply it with `git apply` from a temp file, or by hand hunk by hunk — the `-` lines must match verbatim):

```diff
--- a/README.md
+++ b/README.md
@@ -55,8 +55,8 @@
 
 | Tab | You give him | He gives you |
 |-----|--------------|--------------|
-| **Research a topic** | a research idea — or PDFs / a ZIP of them | a corpus built from the literature, and a synthesis of what's already been done on your idea |
-| **Cite a draft** | one sentence | the sentence rewritten with `\cite{key}`, the source, and why that source supports the claim |
+| **Research a topic** | a research idea — or PDFs / a ZIP of them | a corpus built from the literature, and a synthesis of what's already been done on your idea — narrated as he goes: the seed, the references, each paper's summary as he writes it |
+| **Cite a sentence** | one sentence | the sentence rewritten with `\cite{key}`, the source, and why that source supports the claim |
 | **Cite a whole draft** | a `.txt` draft | the cited draft, a key → source map for BibTeX, a sentence-by-sentence report, and a verdict on every citation: *supports*, *partially*, *contradicts*, *does not support*, or *unclear* |
 | **Research chat** | questions | streamed answers grounded in the corpus, with the sources he used |
 | **How to use** | nothing | [HOW_TO_USE.md](HOW_TO_USE.md), the step-by-step guide |
```

- [ ] **Step 2: Check the in-app rendering of the guide** — open Tab 5; the relative links are shown as filenames (existing behaviour), the new sections read.

- [ ] **Step 3: Commit**

```bash
git add HOW_TO_USE.md PIPELINE.md ARCHITECTURE.md README.md
git commit -m "docs: narrated runs — what the reader sees, the transcript on disk, the thread, stopping; Marvin's name in the guide

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: The live check (spec §6)

No code. Run the page against the v2 index — on this machine `.claude/launch.json` → `marvin-v2` on port 8601 with `CITATION_INDEX_VERSION=2` — and record the outcome in a short `notimportant/superpowers/plans/2026-09-12-narrated-runs-verification.md` (what was run, which gates passed, what was seen). GROBID must be up at `localhost:8070` and Ollama at `localhost:11434` with `gemma4:e2b` and `nomic-embed-text`.

If you would rather not touch the real corpus: copy `data/physics_vectordb`, `data/bm25_index_v2.pkl`, `data/ingested_v2.json`, `data/downloaded.json`, `data/seed_papers.json` into a scratch directory and set `CITATION_DATA_DIR` to it — that is how this plan was validated.

- [ ] **Gate 1 — a `cite` run, watched.** Tab 3: paste three sentences (two factual, one not), *Cite the draft*. Expected: the form is replaced by the header `● Sentence 1 of 3 · …` with *Stop*, the section *Citing — 0 of 3* with the narration line; rows appear one at a time (✅ with `\cite{}` and sources, grey with the reason); the sidebar's *Marvin* line reads `Sentence k of 3`; when done the header reads `○ Done — k of 3 sentences cited`, the cited draft, download, mapping and report render under it, and *Verify the citations* is offered. Then verify: verdicts land one at a time; the tiles fill; the report expander opens if anything needs review.

- [ ] **Gate 2 — a research-idea run, watched, stopped.** Tab 1: *A research idea*, a physics idea, *Go*. Expected, in order and while it runs: `● Starting`, then *The brief* (its text: two sentences, three numbered phrasings, one on the seed) while the search runs; *The seed* with `arxiv: “…” → n results, m with a PDF`, the seed card, the narration line; *References — n found · …* with the numbers line, *The first 10*, the fetched/out-of-reach rows landing as they happen (the `<sub>` provider line rendered, not raw); the sidebar `Fetching — k of n checked`. Press **Stop** during fetch: `Stopping after the current item…`, then within one paper the header `○ Stopped by you`, the transcript kept, the form back beneath a divider, the sidebar *Idle*. Then:

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. python scripts/run_check.py
```

Expected: the events printed, `G1 … G6: PASS` (G5 PASS, the run is cancelled), exit 0.

- [ ] **Gate 3 — refresh mid-run; the run outlives its session.** Start another `cite` run (or the same idea again — the seed is cached now), reload the browser tab while it is running. Expected: the new session shows *He is busy with another run…* on Tab 1 and the feed on Tab 3 with the same header; the run finishes (`run.json` → `done`) although the launching session is gone. Open a second browser to the same page: same feed.

- [ ] **Gate 4 — a full run to the synthesis and the audit.** Tab 1: upload one PDF you have (single file, all four toggles on), *Go*. Watch it through *Reading* (one card per fetched paper, summaries landing later in the stage, figures behind the expander when *Read figures* is on) to *The synthesis* (shortlist with scores, notes if the synthesis-depth build is in, the text with its key legend) and *Citation audit* (tiles filling, one expander per claim). Expected: `○ Done`, `run_check.py` all PASS, and the *What he has read* expander lists the run.

- [ ] **Gate 5 — the Streamlit log is clean.** After the above, the server's output contains no `missing ScriptRunContext` (the run thread made no `st.*` call) and no traceback.

- [ ] **Gate 6 — the CLI path leaves the same transcript.** With the page open on Tab 1 and idle:

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. python orchestrate.py --query "<the same idea>" --ask
```

Expected: the page shows the run live within two seconds — the CLI opened the run through the same `runlog.run` — and `data/runs/current` names it.

- [ ] **Record** the six gates with dates, the run ids, and anything seen that the reader should know, in the verification file; commit it.

```bash
git add notimportant/superpowers/plans/2026-09-12-narrated-runs-verification.md
git commit -m "chore(ui): narrated runs — live check on the v2 index

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** §2A (the run on disk, `current`, kinds, `runlog` API, forked-child no-op, `pipeline_status.run_id`) → Task 1. §2B (every entry point wraps itself; `ui/runs.launch`; refusal while live; daemon thread with no `st.*`) → Tasks 7, 8, 9, 10, 12. §2C (cancel file, `check_cancelled` at every listed boundary, `track_stage` treats it as a stop, the unsaved batch narrated) → Tasks 2, 5, 6, 7, 9. §2D (every event and payload in the table, `cite`/`verify` events, the ingest ordering note) → Tasks 4–9, fixed by `tests/test_transcript.py`; the audit added on the base commit gets the same treatment (`audit_started` / `audit_item` / `audit_done`, stage `audit`, section between synthesis and cite) → Tasks 3, 9, 11, 12. §2E (the brief before the search, `parse_brief`, alternates as real fallbacks, `via_query`, `CITATION_BRIEF=0`, templated lines for file/URL seeds) → Tasks 4, 7. §2F (form-or-run, the 2 s fragment re-reading the whole file, `transcript.py` pure and tested, the section layout, latest open, any session sees the run, past runs, Tab 3 rows, the sidebar split) → Tasks 3, 11, 12. §2G (name, icon, tagline, copy in one file, `HOW_TO_USE.md`, no theme) → Tasks 10, 12, 13. §2H (`app.py` shrinks; the guard test stays) → Tasks 8, 12. §3 honoured: `hybrid_search`, the index, ingestion's batch structure, `pipeline_status` behaviour, `num_ctx`, Tab 2/4 logic, the theme and `config.toml` untouched. §4 config names → Task 1. §5 errors → `RunActive` (Task 1/10), stale `running` (Task 1), torn line (Task 1), brief failure (Task 4/7), `research_answer` without `on_progress` (Task 7), the `ScriptRunContext` grep (Task 14). §6 → the unit tests per task and Task 14's gates.

**Placeholder scan.** No TBD/TODO; every step has its code or its diff; every command has its expected result.

**Type consistency.** Event names and payload keys used by the emitters (Tasks 4–9) are the ones `transcript.py` (Task 3) reads and `feed.py` (Task 11) draws — `tests/test_transcript.py` and `tests/test_ui_feed.py` are written against the same synthetic events. `runs.launch(kind, inputs, target, *args, cleanup=None, **kwargs)` is what `app._launch` / `_start_pipeline` / `_start_ingest` / Tab 3 call. `feed.render_run(run_id, allow_stop, slot)` is what `app._live_run(run_id, slot)` and the finished-state renders call. `fetch_papers()`'s return keys (`fetched`, `unavailable`) are what `orchestrate.fetch` reads. `research_answer_narrated` is imported by both `orchestrate.respond` and `batch_ingest`. `seed_found.path` (Agent 0 and the orchestrator) is what `app._render_audit_offer` reads.
