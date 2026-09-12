"""
Tests for research_assistant/shared/pipeline_status.py
"""

import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import time
import pytest

from research_assistant.shared import pipeline_status


@pytest.fixture
def temp_status_file(tmp_path):
    status_path = str(tmp_path / "pipeline_status.json")
    pipeline_status.set_status_path(status_path)
    yield status_path
    pipeline_status.set_status_path(None)


def test_default_status(temp_status_file):
    status = pipeline_status.get_status()
    assert status["active"] is False
    assert status["stage"] == "idle"
    assert status["stage_label"] == "Idle"
    assert status["current_step"] == 0
    assert status["total_steps"] == 5
    assert status["item_current"] == 0
    assert status["item_total"] == 0
    assert status["current_item_name"] == ""
    assert status["detail"] == ""
    assert status["recent_events"] == []


def test_set_status_and_update_progress(temp_status_file):
    res = pipeline_status.set_status(
        active=True,
        stage="fetch",
        stage_label="Fetching referenced papers",
        current_step=4,
        total_steps=5,
        item_total=10,
        item_current=0,
        detail="Starting fetch",
    )
    assert res["active"] is True
    assert res["stage"] == "fetch"
    assert res["started_at"] != ""
    assert res["updated_at"] != ""
    assert res["pid"] == os.getpid()

    updated = pipeline_status.update_progress(
        item_current=3,
        current_item_name="Paper 3 title",
        detail="Downloaded 3 of 10",
    )
    assert updated["item_current"] == 3
    assert updated["item_total"] == 10
    assert updated["current_item_name"] == "Paper 3 title"
    assert updated["detail"] == "Downloaded 3 of 10"

    fetched = pipeline_status.get_status()
    assert fetched["item_current"] == 3
    assert fetched["current_item_name"] == "Paper 3 title"


def test_add_event_and_capping(temp_status_file):
    for i in range(20):
        pipeline_status.add_event(f"Event {i}")

    status = pipeline_status.get_status()
    events = status["recent_events"]
    assert len(events) == 15
    assert events[0] == "Event 5"
    assert events[-1] == "Event 19"


def test_clear_status(temp_status_file):
    pipeline_status.set_status(
        active=True,
        stage="fetch",
        current_step=4,
        item_current=5,
        item_total=10,
        current_item_name="Some paper",
        detail="Fetching...",
        last_completed_at="2026-09-12T10:00:00Z",
        last_summary="+5 papers",
    )
    pipeline_status.add_event("Event 1")

    # Clear status keeping events
    cleared = pipeline_status.clear_status(keep_events=True)
    assert cleared["active"] is False
    assert cleared["stage"] == "idle"
    assert cleared["current_step"] == 0
    assert cleared["item_current"] == 0
    assert cleared["recent_events"] == ["Event 1"]
    assert cleared["last_completed_at"] == "2026-09-12T10:00:00Z"
    assert cleared["last_summary"] == "+5 papers"

    # Clear status discarding events
    cleared2 = pipeline_status.clear_status(keep_events=False)
    assert cleared2["recent_events"] == []


def test_missing_or_corrupted_file(temp_status_file):
    # File doesn't exist
    if os.path.exists(temp_status_file):
        os.remove(temp_status_file)
    status = pipeline_status.get_status()
    assert status["active"] is False

    # Corrupted / invalid JSON
    with open(temp_status_file, "w") as f:
        f.write("{corrupt json...")
    corrupt_status = pipeline_status.get_status()
    assert corrupt_status["active"] is False
    assert corrupt_status["stage"] == "idle"

    # Empty file
    with open(temp_status_file, "w") as f:
        f.write("")
    empty_status = pipeline_status.get_status()
    assert empty_status["active"] is False


def test_track_stage_success(temp_status_file):
    with pipeline_status.track_stage("discover", "Finding seed paper", detail="Searching arXiv"):
        st = pipeline_status.get_status()
        assert st["active"] is True
        assert st["stage"] == "discover"
        assert st["stage_label"] == "Finding seed paper"
        assert st["current_step"] == 1
        assert st["detail"] == "Searching arXiv"

    # Should still be active by default (for pipeline node transitions)
    after = pipeline_status.get_status()
    assert after["active"] is True


def test_track_stage_mark_idle_on_exit(temp_status_file):
    with pipeline_status.track_stage("ingest_refs", "Ingesting papers", mark_idle_on_exit=True):
        assert pipeline_status.get_status()["active"] is True

    after = pipeline_status.get_status()
    assert after["active"] is False
    assert after["stage"] == "idle"


def test_track_stage_exception(temp_status_file):
    with pytest.raises(ValueError, match="Something went wrong"):
        with pipeline_status.track_stage("extract", "Extracting references"):
            raise ValueError("Something went wrong")

    status = pipeline_status.get_status()
    assert status["active"] is False
    assert status["stage"] == "idle"
    assert "Failed in Extracting references" in status["detail"]
    assert any("❌ Error in Extracting references" in ev for ev in status["recent_events"])


def test_stale_detection_timed_out(temp_status_file, monkeypatch):
    # Active, but updated_at is 6 minutes ago, no lock, dead pid
    six_mins_ago = (datetime.now(timezone.utc) - timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(temp_status_file, "w") as f:
        json.dump(
            {
                "active": True,
                "stage": "fetch",
                "stage_label": "Fetching referenced papers",
                "current_step": 4,
                "total_steps": 5,
                "item_current": 10,
                "item_total": 50,
                "detail": "Long running fetch",
                "started_at": six_mins_ago,
                "updated_at": six_mins_ago,
                "recent_events": [],
                "pid": 9999999,  # Unlikely to be a running PID
            },
            f,
        )

    # Mock pid running to False and lock to False
    monkeypatch.setattr(pipeline_status, "_is_pid_running", lambda pid: False)
    monkeypatch.setattr(pipeline_status, "_is_ingest_locked", lambda: False)

    status = pipeline_status.get_status()
    assert status["active"] is False
    assert status["stage"] == "idle"
    assert "timed out" in status["detail"]
    assert any("timed out" in ev for ev in status["recent_events"])


def test_stale_detection_not_stale_if_recent(temp_status_file, monkeypatch):
    one_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(temp_status_file, "w") as f:
        json.dump(
            {
                "active": True,
                "stage": "fetch",
                "updated_at": one_min_ago,
                "pid": 9999999,
            },
            f,
        )
    monkeypatch.setattr(pipeline_status, "_is_pid_running", lambda pid: False)
    monkeypatch.setattr(pipeline_status, "_is_ingest_locked", lambda: False)

    status = pipeline_status.get_status()
    assert status["active"] is True


def test_stale_detection_not_stale_if_lock_held(temp_status_file, monkeypatch):
    six_mins_ago = (datetime.now(timezone.utc) - timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(temp_status_file, "w") as f:
        json.dump(
            {
                "active": True,
                "stage": "ingest_refs",
                "updated_at": six_mins_ago,
                "pid": 9999999,
            },
            f,
        )
    monkeypatch.setattr(pipeline_status, "_is_pid_running", lambda pid: False)
    monkeypatch.setattr(pipeline_status, "_is_ingest_locked", lambda: True)

    status = pipeline_status.get_status()
    # Ingest lock is held, so not marked stale
    assert status["active"] is True


def test_stale_detection_not_stale_if_pid_running(temp_status_file, monkeypatch):
    six_mins_ago = (datetime.now(timezone.utc) - timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(temp_status_file, "w") as f:
        json.dump(
            {
                "active": True,
                "stage": "ingest_refs",
                "updated_at": six_mins_ago,
                "pid": 1234,
            },
            f,
        )
    monkeypatch.setattr(pipeline_status, "_is_pid_running", lambda pid: True)
    monkeypatch.setattr(pipeline_status, "_is_ingest_locked", lambda: False)

    status = pipeline_status.get_status()
    # Process is running, so not marked stale
    assert status["active"] is True


def test_thread_safety(temp_status_file):
    # Concurrent event additions from multiple threads
    def worker(worker_id):
        for i in range(10):
            pipeline_status.add_event(f"w{worker_id}-{i}")
            time.sleep(0.001)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    status = pipeline_status.get_status()
    assert len(status["recent_events"]) == 15  # Capped at 15


def test_agent2_fetcher_status_integration(temp_status_file, monkeypatch, tmp_path):
    from research_assistant.agents import agent2_fetcher

    refs = [
        {"title": "Paper One", "doi": "10.1001/p1", "doi_confidence": "high"},
        {"title": "Paper Two", "doi": "10.1002/p2", "doi_confidence": "high"},
    ]

    monkeypatch.setattr(agent2_fetcher, "_load_references", lambda: refs)
    monkeypatch.setattr(agent2_fetcher, "_load_state", lambda path, default: {})
    monkeypatch.setattr(agent2_fetcher, "_checkpoint", lambda dl, f: None)
    monkeypatch.setattr(agent2_fetcher, "PULLED_PDFS_DIR", str(tmp_path))

    def mock_resolve_doi(ref):
        return ref["doi"], "grobid"

    def mock_try_unpaywall(doi, dest):
        if "p1" in doi:
            return True, "unpaywall: saved"
        return False, "paywalled"

    monkeypatch.setattr(agent2_fetcher, "resolve_doi", mock_resolve_doi)
    monkeypatch.setattr(agent2_fetcher, "try_unpaywall", mock_try_unpaywall)
    monkeypatch.setattr(agent2_fetcher, "try_europepmc", lambda doi, ref, dest: (False, "not found"))
    monkeypatch.setattr(agent2_fetcher, "try_arxiv", lambda ref, dest: (False, "not found"))

    agent2_fetcher.fetch_papers()

    status = pipeline_status.get_status()
    assert status["item_total"] == 2
    assert status["item_current"] == 2
    events = status["recent_events"]
    assert any("✅ Downloaded" in e for e in events)
    assert any("⚠️" in e for e in events)
    assert any("🌐 Fetch complete: 1 downloaded, 1 unavailable" in e for e in events)


def test_ingestion_status_integration(temp_status_file, monkeypatch, tmp_path):
    from research_assistant.shared import ingestion

    p1 = str(tmp_path / "paper1.pdf")
    p2 = str(tmp_path / "paper2.pdf")
    with open(p1, "w") as f:
        f.write("%PDF-1.4 dummy")
    with open(p2, "w") as f:
        f.write("%PDF-1.4 dummy")

    monkeypatch.setattr(ingestion, "get_ingested_documents", lambda: set())
    monkeypatch.setattr(
        ingestion,
        "process_pdf",
        lambda path, label, *a, **kw: [
            {"document": os.path.basename(path), "type": "text", "content": "text chunk", "page": 1, "citation": label}
        ],
    )
    monkeypatch.setattr(ingestion, "upsert_corpus", lambda corpus: 2)
    monkeypatch.setattr(ingestion, "rebuild_bm25", lambda: None)
    monkeypatch.setattr(ingestion.manifest, "add_many", lambda items: None)

    res = ingestion._ingest_pdfs_locked(
        {p1: "Paper 1", p2: "Paper 2"},
        workers=1,
        skip_ingested=True,
        rebuild_index=True,
        log_prefix="",
    )

    assert res["processed"] == 2
    status = pipeline_status.get_status()
    assert status["item_total"] == 2
    assert status["item_current"] == 2
    events = status["recent_events"]
    assert any("✅ Ingested 2 PDFs" in e for e in events)


def test_orchestrate_lifecycle_status_integration(temp_status_file, monkeypatch):
    import orchestrate

    class MockApp:
        def stream(self, inputs, config, stream_mode):
            yield {"query": inputs["query"], "seed_label": "Seed Paper"}

    monkeypatch.setattr(orchestrate, "build_graph", lambda: MockApp())

    code = orchestrate.run(query="topological wires")
    assert code == 0

    status = pipeline_status.get_status()
    assert status["active"] is False
    assert status["stage"] == "idle"
    assert "topological wires" in status["last_summary"]
    events = status["recent_events"]
    assert any("🚀 Pipeline started" in e for e in events)
    assert any("✅ Pipeline completed" in e for e in events)


def test_app_sidebar_widget_rendering(temp_status_file, monkeypatch):
    import app

    # Test idle rendering
    pipeline_status.clear_status()
    pipeline_status.set_status(
        active=False,
        last_completed_at="2026-09-12T10:00:00Z",
        last_summary="+10 papers indexed",
    )

    markdown_calls = []
    caption_calls = []
    info_calls = []
    progress_calls = []

    monkeypatch.setattr(app.st, "markdown", lambda text: markdown_calls.append(text))
    monkeypatch.setattr(app.st, "caption", lambda text: caption_calls.append(text))
    monkeypatch.setattr(app.st, "info", lambda text: info_calls.append(text))
    monkeypatch.setattr(app.st, "progress", lambda pct, text="": progress_calls.append((pct, text)))

    # Call the fragment function directly
    # Streamlit's @st.fragment wraps the function; __wrapped__ gives original if wrapped
    render_fn = getattr(app._render_sidebar_pipeline_status, "__wrapped__", app._render_sidebar_pipeline_status)
    render_fn()

    assert any("🟢 **Pipeline: Idle**" in m for m in markdown_calls)
    assert any("+10 papers indexed" in c for c in caption_calls)

    # Test active rendering
    markdown_calls.clear()
    caption_calls.clear()
    info_calls.clear()
    progress_calls.clear()

    pipeline_status.set_status(
        active=True,
        stage="fetch",
        stage_label="Fetching referenced papers",
        current_step=4,
        total_steps=5,
        item_current=45,
        item_total=111,
        current_item_name="doi:10.1038/sample",
        detail="Generating summary with Qwen2.5 (45/111)",
        started_at="2026-09-12T10:00:00Z",
    )

    render_fn()

    assert any("⚡ **Pipeline Active**" in i for i in info_calls)
    assert any("Fetching referenced papers" in m for m in markdown_calls)
    assert any("doi:10.1038/sample" in m for m in markdown_calls)
    assert any(p[0] == pytest.approx(45 / 111, rel=1e-2) for p in progress_calls)


def test_started_at_resets_on_new_pipeline_run(temp_status_file):
    old_time = "2020-01-01T00:00:00Z"
    pipeline_status.set_status(active=True, started_at=old_time)
    s1 = pipeline_status.get_status()
    assert s1["started_at"] == old_time

    # Run completes / transitions to idle
    pipeline_status.set_status(active=False, stage="idle")
    assert pipeline_status.get_status()["started_at"] == ""

    # Next run starts
    s2 = pipeline_status.set_status(active=True)
    assert s2["started_at"] != ""
    assert s2["started_at"] != old_time


def test_is_pid_running_with_invalid_types():
    assert pipeline_status._is_pid_running(None) is False
    assert pipeline_status._is_pid_running(-10) is False
    assert pipeline_status._is_pid_running(0) is False
    assert pipeline_status._is_pid_running("invalid_str") is False
    assert pipeline_status._is_pid_running([1, 2, 3]) is False
    assert pipeline_status._is_pid_running({"pid": 123}) is False


def test_track_stage_suppresses_duplicate_error_events(temp_status_file):
    try:
        with pipeline_status.track_stage("ingest_refs", "Ingesting papers"):
            with pipeline_status.track_stage("ingest_refs", "Ingesting papers"):
                raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass

    st = pipeline_status.get_status()
    errors = [e for e in st["recent_events"] if "❌ Error in Ingesting papers: Simulated failure" in e]
    assert len(errors) == 1


def test_sidebar_and_tab1_widget_corrupt_data(temp_status_file, monkeypatch):
    import app

    pipeline_status.set_status(
        active=True,
        current_step="invalid",
        total_steps="invalid",
        item_current="invalid",
        item_total="invalid",
        started_at="invalid_date",
    )

    markdown_calls = []
    progress_calls = []
    monkeypatch.setattr(app.st, "markdown", lambda text: markdown_calls.append(text))
    monkeypatch.setattr(app.st, "caption", lambda text: None)
    monkeypatch.setattr(app.st, "info", lambda text, **kw: None)
    monkeypatch.setattr(app.st, "progress", lambda pct, text="": progress_calls.append((pct, text)))

    render_sidebar = getattr(app._render_sidebar_pipeline_status, "__wrapped__", app._render_sidebar_pipeline_status)
    render_tab1 = getattr(app._render_tab1_live_status, "__wrapped__", app._render_tab1_live_status)

    render_sidebar()
    render_tab1()
    assert len(progress_calls) >= 1


def test_all_candidates_skipped_ingestion(temp_status_file, monkeypatch, tmp_path):
    from research_assistant.shared import ingestion

    p1 = str(tmp_path / "paper1.pdf")
    with open(p1, "w") as f:
        f.write("%PDF-1.4 dummy")

    monkeypatch.setattr(ingestion, "get_ingested_documents", lambda: {ingestion.pdf_key(p1)})

    res = ingestion._ingest_pdfs_locked({p1: "Paper 1"}, workers=1, skip_ingested=True, rebuild_index=False, log_prefix="")
    assert res["skipped"] == 1
    assert res["processed"] == 0

    status = pipeline_status.get_status()
    assert any("already indexed" in e for e in status["recent_events"])


def test_stage_validation(temp_status_file):
    res = pipeline_status.set_status(stage="non_existent_stage")
    assert res["stage"] == "idle"

    res2 = pipeline_status.set_status(stage="ingest_refs")
    assert res2["stage"] == "ingest_refs"
    assert res2["current_step"] == 5


def test_status_lock_reentrancy(temp_status_file):
    """Verify that _status_lock() is re-entrant within the same process/thread without deadlocking."""
    with pipeline_status._status_lock():
        with pipeline_status._status_lock():
            with pipeline_status._status_lock():
                pipeline_status.set_status(detail="Inside triply-nested lock")
    st = pipeline_status.get_status()
    assert st["detail"] == "Inside triply-nested lock"


def test_inactive_stage_resets_to_idle(temp_status_file):
    """Calling set_status(active=False) without explicit stage must reset stage to idle."""
    pipeline_status.set_status(active=True, stage="fetch", current_step=4)
    st_active = pipeline_status.get_status()
    assert st_active["active"] is True
    assert st_active["stage"] == "fetch"
    assert st_active["current_step"] == 4

    st_inactive = pipeline_status.set_status(active=False)
    assert st_inactive["active"] is False
    assert st_inactive["stage"] == "idle"
    assert st_inactive["stage_label"] == "Idle"
    assert st_inactive["current_step"] == 0


def test_progress_callbacks(temp_status_file):
    """Verify thread-local and global progress callbacks receive live telemetry updates."""
    received_local = []
    received_global = []

    unreg_local = pipeline_status.register_progress_callback(
        lambda s: received_local.append(s["detail"]),
        current_thread_only=True,
    )
    unreg_global = pipeline_status.register_progress_callback(
        lambda s: received_global.append(s["detail"]),
        current_thread_only=False,
    )

    pipeline_status.update_progress(detail="Update 1")
    pipeline_status.add_event("Event 1")

    assert "Update 1" in received_local
    assert "Update 1" in received_global

    unreg_local()
    pipeline_status.update_progress(detail="Update 2")

    assert "Update 2" not in received_local
    assert "Update 2" in received_global

    unreg_global()
    pipeline_status.update_progress(detail="Update 3")
    assert "Update 3" not in received_global


def test_agent1_standalone_crash_safety(temp_status_file, monkeypatch, tmp_path):
    """Verify unhandled crash in agent1_extractor resets active=False and records error event."""
    from research_assistant.agents import agent1_extractor

    pipeline_status.clear_status()
    dummy_pdf = str(tmp_path / "seed.pdf")
    with open(dummy_pdf, "w") as f:
        f.write("%PDF-1.4 dummy")

    monkeypatch.setattr(agent1_extractor, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(
        agent1_extractor,
        "grobid_alive",
        lambda: (_ for _ in ()).throw(RuntimeError("GROBID unhandled socket crash")),
    )

    with pytest.raises(RuntimeError, match="GROBID unhandled socket crash"):
        agent1_extractor.run_extractor()

    st = pipeline_status.get_status()
    assert st["active"] is False
    assert st["stage"] == "idle"
    assert any("❌ Error in Extracting reference list" in e for e in st["recent_events"])


def test_agent2_standalone_crash_safety(temp_status_file, monkeypatch, tmp_path):
    """Verify unhandled crash in agent2_fetcher resets active=False and records error event."""
    from research_assistant.agents import agent2_fetcher

    pipeline_status.clear_status()
    refs = [{"raw_reference": "Citation 1", "title": "Paper 1"}]
    monkeypatch.setattr(agent2_fetcher, "_load_references", lambda: refs)
    monkeypatch.setattr(agent2_fetcher, "_load_state", lambda path, default: {})
    monkeypatch.setattr(agent2_fetcher, "PULLED_PDFS_DIR", str(tmp_path))
    monkeypatch.setattr(
        agent2_fetcher,
        "_checkpoint",
        lambda d, f: (_ for _ in ()).throw(IOError("Disk write failure")),
    )

    with pytest.raises(IOError, match="Disk write failure"):
        agent2_fetcher.fetch_papers()

    st = pipeline_status.get_status()
    assert st["active"] is False
    assert st["stage"] == "idle"
    assert any("❌ Error in Fetching referenced papers" in e for e in st["recent_events"])


def test_ingestion_standalone_lifecycle(temp_status_file, monkeypatch, tmp_path):
    """Verify _ingest_pdfs_locked tracks standalone lifecycle properly."""
    from research_assistant.shared import ingestion

    pipeline_status.clear_status()
    p = str(tmp_path / "p1.pdf")
    with open(p, "w") as f:
        f.write("%PDF-1.4 dummy")

    monkeypatch.setattr(ingestion, "get_ingested_documents", lambda: set())
    monkeypatch.setattr(
        ingestion,
        "process_pdf",
        lambda path, label, *a, **kw: [
            {"document": os.path.basename(path), "type": "text", "content": "chunk", "page": 1}
        ],
    )
    monkeypatch.setattr(ingestion, "upsert_corpus", lambda corpus: 1)
    monkeypatch.setattr(ingestion, "rebuild_bm25", lambda: None)
    monkeypatch.setattr(ingestion.manifest, "add_many", lambda items: None)

    res = ingestion._ingest_pdfs_locked({p: "Paper 1"}, workers=1, skip_ingested=True, rebuild_index=True, log_prefix="")
    assert res["processed"] == 1

    st = pipeline_status.get_status()
    assert st["active"] is False
    assert st["stage"] == "idle"
    assert "+1 papers ingested" in st["last_summary"]
    assert any("✅ Ingested 1 PDFs" in e for e in st["recent_events"])


def test_is_ingest_locked_fallback(monkeypatch):
    """Verify _is_ingest_locked falls back to thread lock when fcntl is None."""
    from research_assistant.shared import ingestion

    monkeypatch.setattr(pipeline_status, "fcntl", None)
    # Initially not locked
    assert pipeline_status._is_ingest_locked() is False

    # Simulate thread lock acquired
    with ingestion._ingest_thread_lock:
        assert pipeline_status._is_ingest_locked() is True


def test_track_stage_keyboard_interrupt(temp_status_file):
    """Verify KeyboardInterrupt inside track_stage resets active to False and logs cancellation."""
    pipeline_status.clear_status()
    with pytest.raises(KeyboardInterrupt):
        with pipeline_status.track_stage("discover", "Finding seed paper"):
            raise KeyboardInterrupt()

    st = pipeline_status.get_status()
    assert st["active"] is False
    assert st["stage"] == "idle"
    assert "Cancelled in Finding seed paper" in st["detail"]
    assert any("cancelled by user" in ev for ev in st["recent_events"])


def _mp_worker(proc_id, path):
    pipeline_status.set_status_path(path)
    for i in range(15):
        pipeline_status.add_event(f"p{proc_id}-ev{i}")
        pipeline_status.update_progress(item_current=i, detail=f"p{proc_id}-i{i}")
        time.sleep(0.001)


def test_multiprocess_concurrency(temp_status_file):
    """Verify multiple concurrent OS processes writing events and progress do not deadlock or corrupt JSON."""
    import multiprocessing

    procs = [
        multiprocessing.Process(target=_mp_worker, args=(p, temp_status_file))
        for p in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=10)
        assert not p.is_alive(), f"Process {p.pid} hung or deadlocked"
        assert p.exitcode == 0

    st = pipeline_status.get_status()
    assert len(st["recent_events"]) == 15
    assert st["active"] is True or st["active"] is False


def _fork_worker(path):
    pipeline_status.set_status_path(path)
    pipeline_status.add_event("child event")
    pipeline_status.update_progress(detail="child progress")


def test_fork_lock_safety(temp_status_file):
    """Verify that child process forked while parent has lock depth resets lock depth cleanly."""
    import multiprocessing

    with pipeline_status._status_lock():
        p = multiprocessing.Process(target=_fork_worker, args=(temp_status_file,))
        p.start()
    p.join(timeout=5)
    assert not p.is_alive()
    assert p.exitcode == 0

    st = pipeline_status.get_status()
    assert any("child event" in e for e in st["recent_events"])


def test_stale_detection_notifies_callbacks(temp_status_file, monkeypatch):
    """Verify that when get_status() resolves a stale run, registered callbacks are notified."""
    six_mins_ago = (datetime.now(timezone.utc) - timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(temp_status_file, "w") as f:
        json.dump(
            {
                "active": True,
                "stage": "fetch",
                "updated_at": six_mins_ago,
                "pid": 9999999,
            },
            f,
        )
    monkeypatch.setattr(pipeline_status, "_is_pid_running", lambda pid: False)
    monkeypatch.setattr(pipeline_status, "_is_ingest_locked", lambda: False)

    received = []
    unreg = pipeline_status.register_progress_callback(lambda s: received.append(s), current_thread_only=False)
    try:
        status = pipeline_status.get_status()
        assert status["active"] is False
        assert len(received) >= 1
        assert received[-1]["active"] is False
        assert received[-1]["stage"] == "idle"
    finally:
        unreg()


def test_set_status_non_list_events_and_non_serializable(temp_status_file):
    """Verify that passing non-list recent_events or non-serializable objects does not crash."""
    res = pipeline_status.set_status(recent_events=None)
    assert isinstance(res["recent_events"], list)

    res2 = pipeline_status.set_status(recent_events="string instead of list")
    assert isinstance(res2["recent_events"], list)

    # Pass an exception instance into detail
    res3 = pipeline_status.set_status(detail=RuntimeError("test error"))
    assert "test error" in str(res3["detail"])

    # Ensure get_status loads cleanly from disk
    st = pipeline_status.get_status()
    assert "test error" in str(st["detail"])


def test_app_sidebar_summary_parentheses(temp_status_file, monkeypatch):
    """Verify that last_summary with preexisting parentheses is not double-wrapped."""
    import app

    pipeline_status.clear_status()
    pipeline_status.set_status(
        active=False,
        last_completed_at="2026-09-12T10:00:00Z",
        last_summary="(+111 papers indexed)",
    )

    caption_calls = []
    monkeypatch.setattr(app.st, "markdown", lambda text: None)
    monkeypatch.setattr(app.st, "caption", lambda text: caption_calls.append(text))
    monkeypatch.setattr(app.st, "info", lambda text: None)

    render_fn = getattr(app._render_sidebar_pipeline_status, "__wrapped__", app._render_sidebar_pipeline_status)
    render_fn()

    assert any("Last run completed: 10:00 UTC (+111 papers indexed)" in c for c in caption_calls)
    assert not any("((+111 papers indexed))" in c for c in caption_calls)




