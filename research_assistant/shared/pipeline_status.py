"""
pipeline_status.py — Real-time pipeline status tracking and telemetry recorder.

Provides thread-safe and process-safe telemetry for background pipeline activity:
seed indexing, reference extraction, paper fetching, LLM summarization,
chunk embedding, and BM25 index rebuilding.

State is persisted to data/pipeline_status.json using atomic writes
(atomic_write_json) so readers never see partial or truncated JSON.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
import threading
from typing import Any, Callable, Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

from research_assistant.shared.atomic import atomic_write_json

MAX_RECENT_EVENTS = 15
STALE_TIMEOUT_SECONDS = 300  # 5 minutes

VALID_STAGES = {
    "idle",
    "discover",
    "ingest_seed",
    "extract",
    "fetch",
    "ingest_refs",
    "respond",
}

STAGE_METADATA: dict[str, tuple[int, str]] = {
    "idle": (0, "Idle"),
    "discover": (1, "Finding seed paper"),
    "ingest_seed": (2, "Indexing seed paper"),
    "extract": (3, "Extracting reference list"),
    "fetch": (4, "Fetching referenced papers"),
    "ingest_refs": (5, "Ingesting and summarizing papers"),
    "respond": (5, "Synthesizing answer"),
}

DEFAULT_STATUS: dict[str, Any] = {
    "active": False,
    "stage": "idle",
    "stage_label": "Idle",
    "current_step": 0,
    "total_steps": 5,
    "item_current": 0,
    "item_total": 0,
    "current_item_name": "",
    "detail": "",
    "started_at": "",
    "updated_at": "",
    "recent_events": [],
    "pid": None,
    "last_completed_at": "",
    "last_summary": "",
}

STATUS_PATH_OVERRIDE: Optional[str] = None
_thread_lock = threading.RLock()
_lock_depth = 0
_lock_handle: Optional[Any] = None

# Progress callback support
_local = threading.local()
_global_callbacks: list[Callable[[dict[str, Any]], None]] = []
_global_callbacks_lock = threading.Lock()


def _reset_at_fork() -> None:
    """Reset lock state in forked child processes so they acquire independent locks."""
    global _lock_depth, _lock_handle, _thread_lock
    _lock_depth = 0
    _lock_handle = None
    _thread_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_at_fork)


def register_progress_callback(
    cb: Callable[[dict[str, Any]], None],
    current_thread_only: bool = True,
) -> Callable[[], None]:
    """Register a callback invoked synchronously on update_progress or set_status.

    If current_thread_only is True (default), the callback is bound to the
    calling thread (e.g. Streamlit's script execution thread), preventing
    cross-thread session context collisions.
    """
    if current_thread_only:
        if not hasattr(_local, "callbacks"):
            _local.callbacks = []
        _local.callbacks.append(cb)

        def unregister():
            if hasattr(_local, "callbacks") and cb in _local.callbacks:
                _local.callbacks.remove(cb)

        return unregister
    else:
        with _global_callbacks_lock:
            _global_callbacks.append(cb)

        def unregister():
            with _global_callbacks_lock:
                if cb in _global_callbacks:
                    _global_callbacks.remove(cb)

        return unregister


def _notify_callbacks(status: dict[str, Any]) -> None:
    """Notify registered callbacks synchronously with a copy of status."""
    st_copy = dict(status)
    # 1. Thread-local callbacks
    local_cbs = getattr(_local, "callbacks", None)
    if local_cbs:
        for cb in list(local_cbs):
            try:
                cb(st_copy)
            except Exception:
                pass
    # 2. Global callbacks
    with _global_callbacks_lock:
        global_cbs = list(_global_callbacks)
    for cb in global_cbs:
        try:
            cb(st_copy)
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_status_path() -> str:
    """Return the filesystem path to pipeline_status.json."""
    if STATUS_PATH_OVERRIDE is not None:
        return STATUS_PATH_OVERRIDE
    try:
        from research_assistant import config

        return getattr(
            config,
            "PIPELINE_STATUS_PATH",
            os.path.join(config.DATA_DIR, "pipeline_status.json"),
        )
    except Exception:
        return os.path.join("data", "pipeline_status.json")


def set_status_path(path: Optional[str]) -> None:
    """Override the status path (useful for testing)."""
    global STATUS_PATH_OVERRIDE
    STATUS_PATH_OVERRIDE = path


def _get_lock_path() -> str:
    return get_status_path() + ".lock"


@contextlib.contextmanager
def _status_lock():
    """Acquire thread and process locks for reading/modifying status state.

    Fully re-entrant across both thread and process boundaries within the
    same process.
    """
    global _lock_depth, _lock_handle
    with _thread_lock:
        if _lock_depth == 0:
            lock_file = _get_lock_path()
            os.makedirs(os.path.dirname(os.path.abspath(lock_file)) or ".", exist_ok=True)
            if fcntl is not None:
                try:
                    _lock_handle = open(lock_file, "a")
                    fcntl.flock(_lock_handle, fcntl.LOCK_EX)
                except BaseException:
                    if _lock_handle is not None:
                        try:
                            _lock_handle.close()
                        except OSError:
                            pass
                    _lock_handle = None
                    raise
        _lock_depth += 1
        try:
            yield
        finally:
            _lock_depth -= 1
            if _lock_depth == 0 and _lock_handle is not None:
                if fcntl is not None:
                    try:
                        fcntl.flock(_lock_handle, fcntl.LOCK_UN)
                    except OSError:
                        pass
                try:
                    _lock_handle.close()
                except OSError:
                    pass
                _lock_handle = None


def _is_ingest_locked() -> bool:
    """Check whether ingest.lock is currently held by any process."""
    try:
        from research_assistant.shared import ingestion

        lock = getattr(ingestion, "_ingest_thread_lock", None)
        if lock is not None:
            if getattr(lock, "_is_owned", lambda: False)():
                return True
            if not lock.acquire(blocking=False):
                return True
            lock.release()
    except Exception:
        pass

    if fcntl is None:
        return False

    try:
        from research_assistant import config

        lock_path = getattr(
            config,
            "INGEST_LOCK_PATH",
            os.path.join(config.DATA_DIR, "ingest.lock"),
        )
    except Exception:
        lock_path = os.path.join("data", "ingest.lock")

    if not os.path.exists(lock_path):
        return False
    try:
        with open(lock_path, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                return False
            except OSError:
                return True
    except Exception:
        return False


def _is_pid_running(pid: Any) -> bool:
    """Check if process with pid is running."""
    if not pid:
        return False
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
        os.kill(pid_int, 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _load_status_from_disk() -> dict[str, Any]:
    """Read the status file from disk without applying stale resolution."""
    path = get_status_path()
    if not os.path.exists(path):
        res = dict(DEFAULT_STATUS)
        res["recent_events"] = []
        return res
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                res = dict(DEFAULT_STATUS)
                res["recent_events"] = []
                return res
            merged = dict(DEFAULT_STATUS)
            merged.update(data)
            if not isinstance(merged["recent_events"], list):
                merged["recent_events"] = []
            else:
                merged["recent_events"] = list(merged["recent_events"])
            return merged
    except (json.JSONDecodeError, OSError, ValueError):
        res = dict(DEFAULT_STATUS)
        res["recent_events"] = []
        return res


def _persist_status_locked(status: dict[str, Any]) -> None:
    """Write status atomically. Must be called while holding _status_lock or atomic."""
    path = get_status_path()
    atomic_write_json(path, status, ensure_ascii=False, default=str)


def _check_and_resolve_stale(status: dict[str, Any]) -> dict[str, Any]:
    """Detect and reset stale active status if the process died or timed out."""
    if not status.get("active", False):
        return status

    updated_at_str = status.get("updated_at")
    pid = status.get("pid")
    is_stale = False

    if not updated_at_str:
        is_stale = not _is_ingest_locked() and not _is_pid_running(pid)
    else:
        try:
            cleaned = updated_at_str.replace("Z", "+00:00")
            updated_time = datetime.fromisoformat(cleaned)
            if updated_time.tzinfo is None:
                updated_time = updated_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age = (now - updated_time).total_seconds()
            if age > STALE_TIMEOUT_SECONDS:
                is_stale = not _is_ingest_locked() and not _is_pid_running(pid)
        except (ValueError, TypeError):
            is_stale = not _is_ingest_locked() and not _is_pid_running(pid)

    if is_stale:
        with _status_lock():
            # Re-read status under lock to avoid clobbering a newly started active process
            disk_status = _load_status_from_disk()
            if not disk_status.get("active", False):
                return disk_status

            disk_updated_at = disk_status.get("updated_at")
            disk_pid = disk_status.get("pid")
            disk_stale = False
            if not disk_updated_at:
                disk_stale = not _is_ingest_locked() and not _is_pid_running(disk_pid)
            else:
                try:
                    c = disk_updated_at.replace("Z", "+00:00")
                    u_t = datetime.fromisoformat(c)
                    if u_t.tzinfo is None:
                        u_t = u_t.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - u_t).total_seconds() > STALE_TIMEOUT_SECONDS:
                        disk_stale = not _is_ingest_locked() and not _is_pid_running(disk_pid)
                except (ValueError, TypeError):
                    disk_stale = not _is_ingest_locked() and not _is_pid_running(disk_pid)

            if not disk_stale:
                return disk_status

            disk_status["active"] = False
            disk_status["stage"] = "idle"
            disk_status["stage_label"] = "Idle"
            disk_status["detail"] = "Pipeline timed out (stale process detected)"
            disk_status["started_at"] = ""
            disk_status["pid"] = None
            events = list(disk_status.get("recent_events", []))
            events.append("⚠️ Previous run timed out (no active process)")
            disk_status["recent_events"] = events[-MAX_RECENT_EVENTS:]
            disk_status["updated_at"] = _now_iso()
            _persist_status_locked(disk_status)
            res = dict(disk_status)

        _notify_callbacks(res)
        return res

    return status


def get_status() -> dict[str, Any]:
    """Return the current pipeline status, handling missing/corrupt files and staleness."""
    status = _load_status_from_disk()
    return _check_and_resolve_stale(status)


def set_status(**kwargs) -> dict[str, Any]:
    """Update status fields atomically and return the updated status dictionary."""
    with _status_lock():
        status = _load_status_from_disk()
        was_active = status.get("active", False)
        now_active = kwargs.get("active", was_active)

        for k, v in kwargs.items():
            status[k] = v

        if now_active and not was_active:
            # Fresh start: record new started_at unless caller explicitly provided one
            if "started_at" not in kwargs:
                status["started_at"] = _now_iso()
            if "pid" not in kwargs:
                status["pid"] = os.getpid()
        elif now_active:
            if not status.get("started_at"):
                status["started_at"] = kwargs.get("started_at") or _now_iso()
            if not status.get("pid") and "pid" not in kwargs:
                status["pid"] = os.getpid()
        else:
            # Idle / inactive
            if "started_at" not in kwargs:
                status["started_at"] = ""
            if "pid" not in kwargs:
                status["pid"] = None
            if "stage" not in kwargs:
                status["stage"] = "idle"
                status["stage_label"] = "Idle"
                status["current_step"] = 0
            elif status.get("stage") == "idle":
                if "current_step" not in kwargs:
                    status["current_step"] = 0
                if "stage_label" not in kwargs:
                    status["stage_label"] = "Idle"

        # Validate stage and fill sensible defaults if omitted
        stage = status.get("stage", "idle")
        if "stage" in kwargs:
            if stage not in VALID_STAGES:
                stage = "idle"
                status["stage"] = "idle"
            if "stage_label" not in kwargs and stage in STAGE_METADATA:
                status["stage_label"] = STAGE_METADATA[stage][1]
            if "current_step" not in kwargs and stage in STAGE_METADATA:
                status["current_step"] = STAGE_METADATA[stage][0]

        if "recent_events" in kwargs:
            raw_ev = kwargs["recent_events"]
            if isinstance(raw_ev, list):
                status["recent_events"] = list(raw_ev)[-MAX_RECENT_EVENTS:]
            else:
                status["recent_events"] = []
        elif not isinstance(status.get("recent_events"), list):
            status["recent_events"] = []
        else:
            status["recent_events"] = status["recent_events"][-MAX_RECENT_EVENTS:]

        if "updated_at" not in kwargs:
            status["updated_at"] = _now_iso()

        _persist_status_locked(status)
        result = dict(status)

    _notify_callbacks(result)
    return result


def update_progress(
    item_current: Optional[int] = None,
    item_total: Optional[int] = None,
    current_item_name: Optional[str] = None,
    detail: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience helper to record item-by-item progress."""
    updates: dict[str, Any] = dict(kwargs)
    if item_current is not None:
        updates["item_current"] = item_current
    if item_total is not None:
        updates["item_total"] = item_total
    if current_item_name is not None:
        updates["current_item_name"] = current_item_name
    if detail is not None:
        updates["detail"] = detail
    return set_status(**updates)


def add_event(msg: str) -> None:
    """Append a user-facing event to recent_events (FIFO capped at MAX_RECENT_EVENTS)."""
    if not msg:
        return
    with _status_lock():
        status = _load_status_from_disk()
        events = list(status.get("recent_events", []))
        events.append(str(msg))
        status["recent_events"] = events[-MAX_RECENT_EVENTS:]
        status["updated_at"] = _now_iso()
        _persist_status_locked(status)
        result = dict(status)
    _notify_callbacks(result)


def clear_status(keep_events: bool = True) -> dict[str, Any]:
    """Reset status back to idle, optionally preserving recent events."""
    with _status_lock():
        status = _load_status_from_disk()
        events = status.get("recent_events", []) if keep_events else []
        last_completed = status.get("last_completed_at", "")
        last_summary = status.get("last_summary", "")

        reset = dict(DEFAULT_STATUS)
        reset["recent_events"] = events[-MAX_RECENT_EVENTS:] if events else []
        reset["last_completed_at"] = last_completed
        reset["last_summary"] = last_summary
        reset["updated_at"] = _now_iso()

        _persist_status_locked(reset)
        result = dict(reset)
    _notify_callbacks(result)
    return result


@contextlib.contextmanager
def track_stage(
    stage: str,
    stage_label: Optional[str] = None,
    total_steps: int = 5,
    current_step: Optional[int] = None,
    detail: str = "",
    item_total: int = 0,
    mark_idle_on_exit: bool = False,
    last_summary: Optional[str] = None,
):
    """Context manager for safely tracking execution of a pipeline stage."""
    meta_step, meta_label = STAGE_METADATA.get(stage, (0, stage.replace("_", " ").capitalize()))
    eff_step = current_step if current_step is not None else meta_step
    eff_label = stage_label or meta_label

    set_status(
        active=True,
        stage=stage,
        stage_label=eff_label,
        total_steps=total_steps,
        current_step=eff_step,
        detail=detail,
        item_total=item_total,
        item_current=0,
    )

    stop_heartbeat = threading.Event()

    def _heartbeat():
        while not stop_heartbeat.wait(timeout=15.0):
            try:
                set_status(updated_at=_now_iso())
            except Exception:
                pass

    hb_thread = threading.Thread(target=_heartbeat, daemon=True, name="pipeline_heartbeat")
    hb_thread.start()

    try:
        yield
    except BaseException as exc:
        stop_heartbeat.set()
        if hb_thread.is_alive():
            hb_thread.join(timeout=1.0)
        if isinstance(exc, KeyboardInterrupt):
            err_msg = f"⚠️ {eff_label} cancelled by user (SIGINT)"
            detail_msg = f"Cancelled in {eff_label}"
        elif isinstance(exc, SystemExit):
            err_msg = f"⚠️ {eff_label} stopped (SystemExit)"
            detail_msg = f"Stopped in {eff_label}"
        else:
            err_msg = f"❌ Error in {eff_label}: {exc}"
            detail_msg = f"Failed in {eff_label}: {exc}"
        status = _load_status_from_disk()
        events = status.get("recent_events", [])
        if not events or events[-1] != err_msg:
            add_event(err_msg)
        set_status(
            active=False,
            stage="idle",
            stage_label="Idle",
            detail=detail_msg,
        )
        raise
    else:
        stop_heartbeat.set()
        if hb_thread.is_alive():
            hb_thread.join(timeout=1.0)
        if mark_idle_on_exit:
            summary = last_summary or (f"+{item_total} papers indexed" if item_total else "Stage completed")
            set_status(
                active=False,
                stage="idle",
                stage_label="Idle",
                detail="Stage completed",
                last_completed_at=_now_iso(),
                last_summary=summary,
            )
        else:
            set_status(updated_at=_now_iso())
    finally:
        stop_heartbeat.set()
        if hb_thread.is_alive():
            hb_thread.join(timeout=1.0)
