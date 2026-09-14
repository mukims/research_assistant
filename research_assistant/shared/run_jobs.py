"""Pipeline runs that outlive the browser session.

A browser refresh ends a Streamlit session, and Streamlit stops that
session's script at its next widget call. When the ten-minute pipeline ran
inside the session — and reported progress through the session's own
progress bar from deep inside `pipeline_status.update_progress` — that next
widget call was the pipeline's, and the run died wherever it stood. The
result had only ever lived in the session's state, so even a run that had
finished was gone with the tab.

Here a run is a daemon thread in the server process. It talks to the UI
only through `pipeline_status` (a file every session already polls) and
this registry, and its result is written to disk under a job id that the
UI keeps in the query string, so the session that started it, the same
session after a refresh, or another browser can all pick it up.

Nothing in this module may touch Streamlit: a job thread has no
ScriptRunContext.
"""

import hashlib
import json
import os
import threading
import traceback
from datetime import datetime
from typing import Callable, Optional

from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger

logger = get_logger("run_jobs")

_LOCK = threading.Lock()
_JOBS: dict[str, "Job"] = {}


class JobBusy(RuntimeError):
    """Another run is active. The pipeline holds the ingest lock and the
    model backend; two at once would fight over both."""

    def __init__(self, job: "Job"):
        super().__init__(f"A run is already active: {job.label}")
        self.job = job


class Job:
    def __init__(self, job_id: str, label: str, origin: str):
        self.job_id = job_id
        self.label = label
        self.origin = origin
        self.state = "running"          # running | done | failed | cancelled
        self.started_at = _now()
        self.finished_at: Optional[str] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.snapshot: dict = {}        # the latest partial pipeline state, for a live view
        self.thread: Optional[threading.Thread] = None

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id, "label": self.label, "origin": self.origin,
            "state": self.state, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error,
        }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def job_id_for(seed: str) -> str:
    """Deterministic id for a run — the same sha1 prefix the graph uses as
    its checkpoint thread id, so the two never disagree."""
    return hashlib.sha1(seed.encode()).hexdigest()[:16]


def get_job(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)


def active_job() -> Optional[Job]:
    with _LOCK:
        for job in _JOBS.values():
            if job.state == "running":
                return job
    return None


def start_job(job_id: str, label: str, runner: Callable, origin: str = "audit") -> Job:
    """Run *runner* in a daemon thread and register it under *job_id*.

    *runner* is called with no arguments, or with ``on_snapshot`` if it
    accepts one — a callable the runner may feed partial state to for the
    live view. It returns the final state dict, which is saved to disk on
    success. A job with the same id that is still running is returned as-is
    (a re-submit attaches rather than restarts); a different running job
    raises JobBusy.
    """
    with _LOCK:
        existing = _JOBS.get(job_id)
        if existing is not None and existing.state == "running":
            return existing
        for other in _JOBS.values():
            if other.state == "running":
                raise JobBusy(other)
        job = Job(job_id, label, origin)
        _JOBS[job_id] = job

    def _run():
        try:
            final = _call_runner(runner, job)
            job.result = final if isinstance(final, dict) else {}
            try:
                save_run(job_id, job.result, label=label, origin=origin)
            except Exception as exc:  # the run succeeded; only its record failed
                logger.warning("Could not save run %s: %s", job_id, exc)
            # State flips last: a reader that sees "done" may rely on the
            # record being on disk already.
            job.state = "done"
        except BaseException as exc:  # noqa: BLE001 — a thread must never die silently
            if pipeline_status.is_cancellation(exc):
                job.state = "cancelled"
                job.error = pipeline_status.format_exception_detail(exc)
            else:
                job.state = "failed"
                job.error = pipeline_status.format_exception_detail(exc)
                logger.error("Run %s failed: %s\n%s", job_id, job.error, traceback.format_exc())
        finally:
            job.finished_at = _now()

    job.thread = threading.Thread(target=_run, name=f"run-{job_id}", daemon=True)
    job.thread.start()
    return job


def _call_runner(runner: Callable, job: Job):
    import inspect

    def on_snapshot(partial: dict) -> None:
        job.snapshot = dict(partial or {})

    try:
        params = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        params = {}
    if "on_snapshot" in params:
        return runner(on_snapshot=on_snapshot)
    return runner()


# ─── On disk ─────────────────────────────────────────────────────────────────

def runs_dir() -> str:
    """Beside the seed audits, so one directory holds everything a run
    produced. Read at call time so a test can redirect it."""
    from research_assistant.shared.seed_audit import AUDIT_DIR
    return os.path.join(AUDIT_DIR, "runs")


def run_path(job_id: str) -> str:
    return os.path.join(runs_dir(), f"{job_id}.json")


def save_run(job_id: str, final: dict, label: str, origin: str) -> str:
    """The run's final state, JSON-safe: anything json can't take is stored
    as its str(), because losing one odd value must not lose the run."""
    os.makedirs(runs_dir(), exist_ok=True)
    record = {
        "job_id": job_id,
        "label": label,
        "origin": origin,
        "saved_at": _now(),
        "seed_name": os.path.basename(final.get("seed_path") or "") if isinstance(final, dict) else "",
        "final": final,
    }
    path = run_path(job_id)
    # atomic_write_json serialises with json.dump; make the payload safe first.
    safe = json.loads(json.dumps(record, default=str))
    atomic_write_json(path, safe)
    return path


def load_run(job_id: str) -> Optional[dict]:
    path = run_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Unreadable run record %s: %s", path, exc)
        return None


def list_runs(limit: int = 20) -> list[dict]:
    """Newest first: what the "Recent runs" picker shows."""
    directory = runs_dir()
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        record = load_run(name[:-5])
        if not record:
            continue
        entries.append({
            "job_id": record.get("job_id", name[:-5]),
            "label": record.get("label", ""),
            "origin": record.get("origin", ""),
            "seed_name": record.get("seed_name", ""),
            "saved_at": record.get("saved_at", ""),
            "mtime": os.path.getmtime(path),
            "path": path,
        })
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries[:limit]


def _reset_registry_for_tests() -> None:
    with _LOCK:
        _JOBS.clear()
