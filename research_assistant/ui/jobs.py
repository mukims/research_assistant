"""Running the pipeline in the background, and finding, adopting and showing its runs."""

import os
import time

import streamlit as st

from research_assistant import config
from research_assistant.shared import pipeline_status
from research_assistant.ui.common import _graph
from research_assistant.ui.results import _render_seed_and_downloads

def _execute_pipeline(inputs: dict, cfg: dict, display_q: str, on_snapshot=None) -> dict:
    """Run the graph to completion and return its final state.

    Runs in a job thread (see shared/run_jobs.py): it must never touch
    Streamlit. Progress goes through pipeline_status — the file every
    session already polls — and *on_snapshot* receives the partial state
    after each node so a session can show the seed and its downloads while
    the run continues. Cancellation is the pipeline_status flag the sidebar
    button sets. Failure raises; the job records it.
    """
    graph = _graph()
    final: dict = {}
    pipeline_status.set_status(
        active=True,
        stage="discover",
        stage_label="Finding seed paper",
        current_step=1,
        total_steps=5,
        detail=f"Starting pipeline for: {display_q[:50]}",
    )
    pipeline_status.add_event(f"🚀 Pipeline started for: {display_q[:40]}")
    try:
        for update in graph.stream(inputs, cfg, stream_mode="updates"):
            if pipeline_status.is_cancel_requested():
                break
            for node, payload in update.items():
                final.update(payload or {})
                if on_snapshot is not None:
                    on_snapshot({**final, "_last_node": node})
                if payload and payload.get("stopped"):
                    break
            if pipeline_status.is_cancel_requested():
                break
        if pipeline_status.is_cancel_requested():
            pipeline_status.add_event("⏹️ Pipeline stopped by user")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail="Pipeline stopped by user")
            raise pipeline_status.PipelineCancelledError("Pipeline stopped by user")
        final = graph.get_state(cfg).values
        if final.get("stopped"):
            pipeline_status.add_event(f"⚠️ Pipeline stopped early: {final['stopped'][:60]}")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail=f"Stopped: {final['stopped'][:60]}")
        else:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            pipeline_status.add_event(f"✅ Pipeline completed: {display_q[:40]}")
            pipeline_status.set_status(
                active=False, stage="idle", stage_label="Idle", detail="Pipeline complete",
                last_completed_at=now_iso, last_summary=f"Completed {display_q[:40]}",
            )
        return final
    except pipeline_status.PipelineCancelledError:
        raise
    except BaseException as e:  # noqa: BLE001
        if pipeline_status.is_cancellation(e):
            pipeline_status.add_event("⚠️ Pipeline cancelled")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail="Pipeline cancelled")
        else:
            detail_str = pipeline_status.format_exception_detail(e)
            pipeline_status.add_event(f"❌ Pipeline failed: {detail_str}")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail=f"Pipeline failed: {detail_str}")
        raise
    finally:
        seed_file_path = inputs.get("seed_file")
        if seed_file_path:
            try:
                raw_dir_abs = os.path.abspath(config.RAW_DIR)
                seed_abs = os.path.abspath(seed_file_path)
                if not (seed_abs == raw_dir_abs or seed_abs.startswith(raw_dir_abs + os.sep)):
                    if os.path.exists(seed_file_path):
                        os.unlink(seed_file_path)
            except OSError:
                pass


def _start_pipeline_job(
    query: str,
    seed_url_val: str | None = None,
    seed_file_path: str | None = None,
    ask: bool = True,
    force: bool = False,
    audit_citations: bool = True,
    synthesis_mode: str | None = None,
    origin: str = "audit",
    enable_ocr: bool | None = None,
) -> str | None:
    """Start the pipeline as a server-side job and remember it in the URL.

    Returns the job id, or None when another run is active (a warning is
    shown). The id is deterministic in the input, so re-submitting the same
    paper while its run is active attaches to that run instead of starting
    a second one.
    """
    from research_assistant.shared import run_jobs

    thread_seed = query or (os.path.basename(seed_file_path) if seed_file_path else "run")
    job_id = run_jobs.job_id_for(thread_seed)
    cfg = {"configurable": {"thread_id": job_id}}
    inputs = {
        "query": query,
        "workers": 1,
        "force": force,
        "ask": ask,
        "seed_url": seed_url_val,
        "seed_file": seed_file_path,
        "audit_citations": audit_citations,
        "synthesis_mode": synthesis_mode,
        "enable_ocr": enable_ocr,
    }
    display_q = query or (os.path.basename(seed_file_path) if seed_file_path else "") or (seed_url_val or "") or "topic"

    def runner(on_snapshot):
        return _execute_pipeline(inputs, cfg, display_q, on_snapshot=on_snapshot)

    try:
        run_jobs.start_job(job_id, display_q, runner, origin=origin)
    except run_jobs.JobBusy as busy:
        st.warning(
            f"A run is already active on the server ({busy.job.label}). "
            "Wait for it to finish, or stop it from the sidebar, then submit again.",
            icon="⏳",
        )
        return None
    st.query_params["job"] = job_id
    st.session_state["active_job"] = job_id
    st.session_state["active_job_origin"] = origin
    return job_id


def _resolve_job(job_id: str):
    """(state, final, error, origin) for a job id, from the registry first
    and the run record on disk second — the disk is what survives a server
    restart. state is running | done | failed | cancelled | None."""
    from research_assistant.shared import run_jobs

    job = run_jobs.get_job(job_id)
    if job is not None:
        if job.state == "running":
            return "running", None, None, job.origin
        if job.state == "done":
            return "done", job.result, None, job.origin
        return job.state, None, job.error, job.origin
    record = run_jobs.load_run(job_id)
    if record:
        return "done", record.get("final") or {}, None, record.get("origin", "audit")
    return None, None, None, None


def _adopt_job_result(final: dict, origin: str) -> None:
    """Put a finished run where the tabs already look for it."""
    label = final.get("query") or final.get("seed_label") or os.path.basename(final.get("seed_path") or "") or ""
    if origin == "idea":
        st.session_state["idea_result"] = final
        st.session_state["idea_query"] = label
    else:
        st.session_state["audit_result"] = final
        st.session_state["audit_query"] = label
        st.session_state["build_result"] = final
        st.session_state["build_query"] = label
    st.session_state.pop("active_job", None)
    st.session_state.pop("active_job_origin", None)


@st.fragment(run_every="2s")
def _render_job_progress(job_id: str):
    """The live view of a running job. Rerenders every 2 s from
    pipeline_status and the job's snapshot; when the job ends, reruns the
    whole app so the tab renders the result."""
    from research_assistant.shared import run_jobs

    job = run_jobs.get_job(job_id)
    if job is None or job.state != "running":
        st.rerun(scope="app")
        return
    with st.container(border=True):
        st.markdown(f"#### ⏳ Running: `{job.label}`")
        st.caption(
            "This run continues on the server. You can refresh this page, open it in another tab, "
            "or come back later — the URL keeps the job id, and the report will be here when it finishes."
        )
        status = pipeline_status.get_status()
        stage = status.get("stage_label") or "Starting"
        try:
            step, total = int(status.get("current_step") or 1), int(status.get("total_steps") or 5)
        except (ValueError, TypeError):
            step, total = 1, 5
        try:
            item_c, item_t = int(status.get("item_current") or 0), int(status.get("item_total") or 0)
        except (ValueError, TypeError):
            item_c, item_t = 0, 0
        if item_t > 0:
            st.progress(min(1.0, max(0.0, item_c / item_t)), text=f"Step {step}/{total}: {stage} — {item_c}/{item_t}")
        else:
            st.progress(min(1.0, max(0.0, step / max(1, total))), text=f"Step {step}/{total}: {stage}")
        if status.get("detail"):
            st.caption(f"⚙️ {status['detail']}")
        if status.get("recent_events"):
            with st.expander("📋 Live Activity Log", expanded=False):
                for ev in reversed(status["recent_events"][-8:]):
                    st.write(ev)
        if job.snapshot:
            _render_seed_and_downloads(job.snapshot.get("query") or job.label, job.snapshot)


def _handle_job_state(origin: str) -> bool:
    """Show a running job, adopt a finished one, explain a failed one.

    The job id comes from this session (it started the run) or from the
    URL (the page was refreshed, or opened elsewhere). Returns True when a
    running job was rendered and the tab should show nothing else.
    """
    job_id = st.session_state.get("active_job") or st.query_params.get("job")
    if not job_id:
        return False
    state, final, error, job_origin = _resolve_job(job_id)
    if job_origin not in (None, origin):
        return False  # belongs to the other tab
    if state == "running":
        _render_job_progress(job_id)
        return True
    if state == "done":
        already = st.session_state.get("idea_result" if origin == "idea" else "audit_result")
        if not already or st.session_state.get("active_job") == job_id:
            _adopt_job_result(final or {}, job_origin or origin)
        return False
    if state in ("failed", "cancelled"):
        st.session_state.pop("active_job", None)
        if state == "failed":
            st.error(f"The run failed: {error}", icon="❌")
        else:
            st.warning("The run was stopped before it finished.", icon="⏹️")
        return False
    st.info(
        f"No record of run `{job_id}` on this server — it may have been started before the last restart "
        "and not finished. Pick a recent run below or start a new one.",
        icon="ℹ️",
    )
    st.session_state.pop("active_job", None)
    return False


def _render_recent_runs(origin: str) -> None:
    """Open a finished run from disk without re-uploading anything."""
    from research_assistant.shared import run_jobs

    runs = [r for r in run_jobs.list_runs() if r.get("origin", "audit") == origin]
    if not runs:
        return
    options = {f"{r['label'] or r['seed_name'] or r['job_id']} — {r['saved_at'][:16].replace('T', ' ')}": r["job_id"] for r in runs}
    with st.container(border=True):
        st.markdown("###### 🗂️ Recent runs on this server")
        choice = st.selectbox("Open a finished run", list(options), key=f"recent_runs_{origin}", label_visibility="collapsed")
        if st.button("Open", key=f"open_recent_run_{origin}"):
            st.query_params["job"] = options[choice]
            st.session_state["active_job"] = options[choice]
            st.session_state["active_job_origin"] = origin
            st.rerun()
