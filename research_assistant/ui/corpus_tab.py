"""Tab: manage, explore, and batch-ingest documents into the corpus."""

import os
import streamlit as st

from research_assistant import config
from research_assistant.shared import manifest, pipeline_status
from research_assistant.shared.batch_uploader import (
    fetch_and_stage_urls_or_dois,
    scan_and_stage_directory,
    unpack_and_stage_uploads,
)
from research_assistant.shared.ingestion import ingest_pdfs
from research_assistant.ui.common import (
    _corpus_stats,
    _manifest,
    get_cached_search_resources,
)
from research_assistant.ui.status import _render_live_pipeline_status


def render_corpus_tab(chunks: int, papers: int):
    """Tab: manage, explore, and batch-ingest documents into the corpus."""
    _render_live_pipeline_status("tab_corpus")

    st.subheader("📚 Corpus Library & Batch Document Ingestor")
    st.caption(
        "Dump research papers directly into the database. Ingest multiple PDFs or ZIP archives, "
        "scan local folders on disk, or fetch batches of arXiv IDs, DOIs, and URLs into the vector database and BM25 index."
    )

    # ─── Corpus Summary Metrics ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Papers", papers)
    m2.metric("Corpus Chunks", chunks)
    m3.metric("Index Version", f"v{config.INDEX_VERSION}")
    m4.metric("Embed Engine", f"{config.EMBED_MODEL}")

    st.markdown("---")

    # ─── Ingestion Source Tabs ────────────────────────────────────────────────
    st.markdown("### 📥 Add Documents to Database")
    st.caption("Choose an ingestion source below to bulk-index papers into your knowledge base.")

    tab_upload, tab_dir, tab_urls = st.tabs([
        "📁 Multi-File / ZIP Upload",
        "📂 Scan Local Directory",
        "🌐 Paste DOIs / arXiv IDs / URLs",
    ])

    uploaded_files = None
    folder_path = ""
    recursive = True
    urls_text = ""

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Select one or multiple PDF papers (or a .zip archive)",
            type=["pdf", "zip"],
            accept_multiple_files=True,
            key="corpus_batch_files",
            help="Upload PDF files or ZIP folders containing research papers.",
        )

    with tab_dir:
        folder_path = st.text_input(
            "Local folder path on machine",
            placeholder="/Users/username/Downloads/physics_papers/",
            key="corpus_folder_path",
            help="Absolute or relative path to a directory containing PDF papers.",
        )
        recursive = st.checkbox(
            "Scan subdirectories recursively",
            value=True,
            key="corpus_recursive",
            help="Recursively find all .pdf files inside subfolders.",
        )

    with tab_urls:
        urls_text = st.text_area(
            "Paste DOIs, arXiv IDs, or direct PDF URLs (one per line)",
            placeholder="10.1103/PhysRevLett.100.016601\n2401.12345\nhttps://arxiv.org/pdf/2401.12345.pdf",
            key="corpus_urls_text",
            height=130,
            help="Enter one identifier or URL per line to download and index in bulk.",
        )

    # ─── Ingestion Options ────────────────────────────────────────────────────
    with st.expander("⚙️ Batch Ingestion Settings", expanded=False):
        c_work, c_force = st.columns(2)
        workers = c_work.slider(
            "Worker processes (concurrency)",
            min_value=1,
            max_value=4,
            value=1,
            help="Number of parallel worker processes for PDF parsing and chunk extraction.",
        )
        force = c_force.toggle(
            "Force re-index (re-process already-ingested papers)",
            value=False,
            help="Re-extract and re-embed documents even if they are already recorded as indexed.",
        )
        c_ocr, c_engine = st.columns(2)
        enable_ocr = c_ocr.toggle(
            "Enable OCR for scanned papers & notes",
            value=True,
            help="Automatically run Optical Character Recognition on scanned manuscripts, photocopy reprints, and notes that lack a digital text layer.",
        )
        ocr_engine = c_engine.selectbox(
            "OCR Engine",
            options=["easyocr", "vlm"],
            format_func=lambda e: "EasyOCR (Local MPS GPU/CPU)" if e == "easyocr" else "Vision LLM (Notes & Formulas)",
            help="EasyOCR runs fast locally on device; Vision LLM provides enhanced transcription for handwritten notes and non-standard formulas.",
        )

    # ─── Ingest Action ────────────────────────────────────────────────────────
    if st.button("🚀 Ingest Documents into Database", type="primary", use_container_width=True):
        staged = []

        if uploaded_files:
            with st.spinner("Unpacking and validating uploaded papers…"):
                staged = unpack_and_stage_uploads(uploaded_files, destination_dir=config.RAW_DIR)
        elif folder_path.strip():
            with st.spinner(f"Scanning directory {folder_path} for PDFs…"):
                staged = scan_and_stage_directory(folder_path.strip(), destination_dir=config.RAW_DIR, recursive=recursive)
        elif urls_text.strip():
            items = [line.strip() for line in urls_text.splitlines() if line.strip()]
            with st.spinner(f"Fetching {len(items)} online papers…"):
                staged = fetch_and_stage_urls_or_dois(items, destination_dir=config.RAW_DIR)
        else:
            st.warning("Please upload files, specify a folder path, or paste paper identifiers to begin.", icon="⚠️")

        if staged:
            candidates = {p["path"]: p.get("title") or os.path.basename(p["path"]) for p in staged}
            st.info(f"Staged {len(candidates)} valid research manuscript(s). Beginning ingestion…", icon="📄")

            with st.status(f"Ingesting {len(candidates)} paper(s) into database…", expanded=True) as status_box:
                batch_progress = st.progress(0.0, text=f"Preparing to ingest {len(candidates)} papers…")

                def _on_batch_progress(item_c, item_t, name):
                    if item_t > 0:
                        pct = min(1.0, max(0.0, item_c / item_t))
                        batch_progress.progress(pct, text=f"Parsing paper [{item_c}/{item_t}]: {name[:40]}")

                pipeline_status.register_progress_callback(_on_batch_progress)
                try:
                    ingest_res = ingest_pdfs(
                        candidates,
                        workers=workers,
                        skip_ingested=not force,
                        enable_ocr=enable_ocr,
                        ocr_engine=ocr_engine,
                    )
                finally:
                    pipeline_status.unregister_progress_callback(_on_batch_progress)

                if pipeline_status.is_cancel_requested():
                    status_box.update(label="Batch ingestion stopped by user", state="error")
                    st.warning("Batch ingestion stopped by user.", icon="⏹️")
                else:
                    inserted = ingest_res.get("inserted", 0)
                    processed = ingest_res.get("processed", len(candidates))
                    skipped = ingest_res.get("skipped", 0)
                    figures = ingest_res.get("figure_images", 0)
                    extraction_counts = ingest_res.get("extraction", {})
                    ocr_count = extraction_counts.get("ocr", 0)
                    extra_info = f" (including {ocr_count} scanned/OCR paper(s))" if ocr_count else ""

                    status_box.update(
                        label=f"✓ Complete — {processed} parsed, {skipped} skipped duplicates, {inserted} chunks inserted!",
                        state="complete",
                    )
                    st.success(
                        f"Successfully indexed {processed} paper(s){extra_info}! Added {inserted} new chunks and {figures} figures/tables into the database.",
                        icon="✅",
                    )

            # Invalidate cached resources & sidebar metrics so other tabs reflect the new corpus immediately
            _corpus_stats.clear()
            get_cached_search_resources.clear()

    st.markdown("---")

    # ─── Corpus Explorer & Library Inspector ──────────────────────────────────
    st.markdown("### 🔍 Corpus Library Explorer")
    st.caption("Inspect and filter documents currently stored in your knowledge base.")

    ingested_map = manifest.load()
    if not ingested_map:
        st.info("No documents indexed in database yet. Use the upload or folder scan tools above to build your corpus.", icon="📭")
        return

    filter_query = st.text_input(
        "Filter indexed documents",
        placeholder="Search by title, filename, or DOI…",
        key="corpus_search_filter",
    ).strip().lower()

    downloaded = _manifest(config.DOWNLOADED_JSON_PATH)
    seeds = _manifest(config.SEED_PAPERS_PATH)

    # Build list of records
    doc_records = []
    for doc_key, timestamp in ingested_map.items():
        doc_meta = downloaded.get(doc_key) or seeds.get(doc_key) or {}
        title = doc_meta.get("title") or doc_key
        doi = doc_meta.get("doi") or ""
        arxiv_id = doc_meta.get("arxiv_id") or ""
        url = doc_meta.get("url") or ""

        search_corpus = f"{doc_key} {title} {doi} {arxiv_id}".lower()
        if filter_query and filter_query not in search_corpus:
            continue

        doc_records.append({
            "key": doc_key,
            "title": title,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": url,
            "timestamp": timestamp,
        })

    st.markdown(f"**Showing {len(doc_records)} of {len(ingested_map)} indexed document(s)**")

    if doc_records:
        for idx, rec in enumerate(doc_records[:100]):
            bits = [f"Key: `{rec['key']}`"]
            if rec["doi"]:
                bits.append(f"[doi:{rec['doi']}](https://doi.org/{rec['doi']})")
            if rec["arxiv_id"]:
                bits.append(f"[arXiv:{rec['arxiv_id']}](https://arxiv.org/abs/{rec['arxiv_id']})")
            if rec["url"]:
                bits.append(f"[PDF link]({rec['url']})")
            if rec["timestamp"]:
                bits.append(f"Indexed: {rec['timestamp'][:10]}")

            with st.container(border=True):
                st.markdown(f"**{idx + 1}. {rec['title']}**")
                st.caption(" · ".join(bits))
