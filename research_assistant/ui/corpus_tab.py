"""Tab: manage, explore, and batch-ingest documents into the corpus."""

import os
import re
import streamlit as st

from research_assistant import config
from research_assistant.shared import manifest, pipeline_status
from research_assistant.shared.batch_uploader import (
    fetch_and_stage_urls_or_dois,
    scan_and_stage_directory,
    unpack_and_stage_uploads,
)
from research_assistant.shared.ingestion import ingest_pdfs
from research_assistant.shared.paper_index import PaperIndex, normalise_arxiv
from research_assistant.shared.source_key import normalise_doi
from research_assistant.ui.common import (
    _corpus_stats,
    _manifest,
    get_cached_search_resources,
)
from research_assistant.ui.status import _render_live_pipeline_status

_ARXIV_FILENAME_RE = re.compile(r"^arxiv_(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf$", re.I)


def _doc_key_from_path(path: str | None) -> str:
    if not path:
        return ""
    return os.path.basename(path).strip().replace(" ", "_").lower()


def _index_manifest_records(raw_manifest: dict | None) -> tuple[dict, dict, dict, dict]:
    """Index manifest records by raw key, PDF doc_key, normalized DOI, and normalized arXiv ID."""
    by_key: dict[str, dict] = {}
    by_doc: dict[str, dict] = {}
    by_doi: dict[str, dict] = {}
    by_arxiv: dict[str, dict] = {}

    if not isinstance(raw_manifest, dict):
        return by_key, by_doc, by_doi, by_arxiv

    for raw_k, rec in raw_manifest.items():
        if not isinstance(rec, dict):
            continue
        by_key[raw_k] = rec
        by_key[f"abstract_only:{raw_k}"] = rec
        rec_key = rec.get("key")
        if isinstance(rec_key, str) and rec_key:
            by_key.setdefault(rec_key, rec)
            by_key.setdefault(f"abstract_only:{rec_key}", rec)
        dk = _doc_key_from_path(rec.get("path"))
        if dk:
            by_doc.setdefault(dk, rec)
        nd = normalise_doi(rec.get("doi"))
        if nd:
            by_doi.setdefault(nd, rec)
        na = normalise_arxiv(rec.get("arxiv_id"))
        if na:
            by_arxiv.setdefault(na, rec)

    return by_key, by_doc, by_doi, by_arxiv


def build_corpus_explorer_records(
    ingested_map: dict[str, str],
    downloaded: dict | None = None,
    seeds: dict | None = None,
    paper_index: PaperIndex | None = None,
    filter_query: str = "",
) -> list[dict]:
    """Join ingested document keys with PaperIndex, downloaded.json, and seed_papers.json."""
    if not ingested_map:
        return []

    if paper_index is None:
        paper_index = PaperIndex.load(config.PAPER_INDEX_PATH)
    if downloaded is None:
        downloaded = _manifest(config.DOWNLOADED_JSON_PATH)
    if seeds is None:
        seeds = _manifest(config.SEED_PAPERS_PATH)

    dl_by_key, dl_by_doc, dl_by_doi, dl_by_arxiv = _index_manifest_records(downloaded)
    sd_by_key, sd_by_doc, sd_by_doi, sd_by_arxiv = _index_manifest_records(seeds)

    q = (filter_query or "").strip().lower()
    doc_records: list[dict] = []

    for doc_key, timestamp in ingested_map.items():
        pe = paper_index.resolve(doc_key) if paper_index is not None else None
        m_arx = _ARXIV_FILENAME_RE.match(doc_key)
        fn_arxiv = m_arx.group(1) if m_arx else ""

        def _lookup(by_key: dict, by_doc: dict, by_doi: dict, by_arxiv: dict) -> dict:
            hit = by_key.get(doc_key) or by_doc.get(doc_key)
            if not hit and pe is not None:
                hit = by_doc.get(pe.document)
                if not hit:
                    for d in pe.dois:
                        nd = normalise_doi(d)
                        if nd and nd in by_doi:
                            hit = by_doi[nd]
                            break
                if not hit:
                    for a in pe.arxiv_ids:
                        na = normalise_arxiv(a)
                        if na and na in by_arxiv:
                            hit = by_arxiv[na]
                            break
            if not hit and fn_arxiv:
                na = normalise_arxiv(fn_arxiv)
                if na:
                    hit = by_arxiv.get(na)
            return hit or {}

        dl_meta = _lookup(dl_by_key, dl_by_doc, dl_by_doi, dl_by_arxiv)
        sd_meta = _lookup(sd_by_key, sd_by_doc, sd_by_doi, sd_by_arxiv)

        pe_title = ""
        if pe and pe.title and pe.title.strip() and pe.title.strip().lower() != doc_key.lower():
            pe_title = pe.title.strip()

        title = (
            pe_title
            or (dl_meta.get("title") or "").strip()
            or (sd_meta.get("title") or "").strip()
            or doc_key
        )
        doi = (
            (pe.dois[0] if (pe and pe.dois) else None)
            or dl_meta.get("doi")
            or sd_meta.get("doi")
            or ""
        )
        arxiv_id = (
            (pe.arxiv_ids[0] if (pe and pe.arxiv_ids) else None)
            or dl_meta.get("arxiv_id")
            or sd_meta.get("arxiv_id")
            or fn_arxiv
            or ""
        )
        url = dl_meta.get("url") or sd_meta.get("url") or ""
        venue = (
            dl_meta.get("venue")
            or dl_meta.get("container")
            or sd_meta.get("venue")
            or ""
        )
        authors = (
            list(pe.first_authors)
            if (pe and pe.first_authors)
            else list(dl_meta.get("authors") or ())
        )

        search_corpus = f"{doc_key} {title} {doi} {arxiv_id} {venue} {' '.join(authors)}".lower()
        if q and q not in search_corpus:
            continue

        doc_records.append({
            "key": doc_key,
            "title": title,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": url,
            "venue": venue,
            "authors": authors,
            "timestamp": timestamp,
        })

    return doc_records


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
            raw_path = folder_path.strip()
            real_path = os.path.realpath(os.path.expanduser(raw_path))
            from research_assistant.shared.batch_uploader import is_safe_path
            if not is_safe_path(real_path):
                st.error(f"Access denied: Path '{raw_path}' is outside permitted workspace boundaries.", icon="🚫")
            else:
                with st.spinner(f"Scanning directory {real_path} for PDFs…"):
                    staged = scan_and_stage_directory(real_path, destination_dir=config.RAW_DIR, recursive=recursive)
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

                def _on_batch_progress(status_or_item_c=None, item_t=None, name=None, **kwargs):
                    try:
                        if isinstance(status_or_item_c, dict):
                            st_dict = status_or_item_c
                            item_c = st_dict.get("item_current", 0)
                            item_t = st_dict.get("item_total", 0)
                            name = st_dict.get("current_item_name") or st_dict.get("name") or ""
                            progress = st_dict.get("progress")
                            message = st_dict.get("detail") or st_dict.get("message") or ""
                            stage = st_dict.get("stage") or ""
                        else:
                            item_c = status_or_item_c if status_or_item_c is not None else 0
                            item_t = item_t if item_t is not None else 0
                            name = name or ""
                            progress = None
                            message = ""
                            stage = ""

                        if progress is not None:
                            try:
                                pct = min(1.0, max(0.0, float(progress)))
                            except (ValueError, TypeError):
                                pct = 0.0
                        else:
                            try:
                                c_flt = float(item_c) if item_c is not None else 0.0
                                t_flt = float(item_t) if item_t is not None else 0.0
                                pct = min(1.0, max(0.0, c_flt / t_flt)) if t_flt > 0.0 else 0.0
                            except (ValueError, TypeError, ZeroDivisionError):
                                pct = 0.0

                        name_str = str(name)[:40] if name is not None else ""
                        try:
                            has_t = float(item_t) > 0 if item_t is not None else False
                        except (ValueError, TypeError):
                            has_t = False

                        display_txt = str(message) if message else (
                            f"Parsing paper [{item_c}/{item_t}]: {name_str}" if has_t else f"Parsing paper: {name_str}"
                        )
                        batch_progress.progress(pct, text=display_txt)
                    except Exception:
                        pass

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

    doc_records = build_corpus_explorer_records(
        ingested_map,
        downloaded=_manifest(config.DOWNLOADED_JSON_PATH),
        seeds=_manifest(config.SEED_PAPERS_PATH),
        paper_index=PaperIndex.load(config.PAPER_INDEX_PATH),
        filter_query=filter_query,
    )

    st.markdown(f"**Showing {len(doc_records)} of {len(ingested_map)} indexed document(s)**")

    if doc_records:
        for idx, rec in enumerate(doc_records[:100]):
            bits = [f"Key: `{rec['key']}`"]
            if rec.get("authors"):
                bits.append(f"Author: {rec['authors'][0]}")
            if rec.get("venue"):
                bits.append(rec["venue"])
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
