"""
Agent 2 — reference fetcher.

Takes the reference list Agent 1 extracted, resolves a DOI per reference
(trusting Agent 1 when its confidence was high/medium, otherwise asking
Crossref), then tries to download an open-access PDF through a four-source
ladder: Crossref (DOI resolution) -> Unpaywall -> Europe PMC -> arXiv.
Checkpoints downloaded.json / failed_downloads.json after every paper so a
crashed run resumes instead of starting over.

Successful downloads are written through schemas.DownloadedPaper — the typed
contract Agent 3 reads back. failed_downloads.json stays plain dicts: those
records carry a `reason` and no `path`, and have no downstream consumer
needing a contract.
"""

import html
import json
import os
import re
import requests
import time
import urllib.parse
import xml.etree.ElementTree as ET

from research_assistant.config import (
    EXTRACTED_CITATIONS_PATH,
    DOWNLOADED_JSON_PATH,
    FAILED_DOWNLOADS_PATH,
    PULLED_PDFS_DIR,
    UNPAYWALL_EMAIL,
    MAX_CITATION_LEN,
    ARXIV_RATE_LIMIT,
    UNPAYWALL_SLEEP,
    PAPER_INDEX_PATH,
)
from research_assistant.schemas import DownloadedPaper
from research_assistant.shared.log import get_logger
from research_assistant.shared.paper_index import PaperIndex
from research_assistant.shared.source_key import source_key, normalise_doi, is_authoritative
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.fetch import HEADERS, download_pdf, filename_for, is_saved_pdf
from research_assistant.shared import pipeline_status
from research_assistant.shared import academic_apis


def fetch_bibliometrics_and_abstract(*args, **kwargs):
    return academic_apis.fetch_bibliometrics_and_abstract(*args, **kwargs)


logger = get_logger("agent2")

TRUSTED_CONFIDENCE = ("high", "medium")


# ─── Retry policy ────────────────────────────────────────────────────────────
# A failed key used to be excluded from the work list forever, which made a
# ten-second network blip permanently drop those references from the corpus.
# These markers separate "the answer might differ next run" from "this paper is
# settled" — a paywall, a 404 or an absence from an index is a real answer and
# stays cached; a dropped connection, a rate limit or a 5xx is not.
_TRANSIENT_MARKERS = (
    "download error:",
    "incomplete download",
    "unpaywall error:",
    "europe pmc error:",
    "arxiv error:",
    "empty response",
    "malformed json",
)
_TRANSIENT_HTTP = re.compile(r"http (?:429|5\d\d)")


def _is_retryable(reason: str | None) -> bool:
    """True when *reason* describes a fault that could clear on its own.

    `reason` is every rung of the download ladder joined together, so one
    transient leg is enough: the paper may well be reachable next time.
    """
    text = (reason or "").lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return True
    return bool(_TRANSIENT_HTTP.search(text))


def _checkpoint(downloaded, failed):
    """Write state to disk after every paper — crash-safe incremental saves.

    Atomic because the checkpoint is the resume point: _load_state() treats an
    unparseable file as empty and starts the whole fetch again, so a truncated
    downloaded.json costs more than no checkpoint at all.
    """
    atomic_write_json(DOWNLOADED_JSON_PATH, downloaded, ensure_ascii=False)
    atomic_write_json(FAILED_DOWNLOADS_PATH, failed, ensure_ascii=False)


def _load_state(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("%s is unreadable; starting fresh.", path)
        return default


def _load_references():
    """Read agent1's output, tolerating the old flat-string format."""
    with open(EXTRACTED_CITATIONS_PATH, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return payload.get("references", [])

    # Legacy format: a list of raw citation strings.
    logger.warning("Legacy citation format detected; only raw strings available.")
    return [{"raw_reference": c, "title": None, "doi": None, "authors": []} for c in payload]


def resolve_doi(ref):
    """
    Return (doi, how). Trust agent1's DOI when it was confidently matched;
    otherwise ask Crossref using the parsed title, which is a better query
    than the whole raw reference string.
    """
    doi = normalise_doi(ref.get("doi"))
    if doi and ref.get("doi_confidence") in TRUSTED_CONFIDENCE:
        return doi, "grobid"

    query = ref.get("title") or ref.get("raw_reference")
    if not query:
        return None, None

    params = {"query.bibliographic": query, "rows": 1, "select": "DOI,title,author,issued"}
    if ref.get("authors"):
        params["query.author"] = ref["authors"][0]

    try:
        res = requests.get(
            "https://api.crossref.org/works", params=params, headers=HEADERS, timeout=15
        )
        res.raise_for_status()
        items = res.json()["message"]["items"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("Crossref lookup failed: %s", exc)
        return (doi, "grobid-unverified") if doi else (None, None)

    if not items:
        return (doi, "grobid-unverified") if doi else (None, None)

    cand = items[0]
    cand_titles = cand.get("title", [])
    cand_title = cand_titles[0] if isinstance(cand_titles, list) and cand_titles else str(cand_titles)
    cand_authors = cand.get("author", [])
    cand_author = None
    if cand_authors and isinstance(cand_authors, list):
        first_a = cand_authors[0]
        cand_author = first_a.get("family") or first_a.get("name")
    issued = cand.get("issued", {}).get("date-parts", [[None]])
    cand_year = issued[0][0] if issued and issued[0] else None

    target_title = ref.get("title")
    target_author = ref.get("authors")[0] if ref.get("authors") else None
    target_year = ref.get("year")

    from research_assistant.shared.academic_apis import verify_candidate_match
    if target_title:
        if not verify_candidate_match(target_title, cand_title, target_author, cand_author, target_year, cand_year):
            logger.info("Crossref candidate '%s' failed similarity check against target '%s'", cand_title[:40], target_title[:40])
            return (doi, "grobid-unverified") if doi else (None, None)

    return normalise_doi(cand.get("DOI")), "crossref"



def try_unpaywall(doi, dest_path):
    """
    Look for an open-access PDF for this DOI.

    Unpaywall's best_oa_location is usually the publisher's own site, which
    is also the one most likely to block an automated request. Every known
    location is tried, repositories first.

    Unpaywall refuses requests without a contact email, so with UNPAYWALL_EMAIL
    unset this makes no request at all.
    """
    if not UNPAYWALL_EMAIL:
        return False, "Unpaywall skipped: set UNPAYWALL_EMAIL"
    try:
        res = requests.get(
            f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}",
            params={"email": UNPAYWALL_EMAIL},
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as exc:
        return False, f"Unpaywall error: {exc}"

    if res.status_code != 200:
        return False, f"Unpaywall lookup failed (HTTP {res.status_code})"

    try:
        data = res.json()
    except ValueError:
        return False, "Unpaywall returned malformed JSON"

    if not data.get("is_oa"):
        return False, "paywalled / not open access"

    # Collect every location with a PDF, best_oa_location included, then
    # de-duplicate on URL since it usually repeats one of the others.
    locations = list(data.get("oa_locations") or [])
    best = data.get("best_oa_location")
    if best:
        locations.append(best)

    candidates = []
    seen = set()
    for loc in locations:
        url = (loc or {}).get("url_for_pdf")
        if url and url not in seen:
            seen.add(url)
            candidates.append({"url": url, "host_type": loc.get("host_type") or "?"})

    if not candidates:
        return False, "open access, but no direct PDF URL"

    candidates.sort(key=lambda c: c["host_type"] != "repository")

    reasons = []
    for candidate in candidates:
        ok, reason = download_pdf(candidate["url"], dest_path)
        if ok:
            return True, None
        reasons.append(f"{candidate['host_type']}: {reason}")

    return False, "Unpaywall PDF failed: " + "; ".join(reasons)



def try_europepmc(doi, ref, dest_path):
    """Europe PMC covers biomedical literature that arXiv does not."""
    query = f'DOI:"{doi}"' if doi else f'TITLE:"{ref.get("title", "")}"'
    try:
        res = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": 1,
                    "resultType": "core"},
            headers=HEADERS, timeout=20,
        )
        res.raise_for_status()
        results = res.json().get("resultList", {}).get("result", [])
    except (requests.RequestException, ValueError) as exc:
        return False, f"Europe PMC error: {exc}"

    if not results:
        return False, "not in Europe PMC"

    hit = results[0]
    pmcid = hit.get("pmcid")
    if not pmcid or hit.get("isOpenAccess") != "Y":
        return False, "in Europe PMC but not open access"

    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
           f"{pmcid}/fullTextPdf")
    ok, reason = download_pdf(url, dest_path)
    return ok, None if ok else f"Europe PMC PDF failed: {reason}"




# Inline formatting is part of the word — "WS<sub>2</sub>" is WS2 on arXiv, not
# "WS 2" — so those tags close up. Everything else stands between two words.
_INLINE_TAG_RE = re.compile(r"</?(?:sub|sup|i|b|em|strong|italic|bold|sc|span)\b[^>]*>", re.I)
_MARKUP_RE = re.compile(r"<[^>]+>")


def arxiv_query_title(title) -> str:
    """A title arXiv's phrase index can match.

    arXiv matches ti:"..." as an exact phrase, so anything the reference
    metadata carries that the arXiv title does not makes the paper
    unfindable. Crossref hands back raw MathML for titles with an equation —
    15 of the 369 recorded fetch failures have <mml:math> in the title — and
    an embedded double quote would terminate the phrase early.
    """
    text = _INLINE_TAG_RE.sub("", str(title or ""))
    text = _MARKUP_RE.sub(" ", text)
    text = html.unescape(text).replace('"', " ")
    return " ".join(text.split())


def try_arxiv(ref, dest_path):
    """Fall back to arXiv, searching on title or explicit arxiv_id where we have one."""
    arxiv_id = ref.get("arxiv_id")
    title = arxiv_query_title(ref.get("title"))
    raw = ref.get("raw_reference")
    if not (arxiv_id or title or raw):
        return False, "nothing to search arXiv with"

    if arxiv_id:
        url = (
            "http://export.arxiv.org/api/query?id_list="
            f"{urllib.parse.quote(arxiv_id)}&max_results=1"
        )
    else:
        query = f'ti:"{title}"' if title else f'all:"{raw[:200]}"'
        url = (
            "http://export.arxiv.org/api/query?search_query="
            f"{urllib.parse.quote(query)}&max_results=1"
        )

    try:
        res = requests.get(url, headers=HEADERS, timeout=20)
        res.raise_for_status()
        root = ET.fromstring(res.content)
    except (requests.RequestException, ET.ParseError) as exc:
        return False, f"arXiv error: {exc}"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        return False, "not found on arXiv"

    # Enforce candidate verification
    cand_title = " ".join((entries[0].findtext("atom:title", "", ns)).split())
    cand_authors = [a.findtext("atom:name", "", ns) for a in entries[0].findall("atom:author", ns)]
    cand_author = cand_authors[0] if cand_authors else None
    pub = entries[0].findtext("atom:published", "", ns)
    cand_year = int(pub[:4]) if pub and pub[:4].isdigit() else None

    target_title = ref.get("title")
    target_author = ref.get("authors")[0] if ref.get("authors") else None
    target_year = ref.get("year")

    from research_assistant.shared.academic_apis import verify_candidate_match
    if target_title:
        if not verify_candidate_match(target_title, cand_title, target_author, cand_author, target_year, cand_year):
            return False, "arXiv match was a different paper (similarity verification failed)"
    elif not arxiv_id:
        return False, "arXiv match lacked title/ID verification"

    pdf_link = None
    for link in entries[0].findall("atom:link", ns):
        if link.attrib.get("title") == "pdf":
            pdf_link = link.attrib.get("href", "").replace("http://", "https://")
            break

    if not pdf_link:
        return False, "no PDF link on arXiv"
    if not pdf_link.endswith(".pdf"):
        pdf_link += ".pdf"

    ok, reason = download_pdf(pdf_link, dest_path)
    return ok, None if ok else f"arXiv PDF failed: {reason}"


def fetch_papers():
    references = _load_references()
    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)

    # One entry per distinct source, even when several papers cite it.
    unique = {}
    for ref in references:
        raw = ref.get("raw_reference") or ""
        if raw and len(raw) > MAX_CITATION_LEN:
            continue
        key = source_key(ref)
        if not key:
            continue
        unique.setdefault(key, ref)

    logger.info("%d references collapse to %d distinct sources.", len(references), len(unique))

    downloaded = _load_state(DOWNLOADED_JSON_PATH, {})
    failed = _load_state(FAILED_DOWNLOADS_PATH, {})
    if isinstance(failed, list):  # legacy shape
        failed = {}

    retryable = {
        k for k, record in failed.items()
        if _is_retryable((record or {}).get("reason") if isinstance(record, dict) else None)
    }
    remaining = [
        (k, r) for k, r in unique.items()
        if k not in downloaded and (k not in failed or k in retryable)
    ]
    if len(remaining) < len(unique):
        logger.info("Skipping %d already processed.", len(unique) - len(remaining))
    retrying = len(retryable & {k for k, _ in remaining})
    if retrying:
        logger.info("Retrying %d that previously failed for a transient reason.", retrying)

    paper_index = PaperIndex.load(PAPER_INDEX_PATH)
    standalone = not pipeline_status.get_status().get("active")
    with pipeline_status.track_stage(
        "fetch",
        "Fetching referenced papers",
        current_step=4,
        total_steps=5,
        item_total=len(remaining),
        detail=f"0/{len(remaining)} papers checked ({len(downloaded)} downloaded, {len(failed)} unavailable)",
        mark_idle_on_exit=standalone,
        last_summary=f"+{len(downloaded)} papers fetched",
    ):
        for i, (key, ref) in enumerate(remaining, start=1):
            if pipeline_status.is_cancel_requested():
                logger.info("Fetch cancelled by user via UI at paper %d/%d", i, len(remaining))
                pipeline_status.add_event(f"⏹️ Fetch stopped by user ({len(downloaded)} downloaded)")
                break

            label = (ref.get("title") or ref.get("raw_reference") or key)[:70]
            logger.info("[%d/%d] %s", i, len(remaining), label)
            pipeline_status.update_progress(
                item_current=i,
                item_total=len(remaining),
                current_item_name=label,
                detail=f"Checking [{i}/{len(remaining)}]: {label[:50]} ({len(downloaded)} downloaded)",
            )

            dest = os.path.join(PULLED_PDFS_DIR, filename_for(key))
            record_path = dest
            reasons = []
            saved = False
            doi = how = None
            provider = None

            try:
                doi, how = resolve_doi(ref)
                arxiv_id = ref.get("arxiv_id")

                corpus_hit = paper_index.find_by_ids(doi=doi, arxiv_id=arxiv_id) if (doi or arxiv_id) else None
                if corpus_hit and os.path.isfile(corpus_hit.path):
                    saved, provider = True, "corpus"
                    record_path = corpus_hit.path
                    logger.info("    already in corpus -> %s", corpus_hit.path)
                elif is_saved_pdf(dest):
                    # An earlier run saved this paper but downloaded.json does not
                    # know it (lost, reset, or older than the files). Adopt the file
                    # rather than fetching it again; the success path below records it.
                    saved, provider = True, "local_file"
                    logger.info("    already on disk -> %s", os.path.basename(dest))
                elif doi:
                    saved, reason = try_unpaywall(doi, dest)
                    if saved:
                        provider = "unpaywall"
                    else:
                        reasons.append(reason)
                        time.sleep(UNPAYWALL_SLEEP)
                else:
                    reasons.append("no DOI resolved")

                # Europe PMC carries free full text for much of the biomedical
                # literature that Unpaywall's best_oa_location does not surface, so
                # it belongs between Unpaywall and the arXiv preprint fallback.
                if not saved and doi:
                    saved, reason = try_europepmc(doi, ref, dest)
                    if saved:
                        provider = "europepmc"
                    else:
                        reasons.append(reason)

                if not saved:
                    saved, reason = try_arxiv(ref, dest)
                    if saved:
                        provider = "arxiv"
                    else:
                        reasons.append(reason)
                    time.sleep(ARXIV_RATE_LIMIT)

            except Exception as exc:
                logger.exception("Unhandled error on %s", key)
                reasons.append(str(exc))

            biblio = {}
            if not saved:
                try:
                    biblio = fetch_bibliometrics_and_abstract(doi or ref.get("title") or key)
                    oa_url = biblio.get("oa_pdf_url")
                    if oa_url:
                        logger.info("    attempting direct OA PDF from bibliometrics: %s", oa_url[:60])
                        ok, reason = download_pdf(oa_url, dest)
                        if ok:
                            downloaded[key] = DownloadedPaper(
                                key=key,
                                path=dest,
                                evidence_tier="full_text",
                                abstract=(biblio.get("abstract") or "").strip(),
                                is_retracted=bool(biblio.get("is_retracted", False)),
                                citation_count=biblio.get("citation_count") if biblio.get("citation_count") is not None else 0,
                                provider="oa_pdf_url",
                                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                title=ref.get("title") or biblio.get("title") or "",
                                raw_reference=ref.get("raw_reference") or "",
                                doi=doi or biblio.get("doi"),
                                arxiv_id=ref.get("arxiv_id"),
                                doi_source=how,
                                authoritative=is_authoritative(key),
                                cited_by=ref.get("source_file"),
                                xml_id=ref.get("xml_id"),
                            ).to_dict()
                            saved = True
                            failed.pop(key, None)
                            logger.info("    saved full text via OA URL -> %s", label[:50])
                            pipeline_status.add_event(f"⬇️ Fetched OA PDF: {label[:50]}")
                        else:
                            reasons.append(f"oa_pdf_url failed: {reason}")

                    if not saved:
                        abstract_cand = (biblio.get("abstract") or "").strip()
                        if len(abstract_cand) >= 50:
                            downloaded[key] = DownloadedPaper(
                                key=key,
                                path=None,
                                evidence_tier="abstract_only",
                                abstract=abstract_cand,
                                is_retracted=bool(biblio.get("is_retracted", False)),
                                citation_count=biblio.get("citation_count") if biblio.get("citation_count") is not None else 0,
                                provider="openalex" if biblio.get("doi") else "abstract_only",
                                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                title=ref.get("title") or biblio.get("title") or "",
                                raw_reference=ref.get("raw_reference") or "",
                                doi=doi or biblio.get("doi"),
                                arxiv_id=ref.get("arxiv_id"),
                                doi_source=how,
                                authoritative=is_authoritative(key),
                                cited_by=ref.get("source_file"),
                                xml_id=ref.get("xml_id"),
                            ).to_dict()
                            saved = True
                            failed.pop(key, None)
                            logger.info("    saved abstract-only -> %s", label[:50])
                            pipeline_status.add_event(f"📄 Abstract-only: {label[:50]}")
                except Exception as exc:
                    logger.debug("Abstract fallback check failed for %s: %s", key, exc)

            if saved:
                if key not in downloaded:
                    try:
                        biblio = fetch_bibliometrics_and_abstract(doi or ref.get("title") or key)
                    except Exception:
                        biblio = {}
                    downloaded[key] = DownloadedPaper(
                        key=key,
                        path=record_path,
                        provider=provider,
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        title=ref.get("title"),
                        raw_reference=ref.get("raw_reference"),
                        doi=doi,
                        arxiv_id=ref.get("arxiv_id"),
                        doi_source=how,
                        authoritative=is_authoritative(key),
                        cited_by=ref.get("source_file"),
                        xml_id=ref.get("xml_id"),
                        abstract=ref.get("abstract") or biblio.get("abstract") or "",
                        citation_count=ref.get("citation_count") if ref.get("citation_count") is not None else (biblio.get("citation_count") if biblio.get("citation_count") is not None else 0),
                        is_retracted=bool(ref.get("is_retracted", False) or biblio.get("is_retracted", False)),
                        evidence_tier="full_text",
                    ).to_dict()
                    failed.pop(key, None)   # a retry that worked is no longer a failure
                    logger.info("    saved -> %s", os.path.basename(dest))
                    pipeline_status.add_event(f"✅ Downloaded ({provider or 'oa'}): {label[:50]}")
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(remaining),
                    current_item_name=label,
                    detail=f"Downloaded {len(downloaded)} / Unavailable {len(failed)}",
                )
            else:
                reason_str = " | ".join(r for r in reasons if r) or "fetch failed"
                record = {
                    "key": key,
                    "doi": doi,
                    "doi_source": how,
                    "authoritative": is_authoritative(key),
                    "title": ref.get("title"),
                    "cited_by": ref.get("source_file"),
                    "xml_id": ref.get("xml_id"),
                    "raw_reference": ref.get("raw_reference"),
                    "reason": reason_str,
                }
                failed[key] = record
                logger.info("    unavailable: %s", record["reason"])
                reason_short = reason_str[:35]
                pipeline_status.add_event(f"⚠️ {label[:45]} ({reason_short})")
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(remaining),
                    current_item_name=label,
                    detail=f"Downloaded {len(downloaded)} / Unavailable {len(failed)}",
                )

            if i % 5 == 0 or i == len(remaining):
                _checkpoint(downloaded, failed)

        _checkpoint(downloaded, failed)

        if pipeline_status.is_cancel_requested():
            logger.info("Fetch aborted early due to user stop request.")
            return

        manual = [r for r in failed.values() if r.get("doi")]
        pipeline_status.add_event(
            f"🌐 Fetch complete: {len(downloaded)} downloaded, {len(failed)} unavailable"
        )
        pipeline_status.update_progress(
            item_current=len(remaining),
            item_total=len(remaining),
            detail=f"Fetch complete: {len(downloaded)} downloaded, {len(failed)} unavailable",
        )
        logger.info(
            "Done. %d downloaded, %d unavailable (%d have a DOI and can be fetched by hand).",
            len(downloaded),
            len(failed),
            len(manual),
        )



def backfill_failed_downloads(
    failed_path: str = FAILED_DOWNLOADS_PATH,
    downloaded_path: str = DOWNLOADED_JSON_PATH,
    output_dir: str = PULLED_PDFS_DIR,
) -> dict[str, int]:
    """Unblock and backfill failed downloads in failed_downloads.json.
    Attempts to resolve via direct oa_pdf_url or fall back to abstract_only.
    """
    failed = _load_state(failed_path, {})
    downloaded = _load_state(downloaded_path, {})
    if not isinstance(failed, dict) or not failed:
        return {"attempted": 0, "resolved_oa": 0, "resolved_abstract": 0, "remaining_failed": 0}

    from research_assistant.shared.academic_apis import fetch_bibliometrics_and_abstract

    os.makedirs(output_dir, exist_ok=True)
    resolved_oa = 0
    resolved_abstract = 0
    to_check = [
        (k, rec) for k, rec in list(failed.items())
        if isinstance(rec, dict) and k not in downloaded and (rec.get("doi") or rec.get("title"))
    ]

    for k, rec in to_check:
        doi = rec.get("doi")
        title = rec.get("title")
        raw = rec.get("raw_reference") or ""
        key = rec.get("key") or k
        dest_path = os.path.join(output_dir, f"{key.replace(':', '_')}.pdf")

        try:
            biblio = fetch_bibliometrics_and_abstract(doi or title or key)
            oa_url = biblio.get("oa_pdf_url")
            saved = False
            if oa_url:
                ok, _ = download_pdf(oa_url, dest_path)
                if ok:
                    downloaded[key] = DownloadedPaper(
                        key=key,
                        path=dest_path,
                        evidence_tier="full_text",
                        abstract=(biblio.get("abstract") or "").strip(),
                        is_retracted=bool(biblio.get("is_retracted", False)),
                        citation_count=biblio.get("citation_count") or 0,
                        provider="oa_pdf_url",
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        title=title or biblio.get("title") or "",
                        raw_reference=raw,
                        doi=doi or biblio.get("doi"),
                        arxiv_id=rec.get("arxiv_id"),
                        doi_source="oa_backfill",
                        authoritative=is_authoritative(key),
                        cited_by=rec.get("cited_by"),
                        xml_id=rec.get("xml_id"),
                    ).to_dict()
                    failed.pop(k, None)
                    resolved_oa += 1
                    saved = True

            if not saved:
                abstract = (biblio.get("abstract") or "").strip()
                if len(abstract) >= 50:
                    downloaded[key] = DownloadedPaper(
                        key=key,
                        path=None,
                        evidence_tier="abstract_only",
                        abstract=abstract,
                        is_retracted=bool(biblio.get("is_retracted", False)),
                        citation_count=biblio.get("citation_count") or 0,
                        provider="openalex" if biblio.get("doi") else "abstract_only",
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        title=title or biblio.get("title") or "",
                        raw_reference=raw,
                        doi=doi or biblio.get("doi"),
                        arxiv_id=rec.get("arxiv_id"),
                        doi_source="abstract_backfill",
                        authoritative=is_authoritative(key),
                        cited_by=rec.get("cited_by"),
                        xml_id=rec.get("xml_id"),
                    ).to_dict()
                    failed.pop(k, None)
                    resolved_abstract += 1
        except Exception as exc:
            logger.debug("Failed backfill for %s: %s", k, exc)

    _checkpoint(downloaded, failed)
    return {
        "attempted": len(to_check),
        "resolved_oa": resolved_oa,
        "resolved_abstract": resolved_abstract,
        "remaining_failed": len(failed),
    }


fetch_missing_citations = fetch_papers


if __name__ == "__main__":
    fetch_papers()
