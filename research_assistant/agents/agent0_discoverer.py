"""
Agent 0 — Seed paper discoverer.

Turns a free-text research idea into the first PDF the rest of the pipeline
needs. It searches a scholarly index (OpenAlex, then Semantic Scholar — see
SEARCH_PROVIDERS), whose results come back ranked by relevance, picks the top
result that actually has an open-access PDF, downloads it into RAW_DIR under a
source_key-derived name, and records the choice in seed_papers.json.

Nothing downstream knows Agent 0 exists: it just leaves a PDF in RAW_DIR, which
is exactly what Agent 1 already reads. The returned path also lets an
orchestrator hand the seed straight to shared.ingestion.ingest_pdfs() so the
seed paper itself is indexed, not only the papers it cites.

    python agent0_discoverer.py --query "topological protection in disordered wires"
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Optional
import uuid

import requests

from research_assistant.shared.pdf_metadata import (
    _clean_doi,
    _clean_title,
    _decode_pdf_title,
    _inspect_pdf,
    _is_pdf_content,
    _ARXIV_RE_BYTES,
    _CJK_RE,
    _DOI_RE,
    _HEADER_JUNK_RE,
    _PDF_TITLE_RE,
    _TITLE_BANNER_WORDS,
    _TITLE_FILE_RE,
    _TITLE_ID_RE,
    _XMP_DOI_RE,
    _XMP_TITLE_FALLBACK_RE,
    _XMP_TITLE_RE,
    clean_doi,
    clean_title,
    decode_pdf_title,
    inspect_pdf_metadata,
    is_pdf_content,
)

from research_assistant.config import (
    RAW_DIR,
    SEED_PAPERS_PATH,
    SEARCH_PROVIDERS,
    SEARCH_LIMIT,
    OPENALEX_URL,
    OPENALEX_MAILTO,
    S2_API_KEY,
    S2_SEARCH_URL,
    CHAT_OLLAMA_OPTIONS,
    DATA_DIR,
)
from research_assistant.prompts import (
    DISCOVERY_CONTEXT_SYSTEM,
    DISCOVERY_CONTEXT_USER,
    DISCOVERY_REFINE_QUERY_ITER2_SYSTEM,
    DISCOVERY_REFINE_QUERY_ITER2_USER,
    DISCOVERY_REFINE_QUERY_ITER3_SYSTEM,
    DISCOVERY_REFINE_QUERY_ITER3_USER,
    DISCOVERY_EVAL_CANDIDATES_SYSTEM,
    DISCOVERY_EVAL_CANDIDATES_USER,
)
from research_assistant.schemas import SeedPaper
from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.fetch import HEADERS, download_pdf, filename_for, _is_pdf
import research_assistant.shared.llm as llm
from research_assistant.shared.log import get_logger
from research_assistant.shared.source_key import source_key, normalise_doi, _fold_title, _hash


def chat(*args, **kwargs):
    return llm.chat(*args, **kwargs)

logger = get_logger("agent0")
DISCOVERY_TRACES_PATH = os.path.join(DATA_DIR, "discovery_traces.json")


# ─── Provider searches ───────────────────────────────────────────────────────
# Each returns a list of normalised candidate dicts, most relevant first:
#   {title, doi, arxiv_id, pmid, authors: [str], year, pdf_url}
# pdf_url may be None; pick_best() skips those.


def _search_openalex(query, limit):
    res = requests.get(
        OPENALEX_URL,
        params={"search": query, "per_page": limit, "mailto": OPENALEX_MAILTO},
        headers=HEADERS,
        timeout=20,
    )
    res.raise_for_status()
    out = []
    for w in res.json().get("results") or []:
        loc = w.get("best_oa_location") or w.get("primary_location") or {}
        ids = w.get("ids") or {}
        out.append({
            "title": w.get("display_name") or w.get("title"),
            "doi": w.get("doi"),
            "arxiv_id": None,
            "pmid": (ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
            "authors": [
                a["author"]["display_name"]
                for a in w.get("authorships") or []
                if a.get("author", {}).get("display_name")
            ],
            "year": w.get("publication_year"),
            "pdf_url": loc.get("pdf_url") or (w.get("open_access") or {}).get("oa_url"),
        })
    return out


def _search_semanticscholar(query, limit, retries=4):
    headers = dict(HEADERS)
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    fields = "title,year,authors,externalIds,openAccessPdf"
    for attempt in range(retries):
        res = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": limit, "fields": fields},
            headers=headers,
            timeout=20,
        )
        # The keyless pool 429s constantly; back off on those, but not on any
        # other HTTP error.
        if res.status_code == 429 and attempt < retries - 1:
            wait = float(res.headers.get("Retry-After", 2 ** (attempt + 1)))
            logger.warning("Semantic Scholar rate limited; retrying in %.0fs…", wait)
            time.sleep(wait)
            continue
        res.raise_for_status()
        out = []
        for p in res.json().get("data") or []:
            ext = p.get("externalIds") or {}
            out.append({
                "title": p.get("title"),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
                "pmid": str(ext["PubMed"]) if ext.get("PubMed") else None,
                "authors": [a["name"] for a in p.get("authors") or [] if a.get("name")],
                "year": p.get("year"),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
            })
        return out
    return []


def _search_arxiv(query, limit):
    import xml.etree.ElementTree as ET

    res = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": limit,
                "sortBy": "relevance"},
        headers=HEADERS,
        timeout=20,
    )
    res.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(res.content)

    out = []
    for e in root.findall("a:entry", ns):
        raw_id = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
        arxiv_id = raw_id.split("v")[0] or None

        pdf = None
        for link in e.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf = link.get("href", "").replace("http://", "https://")
        if pdf and not pdf.endswith(".pdf"):
            pdf += ".pdf"

        out.append({
            "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
            "doi": None,
            "arxiv_id": arxiv_id,
            "pmid": None,
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "year": (e.findtext("a:published", "", ns) or "")[:4] or None,
            "pdf_url": pdf,
        })
    return out


_PROVIDERS = {
    "openalex": _search_openalex,
    "arxiv": _search_arxiv,
    "semanticscholar": _search_semanticscholar,
}


# ─── Discovery Traces Store ──────────────────────────────────────────────────


def _load_discovery_traces() -> dict[str, Any]:
    if not os.path.exists(DISCOVERY_TRACES_PATH):
        return {}
    try:
        with open(DISCOVERY_TRACES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_discovery_trace(query: str, trace: dict[str, Any]) -> None:
    traces = _load_discovery_traces()
    traces[query] = trace
    try:
        atomic_write_json(DISCOVERY_TRACES_PATH, traces, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Could not persist discovery trace to %s: %s", DISCOVERY_TRACES_PATH, exc)


def get_discovery_trace(query: str) -> dict[str, Any] | None:
    return _load_discovery_traces().get(query)


# ─── Context Formulation (Requirement R1) ───────────────────────────────────

DOMAINS_TAXONOMY: dict[str, list[str]] = {
    "Condensed Matter Physics": [
        "topological", "nanowire", "superconduct", "majorana", "graphene",
        "semiconductor", "ferromagnet", "lattice", "phonon", "spintronic",
        "correlated", "hall effect", "hubbard", "2d material", "spin glass",
        "quantum dot", "insulator", "bandgap", "andreev", "kitaev",
    ],
    "Quantum Information & Computing": [
        "qubit", "quantum computing", "quantum circuit", "entanglement",
        "decoherence", "quantum error", "teleportation", "quantum memory",
        "quantum algorithm", "quantum simulation", "bell inequality",
        "transmon", "rydberg", "trapped ion", "cat qubit", "zk-snark", "cryptograph",
    ],
    "High Energy Physics & Cosmology": [
        "dark matter", "neutrino", "cosmology", "gravitational wave",
        "black hole", "string theory", "higgs", "quantum gravity",
        "standard model", "collider", "general relativity", "cern",
        "inflation", "baryon",
    ],
    "Biophysics & Molecular Biology": [
        "protein", "dna", "rna", "folding", "lipid", "membrane",
        "molecular dynamic", "enzyme", "biomolecule", "ribosome",
        "macromolecule", "allostery", "cryo-em", "crispr", "perturb-seq",
    ],
    "Machine Learning & Artificial Intelligence": [
        "machine learning", "neural network", "deep learning", "transformer",
        "reinforcement learning", "diffusion model", "llm", "foundation model",
        "graph neural", "convolutional", "representation learning", "fourier neural",
    ],
    "Optics & Photonics": [
        "laser", "photon", "optics", "metamaterial", "plasmon", "waveguide",
        "interferometer", "quantum optics", "cavity", "diffraction",
        "nonlinear optics", "polariton",
    ],
    "Materials Science & Nanotechnology": [
        "perovskite", "thin film", "alloy", "synthesis", "nanoparticle",
        "nanotube", "polymer", "crystallography", "defect", "photovoltaic",
        "lithography", "heterostructure",
    ],
}
DEFAULT_DOMAIN = "General Physics & Physical Sciences"

_HEURISTIC_STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while
who who's whom why why's with won't would wouldn't you you'd you'll you're you've
investigation study analysis approach method research paper based using towards
""".split())

VALID_PAPER_TYPES = {
    "foundational_theory",
    "experimental_breakthrough",
    "comprehensive_review",
    "methodological_benchmark",
    "survey",
    "review",
    "breakthrough",
    "benchmark",
}


def _extract_json_text(text: str) -> str:
    """Extract JSON string from raw model output, handling fences and surrounding prose."""
    if not text or not text.strip():
        return ""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        if last_bracket != -1 and last_bracket > first_bracket:
            return cleaned[first_bracket : last_bracket + 1].strip()
    elif first_brace != -1:
        if last_brace != -1 and last_brace > first_brace:
            return cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def _heuristic_context(query: str) -> dict[str, Any]:
    """Deterministically synthesize a complete ContextProfile from query."""
    raw = (query or "").strip()
    clean = re.sub(r"\s+", " ", re.sub(r"[^\w\s\-]", " ", raw)).strip()
    if not clean:
        clean = "Foundational Scientific Research"

    query_lower = clean.lower()

    # 1. Primary domain classification
    domain_scores = {
        domain: sum(1 for kw in kws if kw in query_lower)
        for domain, kws in DOMAINS_TAXONOMY.items()
    }
    best_domain = max(domain_scores.items(), key=lambda x: x[1])
    primary_domain = best_domain[0] if best_domain[1] > 0 else DEFAULT_DOMAIN

    # 2. Extract content tokens and bigrams
    words = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", clean)
    content_words = [w for w in words if w.lower() not in _HEURISTIC_STOPWORDS and len(w) >= 3]

    tokens = clean.split()
    bigrams = [
        f"{tokens[i]} {tokens[i+1]}"
        for i in range(len(tokens) - 1)
        if tokens[i].lower() not in _HEURISTIC_STOPWORDS
        and tokens[i+1].lower() not in _HEURISTIC_STOPWORDS
        and len(tokens[i]) >= 3 and len(tokens[i+1]) >= 3
    ]

    # 3. Key concepts
    key_concepts: list[str] = []
    for b in bigrams:
        if b.lower() not in [k.lower() for k in key_concepts]:
            key_concepts.append(b)
    for w in content_words:
        if w.lower() not in [k.lower() for k in key_concepts]:
            key_concepts.append(w)
    if not key_concepts:
        key_concepts = [clean, "theoretical foundations", "experimental methodology"]
    key_concepts = key_concepts[:6]
    while len(key_concepts) < 3:
        key_concepts.append(f"{clean} principles")

    # 4. Subtopics
    subtopics = [b.title() for b in bigrams]
    for w in content_words:
        subtopics.append(f"{w.title()} Dynamics")
    subtopics = list(dict.fromkeys(subtopics))[:4]
    if len(subtopics) < 2:
        subtopics.extend([f"{clean.title()} Theory", f"{clean.title()} Applications"])
    subtopics = subtopics[:4]

    # 5. Synonyms
    synonyms = [
        f"{clean} review",
        f"{clean} benchmark",
        f"advances in {clean}",
    ]
    if len(content_words) >= 2:
        synonyms.append(f"{content_words[1]} in {content_words[0]}")
    synonyms = list(dict.fromkeys(synonyms))[:6]
    while len(synonyms) < 3:
        synonyms.append(f"{clean} analysis")

    # 6. Ideal paper type
    if any(w in query_lower for w in ("review", "survey", "overview", "progress")):
        ideal_paper_type = "comprehensive_review"
    elif any(w in query_lower for w in ("experiment", "measurement", "device", "observation", "synthesis", "fabrication", "breakthrough")):
        ideal_paper_type = "experimental_breakthrough"
    elif any(w in query_lower for w in ("benchmark", "protocol", "algorithm", "technique", "method", "dataset")):
        ideal_paper_type = "methodological_benchmark"
    else:
        ideal_paper_type = "foundational_theory"

    core_problem = clean
    seed_criteria = (
        f"Seminal theoretical derivation, pivotal experimental breakthrough, or canonical review paper "
        f"establishing foundational principles and benchmarks for {clean}."
    )

    return {
        "core_problem": core_problem,
        "primary_domain": primary_domain,
        "subtopics": subtopics,
        "key_concepts": key_concepts,
        "synonyms": synonyms,
        "seed_criteria": seed_criteria,
        "ideal_paper_type": ideal_paper_type,
    }


def _sanitize_and_merge_context(raw_dict: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_dict, dict):
        return fallback

    norm = {
        str(k).strip().lower().replace("-", "_").replace(" ", "_"): v
        for k, v in raw_dict.items()
    }

    def _as_str(val, default):
        if val is None:
            return default
        s = str(val).strip()
        return s if s else default

    def _as_str_list(val, min_items, default):
        items = []
        if isinstance(val, list):
            items = [str(x).strip() for x in val if str(x).strip()]
        elif isinstance(val, str) and val.strip():
            items = [s.strip() for s in val.split(",") if s.strip()]
        return items if len(items) >= min_items else default

    core_problem = _as_str(norm.get("core_problem"), fallback["core_problem"])

    pd = norm.get("primary_domain") or norm.get("domain") or norm.get("field")
    primary_domain = _as_str(pd, fallback["primary_domain"])

    st = norm.get("subtopics") or norm.get("sub_topics")
    subtopics = _as_str_list(st, 1, fallback["subtopics"])[:4]

    kc = norm.get("key_concepts") or norm.get("concepts") or norm.get("keywords")
    key_concepts = _as_str_list(kc, 1, fallback["key_concepts"])[:6]

    syn = norm.get("synonyms") or norm.get("alternative_terms")
    if isinstance(syn, dict) and syn:
        synonyms = syn
    elif isinstance(syn, list) and syn:
        synonyms = [str(x).strip() for x in syn if str(x).strip()]
    elif isinstance(syn, str) and syn.strip():
        synonyms = [s.strip() for s in syn.split(",") if s.strip()]
    else:
        synonyms = fallback["synonyms"]

    sc = norm.get("seed_criteria") or norm.get("criteria")
    seed_criteria = _as_str(sc, fallback["seed_criteria"])

    ipt = norm.get("ideal_paper_type") or norm.get("paper_type")
    ipt_str = str(ipt).strip().lower() if ipt else ""
    ideal_paper_type = ipt_str if ipt_str in VALID_PAPER_TYPES else fallback["ideal_paper_type"]

    return {
        "core_problem": core_problem,
        "primary_domain": primary_domain,
        "subtopics": subtopics,
        "key_concepts": key_concepts,
        "synonyms": synonyms,
        "seed_criteria": seed_criteria,
        "ideal_paper_type": ideal_paper_type,
    }


def formulate_context(query: str) -> dict[str, Any]:
    """Formulate a structured scientific context profile from a research query.

    Analyzes the query using research_assistant.shared.llm.chat with Ollama options
    (num_ctx: 4096). If the LLM is unavailable, times out, or produces unparseable
    output, seamlessly falls back to a deterministic keyword-based heuristic.
    """
    clean_query = (query or "").strip()
    fallback = _heuristic_context(clean_query)

    try:
        pipeline_status.update_progress(
            detail="Formulating scientific context profile...",
            current_item_name=clean_query[:50],
        )
    except Exception:
        pass

    try:
        opts = dict(CHAT_OLLAMA_OPTIONS) if CHAT_OLLAMA_OPTIONS else {}
        opts["num_ctx"] = 4096
        reply = chat(
            messages=[
                {"role": "system", "content": DISCOVERY_CONTEXT_SYSTEM},
                {"role": "user", "content": DISCOVERY_CONTEXT_USER.format(query=clean_query)},
            ],
            temperature=0.1,
            options=opts,
        )
        json_text = _extract_json_text(reply.content)
        parsed = json.loads(json_text)
        profile = _sanitize_and_merge_context(parsed, fallback)
        try:
            pipeline_status.add_event(
                f"🧠 Context formulated: {profile['primary_domain']} | Type: {profile['ideal_paper_type']}"
            )
        except Exception:
            pass
        return profile
    except Exception as exc:
        logger.warning(
            "Context formulation LLM call failed (%s); using deterministic heuristic fallback.",
            exc,
        )
        try:
            pipeline_status.add_event(
                f"🧠 Context formulated (fallback): {fallback['primary_domain']}"
            )
        except Exception:
            pass
        return fallback


# ─── Candidate Pool & Deduplication (Requirement R2) ─────────────────────────

class CandidatePool:
    """Accumulates and deduplicates academic paper candidates across providers and iterations."""

    def __init__(self):
        self._candidates: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}

    def _make_keys(self, cand: dict[str, Any]) -> tuple[str | None, list[str]]:
        keys = []
        # 1. DOI
        doi = cand.get("doi")
        if doi:
            clean_doi = normalise_doi(doi) or str(doi).strip().lower()
            if clean_doi:
                keys.append(f"doi:{clean_doi}")

        # 2. arXiv ID
        arx = cand.get("arxiv_id")
        if arx:
            clean_arx = re.sub(r"(?i)^arxiv:\s*", "", str(arx).strip()).split("v")[0].lower()
            if clean_arx:
                keys.append(f"arxiv:{clean_arx}")

        # 3. PMID
        pmid = cand.get("pmid")
        if pmid:
            clean_pmid = str(pmid).strip()
            if clean_pmid:
                keys.append(f"pmid:{clean_pmid}")

        # 4. Title Hash
        title = cand.get("title")
        if title:
            authors = cand.get("authors") or []
            auth = authors[0].split()[-1] if authors else ""
            year = str(cand.get("year") or "")
            folded = _fold_title(title)
            if folded:
                keys.append(_hash("title", f"{folded}|{auth.lower()}|{year}"))

        if not keys:
            sk = source_key(cand)
            if sk:
                keys.append(sk)

        primary = keys[0] if keys else None
        return primary, keys

    def add(self, cand: dict[str, Any], provider: str = "unknown", iteration: int = 1) -> bool:
        primary, keys = self._make_keys(cand)
        if not primary:
            return False

        matched_canonical = None
        for k in keys:
            if k in self._aliases:
                matched_canonical = self._aliases[k]
                break

        if matched_canonical and matched_canonical in self._candidates:
            rec = self._candidates[matched_canonical]
            if cand.get("doi") and not rec.get("doi"):
                rec["doi"] = cand["doi"]
            if cand.get("arxiv_id") and not rec.get("arxiv_id"):
                rec["arxiv_id"] = cand["arxiv_id"]
            if cand.get("pmid") and not rec.get("pmid"):
                rec["pmid"] = cand["pmid"]
            if cand.get("pdf_url"):
                if not rec.get("pdf_url") or ("arxiv.org/pdf" in cand["pdf_url"] and "arxiv.org/pdf" not in rec.get("pdf_url", "")):
                    rec["pdf_url"] = cand["pdf_url"]
                rec.setdefault("all_pdf_urls", [])
                if cand["pdf_url"] not in rec["all_pdf_urls"]:
                    rec["all_pdf_urls"].append(cand["pdf_url"])
            if cand.get("authors") and len(cand["authors"]) > len(rec.get("authors", [])):
                rec["authors"] = cand["authors"]
            if cand.get("year") and not rec.get("year"):
                rec["year"] = cand["year"]

            rec.setdefault("providers_seen", set()).add(provider)
            rec.setdefault("iterations_seen", set()).add(iteration)
            rec["hit_count"] = rec.get("hit_count", 1) + 1
            rec["occurrence_count"] = rec["hit_count"]

            for k in keys:
                self._aliases[k] = matched_canonical
            return False

        rec = dict(cand)
        rec["canonical_key"] = primary
        rec["providers_seen"] = {provider}
        rec["iterations_seen"] = {iteration}
        rec["hit_count"] = 1
        rec["occurrence_count"] = 1
        rec["all_pdf_urls"] = [cand["pdf_url"]] if cand.get("pdf_url") else []

        self._candidates[primary] = rec
        for k in keys:
            self._aliases[k] = primary
        return True

    def to_list(self) -> list[dict[str, Any]]:
        out = []
        for c in self._candidates.values():
            rec = dict(c)
            rec["providers_seen"] = sorted(list(rec.get("providers_seen", [])))
            rec["iterations_seen"] = sorted(list(rec.get("iterations_seen", [])))
            out.append(rec)
        return out


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = CandidatePool()
    for c in candidates:
        pool.add(c, provider=c.get("search_provider", "unknown"), iteration=1)
    return pool.to_list()


# ─── Adaptive 3-Stage Iterative Search (Requirement R2) ─────────────────────

def _construct_initial_query(context_profile: dict[str, Any]) -> str:
    core = context_profile.get("core_problem") or ""
    clean_core = re.sub(r"[^\w\s\-]", " ", core)
    words = [w for w in clean_core.split() if w.lower() not in _HEURISTIC_STOPWORDS and len(w) >= 3]
    if words:
        return " ".join(words[:5])
    concepts = context_profile.get("key_concepts") or []
    if concepts:
        c_words = [w for w in " ".join(concepts).split() if w.lower() not in _HEURISTIC_STOPWORDS and len(w) >= 3]
        return " ".join(c_words[:5])
    domain = context_profile.get("primary_domain") or "scientific research"
    return domain


def _fallback_query_iter2(context_profile: dict[str, Any]) -> str:
    subtopics = context_profile.get("subtopics") or []
    synonyms = context_profile.get("synonyms") or []
    if isinstance(synonyms, dict):
        syn_list = [item for sub in synonyms.values() for item in sub]
    else:
        syn_list = synonyms
    if subtopics and syn_list:
        cand = f"{subtopics[0]} {syn_list[0]}"
    elif subtopics:
        cand = f"{subtopics[0]} mechanism"
    else:
        cand = context_profile.get("primary_domain", "physics") + " review"
    clean = re.sub(r"[^\w\s\-]", " ", cand)
    return " ".join(clean.split()[:5])


def _fallback_query_iter3(context_profile: dict[str, Any]) -> str:
    ideal = (context_profile.get("ideal_paper_type") or "").lower()
    concepts = context_profile.get("key_concepts") or []
    core = concepts[0] if concepts else (context_profile.get("primary_domain") or "research")
    if "survey" in ideal or "review" in ideal:
        mod = "survey review progress"
    elif "benchmark" in ideal or "dataset" in ideal:
        mod = "benchmark comparative analysis"
    elif "experimental" in ideal or "breakthrough" in ideal:
        mod = "experimental demonstration breakthrough"
    else:
        mod = "foundational theoretical principles"
    cand = f"{core} {mod}"
    clean = re.sub(r"[^\w\s\-]", " ", cand)
    return " ".join(clean.split()[:5])


def _parse_refined_query(content: str, fallback_query: str) -> tuple[str, str, str]:
    if not content or not content.strip():
        return fallback_query, "", "LLM empty response; used fallback."
    cleaned = _extract_json_text(content)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            q = data.get("refined_query") or data.get("query")
            gap = data.get("gap_analysis") or data.get("pool_evaluation") or ""
            if q and str(q).strip():
                clean_q = re.sub(r"[^\w\s\-]", " ", str(q))
                clean_q = " ".join(clean_q.split()[:6])
                return clean_q, str(gap), "LLM refined query."
    except Exception:
        pass
    plain = content.strip().splitlines()[0]
    plain = re.sub(r'["\']', '', plain)
    plain = re.sub(r"[^\w\s\-]", " ", plain)
    clean_plain = " ".join(plain.split()[:6]).strip()
    if clean_plain and len(clean_plain) > 2:
        return clean_plain, "", "LLM direct string query."
    return fallback_query, "", "LLM parse failed; used fallback."


def iterative_search(
    context_profile: dict[str, Any],
    providers: list[str] | None = None,
    limit: int = 5,
    on_iteration: Optional[Callable[[int, str, int], None]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_providers = [p.strip() for p in (providers or SEARCH_PROVIDERS)]
    pool = CandidatePool()
    t_start = datetime.now(timezone.utc)
    trace_iterations = []

    def _run_iteration(iter_idx: int, query: str, stage_name: str, query_source: str, rationale: str, gap_analysis: str | None = None):
        logger.info("Search Iteration %d/3 (%s): %s", iter_idx, stage_name, query)
        try:
            pipeline_status.update_progress(
                item_current=iter_idx,
                item_total=3,
                detail=f"Search Iteration {iter_idx}/3: '{query[:40]}'",
            )
        except Exception:
            pass

        provider_results = {}
        total_raw = 0
        pool_before = len(pool.to_list())

        for pname in active_providers:
            p_func = _PROVIDERS.get(pname)
            if not p_func:
                logger.warning("Provider %r not found in _PROVIDERS; skipping.", pname)
                continue

            t_prov_start = time.time()
            try:
                results = p_func(query, limit)
                latency = round((time.time() - t_prov_start) * 1000, 1)
                provider_results[pname] = {
                    "hits": len(results),
                    "status": "ok",
                    "latency_ms": latency,
                    "error_msg": None,
                }
                total_raw += len(results)
                for cand in results:
                    pool.add(cand, provider=pname, iteration=iter_idx)
            except requests.RequestException as exc:
                latency = round((time.time() - t_prov_start) * 1000, 1)
                err_str = str(exc)
                logger.warning("Provider %s failed for query %r: %s", pname, query, err_str)
                provider_results[pname] = {
                    "hits": 0,
                    "status": "error",
                    "latency_ms": latency,
                    "error_msg": err_str,
                }
            except Exception as exc:
                latency = round((time.time() - t_prov_start) * 1000, 1)
                logger.warning("Provider %s raised unexpected error: %s", pname, exc)
                provider_results[pname] = {
                    "hits": 0,
                    "status": "error",
                    "latency_ms": latency,
                    "error_msg": str(exc),
                }

        pool_after = len(pool.to_list())
        new_unique = pool_after - pool_before

        try:
            pipeline_status.add_event(
                f"🔍 Iteration {iter_idx}/3: '{query[:40]}' ({new_unique} new candidates)"
            )
        except Exception:
            pass

        iter_trace = {
            "iteration": iter_idx,
            "stage_name": stage_name,
            "query": query,
            "query_source": query_source,
            "llm_gap_analysis": gap_analysis,
            "llm_rationale": rationale,
            "providers_queried": active_providers,
            "provider_results": provider_results,
            "total_raw_hits": total_raw,
            "new_unique_candidates": new_unique,
            "pool_size_after": pool_after,
        }
        trace_iterations.append(iter_trace)
        if on_iteration:
            try:
                on_iteration(iter_idx, query, new_unique)
            except Exception:
                pass
        return iter_trace

    # ── Iteration 1: Core Scope ──
    q1 = _construct_initial_query(context_profile)
    _run_iteration(
        iter_idx=1,
        query=q1,
        stage_name="core_conceptual_search",
        query_source="core_problem_and_concepts",
        rationale="Initial baseline query targeting core scientific mechanisms.",
    )

    # ── Iteration 2: Subtopics & Gap Expansion ──
    cand_summaries_1 = []
    for c in pool.to_list()[:8]:
        auth = ", ".join(c.get("authors", [])[:2]) or "Unknown"
        cand_summaries_1.append(f"- \"{c.get('title')}\" ({c.get('year') or 'n/d'}, {auth})")
    cand_text_1 = "\n".join(cand_summaries_1) if cand_summaries_1 else "No candidates found in iteration 1."

    synonyms_val = context_profile.get("synonyms") or []
    if isinstance(synonyms_val, dict):
        syn_str = ", ".join([f"{k}: {', '.join(v)}" for k, v in synonyms_val.items()])
    elif isinstance(synonyms_val, list):
        syn_str = ", ".join(synonyms_val)
    else:
        syn_str = str(synonyms_val)

    user_prompt_2 = DISCOVERY_REFINE_QUERY_ITER2_USER.format(
        core_problem=context_profile.get("core_problem", ""),
        primary_domain=context_profile.get("primary_domain", ""),
        subtopics=", ".join(context_profile.get("subtopics", [])),
        key_concepts=", ".join(context_profile.get("key_concepts", [])),
        synonyms=syn_str,
        prior_query=q1,
        candidate_count=len(pool.to_list()),
        candidate_summaries=cand_text_1,
    )

    q2_fallback = _fallback_query_iter2(context_profile)
    q2 = q2_fallback
    gap_2 = None
    rationale_2 = "Fallback query targeting subtopics and synonyms."
    source_2 = "rule_based_fallback"

    try:
        opts = dict(CHAT_OLLAMA_OPTIONS) if CHAT_OLLAMA_OPTIONS else {}
        opts["num_ctx"] = 4096
        reply2 = chat(
            messages=[
                {"role": "system", "content": DISCOVERY_REFINE_QUERY_ITER2_SYSTEM},
                {"role": "user", "content": user_prompt_2},
            ],
            temperature=0.1,
            options=opts,
        )
        parsed_q2, gap_2, rationale_2 = _parse_refined_query(reply2.content, q2_fallback)
        if parsed_q2 and parsed_q2 != q2_fallback:
            q2 = parsed_q2
            source_2 = "llm_refinement"
    except Exception as exc:
        logger.warning("Iteration 2 LLM refinement failed (%s); using fallback query.", exc)

    _run_iteration(
        iter_idx=2,
        query=q2,
        stage_name="subtopic_and_gap_expansion",
        query_source=source_2,
        rationale=rationale_2,
        gap_analysis=gap_2,
    )

    # ── Iteration 3: Foundational Targeted Sweep ──
    cand_summaries_2 = []
    for c in pool.to_list()[:10]:
        auth = ", ".join(c.get("authors", [])[:2]) or "Unknown"
        cand_summaries_2.append(f"- \"{c.get('title')}\" ({c.get('year') or 'n/d'}, {auth})")
    cand_text_2 = "\n".join(cand_summaries_2) if cand_summaries_2 else "No candidates found in iterations 1 and 2."

    user_prompt_3 = DISCOVERY_REFINE_QUERY_ITER3_USER.format(
        core_problem=context_profile.get("core_problem", ""),
        ideal_paper_type=context_profile.get("ideal_paper_type", "foundational_theory"),
        seed_criteria=context_profile.get("seed_criteria", ""),
        candidate_count=len(pool.to_list()),
        candidate_summaries=cand_text_2,
    )

    q3_fallback = _fallback_query_iter3(context_profile)
    q3 = q3_fallback
    eval_3 = None
    rationale_3 = "Fallback query targeting foundational benchmark literature."
    source_3 = "rule_based_fallback"

    try:
        opts = dict(CHAT_OLLAMA_OPTIONS) if CHAT_OLLAMA_OPTIONS else {}
        opts["num_ctx"] = 4096
        reply3 = chat(
            messages=[
                {"role": "system", "content": DISCOVERY_REFINE_QUERY_ITER3_SYSTEM},
                {"role": "user", "content": user_prompt_3},
            ],
            temperature=0.1,
            options=opts,
        )
        parsed_q3, eval_3, rationale_3 = _parse_refined_query(reply3.content, q3_fallback)
        if parsed_q3 and parsed_q3 != q3_fallback:
            q3 = parsed_q3
            source_3 = "llm_refinement"
    except Exception as exc:
        logger.warning("Iteration 3 LLM refinement failed (%s); using fallback query.", exc)

    _run_iteration(
        iter_idx=3,
        query=q3,
        stage_name="foundational_targeted_sweep",
        query_source=source_3,
        rationale=rationale_3,
        gap_analysis=eval_3,
    )

    t_end = datetime.now(timezone.utc)
    candidates = pool.to_list()

    prov_dist = {}
    for it in trace_iterations:
        for p, res in it.get("provider_results", {}).items():
            prov_dist[p] = prov_dist.get(p, 0) + res.get("hits", 0)

    errors = []
    for it in trace_iterations:
        for p, res in it.get("provider_results", {}).items():
            if res.get("error_msg"):
                errors.append(f"Iteration {it['iteration']} {p}: {res['error_msg']}")

    search_trace = {
        "search_id": f"trace_{t_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "timestamp_start": t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_end": t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_duration_sec": round((t_end - t_start).total_seconds(), 2),
        "context_summary": {
            "core_problem": context_profile.get("core_problem"),
            "primary_domain": context_profile.get("primary_domain"),
            "subtopics": context_profile.get("subtopics", []),
            "ideal_paper_type": context_profile.get("ideal_paper_type"),
        },
        "all_candidates": candidates,
        "iterations": trace_iterations,
        "summary": {
            "total_iterations": 3,
            "total_raw_hits": sum(it["total_raw_hits"] for it in trace_iterations),
            "total_unique_candidates": len(candidates),
            "oa_pdf_available_count": sum(1 for c in candidates if c.get("pdf_url")),
            "provider_hit_distribution": prov_dist,
            "errors_encountered": errors,
        },
    }

    return candidates, search_trace


# ─── Candidate Evaluation & Downloader Ladder (Requirement R3) ───────────────

def filter_open_access(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize and filter candidates with verifiable open-access PDF URLs."""
    valid = []
    seen_keys = set()

    for c in candidates:
        cand = dict(c)
        arxiv_id = cand.get("arxiv_id")
        pdf_url = cand.get("pdf_url")
        if arxiv_id and (not pdf_url or "/abs/" in pdf_url):
            cand["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_url = cand["pdf_url"]

        if not pdf_url or not isinstance(pdf_url, str):
            continue

        pdf_url = pdf_url.strip()
        if not pdf_url:
            continue
        if not (pdf_url.startswith("http://") or pdf_url.startswith("https://")):
            continue

        key = source_key(cand)
        if not key or key in seen_keys:
            continue

        seen_keys.add(key)
        valid.append(cand)

    return valid


def _heuristic_score_candidate(cand: dict[str, Any], context_profile: dict[str, Any]) -> tuple[float, str]:
    title = (cand.get("title") or "").lower()

    key_concepts = context_profile.get("key_concepts") or []
    subtopics = context_profile.get("subtopics") or []
    synonyms = context_profile.get("synonyms") or []
    if isinstance(synonyms, dict):
        syn_list = [item for sub in synonyms.values() for item in sub]
    else:
        syn_list = synonyms
    core_problem = context_profile.get("core_problem") or ""

    matches = 0.0
    for kc in key_concepts:
        if kc.lower() in title:
            matches += 2.0
    for st in subtopics:
        if st.lower() in title:
            matches += 1.5
    for syn in syn_list:
        if syn.lower() in title:
            matches += 1.0

    core_words = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", core_problem)
        if w.lower() not in _HEURISTIC_STOPWORDS
    ]
    title_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", title))
    overlap = len(title_words.intersection(core_words))
    matches += min(3.0, overlap * 0.75)

    s_overlap = min(1.0, matches / 4.0)

    survey_kws = (
        "survey", "review", "benchmark", "overview", "progress in",
        "advances in", "tutorial", "roadmap", "foundation", "foundations",
        "perspective", "introduction to",
    )
    is_survey = any(kw in title for kw in survey_kws)
    s_survey = 1.0 if is_survey else 0.0

    try:
        year = int(cand.get("year") or 0)
    except Exception:
        year = 0
    if year >= 2023:
        s_recency = 1.0
    elif year >= 2018:
        s_recency = 0.8
    elif year >= 2012:
        s_recency = 0.6
    elif year > 0:
        s_recency = 0.4
    else:
        s_recency = 0.5

    s_prov = 0.0
    if cand.get("arxiv_id"):
        s_prov += 0.5
    if cand.get("hit_count", 1) > 1 or cand.get("occurrence_count", 1) > 1:
        s_prov += 0.5
    s_prov = min(1.0, s_prov)

    total_score = round(10.0 * (0.45 * s_overlap + 0.25 * s_survey + 0.15 * s_recency + 0.15 * s_prov), 2)
    reasoning = (
        f"Heuristic ranking: {matches:.1f} weighted concept matches, "
        f"survey_boost={'yes' if is_survey else 'no'}, year={year or 'unknown'}"
    )
    return total_score, reasoning


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    context_profile: dict[str, Any],
    max_candidates_to_score: int = 12,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    scored_prelim = []
    for c in candidates:
        rec = dict(c)
        h_score, h_reason = _heuristic_score_candidate(rec, context_profile)
        rec["score"] = h_score
        rec["reasoning"] = h_reason
        rec["eval_method"] = "heuristic_fallback"
        scored_prelim.append(rec)

    scored_prelim.sort(
        key=lambda c: (
            float(c.get("score", 0.0)),
            1 if c.get("doi") else 0,
            1 if c.get("arxiv_id") else 0,
            int(c.get("year") or 0),
            c.get("title") or "",
        ),
        reverse=True,
    )

    shortlist = scored_prelim[:max_candidates_to_score]

    try:
        cand_lines = []
        id_map = {}
        for i, c in enumerate(shortlist):
            cid = f"C{i+1}"
            id_map[cid] = c
            id_map[i] = c
            id_map[str(i)] = c
            id_map[str(i+1)] = c
            auth = ", ".join(c.get("authors", [])[:3]) or "Unknown"
            cand_lines.append(f"[{cid}] Title: {c.get('title')} | Year: {c.get('year')} | Authors: {auth}")

        subtopics_str = ", ".join(context_profile.get("subtopics", []))
        concepts_str = ", ".join(context_profile.get("key_concepts", []))

        user_prompt = DISCOVERY_EVAL_CANDIDATES_USER.format(
            core_problem=context_profile.get("core_problem", ""),
            primary_domain=context_profile.get("primary_domain", ""),
            subtopics=subtopics_str,
            key_concepts=concepts_str,
            seed_criteria=context_profile.get("seed_criteria", ""),
            candidate_summaries="\n".join(cand_lines),
        )

        res = chat(
            messages=[
                {"role": "system", "content": DISCOVERY_EVAL_CANDIDATES_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            options={"num_ctx": 4096},
        )

        raw_json = _extract_json_text(res.content)
        data = json.loads(raw_json)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                cid = item.get("id")
                idx = item.get("index")
                target = id_map.get(cid) if cid is not None else None
                if target is None and idx is not None:
                    target = id_map.get(idx)

                if target is not None:
                    raw_score = item.get("score") if item.get("score") is not None else item.get("total_score")
                    if raw_score is not None:
                        try:
                            target["score"] = float(raw_score)
                        except (ValueError, TypeError):
                            pass
                    if item.get("reasoning"):
                        target["reasoning"] = str(item["reasoning"])
                    target["eval_method"] = "llm"

            shortlist.sort(
                key=lambda c: (
                    float(c.get("score", 0.0)),
                    1 if c.get("doi") else 0,
                    1 if c.get("arxiv_id") else 0,
                    int(c.get("year") or 0),
                    c.get("title") or "",
                ),
                reverse=True,
            )
            return shortlist
    except Exception as exc:
        logger.warning("LLM candidate evaluation failed (%s); using heuristic fallback.", exc)

    return shortlist


def download_seed_cascade(
    ranked_candidates: list[dict[str, Any]],
    force: bool = False,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    os.makedirs(RAW_DIR, exist_ok=True)

    for rank_idx, cand in enumerate(ranked_candidates, 1):
        key = source_key(cand)
        if not key:
            continue

        dest = os.path.join(RAW_DIR, filename_for(key))

        if os.path.exists(dest) and not force:
            try:
                with open(dest, "rb") as f:
                    if _is_pdf(f.read(5)):
                        logger.info("Candidate #%d [%s] already cached on disk.", rank_idx, key)
                        cand["download_status"] = "cached"
                        return cand, key, dest
            except OSError:
                pass

        pdf_url = cand.get("pdf_url")
        if not pdf_url:
            continue

        logger.info("Attempting download of candidate #%d [%s]: %s", rank_idx, key, pdf_url)
        ok, reason = download_pdf(pdf_url, dest)

        if not ok and cand.get("arxiv_id"):
            arxiv_fallback = f"https://arxiv.org/pdf/{cand['arxiv_id']}.pdf"
            if arxiv_fallback != pdf_url:
                logger.info("Retrying candidate [%s] via direct arXiv link: %s", key, arxiv_fallback)
                ok, reason = download_pdf(arxiv_fallback, dest)
                if ok:
                    cand["pdf_url"] = arxiv_fallback

        if ok:
            logger.info("Successfully downloaded seed candidate #%d: [%s]", rank_idx, key)
            cand["download_status"] = "downloaded"
            return cand, key, dest

        cand["download_status"] = f"failed: {reason}"
        logger.warning("Candidate #%d [%s] download failed (%s). Cascading...", rank_idx, key, reason)
        try:
            pipeline_status.add_event(
                f"⚠️ Candidate #{rank_idx} ({cand.get('title', key)[:35]}) download failed ({reason}). Cascading to next candidate..."
            )
        except Exception:
            pass

    logger.error("Exhausted all %d candidates without a successful PDF download.", len(ranked_candidates))
    return None, None, None


def find_and_fetch_seed(query, force=False, limit=SEARCH_LIMIT, on_progress=None):
    """Adaptive literature discovery ladder: formulate context, iterative search,
    evaluate candidates, and download in cascade until a PDF validates.

    Returns (candidate, key, dest_path) or (None, None, None).
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return None, None, None

    os.makedirs(RAW_DIR, exist_ok=True)
    context_profile = formulate_context(clean_query)
    candidates, search_trace = iterative_search(context_profile, providers=SEARCH_PROVIDERS, limit=limit)
    oa_candidates = filter_open_access(candidates)
    if not oa_candidates:
        logger.warning("No candidate papers in pool possess an open-access PDF link.")
        try:
            pipeline_status.add_event("⚠️ No open-access PDF links found among search candidates.")
        except Exception:
            pass
        return None, None, None

    ranked = evaluate_candidates(oa_candidates, context_profile)
    cand, key, dest = download_seed_cascade(ranked, force=force)
    return cand, key, dest


# ─── Manifest ────────────────────────────────────────────────────────────────


def _load_seeds():
    if not os.path.exists(SEED_PAPERS_PATH):
        return {}
    try:
        with open(SEED_PAPERS_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("%s is unreadable; starting fresh.", SEED_PAPERS_PATH)
        return {}


def _save_seeds(seeds):
    """Atomic: _load_seeds() treats an unparseable manifest as empty, so a
    truncated write silently forgets every paper already seeded."""
    atomic_write_json(SEED_PAPERS_PATH, seeds, ensure_ascii=False)


def get_seed(query):
    """Return Agent 0's manifest record for *query* (key, title, path, …) or None."""
    return _load_seeds().get(query)


def get_seed_by_path(path):
    """Return (record, query) for a seed record with matching path, or (None, None).
    Iterates in reverse so that the most recent record for a path is returned."""
    if not path:
        return None, None
    norm = os.path.abspath(path)
    for q, rec in reversed(list(_load_seeds().items())):
        p = rec.get("path")
        if p and os.path.abspath(p) == norm:
            return rec, q
    return None, None


# ─── Direct-URL fallback ─────────────────────────────────────────────────────

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z-]+/[0-9]{7})")


def _from_arxiv_url(url):
    """(pdf_url, arxiv_id) for an arxiv.org link, else (url, None)."""
    m = _ARXIV_RE.search(url or "")
    if not m:
        return url, None
    arxiv_id = m.group(1)
    return f"https://arxiv.org/pdf/{arxiv_id}", arxiv_id


def _download_and_record(query, key, pdf_url, seeds, force, arxiv_id=None, source="manual-url"):
    """Download *pdf_url* into RAW_DIR, write the seed manifest, return the path."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, filename_for(key))

    if os.path.exists(dest) and not force:
        logger.info("PDF already on disk: %s", os.path.basename(dest))
    else:
        ok, reason = download_pdf(pdf_url, dest)
        if not ok:
            logger.error("Could not download %s: %s", pdf_url, reason)
            return None
        logger.info("Saved -> %s", os.path.basename(dest))

    seeds[query] = SeedPaper(
        key=key,
        url=pdf_url,
        path=dest,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        arxiv_id=arxiv_id,
        source=source,
    ).to_dict()
    _save_seeds(seeds)
    return dest


def discover_from_url(query, url, force=False):
    """Seed from a PDF link the user supplied — the fallback when the search
    turns up nothing open-access. arXiv abstract links are rewritten to the PDF.

    Returns the local PDF path, or None if the download failed.
    """
    seeds = _load_seeds()
    prior = seeds.get(query)
    if prior and not force and os.path.exists(prior.get("path", "")):
        logger.info("Already seeded for this query: %s", prior["path"])
        return prior["path"]

    pdf_url, arxiv_id = _from_arxiv_url(url.strip())
    key = f"arxiv:{arxiv_id}" if arxiv_id else (source_key({"url": pdf_url}) or "url:manual")
    logger.info("Seeding from supplied link [%s]: %s", key, pdf_url)
    return _download_and_record(
        query, key, pdf_url, seeds, force, arxiv_id=arxiv_id, source="manual-url"
    )


# ─── Direct-file fallback ────────────────────────────────────────────────────
# Note: _DOI_RE, _ARXIV_RE_BYTES, _PDF_TITLE_RE, _XMP_TITLE_RE, _XMP_TITLE_FALLBACK_RE,
# _XMP_DOI_RE, _HEADER_JUNK_RE, _TITLE_ID_RE, _TITLE_FILE_RE, _CJK_RE,
# _TITLE_BANNER_WORDS, _decode_pdf_title, _clean_title, _clean_doi, _is_pdf_content,
# and _inspect_pdf are imported from research_assistant.shared.pdf_metadata and
# re-exported at module level above for backward compatibility.


def discover_from_file(
    query: str = "",
    file_input: str | bytes | os.PathLike = "",
    filename: str | None = None,
    force: bool = False,
) -> str | None:
    """Seed directly from an uploaded or local PDF file.

    Validates that the file is a PDF, extracts or infers metadata (title, DOI, arXiv ID),
    computes a deterministic key, copies/writes it to RAW_DIR, and records the seed paper.
    If *query* is empty, the paper's title or cleaned filename is used as the query.

    Returns the local path in RAW_DIR, or None if the file is invalid or cannot be read.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    if isinstance(file_input, (str, os.PathLike)):
        file_path = str(file_input)
        if not os.path.exists(file_path):
            logger.error("Seed file does not exist: %s", file_path)
            return None
        filename = filename or os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except OSError as exc:
            logger.error("Could not read seed file %s: %s", file_path, exc)
            return None
    elif hasattr(file_input, "read"):
        filename = filename or getattr(file_input, "name", "uploaded_paper.pdf")
        content = file_input.read()
        file_path = None
    elif isinstance(file_input, (bytes, bytearray)):
        content = bytes(file_input)
        filename = filename or "uploaded_paper.pdf"
        file_path = None
    else:
        logger.error("Unsupported file input type: %s", type(file_input))
        return None

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    if not _is_pdf_content(content[:1024]):
        logger.error("File is not a valid PDF (header does not contain %%PDF-): %s", filename)
        return None

    stem = os.path.splitext(os.path.basename(filename))[0]
    clean_stem = re.sub(r"[_\-]+", " ", stem).strip()
    title, doi, arxiv_id = _inspect_pdf(file_path or "", raw_bytes=content)

    if arxiv_id:
        key = f"arxiv:{arxiv_id}"
    elif doi:
        key = f"doi:{normalise_doi(doi) or doi.lower()}"
    else:
        digest = hashlib.sha1(content).hexdigest()[:16]
        key = f"file:{digest}"

    dest = os.path.join(RAW_DIR, filename_for(key))

    if os.path.exists(dest) and not force:
        logger.info("PDF already on disk in RAW_DIR: %s", os.path.basename(dest))
    else:
        tmp_path = dest + ".part"
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            os.replace(tmp_path, dest)
        except OSError as exc:
            logger.error("Failed writing seed PDF to %s: %s", dest, exc)
            return None
        logger.info("Saved seed PDF -> %s", os.path.basename(dest))

    seeds = _load_seeds()
    supplied_query = (query or "").strip()
    effective_query = supplied_query or title or clean_stem or "Uploaded seed paper"

    # The manifest is keyed by query, which is the right identity when the
    # caller supplied one — discover() and discover_from_url() both rely on
    # seeds[query] to answer "have I already seeded this query?". But an
    # *inferred* query is derived from the paper's title or filename stem, and
    # two different papers can infer the same string. Keying on it would drop
    # the earlier paper's record while leaving its PDF in RAW_DIR, so the file
    # would still be extracted and ingested with no seed record to point at.
    # The paper's own key is unique, so it disambiguates.
    if not supplied_query:
        clash = seeds.get(effective_query)
        if clash and clash.get("key") != key:
            effective_query = f"{effective_query} [{key}]"

    seeds[effective_query] = SeedPaper(
        key=key,
        path=dest,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="upload",
        title=title or clean_stem,
        doi=doi,
        arxiv_id=arxiv_id,
    ).to_dict()
    _save_seeds(seeds)

    return dest


# ─── Entry point ─────────────────────────────────────────────────────────────


def discover(query, force=False, on_progress=None):
    """Find the top relevant open-access paper for *query* and pull it to RAW_DIR.

    Returns the local PDF path, or None when nothing usable was found.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return None

    seeds = _load_seeds()
    prior = seeds.get(clean_query)
    if prior and not force and os.path.exists(prior.get("path", "")):
        logger.info("Already seeded for this query: %s", prior["path"])
        return prior["path"]

    logger.info("Searching for: %s", clean_query)
    try:
        pipeline_status.update_progress(
            detail="Initiating adaptive literature discovery...",
            current_item_name=clean_query[:50],
        )
    except Exception:
        pass
    if on_progress:
        try:
            on_progress("init", 0.05, "Initiating discovery...")
        except Exception:
            pass

    os.makedirs(RAW_DIR, exist_ok=True)
    context_profile = formulate_context(clean_query)
    if on_progress:
        try:
            on_progress("context", 0.25, f"Context formulated: {context_profile.get('primary_domain')}")
        except Exception:
            pass

    candidates, search_trace = iterative_search(context_profile, providers=SEARCH_PROVIDERS, limit=SEARCH_LIMIT)
    if on_progress:
        try:
            on_progress("search", 0.60, f"Search completed: {len(candidates)} candidates")
        except Exception:
            pass

    oa_candidates = filter_open_access(candidates)
    if not oa_candidates:
        logger.error("No downloadable open-access PDF for query: %s", clean_query)
        try:
            pipeline_status.add_event("⚠️ No open-access PDF links found among search candidates.")
        except Exception:
            pass
        return None

    ranked = evaluate_candidates(oa_candidates, context_profile)
    if on_progress:
        try:
            on_progress("evaluation", 0.80, f"Scored {len(ranked)} candidates")
        except Exception:
            pass

    cand, key, dest = download_seed_cascade(ranked, force=force)
    if not dest:
        logger.error("No downloadable open-access PDF for query: %s", clean_query)
        return None

    seeds[clean_query] = SeedPaper(
        key=key,
        path=dest,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="search",
        title=cand.get("title"),
        url=cand.get("pdf_url"),
        doi=cand.get("doi"),
        arxiv_id=cand.get("arxiv_id"),
    ).to_dict()
    _save_seeds(seeds)

    # Persist decoupled discovery trace
    trace_record = {
        "query": clean_query,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context_profile": context_profile,
        "search_trace": search_trace,
        "ranked_candidates": [
            {
                "rank": idx + 1,
                "title": c.get("title"),
                "doi": c.get("doi"),
                "arxiv_id": c.get("arxiv_id"),
                "score": c.get("score"),
                "reasoning": c.get("reasoning"),
                "eval_method": c.get("eval_method"),
            }
            for idx, c in enumerate(ranked)
        ],
        "selected_seed": {
            "key": key,
            "title": cand.get("title"),
            "path": dest,
            "score": cand.get("score"),
            "reasoning": cand.get("reasoning"),
        },
    }
    _save_discovery_trace(clean_query, trace_record)

    try:
        score_val = cand.get("score", 0.0)
        pipeline_status.add_event(f"🏆 Selected seed paper: '{cand.get('title', key)[:40]}' (score: {score_val})")
    except Exception:
        pass

    if on_progress:
        try:
            on_progress("done", 1.0, f"Selected seed: {cand.get('title', key)}")
        except Exception:
            pass

    return dest


def main():
    parser = argparse.ArgumentParser(description="Agent 0 — Seed paper discoverer")
    parser.add_argument("--query", default="", help="Research idea to seed from.")
    parser.add_argument(
        "--url",
        help="Seed directly from this PDF / arXiv link instead of searching.",
    )
    parser.add_argument(
        "--file",
        help="Seed directly from a local PDF file instead of searching.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-search and re-download even if this query was already seeded.",
    )
    args = parser.parse_args()

    if args.file:
        path = discover_from_file(args.query, args.file, force=args.force)
    elif args.url:
        path = discover_from_url(args.query, args.url, force=args.force)
    else:
        if not args.query:
            parser.error("Must provide --query, --file, or --url")
        path = discover(args.query, force=args.force)
    if not path:
        raise SystemExit(1)
    print(path)


if __name__ == "__main__":
    main()
