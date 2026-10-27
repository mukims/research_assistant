"""
research_assistant.audit — Modular citation audit, claim verification, and reliability evaluation package.
"""

from research_assistant.audit.extractor import (
    XML_OUTPUT_DIR,
    extract_seed_citation_claims,
    find_tei_for_seed,
)
from research_assistant.audit.resolver import (
    _load_downloaded_manifest,
    _match_downloaded_paper,
    _resolve_ref,
    _title_key,
    attach_cited_summaries,
    contextualize_citation_queries,
)
from research_assistant.audit.report import (
    _audit_item_badge,
    explain_rubric_verdict,
    explain_verdict_steps,
    generate_seed_audit_markdown,
    unassessed_bucket,
)

__all__ = [
    "XML_OUTPUT_DIR",
    "extract_seed_citation_claims",
    "find_tei_for_seed",
    "_load_downloaded_manifest",
    "_match_downloaded_paper",
    "_resolve_ref",
    "_title_key",
    "attach_cited_summaries",
    "contextualize_citation_queries",
    "_audit_item_badge",
    "explain_rubric_verdict",
    "explain_verdict_steps",
    "generate_seed_audit_markdown",
    "unassessed_bucket",
]
