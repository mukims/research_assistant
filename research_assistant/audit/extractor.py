"""
extractor.py — Extract in-text citation claims and bibliography references from TEI XML.
Unified with shared.seed_audit to eliminate signature mismatches and parsing defects.
"""

from __future__ import annotations

import os
from typing import Optional
from bs4 import BeautifulSoup

from research_assistant.config import RAW_DIR
from research_assistant.shared.seed_audit import (
    extract_seed_citation_claims,
    find_tei_for_seed,
)

XML_OUTPUT_DIR = os.path.join(RAW_DIR, "grobid_output")

__all__ = [
    "XML_OUTPUT_DIR",
    "extract_seed_citation_claims",
    "find_tei_for_seed",
]
