"""Packaging guards.

research_assistant/judgement ships two non-Python files that judge.py reads at
MODULE scope. app.py imports the verifier at module scope too, so if these are
missing from an install the Streamlit app fails to start — not just Agent 8.
setuptools does not ship non-Python files unless told to, and cases/ has no
__init__.py so packages.find never sees it. These tests guard the declaration.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


class TestJudgementDataFilesAreDeclared(unittest.TestCase):
    def setUp(self):
        self.text = PYPROJECT.read_text(encoding="utf-8")

    def test_package_data_section_exists(self):
        self.assertIn(
            "[tool.setuptools.package-data]",
            self.text,
            "pyproject.toml must declare package-data or the judgement "
            "prompt and cases are omitted from any non-editable install.",
        )

    def test_prompt_is_declared(self):
        section = self.text.split("[tool.setuptools.package-data]", 1)[-1]
        self.assertIn("prompt.md", section)

    def test_cases_are_declared(self):
        section = self.text.split("[tool.setuptools.package-data]", 1)[-1]
        self.assertTrue(
            re.search(r"cases/\*\.jsonl|cases\.jsonl", section),
            "cases/cases.jsonl must be covered by a package-data pattern.",
        )


class TestDataFilesExistWhereJudgeExpectsThem(unittest.TestCase):
    """The paths judge.py resolves at import time."""

    def test_prompt_present(self):
        self.assertTrue(
            (REPO_ROOT / "research_assistant" / "judgement" / "prompt.md").is_file()
        )

    def test_cases_present(self):
        self.assertTrue(
            (REPO_ROOT / "research_assistant" / "judgement" / "cases" / "cases.jsonl").is_file()
        )
