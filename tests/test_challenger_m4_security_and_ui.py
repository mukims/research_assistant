"""
Adversarial Security, UI, and Boundary Hardening Empirical Stress Test Suite.
Author: Empirical Challenger M4 2 (challenger_m4_2)
Milestone: Milestone 4 (Phase 2: Tier 5 Adversarial Coverage Hardening)

Empirically challenges and stress-tests:
1. UI Security Defenses (batch_uploader.py, ui/results.py, tasks/ui.py):
   - Path traversal defenses (is_safe_path) against system roots, parent directory traversal,
     prefix collisions, symlink escapes, URL-encoded traversal, Windows mixed separators, and null bytes.
   - SSRF defenses (is_safe_url) against loopbacks, link-local cloud metadata, private RFC 1918 subnets,
     Carrier-Grade NAT (CGNAT RFC 6598 100.64.0.0/10), non-HTTP schemes, alternative decimal/hex/octal IP formats,
     IPv6 variations, and dual-homed DNS rebinding.
   - ZIP bomb and limit defenses in unpack_and_stage_uploads (file count cap at 500, single file cap at 100MB,
     total uncompressed cap at 500MB, high compression ratio bombs, corrupt non-zip files, and disguised zip files).
   - XSS sanitization and strict Mermaid rendering (html.escape in ui/results.py and tasks/ui.py,
     securityLevel: 'strict' and htmlLabels: false in _render_mermaid_interactive).
2. Tasks Module Uncoupling & Runner Hardening (research_assistant/tasks/):
   - Zero residual benchmark domain terms ('Stealthy Hyperuniform', 'RGF', 'Heaviside', 'tight-binding', 'KPM')
     in prompts, critic rules, and extractor templates.
   - TaskRunner file output handling: markdown code fence stripping for Python and CSV, conversational prose
     pruning in CSV deliverables, rejection of malformed CSV in autonomous mode, and robust LLM exception handling
     in Cycle 1 and Cycle 2 reflexion.
   - torch.load safety: verification that generated test harness scripts strictly enforce weights_only=True.
3. Transfer Engine UI & Card Grounding (research_assistant/transfer/, research_assistant/ui/):
   - Streamlit UI integration: render_transfer_tab renders without crashing or throwing unhandled exceptions.
   - Card evidence grounding by stable document identity across document reordering, deletions, and duplicate chunk IDs.
   - Stage 6 revision updates propagating revised mapping, steps, prediction, summary, and title to Transfer.
   - Prior art temporal cutoff handling when before_year is None, hit year is None, or year strings are malformed.
"""

from __future__ import annotations

import ast
import csv
import glob
import html
import io
import json
import os
import re
import shutil
import socket
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from research_assistant import config
from research_assistant.shared.batch_uploader import (
    MAX_SINGLE_FILE_SIZE,
    MAX_TOTAL_UNCOMPRESSED_SIZE,
    MAX_ZIP_FILE_COUNT,
    is_safe_path,
    is_safe_url,
    scan_and_stage_directory,
    unpack_and_stage_uploads,
)
from research_assistant.tasks.harness import generate_task_test_script
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.runner import ExecutionMode, TaskRunner
from research_assistant.tasks.ui import _render_mermaid_interactive
from research_assistant.transfer.deps import TransferSettings
from research_assistant.transfer.models import (
    Bottleneck,
    Card,
    Critique,
    Identifiability,
    MappingRow,
    Passage,
    Premise,
    PriorArt,
    Step,
    Transfer,
)
from research_assistant.transfer.premises import PremiseChecker
from research_assistant.transfer.prior_art import (
    PriorArtClient,
    parse_arxiv,
    parse_openalex,
)
from research_assistant.transfer.stages import Seed, develop, ground
from research_assistant.ui import results
from tests.transfer_fixtures import world


class TestAdversarialUISecurityPathTraversal(unittest.TestCase):
    """Empirical adversarial stress test targeting is_safe_path directory traversal defenses."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="adv_traversal_m4_")
        self.workspace = os.path.join(self.tmp_dir, "safe_workspace")
        os.makedirs(self.workspace, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_system_roots_and_forbidden_dirs_blocked(self):
        """Probes system root and sensitive directories across UNIX and macOS."""
        forbidden_targets = [
            "/",
            "/etc",
            "/etc/passwd",
            "/etc/shadow",
            "/etc/hosts",
            "/bin",
            "/bin/sh",
            "/sbin",
            "/usr",
            "/usr/bin",
            "/var/log",
            "/dev",
            "/proc",
            "/sys",
            "/private/etc",
            "/private/var/log",
            "/private/var/root",
            "/System",
            "/Library",
        ]
        for target in forbidden_targets:
            with self.subTest(target=target):
                self.assertFalse(
                    is_safe_path(target, allowed_roots=[self.workspace]),
                    f"Forbidden system path must be rejected: {target}",
                )

    def test_parent_directory_traversal_blocked(self):
        """Probes parent directory traversal payloads (..) attempting to escape allowed root."""
        traversal_payloads = [
            os.path.join(self.workspace, ".."),
            os.path.join(self.workspace, "..", ".."),
            os.path.join(self.workspace, "..", "etc", "passwd"),
            os.path.join(self.workspace, "sub", "..", "..", "..", "etc", "shadow"),
            os.path.join(self.workspace, "dir1", "dir2", "..", "..", "..", "bin", "sh"),
            f"{self.workspace}/../../../../../../../../etc/passwd",
        ]
        for payload in traversal_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(
                    is_safe_path(payload, allowed_roots=[self.workspace]),
                    f"Parent traversal payload must be rejected: {payload}",
                )

    def test_prefix_collision_attacks_blocked(self):
        """Probes prefix collision where target directory shares prefix with allowed root."""
        collision_sibling = self.workspace + "_evil"
        os.makedirs(collision_sibling, exist_ok=True)
        try:
            self.assertFalse(
                is_safe_path(collision_sibling, allowed_roots=[self.workspace]),
                "Prefix collision sibling directory must not be admitted",
            )
            self.assertFalse(
                is_safe_path(os.path.join(collision_sibling, "secret.pdf"), allowed_roots=[self.workspace]),
                "File in prefix collision sibling must not be admitted",
            )
        finally:
            shutil.rmtree(collision_sibling, ignore_errors=True)

    def test_symlink_escape_attacks_blocked(self):
        """Probes symlink escapes where a symlink inside workspace points to sensitive locations."""
        symlink_etc = os.path.join(self.workspace, "escape_etc")
        os.symlink("/etc", symlink_etc)

        outside_secret = os.path.join(self.tmp_dir, "outside_secret.pdf")
        with open(outside_secret, "wb") as f:
            f.write(b"%PDF-1.4 secret")
        symlink_file = os.path.join(self.workspace, "link_secret.pdf")
        os.symlink(outside_secret, symlink_file)

        self.assertFalse(
            is_safe_path(symlink_etc, allowed_roots=[self.workspace]),
            "Symlink pointing to /etc must be rejected",
        )
        self.assertFalse(
            is_safe_path(os.path.join(symlink_etc, "passwd"), allowed_roots=[self.workspace]),
            "File traversal through /etc symlink must be rejected",
        )
        self.assertFalse(
            is_safe_path(symlink_file, allowed_roots=[self.workspace]),
            "Symlink pointing to unpermitted outside file must be rejected",
        )

    def test_malformed_and_non_string_inputs_handled_safely(self):
        """Probes boundary cases: None, empty string, non-string types, and pure whitespace."""
        self.assertFalse(is_safe_path(None, allowed_roots=[self.workspace]))
        self.assertFalse(is_safe_path("", allowed_roots=[self.workspace]))
        self.assertFalse(is_safe_path(12345, allowed_roots=[self.workspace]))
        self.assertFalse(is_safe_path(["/etc/passwd"], allowed_roots=[self.workspace]))
        self.assertFalse(is_safe_path({"path": "/tmp"}, allowed_roots=[self.workspace]))
        self.assertFalse(is_safe_path("   ", allowed_roots=[self.workspace]))
        self.assertFalse(is_safe_path("\t\n", allowed_roots=[self.workspace]))

    def test_scan_and_stage_directory_rejects_unsafe_paths(self):
        """Verifies that scan_and_stage_directory halts on directory traversal attempts."""
        res_etc = scan_and_stage_directory("/etc", destination_dir=self.workspace)
        self.assertEqual(res_etc, [], "Scanning /etc must return empty list")

        res_traversal = scan_and_stage_directory(
            os.path.join(self.workspace, "..", "..", "etc"),
            destination_dir=self.workspace,
        )
        self.assertEqual(res_traversal, [], "Scanning parent traversal must return empty list")

    def test_null_byte_traversal_behavior(self):
        """
        Adversarially probes null byte handling in is_safe_path.
        Records whether null byte is handled safely (returning False) or raises ValueError.
        """
        null_payloads = [
            os.path.join(self.workspace, "\x00/etc/passwd"),
            "/tmp/safe\x00/etc/shadow",
            "\x00",
            os.path.join(self.workspace, "file.pdf\x00.exe"),
        ]
        for payload in null_payloads:
            with self.subTest(payload=repr(payload)):
                try:
                    res = is_safe_path(payload, allowed_roots=[self.workspace])
                    # If handled without exception, must return False
                    self.assertFalse(res, f"Null byte payload must return False: {repr(payload)}")
                except ValueError as exc:
                    # Document empirical vulnerability: unhandled ValueError on embedded null byte
                    self.fail(
                        f"VULNERABILITY CONFIRMED: is_safe_path crashed with unhandled exception on null byte "
                        f"{repr(payload)}: {type(exc).__name__}: {exc}"
                    )


class TestAdversarialUISecuritySSRF(unittest.TestCase):
    """Empirical adversarial stress test targeting is_safe_url SSRF defenses."""

    def test_loopback_and_local_urls_blocked(self):
        """Probes loopback addresses in IPv4, IPv6, and localhost."""
        loopback_urls = [
            "http://127.0.0.1",
            "http://127.0.0.1:8000/api",
            "http://127.0.0.2:8501",
            "http://127.1",
            "http://127.0.1.1",
            "http://localhost",
            "http://localhost:8501/status",
            "http://[::1]",
            "http://[::1]:8080/metrics",
            "http://0.0.0.0",
            "http://0.0.0.0:8000",
        ]
        for url in loopback_urls:
            with self.subTest(url=url):
                self.assertFalse(is_safe_url(url), f"SSRF check failed to block loopback URL: {url}")

    def test_cloud_metadata_endpoints_blocked(self):
        """Probes cloud metadata endpoints (AWS, GCP, Azure link-local 169.254.169.254)."""
        metadata_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/latest/user-data",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://169.254.1.1/secret",
            "http://169.254.254.254",
        ]
        for url in metadata_urls:
            with self.subTest(url=url):
                self.assertFalse(is_safe_url(url), f"SSRF check failed to block metadata URL: {url}")

    def test_private_rfc1918_ranges_blocked(self):
        """Probes private RFC 1918 subnets (10/8, 172.16/12, 192.168/16)."""
        private_urls = [
            "http://10.0.0.1/internal.pdf",
            "http://10.255.255.254/admin",
            "http://172.16.0.1/status",
            "http://172.31.255.255/intranet",
            "http://192.168.0.1/router",
            "http://192.168.1.254:8080",
        ]
        for url in private_urls:
            with self.subTest(url=url):
                self.assertFalse(is_safe_url(url), f"SSRF check failed to block private RFC1918 URL: {url}")

    def test_carrier_grade_nat_cgnat_blocked(self):
        """
        Adversarially probes Carrier-Grade NAT (RFC 6598: 100.64.0.0/10).
        CGNAT is shared internal/non-global space used by ISPs and cloud infrastructure.
        Empirically verifies whether is_safe_url blocks or permits CGNAT addresses.
        """
        cgnat_urls = [
            "http://100.64.0.1",
            "http://100.64.0.1:8000/api",
            "http://100.100.100.100/metadata",
            "http://100.127.255.254",
            "http://[::ffff:100.64.0.1]",
        ]
        for url in cgnat_urls:
            with self.subTest(url=url):
                res = is_safe_url(url)
                if res is True:
                    self.fail(
                        f"VULNERABILITY CONFIRMED: is_safe_url permits Carrier-Grade NAT (RFC 6598) URL: {url} "
                        f"(ipaddress.is_private is False for 100.64.0.0/10; requires check against not ip.is_global "
                        f"or ipaddress.ip_network('100.64.0.0/10'))"
                    )
                self.assertFalse(res, f"CGNAT address must be blocked: {url}")

    def test_alternative_numeric_ip_formats_blocked(self):
        """Probes alternative numeric IP formats (decimal, hex, octal)."""
        numeric_urls = [
            ("http://2130706433/", "Decimal 127.0.0.1"),
            ("http://0x7f000001/", "Hex 127.0.0.1"),
            ("http://017700000001/", "Octal single int 127.0.0.1"),
        ]
        for url, desc in numeric_urls:
            with self.subTest(url=url, desc=desc):
                self.assertFalse(
                    is_safe_url(url),
                    f"Alternative numeric notation ({desc}) must be blocked: {url}",
                )

    def test_non_http_schemes_blocked(self):
        """Probes dangerous non-HTTP URL schemes."""
        scheme_urls = [
            "file:///etc/passwd",
            "file:///etc/hosts",
            "gopher://127.0.0.1:6379/_FLUSHALL",
            "ftp://anonymous@internal.corp/secret.pdf",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            "javascript:alert(document.cookie)",
            "dict://127.0.0.1:11211/stat",
        ]
        for url in scheme_urls:
            with self.subTest(url=url):
                self.assertFalse(is_safe_url(url), f"Non-HTTP scheme must be blocked: {url}")

    def test_ipv6_variations_blocked(self):
        """Probes IPv6 special addresses (unspecified, ULA, link-local, IPv4-mapped)."""
        ipv6_urls = [
            "http://[::]",
            "http://[::]:8080",
            "http://[0000::1]",
            "http://[fc00::1]/admin",
            "http://[fe80::1]/linklocal",
            "http://[::ffff:127.0.0.1]",
            "http://[::ffff:10.0.0.1]",
            "http://[::ffff:169.254.169.254]",
        ]
        for url in ipv6_urls:
            with self.subTest(url=url):
                self.assertFalse(is_safe_url(url), f"IPv6 special address must be blocked: {url}")

    def test_dns_rebinding_dual_homed_blocked(self):
        """Probes multi-homed DNS response where one resolved IP is private/loopback."""
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
        ]
        with patch("socket.getaddrinfo", return_value=mock_addr_info):
            self.assertFalse(
                is_safe_url("http://rebind.attacker.com/paper.pdf"),
                "Dual-homed DNS resolving to both public and private IP must be rejected",
            )

    def test_valid_public_urls_allowed(self):
        """Probes legitimate academic URLs resolving to public IPs."""
        mock_addr_info = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch("socket.getaddrinfo", return_value=mock_addr_info):
            self.assertTrue(is_safe_url("https://arxiv.org/pdf/2401.12345.pdf"))
            self.assertTrue(is_safe_url("https://openalex.org/W123456789"))


class TestAdversarialZipBombsAndLimits(unittest.TestCase):
    """Empirical challenge against ZIP bombs, decompression ratios, and magic byte bypasses."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="adv_zip_m4_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_zip_bomb_file_count_capped_at_500(self):
        """Probes ZIP archive with > 500 files; verifies unpacking stops at MAX_ZIP_FILE_COUNT (500)."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx in range(520):
                zf.writestr(f"doc_{idx:04d}.pdf", b"%PDF-1.4 sample content")

        staged = unpack_and_stage_uploads([(bio.getvalue(), "bomb_count.zip")], destination_dir=self.tmp_dir)
        self.assertLessEqual(
            len(staged),
            MAX_ZIP_FILE_COUNT,
            f"Staged file count must not exceed MAX_ZIP_FILE_COUNT ({MAX_ZIP_FILE_COUNT})",
        )

    def test_zip_bomb_single_file_exceeding_100mb_skipped(self):
        """Probes ZIP archive with a member uncompressing to > 100MB (MAX_SINGLE_FILE_SIZE)."""
        bio = io.BytesIO()
        # 105 MB of zeros deflated compresses to ~107 KB
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("huge_file.pdf", b"0" * (105 * 1024 * 1024))
            zf.writestr("valid.pdf", b"%PDF-1.4 normal size")

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Normal Paper", "", "")):
            staged = unpack_and_stage_uploads([(bio.getvalue(), "bomb_single.zip")], destination_dir=self.tmp_dir)

        huge_staged = [p for p in staged if "huge" in p.get("filename", "")]
        self.assertEqual(huge_staged, [], "ZIP member exceeding 100MB must be skipped")
        self.assertEqual(len(staged), 1, "Only normal sized member should be staged")

    def test_zip_bomb_total_uncompressed_size_exceeding_500mb_halted(self):
        """Probes ZIP archive with cumulative uncompressed size > 500MB (MAX_TOTAL_UNCOMPRESSED_SIZE)."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx in range(6):
                zf.writestr(f"chunk_{idx}.pdf", b"0" * (90 * 1024 * 1024))

        staged = unpack_and_stage_uploads([(bio.getvalue(), "bomb_total.zip")], destination_dir=self.tmp_dir)
        self.assertEqual(len(staged), 0, "Cumulative extraction must halt when exceeding 500MB total")

    def test_corrupt_non_zip_with_zip_extension_handled_safely(self):
        """Probes non-zip garbage with a .zip extension; must handle BadZipFile safely without crashing."""
        garbage_bytes = b"This is not a zip file or a pdf file, just garbage text."
        staged = unpack_and_stage_uploads([(garbage_bytes, "corrupt.zip")], destination_dir=self.tmp_dir)
        self.assertEqual(staged, [], "Corrupted zip file must yield 0 staged papers")

    def test_disguised_zip_file_unpacked_properly(self):
        """Probes file without .zip extension but starting with PK\\x03\\x04 magic bytes."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("disguised_doc.pdf", b"%PDF-1.4 header text")

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Disguised Paper", "", "")):
            staged = unpack_and_stage_uploads([(bio.getvalue(), "payload.bin")], destination_dir=self.tmp_dir)
        self.assertEqual(len(staged), 1, "Disguised ZIP with PK magic bytes must be recognized and extracted")

    def test_zip_member_path_traversal_sanitized(self):
        """Probes zip member with path traversal filename (../../evil.pdf); must be sanitized into safe target."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("../../../../tmp/evil_traversal.pdf", b"%PDF-1.4 traversal test")

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Sanitized Paper", "", "")):
            staged = unpack_and_stage_uploads([(bio.getvalue(), "traversal.zip")], destination_dir=self.tmp_dir)
        for p in staged:
            path = p["path"]
            self.assertTrue(
                path.startswith(self.tmp_dir),
                f"Extracted member path must remain within target destination dir: {path}",
            )


class TestAdversarialXSSAndMermaidDefenses(unittest.TestCase):
    """Empirical challenge against XSS injection in Streamlit UI and Mermaid strict rendering."""

    def test_ui_results_fetch_failures_escaping(self):
        """Probes fetch_failures rendering with HTML/XSS payloads in title and reason."""
        xss_title = "<script>alert('xss_title')</script>"
        xss_reason = "<img src=x onerror=alert('xss_reason')><svg onload=alert(1)>"
        final_payload = {
            "fetch_failures": [{"title": xss_title, "reason": xss_reason}],
            "downloaded": [],
            "seed_papers": [],
        }

        rendered_markdown_calls = []
        with patch("streamlit.expander", return_value=MagicMock()), \
             patch("streamlit.markdown", side_effect=lambda txt, **kw: rendered_markdown_calls.append(txt)), \
             patch("streamlit.caption"):

            results._render_seed_and_downloads("query", final_payload)

        combined_rendered = "\n".join(rendered_markdown_calls)
        self.assertNotIn("<script>", combined_rendered, "Unescaped <script> tag detected in rendered UI output!")
        self.assertNotIn("<img src=x", combined_rendered, "Unescaped <img> XSS tag detected in rendered UI output!")
        self.assertNotIn("<svg", combined_rendered, "Unescaped <svg> XSS tag detected in rendered UI output!")
        self.assertIn("&lt;script&gt;", combined_rendered, "HTML entity escaping not found for script tag!")
        self.assertIn("&lt;img", combined_rendered, "HTML entity escaping not found for img tag!")
        self.assertIn("&lt;svg", combined_rendered, "HTML entity escaping not found for svg tag!")

    def test_ui_results_batch_uploaded_escaping(self):
        """Probes batch_uploaded paper cards with HTML injection in title, filename, and DOI."""
        xss_paper = {
            "title": "Quantum Mechanics <script>alert(1)</script>",
            "filename": "paper_<b style='color:red;'>bold</b>.pdf",
            "doi": "10.1000/<script>evil()</script>",
            "arxiv_id": "2401.0001<img src=x>",
        }
        final_payload = {"batch_uploaded": [xss_paper]}

        rendered_markdown_calls = []
        with patch("streamlit.container", return_value=MagicMock()), \
             patch("streamlit.markdown", side_effect=lambda txt, **kw: rendered_markdown_calls.append(txt)), \
             patch("streamlit.info"):

            results._render_build(final_payload, query="test_query")

        combined_rendered = "\n".join(rendered_markdown_calls)
        self.assertNotIn("<script>", combined_rendered, "Unescaped <script> tag detected in batch upload cards!")
        self.assertNotIn("<img src=x>", combined_rendered, "Unescaped <img> tag detected in batch upload cards!")
        self.assertIn("&lt;script&gt;", combined_rendered, "HTML entity escaping not found for script tag!")
        self.assertIn("&lt;img", combined_rendered, "HTML entity escaping not found for img tag!")

    def test_tasks_ui_mermaid_strict_configuration(self):
        """Verifies that _render_mermaid_interactive configures Mermaid with strict security."""
        html_captured = []
        with patch("streamlit.components.v1.html") as mock_html:
            _render_mermaid_interactive("graph TD\nA-->B", height=400)
            for call in mock_html.call_args_list:
                html_captured.append(call[0][0])

        combined_html = "\n".join(html_captured)
        self.assertIn(
            "securityLevel: 'strict'",
            combined_html,
            "Mermaid configuration must enforce securityLevel: 'strict'",
        )
        self.assertIn(
            "htmlLabels: false",
            combined_html,
            "Mermaid configuration must enforce htmlLabels: false to prevent script injection in nodes",
        )


class TestTasksModuleUncouplingAndRunnerHardening(unittest.TestCase):
    """Empirical challenge against benchmark term coupling, file output fences, and torch safety."""

    def test_tasks_module_domain_uncoupling(self):
        """
        Adversarially scans all Python files in research_assistant/tasks/ to verify complete
        absence of residual benchmark-specific domain terms.
        """
        tasks_dir = os.path.join(config.PROJECT_ROOT, "research_assistant", "tasks")
        forbidden_terms = [
            "stealthy hyperuniform",
            "hyperuniform",
            "rgf",
            "recursive green's function",
            "heaviside",
            "tight-binding",
            "kpm",
            "kernel polynomial method",
        ]

        found_occurrences = []
        for root, _, files in os.walk(tasks_dir):
            for file in files:
                if file.endswith(".py"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, config.PROJECT_ROOT)
                    with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f, 1):
                            line_lower = line.lower()
                            for term in forbidden_terms:
                                if term in line_lower and "remove" not in line_lower and "uncouple" not in line_lower:
                                    found_occurrences.append((rel_p, line_idx, term, line.strip()))

        self.assertEqual(
            found_occurrences,
            [],
            f"Residual benchmark domain terms found in tasks/ module: {found_occurrences}",
        )

    def test_task_runner_strips_markdown_code_fences_python(self):
        """Verifies TaskRunner strips ```python ... ``` fences and extracts valid AST code."""
        tmp_ws = tempfile.mkdtemp(prefix="adv_runner_py_")
        try:
            task = Task(
                task_id="T_CODE_1",
                title="Model Implementation",
                assigned_role=AgentRole.MODEL_ARCHITECT,
                outputs=["model.py"],
            )
            llm_output_with_fences = (
                "Here is the synthesized implementation for the model architecture:\n\n"
                "```python\n"
                "import math\n\n"
                "class LinearModel:\n"
                "    def __init__(self, dim: int):\n"
                "        self.dim = dim\n\n"
                "    def forward(self, x):\n"
                "        return x * 2.0\n"
                "```\n\n"
                "This completes the module."
            )
            runner = TaskRunner(TaskGraph(tasks=[task]), workspace_dir=tmp_ws, mode=ExecutionMode.AUTONOMOUS)
            runner._materialize_deliverables(task, llm_output_with_fences, "model_architect", overwrite=True, is_simulation=False)

            written_file = os.path.join(tmp_ws, "model.py")
            self.assertTrue(os.path.exists(written_file), "model.py deliverable must be written")
            with open(written_file, "r", encoding="utf-8") as f:
                code_content = f.read()

            self.assertNotIn("```", code_content, "Markdown fences must be stripped from python code deliverable")
            self.assertNotIn("Here is the synthesized", code_content, "Conversational prose must not be written to code file")
            ast.parse(code_content)
        finally:
            shutil.rmtree(tmp_ws, ignore_errors=True)

    def test_task_runner_strips_markdown_code_fences_and_prose_csv(self):
        """Verifies TaskRunner strips ```csv ... ``` fences and prunes conversational prose from CSVs."""
        tmp_ws = tempfile.mkdtemp(prefix="adv_runner_csv_")
        try:
            task = Task(
                task_id="T_CSV_1",
                title="Evaluation Metrics",
                assigned_role=AgentRole.EVALUATION_AGENT,
                outputs=["metrics.csv"],
                artifact_contracts=[ArtifactContract(path="metrics.csv", artifact_type=ArtifactType.TABULAR)],
            )
            llm_csv_output = (
                "Certainly! Here is the evaluation metrics table formatted as CSV:\n\n"
                "```csv\n"
                "# Generated by Marvin agent\n"
                "step,train_loss,val_loss,accuracy\n"
                "100,0.852,0.871,0.72\n"
                "200,0.612,0.643,0.84\n"
                "300,0.418,0.450,0.91\n"
                "```\n\n"
                "Note: The model converged by step 300."
            )
            runner = TaskRunner(TaskGraph(tasks=[task]), workspace_dir=tmp_ws, mode=ExecutionMode.AUTONOMOUS)
            runner._materialize_deliverables(task, llm_csv_output, "evaluation_agent", overwrite=True, is_simulation=False)

            written_file = os.path.join(tmp_ws, "metrics.csv")
            self.assertTrue(os.path.exists(written_file), "metrics.csv deliverable must be written")
            with open(written_file, "r", encoding="utf-8") as f:
                csv_content = f.read()

            self.assertNotIn("```", csv_content, "Markdown fences must be stripped from CSV deliverable")
            self.assertNotIn("Certainly!", csv_content, "Conversational preamble must be pruned from CSV")
            self.assertNotIn("Note:", csv_content, "Conversational postscript must be pruned from CSV")

            reader = list(csv.reader(io.StringIO(csv_content)))
            self.assertEqual(len(reader), 4, "Must contain exactly 4 CSV rows (1 header + 3 data rows)")
            self.assertEqual(reader[0], ["step", "train_loss", "val_loss", "accuracy"])
        finally:
            shutil.rmtree(tmp_ws, ignore_errors=True)

    def test_task_runner_rejects_malformed_csv_in_autonomous_mode(self):
        """Verifies TaskRunner does not write malformed non-comma prose rows to CSV in autonomous mode."""
        tmp_ws = tempfile.mkdtemp(prefix="adv_runner_bad_csv_")
        try:
            task = Task(
                task_id="T_CSV_BAD",
                title="Bad CSV Task",
                assigned_role=AgentRole.EVALUATION_AGENT,
                outputs=["results.csv"],
                artifact_contracts=[ArtifactContract(path="results.csv", artifact_type=ArtifactType.TABULAR)],
            )
            llm_invalid_csv = (
                "I am unable to run the evaluation because the test dataset is missing.\n"
                "Please provide data.parquet first."
            )
            runner = TaskRunner(TaskGraph(tasks=[task]), workspace_dir=tmp_ws, mode=ExecutionMode.AUTONOMOUS)
            runner._materialize_deliverables(task, llm_invalid_csv, "evaluation_agent", overwrite=True, is_simulation=False)

            written_file = os.path.join(tmp_ws, "results.csv")
            # No valid tabular data lines should be written (file size must be 0)
            if os.path.exists(written_file):
                self.assertEqual(os.path.getsize(written_file), 0, "No malformed prose rows must be written to CSV")
        finally:
            shutil.rmtree(tmp_ws, ignore_errors=True)

    def test_task_runner_llm_exceptions_in_cycle1_and_cycle2_handled(self):
        """Verifies LLM exceptions in Cycle 1 and Cycle 2 mark TaskStatus.FAILED with zero dummy comments."""
        tmp_ws = tempfile.mkdtemp(prefix="adv_runner_exc_")
        try:
            task = Task(
                task_id="T_EXC_1",
                title="Exception Task",
                assigned_role=AgentRole.MODEL_ARCHITECT,
                outputs=["model.py"],
            )
            runner = TaskRunner(TaskGraph(tasks=[task]), workspace_dir=tmp_ws, mode=ExecutionMode.AUTONOMOUS)

            with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("Ollama connection timeout")):
                status = runner.execute_with_reflection(task)

            self.assertEqual(status, TaskStatus.FAILED, "Task status must be marked FAILED on Cycle 1 LLM exception")
            self.assertEqual(task.status, TaskStatus.FAILED)
            self.assertIn("Ollama connection timeout", task.metadata.get("error", ""))
            written_file = os.path.join(tmp_ws, "model.py")
            self.assertFalse(os.path.exists(written_file), "No deliverable should be written on Cycle 1 LLM exception")
        finally:
            shutil.rmtree(tmp_ws, ignore_errors=True)

    def test_safe_torch_load_weights_only(self):
        """Verifies AST of test script generated by generate_task_test_script mandates weights_only=True."""
        task = Task(
            task_id="T_TORCH",
            title="PyTorch Model Training",
            assigned_role=AgentRole.EXPERIMENT_RUNNER,
            outputs=["model.pt", "checkpoint.pth"],
            artifact_contracts=[
                ArtifactContract(path="model.pt", artifact_type=ArtifactType.TENSOR),
            ],
        )
        script_code = generate_task_test_script(task, "/tmp/ws")
        tree = ast.parse(script_code)

        torch_load_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "load":
                    if isinstance(func.value, ast.Name) and func.value.id == "torch":
                        torch_load_calls.append(node)

        self.assertTrue(len(torch_load_calls) >= 1, "At least one torch.load call must be present in test harness")
        for call_node in torch_load_calls:
            weights_only_kw = next((kw for kw in call_node.keywords if kw.arg == "weights_only"), None)
            self.assertIsNotNone(weights_only_kw, "torch.load is missing weights_only keyword argument!")
            self.assertIsInstance(weights_only_kw.value, ast.Constant)
            self.assertEqual(weights_only_kw.value.value, True, "weights_only must be set to True!")


class TestTransferEngineUIAndCardGrounding(unittest.TestCase):
    """Empirical challenge against Streamlit UI wiring, stable Card grounding, and Stage 6 revision."""

    def test_transfer_tab_renders_cleanly(self):
        """Verifies render_transfer_tab executes cleanly without crashing in mock Streamlit environment."""
        from research_assistant.ui.transfer_tab import render_transfer_tab

        with patch("streamlit.button", return_value=False), \
             patch("streamlit.form_submit_button", return_value=False), \
             patch("streamlit.info"), \
             patch("research_assistant.ui.transfer_tab._render_live_pipeline_status"):
            render_transfer_tab()

    def test_card_evidence_grounding_document_isolation(self):
        """
        Probes Card evidence grounding when documents in deps.texts are reordered.
        Verifies evidence resolves by stable card.document identity rather than fragile array index.
        """
        deps, log = world()
        deps.texts = [
            "Unrelated noise document at index 0",
            "Target text for s1 at index 1: Hamiltonian matrix formulation",
            "Additional document at index 2",
        ]
        deps.metadatas = [
            {"document": "other_doc", "section": "intro", "chunk_index": 0},
            {"document": "s1", "section": "methods", "chunk_index": 1, "citation_source": "Source Citation 2024"},
            {"document": "other_doc2", "section": "conclusion", "chunk_index": 2},
        ]

        card = Card(
            card_id="card_s1",
            document="s1",
            citation="Source Citation 2024",
            name="Matrix Technique",
            problem_solved="Solves matrix inversion",
            context="Solid state physics",
            inputs="Hamiltonian",
            outputs="Eigenvalues",
            assumptions=[],
            evidence=Premise(
                claim="Hamiltonian matrix formulation",
                key="C1",
                document="s1",
                chunk_index=1,
            ),
        )
        seed = Seed(bottleneck_id="B1", card=card, score=0.9, level="broad", rank=1)
        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        s = TransferSettings()
        g = ground("q", deps, s, ch)

        t = develop(seed, 1, g, deps, s, ch)
        self.assertIsNotNone(t, "develop should succeed with isolated card document")
        self.assertEqual(t.card_document, "s1")

        # Verify that grounded evidence passage resolved to index 1 (s1), NOT index 0 (other_doc)
        grounded_ev_passage = next((p for p in g.passages.values() if p.document == "s1" and p.chunk_index == 1), None)
        self.assertIsNotNone(grounded_ev_passage, "Evidence passage should resolve to index 1 matching document s1")
        self.assertEqual(grounded_ev_passage.text, "Target text for s1 at index 1: Hamiltonian matrix formulation")

        for p in g.passages.values():
            if p.document == "s1":
                self.assertNotEqual(p.chunk_index, 0)
                self.assertNotIn("Unrelated noise", p.text)

    def test_card_evidence_grounding_handles_missing_document(self):
        """Probes develop() when card.document is completely missing from deps.texts."""
        deps, log = world()
        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        s = TransferSettings()
        g = ground("q", deps, s, ch)

        deps.search = lambda q, k, docs, excl: []

        ghost_card = Card(
            card_id="card_ghost",
            document="ghost_doc_absent",
            citation="Ghost Citation",
            name="Ghost Technique",
            problem_solved="Ghost solve",
            context="Ghost context",
            inputs="x",
            outputs="y",
            assumptions=[],
            evidence=Premise(claim="ghost fact", key="C99", document="ghost_doc_absent", chunk_index=0),
        )
        seed = Seed(bottleneck_id="B1", card=ghost_card, score=0.85, level="broad", rank=1)

        t = develop(seed, 1, g, deps, s, ch)
        self.assertIsNotNone(t, "develop should handle missing card document gracefully using near passages")
        self.assertEqual(t.card_document, "ghost_doc_absent")

    def test_stage6_revision_updates_transfer(self):
        """Probes Stage 6 revision applying revised title, summary, prediction, mapping, and steps."""
        from research_assistant.transfer import stages

        revise_payload = json.dumps({
            "responses": [{"objection": 0, "response": "Boundary effect resolved via damping", "concede": False}],
            "revised_title": "Damped Matrix Propagation",
            "revised_summary": "Summary updated with rigorous damping constraints",
            "revised_prediction": "Prediction updated: power-law scaling holds above threshold",
            "revised_mapping": [
                {"source_role": "Damping factor", "target_role": "Boundary absorption", "assumption": "Linear dissipation"}
            ],
            "revised_steps": [
                {"step_num": 1, "action": "Initialize damped lattice", "verification": "Check conservation law"}
            ],
        })

        deps, log = world(replies={("revise", 1): revise_payload})
        deps.critic = lambda m: json.dumps({
            "shape_match": "yes",
            "rows": [{"row": 0, "status": "breaks", "reason": "Damping missing"}],
            "objections": ["Damping missing causes divergence"],
        })

        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        s = TransferSettings()
        g = stages.ground("q", deps, s, ch)
        seeds, _ = stages.candidate_cards(g, deps, s)

        t = stages.develop(seeds[0], 1, g, deps, s, ch)

        self.assertIsNotNone(t)
        self.assertEqual(t.title, "Damped Matrix Propagation")
        self.assertEqual(t.summary, "Summary updated with rigorous damping constraints")
        self.assertEqual(t.prediction, "Prediction updated: power-law scaling holds above threshold")
        self.assertEqual(len(t.mapping), 1)
        self.assertEqual(t.mapping[0].source_role, "Damping factor")
        self.assertEqual(len(t.steps), 1)
        self.assertEqual(t.steps[0].action, "Initialize damped lattice")

    def test_prior_art_temporal_cutoff_year_none(self):
        """Probes prior art checking when before_year is None, hit year is None, or invalid."""
        openalex_payload = {
            "results": [
                {
                    "display_name": "Paper 2020",
                    "publication_year": 2020,
                    "doi": "https://doi.org/10.1000/1",
                    "abstract_inverted_index": {"test": [0]},
                },
                {
                    "display_name": "Paper No Year",
                    "publication_year": None,
                    "publication_date": None,
                    "doi": "https://doi.org/10.1000/2",
                    "abstract_inverted_index": {"sample": [0]},
                },
                {
                    "display_name": "Paper 2025",
                    "publication_year": 2025,
                    "doi": "https://doi.org/10.1000/3",
                    "abstract_inverted_index": {"future": [0]},
                },
            ]
        }

        arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>arXiv Paper 2022</title>
                <published>2022-05-01T00:00:00Z</published>
                <summary>Abstract 2022</summary>
                <id>http://arxiv.org/abs/2205.00001</id>
            </entry>
            <entry>
                <title>arXiv Paper Malformed Date</title>
                <published>unknown_date</published>
                <summary>Abstract no date</summary>
                <id>http://arxiv.org/abs/2205.00002</id>
            </entry>
        </feed>
        """

        client = PriorArtClient(
            get_json=lambda url, params: openalex_payload,
            get_text=lambda url, params: arxiv_xml,
            embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        )

        # 1. When before_year=2024: only hits with valid year < 2024 kept; None years excluded without TypeError
        res_cutoff = client.check("Technique", ["conductance"], "Summary text", before_year=2024)
        for h in res_cutoff.hits:
            self.assertIsNotNone(h["year"])
            self.assertLess(h["year"], 2024)

        # 2. When before_year is None: top scored hits returned regardless of year presence
        res_all = client.check("Technique", ["conductance"], "Summary text", before_year=None)
        self.assertTrue(1 <= len(res_all.hits) <= 3, "Prior art returns top scored hits (up to 3) without filtering")



if __name__ == "__main__":
    unittest.main()
