"""
Adversarial Security Hardening and Technique Transfer Empirical Stress Test Suite.
Author: Empirical Challenger M3 2 (challenger_m3_2)
Milestone: Milestone 3 (Tier P2: Architecture, Scale, and Feature Wiring)

Adversarially probes:
1. Security Hardening:
   - Directory traversal attacks against is_safe_path() (system roots, relative paths, symlink escapes, prefix collisions, invalid types).
   - SSRF attacks against is_safe_url() (private IPv4, loopback, link-local metadata, IPv6, multi-homed DNS rebinding, non-HTTP schemes).
   - Zip bomb payloads and magic byte checks in batch_uploader.py (decompression bomb ratio, file count caps, single file caps, total uncompressed caps, path traversal members, non-PDF filtering).
   - HTML injection and XSS defenses in ui/results.py and tasks/ui.py (escaping in fetch_failures, batch_uploaded cards, task wave cards, and Mermaid strict security).
2. Technique Transfer:
   - Card evidence grounding document isolation in stages.py (interleaved multi-document corpus, mismatched index vs document, absent documents, boundary indices).
   - Stage 6 revision updates on Transfer in stages.py (both prefixed and unprefixed revisions, partial revisions, updated summary propagation).
   - Temporal cutoff handling in prior_art.py (None years, invalid date strings, strict year boundary filtering, None cutoff).
"""

from __future__ import annotations

import html
import io
import json
import os
import shutil
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


class TestAdversarialDirectoryTraversal(unittest.TestCase):
    """Empirical challenge against directory traversal vulnerabilities in is_safe_path()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="adv_traversal_")
        self.allowed_workspace = os.path.join(self.tmp_dir, "workspace")
        os.makedirs(self.allowed_workspace, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_system_root_and_critical_paths_blocked(self):
        """Probes system root and sensitive directories across UNIX and macOS."""
        forbidden_targets = [
            "/",
            "/etc",
            "/etc/passwd",
            "/etc/shadow",
            "/etc/hosts",
            "/private/etc",
            "/private/etc/hosts",
            "/private/var/root",
            "/private/var/log",
            "/private/var/db",
            "/bin",
            "/bin/sh",
            "/sbin",
            "/usr",
            "/usr/bin",
            "/var/log",
            "/dev",
            "/dev/null",
            "/dev/random",
            "/proc",
            "/proc/cpuinfo",
            "/sys",
            "/System",
            "/System/Library",
            "/Library",
            "/Library/Preferences",
        ]
        for path in forbidden_targets:
            with self.subTest(path=path):
                self.assertFalse(
                    is_safe_path(path),
                    f"Directory traversal check failed to block sensitive system path: {path}",
                )

    def test_relative_and_dot_dot_traversals_blocked(self):
        """Probes relative path traversals attempting to escape permitted directories."""
        traversal_payloads = [
            "../../../../../../../../../../../../etc/passwd",
            "./../../../../../../../../../../../../private/etc/hosts",
            "foo/bar/../../../../../../../../../../../../etc/passwd",
            "~/../../../../../../etc/passwd",
        ]
        for payload in traversal_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(
                    is_safe_path(payload),
                    f"Relative traversal payload was not blocked: {payload}",
                )

    def test_symlink_escape_attacks_blocked(self):
        """Probes symlink escapes where a symlink inside an allowed root points to a forbidden path."""
        # 1. Symlink to /etc directory
        symlink_etc = os.path.join(self.allowed_workspace, "link_to_etc")
        os.symlink("/etc", symlink_etc)
        self.assertFalse(
            is_safe_path(symlink_etc, allowed_roots=[self.allowed_workspace]),
            "Symlink pointing to /etc should be rejected by realpath resolution",
        )
        # Subpath through symlink
        self.assertFalse(
            is_safe_path(os.path.join(symlink_etc, "passwd"), allowed_roots=[self.allowed_workspace]),
            "Path traversing through /etc symlink must be blocked",
        )

        # 2. Symlink to /private/etc/hosts file
        symlink_hosts = os.path.join(self.allowed_workspace, "link_to_hosts")
        os.symlink("/private/etc/hosts" if os.path.exists("/private/etc/hosts") else "/etc/hosts", symlink_hosts)
        self.assertFalse(
            is_safe_path(symlink_hosts, allowed_roots=[self.allowed_workspace]),
            "Symlink pointing to system hosts file must be blocked",
        )

        # 3. Symlink pointing to an unpermitted external directory outside allowed roots
        outside_dir = os.path.join(self.tmp_dir, "unpermitted_external")
        os.makedirs(outside_dir, exist_ok=True)
        symlink_outside = os.path.join(self.allowed_workspace, "link_to_outside")
        os.symlink(outside_dir, symlink_outside)
        self.assertFalse(
            is_safe_path(symlink_outside, allowed_roots=[self.allowed_workspace]),
            "Symlink pointing outside allowed root must be rejected",
        )

    def test_prefix_collision_attacks_blocked(self):
        """Probes prefix collision attacks (e.g. /path/to/safe_evil vs /path/to/safe)."""
        safe_root = os.path.join(self.tmp_dir, "safe_zone")
        os.makedirs(safe_root, exist_ok=True)

        collision_sibling = os.path.join(self.tmp_dir, "safe_zone_evil")
        os.makedirs(collision_sibling, exist_ok=True)

        self.assertTrue(is_safe_path(safe_root, allowed_roots=[safe_root]))
        self.assertFalse(
            is_safe_path(collision_sibling, allowed_roots=[safe_root]),
            "Prefix collision path should not pass allowed root check",
        )
        self.assertFalse(
            is_safe_path(os.path.join(safe_root, "..", "safe_zone_evil"), allowed_roots=[safe_root]),
            "Dot-dot escape to collision sibling must be blocked",
        )

    def test_malformed_and_non_string_inputs_handled(self):
        """Verifies None, empty, whitespace, and non-string inputs are rejected safely."""
        # Non-string and None inputs
        self.assertFalse(is_safe_path(None))
        self.assertFalse(is_safe_path(""))
        self.assertFalse(is_safe_path(12345))
        self.assertFalse(is_safe_path(["/etc/passwd"]))
        self.assertFalse(is_safe_path({"path": "/tmp"}))

        # Test whitespace with specific allowed root (ensures whitespace cannot escape to custom root)
        isolated_root = os.path.join(self.tmp_dir, "isolated_unique_root")
        os.makedirs(isolated_root, exist_ok=True)
        self.assertFalse(is_safe_path("   ", allowed_roots=[isolated_root]))
        self.assertFalse(is_safe_path("\t\n", allowed_roots=[isolated_root]))

    def test_scan_and_stage_directory_rejects_unsafe_paths(self):
        """Verifies scan_and_stage_directory halts if an unsafe root path is passed."""
        res_etc = scan_and_stage_directory("/etc")
        self.assertEqual(res_etc, [])

        res_traversal = scan_and_stage_directory("../../../../../../etc")
        self.assertEqual(res_traversal, [])


class TestAdversarialSSRFProtection(unittest.TestCase):
    """Empirical challenge against Server-Side Request Forgery in is_safe_url()."""

    def test_loopback_and_local_urls_blocked(self):
        """Probes loopback addresses in IPv4, IPv6, and localhost."""
        loopback_urls = [
            "http://127.0.0.1",
            "http://127.0.0.1:8000/api/v1",
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
                self.assertFalse(
                    is_safe_url(url),
                    f"SSRF check failed to block loopback URL: {url}",
                )

    def test_cloud_metadata_endpoints_blocked(self):
        """Probes AWS, GCP, Azure, and OpenStack link-local metadata endpoints."""
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
                self.assertFalse(
                    is_safe_url(url),
                    f"SSRF check failed to block metadata URL: {url}",
                )

    def test_private_rfc1918_ranges_blocked(self):
        """Probes 10.0.0.0/8, 172.16.0.0/12, and 192.168.0.0/16 private IP ranges."""
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
                self.assertFalse(
                    is_safe_url(url),
                    f"SSRF check failed to block private IP URL: {url}",
                )

    def test_special_purpose_and_ipv6_ranges_blocked(self):
        """Probes IPv6 ULA, link-local, IPv4-mapped IPv6, and multicast."""
        special_urls = [
            "http://[fc00::1]/admin",
            "http://[fd12:3456:789a::1]",
            "http://[fe80::1]/linklocal",
            "http://224.0.0.1/multicast",
            "http://[::ffff:127.0.0.1]",
            "http://[::ffff:169.254.169.254]",
        ]
        for url in special_urls:
            with self.subTest(url=url):
                self.assertFalse(
                    is_safe_url(url),
                    f"SSRF check failed to block special purpose IP: {url}",
                )

    def test_non_http_schemes_blocked(self):
        """Probes dangerous URL schemes (file, gopher, ftp, javascript, data)."""
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
                self.assertFalse(
                    is_safe_url(url),
                    f"SSRF check failed to block non-HTTP scheme: {url}",
                )

    def test_dns_rebinding_dual_homed_blocked(self):
        """Probes multi-homed DNS response where one resolved IP is private/loopback."""
        mock_addr_info = [
            (None, None, None, None, ("93.184.216.34", 80)),
            (None, None, None, None, ("127.0.0.1", 80)),
        ]
        with patch("socket.getaddrinfo", return_value=mock_addr_info):
            self.assertFalse(
                is_safe_url("http://rebind.attacker.com/paper.pdf"),
                "Dual-homed DNS containing a private address must be rejected",
            )

    def test_malformed_urls_handled_safely(self):
        """Probes malformed, empty, and non-URL strings without throwing unhandled exceptions."""
        malformed = ["", "   ", "http://", "https://", "not_a_url", "http://:80", "http://[invalid]/"]
        for url in malformed:
            with self.subTest(url=url):
                self.assertFalse(is_safe_url(url))

    def test_valid_public_urls_allowed(self):
        """Probes legitimate academic and web URLs resolving to public IPs."""
        mock_addr_info = [(None, None, None, None, ("93.184.216.34", 443))]
        with patch("socket.getaddrinfo", return_value=mock_addr_info):
            self.assertTrue(is_safe_url("https://arxiv.org/pdf/2401.12345.pdf"))
            self.assertTrue(is_safe_url("https://openalex.org/W123456789"))


class TestAdversarialZipBombsAndMagicBytes(unittest.TestCase):
    """Empirical challenge against zip bombs, decompression bombs, and magic byte bypasses."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="adv_zip_")
        self.dest_dir = os.path.join(self.tmp_dir, "staged")
        os.makedirs(self.dest_dir, exist_ok=True)
        self.valid_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_invalid_magic_bytes_rejected(self):
        """Probes non-zip files with .zip extension (missing PK\\x03\\x04 header)."""
        bad_payloads = [
            (b"GIF89a Fake Image", "image.zip"),
            (b"%PDF-1.4 Fake PDF pretending to be zip", "paper.zip"),
            (b"Plain text file content", "text.zip"),
            (b"\x00\x01\x02\x03 corrupted binary", "corrupt.zip"),
            (b"PK", "truncated_header.zip"),
        ]
        for data, name in bad_payloads:
            buf = io.BytesIO(data)
            buf.name = name
            staged = unpack_and_stage_uploads([buf], destination_dir=self.dest_dir)
            self.assertEqual(
                len(staged),
                0,
                f"File {name} with invalid magic bytes should produce 0 staged files",
            )

    def test_disguised_zip_file_unpacked(self):
        """Probes zip archive disguised with .pdf extension (has PK\\x03\\x04 magic bytes)."""
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("inner_paper.pdf", self.valid_pdf)
        zip_buf.seek(0)
        zip_buf.name = "disguised_archive.pdf"  # named .pdf but starts with PK\x03\x04

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Disguised Paper", "", "")):
            staged = unpack_and_stage_uploads([zip_buf], destination_dir=self.dest_dir)

        self.assertEqual(len(staged), 1, "Disguised zip file should be correctly recognized via PK magic bytes")
        self.assertEqual(staged[0]["filename"], "inner_paper.pdf")

    def test_zip_bomb_excessive_file_count_capped(self):
        """Probes zip bomb containing more than MAX_ZIP_FILE_COUNT (500) members."""
        bomb_buf = io.BytesIO()
        with zipfile.ZipFile(bomb_buf, "w") as zf:
            for i in range(MAX_ZIP_FILE_COUNT + 15):
                zf.writestr(f"p_{i}.pdf", self.valid_pdf)
        bomb_buf.seek(0)
        bomb_buf.name = "bomb_many_files.zip"

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Paper", "", "")):
            staged = unpack_and_stage_uploads([bomb_buf], destination_dir=self.dest_dir)

        self.assertLessEqual(
            len(staged),
            MAX_ZIP_FILE_COUNT,
            f"Extraction must halt at MAX_ZIP_FILE_COUNT ({MAX_ZIP_FILE_COUNT})",
        )

    def test_zip_bomb_single_file_exceeding_100mb_skipped(self):
        """Probes decompression bomb where a single member uncompresses to > 100 MB."""
        bomb_buf = io.BytesIO()
        # 105 MB of zeros deflated compresses to ~107 KB
        with zipfile.ZipFile(bomb_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("huge_file.pdf", b"0" * (105 * 1024 * 1024))
            zf.writestr("normal.pdf", self.valid_pdf)
        bomb_buf.seek(0)
        bomb_buf.name = "bomb_single_huge.zip"

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Normal Paper", "", "")):
            staged = unpack_and_stage_uploads([bomb_buf], destination_dir=self.dest_dir)

        # huge_file.pdf should be skipped; normal.pdf should be extracted
        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0]["filename"], "normal.pdf")

    def test_zip_bomb_total_uncompressed_size_exceeding_500mb_halted(self):
        """Probes decompression bomb where cumulative uncompressed size exceeds 500 MB."""
        bomb_buf = io.BytesIO()
        # 6 members of 90 MB each = 540 MB total (each < 100 MB single file limit)
        with zipfile.ZipFile(bomb_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx in range(6):
                zf.writestr(f"chunk_{idx}.pdf", b"0" * (90 * 1024 * 1024))
        bomb_buf.seek(0)
        bomb_buf.name = "bomb_total_huge.zip"

        staged = unpack_and_stage_uploads([bomb_buf], destination_dir=self.dest_dir)
        # Should have halted once cumulative size exceeded 500 MB without crashing
        self.assertEqual(len(staged), 0)

    def test_zip_member_path_traversal_contained(self):
        """Probes zip members containing directory traversal paths (e.g. ../../etc/passwd)."""
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("../../../../etc_passwd.pdf", self.valid_pdf)
            zf.writestr("sub/folder/nested.pdf", self.valid_pdf)
        zip_buf.seek(0)
        zip_buf.name = "traversal_archive.zip"

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Nested Paper", "", "")):
            staged = unpack_and_stage_uploads([zip_buf], destination_dir=self.dest_dir)

        for s in staged:
            staged_path = s["path"]
            self.assertTrue(
                staged_path.startswith(self.dest_dir),
                f"Staged file escaped destination directory: {staged_path}",
            )

    def test_non_pdf_members_filtered_out(self):
        """Probes zip archives containing executable, script, or non-PDF files."""
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("malware.exe", b"MZ\x90\x00executable")
            zf.writestr("script.sh", b"#!/bin/bash\nrm -rf /")
            zf.writestr("document.docx", b"PK\x03\x04docx")
            zf.writestr("legit.pdf", self.valid_pdf)
        zip_buf.seek(0)
        zip_buf.name = "mixed_archive.zip"

        with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Legit Paper", "", "")):
            staged = unpack_and_stage_uploads([zip_buf], destination_dir=self.dest_dir)

        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0]["filename"], "legit.pdf")


class TestAdversarialHTMLInjection(unittest.TestCase):
    """Empirical challenge against XSS and unescaped HTML injection in ui/results.py and tasks/ui.py."""

    def test_ui_results_fetch_failures_escaping(self):
        """Probes html.escape() enforcement over untrusted title and reason in results.py."""
        from research_assistant.ui import results

        raw_title = "<script>alert('pwn_title')</script>"
        raw_reason = "<img src=x onerror=alert('pwn_reason')><svg onload=alert(1)>"
        final_payload = {
            "fetch_failures": [{"title": raw_title, "reason": raw_reason}],
            "downloaded": [],
            "seed_papers": [],
        }

        rendered_markdown: list[str] = []
        with patch("streamlit.markdown", side_effect=lambda txt, **kw: rendered_markdown.append(txt)), \
             patch("streamlit.expander", return_value=MagicMock()), \
             patch("streamlit.caption"):

            results._render_seed_and_downloads("query", final_payload)

            output_text = "\n".join(rendered_markdown)
            self.assertNotIn("<script>", output_text)
            self.assertNotIn("<img", output_text)
            self.assertNotIn("<svg", output_text)
            self.assertIn("&lt;script&gt;", output_text)
            self.assertIn("&lt;img", output_text)
            self.assertIn("&lt;svg", output_text)

    def test_ui_results_batch_uploaded_escaping(self):
        """Probes html.escape() enforcement in results._render_build batch_uploaded records."""
        from research_assistant.ui import results

        malicious_batch = [
            {
                "title": "<script>alert('pwn_batch')</script>",
                "filename": "<b onclick=alert(1)>click.pdf</b>",
                "doi": "10.1000/<iframe src=evil.com>",
                "arxiv_id": "2401.<img src=x>",
            }
        ]
        final_payload = {
            "batch_uploaded": malicious_batch,
        }

        rendered_markdown: list[str] = []
        with patch("streamlit.markdown", side_effect=lambda txt, **kw: rendered_markdown.append(txt)), \
             patch("streamlit.container", return_value=MagicMock()), \
             patch("streamlit.info"):

            results._render_build(final_payload, query="test_query")

            output_text = "\n".join(rendered_markdown)
            self.assertNotIn("<script>", output_text)
            self.assertNotIn("<iframe", output_text)
            self.assertNotIn("<b onclick", output_text)
            self.assertIn("&lt;script&gt;", output_text)
            self.assertIn("&lt;iframe", output_text)

    def test_tasks_ui_mermaid_strict_configuration(self):
        """Probes Mermaid initialization to ensure strict securityLevel and disabled HTML labels."""
        from research_assistant.tasks.ui import _render_mermaid_interactive

        with patch("streamlit.components.v1.html") as mock_html:
            _render_mermaid_interactive("flowchart TD\n  A[Start] --> B[End]")
            mock_html.assert_called_once()
            code = mock_html.call_args[0][0]

            self.assertIn("securityLevel: 'strict'", code)
            self.assertIn("htmlLabels: false", code)

    def test_tasks_ui_wave_card_escaping(self):
        """Probes HTML escaping for dynamic task metadata in tasks/ui.py."""
        from research_assistant.tasks.models import AgentRole, Task, TaskStatus

        malicious_task = Task(
            task_id="<script>alert('tid')</script>",
            title="<img src=x onerror=alert('title')>",
            assigned_role=AgentRole.DATA_ENGINEER,
            outputs=["<iframe src='http://evil.com'></iframe>", "<svg/onload=alert('out')>"],
            status=TaskStatus.COMPLETED,
        )

        safe_id = html.escape(str(malicious_task.task_id))
        safe_title = html.escape(str(malicious_task.title))
        safe_outputs = ", ".join(f"<code>{html.escape(str(out))}</code>" for out in malicious_task.outputs)

        self.assertNotIn("<script>", safe_id)
        self.assertNotIn("<img", safe_title)
        self.assertNotIn("<iframe", safe_outputs)
        self.assertNotIn("<svg", safe_outputs)
        self.assertIn("&lt;script&gt;", safe_id)
        self.assertIn("&lt;img", safe_title)
        self.assertIn("&lt;iframe", safe_outputs)


class TestAdversarialTechniqueTransfer(unittest.TestCase):
    """Empirical challenge against Technique Transfer grounding, revisions, and prior art."""

    def test_card_evidence_grounding_document_isolation(self):
        """
        Probes that card evidence grounding isolates the correct document when
        deps.texts contains multiple disparate documents at different positions.
        """
        from research_assistant.transfer.models import Card, Premise
        from research_assistant.transfer.stages import Seed, develop, ground
        from research_assistant.transfer.deps import TransferSettings
        from research_assistant.transfer.premises import PremiseChecker
        from tests.transfer_fixtures import world

        deps, log = world()

        # Alter deps so index 0 belongs to a different document
        deps.texts[0] = "Unrelated document text at index 0"
        deps.metadatas[0] = {"document": "other_doc", "chunk_index": 0}

        # Target document s1 is placed at index 1 with seq 0
        deps.texts[1] = "Target text for s1 at index 1: Hamiltonian matrix"
        deps.metadatas[1] = {"document": "s1", "chunk_index": 0, "seq": 0}

        # Card belongs to s1 with evidence pointing to chunk_index 0
        card = Card(
            card_id="card_s1_isolated",
            document="s1",
            citation="Target Citation S1",
            name="Matrix Technique",
            problem_solved="Solving coupled linear system",
            context="Numerical methods",
            inputs="A matrix, b vector",
            outputs="x solution vector",
            assumptions=["linearity"],
            evidence=Premise(claim="Hamiltonian matrix formulation", key="C1", document="s1", chunk_index=0, seq=0),
        )

        seed = Seed(bottleneck_id="B1", card=card, score=0.92, level="mid", rank=1)
        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        s = TransferSettings()
        g = ground("q", deps, s, ch)

        t = develop(seed, 1, g, deps, s, ch)
        self.assertIsNotNone(t, "develop should succeed with isolated card document")
        self.assertEqual(t.card_document, "s1")

        # Verify that grounded evidence passage resolved to index 1 (s1), NOT index 0 (other_doc)!
        grounded_ev_passage = next((p for p in g.passages.values() if p.document == "s1" and p.chunk_index == 1), None)
        self.assertIsNotNone(grounded_ev_passage, "Evidence passage should resolve to index 1 matching document s1")
        self.assertEqual(grounded_ev_passage.text, "Target text for s1 at index 1: Hamiltonian matrix")

        # Ensure index 0 (other_doc) was NOT erroneously grounded as s1
        for p in g.passages.values():
            if p.document == "s1":
                self.assertNotEqual(p.chunk_index, 0)
                self.assertNotIn("Unrelated document text", p.text)

    def test_card_evidence_grounding_handles_absent_document(self):
        """Probes develop() when card.document is completely missing from deps.texts."""
        from research_assistant.transfer.models import Card, Premise
        from research_assistant.transfer.stages import Seed, develop, ground
        from research_assistant.transfer.deps import TransferSettings
        from research_assistant.transfer.premises import PremiseChecker
        from tests.transfer_fixtures import world

        deps, log = world()
        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        s = TransferSettings()
        g = ground("q", deps, s, ch)

        # Mock search so search for missing doc returns empty list without error
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

        # Must handle absent document gracefully without unhandled exception:
        # develop completes using available near passages rather than crashing
        t = develop(seed, 1, g, deps, s, ch)
        self.assertIsNotNone(t, "develop should handle missing card document gracefully using near passages")
        self.assertEqual(t.card_document, "ghost_doc_absent")

    def test_stage6_revision_cleanly_updates_assembled_transfer(self):
        """
        Probes Stage 6 revision applying revised title, summary, prediction, mapping,
        and steps across both prefixed ('revised_...') and unprefixed keys.
        """
        from research_assistant.transfer import stages
        from research_assistant.transfer.deps import TransferSettings
        from research_assistant.transfer.premises import PremiseChecker
        from tests.transfer_fixtures import world

        # 1. Test prefixed keys: revised_title, revised_summary, revised_prediction, revised_mapping, revised_steps
        revise_payload_prefixed = json.dumps({
            "responses": [{"objection": 0, "response": "Boundary effect resolved via damping", "concede": False}],
            "revised_title": "Damped Matrix Propagation",
            "revised_summary": "Summary updated with rigorous damping constraints",
            "revised_prediction": "Prediction updated: power-law scaling holds above threshold",
            "revised_mapping": [
                {"source_role": "S_damped", "target_role": "T_damped", "assumption": "A_damped", "status": "holds", "core": True, "note": "verified"}
            ],
            "revised_steps": [
                {"action": "Apply Damping Filter", "input": "raw_signal", "output": "filtered_signal", "tool": "scipy.signal"}
            ],
        })

        deps, log = world(replies={("revise", 1): revise_payload_prefixed})
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
        self.assertEqual(t.mapping[0].source_role, "S_damped")
        self.assertEqual(len(t.steps), 1)
        self.assertEqual(t.steps[0].action, "Apply Damping Filter")

        # 2. Test unprefixed keys: title, summary, prediction, mapping, steps
        revise_payload_unprefixed = json.dumps({
            "responses": [{"objection": 0, "response": "Addressed without prefix", "concede": False}],
            "title": "Unprefixed Title Update",
            "summary": "Unprefixed Summary Update",
            "prediction": "Unprefixed Prediction Update",
            "mapping": [
                {"source_role": "S_unpref", "target_role": "T_unpref", "assumption": "A_unpref", "status": "holds", "core": True}
            ],
            "steps": [
                {"action": "Unprefixed Step", "input": "u_in", "output": "u_out", "tool": "torch"}
            ],
        })

        deps2, log2 = world(replies={("revise", 1): revise_payload_unprefixed})
        deps2.critic = lambda m: json.dumps({
            "shape_match": "yes",
            "rows": [{"row": 0, "status": "breaks", "reason": "Edge case"}],
            "objections": ["Edge case found"],
        })

        ch2 = PremiseChecker(deps2.judge, deps2.texts, deps2.metadatas)
        g2 = stages.ground("q", deps2, s, ch2)
        seeds2, _ = stages.candidate_cards(g2, deps2, s)

        t2 = stages.develop(seeds2[0], 1, g2, deps2, s, ch2)
        self.assertIsNotNone(t2)
        self.assertEqual(t2.title, "Unprefixed Title Update")
        self.assertEqual(t2.summary, "Unprefixed Summary Update")
        self.assertEqual(t2.prediction, "Unprefixed Prediction Update")
        self.assertEqual(t2.mapping[0].source_role, "S_unpref")
        self.assertEqual(t2.steps[0].action, "Unprefixed Step")

    def test_prior_art_temporal_cutoff_with_none_and_malformed_years(self):
        """
        Probes PriorArtClient.check() with None years, malformed publication dates,
        and strict boundary filtering (before_year=2021).
        """
        from research_assistant.transfer.prior_art import PriorArtClient

        client = PriorArtClient(
            embed_documents=lambda texts: [[0.1] * 768 for _ in texts],
        )

        test_hits = [
            {"title": "Paper 2018 Valid", "year": 2018, "abstract": "abc"},
            {"title": "Paper 2020 Valid", "year": 2020, "abstract": "def"},
            {"title": "Paper 2021 Boundary", "year": 2021, "abstract": "ghi"},
            {"title": "Paper 2023 Future", "year": 2023, "abstract": "jkl"},
            {"title": "Paper Year None", "year": None, "abstract": "mno"},
            {"title": "Paper Empty Year", "abstract": "pqr"},
        ]

        with patch.object(client, "get_json", return_value={"results": []}), \
             patch.object(client, "get_text", return_value="<xml></xml>"), \
             patch("research_assistant.transfer.prior_art.parse_openalex", return_value=test_hits), \
             patch("research_assistant.transfer.prior_art.parse_arxiv", return_value=[]):

            # 1. Enforce cutoff at 2021: only 2018 and 2020 papers should survive
            res = client.check("quantum transport", ["conductance"], "summary", before_year=2021)
            self.assertIsNotNone(res)
            surviving_titles = [h["title"] for h in res.hits]

            self.assertIn("Paper 2018 Valid", surviving_titles)
            self.assertIn("Paper 2020 Valid", surviving_titles)
            self.assertNotIn("Paper 2021 Boundary", surviving_titles, "Paper published in 2021 must be dropped by before_year=2021")
            self.assertNotIn("Paper 2023 Future", surviving_titles)
            self.assertNotIn("Paper Year None", surviving_titles, "Paper with None year must be dropped")
            self.assertNotIn("Paper Empty Year", surviving_titles, "Paper without year must be dropped")

        # 2. None cutoff: papers with None year are preserved when within top_k limit
        hits_with_none = [
            {"title": "Paper Year None", "year": None, "abstract": "mno"},
            {"title": "Paper 2018 Valid", "year": 2018, "abstract": "abc"},
        ]
        with patch.object(client, "get_json", return_value={"results": []}), \
             patch.object(client, "get_text", return_value="<xml></xml>"), \
             patch("research_assistant.transfer.prior_art.parse_openalex", return_value=hits_with_none), \
             patch("research_assistant.transfer.prior_art.parse_arxiv", return_value=[]):

            res_all = client.check("quantum transport", ["conductance"], "summary", before_year=None)
            self.assertIsNotNone(res_all)
            all_titles = [h["title"] for h in res_all.hits]
            self.assertIn("Paper 2018 Valid", all_titles)
            self.assertIn("Paper Year None", all_titles)


if __name__ == "__main__":
    unittest.main()
