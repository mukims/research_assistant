"""
Unit tests for Paper Workspace Manager & Markdown Bundle Generator (workspace.py).
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
)
from research_assistant.tasks.workspace import (
    get_paper_workspace_dir,
    materialize_deconstructed_paper_bundle,
    materialize_paper_upload,
    render_paper_summary_markdown,
)
from research_assistant.tasks.extractor import deconstruct_paper_into_task_graph


SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Autonomous Multi-Agent Materials Discovery</title>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We present an autonomous agentic framework for materials science.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1. Methodology and Experimental Design</head>
        <p>We formulate crystal structure prediction using equivariant graph neural networks.</p>
      </div>
      <figure type="table">
        <figDesc>Table 1: Target formation energies and band gap benchmarks.</figDesc>
      </figure>
    </body>
  </text>
</TEI>
"""


def _sample_graph() -> TaskGraph:
    g = TaskGraph(paper_id="test_discovery_2026", paper_title="Materials Discovery 2026")
    t1 = Task(
        task_id="t1_env",
        title="Setup Environment",
        category=TaskCategory.ENVIRONMENT_SETUP,
        assigned_role=AgentRole.DEVOPS_AGENT,
        outputs=["env.lock"],
        description="Configure runtime dependencies and lockfiles.",
    )
    t2 = Task(
        task_id="t2_model",
        title="Synthesize Graph Network",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["t1_env"],
        outputs=["src/model.py"],
        description="Implement equivariant neural network message passing.",
    )
    g.add_task(t1)
    g.add_task(t2)
    return g


class TestTaskWorkspace(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_paper_workspace_dir(self):
        ws = get_paper_workspace_dir("2608.11188v1", base_dir=self.tmp_dir)
        self.assertTrue(os.path.isdir(ws))
        self.assertTrue(ws.endswith("2608_11188v1"))

        # Test dirty characters sanitized
        ws_dirty = get_paper_workspace_dir("Paper: Sub-Domain & Wave / Tests!", base_dir=self.tmp_dir)
        self.assertTrue(os.path.isdir(ws_dirty))
        self.assertNotIn(":", ws_dirty)
        self.assertNotIn("!", ws_dirty)
        self.assertNotIn("/", os.path.basename(ws_dirty))

    def test_materialize_paper_upload_bytes(self):
        pdf_bytes = b"%PDF-1.4 sample pdf content for unit testing"
        ws = materialize_paper_upload(
            source=pdf_bytes,
            filename="superconductivity_nature2026.pdf",
            base_dir=self.tmp_dir,
        )
        self.assertTrue(os.path.isdir(ws))

        # Check uploaded file is copied into root dir
        self.assertTrue(os.path.exists(os.path.join(ws, "superconductivity_nature2026.pdf")))

        # Check README.md and paper_summary.md exist
        readme_path = os.path.join(ws, "README.md")
        self.assertTrue(os.path.exists(readme_path))
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_text = f.read()
        self.assertIn("superconductivity_nature2026", readme_text)
        self.assertIn("paper_summary.md", readme_text)

        summary_path = os.path.join(ws, "paper_summary.md")
        self.assertTrue(os.path.exists(summary_path))
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
        self.assertIn("# Paper Summary:", summary_text)

    def test_materialize_deconstructed_paper_bundle(self):
        graph = _sample_graph()
        context = {
            "abstract": "Testing paper abstract.",
            "sections": ["## 1. Hamiltonian\nH = sum t_ij c_i^dag c_j"],
            "tables": ["Table 1: Ground state energies"],
        }
        manifest = materialize_deconstructed_paper_bundle(
            graph=graph,
            workspace_dir=self.tmp_dir,
            context=context,
        )

        ws_dir = os.path.join(self.tmp_dir, graph.paper_id)
        self.assertTrue(os.path.isdir(ws_dir))

        # Check all required markdown files exist
        expected_md_files = [
            "plan.md",
            "dag.md",
            "scorecard.md",
            "paper_summary.md",
            "rationale.md",
            "README.md",
            "rationales/t1_env_rationale.md",
            "rationales/t2_model_rationale.md",
        ]
        for f in expected_md_files:
            file_path = os.path.join(ws_dir, f)
            self.assertTrue(os.path.exists(file_path), f"Missing expected file: {f}")
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()
                self.assertGreater(len(content.strip()), 10)

        # Check machine-readable JSON specs also materialized
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "plan.json")))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "antigravity_specs.json")))

        # Verify manifest content
        self.assertIn("plan.md", manifest)
        self.assertIn("dag.md", manifest)
        self.assertIn("scorecard.md", manifest)
        self.assertIn("paper_summary.md", manifest)
        self.assertIn("rationale.md", manifest)
        self.assertIn("README.md", manifest)

    def test_deconstruct_paper_automatically_materializes_bundle(self):
        # Write temporary TEI file
        tei_path = os.path.join(self.tmp_dir, "sample_paper.tei.xml")
        with open(tei_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TEI_XML)

        target_ws = os.path.join(self.tmp_dir, "auto_ws")
        graph = deconstruct_paper_into_task_graph(
            tei_or_path=tei_path,
            paper_id="mat_disc_test",
            force_heuristic=True,
            workspace_dir=target_ws,
        )

        paper_ws = os.path.join(target_ws, "mat_disc_test")
        self.assertTrue(os.path.isdir(paper_ws))

        # Verify all markdown files dropped
        self.assertTrue(os.path.exists(os.path.join(paper_ws, "plan.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_ws, "dag.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_ws, "scorecard.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_ws, "paper_summary.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_ws, "rationale.md")))
        self.assertTrue(os.path.exists(os.path.join(paper_ws, "README.md")))
        self.assertTrue(os.path.isdir(os.path.join(paper_ws, "rationales")))


if __name__ == "__main__":
    unittest.main()
