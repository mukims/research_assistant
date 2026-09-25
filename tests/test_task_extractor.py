"""
Unit tests for scientific protocol miner and task extractor.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from research_assistant.tasks.models import (
    TaskGraph,
    TaskCategory,
    AgentRole,
)
from research_assistant.tasks.extractor import (
    extract_paper_planning_context,
    _heuristic_task_graph,
    deconstruct_paper_into_task_graph,
)


SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Deep Learning for Quantum State Reconstruction</title>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We present a neural network architecture for quantum tomography that scales linearly with qubit count.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1. Introduction</head>
        <p>Quantum state tomography is a foundational challenge in quantum information processing.</p>
      </div>
      <div>
        <head>2. Method and Model Architecture</head>
        <p>Our model consists of a parameterized variational autoencoder with complex-valued weights. We minimize the Kullback-Leibler divergence between predicted and observed measurement probabilities.</p>
      </div>
      <div>
        <head>3. Experimental Setup and Dataset</head>
        <p>We simulated Pauli measurement outcomes on random Haar-distributed 4-qubit states using Qiskit. The dataset contains 100,000 synthetic projective measurements.</p>
      </div>
      <figure type="table">
        <figDesc>Table 1: State fidelity comparison across 2, 4, and 8 qubit benchmarks.</figDesc>
      </figure>
    </body>
  </text>
</TEI>
"""


def test_extract_paper_planning_context_tei(tmp_path):
    tei_file = tmp_path / "paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    context = extract_paper_planning_context(str(tei_file))
    assert context["title"] == "Deep Learning for Quantum State Reconstruction"
    assert "neural network architecture for quantum tomography" in context["abstract"]
    assert any("Method and Model Architecture" in s for s in context["sections"])
    assert any("Table 1" in t for t in context["tables"])


def test_extract_paper_planning_context_plain_text(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Title: Simple Paper\nMethod: Train a transformer on dataset X.", encoding="utf-8")

    context = extract_paper_planning_context(str(txt_file))
    assert context["title"] == "notes.txt"
    assert "Train a transformer" in context["sections"][0]


def test_extract_paper_planning_context_raw_string():
    raw_text = "Raw paper content with methodology description."
    context = extract_paper_planning_context(raw_text)
    assert context["title"] == "Scientific Paper"
    assert raw_text in context["sections"][0]


def test_heuristic_task_graph():
    context = {
        "title": "Quantum Tomography",
        "abstract": "Scaling quantum tomography using neural networks.",
        "sections": ["Methodology details here."],
        "tables": ["Table 1: Benchmark numbers"],
    }
    graph = _heuristic_task_graph("quantum_01", "Quantum Tomography", context)

    assert isinstance(graph, TaskGraph)
    assert len(graph.tasks) == 5
    assert graph.validate() == []
    waves = graph.topological_waves()
    assert len(waves) >= 4

    # Verify task 01 is env
    t1 = graph.get_task("task_01_env")
    assert t1 is not None
    assert t1.category == TaskCategory.ENVIRONMENT_SETUP
    assert t1.assigned_role == AgentRole.DEVOPS_AGENT

    # Verify task 05 depends on task 04
    t5 = graph.get_task("task_05_eval")
    assert "task_04_experiment" in t5.dependencies


def test_deconstruct_paper_into_task_graph_force_heuristic(tmp_path):
    tei_file = tmp_path / "paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    graph = deconstruct_paper_into_task_graph(
        str(tei_file),
        paper_id="quantum_test",
        force_heuristic=True,
    )
    assert len(graph.tasks) == 5
    assert graph.paper_id == "quantum_test"
    assert graph.validate() == []


def test_deconstruct_paper_into_task_graph_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CITATION_TASK_HEURISTIC_ONLY", "1")
    tei_file = tmp_path / "paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    graph = deconstruct_paper_into_task_graph(str(tei_file), paper_id="env_test")
    assert len(graph.tasks) == 5
    assert graph.validate() == []


@patch("research_assistant.tasks.extractor.chat")
def test_deconstruct_paper_into_task_graph_llm_success(mock_chat, tmp_path):
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "paper_title": "Quantum Tomography AI",
        "tasks": [
            {
                "task_id": "task_1",
                "title": "Install Qiskit & PyTorch",
                "category": "environment_setup",
                "assigned_role": "devops_agent",
                "dependencies": [],
                "description": "Install dependencies",
                "inputs": ["requirements.txt"],
                "outputs": ["env.lock"],
                "acceptance_criteria": ["pip install succeeds"],
            },
            {
                "task_id": "task_2",
                "title": "Build Complex VAE",
                "category": "model_implementation",
                "assigned_role": "model_architect",
                "dependencies": ["task_1"],
                "description": "Construct VAE",
                "inputs": ["env.lock"],
                "outputs": ["vae.py"],
                "acceptance_criteria": ["Model compiles"],
            },
            {
                "task_id": "task_3",
                "title": "Evaluate State Fidelity",
                "category": "metric_evaluation",
                "assigned_role": "evaluation_agent",
                "dependencies": ["task_2"],
                "description": "Compute fidelity",
                "inputs": ["vae.py"],
                "outputs": ["fidelity.json"],
                "acceptance_criteria": ["Fidelity > 0.95"],
            }
        ]
    })
    mock_chat.return_value = mock_response

    tei_file = tmp_path / "paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    graph = deconstruct_paper_into_task_graph(str(tei_file), paper_id="llm_paper")
    assert len(graph.tasks) == 3
    assert graph.paper_title == "Quantum Tomography AI"
    assert graph.get_task("task_2").category == TaskCategory.MODEL_IMPLEMENTATION
    assert graph.get_task("task_2").assigned_role == AgentRole.MODEL_ARCHITECT
    assert graph.validate() == []


@patch("research_assistant.tasks.extractor.chat")
def test_deconstruct_paper_into_task_graph_llm_malformed_fallback(mock_chat, tmp_path):
    mock_response = MagicMock()
    mock_response.content = "Not valid JSON at all! Just raw text."
    mock_chat.return_value = mock_response

    tei_file = tmp_path / "paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    # Should fallback gracefully to heuristic without throwing
    graph = deconstruct_paper_into_task_graph(str(tei_file), paper_id="fallback_paper")
    assert len(graph.tasks) == 5
    assert graph.validate() == []


@patch("research_assistant.tasks.extractor.chat")
def test_deconstruct_paper_into_task_graph_llm_cycle_fallback(mock_chat, tmp_path):
    # If LLM returns a cyclic graph, validation fails, and it falls back to heuristic
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "paper_title": "Cyclic Paper",
        "tasks": [
            {"task_id": "A", "title": "Task A", "dependencies": ["B"]},
            {"task_id": "B", "title": "Task B", "dependencies": ["A"]},
        ]
    })
    mock_chat.return_value = mock_response

    tei_file = tmp_path / "paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    graph = deconstruct_paper_into_task_graph(str(tei_file), paper_id="cycle_fallback")
    assert len(graph.tasks) == 5
    assert graph.validate() == []


def test_extract_paper_planning_context_pdf(tmp_path):
    import fitz

    # 1. Create a synthetic PDF with title, abstract, methodology section, table/figure caption
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Autonomous Multi-Agent Systems for Quantum Tomography")
    page.insert_text((50, 100), "Abstract: We present an autonomous multi-agent system for quantum state reconstruction.")
    page.insert_text((50, 150), "1. Introduction\nQuantum tomography is foundational for computing.")
    page.insert_text((50, 200), "2. Methodology and Architecture\nWe implement variational autoencoders with PyTorch and Qiskit.")
    page.insert_text((50, 250), "Figure 1: Architecture schematic of distributed agents.")
    page.insert_text((50, 280), "Table 1: State fidelity comparison across benchmarks.")
    pdf_path = tmp_path / "quantum_paper.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # Extract via direct PyMuPDF fallback when no cached TEI XML exists
    context = extract_paper_planning_context(str(pdf_path))
    assert "Autonomous Multi-Agent Systems" in context["title"]
    assert "quantum state reconstruction" in context["abstract"].lower()
    assert any("Methodology and Architecture" in s for s in context["sections"])
    assert any("Figure 1" in t for t in context["tables"])
    assert any("Table 1" in t for t in context["tables"])

    # 2. When cached GROBID TEI XML exists on disk, it should parse via TEI XML parser
    tei_file = tmp_path / "cached.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")
    with patch("research_assistant.tasks.extractor._find_cached_tei", return_value=str(tei_file)):
        cached_context = extract_paper_planning_context(str(pdf_path))
        assert cached_context["title"] == "Deep Learning for Quantum State Reconstruction"
        assert "neural network architecture for quantum tomography" in cached_context["abstract"]


def test_extract_paper_planning_context_binary_bytes(tmp_path):
    import fitz

    # 1. Test in-memory raw bytes stream
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Stream Processed Paper")
    page.insert_text((50, 100), "Abstract: Testing stream byte processing without filesystem write.")
    page.insert_text((50, 150), "2. Experimental Setup and Evaluation\nBenchmark evaluation details.")
    raw_pdf_bytes = doc.tobytes()
    doc.close()

    context_stream = extract_paper_planning_context(raw_pdf_bytes)
    assert "Stream Processed Paper" in context_stream["title"]
    assert "stream byte processing" in context_stream["abstract"]
    assert any("Experimental Setup" in s for s in context_stream["sections"])

    # 2. Test exact crash reproduction: binary sequence containing 0x8f and null bytes
    # ('utf-8' codec can't decode byte 0x8f in position 10: invalid start byte)
    corrupted_pdf_bytes = b"%PDF-1.4\n%\x8f\x9a\x00\xff" + b"Corrupted PDF payload with binary markers\x8f\x90"
    context_corrupt = extract_paper_planning_context(corrupted_pdf_bytes)
    assert isinstance(context_corrupt, dict)
    assert "title" in context_corrupt
    assert "sections" in context_corrupt
    assert "abstract" in context_corrupt

    # 3. Test binary file on disk without .pdf extension
    bin_file = tmp_path / "paper_binary.dat"
    bin_file.write_bytes(b"%PDF-1.4\n%\x8f\x9a\x00\xff" + raw_pdf_bytes[15:])
    context_bin_file = extract_paper_planning_context(str(bin_file))
    assert isinstance(context_bin_file, dict)
    assert context_bin_file["title"] != ""


def test_extract_paper_planning_context_corrupt_text(tmp_path):
    # 1. Text file with corrupt non-utf8 bytes opened with errors="replace"
    corrupt_text_file = tmp_path / "corrupt_notes.txt"
    corrupt_text_file.write_bytes(b"Title: Corrupt Paper\nMethod: Train model with \x8f\x9a\xff\xfe invalid bytes.")

    context = extract_paper_planning_context(str(corrupt_text_file))
    assert isinstance(context, dict)
    assert context["title"] == "corrupt_notes.txt"
    assert len(context["sections"]) > 0
    assert "Train model with" in context["sections"][0]

    # 2. Text file with embedded null byte routing to PyMuPDF safely
    null_byte_file = tmp_path / "null_byte.txt"
    null_byte_file.write_bytes(b"Title: Null Byte Paper\nMethod: Embedded \x00 bytes.")
    context_null = extract_paper_planning_context(str(null_byte_file))
    assert isinstance(context_null, dict)
    assert "Null Byte Paper" in context_null["title"] or context_null["title"] == "null_byte.txt"

    # 3. In-memory string with embedded null bytes and replacement chars
    raw_corrupt_str = "Scientific Protocol with \x00 null bytes and \ufffd invalid characters.\nMethod: Step 1."
    context_str = extract_paper_planning_context(raw_corrupt_str)
    assert isinstance(context_str, dict)
    assert context_str["title"] == "Scientific Paper"
    assert len(context_str["sections"]) > 0
    assert "Scientific Protocol" in context_str["sections"][0]

    # 4. Malformed XML string with corrupt unclosed tags
    corrupt_xml = "<TEI><teiHeader><titleStmt><title>Malformed XML Paper</title></titleStmt></teiHeader><text><body><div><head>Method</head><p>Testing unclosed tags"
    context_xml = extract_paper_planning_context(corrupt_xml)
    assert isinstance(context_xml, dict)
    assert context_xml["title"] == "Malformed XML Paper"


def test_extract_paper_planning_context_encrypted_pdf(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Confidential Secret Paper")
    enc_path = tmp_path / "encrypted_paper.pdf"
    doc.save(str(enc_path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="user_secret", owner_pw="owner_secret")
    doc.close()

    context = extract_paper_planning_context(str(enc_path))
    assert isinstance(context, dict)
    assert context["sections"] == []
    assert "encrypted" in context["title"].lower()

    # Also test deconstruct_paper_into_task_graph on encrypted PDF
    graph = deconstruct_paper_into_task_graph(str(enc_path), force_heuristic=True)
    assert len(graph.tasks) == 5
    for task in graph.tasks.values():
        for excerpt in task.paper_context_excerpts:
            assert str(enc_path) not in excerpt


def test_extract_paper_planning_context_empty_pdf(tmp_path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    context = extract_paper_planning_context(str(empty_file))
    assert isinstance(context, dict)
    assert context["sections"] == []
    assert context["title"] == "empty.pdf"

    # Also test deconstruct_paper_into_task_graph on empty PDF
    graph = deconstruct_paper_into_task_graph(str(empty_file), force_heuristic=True)
    assert len(graph.tasks) == 5
    for task in graph.tasks.values():
        for excerpt in task.paper_context_excerpts:
            assert str(empty_file) not in excerpt


def test_extract_paper_planning_context_pathlib_and_streams(tmp_path):
    import io
    from pathlib import Path
    import fitz

    # 1. Test pathlib.Path
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Title: PathLib Compatible Paper")
    page.insert_text((50, 100), "2. Methodology\nTesting pathlib Path support.")
    pdf_path = tmp_path / "pathlib_paper.pdf"
    doc.save(str(pdf_path))
    raw_bytes = doc.tobytes()
    doc.close()

    p = Path(pdf_path)
    context_path = extract_paper_planning_context(p)
    assert "PathLib Compatible Paper" in context_path["title"]
    assert any("Methodology" in s for s in context_path["sections"])

    graph = deconstruct_paper_into_task_graph(p, force_heuristic=True)
    assert graph.paper_id == "pathlib_paper"

    # 2. Test io.BytesIO stream
    buf = io.BytesIO(raw_bytes)
    context_buf = extract_paper_planning_context(buf)
    assert "PathLib Compatible Paper" in context_buf["title"]
    assert any("Methodology" in s for s in context_buf["sections"])

    # 3. Test None and empty string
    ctx_none = extract_paper_planning_context(None)
    assert ctx_none["sections"] == []
    ctx_empty_str = extract_paper_planning_context("")
    assert ctx_empty_str["sections"] == []


def test_extract_paper_planning_context_nonexistent_pdf():
    # Should not treat nonexistent filepath as plain text body
    context = extract_paper_planning_context("nonexistent_paper_123.pdf")
    assert context["sections"] == []
    assert context["title"] == "nonexistent_paper_123.pdf"


def test_heuristic_task_graph_empty_context_fallbacks():
    context = {"title": "Empty Paper", "abstract": "", "sections": [], "tables": []}
    graph = _heuristic_task_graph("empty_test", "Empty Paper", context)
    assert len(graph.tasks) == 5
    t2 = graph.get_task("task_02_data")
    assert t2.paper_context_excerpts == ["Dataset and preprocessing details."]
    t3 = graph.get_task("task_03_model")
    assert t3.paper_context_excerpts == ["Model architecture and methodology."]
    t4 = graph.get_task("task_04_experiment")
    assert t4.paper_context_excerpts == ["Experimental setup and training protocol."]
    t5 = graph.get_task("task_05_eval")
    assert t5.paper_context_excerpts == ["Evaluation results and comparison."]


def test_extract_paper_planning_context_short_headings(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Short Headings Benchmark Paper")
    page.insert_text((50, 100), "Abstract: Validating extraction of short section headings under 10 chars.")
    # 2  THEORY is 9 chars; METHODS is 7 chars; RESULTS is 7 chars; 1. DATA is 7 chars
    page.insert_text((50, 150), "2  THEORY")
    page.insert_text((50, 180), "We develop the mathematical theoretical framework for quantum networks.")
    page.insert_text((50, 220), "METHODS")
    page.insert_text((50, 250), "The experimental method applies recursive Green functions.")
    page.insert_text((50, 290), "RESULTS")
    page.insert_text((50, 320), "The simulation yields 99.4 percent accuracy across tests.")

    pdf_file = tmp_path / "short_headings.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert any("2  THEORY" in s for s in context["sections"])
    assert any("METHODS" in s for s in context["sections"])
    assert any("RESULTS" in s for s in context["sections"])


def test_extract_paper_planning_context_spaced_summary(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Target-Oriented Modeling with Patched Functions")
    page.insert_text((50, 100), "S U M M A R Y\nWe develop the mathematical framework of recursive patched functions for wavefields.")
    page.insert_text((50, 180), "Key words: Waveform Inversion; Numerical Modeling.\n1  INTRODUCTION\nThe need for subsurface information is critical.")

    pdf_file = tmp_path / "spaced_summary.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert "Target-Oriented" in context["title"]
    assert "recursive patched functions" in context["abstract"]
    assert any("INTRODUCTION" in s for s in context["sections"])


def test_extract_paper_planning_context_multiline_title_and_captions(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # Multiline title block
    page.insert_text((50, 50), "Target-oriented modelling and inversion using Recursive Patched\nGreen Functions")
    page.insert_text((50, 100), "Abstract: Multiline caption and title test paper.")
    page.insert_text((50, 150), "1. Introduction\nBackground details on wave propagation.")
    # Multiline figure caption
    page.insert_text((50, 200), "Figure 1: Schematic diagram of the multi-terminal setup\nconsidered here and used as a benchmark for comparison.")
    page.insert_text((50, 250), "Table 1: Boldface integers indicate the number of impurities\nobtained from the inversion workflow across runs.")

    pdf_file = tmp_path / "multiline_paper.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert context["title"] == "Target-oriented modelling and inversion using Recursive Patched Green Functions"
    assert any("multi-terminal setup considered here" in t for t in context["tables"])
    assert any("Boldface integers indicate the number" in t for t in context["tables"])


def test_extract_paper_planning_context_tei_no_duplicate_abstract():
    # TEI XML with abstract div should not include abstract div as a methodology section
    tei_with_abstract_div = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Deduplication Test Paper</title></titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <div><p>This is the actual abstract text that should never appear as a body section.</p></div>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1. Introduction</head>
        <p>This is genuine introduction body prose describing the problem setting.</p>
      </div>
      <div>
        <head>2. Methodology and Architecture</head>
        <p>This describes our variational neural network model architecture.</p>
      </div>
    </body>
  </text>
</TEI>"""
    context = extract_paper_planning_context(tei_with_abstract_div)
    assert "actual abstract text" in context["abstract"]
    for s in context["sections"]:
        assert "This is the actual abstract text" not in s
        assert not s.startswith("## Section\nThis is the actual abstract")
    assert any("Methodology and Architecture" in s for s in context["sections"])


def test_find_cached_tei_short_stem_and_glob_characters(tmp_path, monkeypatch):
    from research_assistant import config
    from research_assistant.tasks.extractor import _find_cached_tei

    grobid_dir = tmp_path / "raw" / "grobid_output"
    grobid_dir.mkdir(parents=True)

    # File with 'a' in name
    (grobid_dir / "title_177849b11a7322dd.grobid.tei.xml").write_text("<TEI/>", encoding="utf-8")
    # File with brackets in name
    (grobid_dir / "paper_bracket.tei.xml").write_text("<TEI/>", encoding="utf-8")

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "GROBID_TEI_DIR", str(grobid_dir))

    # Short stem 'a.pdf' must NOT false-positive match 'title_177849b11a7322dd.grobid.tei.xml'
    assert _find_cached_tei("a.pdf") is None

    # Brackets in filename should not cause glob exception
    res_bracket = _find_cached_tei("paper [2024].pdf")
    assert res_bracket is None


def test_extract_paper_planning_context_corrupt_pdf_file_no_binary_leak(tmp_path):
    # A corrupt .pdf file on disk that does not start with %PDF- (e.g. truncated binary download)
    corrupt_pdf = tmp_path / "broken_corrupt.pdf"
    corrupt_pdf.write_bytes(b"\x8f\x9a\x00\xff\xfe\xca\xfe\xba\xbe\x00\x01\x02RandomBinaryTrash")

    context = extract_paper_planning_context(str(corrupt_pdf))
    assert isinstance(context, dict)
    # Sections must not contain binary replacement characters
    assert context["sections"] == []


def test_paper_id_path_traversal_sanitization():
    # Dangerous file names with dots or path traversal
    graph1 = deconstruct_paper_into_task_graph("..pdf", force_heuristic=True)
    assert graph1.paper_id == "paper"
    assert ".." not in graph1.paper_id

    graph2 = deconstruct_paper_into_task_graph("../../../secret.pdf", force_heuristic=True)
    assert graph2.paper_id == "secret"
    assert ".." not in graph2.paper_id


def test_stream_seek_reset():
    import io
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Title: Stream Reset Test Paper")
    page.insert_text((50, 100), "2. Methodology\nTesting stream seek reset.")
    pdf_bytes = doc.tobytes()
    doc.close()

    stream = io.BytesIO(pdf_bytes)
    # Move pointer to end of file
    stream.seek(0, io.SEEK_END)
    assert stream.tell() > 0

    # extract_paper_planning_context should seek to 0 and read all bytes
    context = extract_paper_planning_context(stream)
    assert "Stream Reset Test Paper" in context["title"]
    assert any("Methodology" in s for s in context["sections"])


def test_extract_paper_planning_context_utf16_and_bom(tmp_path):
    # UTF-16 encoded text paper
    utf16_text = "Title: Quantum Communication Protocol\nMethod: Train an entanglement purification model."
    utf16_file = tmp_path / "paper_utf16.txt"
    utf16_file.write_bytes(utf16_text.encode("utf-16"))

    context = extract_paper_planning_context(str(utf16_file))
    assert isinstance(context, dict)
    assert len(context["sections"]) > 0
    assert "entanglement purification" in context["sections"][0]
    assert "\x00" not in context["sections"][0]


def test_extract_paper_planning_context_rejects_junk_metadata_titles(tmp_path):
    import fitz

    doc = fitz.open()
    doc.set_metadata({"title": "ldos.eps", "subject": "gnuplot plot"})
    page = doc.new_page()
    page.insert_text((50, 50), "Real Scientific Discovery in Condensed Matter")
    page.insert_text((50, 100), "Abstract: Paper with junk eps metadata.")
    page.insert_text((50, 150), "1. Introduction\nDetails here.")
    pdf_file = tmp_path / "junk_meta.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert context["title"] != "ldos.eps"
    assert "Real Scientific Discovery" in context["title"]


def test_find_cached_tei_none_and_invalid_types():
    from research_assistant.tasks.extractor import _find_cached_tei

    assert _find_cached_tei(None) is None
    assert _find_cached_tei(12345) is None
    assert _find_cached_tei([]) is None
    assert _find_cached_tei(b"") is None
    assert _find_cached_tei(b"nonexistent.pdf") is None


def test_extract_from_tei_xml_none_and_invalid_types():
    from research_assistant.tasks.extractor import _extract_from_tei_xml

    res_none = _extract_from_tei_xml(None)
    assert isinstance(res_none, dict)
    assert res_none["sections"] == []
    assert res_none["title"] == "Scientific Paper"

    res_int = _extract_from_tei_xml(123)
    assert isinstance(res_int, dict)
    assert res_int["sections"] == []

    res_empty = _extract_from_tei_xml("")
    assert isinstance(res_empty, dict)
    assert res_empty["sections"] == []


def test_extract_paper_planning_context_plain_text_with_pdf_extension(tmp_path):
    # Plain text file mistakenly named or renamed with a .pdf extension
    fake_pdf = tmp_path / "text_as_pdf.pdf"
    fake_pdf.write_text("Title: Text Paper in PDF Clothing\nMethod: Train an unsupervised network.", encoding="utf-8")

    context = extract_paper_planning_context(str(fake_pdf))
    assert isinstance(context, dict)
    assert len(context["sections"]) > 0
    assert "Train an unsupervised network" in context["sections"][0]


def test_extract_paper_planning_context_nonexistent_txt_file():
    # Nonexistent .txt path must return empty sections rather than treating the path as body prose
    context = extract_paper_planning_context("nonexistent_paper_456.txt")
    assert context["sections"] == []
    assert context["title"] == "nonexistent_paper_456.txt"


def test_extract_paper_planning_context_open_file_object_with_name_cached_tei(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Dummy PDF Text")
    pdf_file = tmp_path / "paper_with_cached.pdf"
    doc.save(str(pdf_file))
    doc.close()

    tei_file = tmp_path / "cached_tei_paper.tei.xml"
    tei_file.write_text(SAMPLE_TEI_XML, encoding="utf-8")

    with patch("research_assistant.tasks.extractor._find_cached_tei", return_value=str(tei_file)):
        with open(pdf_file, "rb") as f:
            context = extract_paper_planning_context(f)
            assert context["title"] == "Deep Learning for Quantum State Reconstruction"
            assert any("Method and Model Architecture" in s for s in context["sections"])


def test_deconstruct_paper_into_task_graph_file_object_pid(tmp_path):
    txt_file = tmp_path / "entanglement_sim.txt"
    txt_file.write_text("Method: Perform entanglement purification.", encoding="utf-8")

    with open(txt_file, "r", encoding="utf-8") as f:
        graph = deconstruct_paper_into_task_graph(f, force_heuristic=True)
        assert graph.paper_id == "entanglement_sim"
        assert len(graph.tasks) == 5


def test_extract_paper_planning_context_two_line_headings(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Two-Line Headings Quantum Paper")
    page.insert_text((50, 100), "Abstract: Validating two-line headings extraction.")
    # Line 0 is prefix, line 1 is heading title
    page.insert_text((50, 150), "I.\nINTRODUCTION")
    page.insert_text((50, 180), "Introduction text describing the quantum lattice system.")
    page.insert_text((50, 220), "II.\nAUBRY-ANDRE MODEL")
    page.insert_text((50, 250), "Details on the discrete Schroedinger operator and quasi-periodic potential.")
    page.insert_text((50, 300), "III.\nNUMERICAL RESULTS")
    page.insert_text((50, 330), "Numerical simulation and scaling calculations across lattices.")
    pdf_file = tmp_path / "two_line_headings.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert any("I. INTRODUCTION" in s for s in context["sections"])
    assert any("II. AUBRY-ANDRE MODEL" in s for s in context["sections"])
    assert any("III. NUMERICAL RESULTS" in s for s in context["sections"])


def test_extract_paper_planning_context_unstructured_pdf_fallback(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Title: Unstructured Scientific Report")
    page.insert_text((50, 100), "Paragraph one describes the experimental methodology and dataset.")
    page.insert_text((50, 150), "Paragraph two describes the neural network training and validation.")
    page.insert_text((50, 200), "Paragraph three presents the benchmark results and discussion.")
    pdf_file = tmp_path / "unstructured.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert isinstance(context, dict)
    assert len(context["sections"]) > 0
    assert "experimental methodology" in context["sections"][0]
    assert "benchmark results" in context["sections"][0]


def test_extract_paper_planning_context_utf32_bom_text_file(tmp_path):
    utf32_file = tmp_path / "paper_utf32.txt"
    text = "Title: Quantum Computing UTF32 Paper\nMethod: Quantum circuit optimization with unitary matrices."
    utf32_file.write_bytes(text.encode("utf-32"))

    context = extract_paper_planning_context(str(utf32_file))
    assert isinstance(context, dict)
    assert len(context["sections"]) > 0
    assert "Quantum circuit optimization" in context["sections"][0]


def test_extract_paper_planning_context_fitz_document(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Open Fitz Document Paper")
    page.insert_text((50, 100), "Abstract: Testing fitz.Document object passing.")
    page.insert_text((50, 150), "1. Introduction\nDocument context details.")

    context = extract_paper_planning_context(doc)
    assert "Open Fitz Document Paper" in context["title"]
    assert "fitz.Document" in context["abstract"]
    assert not doc.is_closed  # Should not close external document
    doc.close()


def test_extract_paper_planning_context_fallback_abstract(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "RevTex Style Physics Paper")
    # Authors and affiliations without sentence-ending punctuation
    page.insert_text((50, 80), "John Doe,1 Jane Smith,2 and Bob Wilson1")
    page.insert_text((50, 100), "1Department of Physics, Stanford University\n2Department of Physics, MIT")
    # Abstract without "Abstract:" label, but with complete sentences > 150 chars
    abstract_text = (
        "We consider quantum wave propagation in one-dimensional quasiperiodic lattices. "
        "We propose an iterative construction of quasiperiodic potentials from sequences of potentials. "
        "At each finite iteration step the eigenstates reflect the properties of the limiting potential."
    )
    page.insert_textbox(fitz.Rect(50, 130, 550, 220), abstract_text)
    page.insert_text((50, 240), "1. Introduction\nIntroduction prose starts here.")
    pdf_file = tmp_path / "revtex_paper.pdf"
    doc.save(str(pdf_file))
    doc.close()

    context = extract_paper_planning_context(str(pdf_file))
    assert "John Doe" not in context["abstract"]
    assert "iterative construction of quasiperiodic potentials" in context["abstract"]





