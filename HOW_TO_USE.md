# 📚 Autonomous Academic Research Assistant — User & Developer Guide

Welcome to the **Autonomous Academic Research Assistant**. This guide provides comprehensive documentation on using the web application across all five tabs, understanding the AI pipeline outputs, and running or developing the system locally.

---

## 🔗 Quick Links & Access

- **GitHub Repository**: [https://github.com/mukims/research_assistant](https://github.com/mukims/research_assistant)
- **Live Cloud Web Application**:
  - Secure Custom Domain: [https://marvin-the-citebot.duckdns.org](https://marvin-the-citebot.duckdns.org)
  - Direct HTTP: [http://34.62.178.182:8080](http://34.62.178.182:8080)
- **Active AI Models**:
  - Chat, Synthesis & Claim Judgement: `gemma4:e2b` (Google Gemma 4, 2B parameters)
  - Dense Vector Embeddings: `nomic-embed-text`
- **Managed Microservices**:
  - GROBID TEI Extraction Service running on port `8070` (`consolidate_citations=0` for rapid parsing).
  - ChromaDB vector store + BM25 keyword index on persistent storage.

---

## 🧭 Interface Overview (The 5 Tabs)

| Tab | Purpose | Input | Expected Output |
|---|---|---|---|
| **Tab 1: Research a topic** | Ingest literature, download references, audit citations, synthesize review | PDF research paper or topic query | Downloaded PDFs, indexed corpus, in-text citation audit, related-work synthesis |
| **Tab 2: Cite a draft** | Suggest and verify citations for a single claim | Single scientific statement | Formatted `\cite{...}`, source metadata, support rationale, evidence passages |
| **Tab 3: Cite a whole draft** | Batch-cite and audit an entire academic manuscript | Plain-text draft (`.txt` or pasted) | Fully cited manuscript, BibTeX source mapping, and Agent 8 sentence-by-sentence verification audit |
| **Tab 4: Research chat** | Multi-turn conversational research grounded in your library | Plain-English research questions | Token-by-token streaming answers with expandable evidence excerpts |
| **Tab 5: How to use** | Complete documentation, pipeline explanations, and developer guide | None | This reference guide! |

---

## 📑 Detailed Tab Instructions

### Tab 1 — Research a topic (Corpus Builder & Citation Auditor)

This tab is the primary entry point for building your literature database.

#### 1. Uploading Research Papers (PDF or ZIP)
* **Single PDF Upload**:
  1. Select **Upload research paper(s)**.
  2. Drag and drop your `.pdf` file.
  3. *(Optional)* Enter a research topic question to focus the final synthesis, or leave blank to infer from the paper.
  4. Ensure **Audit citations** is toggled ON if you want the system to check whether the cited papers support the uploaded paper's statements.
  5. Click **Process and Index Paper(s)**.
* **Batch Upload (Multiple PDFs or ZIP)**:
  1. Select and upload multiple PDFs or a `.zip` archive.
  2. The system executes direct batch ingestion: parses all documents, chunks text, computes embeddings, and updates the search index in one pass.

#### 2. Automatic Topic Search
1. Select **Search for a paper**.
2. Type an academic topic into **Research idea** (e.g. `GPU-accelerated filtered density function simulator`).
3. *(Optional)* Provide an arXiv or direct PDF URL in **Seed paper URL** to pin a specific seed.
4. Click **Build corpus**.

#### 3. Understanding Tab 1 Outputs

* **What does the "Score" mean?**
  * The score next to each paper in **Relevant prior work** is a **semantic topic relevance score**:
    $$\text{score} = 1.0 - \text{cosine\_distance}$$
  * Out of all reference PDFs fetched, only papers that pass the Stage-1 vector retrieval and LLM relevance gate are shortlisted. It measures topic alignment with your research query, **not** claim truthfulness.
* **What is "Related-Work Overview"? Does it search the web?**
  * **No, it does NOT search the live web.**
  * It is an AI synthesis grounded exclusively in the **locally downloaded PDFs** in your database. The local LLM (`gemma4:e2b`) reviews the retrieved excerpts from your library and writes a structured literature review citing each paper by key.
* **In-Text Citation Audit (Seed Paper Claim Verification)**:
  * For uploaded seed papers, GROBID extracts in-text citations (`[14]`, `[Smith et al.]`) and maps them to the bibliography.
  * For each referenced paper that was downloaded into your local library, the system retrieves evidence chunks and evaluates the statement using the **Gemma 4 Judgement Rubric**:
    * **Verdicts**: `🟢 Supports`, `🟡 Partially Supports`, `🔴 Contradicts`, `🟠 Does Not Support`, `⚪ Unclear`.
    * **Verbatim Evidence**: The exact sentence quoted from the cited PDF.
    * **Confidence & Rationale**: The model's reasoning.
    * **Paywalled References**: Clearly flagged if the paper was paywalled and could not be downloaded.
  * You can re-run or trigger the audit anytime using the **"Audit Seed Paper Citations"** button.

---

### Tab 2 — Cite a draft (Single-Sentence Verification)

Use this tab to verify and attribute citations for an individual claim.

1. Ensure the sidebar shows indexed papers (`Chunks > 0`).
2. Enter your sentence into the claim box (e.g. `Topological edge states exhibit robust protection against non-magnetic impurities.`).
3. Select how many passages to evaluate (default: 5).
4. Click **Suggest a citation**.
5. **Output**:
   * **Suggested sentence** with LaTeX `\cite{...}` markup.
   * **Source attribution** and confidence.
   * **Support rationale** explaining how the evidence verifies the claim.
   * **Retrieved passages** showing the source text.

---

### Tab 3 — Cite a whole draft (Batch Citer & Agent 8 Verifier)

Use this tab to process an entire draft manuscript at once.

1. **Upload or Paste**: Upload a `.txt` manuscript or paste your draft into the editor.
2. Click **Cite the draft**:
   * **Agent 5** splits the draft into sentences, identifies factual assertions requiring citation, and matches them to your library.
   * Produces a fully cited draft with `\cite{cite_1}`, `\cite{cite_2}` tags and a `_citations.json` registry.
3. Click **Verify citations**:
   * **Agent 8** re-retrieves evidence from the cited papers and audits every citation.
   * Displays 5 metric tiles (`Checked`, `Supports`, `Partially`, `Need review`, `Not judged`).
   * Generates a detailed audit report with verbatim quotations and slot evaluations (finding, scope, strength).

---

### Tab 4 — Research chat (Conversational Exploration)

Use this tab to conduct multi-turn literature question-answering grounded in your vector library.

1. Type any research question into the chat input.
2. The response streams token-by-token in real time.
3. Under each response, expand **Sources** to view the exact text passages that informed the answer.
4. Export the conversation to Markdown or clear the history at any time.

---

## ⚡ Live Telemetry & Pipeline Status

The left sidebar includes an automatic real-time telemetry widget:
- **Active State (`⚡ Pipeline Active`)**: Displays the current pipeline stage (Steps 1 to 5), the paper currently being downloaded or summarized, an item progress bar, and elapsed time.
- **Live Activity Log**: An expandable rolling log of recent background events (e.g. `✅ Downloaded 14 reference PDFs`, `✓ Generated paper summaries with gemma4:e2b`).
- **Idle State (`🟢 Pipeline: Idle`)**: Shows when the server is ready for new jobs and displays the completion summary of the last run.

---

## ❓ Frequently Asked Questions

### Why did the pipeline fetch 14 references but only score 6 of them?
Stage 1 of the retrieval engine searches your library for papers relevant to your specific research query. After initial vector ranking, an LLM relevance gate filters out tangential citations (such as general software packages, standard mathematical identities, or peripheral background papers). The 6 papers that passed the gate are shortlisted and scored for the literature review.

### Why are some citations marked "Paywalled / Not In Corpus"?
The assistant downloads papers through open-access repositories (Unpaywall, Europe PMC, arXiv). If a cited paper is behind an academic paywall, it cannot be downloaded automatically. You can manually add any PDF to the library by uploading it in Tab 1.

### How are citations extracted from PDFs?
The pipeline uses GROBID (`processFulltextDocument`) running on port 8070 to parse PDFs into structured TEI XML. This extracts titles, authors, DOIs, bibliography items, and body paragraphs with tagged in-text citation markers (`<ref type="bibr">`).

---

## 💻 Local Developer Setup (Linux / macOS)

To clone and run the project locally on your own machine:

```bash
# 1. Clone repository
git clone https://github.com/mukims/research_assistant.git
cd research_assistant

# 2. Set up Python virtual environment (Python 3.11 or 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Pull Ollama models
ollama pull nomic-embed-text
ollama pull gemma4:e2b

# 5. Launch GROBID Docker microservice
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1

# 6. Run test suite
pytest -q

# 7. Start the Streamlit application
export UNPAYWALL_EMAIL="your.email@example.com"
export CITATION_CHAT_MODEL="gemma4:e2b"
streamlit run app.py
```
