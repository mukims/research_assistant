# 📚 How to use the Research Assistant

Welcome to the **Autonomous Academic Research Assistant**. This guide explains how to drive the application across all five interface tabs, whether running on the cloud deployment or locally.

---

## 🌐 Quick Access: Cloud Web Application

If you are using the cloud deployment, everything is already installed, configured, and running:

- **Web UI URLs**:
  - Direct HTTP: [http://34.62.178.182/](http://34.62.178.182/) *(no port :8080 needed!)*
  - Secure Custom Domain: [https://marvin-the-citebot.duckdns.org](https://marvin-the-citebot.duckdns.org)
- **Active AI Models**:
  - Chat & Synthesis: `gemma4:e2b` (Google Gemma 4, 2B parameters, optimized for CPU inference)
  - Dense Vector Embeddings: `nomic-embed-text`
- **Reference Extraction**: GROBID TEI Extraction Engine running as a managed background microservice on port 8070.
- **Persistent Storage**: 50 GB persistent SSD disk mounted at `/mnt/disks/data` preserving your ingested PDFs, extracted citations, and ChromaDB vector index across reboots.

---

## 🧭 Interface Overview (The 5 Tabs)

The Research Assistant is organized into 5 functional tabs:

| Tab | Purpose | Key Input | Expected Output |
|---|---|---|---|
| **Tab 1: Research a topic** | Build a grounded corpus from a seed paper | PDF file or research topic string | Vector database index, downloaded open-access reference papers, related-work synthesis |
| **Tab 2: Cite a draft** | Suggest and verify a citation for a single sentence | One claim or sentence | Formatted citation with `\cite{...}`, source attribution, support rationale, and matched passages |
| **Tab 3: Cite a whole draft** | Batch-cite an entire academic manuscript or draft | Full text draft (`.txt` or pasted) | Fully cited document with `\cite{cite_N}` markers, BibTeX bibliography mapping, and per-sentence verification report |
| **Tab 4: Research chat** | Interactive multi-turn literature dialogue | Plain-English research questions | Token-by-token streaming answers grounded in the retrieved vector context with source accordions |
| **Tab 5: How to use** | Complete documentation and troubleshooting guide | None | This guide! |

---

## 📑 Detailed Tab Instructions

### Tab 1 — Research a topic (Building the Corpus)

This tab seeds the assistant with academic literature. Everything in Tabs 2–4 retrieves from the corpus built here.

#### Option A: Upload research paper(s) (Single PDF, Multi-PDF, or ZIP)
1. Select **Upload research paper(s) (PDF or ZIP)** in the mode selector.
2. Drag and drop your `.pdf` research paper(s) or a `.zip` archive into the file uploader:
   - **Single PDF**: Runs the full end-to-end pipeline (mines the paper's bibliography with GROBID, downloads open-access references, builds the corpus, and formulates a synthesis).
   - **Multiple PDFs or ZIP archive**: Direct batch ingestion — parses all papers, extracts text chunks, embeds them into ChromaDB, and updates the BM25 keyword index in one step.
3. *(Optional)* Enter a **Research topic / question** to focus the final synthesis on a specific question across your papers. If left blank, the topic is automatically inferred.
4. Click **Process and Index Paper(s)**.

#### Option B: Search for a paper automatically
1. Select **Search for a paper**.
2. Type a specific academic topic into **Research idea** (e.g., `spin-orbit coupling in monolayer transition metal dichalcogenides`).
3. *(Optional)* Provide a direct arXiv or PDF link in **Seed paper URL** to pin the exact paper.
4. Click **Build corpus**.

#### What happens during a build:
1. **Discover & Unpack**: Uploaded files (including nested PDFs inside ZIP archives) are validated and staged.
2. **Ingest**: PyMuPDF extracts text chunks and `nomic-embed-text` vectors are indexed into ChromaDB.
3. **Reference Mining (Single Seed)**: GROBID parses the bibliography into structured references (authors, title, year, DOI) and open-access resolvers download available PDFs.
4. **Respond**: An extractive synthesis summarizing prior literature relevant to your query is produced.

---

### Tab 2 — Cite a draft (Single-Sentence Verification)

Use this tab when polishing a specific assertion in your manuscript.

1. **Prerequisite**: Ensure the sidebar indicates papers are indexed (`Chunks > 0`).
2. Paste **one sentence** into the claim input box (e.g., `Topological edge states exhibit robust protection against non-magnetic impurities.`).
3. Select how many relevant passages to retrieve (default: 5).
4. Click **Suggest a citation**.
5. **Review the output**:
   - **Suggested sentence**: Your sentence with LaTeX `\cite{...}` inserted.
   - **Sources cited**: Detailed publication metadata for the matched papers.
   - **Rationale**: Explanation of why the cited passage supports or qualifies the claim.
   - **Retrieved passages**: Expand to read the exact text chunks matched from the corpus.

---

### Tab 3 — Cite a whole draft (Batch Document Citation)

Use this tab to process an entire draft manuscript at once.

1. **Upload or Paste**: Upload a `.txt` file or paste your complete draft into the text area.
2. Click **Cite the draft**.
3. **Pipeline Stages**:
   - The manuscript is split into individual sentences using sentence boundary detection.
   - Sentences are evaluated for whether they assert a verifiable scientific claim.
   - Each qualifying sentence queries the vector database for matching evidence.
4. **Results Provided**:
   - **Cited Draft**: Your complete manuscript with sequential `\cite{cite_1}`, `\cite{cite_2}` tags.
   - **BibTeX Source Mapping**: JSON dictionary linking citation keys to full metadata.
   - **Decisions Expander**: A complete breakdown showing why each sentence was cited, skipped, or had insufficient evidence.

---

### Tab 4 — Research chat (Conversational Exploration)

Use this tab for multi-turn literature question-answering grounded in your vector corpus.

1. Type any research question into the chat box at the bottom.
2. **Streaming Response**: The response streams in real time token-by-token.
3. **Context Accordion**: Underneath each assistant message, expand **Sources** to see the exact excerpts from your papers that informed the answer.
4. **Export & Reset**:
   - Click **Export conversation** to download your session transcript as a Markdown document.
   - Click **Clear** to start a new chat session.

---

### Tab 5 — How to use

This in-app tab displays this documentation file directly from the repository.

---

## 🛠️ Sidebar Controls & Services

The left sidebar provides live status indicators and microservice controls:

- **Corpus Statistics**: Displays total indexed papers and vector chunks in ChromaDB.
- **Model Runtimes**: Shows active chat model (`gemma4:e2b`) and embedding model (`nomic-embed-text`).
- **GROBID Service Manager**:
  - **Status Indicator**: Green dot indicates GROBID is alive and answering on port 8070; red indicates unreachable.
  - **Start GROBID**: Launches the GROBID Docker microservice if stopped.
  - **Stop / Restart**: Controls the GROBID daemon.
  - **Status**: Runs health diagnostics and checks memory allocation.

---

## ❓ Troubleshooting & Frequently Asked Questions

### Why does a chat query or citation take 20–30 seconds to start streaming?
On CPU-based deployments without GPU acceleration, the language model must perform an initial **prompt evaluation** phase (reading the conversation history plus 4,000 tokens of retrieved academic passages). Once this evaluation finishes, the token generation phase runs at ~20 tokens/second. The loading spinner indicates that the system is actively evaluating the literature context.

### Why do some references show as unavailable?
The assistant exclusively fetches legal, open-access PDFs through official public APIs (Unpaywall, arXiv, Crossref). If a paper is behind a publisher paywall, it is safely recorded under unavailable references. You can always manually upload a PDF into the corpus via Tab 1 or the `pulled_pdfs/` folder.

### What should I do if GROBID shows a red dot?
Click the **Start GROBID** or **Restart** button in the sidebar under Server & Services. If running outside Docker, ensure GROBID is running on port 8070 via `curl http://localhost:8070/api/isalive`.

---

## 💻 Local Developer Setup (Mac / Linux)

If you wish to clone and run the project locally on your own machine:

```bash
# 1. Clone repository & set up Python 3.11 or 3.12 virtual environment
git clone <repo_url> research_assistant
cd research_assistant
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 3. Pull Ollama models
ollama pull nomic-embed-text
ollama pull gemma4:e2b

# 4. Launch GROBID container
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1

# 5. Run Streamlit UI
export UNPAYWALL_EMAIL="your.email@example.com"
export CITATION_CHAT_MODEL="gemma4:e2b"
streamlit run app.py
```
