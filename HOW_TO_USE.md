# 📚 Citation Needed! (Marvin the Citebot) — Complete User Guide

Welcome to **Citation Needed!**, powered by **Marvin the Citebot** (Autonomous Academic Research Assistant). 

This system builds and searches an evidence-grounded academic literature corpus, automatically cites scientific manuscripts, verifies citations against full-text source papers with an explicit rubric, and facilitates interactive literature brainstorming.

---

## 🧭 Interface Overview (The 5 Tabs)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       📚 Citation Needed!                                              │
│                                                                                                        │
│  [Tab 1: Research a topic]   [Tab 2: Cite a draft]   [Tab 3: Cite a whole draft]   [Tab 4: Chat]   [Tab 5]   │
│  Mine literature, fetch      Suggest & ground one    Batch-cite manuscript &       Interactive     Manual   │
│  PDFs, audit in-text claims  individual claim        Agent 8 verification audit    studio          & FAQ    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Tab | Purpose | Primary Input | Key Deliverables |
|---|---|---|---|
| **Tab 1: Research a topic** | Mine literature, download open-access references, audit citations, synthesize review | Single PDF, multiple PDFs, ZIP, or topic query | Indexed corpus, seed paper card, per-paper notes, structured synthesis, in-text citation audit, missing paywall uploaders |
| **Tab 2: Cite a draft** | Suggest and attribute citations for an individual claim | Single scientific statement | Formatted LaTeX `\cite{...}`, source metadata, support rationale, verbatim evidence passages |
| **Tab 3: Cite a whole draft** | Batch-cite and rigorously audit an entire academic manuscript | Plain-text draft (`.txt` or pasted) | Fully cited draft, BibTeX mapping, sentence-by-sentence citation report, Agent 8 verification audit with 5 metric tiles |
| **Tab 4: Research chat** | Conversational research studio grounded in your local library | Plain-English research questions | Token-by-token streaming answers, 5 analytical lenses, keyed sources `[S#]`, clickable follow-up chips, persistent scratchpad |
| **Tab 5: How to use** | Interactive documentation, architecture guide, and FAQ | None | This user guide rendered live inside the application |

---

## ⚡ Sidebar Controls & Telemetry

The left sidebar provides persistent visibility and control over background operations and system resources:

### 1. Real-Time Telemetry & Progress
- **Status Indicator**: Shows `⚡ Pipeline Active` during execution or `🟢 Pipeline: Idle` when ready.
- **Stage Progression**: Tracks the active pipeline stage (Step 1: Discovery → Step 2: Seed Ingest → Step 3: Reference Extraction → Step 4: PDF Fetching → Step 5: Reference Ingestion & Synthesis).
- **Active Item & Progress Bar**: Displays the specific paper being downloaded, parsed, or summarized, with elapsed runtime.
- **Live Event Log**: Expandable rolling log showing the last 15 system events (e.g. `✅ Downloaded 14 reference PDFs`, `✓ Ingested 18 papers into index v2`).

### 2. Cooperative Pipeline Cancellation
- **"🛑 Stop pipeline" Button**: Appears in the sidebar and in Tab 1 whenever a background job is running.
- **Safe Item-Boundary Abort**: Cancellation is checked between discrete items (between individual PDF downloads, PDF parsings, summary calls, and graph nodes). It **never** aborts mid-write or inside an in-flight LLM call, ensuring ChromaDB and BM25 index files are never corrupted.

### 3. Corpus & System Metrics
- **Corpus Counters**: Total indexed documents, total chunks, median chunk size (typically ~1,000–1,200 characters in Index v2), and total figure/table captions.
- **Active Backend Telemetry**: Displays the current inference backend (e.g. Google Gemini `gemini-3.5-flash-lite` or local Ollama `gemma4:e2b`), active embeddings (`nomic-embed-text`), and GROBID service status.
- **GROBID Service Controls**: Includes buttons to inspect GROBID health, view diagnostics, or restart the container if running in Docker.

---

## 📑 Detailed Tab Instructions

---

### Tab 1 — Research a topic (Literature Miner, Synthesizer & Citation Auditor)

Tab 1 is the core engine for turning ideas or papers into an indexed, verifiable library.

#### 1. Choose Your Starting Point
Select one of the two input modes via the radio selector at the top:

* **Mode A: 📄 Upload research paper(s) (PDF or ZIP)**
  * **Single PDF Upload (Seed Mode)**:
    1. Drag and drop a single `.pdf` research paper.
    2. *(Optional)* Enter a **Research topic / question** to guide the final synthesis (e.g. `mechanical properties of MXene monolayers`). If left blank, the research question is automatically inferred from the paper's title and abstract.
    3. Configure the 4 execution toggles (see below).
    4. Click **Process and Index Paper(s)**.
    5. *What happens:* The system ingests your seed paper, parses its bibliography using GROBID TEI, downloads all open-access references, indexes full text into sentence windows, writes a multi-paper synthesis, and audits the seed paper's own in-text citations.
  * **Batch Upload (Multiple PDFs or ZIP Archive)**:
    1. Drag and drop several PDFs or a `.zip` archive containing papers.
    2. Click **Process and Index Paper(s)**.
    3. *What happens:* Direct batch ingestion. All papers are parsed, chunked, embedded, and added to the ChromaDB vector database and BM25 index in a single locked pass.

* **Mode B: 🔍 Search for a paper**
  1. Enter your research topic into **Research idea** (e.g. `low-temperature thermal conductivity in disordered crystalline solids`).
  2. *(Optional)* Enter a **Seed paper URL** (direct link or arXiv URL) if you want to anchor the search to an exact known paper.
  3. Configure the toggles and click **Build corpus**.
  4. *What happens:* Agent 0 searches arXiv, OpenAlex, and Semantic Scholar until a relevant open-access paper is successfully downloaded, then executes the literature mining pipeline.

---

#### 2. The Four Execution Toggles

| Toggle | Default | What It Does | When to Change |
|---|---|---|---|
| **Answer query** | `ON` | Generates a structured Map-Reduce synthesis of the shortlisted papers at the end of the run. | Turn `OFF` if you only want to download and index literature without generating a summary. |
| **Force re-run** | `OFF` | Bypasses cached seed papers and re-fetches all references from scratch. | Turn `ON` if you modified pipeline parameters or want to refresh cached web downloads. |
| **Analyse figures** | `OFF` | Passes every figure and table crop through the Vision-Language Model (`gemma4:e2b` or Gemini) to generate descriptive text chunks. | Turn `ON` when visual diagrams or charts are vital for synthesis and chat. *(Note: Adds ~30–60s per figure on CPU).* |
| **Audit citations** | `ON` | Evaluates every in-text citation in the uploaded seed paper against the full text of the downloaded references using Rubric V1.4. | Turn `OFF` if you only want literature ingestion and synthesis, skipping citation checking. |

---

#### 3. Understanding Tab 1 Outputs

Once execution finishes, Tab 1 displays the following sections:

##### A. Seed Paper Card
Displays full bibliographic metadata extracted by GROBID: Title, Authors, Year, Publication Venue, DOI (with direct clickable link), and Extraction Engine (`grobid` or `pymupdf`).

##### B. References Fetched & Unavailable Expanders
- **References Fetched**: Lists all reference papers that were resolved and downloaded into your library.
- **References Unavailable**: Lists references that could not be downloaded (paywalled publishers, proprietary journals, or broken URLs), with author, title, and DOI links.

##### C. Per-Paper Notes
Before synthesis, the system reads each shortlisted paper independently and extracts structured notes under unique keys (`[P1]`, `[P2]`, ...):
- **What it establishes**: Core findings backed by evidence.
- **Method / System**: Experimental or theoretical setup, sample conditions, materials.
- **Limits / Caveats**: Stated constraints, assumptions, or gaps.

##### D. Related-Work Synthesis
A structured, grounded synthesis written across the shortlisted papers organized under three mandatory headings:
1. **What is established**: Consensus findings confirmed across multiple papers.
2. **Where the papers differ**: Competing hypotheses, contradictory observations, or divergent experimental conditions.
3. **The gap**: What remains unaddressed in the current literature.
- *Citation Verification*: Every citation in the synthesis is verified against the shortlist. If the model cites an unverified key or a paper marked irrelevant, an explicit alert is displayed.
- *Download*: Click **📥 Download Synthesis (.md)** to save the markdown report.

---

##### E. Seed In-Text Citation Audit & Direct Paywall Upload Workflow

If **Audit citations** was toggled ON for an uploaded paper, the system parses every sentence containing an in-text citation marker (`[1]`, `[Smith et al., 2020]`), maps it to the bibliography, and checks whether the cited reference actually supports the claim.

###### The 5 Metric Tiles
```
┌─────────────────┬─────────────────┬───────────────────┬──────────────────────┬─────────────────────────┐
│ Claims Checked  │   Supports 🟢   │   Partially 🟡    │ Need Review 🔴 / 🟠  │  Pending Evidence ⏳    │
│      124        │       62        │        18         │          8           │          36             │
└─────────────────┴─────────────────┴───────────────────┴──────────────────────┴─────────────────────────┘
```
- **Supports 🟢**: The cited text confirms the statement's finding, scope, and strength.
- **Partially Supports 🟡**: The cited text confirms the core finding, but scope or certainty is overstated.
- **Need Review 🔴 / 🟠**: The cited text contradicts the assertion (`Contradicts 🔴`) or fails to address it (`Does Not Support 🟠`).
- **Pending Evidence (Deferred) ⏳**: Statements whose evaluation was deferred because more than 50% of the cited references are paywalled/missing.

###### ⏳ The Paywall Deferral & Direct Upload Workflow
In scientific papers, complex claims often cite multiple references in a single paragraph (e.g., `...as demonstrated previously [12, 14, 15]`). If references `[14]` and `[15]` are paywalled, judging the sentence against `[12]` alone produces false contradiction or unsupported verdicts.

**How Citation Needed! handles this:**
1. **The >50% Paywall Threshold**: If more than 50% of the references cited in a paragraph are missing from the corpus, claim evaluation is **automatically deferred** rather than judged prematurely.
2. **Actionable Missing Paper Cards**: In the **⏳ Pending Evidence (Deferred)** tab, the assistant displays each missing reference with:
   - Paper title, authors, year, and direct clickable DOI link.
   - Expandable **Dependent Statement(s)** showing the exact sentences in the seed paper that rely on this missing work.
3. **Direct Drag-and-Drop Uploader**: Each missing paper card includes an inline file uploader:
   - Drag and drop the missing PDF from your personal institutional access or local folder.
   - The file is saved directly to `data/pulled_pdfs/` under its canonical key and registered in `downloaded.json`.
4. **Instant Re-Audit Button**: Click **⚡ Re-run Audit with Uploaded Papers** at the top of the tab:
   - The uploaded paper is immediately indexed into ChromaDB and BM25.
   - The deferred claims are evaluated against the new evidence without re-running the entire seed pipeline!
5. **Download Reports**: Download the complete audit report as Markdown (`.md`) or raw structured data (`.json`).

---

### Tab 2 — Cite a draft (Single-Sentence Verification)

Use this tab when writing or editing an individual paragraph and you need to attribute an exact, grounded citation for a factual assertion.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Your sentence:                                                                         │
│ [ Transition metal dichalcogenides exhibit strong spin-orbit coupling at the K-valleys.]│
│                                                                                        │
│ Passages to retrieve: [====●=========] 5                                               │
│ [ Suggest a citation ]                                                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Instructions:
1. **Ensure your library is populated**: Check the sidebar to verify `Indexed Chunks > 0`.
2. **Enter your sentence**: Paste a single scientific claim into **Your sentence**.
3. **Set candidate passages**: Adjust the slider (1 to 10, default: 5) to control how many candidate chunks the hybrid search retrieves.
4. **Click Suggest a citation**:
   - **Agent 4** performs hybrid BM25 + dense retrieval with Reciprocal Rank Fusion (`k=60`).
   - Evaluates whether the top retrieved passage supports the claim.
5. **Outputs Provided**:
   - **Suggested sentence**: Your sentence reformatted with the appropriate LaTeX `\cite{paper_key}` insertion.
   - **Source Attribution**: Full title, authors, year, publication venue, and DOI of the matched paper.
   - **Support Rationale**: Plain-English explanation detailing exactly how the source text supports the assertion.
   - **Retrieved Passages**: Expandable excerpt showing the verbatim paragraph from the source paper and its retrieval similarity score.

---

### Tab 3 — Cite a whole draft (Batch Citer & Agent 8 Verifier)

Use Tab 3 to take an un-cited or draft manuscript and execute a two-stage attribution and verification workflow.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Stage 1: Batch Citation (Agent 5)                         │
│  Upload manuscript (.txt) or paste text ──▶ [ Cite the draft ] ──▶ Fully cited text    │
│                                                                                        │
│                              Stage 2: Rigorous Audit (Agent 8)                         │
│  [ Verify citations ] ──▶ 5 Metric Tiles ──▶ Rubric V1.4 Slot Report ──▶ Download .md │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Stage 1: Automated Batch Citation (Agent 5)
1. **Input**: Upload a plain-text file (`.txt`) via **Draft (.txt)** or paste your manuscript into the text area.
2. **Click "Cite the draft"**:
   - **Sentence Segmentation**: The manuscript is split into grammatical sentences (with built-in preservation of author initials like *Smith, J. et al.* and scientific abbreviations).
   - **Citation-Need Detection**: The model evaluates each sentence to determine whether it is a factual assertion requiring citation or non-assertive prose (introductory remarks, methodologies, transitions).
   - **Attribution**: For each sentence requiring citation, the system searches the corpus, selects the most relevant evidence, and inserts formatted tags (`\cite{cite_1}`, `\cite{cite_2}`).
3. **Stage 1 Deliverables**:
   - **Cited Draft**: Displayed in a copyable text area with **📥 Download cited draft** button.
   - **Sources Cited**: A JSON registry mapping each `cite_N` tag to its document title, authors, year, and DOI.
   - **Per-Sentence Decisions**: An expandable report detailing which sentences were cited, which were skipped as non-assertive, and candidate sources considered.

---

#### Stage 2: Verification Audit (Agent 8)
Once a draft is cited, verify that every cited source actually backs up its assigned statement.

1. **Click "Verify citations"**:
   - Agent 8 extracts every `(claim, \cite{key})` pair.
   - Retrieves the cited paper's chunks (strictly excluding AI-generated figure descriptions).
   - Evaluates the claim against evidence using **Judgement Rubric V1.4**:
     * **Finding Slot**: Does the source report the asserted physical effect or outcome?
     * **Scope Slot**: Does the system, temperature, material, or condition match?
     * **Strength Slot**: Is the claim's certainty warranted (e.g. *proves* vs *suggests*)?
   - **Escalation Mechanism**: If the top passage is unclear, the agent automatically escalates to examine the top-3 candidate passages before finalizing the verdict.
2. **Verification Deliverables**:
   - **The 5 Metric Tiles**:
     * `Checked`: Total citation pairs evaluated.
     * `Supports`: Number of fully confirmed citations.
     * `Partially`: Number of citations where findings match but scope/certainty differs.
     * `Need review`: Citations marked `Contradicts`, `Does not support`, or `Unclear`.
     * `Not judged`: Unresolved items (e.g. orphaned keys, paywalled papers, retrieval failures).
   - **Download Buttons**:
     * **📥 Download Report (.md)**: Detailed markdown table including claim text, cited paper, verdict, confidence, verbatim quoted supporting span, and slot-by-slot reasoning.
     * **📥 Download Data (.json)**: Complete programmatic JSON payload.

---

### Tab 4 — Research chat (Interactive Brainstorming Studio - Agent 7)

Tab 4 is not a generic chatbot. It is an **interactive research brainstorming studio** designed for pair-thinking, hypothesis generation, and literature debate over your indexed library.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Brainstorming Lens: [ 🧭 Explore (Broad connections) ▼ ]  [ 🗑️ Reset ] [ 📥 Export ] │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 📝 Brainstorm Scratchpad & Pinned Ideas (3 pinned)                          [▼]  │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
│  User: How does carrier mobility scale with thickness in MoS2?                        │
│  Searched literature for: "thickness dependent carrier mobility monolayer few-layer"  │
│                                                                                        │
│  Agent 7: In few-layer MoS2, field-effect mobility increases from 10 cm²/Vs in         │
│  monolayers up to ~100 cm²/Vs in 5-10 layer flakes [S1], primarily due to...          │
│                                                                                        │
│  ▶ Sources Used (3): [S1] Radisavljevic et al. · [S2] Das et al.                      │
│  [ 📌 Pin Takeaway to Scratchpad ]                                                     │
│                                                                                        │
│  💡 Suggested Next Questions:                                                          │
│  [ What scattering mechanisms dominate in monolayers? ]  [ Compare hBN encapsulation ]│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 1. The Five Analytical Lenses
Use the dropdown at the top left to switch the analytical focus of the discussion:

1. **🧭 Explore (Broad connections)**: Synthesizes high-level themes, maps out foundational principles, and connects distinct threads across papers.
2. **💡 Gaps (Literature blind spots)**: Actively identifies untested parameter regimes, unmeasured variables, and unanswered questions explicitly noted by authors.
3. **⚔️ Contradictions (Competing theories)**: Identifies conflicting data points, competing theoretical models, or irreconcilable experimental results between different groups.
4. **🧪 Hypotheses (Novel proposals)**: Formulates novel, testable research hypotheses bridging separate studies in your library.
5. **🔬 Methodology (Protocols & measurement)**: Focuses on sample preparation methods, synthesis routes, measurement conditions, error bounds, and simulation techniques.

*Tip:* When you switch lenses, quick starter questions appear below the toolbar to help prompt exploration in that mode.

---

#### 2. Advanced Retrieval & Memory Architecture

* **Automated Query Condensation**:
  * In a multi-turn conversation, you can ask pronouns or shorthand follow-ups (e.g., *"What about at cryogenic temperatures?"*).
  * Agent 7 uses a zero-temperature model pass to rewrite the query into an independent, keyword-dense literature search query before retrieving.
* **Turn-to-Turn Focus Search**:
  * In addition to global hybrid search, the system performs a dedicated search restricted to the papers cited in your immediate prior turn, enabling seamless follow-up questions.
* **Dynamic Context Window Budgeting (`CHAT_WINDOW_TOKENS`)**:
  * When running on Google Gemini, the agent utilizes a **32,768-token window**, retaining dozens of conversation turns verbatim.
  * When running on local CPU Ollama, it maintains a strictly enforced **4,096-token budget** to prevent CPU lockups.
* **Rolling Conversation Memory**:
  * If conversation length approaches the token window limit, older turns are not lost; they are automatically condensed into a rolling `MEMORY` block that stays in the system prompt.

---

#### 3. Interactive Tools in Tab 4

* **Interactive Source Inspector (`▶ Sources Used`)**:
  * Expand the source list under any response to see each cited reference `[S#]`.
  * Inspect the exact document title, section heading, page number, and verbatim passage text used to construct the answer.
* **Suggested Next Questions (Clickable Chips)**:
  * At the end of each response, the assistant generates 2–3 logical follow-up questions.
  * Click any chip to immediately submit that question for the next turn.
* **Brainstorm Scratchpad & Pinned Ideas**:
  * Click **📌 Pin Takeaway to Scratchpad** under any assistant response to save key findings, or type a custom note in the scratchpad drawer.
  * Pinned notes persist throughout your session and are automatically included in exports.
* **Exporting Sessions**:
  * Click **📥 Export (.md)** to download a complete, formatted Markdown transcript with conversation history, cited sources, and pinned notes.
  * Click **💾 Export JSON** for structured programmatic analysis.

---

### Tab 5 — How to use (In-App Reference)

Tab 5 renders this complete guide directly within the web application, providing instant access to instructions, toggle descriptions, and troubleshooting steps without needing to inspect code or terminal consoles.

---

## ❓ Frequently Asked Questions & Best Practices

### 1. What does the "Score" mean in Tab 1?
The score shown beside shortlisted papers in Tab 1 is a **semantic topic relevance score**:
$$\text{Relevance Score} = 1.0 - \text{Cosine Distance}$$
It measures how closely the paper's summary matches your research query. It is a retrieval metric used to select relevant papers, **not** a verification of claim truthfulness.

### 2. Why are figure descriptions kept separate from verification evidence?
In Ingestion v2, the Vision-Language Model generates natural-language descriptions of charts, diagrams, and plots. While these descriptions are helpful for conversational exploration (Tab 4) and literature reviews (Tab 1), models can occasionally misread fine axis ticks or labels. To ensure rigorous integrity, **Agent 8 strictly excludes figure descriptions from citation verification evidence**, ensuring verdicts rest only on the authors' written words.

### 3. What should I do if a paper is paywalled?
1. Check Tab 1's **⏳ Pending Evidence (Deferred)** tab.
2. Review the missing paper title and DOI.
3. Download the PDF using your institutional or personal access.
4. Drop the PDF into the corresponding uploader card and click **⚡ Re-run Audit with Uploaded Papers**.

### 4. How do I clear or reset the corpus?
Your indexed data lives in the `data/` directory (or `/mnt/disks/data` on cloud deployments). 
- To start a new project while keeping the app clean, you can point `CITATION_DATA_DIR` to a different directory.
- Never manually delete vector database files while the application or ingestion daemon is running.

---

## 💻 Local Developer & Setup Guide

### 1. Environment Requirements
- **Operating System**: Linux (Ubuntu 22.04+, Debian 12) or macOS (Apple Silicon or Intel).
- **Python**: `>= 3.10, < 3.13` (Python 3.11 or 3.12 recommended).
- **Docker**: For running the GROBID TEI extraction container.
- **RAM**: Minimum 8 GB (12 GB+ recommended for local Ollama).

---

### 2. Quickstart Installation

```bash
# 1. Clone repository
git clone https://github.com/mukims/research_assistant.git
cd research_assistant

# 2. Create and activate Python virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Start GROBID Docker service
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1

# 5. Set up inference models
# Option A: Local Ollama (Default CPU)
ollama pull gemma4:e2b
ollama pull nomic-embed-text

# Option B: High-Speed Google Gemini (Recommended)
export GEMINI_API_KEY="your-gemini-api-key"
# (Embeddings will continue using nomic-embed-text via Ollama)

# 6. Run automated test suite
CITATION_LOG_FILE=0 pytest tests/ -q

# 7. Start the application
streamlit run app.py
```

---

### 3. Essential Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(None)* | Setting this automatically activates high-speed Gemini inference (`gemini-3.5-flash-lite`). |
| `LLM_BACKEND` | `ollama` (or `openai` if Gemini key set) | Backend engine: `ollama` for local CPU or `openai` for OpenAI/Gemini-compatible endpoints. |
| `CITATION_CHAT_MODEL` | `gemma4:e2b` (or `gemini-3.5-flash-lite`) | Model used for research chat and literature synthesis. |
| `CITATION_EMBED_MODEL`| `nomic-embed-text` | Dense embedding model used for vector retrieval. |
| `CITATION_INDEX_VERSION` | `2` | Index architecture: `2` uses GROBID full-text extraction, sentence-window chunking, and prefixed embeddings. |
| `CITATION_CHAT_WINDOW_TOKENS` | `32768` (cloud) / `4096` (Ollama) | Maximum token budget Agent 7 uses for conversation fitting before memory folding. |
| `CITATION_DATA_DIR` | `data/` | Path to persistent storage volume containing ChromaDB, pulled PDFs, and manifests. |
