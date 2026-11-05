# 🤖 Marvin the Citebot — Complete User Guide

Welcome to **Marvin the Citebot**, your autonomous academic research assistant.

This system builds and searches an evidence-grounded academic literature corpus, automatically cites scientific manuscripts, verifies citations against full-text source papers with an explicit rubric, and facilitates interactive literature brainstorming.

---

## 🧭 Interface Overview (The 7 Tabs)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                  🤖 Marvin the Citebot                                                          │
│                                                                                                                                 │
│ [Tab 1: Citation auditor] [Tab 2: Cite draft] [Tab 3: Idea] [Tab 4: Chat] [Tab 5: Corpus] [Tab 6: Labs] [Tab 7: Help]          │
│ Audit in-text citations   Batch-cite & audit  Discover &    Interactive   Batch library   Planner &       Manual &             │
│ against open-access PDFs  manuscript          synthesize    studio        bulk ingestion  transfer (exp.) FAQ                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Tab | Purpose | Primary Input | Key Deliverables |
|---|---|---|---|
| **Tab 1: Citation auditor** | Upload a published paper, fetch open-access references, and audit whether citations support the claims | One PDF to audit; one or more PDFs, or a ZIP archive, to add to the library | In-text citation audit, 5 metric tiles, missing paywall upload cards, per-claim evidence viewer, synthesis across collection |
| **Tab 2: Cite a draft** | Batch-cite and rigorously audit an entire academic manuscript draft | Draft (`.txt`, `.md`, `.tex` or pasted) | Fully cited draft, BibTeX mapping, sentence-by-sentence citation report, Agent 8 verification audit with 5 metric tiles |
| **Tab 3: Research idea** | Discover literature from an idea, download open-access papers, and synthesize established findings & gaps | Topic query or arXiv/PDF link | Discovered seed paper, per-paper reading notes, structured 3-part synthesis, key evidence passages |
| **Tab 4: Research chat** | Conversational research studio grounded in your local library | Plain-English research questions | Token-by-token streaming answers, 5 analytical lenses, keyed sources `[S#]`, clickable follow-up chips, persistent scratchpad |
| **Tab 5: Corpus library** | Bulk ingest local files, scan directories, and inspect database state | Folder path, ZIP archive, or list of DOIs/arXiv IDs | Direct batch ingestion into LanceDB & BM25, EasyOCR/Vision LLM processing for scanned papers, and interactive corpus inspector |
| **Tab 6 (Labs): Agent task planner (experimental)** | Deconstruct scientific papers into dependency DAGs for autonomous AI agents | Paper PDF, TEI XML, or plain text | Dependency DAG flowchart, topological wave schedule, Antigravity agent specs, replication scorecard |
| **Tab 6 (Labs): Technique transfer (experimental)** | Discover and adapt methods from other scientific fields to solve target bottlenecks | Problem statement or computational bottleneck | Candidate technique cards, role-by-role analogy mappings, identifiability analysis, critic debate, prior-art verification, and Markdown reports |
| **Tab 7: How to use** | Interactive documentation, architecture guide, and FAQ | None | This user guide rendered live inside the application |

---

## ⚡ Sidebar Controls & Telemetry

The left sidebar provides persistent visibility and control over background operations and system resources:

### 1. Real-Time Telemetry & Progress
- **Status Indicator**: Shows `⚡ Pipeline Active` during execution or `🟢 Pipeline: Idle` when ready.
- **Stage Progression**: Tracks the active pipeline stage (Step 1: Discovery → Step 2: Seed Ingest → Step 3: Reference Extraction → Step 4: PDF Fetching → Step 5: Reference Ingestion & Synthesis).
- **Active Item & Progress Bar**: Displays the specific paper being downloaded, parsed, or summarized, with elapsed runtime.
- **Live Event Log**: Expandable rolling log showing the last 15 system events (e.g. `✅ Downloaded 14 reference PDFs`, `✓ Ingested 18 papers into index v3`).

### 2. Cooperative Pipeline Cancellation
- **"🛑 Stop pipeline" Button**: Appears in the sidebar and at the top of active tabs whenever a background job is running.
- **Persistent Job Cancellation Ledger**: Cancellation requests are atomically synchronized across `data/pipeline_status.json` and `data/jobs/<job_id>/status.json`. Flags persist across progress ticks without being overwritten, cleanly halting worker execution loops.
- **Safe Item-Boundary Abort**: Cancellation is checked between discrete items (between individual PDF downloads, PDF parsings, summary calls, and graph nodes). It **never** aborts mid-write or inside an in-flight LLM call, ensuring LanceDB, ChromaDB, and BM25 index files are never corrupted.

### 3. Corpus & System Metrics
- **Corpus Counters**: Total indexed documents, total chunks, median chunk size (typically ~1,000–1,200 characters), and total figure/table captions.
- **Storage Engine**: LanceDB zero-copy vector store (`physics_papers_v3`, `physics_summaries_v3`) with cosine metric and Tantivy FTS, or legacy ChromaDB server.
- **Active Backend Telemetry**: Displays the current inference backend (e.g. Google Gemini `gemini-3.5-flash-lite` or local Ollama `gemma4:e2b`), active embeddings (`embeddinggemma-2`), and GROBID service status.
- **GROBID Service Controls**: Includes buttons to inspect GROBID health (cached with 10s TTL), view diagnostics, or restart the container if running in Docker.

---

## 📑 Detailed Tab Instructions

---

### Tab 1 — Citation auditor (Seed Paper Citation Auditor & Ingestion)

Tab 1 is the dedicated engine for auditing citations in published research papers and ingesting paper collections into your local library.

#### 1. Input Options
Two buttons sit under the upload form. The button you press decides what happens; the number of files only has to agree with it.
* **Audit this paper (exactly one PDF)**:
  1. Drag and drop a single `.pdf` research manuscript. (A `.zip` works too if it holds exactly one PDF.)
  2. *(Optional)* Enter a **Research topic / question** to guide the final synthesis across the literature (e.g. `mechanical properties of MXene monolayers`). If left blank, the topic is automatically inferred from the paper's title.
  3. Configure toggles (Synthesize answer, Force re-run, Enable OCR).
  4. Click **Audit this paper**.
  5. *What happens:* The system ingests your seed paper, parses its bibliography using GROBID TEI, downloads all open-access references from Unpaywall, Europe PMC, and arXiv, indexes full text into sentence windows, writes a multi-paper synthesis, and executes a full scientific audit of the seed paper's in-text citations.
  6. If you uploaded several papers, Marvin stops with *"Audit takes exactly one paper. To index several, use **Add papers to library**."*
* **Add papers to library (one or more PDFs, or a ZIP archive)**:
  1. Drag and drop one or several PDFs, or a `.zip` archive containing papers.
  2. Click **Add papers to library**.
  3. *What happens:* Direct batch ingestion with no citation audit. All papers are parsed, chunked, embedded, and added to the LanceDB vector database (or legacy ChromaDB) and Tantivy/BM25 index in a single locked pass.

---

#### 2. The Three Execution Toggles

| Toggle | Default | What It Does | When to Change |
|---|---|---|---|
| **Synthesize answer** | `ON` | Generates a structured Map-Reduce synthesis of the shortlisted papers at the end of the run. | Turn `OFF` if you only want to download and index literature without generating a summary. |
| **Force re-run** | `OFF` | Bypasses cached seed papers and re-fetches all references from scratch. | Turn `ON` if you modified pipeline parameters or want to refresh cached web downloads. |
| **Enable OCR** | `ON` | Runs OCR on scanned papers or notes that have no digital text layer. | Turn `OFF` if your papers all have a text layer and you want faster ingestion. |

---

#### 3. Understanding Tab 1 Audit Outputs

Once execution finishes, Tab 1 displays the following sections:

##### A. The 5 Metric Tiles
```
┌─────────────────┬─────────────────┬───────────────────┬──────────────────────┬─────────────────────────┐
│ Claims Checked  │   Supports 🟢   │   Partially 🟡    │ Need Review 🔴 / 🟠  │  Pending Evidence ⏳    │
│      124        │       62        │        18         │          8           │          36             │
└─────────────────┴─────────────────┴───────────────────┴──────────────────────┴─────────────────────────┘
```
- **Supports 🟢**: The cited text confirms the statement's finding, scope, and strength.
- **Partially Supports 🟡**: The cited text confirms the core finding, but scope or certainty is overstated.
- **Need Review 🔴 / 🟠**: The cited text contradicts the assertion (`Contradicts 🔴`) or fails to address it (`Does Not Support 🟠`).
- **Pending Evidence (Deferred) ⏳**: Statements whose evaluation was deferred because more than 50% of the cited references in that paragraph are paywalled/missing.

##### B. Filter Tabs & Claim Expanders
- **Supported Tab**: Every verified claim with its verbatim quote from the cited paper.
- **Need Review Tab**: Every contradicted or unsupported claim with slot-by-slot breakdown and explanation.
- **⏳ Pending Evidence (Deferred) Tab**:
  - Missing paper cards with direct DOI links.
  - Inline drag-and-drop PDF uploaders to supply institutional copies.
  - **⚡ Re-run Audit with Uploaded Papers** button to re-evaluate without restarting from scratch.
- **All Citations Tab**: Complete registry of all in-text citations found.

##### C. Reports & Downloads
- Click **📥 Download Audit Report (.md)** for a formatted markdown report.
- Click **📥 Download Audit Data (.json)** for raw JSON data.

---

### Tab 2 — Cite a draft (Batch Manuscript Citer & Agent 8 Verifier)

Use Tab 2 to take an un-cited or draft manuscript and execute a two-stage attribution and verification workflow.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Stage 1: Batch Citation (Agent 5)                         │
│  Upload manuscript (.txt/.md/.tex) or paste ──▶ [ Cite the draft ] ──▶ Fully cited text│
│                                                                                        │
│                              Stage 2: Rigorous Audit (Agent 8)                         │
│  [ Verify citations ] ──▶ 5 Metric Tiles ──▶ Rubric V1.6 Slot Report ──▶ Download .md │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Stage 1: Automated Batch Citation (Agent 5)
1. **Input**: Upload a `.txt`, `.md` or `.tex` file via **Draft (.txt, .md, .tex)**, or paste your manuscript into the text area.
   - **LaTeX (`.tex`)**: only the text between `\begin{document}` and `\end{document}` is used. Comments, equations, figures and tables are dropped, section titles become paragraphs, `\emph`/`\textbf`/`\textit` are unwrapped, and any `\cite`, `\citep` or `\citet` already in the draft are kept as they are.
   - **The output is text with `\cite` keys, not a compilable `.tex` file.** Paste the cited text back into your manuscript yourself.
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
   - Evaluates the claim against evidence using **Judgement Rubric V1.6**:
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

### Tab 3 — Research idea (Literature Discovery & Synthesis)

Use Tab 3 to explore a new research topic or hypothesis from scratch without uploading a PDF.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Research idea:                                                                         │
│ [ topological protection in disordered quantum wires                                  ]│
│                                                                                        │
│ Seed paper URL (optional):                                                             │
│ [ https://arxiv.org/abs/2401.12345                                                    ]│
│                                                                                        │
│ [x] Synthesize answer   [ ] Force re-run                                               │
│ [ Explore Research Idea ]                                                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Instructions:
1. **Enter your topic**: Type your research question or idea into **Research idea**.
2. *(Optional)* **Anchor with a Seed URL**: Paste a direct arXiv or PDF link if you want the search anchored to a specific landmark paper.
3. **Configure Toggles**: Turn on **Synthesize answer** to get a structured literature review.
4. **Click "Explore Research Idea"**:
   - **Agent 0** searches arXiv, OpenAlex, and Semantic Scholar for candidate literature and downloads the seed paper.
   - **Agents 1 & 2** extract the seed's references and fetch open-access connected literature.
   - **Agent 3** chunks, embeds, and indexes all papers into LanceDB (or ChromaDB) and the Tantivy/BM25 index.
   - **Synthesis Engine** reads shortlisted papers and formulates a structured 3-part related-work synthesis.

#### Tab 3 Deliverables:
- **Seed & Downloads Card**: Lists the anchor paper and all successfully downloaded literature with DOIs.
- **Per-Paper Notes**: What each shortlisted paper establishes, its experimental/theoretical methods, and its caveats.
- **Related-Work Synthesis**:
  1. *What is established*: Consensus findings across papers.
  2. *Where the papers differ*: Competing hypotheses, contradictory observations, or divergent conditions.
  3. *The gap*: Unaddressed questions and future research directions.
- **Evidence Passages**: Verbatim excerpts from the literature with cosine similarity scores.

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

### Tab 5 — Corpus library (Bulk Ingestion, Folder Scan & OCR)

Tab 5 bypasses web discovery to ingest your existing literature libraries directly into LanceDB and the BM25 search index.

#### 1. Ingestion Sources
* **📁 Multi-File / ZIP Upload**:
  - Drag and drop dozens of `.pdf` files or an entire `.zip` archive.
  - Validates `%PDF-` file magic bytes, verifies file sizes (up to 500 MB total, 100 MB single file limits to prevent ZIP bombs), extracts titles/metadata, and stages them cleanly.
* **📂 Scan Local Directory**:
  - Point Marvin to any local directory on disk (e.g. `/Users/you/papers/`).
  - Toggle **Recursive scanning** to index nested subdirectories.
* **🌐 Paste DOIs / arXiv IDs / URLs**:
  - Enter a raw list of identifiers (one per line).
  - Resolves metadata across Unpaywall, Europe PMC, and arXiv, downloads open-access PDFs, and embeds them into the database.

#### 2. OCR Engine for Scanned Papers & Notes
Historically, photocopied reprints, older journal scans, handwritten lab notes, and whiteboard photos were discarded as `scanned_or_empty`. Marvin provides built-in OCR:
* **Automatic Detection**: Detects documents lacking digital text layers (< 50 characters/page) and routes them to OCR.
* **Engine Selection**:
  - **Local EasyOCR** (default): Runs offline on Apple Silicon MPS or NVIDIA CUDA GPU using bundled weights. Extracts text in natural reading order and de-hyphenates split words.
  - **Vision LLM Mode** (`vlm`): Routes page renderings to multimodal vision models (`gemini-3.5-flash-lite` or Ollama vision) for complex formulas, handwritten annotations, or low-contrast scans.

#### 3. Interactive Corpus Explorer
* Displays an interactive table of all indexed papers, with document keys, chunk counts, publication years, timestamps, and direct DOI/arXiv links.
* Search and filter papers by title or source key.
* Real-time metrics for total indexed papers, corpus chunks, and embedding configuration.

---

### Tab 6 — Labs (experimental)

Labs holds two features that work but are not yet as mature as the rest. Treat what they produce as leads, not findings. It has two sub-tabs: **Agent task planner (experimental)** and **Technique transfer (experimental)**.

#### Labs: Agent task planner (experimental — Scientific Paper-to-Agent Task Decomposition)

The Agent task planner (the first sub-tab under **Labs**) deconstructs complex scientific papers into structured Directed Acyclic Graphs (DAGs) of executable tasks assigned to specialized AI subagents for autonomous replication.

##### 1. Input Options & Automatic Workspace Creation
* **Select Ingested Paper**: Pick an existing paper from your local library or TEI extraction directory.
* **Upload New Paper**: Upload a new `.pdf`, GROBID `.tei.xml`, or plain text `.txt` methodology notes.
* **📁 Instant Workspace Materialization**: As soon as a paper is uploaded or deconstructed, the system automatically creates a dedicated root directory at `workspace_runs/<paper_id>/` (anchored safely outside persistent databases). It copies the manuscript and immediately drops:
  - `README.md`: Workspace overview and execution waves manifest.
  - `paper_summary.md`: Extracted abstract, theoretical methodology sections, and tables.

##### 2. Configuration & Synthesis
* **⚡ Fast Heuristic Mode**: Instantly generates an authentic replication graph covering all scientific phases without querying an external LLM (ideal for offline use and fast CI).
* **Execution Mode**: Choose between:
  - **🧪 Dry-Run Simulation**: Validates graph acyclicity, generates task rationales, checks prerequisite inputs, and writes deterministic dummy deliverables.
  - **🤖 Autonomous Agent Execution**: Uses specialized coding subagents to synthesize code, executes tasks against generated test harnesses in a subprocess, and applies Two-Cycle Reflexion with structured failure diagnostics if an assertion fails.
* Click **⚡ Deconstruct into Task Graph**.

##### 3. Complete Markdown Bundle Materialization
Upon deconstruction, the paper's root directory (`workspace_runs/<paper_id>/`) is automatically populated with a full suite of Markdown files:
- [`plan.md`](#): The complete replication DAG specification with wave sequencing, inputs, deliverables, and acceptance criteria.
- [`dag.md`](#): Interactive Mermaid flowchart dependency diagram.
- [`scorecard.md`](#): Quantitative rubric scorecard from the Two-Level Critic Engine.
- [`paper_summary.md`](#): Formatted abstract, methodology, equations, and tables extracted from the paper.
- [`rationale.md`](#): Master scientific hypothesis, physical invariants, and complexity bounds.
- [`rationales/<task_id>_rationale.md`](#): Individual task rationales documenting *Problem Knowledge*, *Scientific Insight*, and *Implementation Strategy & Complexity Bounds*.
- [`plan.json`](#) & [`antigravity_specs.json`](#): Machine-readable execution specifications.

##### 4. The Two-Level Critic Engine & Quantitative Rubric
Before any task executes, the graph is scrutinized by a Two-Level Critic:
* **MacroCritic**:
  - Checks for DAG cycles, self-loops, and missing upstream dependencies.
  - Enforces mandatory coverage across the **6 Scientific Replication Phases**: `environment_setup`, `data_preprocessing`, `model_implementation`, `sanity_smoke_test`, `training_or_simulation`, and `metric_evaluation` / `verification_audit`.
  - Flags **cross-task deliverable collisions** (`macro_duplicate_output_*` as `BLOCKING`) to prevent output ownership conflicts.
  - Warns against over-fragmented graph bloat (>12 tasks).
* **MicroCritic**:
  - Enforces operational instruction specificity (mathematical formulations, cutoff wavevectors).
  - Enforces formal `ArtifactContract` schemas (tensor shapes, dtypes, tabular columns, JSON keys).
  - Enforces single-threaded CPU safety (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`).
  - Flags intra-task duplicate outputs (`BLOCKING`), deliverable bloat (>8 deliverables), and unpruned tensor dimensionality bloat (>50,000 dimensions).
* **Rubric Scorecard (0–100)**: Computes quantitative scores for topology, contracts, and testability.

##### 5. Deliverables & Views
* **📊 Interactive DAG Flowchart**: Dynamic pan/zoom Mermaid diagram color-coded by task category and agent role.
* **📑 Execution Schedule & Tasks**: Tasks grouped into parallel execution waves with expandable task cards showing objectives, required input artifacts, deliverables, acceptance criteria checkboxes, and context excerpts from the paper.
* **🤖 Antigravity Agent Specs**: Formatted subagent invocation specs ready for multi-agent delegation.
* **🚀 Dispatch & Execution**: Single-click execution across waves with progress tracking, artifact hash calculation, and automated generation of a **Replication Scorecard** (`CERTIFIED_REPLICABLE`, `PARTIALLY_REPLICATED`, `REPLICATION_FAILED`).
* **📥 Downloads**: Export the complete plan as JSON, Markdown, or Antigravity JSON.

---

#### Labs: Technique transfer (experimental — Cross-Disciplinary Methodology Discovery)

The Technique transfer sub-tab (the second one under **Labs**) discovers and adapts techniques from unrelated scientific disciplines to overcome target bottlenecks in your research.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Technique Transfer Engine                                      │
│                                                                                                  │
│  [Problem / Bottleneck] ──▶ Card Retrieval ──▶ Analogical Mapping ──▶ Critic ──▶ Scored Report  │
│                               (Abstracted)       (Identifiability)      Debate     (Prior Art)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

##### 1. Input & Discovery Options
* **Target Problem or Bottleneck**: Enter your specific physical question or mathematical obstacle (e.g. `recovering continuous parameters from noisy spectra` or `phase retrieval in high-dimensional non-convex optimization`).
* **Example Starters**: Click any of the pre-filled chips to explore canonical transfer scenarios:
  - *Inverse problem parameter recovery under correlated noise*
  - *Topological protection in disordered one-dimensional lattices*
  - *Phase retrieval in high-dimensional non-convex optimization*
  - *Diffusion-based generative sampling with physical conservation laws*
* **Candidate Cards to Explore**: Slider (1 to 10 cards, default 3) controlling how many candidate methods from the card index are mapped and evaluated.
* **Prior Art Cutoff Year**: Optional filter to restrict prior-art validation to literature published before a certain year (useful for retrodiction benchmarks).
* **Proposal Temperature**: Sampling temperature (default 0.2) balancing conservative analogical mappings with creative cross-disciplinary transfers.

##### 2. The 4 Transfer Pipeline Stages
1. **Technique Cards & Bottleneck Abstraction**: Matches your problem against pre-extracted technique cards (`data/transfer/cards/`). Each card encapsulates an invariant method (its general objective, 2–5 operational steps, and verbatim source evidence) stripped of domain-specific nouns.
2. **Analogical Role-by-Role Mapping**: Formulates the transfer:
   - Maps source variables/entities directly to target domain counterparts.
   - Assesses mathematical identifiability (whether target observations mathematically determine unknown parameters).
   - Formulates concrete implementation steps, baseline benchmarks, specific predictions, and refutation criteria.
3. **Adversarial Critic Debate**: An adversarial reviewer raises strict domain objections (e.g., breakdown of scale separation, gauge ambiguity, non-convexity). The proposer must refute each objection with mathematical reasoning or concede the limitation.
4. **Deterministic Scoring & Prior Art Verification**:
   - Gating criteria: Rejects transfers with contradicted premises, fatal conceded assumptions, or unidentified parameters.
   - Depth checklist: Evaluates 10 structural transfer dimensions.
   - Prior-art search: Queries arXiv and OpenAlex to confirm that the suggested transfer is novel and has not already been published.

##### 3. Deliverables & Inspection Views
* **📄 Full Transfer Report**: Formatted Markdown document containing complete mapping tables, mathematical formulations, critic objections and rebuttals, refutation tests, and prior art status.
* **🌳 Methodology Search Tree**: Interactive Mermaid diagram visualizing the lineage from target bottleneck to source cards, transfer proposals, and final verdicts.
* **💾 Raw JSON**: Machine-readable JSON export of the entire transfer payload (`RunResult`).

---

### Tab 7 — How to use (In-App Reference)

Tab 7 renders this complete guide directly within the web application, providing instant access to instructions, toggle descriptions, and troubleshooting steps without needing to inspect code or terminal consoles.

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
Your indexed data lives in the `data/` directory. 
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
ollama pull embeddinggemma-2

# Option B: High-Speed Google Gemini (Recommended)
export GEMINI_API_KEY="your-gemini-api-key"
# (Embeddings will continue using embeddinggemma-2 via Ollama)

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
| `CITATION_USE_LANCEDB` | `true` | Default storage backend: LanceDB zero-copy vector store (`true`) or ChromaDB (`false`). |
| `CITATION_LANCEDB_PATH`| `data/lancedb` | Path to persistent LanceDB tables (`physics_papers_v3`, `physics_summaries_v3`). |
| `LLM_REQUEST_TIMEOUT` | `60` | Request timeout in seconds for LLM client calls. |
| `CITATION_EMBED_MODEL`| `embeddinggemma-2` | Embedding model for text and figure images. An index records which one built it and refuses a different one. A value pinned in `.env` or `data/.env` overrides this default. |
| `CITATION_INDEX_VERSION` | `3` | Which index to read and write. `3` is the current one (Docling/GROBID full text, sentence-window chunks, figure images, embeddinggemma-2). `1` and `2` are older, read-only indexes and need `CITATION_EMBED_MODEL=nomic-embed-text`. |
| `CITATION_CHAT_WINDOW_TOKENS` | `32768` (cloud) / `4096` (Ollama) | Maximum token budget Agent 7 uses for conversation fitting before memory folding. |
| `CITATION_DATA_DIR` | `data/` | Path to persistent storage volume containing LanceDB, ChromaDB, pulled PDFs, and manifests. |
