---
title: "Research Assistant: Complete User & Command Guide"
subtitle: "A step-by-step handbook for non-programmers"
author: "Shardul Mukim"
date: "2026-09-08"
geometry: "margin=0.85in"
fontsize: 11pt
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - \usepackage{fvextra}
  - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
---

# Welcome

The **Research Assistant** is an autonomous AI assistant for scientific literature. You give it a research idea or a seed paper (PDF), and it automatically:

1. Finds relevant research papers.
2. Reads their bibliographies and extracts citations.
3. Downloads the legal, open-access PDFs of those references.
4. Builds a searchable knowledge base on your machine.
5. Synthesizes related work and inserts LaTeX `\cite{...}` citations into your draft papers.

You do **not** need to know programming or write code to use this tool. You only need to know how to copy, paste, and press **Enter** in your computer's **Terminal**.

> **Note on Purpose & Legal Responsibility:**  
> This project is built solely to support researchers and scholars who are not in the business of money-making and are already resource-constrained. The user is solely responsible for any legalities, terms of service, and copyright obligations that may come with retrieving, storing, and citing literature.

---

# Part 1: First-Time Setup (Do this once)

### Step 1: Open Terminal and Navigate to the Folder

Open your **Terminal** app (on Linux/Mac, search for "Terminal" and open it).

Type `cd ` followed by the path to where you downloaded or extracted this project, then press **Enter**:
```bash
cd path/to/research_assistant
```
*(Tip: You can also type `cd ` with a space and drag-and-drop the project folder from your file manager straight into the Terminal window).*

### Step 2: Create a Private Workspace

*(Note: The app requires Python 3.10, 3.11, or 3.12. Do not use Python 3.13 because one of the components, lxml, cannot compile on 3.13).*

Paste these two lines:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```
*(Your prompt will now start with `(.venv)`. This confirms you are in your private workspace).*

> **CRITICAL RULE FOR TERMINAL USERS:**
> When you close your Terminal window, it **forgets** that you turned on this private workspace.
>
> **Every single time you open a new Terminal window to use this app, you must activate it again:**
> ```bash
> source .venv/bin/activate
> ```
> Always check that **`(.venv)`** appears at the beginning of your command prompt before running any commands. If you ever see `command not found: streamlit`, it means you forgot this line!

### Step 3: Install the Components

Paste this command to install all necessary tools:
```bash
pip install -r requirements.txt
pip install -e .
```
*(This downloads required libraries. It takes 2 to 5 minutes depending on your internet connection).*

### Step 4: Download the Free AI Models

Make sure [Ollama](https://ollama.com) is installed and running on your computer. Then, paste these commands one by one to download the local AI models:
```bash
ollama pull nomic-embed-text
ollama pull gemma4:e2b-mlx
```
*(This is a one-time download of the embedding model and the local language model).*

### Step 5: Install Docker and Start GROBID

**GROBID** is the component that reads each paper's reference list properly. Without it the app still runs, but it finds references with no authors, years or DOIs — and it will not warn you that this has happened. Everyone on the project should have it running.

It needs **Docker**. The full walkthrough — installing Docker on your operating system, starting the server, and every error message it can produce — is in the companion handbook:

> **Research Assistant: GROBID Setup Guide** (`Research_Assistant_GROBID_Guide.pdf`)

Once Docker is installed, this is the whole of it:
```bash
docker pull grobid/grobid:0.8.1
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
```

Wait about 30 seconds, then confirm it is awake:
```bash
curl http://localhost:8070/api/isalive
```
*(You want to see the word `true`. If you see anything else, check the GROBID guide.)*

---

# Part 2: Starting the Web App (Recommended)

The easiest way to use the Research Assistant is via its **visual web browser interface**, where you can click buttons and upload files.

### 1. Launch the Web Interface

Copy and paste this command block:
```bash
source .venv/bin/activate
export UNPAYWALL_EMAIL="your.email@example.com"
export CITATION_LLM_MODEL="gemma4:e2b-mlx"
export CITATION_LAYOUT_DETECTION=0
streamlit run app.py
```

> **Note on `CITATION_LAYOUT_DETECTION` (`0` vs `1`):**
> * **Keep `0` (default for most users):** 95%+ of scientific citations and literature summaries rely entirely on written claims in the text. Setting `0` keeps the pipeline blazing fast, lightweight, and free of heavy GPU/Torch dependencies.
> * **Use `1` only if:** You specifically need the system to locate, crop, and cite visual charts, plots, or data tables from papers, and you have installed the optional layout stack from `requirements-layout.txt`.

### 2. Using the Browser App

Your web browser will automatically open at:
```text
http://localhost:8501
```

Inside the browser, you will see four main functional tabs:

* **Tab 1: Research a topic**
  * Type a research idea (e.g., `topological protection in disordered quantum wires`) OR upload your own PDF paper.
  * Click **Build corpus**.
  * Watch the progress live as papers are found, downloaded, and summarized.
* **Tab 2: Cite a draft**
  * Paste a single sentence from your paper (e.g., *Anderson localization suppresses diffusion in 1D.*).
  * Click **Suggest a citation**.
  * The assistant retrieves matching passages and suggests the exact reference with a LaTeX `\cite{...}` tag and an explanation.
* **Tab 3: Cite a whole draft**
  * Upload a `.txt` file containing your full paper draft.
  * Click **Cite the draft**.
  * The assistant inspects every sentence, determines which claims need citations, searches the corpus, and produces:
    1. A cited draft document.
    2. A `_citations.json` file ready for BibTeX.
    3. A sentence-by-sentence evaluation report.
* **Tab 4: Research chat**
  * Have a multi-turn conversation with all the papers in your library.
  * Type questions and receive streaming answers grounded in your downloaded PDFs.

### 3. Stopping the App

To stop the web app at any time, switch to your Terminal window and press **`Ctrl + C`**.

---

# Part 3: Command-Line Shortcuts (Typing Commands)

If you prefer to run specific tasks directly from the command line without opening the web browser, use these commands:

### 1. Research an Idea from Scratch
Searches arXiv/OpenAlex, downloads the seed paper, extracts its citations, downloads available references, and writes a related-work overview:
```bash
python orchestrate.py --query "topological protection in disordered quantum wires" --ask
```

### 2. Research Starting from a PDF File You Have
Seeds directly from a paper you already downloaded:
```bash
python orchestrate.py --seed-file "/path/to/your/paper.pdf" --ask
```

### 3. Research Starting from an Online URL
Seeds directly from an arXiv or PDF link:
```bash
python orchestrate.py --seed-url "https://arxiv.org/abs/2401.12345" --ask
```

### 4. Find a Citation for a Single Sentence
Suggests a citation for an individual sentence claim:
```bash
python -m research_assistant.agents.agent4_assistant --text "Anderson localization suppresses diffusion in 1D."
```

### 5. Cite an Entire Draft Document
Analyzes every sentence in `my_draft.txt` and writes the cited version to `cited_draft.txt`:
```bash
python -m research_assistant.agents.agent5_batch_citer --file my_draft.txt --out cited_draft.txt
```

### 6. Interactive Chat with Downloaded Papers
Chat with your paper library in the Terminal:
```bash
python -m research_assistant.agents.agent7_research_chat
```
* Type `/sources` to view the papers that backed the last answer.
* Type `/export` to save the chat transcript to a markdown file.
* Type `exit` to quit.

### 7. Automatic Background Folder Watcher
Runs quietly in the background. When you drop a PDF into the `data/raw/` folder, it automatically downloads references and adds them to your search library:
```bash
python watch.py
```

---

# Part 4: Everyday Quickstart (Using the App Again)

Whenever you open a fresh Terminal window in the future, follow this simple 3-step process:

### Step 1: Navigate to the folder and turn on `.venv` (MANDATORY)

Because Terminal starts fresh every time, you **must** tell it to activate your private workspace first:
```bash
cd path/to/research_assistant
source .venv/bin/activate
```
*(Always verify that `(.venv)` appears at the start of your line before proceeding).*

### Step 2: Start GROBID

Docker does not restart GROBID for you after a reboot, so start it again:
```bash
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
```
*(Already running? `docker ps` will list it, and you can skip this.)*

### Step 3: Run the App

Now simply start whichever mode you need:

* **To launch the visual web browser app (Recommended):**
  ```bash
  streamlit run app.py
  ```
* **Or to run a research topic from the command line:**
  ```bash
  python orchestrate.py --query "your research topic" --ask
  ```
* **Or to chat with your papers in the terminal:**
  ```bash
  python -m research_assistant.agents.agent7_research_chat
  ```

---

# Part 5: Troubleshooting Guide

| What you see | What it means | How to fix it |
| :--- | :--- | :--- |
| `command not found: python3.12` | Python 3.12 is not installed. | Install Python 3.12 (e.g. from python.org) or use Python 3.10 / 3.11. |
| Red error mentioning `lxml` during install | You are using Python 3.13. | Delete `.venv` folder and recreate it using `python3.12 -m venv .venv`. |
| `command not found: streamlit` | The workspace environment is inactive. | Run `source .venv/bin/activate` first. |
| The app opens but every answer errors | Ollama is not running or model missing. | Open the Ollama app, then run `ollama list` in Terminal to check models. |
| Red **GROBID** dot in the sidebar | GROBID extraction server is stopped. | Start it: `docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1`, or click **Start GROBID** in the sidebar. Do not skip this — the app carries on with much weaker references and does not warn you. |
| `docker: command not found` | Docker is not installed. | See the **GROBID Setup Guide** — Part 1. |
| `curl` on port 8070 says `Connection refused` | GROBID is still starting, or is not running. | Wait 30 seconds and retry, then check `docker ps`. |
| `Address already in use` | The app is already running in another window. | Find that Terminal window and press `Ctrl + C`, or close it. |

\vspace{1em}

> **Hobbes:** *Won’t inventing a robot be more work than making the bed?*  
> **Calvin:** *It’s only work if somebody makes you do it.*

