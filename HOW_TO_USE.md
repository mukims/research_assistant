# How to use the Research Assistant

This is the guide to *driving* the app. For what each stage does and why, see
[ARCHITECTURE.md](ARCHITECTURE.md).

The app renders this file itself, under the **How to use** tab — so if you are
reading this inside the app, it is already running and you can skip straight to
[Before you start](#before-you-start-read-the-sidebar).

---

## Setting it up from scratch

Written for someone who does not write code. You will be typing commands into
**Terminal** — on a Mac, press `Cmd + Space`, type `Terminal`, press Enter. Copy
each block below, paste it in, press Enter, and wait for it to finish before
moving to the next one.

Every command here was run on a clean machine before being written down.

### Step 0 — Check you have the right Python

```bash
python3.12 --version
```

You should see `Python 3.12.something`. If you get "command not found", install
Python 3.12 from [python.org/downloads](https://www.python.org/downloads/) and
run it again.

**Python 3.13 will not work.** One of the components (`lxml`) has no 3.13 build
and the install fails partway through with a wall of red text. Use 3.10, 3.11 or
3.12. This is the single most common way to get stuck.

### Step 1 — Go to the project folder

```bash
cd ~/Downloads/research_assistant
```

If you put the project somewhere else, use that path instead. You can drag the
folder onto the Terminal window and it will paste the path for you.

### Step 2 — Create a private space for the project's components

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. That means it worked. **You need to
run that second line again every time you open a new Terminal window** — it is
how Terminal knows to use this project's components rather than your system's.

### Step 3 — Install everything

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

This downloads a few hundred megabytes and takes several minutes. Some yellow
warning text is normal; red `ERROR` lines are not.

Check it worked:

```bash
python -c "import app; print('install OK')"
```

You want to see `install OK`. Warnings about "missing ScriptRunContext" are
expected here and harmless.

### Step 4 — Install the AI models

The assistant needs a language model to read and write with. Install
[Ollama](https://ollama.com/download), open it once so it is running, then:

```bash
ollama pull nomic-embed-text
ollama pull gemma4:e2b-mlx
```

That is roughly 8 GB and will take a while on a normal connection.

### Step 5 — Set your options

```bash
export UNPAYWALL_EMAIL="your.name@example.com"
export CITATION_LLM_MODEL="gemma4:e2b-mlx"
export CITATION_LAYOUT_DETECTION=0
```

Put your real email address in the first line. It is not used for marketing —
the free services that supply the papers (Unpaywall, Crossref, arXiv) ask
who is calling, and sending them the built-in placeholder is rude and can get
you rate-limited.

The second line matters: the built-in default is `gemma4:31b-cloud`, which runs
on **Ollama's servers**, not yours — it needs an Ollama account (`ollama signin`)
and sends your text off your machine. Setting it to `gemma4:e2b-mlx` uses the
model you just downloaded and keeps everything local. It is smaller, so answers
are less polished; if you would rather have the bigger cloud model, run
`ollama signin` and leave this line out.

The third line turns off figure and table extraction, which needs a large
scientific-imaging component that is difficult to install and is not needed for
citations. Leave it off unless you specifically want figure crops.

Like Step 2, these three lines have to be re-run in each new Terminal window.

### Step 5b — Start GROBID

Agent 1 uses GROBID to read reference lists properly. It runs as a container:

```bash
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
```

Give it about 30 seconds, then check it answers:

```bash
curl http://localhost:8070/api/isalive
```

You want `true`. If you do not have Docker yet, or that command fails,
[Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md) walks
through installing it and every error it can produce.

The app runs without GROBID, but it runs *worse* and it does not tell you —
see **If GROBID is red** below. Start it before the app.

### Step 6 — Start it

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. That is the app.

To stop it, click the Terminal window and press `Ctrl + C`.

### Starting it again next time

Once set up, you only need this:

```bash
cd ~/Downloads/research_assistant
source .venv/bin/activate
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
export UNPAYWALL_EMAIL="your.name@example.com"
export CITATION_LLM_MODEL="gemma4:e2b-mlx"
export CITATION_LAYOUT_DETECTION=0
streamlit run app.py
```

(The GROBID line is only needed if it is not already running. `docker ps` tells
you.)

### When something goes wrong

| What you see | What it means |
|---|---|
| `command not found: python3.12` | Python 3.12 isn't installed — see Step 0. |
| A wall of red text mentioning `lxml` during Step 3 | You are on Python 3.13. Delete the `.venv` folder and redo Step 2 with `python3.12`. |
| `command not found: pip` or `streamlit` | You skipped `source .venv/bin/activate`. Run it and try again. |
| `No module named 'research_assistant'` | The `pip install -e .` in Step 3 didn't finish. Run it again. |
| The app opens but every answer errors | Ollama isn't running, or the model name is wrong. Open the Ollama app, then run `ollama list` and make sure `gemma4:e2b-mlx` is in it. |
| Red **GROBID** dot in the sidebar | GROBID is not running. Start it (Step 5b). The app works without it, but with visibly weaker references — see the next section. |
| `Address already in use` | The app is already running in another Terminal window. |

---

## What it does

You give it a one-line research idea. It finds a paper on that topic, reads that
paper's reference list, downloads the references it can legally get, and builds
a searchable corpus out of them.

Then it answers three kinds of question, one per tab:

- **What has already been done on this idea?** — a related-work synthesis
  (Tab 1, at the end of a build).
- **What should I cite for this sentence?** — paste one sentence of draft text
  and get it back with a `\cite{key}` inserted, plus why that source supports
  the claim (Tab 2).
- **What should I cite across this whole draft?** — upload or paste a full
  document and get every citable sentence cited at once, with a mapping file
  and a per-sentence report (Tab 3).

There's also a **Research chat** tab (Tab 4) for open-ended, multi-turn
conversation grounded in the same corpus, when a single sentence or a fixed
question isn't the shape of what you want to ask.

It builds its corpus from one seed paper's references. It is not a literature
search over everything ever published — the corpus is only as broad as the
seed paper's bibliography (plus anything you drop into `data/pulled_pdfs/` by
hand, which Agent 6 picks up automatically — see Tab 4's note below on where
manual PDFs go).

---

## Before you start: read the sidebar

The sidebar is the app's status panel. Check it first, because two of the three
things there decide whether a run can succeed at all.

| Row | What it means |
|---|---|
| **Papers / Chunks** | How much is in the corpus right now. `0 / 0` means nothing has been ingested yet — Tabs 2–4 will have nothing to search. |
| **Chat / Embeddings** | Which models are wired up, and via which backend. |
| **Layout** | `text-only` is the fast path, and what the packaged and Docker builds use. `on` means layout detection is enabled — it needs detectron2 and costs roughly 15s per figure. Set `CITATION_LAYOUT_DETECTION=0` unless you need figure crops. |
| **GROBID** | Green: reference extraction (Agent 1) is using the richer GROBID path. Red: GROBID isn't responding for this app instance. |

**If GROBID is red, start it before you go further** (Step 5b, or the sidebar
button). The run will complete without it, but with weaker references. Agent
1 automatically falls back to pulling a numbered reference list straight out
of each PDF's text when GROBID is unreachable (or comes back empty for a
particular paper). That fallback has no authors, years or DOIs, so Agent 2
has less to work with when deciding what to fetch, and some references that
GROBID would have resolved confidently may not download at all — but the
pipeline itself does not stop. If Agent 1 or Agent 2 seem to be doing
noticeably worse than usual, check GROBID before anything else.

You can manage GROBID directly from the app's sidebar under **Server & Services**
using the **▶️ Start GROBID**, **⏹️ Stop**, **🔄 Restart**, and **🩺 Status**
buttons, which run the GROBID manager agent to inspect Docker, launch the container,
and verify health. Alternatively, launch it manually from your terminal:

```bash
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
```

`http://localhost:8070` is already the app's own default, so there is nothing
further to set when GROBID runs on the same machine. Point `GROBID_SERVER` at a
hosted instance only if you are using someone else's server.

Installing Docker, starting the container and reading its errors are covered in
[Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).

---

## Tab 1 — Research a topic

This is the tab that builds the corpus. Everything else depends on it. You can begin from your own PDF or let the assistant search for one:

### Option A: Upload a seed PDF
1. **Seed paper (.pdf)** — Upload any research paper in PDF format. The assistant seeds directly from your uploaded paper, indexes it, and mines its bibliography to build the corpus.
2. **Research topic / question** *(optional)* — Specify a query or question to guide the final related-work synthesis. If left blank, the topic is automatically inferred from the paper's title or filename.
3. **Answer my query at the end** *(on by default)* — runs the related-work synthesis once ingestion finishes.
4. **Force re-run every stage** *(off by default)* — ignores saved state and redoes everything.
5. **Build corpus from PDF.** Progress appears live as each stage finishes.

### Option B: Search for a paper
1. **Research idea** — one line of plain English. This gets used as a search
   query against arXiv, OpenAlex and Semantic Scholar, so it should read like a
   topic, not a question.

   - Good: `topological protection in disordered quantum wires`
   - Good: `spin-orbit coupling in monolayer transition metal dichalcogenides`
   - Poor: `what is the best way to protect qubits?` — conversational phrasing
     retrieves badly.

2. **Seed paper URL** *(optional)* — an arXiv link or a direct `.pdf` URL. Use
   this when you want to control which paper the corpus is built from instead of
   letting the search pick. It is also the fallback when the search finds
   nothing open-access.

3. **Answer my query at the end** *(on by default)* — runs the related-work
   synthesis once ingestion finishes. Turn it off if you only want the corpus.

4. **Force re-run every stage** *(off by default)* — ignores saved state and
   redoes everything. Normally leave this off: every stage records what it has
   already done, so a re-run resumes rather than repeats.

5. **Build corpus.** Progress appears live as each stage finishes.

### What you should expect to see

The seed paper appears first, then downloaded PDFs accumulate, then the
shortlist, then the synthesis. A normal run ends with substantially fewer papers
than the seed's reference list — most references are paywalled, and those are
skipped rather than treated as errors.

**Timing.** Minutes, not seconds. On local Ollama over CPU, expect considerably
longer — model calls dominate. The reference-fetching stage is deliberately
rate-limited to stay polite to arXiv and Unpaywall.

---

## Tab 2 — Cite a draft

**This needs a corpus.** If Papers is `0`, build one in Tab 1 first.

1. Paste a sentence of your draft — one claim, not a paragraph. The retrieval
   matches a specific assertion against specific passages, so
   `Anderson localization suppresses diffusion in 1D.` works far better than
   three sentences of background.
2. Set how many passages to retrieve. More passages give the model more to weigh
   but dilute precision; the default is a sensible starting point.
3. **Suggest a citation.**

You get the sentence back with a `\cite{key}` inserted, the sources it drew on,
and an explanation of why each supports the claim. Expand the retrieved-context
panel to see the actual passages — **do this before trusting the citation.** The
model is choosing among what retrieval handed it; if the corpus has nothing
genuinely on-point, it will still pick the closest thing.

Use this tab for one claim at a time. For a whole document, use Tab 3 instead.

---

## Tab 3 — Cite a whole draft

**This also needs a corpus.** This is Tab 2's job applied to an entire
document at once (Agent 5), rather than one sentence pasted by hand.

1. **Upload a `.txt` file, or paste the whole draft** into the text box below
   the uploader.
2. **Cite the draft.** The pipeline splits the text into sentences, asks the
   model in one batched pass which sentences actually make a claim worth
   citing (very short sentences are skipped automatically), then retrieves
   context and attempts a citation for each one that needs it.
3. Read the **cited draft** in the result box, or download it.

What comes back:

- **The cited draft** — your text with `\cite{cite_N}` keys inserted wherever
  the corpus supported a claim.
- **Sources cited** — a JSON mapping from each `cite_N` key to the full source
  citation it stands for, ready to feed into a BibTeX workflow.
- **Per-sentence decisions** (an expander) — for every sentence: whether it was
  cited, skipped as not needing a citation, or retrieved context but the model
  declined to cite it, plus the model's reasoning either way.

**If nothing comes back and you see an error instead:** the batched
citation-need check couldn't be lined up against the draft's own sentences (a
malformed model reply), and the run aborts *before writing anything* rather
than guess which verdict belongs to which sentence. Try again, or shorten the
draft — a very long draft is more likely to make the model truncate its reply
mid-list.

As with Tab 2: expand **Also retrieved, not cited** / check the report before
trusting any individual citation. A batch run makes many small decisions
unattended; spot-check a few before using the output.

---

## Tab 4 — Research chat

**This also needs a corpus.** Unlike Tabs 2 and 3, this is a genuine back-and-
forth (Agent 7) — ask a question, get an answer grounded in the retrieved
passages, ask a follow-up that refers back to what was just said, and keep
going.

1. Type a question in the chat box and send it. The answer streams in token by
   token rather than appearing all at once.
2. If the answer drew on retrieved passages, a **Sources** expander appears
   underneath it — check what actually backed the answer, the same way you
   would for a single citation.
3. **Clear** resets the conversation. **Export conversation** saves the full
   transcript to a timestamped markdown file under `data/drafts/` so you can
   come back to it later.

This tab is for exploring an idea, not for producing a citable sentence or
document — use Tab 2 or Tab 3 for that. It's also the tab to reach for when a
paper you need isn't showing up from Tab 1's automatic search: drop the PDF
into `data/pulled_pdfs/` by hand (or run
`python -m research_assistant.agents.agent6_manual_ingestor --once <file>`)
and it's ingested the same way a fetched paper would be, then it's part of
what every tab — chat included — can retrieve from.

---

## Where your work is stored

The corpus, the downloaded PDFs and the JSON manifests persist between runs,
under `data/` by default.

| How you run it | Where it goes |
|---|---|
| From a checkout | `<project>/data/` |
| Docker | Whatever you mounted at `/home/user/data` — **mount something, or it is lost** |
| Cloud Run | In memory, **lost when the instance scales to zero** unless a volume is mounted |

`CITATION_DATA_DIR` overrides this everywhere.

---

## Things that go wrong

**"No open-access PDF found for that query."** The search found candidates but
every one was paywalled. Either switch Tab 1 to **Upload a seed PDF** and give
it a paper you already have, or paste an arXiv or direct-PDF link into *Seed
paper URL*, and run again. The app will still answer from the model's own
knowledge, but that answer is ungrounded — it is not backed by any corpus.

**"Could not load seed PDF."** The file you uploaded isn't a PDF — the app
checks the file's actual contents, not its name, so something renamed to `.pdf`
is caught here. Re-export it as a real PDF and try again.

**References look thin — no authors, no years, no DOIs.** GROBID was down (or
returned nothing for that paper) and Agent 1 fell back to pattern-matching a
plain reference list. Check the sidebar's GROBID indicator; the run itself
still completed. Start GROBID and re-run to get the full metadata —
[Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).

**Most downloads fail.** Expected outside physics. arXiv coverage is excellent;
biomedical and chemistry much less so. Failures are recorded, not fatal. A
paper that failed for a settled reason (paywalled, not indexed, 404) is not
tried again; one that failed because the network dropped, or the server was
busy, is retried on your next run.

**Citations look plausible but wrong** (Tabs 2 and 3). Check the retrieved
passages. A small or off-topic corpus produces confident, badly-grounded
suggestions — this is the single most important thing to verify manually.

**A batch citation run (Tab 3) wrote nothing at all.** The citation-need check
couldn't be aligned back to the draft's sentences — see Tab 3 above. Re-run,
or shorten the draft.

**It is very slow.** Local Ollama on CPU is the usual cause; a hosted backend is
dramatically faster. If Layout is not `text-only`, figure processing is costing
you roughly 15 seconds per figure.

---

## A first run, end to end

1. Check the sidebar — GROBID green (or accept the regex fallback), models
   listed.
2. Tab 1: pick **Search for a paper**, enter
   `topological protection in disordered quantum wires`, leave the URL blank,
   leave both toggles as they are, **Build corpus**. (Or pick **Upload a seed
   PDF** and give it a paper of your own — the topic is then inferred from the
   paper if you leave the question blank.)
3. Wait. Watch the papers count in the sidebar climb.
4. Read the related-work synthesis at the end.
5. Tab 2: paste `Anderson localization suppresses diffusion in 1D.` and
   **Suggest a citation**. Expand the retrieved passages and check the
   citation is actually supported.
6. Tab 3: paste a short multi-sentence draft and **Cite the draft** to see the
   same idea applied to a whole document, with its report.
7. Tab 4: ask a follow-up question that Tab 1's synthesis didn't quite answer,
   and see it stream in, grounded in the same corpus.
