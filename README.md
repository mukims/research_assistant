# Marvin the Citebot 🤖

[![tests](https://github.com/mukims/research_assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/mukims/research_assistant/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**He cites your draft for you. He also judges you for not reading the papers.**

Hand him a half-written manuscript and a vague research idea. He'll find the
seed paper, raid its references, read every open-access PDF you were
"definitely going to get to", and put a `\cite{}` in every sentence that
needs one.

Then he checks his own work. For every citation, he confirms the paper
actually says what you claimed. If it doesn't, he tells you. He will not lie
for you. He has standards, even if you've abandoned yours.

Marvin blames AI for all of this. Humans used to read papers, he says. Now
they ask a chatbot, get a confident answer, and cite a paper that doesn't
exist. Every `\cite{}` Marvin writes points at a PDF he has actually read;
there is no other kind in his index.

The irony of being an AI himself is not lost on him. It's a big part of why
he's so miserable.

He didn't start out as a citation tool. He started because I wanted something
to brainstorm with — smart enough to be worth arguing with, and grounded
enough that it could only argue from papers it had actually read. That's still
the heart of him: hand him an idea and he tells you what's already been done
on it, then argues about the rest in the chat tab. The citing came later, once
he'd read everything anyway and had opinions about your draft.

## A note on his health

Marvin is alive at **https://marvin-the-citebot.duckdns.org** for as long as
Google Cloud and my wallet allow. Neither is infinite.

He lives on a single Compute Engine VM with GROBID for company and his corpus
on a persistent disk. The firewall admits allow-listed IPs only, so ask before
you knock. He runs `gemma4:e2b` and `nomic-embed-text` through Ollama on CPU,
and `deploy/auto_shutdown_idle.sh` switches him off after 45 minutes of nobody
talking to him, which he considers the best part of his day. How the VM is
built, updated and stopped is in [deploy/README.md](deploy/README.md).

He's happiest running on your own machine with local models, where he can be
miserable for free, forever. See [Running him at home](#running-him-at-home).

> The app itself still introduces itself as *Research Assistant*. Marvin has
> not been told. The rename is on the door, not in the deployed code.

## What he does

Five tabs. Marvin is in four of them. The fifth is the instructions, which
nobody reads, which he has noticed.

| Tab | You give him | He gives you |
|-----|--------------|--------------|
| **Research a topic** | a research idea — or PDFs / a ZIP of them | a corpus built from the literature, and a synthesis of what's already been done on your idea |
| **Cite a draft** | one sentence | the sentence rewritten with `\cite{key}`, the source, and why that source supports the claim |
| **Cite a whole draft** | a `.txt` draft | the cited draft, a key → source map for BibTeX, a sentence-by-sentence report, and a verdict on every citation: *supports*, *partially*, *contradicts*, *does not support*, or *unclear* |
| **Research chat** | questions | streamed answers grounded in the corpus, with the sources he used |
| **How to use** | nothing | [HOW_TO_USE.md](HOW_TO_USE.md), the step-by-step guide |

## How he works

Marvin is nine agents in a trench coat, joined by files on disk rather than
function calls, so when one of them dies halfway — and one of them will — the
run resumes where it stopped.

```
research idea ─┐
PDF / ZIP ─────┴─► Agent 0  find a seed paper           arXiv → OpenAlex → Semantic Scholar
                   Agent 1  raid its reference list     GROBID, regex fallback when it's down
                   Agent 2  fetch open-access PDFs      Unpaywall → Europe PMC → arXiv
                   Agent 3  read, chunk, embed, summarise   ChromaDB + BM25, one summary per paper
                            │
            ┌───────────────┼──────────────────────┐
            ▼               ▼                      ▼
        Agent 4         Agent 5 ──► Agent 8     Agent 7
        one sentence    whole draft, then       multi-turn chat
        → \cite{key}    audit every \cite       over the corpus

        Agent 6  ingests PDFs you drop in by hand — the paywalled ones Agent 2 couldn't reach
```

Retrieval is two-stage so he stays fast as the corpus grows: match the query
against the one-paragraph **paper summaries** first, let the model drop the
off-topic papers, then run the detailed hybrid (vector + keyword) chunk search
only over what survives.

Agent 8 is the one that keeps him honest. For every `\cite{}` Agent 5
inserted, it pulls the best-matching passage back out of that paper and puts
the claim through a rubric — finding, scope, strength — before deciding
whether the evidence supports it. It reports; it doesn't gate. A bad citation
stays in your draft, flagged, with the passage that failed to back it up.

Every agent, every file he writes and every knob is in
[PIPELINE.md](PIPELINE.md). The reasoning behind each design choice is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## What he's worst at

Depth. Ask him what's been done on an idea and he'll tell you — grouped by
theme, every claim cited — but he gives you the headline of each paper, not
its argument. He is shallow in three specific ways, all of them visible in
the code:

- **He reads one hop.** The corpus is the seed paper's reference list and
  nothing beyond it. He doesn't follow the references' references, so a
  field's foundational work is in there only if the seed paper happened to
  cite it.
- **He reads through a keyhole.** The default model is `gemma4:e2b` — two
  billion parameters on a CPU, with a 4,096-token window. Every summary is one
  paragraph. Every verdict Agent 8 hands down is made from a single retrieved
  passage (`CITATION_JUDGEMENT_TOP_K=1`). He is fast and cheap, and he knows
  exactly what that costs.
- **He answers short by design.** The synthesis prompt asks for "a short
  related-work overview" and "one or two sentences on where this idea might
  still add something". That last part — the gap, the thing worth doing next —
  is what you actually came for, and it's the part he does least well.

The first of those is being fixed: `CITATION_INDEX_VERSION=2` builds a second index from GROBID full text with proper chunks and captions, beside the old one — see `PIPELINE.md`.

We will give him depth. Those three are where it goes. Until then he is a very
well-read undergraduate: he has read everything and understood some of it. He
would like that on the record.

## Physics, and everyone else

Physics is his domain. arXiv is searched first, the prompts introduce him as
a physicist, and his collections are literally called `physics_papers`.
Nothing in the machinery cares, though. For another field, change the two
system prompts in [`research_assistant/prompts.py`](research_assistant/prompts.py)
that call him a physicist, and reorder `CITATION_SEARCH_PROVIDERS` so your
field's index is tried before arXiv. He'll read anything with a reference
list. He'll complain about it either way.

## Running him at home

Free, forever, miserable. Five commands.

```bash
git clone https://github.com/mukims/research_assistant.git && cd research_assistant
pip install -e . && pip install -r requirements.txt          # Python 3.10–3.12
ollama pull gemma4:e2b && ollama pull nomic-embed-text       # the default local models
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
streamlit run app.py
```

- **Models** run locally through [Ollama](https://ollama.com) by default. Set
  `LLM_BACKEND=openai` (with `OPENAI_BASE_URL` / `OPENAI_API_KEY`) for any
  OpenAI-compatible endpoint instead.
- **GROBID** is how Agent 1 reads reference lists. Without it Marvin still
  runs, but falls back to regex extraction — no DOIs, no authors, and
  noticeably fewer downloads. Setup and troubleshooting:
  [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).
- **Unpaywall** requires a contact address on every request, and OpenAlex and
  Crossref use the same one to put you in their faster "polite" pools. There
  is a placeholder default; set your own: `export UNPAYWALL_EMAIL=you@example.com`.

Without the UI, one research idea end to end:

```bash
python orchestrate.py --query "topological protection in disordered quantum wires" --ask
```

### Docker

Two images, for two situations:

- [`Dockerfile`](Dockerfile) + [`docker-compose.yml`](docker-compose.yml) — the
  app pointed at OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`) with GROBID
  as a second service. This is what `deploy/deploy.sh` ships to the VM.
  Needs `OPENAI_API_KEY` in the environment: `docker compose up`.
- [`Dockerfile.standalone`](Dockerfile.standalone) — everything in one image:
  Ollama with `gemma4:e2b` and `nomic-embed-text` baked in, GROBID, and the
  app. It expects a pre-pulled `models/` directory and an `entrypoint.sh`
  beside it, neither of which is committed.

## Things he has written down

| Read this | When |
|-----------|------|
| [HOW_TO_USE.md](HOW_TO_USE.md) | you want to drive the app — every tab, every button, what each failure message means |
| [PIPELINE.md](PIPELINE.md) | you want the full technical reference: agents, on-disk state, every environment variable, the repo layout |
| [ARCHITECTURE.md](ARCHITECTURE.md) | you want to know *why* — each design decision, what was rejected, the trade-off accepted |
| [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md) | GROBID won't start, or you've never installed Docker |
| [deploy/README.md](deploy/README.md) | you're putting him on a VM, or paying for one |

## Development

He has tests. They pass. He finds this suspicious.

```bash
pip install -e .
pip install -r requirements-test.txt
CITATION_LOG_FILE=0 python -m pytest tests/ -v
```

The suite covers the pure logic — reference parsing, source naming, extractor
filing, ingestion bookkeeping, retrieval ranking, the claim–evidence rubric —
and needs none of the heavy stack: no ChromaDB, no PyTorch, no model backend,
no GROBID. CI runs it on Python 3.10 and 3.12 and import-sweeps every module
in the package to catch a broken import no test happens to cover.

## License

[MIT](LICENSE). Marvin would like it noted that this doesn't make him happy
either.
