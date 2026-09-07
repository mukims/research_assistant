# Handoff

*Updated 2026-09-07, at completion.*

This file was written mid-build as a session-boundary note. The build is now finished, so
what follows is the closing state rather than a resume guide.

## Status

The merge is complete. All 14 planned tasks landed on `build/merged-pipeline`, each one
reviewed and each ending in a commit. **131 tests pass**, and CI's full sequence —
`pip install -r requirements-test.txt`, `pip install -e .`, pytest, then a 24-module
import sweep — was reproduced end to end in a clean Python 3.12 environment.

## Where to look

| For | Read |
|---|---|
| What the project is and how to run it | [README.md](README.md) |
| How to drive the app | [HOW_TO_USE.md](HOW_TO_USE.md) — also rendered in the app's fifth tab |
| Why the design is what it is | [ARCHITECTURE.md](ARCHITECTURE.md) |
| The design as originally drafted | [the spec](docs/superpowers/specs/2026-09-07-merged-research-assistant-design.md) |
| **Where the code departs from that spec, and what is knowingly left open** | **[DECISIONS.md](docs/superpowers/DECISIONS.md)** |

Read `DECISIONS.md` before trusting the spec — ten parts of the spec turned out to be
wrong once the real code was read, mostly fields it named that did not exist or omitted
that did.

## What is not verified

Everything was verified offline: tests, mutation checks, import sweeps, and a clean-room
dependency install. **Nothing has been run against a live model or a real GROBID server**,
and no end-to-end corpus has been built. That needs an Ollama daemon (or an
OpenAI-compatible endpoint via `LLM_BACKEND=openai`) and is the obvious next step.

## Environment

The project declares Python `>=3.10,<3.13`. The upper bound is real — `lxml==4.9.4` has no
cp313 wheel and will not compile against 3.13's C API. The repo's own `.venv` is 3.13 and
predates that constraint; it still runs the suite because it was installed editable
beforehand, but rebuild it with a 3.10–3.12 interpreter.
