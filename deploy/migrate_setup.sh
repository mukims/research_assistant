#!/usr/bin/env bash
# Get a machine ready to run scripts/migrate_to_v3.py. Safe to re-run; it installs and
# starts things and never deletes or overwrites anything.
#
#   deploy/migrate_setup.sh [--venv DIR] [--skip-grobid] [--check-only]
#
# What it does, in order, and stops at the first thing it cannot fix:
#   1. Python 3.10-3.12 (the package refuses 3.13) and a virtualenv with requirements.txt
#      and this package installed (default DIR: .venv-migrate)
#   2. Ollama 0.36 or newer, running, with embeddinggemma-2 pulled
#   3. GROBID (grobid/grobid:0.8.1 in Docker, container "grobid", port 8070), started and answering
#   4. warns about a CITATION_EMBED_MODEL pin in .env / data/.env, and about having no summary model
#
# --check-only reports what is missing and changes nothing. The summary model is not pulled
# for you (gemma4:e2b is several GB): set GEMINI_API_KEY in .env, or `ollama pull gemma4:e2b`.
#
# Environment (all optional): PYTHON, OLLAMA_HOST, GROBID_SERVER, GROBID_CONTAINER_NAME,
# GROBID_DOCKER_IMAGE, GROBID_WAIT, CITATION_DATA_DIR.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-migrate"
SKIP_GROBID=0
CHECK_ONLY=0
GROBID_SERVER="${GROBID_SERVER:-http://localhost:8070}"
GROBID_CONTAINER_NAME="${GROBID_CONTAINER_NAME:-grobid}"
GROBID_DOCKER_IMAGE="${GROBID_DOCKER_IMAGE:-grobid/grobid:0.8.1}"
GROBID_WAIT="${GROBID_WAIT:-180}"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
case "$OLLAMA_URL" in *://*) ;; *) OLLAMA_URL="http://$OLLAMA_URL" ;; esac
DATA_DIR="${CITATION_DATA_DIR:-$ROOT/data}"
MODEL="embeddinggemma-2"
MIN_OLLAMA="0.36.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --skip-grobid) SKIP_GROBID=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
# act "<what>" cmd...   runs cmd, or only reports it under --check-only
act()  { local what="$1"; shift; if [ "$CHECK_ONLY" -eq 1 ]; then say "would: $what"; return 1; fi; say "$what"; "$@"; }
# version_ge A B  (dotted numbers): true when A >= B
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -n1)" = "$2" ]; }

# 1. Python -----------------------------------------------------------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null && { PY="$c"; break; }
  done
fi
[ -n "$PY" ] || die "no Python 3.10-3.12 found (the package requires >=3.10,<3.13). Install one, or set PYTHON=/path/to/python3.12"
"$PY" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' || die "$PY is $("$PY" -V 2>&1); need 3.10-3.12"
say "python: $("$PY" -V 2>&1) ($PY)"

if [ ! -x "$VENV/bin/python" ]; then
  act "create the virtualenv $VENV" "$PY" -m venv "$VENV" || { [ "$CHECK_ONLY" -eq 1 ] || die "could not create $VENV"; }
fi
if [ -x "$VENV/bin/python" ]; then
  if "$VENV/bin/python" -c 'import chromadb, fitz, lxml, rank_bm25, requests, research_assistant' 2>/dev/null; then
    say "python packages: already installed in $VENV"
  else
    act "install requirements.txt and the package into $VENV (a few minutes)" \
      bash -c "\"$VENV/bin/pip\" install -q -r \"$ROOT/requirements.txt\" && \"$VENV/bin/pip\" install -q -e \"$ROOT\"" \
      || { [ "$CHECK_ONLY" -eq 1 ] || die "pip install failed"; }
  fi
fi

# 2. Ollama -----------------------------------------------------------------
command -v ollama >/dev/null 2>&1 || die "ollama is not installed: https://ollama.com/download (need $MIN_OLLAMA or newer)"
OLLAMA_VERSION="$(curl -sf --max-time 5 "$OLLAMA_URL/api/version" 2>/dev/null | sed -n 's/.*"version" *: *"\([0-9.]*\)".*/\1/p')"
[ -n "$OLLAMA_VERSION" ] || die "Ollama is installed but not answering at $OLLAMA_URL: start it (open the app, or: ollama serve &)"
version_ge "$OLLAMA_VERSION" "$MIN_OLLAMA" || die "Ollama $OLLAMA_VERSION is older than $MIN_OLLAMA, which $MODEL needs: update Ollama"
say "ollama: $OLLAMA_VERSION"
if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$MODEL:"; then
  say "$MODEL: already pulled"
else
  act "pull $MODEL (about 1.3 GB)" ollama pull "$MODEL" || true
fi
BUILD="$(curl -sf --max-time 5 "$OLLAMA_URL/api/tags" 2>/dev/null | python3 -c "
import sys, json
for m in json.load(sys.stdin).get('models', []):
    if m['name'].startswith('$MODEL:'):
        print(m['digest'][:12])
" 2>/dev/null)"
[ -n "$BUILD" ] && say "$MODEL build: $BUILD (an index records this and warns if another machine's build differs)"

# 3. GROBID -----------------------------------------------------------------
if [ "$SKIP_GROBID" -eq 0 ]; then
  grobid_alive() { curl -sf --max-time 3 "$GROBID_SERVER/api/isalive" >/dev/null 2>&1; }
  if grobid_alive; then
    say "grobid: already answering at $GROBID_SERVER"
  else
    command -v docker >/dev/null 2>&1 || die "docker is not installed, and GROBID is not answering at $GROBID_SERVER"
    docker info >/dev/null 2>&1 || die "docker is installed but its daemon is not running"
    if docker inspect "$GROBID_CONTAINER_NAME" >/dev/null 2>&1; then
      act "start the existing container $GROBID_CONTAINER_NAME" docker start "$GROBID_CONTAINER_NAME" >/dev/null || true
    else
      act "run $GROBID_DOCKER_IMAGE as $GROBID_CONTAINER_NAME on 127.0.0.1:8070 (downloads about 1.5 GB; needs ~4 GB of RAM)" \
        docker run -d --name "$GROBID_CONTAINER_NAME" --init --ulimit core=0 -p 127.0.0.1:8070:8070 "$GROBID_DOCKER_IMAGE" >/dev/null || true
    fi
    if [ "$CHECK_ONLY" -eq 0 ]; then
      i=0; while [ "$i" -lt "$GROBID_WAIT" ]; do grobid_alive && break; sleep 3; i=$((i + 3)); done
      grobid_alive && say "grobid: answering" || die "GROBID did not answer within ${GROBID_WAIT}s (docker logs $GROBID_CONTAINER_NAME)"
    fi
  fi
fi

# 4. Configuration the migration trips over ---------------------------------
for f in "$ROOT/.env" "$DATA_DIR/.env"; do
  if [ -f "$f" ] && grep -Eq '^[[:space:]]*CITATION_EMBED_MODEL=' "$f"; then
    warn "$f sets CITATION_EMBED_MODEL, which overrides the $MODEL default and makes the migration refuse to start. Remove that line."
  fi
done
if [ -n "${GEMINI_API_KEY:-}" ] || { [ -f "$ROOT/.env" ] && grep -Eq '^[[:space:]]*(GEMINI_API_KEY|GOOGLE_API_KEY|OPENAI_API_KEY)=.+' "$ROOT/.env"; }; then
  say "summary model: an API key is configured (the migration's preflight sends one test request)"
elif curl -sf --max-time 5 "$OLLAMA_URL/api/tags" | grep -q '"name":"gemma4:e2b'; then
  say "summary model: local gemma4:e2b is present"
else
  warn "no summary model: set GEMINI_API_KEY in $ROOT/.env, or run: ollama pull gemma4:e2b (several GB, needs RAM to load)"
fi

say "ready. Next:"
echo "  cd $ROOT && source $VENV/bin/activate"
echo "  CITATION_DATA_DIR=<where v3 should live> PYTHONPATH=. python scripts/migrate_to_v3.py --from-data-dir <copied data> --dry-run"
