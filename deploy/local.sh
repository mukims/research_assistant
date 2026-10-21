#!/usr/bin/env bash
# Run Marvin on this machine only.
#
#   deploy/local.sh up [--no-grobid]   start the Chroma server and GROBID if needed, then the app on 127.0.0.1
#   deploy/local.sh chroma             start only the Chroma server (for scripts without the app)
#   deploy/local.sh down               stop the Chroma server this script started, and the GROBID container
#   deploy/local.sh status             what is answering, changes nothing
#
# The Chroma server is the one process that opens the vector store; the app and every
# script connect to it (config.chroma_server_from says why: two processes with the store
# open lost 17,201 vectors on 2026-10-21).
#
# The app listens on 127.0.0.1 and nowhere else; Streamlit's own default is every
# interface. GROBID is managed under the same container name and image the app's
# own Start/Stop buttons use (GROBID_CONTAINER_NAME, GROBID_DOCKER_IMAGE), so the
# two never fight. No other container is ever started, stopped or removed.
#
# Environment (all optional):
#   PYTHON                   interpreter that has the requirements installed (default python3)
#   PORT                     app port (default 8501)
#   GROBID_SERVER            default http://localhost:8070; a non-local address is left alone
#   GROBID_CONTAINER_NAME    default grobid
#   GROBID_DOCKER_IMAGE      default grobid/grobid:0.8.1; used only if the container does not exist
#   GROBID_WAIT              seconds to wait for GROBID to answer (default 120)
#   OLLAMA_HOST              default http://127.0.0.1:11434
#   CITATION_DATA_DIR        default <repo>/data; the server owns its physics_vectordb
#   CITATION_CHROMA_SERVER   default http://localhost:8000; empty means the app opens the files itself
#   CHROMA_WAIT              seconds to wait for the Chroma server to answer (default 60)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8501}"
GROBID_SERVER="${GROBID_SERVER:-http://localhost:8070}"
GROBID_CONTAINER_NAME="${GROBID_CONTAINER_NAME:-grobid}"
GROBID_DOCKER_IMAGE="${GROBID_DOCKER_IMAGE:-grobid/grobid:0.8.1}"
GROBID_WAIT="${GROBID_WAIT:-120}"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
DATA_DIR="${CITATION_DATA_DIR:-$ROOT/data}"
CHROMA_SERVER="${CITATION_CHROMA_SERVER-http://localhost:8000}"
CHROMA_WAIT="${CHROMA_WAIT:-60}"
CHROMA_PID="$DATA_DIR/chroma.pid"
CHROMA_LOG="$DATA_DIR/logs/chroma.log"
case "$OLLAMA_URL" in *://*) ;; *) OLLAMA_URL="http://$OLLAMA_URL" ;; esac

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# GROBID_SERVER split into host and port; only a local server is ours to start.
grobid_hostport="${GROBID_SERVER#*://}"; grobid_hostport="${grobid_hostport%%/*}"
grobid_host="${grobid_hostport%%:*}"
grobid_port="${grobid_hostport##*:}"
[ "$grobid_port" = "$grobid_hostport" ] && grobid_port=8070
case "$grobid_host" in localhost|127.0.0.1|0.0.0.0|"") grobid_is_local=1 ;; *) grobid_is_local=0 ;; esac

# CITATION_CHROMA_SERVER split the same way; only a local server is ours to start.
chroma_hostport="${CHROMA_SERVER#*://}"; chroma_hostport="${chroma_hostport%%/*}"
chroma_host="${chroma_hostport%%:*}"
chroma_port="${chroma_hostport##*:}"
[ "$chroma_port" = "$chroma_hostport" ] && chroma_port=8000
case "$chroma_host" in localhost|127.0.0.1|0.0.0.0|"") chroma_is_local=1 ;; *) chroma_is_local=0 ;; esac

chroma_alive() { [ -n "$CHROMA_SERVER" ] && curl -sf --max-time 3 "$CHROMA_SERVER/api/v2/heartbeat" >/dev/null 2>&1; }

chroma_up() {
  if [ -z "$CHROMA_SERVER" ]; then say "CITATION_CHROMA_SERVER is empty: the app opens the store's files itself"; return 0; fi
  if chroma_alive; then say "Chroma server already answering at $CHROMA_SERVER"; return 0; fi
  [ "$chroma_is_local" -eq 1 ] \
    || die "CITATION_CHROMA_SERVER=$CHROMA_SERVER is not answering, and it is not on this machine, so I will not try to start it."
  mkdir -p "$(dirname "$CHROMA_LOG")"
  say "Starting the Chroma server for $DATA_DIR/physics_vectordb on 127.0.0.1:$chroma_port (log: $CHROMA_LOG)"
  nohup "$PYTHON" -c 'import sys; from chromadb.cli.cli import app; sys.argv[0] = "chroma"; sys.exit(app())' \
    run --path "$DATA_DIR/physics_vectordb" --host 127.0.0.1 --port "$chroma_port" >>"$CHROMA_LOG" 2>&1 &
  echo $! > "$CHROMA_PID"
  local waited=0
  while [ "$waited" -lt "$CHROMA_WAIT" ]; do
    chroma_alive && { say "Chroma server is up (${waited}s)"; return 0; }
    sleep 1; waited=$((waited + 1))
  done
  die "the Chroma server did not answer within ${CHROMA_WAIT}s; see $CHROMA_LOG"
}

chroma_down() {
  local pid
  pid="$(cat "$CHROMA_PID" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    say "Stopping the Chroma server (pid $pid)"
    kill "$pid"
  else
    say "No Chroma server started by this script is running; nothing to stop"
  fi
  rm -f "$CHROMA_PID"
}

grobid_alive() { curl -sf --max-time 3 "$GROBID_SERVER/api/isalive" >/dev/null 2>&1; }
container_state() { docker inspect -f '{{.State.Status}}' "$GROBID_CONTAINER_NAME" 2>/dev/null; }
port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

ollama_line() {
  local tags
  if tags="$(curl -sf --max-time 3 "$OLLAMA_URL/api/tags" 2>/dev/null)"; then
    echo "up ($(printf '%s' "$tags" | grep -o '"name"' | wc -l | tr -d ' ') models installed)"
  else
    echo "not reachable (fine if LLM_BACKEND and CITATION_EMBED_BACKEND point at a hosted API)"
  fi
}

grobid_up() {
  if grobid_alive; then say "GROBID already answering at $GROBID_SERVER"; return 0; fi
  if [ "$grobid_is_local" -ne 1 ]; then
    warn "GROBID_SERVER=$GROBID_SERVER is not answering, and it is not on this machine, so I will not try to start it."
    return 0
  fi
  if ! docker info >/dev/null 2>&1; then
    warn "Docker is not running, so GROBID is unavailable. Reference lists fall back to regex extraction: no DOIs, no authors, fewer downloads."
    return 0
  fi
  local state; state="$(container_state)"
  case "$state" in
    running) say "GROBID container '$GROBID_CONTAINER_NAME' is running, waiting for it to answer" ;;
    "")      say "Creating GROBID container '$GROBID_CONTAINER_NAME' from $GROBID_DOCKER_IMAGE (first run downloads the image)"
             docker run -d --name "$GROBID_CONTAINER_NAME" -p "127.0.0.1:$grobid_port:8070" "$GROBID_DOCKER_IMAGE" >/dev/null ;;
    *)       say "Starting GROBID container '$GROBID_CONTAINER_NAME' (was $state)"
             docker start "$GROBID_CONTAINER_NAME" >/dev/null ;;
  esac
  local waited=0
  while [ "$waited" -lt "$GROBID_WAIT" ]; do
    grobid_alive && { say "GROBID is up (${waited}s)"; return 0; }
    state="$(container_state)"
    case "$state" in
      exited|dead)
        warn "GROBID container stopped right after starting ($state). Last log lines:"
        docker logs --tail 15 "$GROBID_CONTAINER_NAME" >&2 2>&1
        warn "Exit code 137 means the kernel killed it for memory: raise Docker Desktop's memory limit, or run with --no-grobid."
        return 0 ;;
    esac
    sleep 1; waited=$((waited + 1))
  done
  warn "GROBID did not answer within ${GROBID_WAIT}s; continuing without it."
}

cmd_status() {
  local app="not running" grobid="not answering" state=""
  port_in_use "$PORT" && app="listening"
  grobid_alive && grobid="alive"
  if [ "$grobid_is_local" -eq 1 ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    state="$(container_state)"
    [ -n "$state" ] && grobid="$grobid (container $GROBID_CONTAINER_NAME: $state)"
  fi
  local chroma="not answering"
  [ -z "$CHROMA_SERVER" ] && chroma="off (the app opens the files itself)"
  chroma_alive && chroma="alive"
  echo "app     http://127.0.0.1:$PORT   $app"
  echo "chroma  ${CHROMA_SERVER:-none}   $chroma"
  echo "grobid  $GROBID_SERVER   $grobid"
  echo "ollama  $OLLAMA_URL   $(ollama_line)"
}

cmd_up() {
  local manage_grobid=1
  for arg in "$@"; do
    case "$arg" in
      --no-grobid) manage_grobid=0 ;;
      *) die "unknown option: $arg (usage: deploy/local.sh up [--no-grobid])" ;;
    esac
  done
  "$PYTHON" -c "import streamlit" >/dev/null 2>&1 \
    || die "'$PYTHON' cannot import streamlit. Install the requirements into it, or point PYTHON at the interpreter that has them: PYTHON=/path/to/python deploy/local.sh up"
  port_in_use "$PORT" && die "port $PORT is already taken on 127.0.0.1. Is Marvin already running? Use PORT=<free port> for a second copy."
  chroma_up
  if [ "$manage_grobid" -eq 1 ]; then grobid_up; else say "Skipping GROBID (--no-grobid)"; fi
  echo "ollama  $OLLAMA_URL   $(ollama_line)"
  say "Marvin on http://127.0.0.1:$PORT  (this machine only; Ctrl-C stops the app, 'deploy/local.sh down' stops GROBID)"
  cd "$ROOT" && exec "$PYTHON" -m streamlit run app.py \
    --server.address 127.0.0.1 --server.port "$PORT" --browser.gatherUsageStats false
}

cmd_down() {
  chroma_down
  docker info >/dev/null 2>&1 || die "Docker is not running, so there is no GROBID container to stop."
  if [ "$(container_state)" = "running" ]; then
    say "Stopping GROBID container '$GROBID_CONTAINER_NAME'"
    docker stop "$GROBID_CONTAINER_NAME" >/dev/null
  else
    say "GROBID container '$GROBID_CONTAINER_NAME' is not running; nothing to stop"
  fi
}

case "${1:-}" in
  up)     shift; cmd_up "$@" ;;
  chroma) chroma_up ;;
  down)   cmd_down ;;
  status) cmd_status ;;
  *)      echo "usage: deploy/local.sh up [--no-grobid] | chroma | down | status" >&2; exit 2 ;;
esac
