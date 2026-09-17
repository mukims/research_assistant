#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting Marvin the Citebot Standalone Container"
echo "=========================================================="

OLLAMA_PID=""
GROBID_PID=""
STREAMLIT_PID=""

cleanup() {
    echo ""
    echo "Caught stop signal. Shutting down background services..."
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "Stopping Streamlit (PID $STREAMLIT_PID)..."
        kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
    fi
    if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Stopping Ollama (PID $OLLAMA_PID)..."
        kill -TERM "$OLLAMA_PID" 2>/dev/null || true
    fi
    if [ -n "$GROBID_PID" ] && kill -0 "$GROBID_PID" 2>/dev/null; then
        echo "Stopping GROBID (PID $GROBID_PID)..."
        kill -TERM "$GROBID_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "Services stopped cleanly."
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# 1. Start Ollama daemon
echo "[1/3] Starting Ollama daemon..."
export OLLAMA_MODELS=/opt/ollama/models
export OLLAMA_HOST=127.0.0.1:11434
/usr/bin/ollama serve &
OLLAMA_PID=$!

# 2. Start GROBID service with -XX:-UseContainerSupport to prevent cgroupv2 NPE
echo "[2/3] Starting GROBID service..."
cd /opt/grobid
export GROBID_SERVICE_OPTS="-XX:-UseContainerSupport -Djava.library.path=grobid-home/lib/lin-64:grobid-home/lib/lin-64/jep --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.io=ALL-UNNAMED"
./grobid-service/bin/grobid-service &
GROBID_PID=$!
cd /home/user/app

# 3. Wait for Ollama readiness
echo "Waiting for Ollama to become ready..."
OLLAMA_READY=0
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is UP and ready! (took ${i}s)"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" -ne 1 ]; then
    echo "ERROR: Ollama failed to respond within 30s."
    exit 1
fi

# Print available pre-baked models
echo "Pre-baked models detected in Ollama:"
curl -s http://127.0.0.1:11434/api/tags | tr ',' '\n' | grep '"name":' || true

# 4. Wait for GROBID readiness
echo "Waiting for GROBID to become ready..."
GROBID_READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8070/api/isalive | grep -q "true" 2>/dev/null; then
        echo "GROBID is UP and ready! (took ${i}s)"
        GROBID_READY=1
        break
    fi
    sleep 1
done

if [ "$GROBID_READY" -ne 1 ]; then
    echo "ERROR: GROBID failed to respond within 60s."
    exit 1
fi

# 5. Start Streamlit Application
PORT="${PORT:-8080}"
echo "[3/3] Starting Streamlit Application on port ${PORT}..."
echo "=========================================================="
echo " Application is ready to receive requests at http://0.0.0.0:${PORT}"
echo "=========================================================="

# Run Streamlit in foreground so signals propagate properly
streamlit run app.py \
    --server.port="${PORT}" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false &
STREAMLIT_PID=$!

wait "$STREAMLIT_PID"
