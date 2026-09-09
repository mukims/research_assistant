# Container for the Streamlit app in app.py.
# Runs on Google Cloud Run (honours $PORT), or any Docker host (defaults to 7860).
FROM python:3.12-slim

# Build tools for the few deps without a pure wheel; curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (also matches Hugging Face Spaces' uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Editable install so `research_assistant.*` resolves regardless of the
# working directory the CMD below happens to run from — pyproject.toml is a
# thin package pointer, not a rebuild of anything already installed above.
RUN pip install --no-cache-dir --user -e .

# Baked-in backend config. Corpus + downloads land in CITATION_DATA_DIR, which
# sits outside the app's own directory tree so a read-only image still runs; on
# the GCE deployment a persistent disk is bind-mounted there. Chat and
# embeddings both go to OpenAI — no Ollama package is imported at runtime under
# these settings, since every ollama import in shared/llm.py is function-local
# and behind a backend branch. OPENAI_API_KEY is NOT baked in; it arrives from
# Secret Manager at container start.
ENV CITATION_DATA_DIR=/home/user/data \
    CITATION_LAYOUT_DETECTION=0 \
    CITATION_LOG_FILE=0 \
    LLM_BACKEND=openai \
    CITATION_EMBED_BACKEND=openai \
    OPENAI_BASE_URL=https://api.openai.com/v1 \
    CITATION_LLM_MODEL=gpt-4.1-mini \
    CITATION_EMBED_MODEL=text-embedding-3-small \
    PORT=8080
RUN mkdir -p /home/user/data

EXPOSE 8080
HEALTHCHECK CMD curl -f "http://localhost:${PORT}/_stcore/health" || exit 1

# Shell form so ${PORT} expands. Disable CORS and XSRF so remote browser access
# over IP or tunnel doesn't drop WebSocket connections.
CMD streamlit run app.py \
      --server.port=${PORT} --server.address=0.0.0.0 \
      --server.headless=true --browser.gatherUsageStats=false \
      --server.enableCORS=false --server.enableXsrfProtection=false
