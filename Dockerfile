FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CITATION_DATA_DIR=/app/data

WORKDIR /app

# System dependencies for PDF/image handling (poppler, OpenCV/Docling runtime libs, healthcheck curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python runtime stack first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project source and install package
COPY . .
RUN pip install --no-deps -e . && mkdir -p /app/data /app/workspace_runs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
