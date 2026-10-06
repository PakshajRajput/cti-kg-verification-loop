FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data/ctinexus_raw ./data/ctinexus_raw

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -e .

# Prepare only the lightweight processed CTINexus cache needed by the API.
RUN /opt/venv/bin/python -c "from pathlib import Path; from src.ingestion.dataset_loader import load_ctinexus_dataset; load_ctinexus_dataset(Path('data'))"

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
COPY streamlit_app.py ./

# Copy only the runtime data required by the deployed pipeline.
COPY data/chroma ./data/chroma
COPY data/mitre/bm25_index.pkl ./data/mitre/bm25_index.pkl
COPY data/mitre/bm25_corpus.jsonl ./data/mitre/bm25_corpus.jsonl
COPY --from=builder /build/data/processed ./data/processed

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
