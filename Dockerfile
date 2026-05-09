# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Keeps Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System libraries required by faiss-cpu and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
# Install CPU-only PyTorch first (saves ~1.5 GB vs the default CUDA build)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else (pip skips torch since it's already satisfied)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app/ ./app/
COPY pipeline/ ./pipeline/

# ── Pre-download embedding model ──────────────────────────────────────────────
# Baking InLegalBERT into the image avoids a ~534 MB download on every cold
# start and prevents startup failures when disk space is tight at runtime.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('law-ai/InLegalBERT'); \
print('Embedding model downloaded.')"

# Create the index directory so the app can write into it at startup
# (populated at runtime from S3, or bind-mounted for local testing)
RUN mkdir -p ./index

# ── Runtime config ────────────────────────────────────────────────────────────
EXPOSE 8000

# Environment variables with safe defaults.
# Override these at `docker run` time — do NOT hardcode secrets here.
#   OPENAI_API_KEY    — required for /query
#   INDEX_DIR         — where to write the downloaded index (default: ./index)
#   EMBED_MODEL       — must match the model used to build the index
#   S3_INDEX_BUCKET   — if set, index is downloaded from S3 on startup
#   S3_INDEX_PREFIX   — S3 key prefix (default: index)
ENV INDEX_DIR=./index
ENV EMBED_MODEL=law-ai/InLegalBERT

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
