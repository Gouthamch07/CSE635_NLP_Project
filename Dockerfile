FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install runtime deps first so layer caches across code-only edits.
COPY requirements.runtime.txt ./
RUN pip install --upgrade pip && pip install -r requirements.runtime.txt

# App code + the BM25 pickle (the rest is excluded by .dockerignore).
COPY config/ ./config/
COPY ub_cse_bot/ ./ub_cse_bot/
COPY data/processed/bm25.pkl ./data/processed/bm25.pkl

# Cloud Run sets $PORT (default 8080). Bind 0.0.0.0 so the container is reachable.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn ub_cse_bot.ui.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
