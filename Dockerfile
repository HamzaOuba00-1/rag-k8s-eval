# DocTruth web app: React front-end + FastAPI API in one image, for Cloud Run.
#
# Build context = repository root, after these local steps (see docs/DEPLOY.md):
#   python -m ragk8s index --all        -> data/index/<strategy>/
#   python -m ragk8s onnx export        -> models/onnx/embedder/model.onnx (fp32)
#   python -m ragk8s app-bundle         -> app_data/bundle.json
# Only the files the app reads are copied: no PyTorch, no raw corpus, no results.

# --- front-end build -------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- runtime ---------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements-app.txt ./
RUN pip install -r requirements-app.txt

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src/ src/
RUN pip install --no-deps . && rm -rf src build

COPY app/ app/
COPY data/index/fixed/chunks.jsonl data/index/fixed/bm25_tokens.json data/index/fixed/dense.faiss data/index/fixed/
COPY data/index/structure/chunks.jsonl data/index/structure/bm25_tokens.json data/index/structure/dense.faiss data/index/structure/
COPY data/index/semantic/chunks.jsonl data/index/semantic/bm25_tokens.json data/index/semantic/dense.faiss data/index/semantic/
COPY models/onnx/embedder/model.onnx models/onnx/embedder/tokenizer.json models/onnx/embedder/ragk8s_onnx.json models/onnx/embedder/
COPY app_data/bundle.json app_data/
COPY --from=web /web/dist web/dist

RUN useradd --uid 10001 --no-create-home doctruth
USER doctruth

ENV PORT=8080
EXPOSE 8080
# One process: the quota locks keep their per-visitor state in memory.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-server-header"]
