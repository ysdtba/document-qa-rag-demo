# Document Q&A with Hybrid Retrieval

A small RAG service that answers questions about a document set. It combines Chroma vector search and BM25 using reciprocal rank fusion, reranks the matches, and streams an answer and source snippets through a FastAPI endpoint. The included browser client uses the same API.

**Data notice:** `docs/synthetic_otc_rules.md` is a synthetic demonstration document. Its rules, dates and monetary amounts are invented. This repository is not a source of Hong Kong regulatory guidance.

## What you can inspect

- `data_init.py`: document loading, Markdown-aware chunks, Chroma and BM25 indexing.
- `rag_pipeline.py`: retrieval fusion, reranking, source metadata and streaming generation.
- `main.py`: `/health`, `/api/chat` and the browser demo at `/`.
- `demo/index.html`: a small same-origin client showing streamed answers and citations.

The default demo does not need a paid LLM key. It returns a clearly labeled extractive mock answer. Set `OPENAI_API_KEY` and the corresponding model settings to use a compatible LLM. `RERANK_MODE=mock` uses lightweight heuristic ordering; `local` loads the Cross-Encoder model.

## Run locally

Use Python 3.11 and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python data_init.py
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. API documentation is at <http://127.0.0.1:8000/docs>. The first run downloads the embedding model and builds local indexes, so it needs internet access and can take several minutes. Subsequent starts reuse `storage/`.

To test the stream from a terminal:

```bash
python test_client.py "示例文档中的理赔申请期限是多少？"
```

To index your own documents, replace the synthetic file in `docs/` with `.md`, `.txt` or text-extractable `.pdf` files, remove the old `storage/` directory, then rerun `python data_init.py`. Check that you have rights to publish any demo documents.

## Container demo

```bash
docker build -t document-rag-demo .
docker run --rm -p 8000:8000 document-rag-demo
```

The image build downloads the embedding model and builds the sample index. Allow several GB of build space and memory for the Python/ML dependencies. The demo image uses mock generation and mock reranking; it makes no paid model calls. For a real document service, use a persistent volume for `storage/`, authenticate the API, rate-limit requests, and configure `CORS_ORIGINS` only if the browser client is hosted on another origin. Those production controls are not implemented here.

## API contract

`POST /api/chat` accepts `{"question":"..."}` and responds as server-sent events: a `sources` JSON event, `token` JSON events and `data: [DONE]`. `GET /health` reports whether an index is loaded. Requests return 503 until indexing is complete.

## Current limits

The provided corpus has one synthetic document, so relevance claims are limited. There are no retrieval-quality evaluations, authentication, upload workflow, or concurrency benchmarks. The service is a portfolio demonstration, not a production compliance system.
