# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multimodal RAG (Retrieval-Augmented Generation) document Q&A application built on LlamaIndex. It supports text and PDF (with embedded images) ingestion, hybrid retrieval, reranking, and a Gradio chat interface.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables (or use .env file)
# DASHSCOPE_API_KEY=<your-key>
# DASHSCOPE_BASE_URL=<base-url>

# Start the app (Gradio serves at http://127.0.0.1:7860)
python main.py
```

**Prerequisites before starting:**
- Redis must be running at `127.0.0.1:6379` (used for document store and index store)
- Local embedding model at the path in `config/settings.py` → `EMBEDDING_MODEL_PATH` (`BAAI/bge-large-zh-v1.5`)
- Local reranker model at `RERANK_MODEL_PATH` (`BAAI/bge-reranker-large`)

## Architecture

### Request Flow

```
main.py
  └── ui/interface.py          # Gradio UI, wires buttons to RAGApplication methods
        └── core/application.py  # RAGApplication: orchestrates ingestion + querying
              ├── core/ingestion.py   # DocumentIngestionPipeline: loads, chunks, embeds, stores
              │     └── core/pdfProcessor.py  # MultimodalPDFProcessor: image+text extraction
              └── core/workflow.py    # RAGWorkflow (LlamaIndex Workflow): retrieve→rerank→generate
                    └── core/events.py  # Typed workflow events (RetrievalEvent, RerankEvent, ResponseEvent)
```

### Storage Layer

| Component | Technology | Location |
|-----------|-----------|---------|
| Vector store | ChromaDB | `./chroma_db` (persistent) |
| Document store | Redis | `127.0.0.1:6379`, namespace `redis_docs` |
| Index store | Redis | `127.0.0.1:6379`, namespace `redis_index` |

### RAG Workflow Steps (`core/workflow.py`)

The `RAGWorkflow` is a LlamaIndex `Workflow` subclass with four `@step` methods executed in sequence:

1. **`retrieve_step`** — Hybrid retrieval: `QueryFusionRetriever` combines `VectorIndexRetriever` (Chroma) + `BM25Retriever` (Redis docstore), returning top `SIMILARITY_TOP_K=5` nodes.
2. **`rerank_step`** — `SentenceTransformerRerank` with local `bge-reranker-large`, keeps top `RERANK_TOP_K=3` nodes.
3. **`generate_step`** — `ResponseSynthesizer` generates an answer from the reranked nodes.
4. **`finalize_step`** — Packages result dict (response text + source nodes with scores and metadata).

### PDF Multimodal Processing (`core/pdfProcessor.py`)

`MultimodalPDFProcessor` uses `unstructured` with `strategy='hi_res'` to extract images, text, and tables. Images are spatially associated with nearby text chunks via Euclidean distance on bounding-box coordinates. Image paths are stored as JSON in node metadata (`image_paths` field) and rendered as base64 `<img>` tags in the chat response.

### Dual Query Mode (`core/application.py`)

`RAGApplication.query_documents()` branches on whether the user selected documents in the dropdown:
- **With documents selected** → runs `RAGWorkflow`, shows sources + inline images.
- **No documents selected** → plain `Settings.llm.complete(query)` call (no RAG).

## Configuration (`config/settings.py`)

All tuneable parameters live here. Key items:

| Setting | Default | Notes |
|---------|---------|-------|
| `OPENAI_API_KEY` | `DASHSCOPE_API_KEY` env var | Actual LLM is DashScope/Qwen, not OpenAI |
| `OPENAI_MODEL` | `qwen-plus-2025-07-14` | DashScope model name |
| `EMBEDDING_MODEL_PATH` | Windows absolute path | Change to your local model path |
| `RERANK_MODEL_PATH` | Windows absolute path | Change to your local model path |
| `CHUNK_SIZE` | `512` | Token chunk size for ingestion |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `SIMILARITY_TOP_K` | `5` | Nodes retrieved before reranking |
| `RERANK_TOP_K` | `3` | Nodes kept after reranking |

## Logging

`utils/logger.py` → `setup_logger(name)` writes to both stdout and `logs/<module-name>.log`. Each module passes `__name__` so log files are named after the module (e.g., `logs/core.workflow.log`).
