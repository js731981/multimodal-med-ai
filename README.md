![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)
![AI](https://img.shields.io/badge/AI-Multimodal-orange)
![VectorDB](https://img.shields.io/badge/VectorDB-Redis-purple)

# Multimodal Medical AI

Educational demo system for **multimodal chest X-ray + clinical text** inference: a FastAPI backend, optional **RAG** explanations from a built-in medical knowledge base, a **Streamlit** UI with follow-up chat, and **Redis-backed session memory** (with in-memory fallback).

**Not for clinical use.** Outputs are for research and education only—not a diagnosis or treatment plan.

## 🚀 Features

- Chest X-ray analysis (ResNet-based)
- Clinical text understanding
- Multimodal fusion model
- Grad-CAM visual explainability
- RAG-based medical reasoning
- Conversational AI with Redis memory
- Semantic recall using vector embeddings
- Fault-tolerant design (Redis fallback)

## What is implemented

- **Inference pipeline**: Image encoder (ResNet-style CXR path with optional checkpoint), text encoder (TF-IDF), fusion classifier, optional Grad-CAM when the image model supports it.
- **RAG**: TF-IDF retrieval over a default in-process medical KB + template explanations (symptom–prediction alignment hints, reference-style text). Pluggable generator (including optional LLM hook in code).
- **API**: `GET /api/v1/health`, `POST /api/v1/infer` (JSON with optional `text` and base64 `image_b64`).
- **Frontend**: Streamlit app calling the same pipeline in-process; chat turns that reuse the last prediction and RAG text without re-running fusion.
- **Session storage**: Chat history and per-session embedding store via **Redis** when `REDIS_HOST` / `REDIS_PORT` reach a live server; otherwise transparent **in-memory** fallback.
- **Docker**: `docker-compose.yml` runs **Qdrant** and the **backend** (Qdrant is wired in config/compose for future use; default RAG does not require it).
- **Tests**: Pytest coverage for RAG, conversational helpers, multimodal and image pipelines, Grad-CAM, text model, and Redis memory integration (where Redis is available).

## Repository layout

```
.
├── backend/                 # FastAPI app
│   ├── app/
│   │   ├── api/v1/routes/   # health, infer
│   │   ├── inference/       # pipeline, models, explanation adapter
│   │   └── config.py        # settings (env / .env)
│   └── Dockerfile
├── frontend/                # Streamlit UI (app.py, backend_client.py)
├── rag/                     # KB, retriever, generator, conversational, vector_store, storage
├── models/                  # Checkpoints and model package code (image / text / fusion)
├── utils/                   # redis_client
├── tests/                   # pytest suites (+ manual redis ping script)
├── docker-compose.yml       # qdrant + backend
├── requirements.txt
└── .env.example             # starter environment variables
```

## Requirements

- Python **3.12** recommended (matches Docker image).
- See `requirements.txt` for packages (FastAPI, PyTorch, scikit-learn, Streamlit, redis, qdrant-client, OpenCV headless, etc.).

Optional: **sentence-transformers** for richer session embeddings in `rag/vector_store.py`; without it, a deterministic stub embedder is used.

## Quickstart (local)

### 1. Environment

Copy `.env.example` to `.env` and adjust. Important variables:

| Variable | Purpose |
|----------|---------|
| `APP_ENV` | `local`, `dev`, `prod`, or `test` (affects CORS in prod). |
| `HOST`, `PORT` | Bind address for uvicorn/gunicorn. |
| `LOG_LEVEL`, `LOG_JSON` | Logging. |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Reserved for future Qdrant-backed RAG; default path uses in-process KB. |
| `REDIS_HOST`, `REDIS_PORT` | Optional; if Redis responds to `PING`, chat/vector session data uses Redis. |
| `MMEDAI_IMAGE_CHECKPOINT_PATH` / `IMAGE_CHECKPOINT_PATH` | Optional ResNet CXR weights (`.pth`). |
| `MMEDAI_RAG_ENABLED` / `RAG_ENABLED` | Enable/disable RAG in the pipeline (default on). |
| `MMEDAI_RAG_TOP_K` / `RAG_TOP_K` | Top-k passages for retrieval. |
| `MMEDAI_LABELS` | Class names; order must match model outputs. |

Model class paths default to `backend.app.inference.*_impl` modules; see `backend/app/config.py` for the full list and aliases.

### 2. Install and run the API

From the repository root (so `backend` and `rag` import correctly):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- OpenAPI docs: http://localhost:8000/docs  

### 3. Run the Streamlit UI

```bash
streamlit run frontend/app.py
```

Ensure the API env vars and model paths are set if you expect parity with production-like runs.

### 4. Docker Compose

From the repository root:

```bash
docker compose up --build
```

Backend on port **8000**, Qdrant on **6333**. The backend image copies `backend/` only; for inference with local checkpoints, mount or extend the image to include `models/` and any weight files.

### 5. Tests

```bash
pytest tests -q
```

Some tests need optional fixtures (e.g. Redis for `test_redis_memory.py`). `tests/test_redis.py` is a small manual connectivity script, not part of the pytest suite.

## API

### `GET /api/v1/health`

Returns `{"status": "ok"}`.

### `POST /api/v1/infer`

Body (JSON):

- `text` (optional): clinical note or symptoms.
- `image_b64` (optional): base64-encoded image.
- `top_k` (optional): present on schema; RAG top-k is driven by server settings unless extended.

Response:

- `disease`, `confidence`, `explanation` (RAG string, may be empty if disabled).

At least one of `text` or `image_b64` must be provided.

## Documentation

- **[PROJECTOVERVIEW.md](PROJECTOVERVIEW.md)** — architecture, data flow, module map, and extension points.

## Author

**Jayendran Subramanian**  
AI Engineer | Data Engineer | System Builder  

- Passionate about building AI-driven decision systems  
- Focused on multimodal AI, RAG architectures, and scalable AI platforms

## Disclaimer

This project is intended for educational and research purposes only.

- It does NOT provide medical advice, diagnosis, or treatment.
- Predictions and explanations are generated by AI models and may be inaccurate.
- Always consult a qualified healthcare professional for medical concerns.

The system simulates clinical reasoning for demonstration purposes only.
