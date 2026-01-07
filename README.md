# Sahayak_AI

**Sahayak_AI** unifies the four historical Sahayak projects into a single multimodal AI assistant platform. It ships with:

- **Multimodal ingestion** (audio, video, image, PDF, text, URL, YouTube) from `Sahayak_09`.
- **Qdrant-powered vector search** with an automatic **SQLite/FAISS fallback** from `sahayak_09_02`.
- **Rich research workflow assets** (data pipelines, notebooks, scripts, dataset layouts) from `Sahayak`.
- **Fine-tuning dataset utilities and lightweight HuggingFace UI** from `ahayak`.

Run everything locally or mix-and-match capabilities per deployment target.

## Repository layout 

```
Sahayak_AI/
├── backend/                 # FastAPI platform API
│   ├── ingestion/           # Multimodal processors
│   ├── routers/             # /ingest, /search, /summaries, /local, /finetune, /admin
│   ├── local_stack/         # SQLite + FAISS fallback pipeline (sahayak_09_02)
│   └── finetune_stack/      # Stand-alone fine-tuning helper service (ahayak)
├── frontend/
│   └── app.py               # Unified Streamlit UI (multiple modes)
├── data/
│   ├── processed/, raw/     # Research datasets from Sahayak
│   ├── ahayak/              # Fine-tune dataset & pdf storage
│   └── sahayak_09_02/       # Local-mode pdf storage + sqlite db
├── embeddings/, models/     # Vector indexes & model checkpoints
├── scripts/                 # Preprocessing + embedding pipelines
└── notebooks/demo.ipynb     # Research playground
```

## Backend usage

```bash
cd Sahayak_AI
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Key routes:

| Route prefix | Description |
|--------------|-------------|
| `/ingest`    | Upload audio/video/image/pdf/text/url assets (auto-chunks + vectorizes). |
| `/search`    | `/vector` for pure retrieval, `/rag` for retrieval-augmented answers. |
| `/summaries` | Text summarization (uses transformers pipeline with graceful fallback). |
| `/local`     | Stand-alone SQLite/FAISS uploader + `/local/ask` endpoint (legacy mode). |
| `/finetune`  | Append/list LoRA/QLoRA training pairs backed by `data/ahayak/fine_tune_dataset.jsonl`. |
| `/admin`     | Qdrant health, collection metadata, and recent payloads. |

The platform automatically prefers a running Qdrant instance; when it is unavailable the system stores vectors locally in SQLite/FAISS.

### Switching vector backends

Every ingestion/search endpoint accepts `target` with values:

- `auto` *(default)* – Qdrant when available, otherwise SQLite/FAISS.
- `qdrant` – force Qdrant (fails fast if unavailable).
- `local` – skip Qdrant and store/query locally.

> **Vector DB cheat-sheet** (pick what matches your deployment):
>
> | DB | Best for |
> | --- | --- |
> | **FAISS** | Local experiments, PyTorch workflows |
> | **Chroma** | Quick RAG prototyping |
> | **Milvus** | Production-grade, self-hosted scale |
> | **Qdrant** | Fast filtering + hybrid queries (default here) |
> | **Pinecone** | Managed, zero-ops deployments |
> | **Weaviate** | Schema-rich, hybrid search |

### Fine-tune dataset workflow

1. Collect question/answer pairs via `/finetune/examples`.
2. Inspect them with `/finetune/examples?limit=50` and `/finetune/stats`.
3. Feed the resulting `data/ahayak/fine_tune_dataset.jsonl` into your preferred PEFT/LoRA trainer (see scripts in `scripts/`).

## Frontend UI

`streamlit run frontend/app.py`

Pick the experience you need from the sidebar mode selector:

- **Pro Dashboard** – tabbed navigation (Upload, Ask, Search, Recommend, Advanced, Roadmap) built from the legacy Sahayak_09 components.
- **Simplified Upload** – the minimalist sahayak_09_02 workflow with health badges, metrics, and one-click Q&A.
- **HuggingFace Mini** – a compact uploader/asker view ideal for Spaces or kiosk deployments.

All modes point at the same backend and share the same component library, so you only have to maintain a single Streamlit file.

## Data & research utilities

- `scripts/preprocessing.py`, `scripts/embedding_pipeline.py`, `scripts/rag_query.py` retain the original research workflows.
- `data/raw` & `data/processed` mirror the Sahayak dataset layout for reproducible experiments.
- `models/` and `embeddings/` host pretrained + fine-tuned checkpoints and FAISS indexes.

## Docker (optional)

Use `docker compose up --build` after crafting your own compose file or reuse the ones from the legacy folders; the backend only needs `uvicorn backend.main:app` and the Streamlit frontend runs via `streamlit run frontend/app.py`.

## Next steps

- Point `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` to your preferred Qdrant cluster (leave defaults for local `docker run qdrant/qdrant`).
- Point ingestion endpoints at your multi-modal data lake.
- Export the fine-tune dataset into any LoRA/QLoRA pipeline for bespoke teaching-assistant models.
- Customize the Streamlit UI (either advanced or simplified) per deployment requirement.
```
