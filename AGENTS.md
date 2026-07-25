# AGENTS.md

Agent instructions for contributors and coding agents working in this repository.

## Purpose

Sahayak AI is a Python-first, full-stack learning platform:
- Backend: FastAPI app with multimodal ingestion + RAG services
- Frontend: Streamlit app calling backend REST endpoints
- Infra: Docker Compose with Qdrant + backend + frontend

Reference docs:
- Project overview and feature list: [README.md](README.md)
- Container wiring: [docker-compose.yml](docker-compose.yml)

## Fast Start Commands

From repo root:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

Run backend:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Run frontend:

```powershell
streamlit run frontend/app.py
```

Run tests:

```powershell
pytest backend/tests -q
```

Docker stack:

```powershell
docker compose up --build
```

## Architecture Boundaries

- API entrypoint: `backend/main.py`
- HTTP routers: `backend/routers/`
- Domain/service logic: `backend/services/`
- Ingestion adapters by modality: `backend/ingestion/`
- Auth system (JWT + DB models): `backend/auth_system/`
- Shared utilities/config: `backend/common/`, `backend/utils/`
- Streamlit UI entrypoint: `frontend/app.py`
- Persistent runtime data: `data/`

When adding new backend behavior:
- Add/extend a router in `backend/routers/` for HTTP surface.
- Keep modality/data processing in `backend/ingestion/` or `backend/services/`.
- Reuse shared path helpers in `backend/common/data_paths.py` rather than hardcoded paths.

## Environment and Security Expectations

Create `.env` in repo root (see README for baseline keys). Commonly used variables include:
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`
- `GROQ_API_KEY`
- `JWT_SECRET_KEY`
- `ALLOWED_ORIGINS` (comma-separated; backend defaults to `http://localhost:8501`)
- `SAHAYAK_API_KEY` (if set, ingestion/search calls require `X-API-Key`)
- `AUTH_DATABASE_URL` (optional; defaults to SQLite under `data/auth/`)
- `BACKEND_URL` (frontend -> backend target)

Do not hardcode secrets in code or tests.

## Repository-Specific Pitfalls

- Auth behavior is environment-dependent: if `SAHAYAK_API_KEY` is unset, API key auth is bypassed by design (`backend/auth.py`).
- CORS is intentionally restricted and credential-aware in `backend/main.py`; do not switch to wildcard origins with credentials.
- Health endpoint may report degraded if DB/vector checks fail; validate `/health` after infra changes.
- `package.json` exists but Node is not the primary runtime for app execution; prefer Python tooling unless task is explicitly Node-related.

## Change Checklist for Agents

Before submitting changes:
- Run targeted tests in `backend/tests/` for touched functionality.
- Smoke test backend startup: `uvicorn backend.main:app --reload --port 8000`.
- If frontend/API contracts changed, run Streamlit app and verify key flows manually.
- Prefer minimal, surgical edits; avoid unrelated refactors.

## High-Value Files to Read First

- `backend/main.py` (middleware, router wiring, health checks)
- `backend/routers/ingestion.py` (multimodal endpoint patterns)
- `backend/auth.py` and `backend/auth_system/database.py` (auth and DB setup)
- `frontend/app.py` (UI state, backend calls, auth headers)
- `requirements.txt` and `requirements-dev.txt` (runtime + test dependencies)
