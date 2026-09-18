# Project Context & System Status: Sahayak AI

*Last Updated: 2026-09-14 18:39 (IST)*

> **Purpose of this file:** single copy-paste onboarding document. Gives any LLM/AI coding agent
> immediate, verified facts about architecture, deployed state, active migration, and hard
> constraints — no chat history required. All facts below were re-audited against the live repo
> on the date above; do not trust stale docs (README.md, AGENTS.md, PROJECT_KNOWLEDGE.md differ
> from code in places flagged in §7).

---

## 1. System Mission & Core Architectural Invariants

**Mission:** Sahayak AI is a multimodal (text / image / audio / video) knowledge assistant for
Indian education — grounded Q&A over ingested PDFs/audio/video, personalized recommendations,
mental-wellness counseling with crisis triage, UPI/Stripe monetization, multilingual (en/hi/es/
fr/de) support, WhatsApp/Telegram delivery.

**The 512 MB Render Free Tier ceiling is the single most important invariant.** Every design
decision below exists to survive it.

| # | Invariant | Enforcement |
|---|-----------|-------------|
| 1 | **Zero local heavy ML compute.** No `torch`, no `sentence-transformers`, no local Whisper/OCR/reranker/embedder. | Both `requirements.txt` files verified clean (only `httpx`, `edge-tts`, `langgraph`, `langchain-groq`, `qdrant-client`, `neo4j`, `upstash-*`). A dev-only `gpu_test.py` imports `torch` at repo root — never deploy it. |
| 2 | All heavy inference offloaded to **managed APIs**: Groq (Llama 3.3/3.1/Whisper), Hugging Face Serverless (`BAAI/bge-m3` embeddings, `BAAI/bge-reranker-base` rerank), Neo4j AuraDB, Qdrant Cloud, Upstash Vector/Redis, qrserver.com (QR render), Bazooka-style UPI gateway. | Enforced per module; verified in `services/`, `agents/`, `rag_agent.py`. |
| 3 | **Pipeline isolation.** Legacy routes (`backend/`, `/api/v1/*`) stay intact. v3 lives under `sahayak_ai_v3/`, gated by `settings.SAHAYAK_PIPELINE_VERSION` (`legacy` \| `v2_advanced`). | `backend/api/main.py:58` conditionally mounts v2 agents under try/except ImportError (Rule 2). |
| 4 | **Graceful degradation everywhere.** Reranker fails → fall back to RRF scores; Graph search fails → fall back to Dense; Groq fails → keyword fallback router. | Verified in `retrieval/fusion.py`, `agents/supervisor.py` (`_fallback_route`, `_retrieve_safely`, `_recommender_context`). |
| 5 | **Canonical chunk schema.** Chunks map to `chunk_id, document_id, modality, bbox, entities, content, metadata`. | `backend/core/models.py` (`RetrievedDocument`). |
| 6 | **External API calls must have backoff/retries + fallback.** | Groq client `max_retries=2` (`services/agents/supervisor.py:78`). |
| 7 | **Reranker MUST be `BAAI/bge-reranker-base` via remote HTTP only, never local weights.** | `agents/rag_agent.py:21`, `retrieval/fusion.py`. |
| 8 | Single uvicorn worker. Run command: `uvicorn ... :app --workers 1`. | Both `render.yaml` files set `--workers 1`. |

**Production UI vs Legacy UI demarcation:**
- **`sahayak-ui/` (Modern UI, React 19+Vite 8)** → the production frontend, deployed via root `render.yaml` / `sahayak-ui/vercel.json`.
- **`frontend/` (Classic UI, Streamlit)** → legacy full-featured learning platform, Dockerized (`frontend/Dockerfile`, port 8501). Documented in README as "Classic UI"; kept alive for backward compatibility.
- **`sahayak_ai_v3/frontend/` (experimental v3 chat SPA, React 18+Vite 5)** → mid-refactor, **not wired to any backend, currently broken** (imports `../api/v2_client` which does not exist).

---

## 2. Directory & Repository Layout

```
SAHAYAK_AI/
├── backend/                     # LEGACY backend (monolith; /api/v1-ish, unversioned paths)
│   ├── main.py                  #   FastAPI app (title "Sahayak AI Platform", no static mount)
│   ├── tests/                   #   9 legacy test files
│   └── vector_store/qdrant_store.py   # legacy Qdrant singleton — v3 agents import this
│
├── frontend/                    # CLASSIC UI — legacy Streamlit app (19 pages, Dockerized :8501)
│   ├── app.py                   #   917-line page_map (Dashboard, Upload, Search, Chat, Quiz, …)
│   ├── pages/login.py           #   auth gate
│   └── src/                     #   ORPHANED React 18 component set (no entry, no vite config — dead)
│       ├── ChatInterface.jsx, CitationViewer.jsx, KnowledgeGraph.jsx, VoiceRecorder.jsx
│       └── api/client.js
│
├── sahayak-ui/                  # MODERN/PRODUCTION UI — React 19.2 + Vite 8 + Tailwind 3 SPA
│   ├── render.yaml              #   (deployed from root render.yaml, static dist)
│   ├── vercel.json              #   SPA rewrites
│   └── src/
│       ├── App.jsx              #   auth gate + currentPage switch (16 routed screens)
│       ├── pages/               #   17 files: AuthPage + Dashboard, Upload, SearchChat, Counselor,
│       │                        #   Quiz, Roadmaps, Books, Progress, KnowledgeGraph, Stories,
│       │                        #   Pricing, LearnHub, SettingsPage, Help, Sync, Contact
│       ├── components/          #   ChatInterface, ChatMessage, CitationViewer, KnowledgeGraphViewer,
│       │   └── ui/              #   VoiceRecorder + 14 primitives (Button, Card, Modal, RAGPipeline…)
│       ├── context/AppContext.jsx   # state-router (currentPage), auth token, ragSessionId
│       ├── hooks/useVoice.js        # browser Web Speech API (NOT Groq Whisper)
│       ├── api/client.js            # callBackend(): fetch → BACKEND_URL, Bearer + X-API-Key
│       └── layouts/MainLayout.jsx   # sidebar (hardcoded "v2.0" badge)
│
├── sahayak_ai_v3/               # V3 ENGINE — LangGraph multi-agent + hybrid CRAG + GraphRAG
│   ├── requirements.txt         #   supplementary deps for v2_advanced env
│   ├── render.yaml              #   ⚠ start cmd `src.main:app` — MISMATCHES real file path
│   ├── backend/
│   │   ├── api/
│   │   │   ├── main.py          #   FastAPI "Sahayak AI V3 - Orchestrator" — route registration
│   │   │   ├── routers/         #   chat.py, ingest.py, v2_chat.py, payments.py, audio.py, health.py
│   │   │   ├── chat.py          #   ⚠ orphan duplicate of /api/v2/chat/orchestrate (NOT mounted)
│   │   │   ├── voice.py         #   ⚠ /api/v3/voice/* — exists, NOT mounted in main.py
│   │   │   ├── payments.py      #   ⚠ /api/v3/payments/* — exists, NOT mounted
│   │   │   └── health.py        #   /api/v2/health/ping + deep-ping (mounted)
│   │   ├── core/                #   config.py (Settings), graph_state.py (CRAGState), models.py,
│   │   │                        #   logging.py, middleware.py, exceptions.py, alerts.py
│   │   ├── agents/              #   supervisor.py, graph_state.py, rag_agent.py,
│   │   │                        #   recommender_agent.py, counseling_agent.py, state.py
│   │   ├── services/
│   │   │   ├── agents/supervisor.py      # ⚙ LIVE supervisor used by /api/v2/chat (Groq JSON-mode)
│   │   │   ├── retrieval/                #   qdrant, sparse(BM25), graph(Neo4j), aggregator,
│   │   │   │                            #   fusion(RRF+rerank), router, decomposition, v3_orchestrator
│   │   │   ├── crag_orchestrator/        #   nodes.py, grader.py, workflow.py
│   │   │   ├── generation/               #   context_engine.py, generation_service.py
│   │   │   ├── payments/                 #   upi.py, verifier.py, webhook.py, worker.py, user_store.py
│   │   │   └── external/hf_client.py
│   │   ├── cache/semantic_cache.py       #   Upstash Vector similarity cache
│   │   ├── audio/tts_service.py          #   edge-tts (hi-IN-Swara / en-IN-Neerja)
│   │   ├── security/                     #   rate_limiter.py, pii_scrubber.py, dependencies.py
│   │   ├── webhooks/                     #   telegram.py, whatsapp.py (Task 4: WhatsApp Cloud API)
│   │   └── tests/                        #   test_endpoints.py, test_retrieval.py, test_v2_logic.py
│   ├── tests/                  #   test_semantic_cache.py, test_rate_limiter.py, test_pii_scrubber.py,
│   │   │                      #   test_all_endpoints.py, tests/agents/, tests/load/ (locust+k6)
│   └── frontend/               #   EXPERIMENTAL v3 chat SPA — broken (see §7 Known Breaks) + render.yaml
│
├── docs/openapi.json           #   5292-line OpenAPI 3.1.0 for LEGACY backend only
├── openapi.json / openapi_utf8.json / openapi_test.json   # generated exports
├── render.yaml                 #   PRODUCTION deploy: static sahayak-ui + web backend
├── docker-compose.yml          #   runs legacy Streamlit frontend + backend
├── data/sahayak_payments.db    #   sqlite payment/session store
├── requirements.txt            #   root/legacy env (clean, no torch)
└── AGENTS.md, PROJECT_KNOWLEDGE.md, PROGRESS.md, SLA_BENCHMARKING.md
```

---

## 3. Active Progress & Migration Checklist

### ✅ Completed backend modules
- **Hybrid retrievers:** QdrantDenseRetriever, BM25SparseRetriever, Neo4jGraphRetriever (`services/retrieval/`).
- **RRF fusion + `BAAI/bge-reranker-base` serverless rerank** (`retrieval/fusion.py`).
- **CRAG LangGraph state machine** (`services/crag_orchestrator/`): retrieve → rerank → grade → (rewrite|web_search) → generate → verify-faithfulness; loop-guarded by `MAX_RETRIEVAL_ATTEMPTS` / `rewrite_count` / `web_search_invoked`.
- **Multi-Agent supervisor graph** (LangGraph): Supervisor → {rag_agent, counseling_agent, recommender_agent}, circuit breaker at `iteration_count >= 3`, distress triage `>= 0.80` → HITL emergency lane (`services/agents/supervisor.py`).
- **Semantic cache** over Upstash Vector (`cache/semantic_cache.py`) — HIT bypasses agent graph, sets `X-Cache-Status`.
- **Voice API:** Groq Whisper STT (`/api/v2/chat/voice-stream`), edge-tts streaming TTS (`/api/v2/audio/synthesize`).
- **Payments:** Stripe checkout (`/api/v2/payments/create-checkout-session` + webhook), UPI deep-link/QR/UTR + HMAC webhook verify (`services/payments/upi.py`), sqlite `user_store.py`.
- **Webhooks:** Telegram bot + WhatsApp Cloud API (`/api/v2/webhooks/*`).
- **Security:** token-bucket rate limiter + PII scrubber (`security/`).
- **Observability:** structured JSON logging with PII sanitization, CorrelationId middleware, custom exception handlers.
- **Render keep-alive:** `/api/v2/health/ping` (<1 ms zero-compute).

### 🚧 In progress / partial
- **Frontend consolidation (v3 → `sahayak-ui`):** streaming tokens, PDF bbox citations, Neo4j force-graph modal, Groq-Whisper voice chat are the stated goal. **Current reality: NONE of the four are live in `saheyak-ui` — see §7.**
- **Ingestion workers:** PDF (PyMuPDF) and audio (Whisper) ingest endpoints are stubs — they accept the file and return `{"status":"processing"}` but the worker dispatch line is **commented out** (`routers/ingest.py:16`).
- **UPI payment reconciliation:** `services/payments/` UPI primitives exist and are unit-testable, but the `/api/v3/payments/*` and `services/payments/webhook.py` routers that expose them are **not mounted** in `main.py` (see §5, §7). UTR verification flow is defined but not reachable via HTTP.

### ⏳ Pending / blockers
1. **Boot blockers (v2_advanced mode):** `api/routers/audio.py:7` — `async def synthesize_audio(req: SynthesizeRequest):` has **an empty body → SyntaxError** → import of the `audio` router crashes at startup in `v2_advanced` mode (SyntaxError is not caught by the `except ImportError` in `main.py:69`).
2. **Mixed/path-fragile imports:** many v3 modules import the *legacy* package via bare `from backend...` (e.g. `services/agents/supervisor.py:23`, `rag_agent.py` → `backend.vector_store.qdrant_store`) while others use `from sahayak_ai_v3.backend...`. Boot only works from the repo root with the legacy `backend/` present.
3. **`render.yaml` mismatch:** `sahayak_ai_v3/render.yaml` starts `src.main:app` but the app is at `backend/api/main.py`. Root `render.yaml` correctly uses `sahayak_ai_v3.backend.api.main:app`.
4. **Data-model break in v3 rerank/grade path:** pydantic `RetrievedDocument` (`core/models.py:43`) declares `reranker_score`, but has no `model_config` permitting extra fields; `fusion.py` `rerank_with_bge` assigns `chunk.rerank_score = value` → **ValidationError** at the rerank node; `sparse.py:89-96` also reads `content/source/modality` from the wrong dict level (`sparse.py:39-42` builds them nested) → every BM25 doc has empty `content`. → the CRAG/rerank retrieval loop is currently broken at runtime.
5. **Legacy tests failing:** Progress report 26 Aug 2026: 14 tests → 7 passed / 7 failed (BUG-1..7, incl. `admin.py` property-vs-method call). v3 test suites + `tests/load/` (locust, k6) exist but were not run in this audit.
6. **`SAHAYAK_PIPELINE_VERSION` drift:** `.env` → `legacy`; both `render.yaml` → `v2_advanced`. Live behavior differs from deployed intent.
7. **CORS spec violation (v3 only):** `main.py:42-45` uses `allow_origins=["*"]` **with** `allow_credentials=True` → browsers reject; legacy backend correctly whitelists.

---

## 4. Multi-Agent & RAG Pipeline Specifications

### 4.1 Two state schemas exist (keep both, do not "unify" blindly)

| Schema | File | Fields (key) | Owner |
|--------|------|--------------|-------|
| `CRAGState` | `backend/core/graph_state.py` | `original_query, active_query, intent, detected_modality, extracted_entities, candidate_pool, refined_pool, relevance_score, confidence_category, rewrite_count, web_search_invoked, compressed_context, draft_response, is_faithful, final_response, filters, sources, learning_mode, user_mode` | CRAG loop (`services/crag_orchestrator/`) |
| `GraphState` / `AgentState` | `backend/agents/graph_state.py`; `services/agents/supervisor.py` + `agents/state.py` | `messages` (Annotated + reducer), `next_agent`, `user_id`, `user_tier`, `distress_level`, `emergency_flag`, `final_output`, `citations`, `iteration_count`, `visual_citations` | Supervisor multi-agent graph |

**Note:** `PROJECT_KNOWLEDGE.md` documents a *third* variant of the supervisor state (`is_emergency`, `iteration_count`) that matches `agents/graph_state.py`. Don't reconcile across all three without checking who reads what.

### 4.2 Agent matrix (LIVE implementation under `/api/v2/chat`)

| Agent | Module | Inference | Behavior / Circuit breakers |
|-------|--------|-----------|------------------------------|
| **Supervisor** (router) | `services/agents/supervisor.py` | Groq JSON-mode (`settings.GROQ_MODEL`, fallback `llama-3.3-70b-versatile`) | Routes to ONE worker; JSON parse failure → `_fallback_route` keyword heuristic; Groq failure → `rag_agent`. |
| **rag_agent** | `services/agents/supervisor.py` (node) + `agents/rag_agent.py` | Groq + `BAAI/bge-m3` embeddings + `bge-reranker-base` (HF serverless) | Hybrid dense/BM25/Graph via `_retrieve_safely` (parallel, never raises), evidence-grounded answer with `[n]` markers; emits `visual_citations` (page + bbox) for image/pdf. |
| **recommender_agent** | `services/agents/supervisor.py` (node) | Groq + Neo4j | Cypher `(User)-[:LIKES]->(Genre)-[:CONTAINS]->(Item)`; Neo4j down → Qdrant `metadata.genre` filter fallback; genre vocab is a hardcoded tuple (`# ponytail` noted). |
| **counseling_agent** | `services/agents/supervisor.py` (node) | Groq, `temperature=0` triage | `distress_score >= 0.80` → returns hardcoded Vandrevala/Tele-MANAS/Emergency helpline payload, NO further LLM generation (HITL lane). |

### 4.3 CRAG loop (compiled `services/crag_orchestrator/workflow.py`)
`retrieve → rerank → grade → route`:
- `GOOD` (≥0.85) → `generate`; `PARTIAL` (0.60–0.84) → `rewrite` (re-retrieve; >1 rewrite → `web_search`); `BAD` (<0.60) → `web_search`; `web_search_invoked` → force `generate`.
- `generate → verify_faithfulness → END`.
- Loop ceilings tuned only via `config.settings.MAX_RETRIEVAL_ATTEMPTS` (default 2) — never a new counter.
- ⚠ This loop's rerank/grade leg is currently **broken at runtime** (see §7-4).

### 4.4 Monolith agent graph (`agents/` package — used by `sahayak_agent_app`, NOT the mounted `/api/v2/chat`)
`agents/supervisor.py` builds a StateGraph over `agents/graph_state.GraphState` (`messages` with `add_messages` reducer), nodes = supervisor/counseling/recommender/rag, pre-screens suicide keywords, `iteration_count >= 3` → END. This is a **second, near-duplicate supervisor** — keep it working or delete it; do not assume it serves the API.

---

## 5. API Endpoint Catalog

### 5.1 Mounted in `backend/api/main.py` (always)
| Method | Path | Source | Notes |
|--------|------|--------|-------|
| POST | `/chat/ask` | `api/routers/chat.py:13` | |
| POST | `/chat/ask_multimodal` | `api/routers/chat.py:35` | |
| POST | `/ingest/pdf` | `api/routers/ingest.py:7` | stub — worker dispatch commented out |
| POST | `/ingest/audio` | `api/routers/ingest.py:20` | stub |
| GET | `/health` | `api/main.py:75` | `{"status":"healthy","version":"v3.0.0"}` |
| GET | `/api/v2/health/ping` | `api/health.py:13` + `routers/health.py:26` | zero-compute keep-alive |
| GET | `/api/v2/health/deep-ping` | `api/health.py:42` | |

### 5.2 Mounted **only when `SAHAYAK_PIPELINE_VERSION == "v2_advanced"`** (`main.py:58-73`)
| Method | Path | Source | Notes |
|--------|------|--------|-------|
| POST | `/api/v2/chat/orchestrate` | `api/routers/v2_chat.py:39` | Semantic-cache bypass → supervisor graph; sets `X-Cache-Status`; rate-limited |
| POST | `/api/v2/chat/voice-stream` | `api/routers/v2_chat.py:101` | MediaRecorder blob → Groq `whisper-large-v3` → orchestrate; 502/422 on failure |
| POST | `/api/v2/audio/synthesize` | `api/routers/audio.py:14` | **file has a SyntaxError at line 7 — boot blocker** |
| POST | `/api/v2/payments/create-checkout-session` | `api/routers/payments.py:32` | Stripe |
| POST | `/api/v2/payments/webhook` | `api/routers/payments.py:68` | Stripe |
| POST | `/api/v2/webhooks/telegram` | `webhooks/telegram.py:69` | |
| GET  | `/api/v2/webhooks/whatsapp` | `webhooks/whatsapp.py:247` | verify handshake echo |
| POST | `/api/v2/webhooks/whatsapp` | `webhooks/whatsapp.py:265` | inbound message → agent |

### 5.3 Exists in code but **NOT mounted** (dead surface — §7)
| Method | Path | Source |
|--------|------|--------|
| POST | `/api/v3/voice/transcribe` | `api/voice.py:48` |
| POST | `/api/v3/voice/tts` | `api/voice.py:95` |
| POST | `/api/v3/voice/process-voice-chat` | `api/voice.py:125` |
| POST | `/api/v3/payments/create-order` | `api/payments.py:32` |
| POST | `/api/v3/payments/verify-utr` | `api/payments.py:71` |
| POST | `/api/v2/payments/verify-utr` | `services/payments/webhook.py:24` |
| POST | `/api/v2/payments/gateway-webhook` | `services/payments/webhook.py:48` |
| POST | `/api/v2/chat/orchestrate` (duplicate) | `api/chat.py:18` — collides with the mounted routing if ever registered |

### 5.4 Legacy backend (`backend/`) serves the **production UI** (`sahayak-ui`)
`sahayak-ui` calls **29 unversioned paths** — `/search/rag`, `/search/vector`, `/quiz/generate`, `/counselor/chat`, `/ingest/batch`, `/document/notes`, ... (see `docs/openapi.json`). **The v3 backend does not implement these.** Production UI ↔ production backend are the legacy FastAPI app, not the v3 orchestrator. This is the single biggest integration seam to know before wiring v3 features into `SearchChat.jsx`.

---

## 6. Environment & Secret Configuration

### 6.1 Required variables (`sahayak_ai_v3/backend/core/config.py` + `.env`)
| Variable | Default | Purpose |
|----------|---------|---------|
| `SAHAYAK_PIPELINE_VERSION` | `legacy` | `legacy` \| `v2_advanced` gate |
| `GROQ_API_KEY` | `""` | Llama + Whisper (`_groq()` uses it) |
| `HF_TOKEN` | `""` | `bge-m3` embeddings + `bge-reranker-base` |
| `QDRANT_URL` / `QDRANT_API_KEY` | `""` | dense vector store |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `""` | graph traversal |
| `UPSTASH_*` (Vector + Redis REST URL/TOKEN) | `""` | semantic cache + token bucket rate limiter |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | chat model (`.env` currently `openai/gpt-oss-20b`) |
| `MAX_RETRIEVAL_ATTEMPTS` | `2` | CRAG loop ceiling |
| `DENSE_TOP_K`/`SPARSE_TOP_K`/`GRAPH_TOP_K`/`RERANK_TOP_K`/`FINAL_CONTEXT_K` | `50/50/20/10/5` | pipleline top-k |
| `CRAG_ENABLED`, `GRAPH_RAG_ENABLED`, `RERANKER_ENABLED`, `HYBRID_SEARCH_ENABLED` | `true` | toggles |
| Stripe: `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID`, `STRIPE_SUCCESS_URL`, `STRIPE_CANCEL_URL` | `""` / localhost | monetization |
| WhatsApp: `WHATSAPP_TOKEN`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_GRAPH_BASE` | `""` / `graph.facebook.com/v19.0` | Cloud API webhook |
| `SAHAYAK_DB_PATH` | `./data/sahayak_payments.db` | sqlite payment/session store |
| Legacy app: `JWT_SECRET_KEY`, `BACKEND_URL`, `VITE_BACKEND_URL`, `AUTH_DATABASE_URL` (Neon), `EMBEDDING_MODEL`... | — | legacy auth/search |

`.env` (var names only): `SAHAYAK_PIPELINE_VERSION=legacy`, `CRAG_ENABLED=true`, `GRAPH_RAG_ENABLED=true`, `RERANKER_ENABLED=true`, `HYBRID_SEARCH_ENABLED=true`, `MAX_RETRIEVAL_ATTEMPTS=2`, `GROQ_API_KEY` set, `HF_API_TOKEN`/`HUGGINGFACEHUB_API_TOKEN` set, `QDRANT_*` set, `UPSTASH_REDIS_*` = placeholders, `UPSTASH_VECTOR_REST_TOKEN` **empty**, `SAHAYAK_API_KEY` **empty**, `TELEGRAM_BOT_TOKEN`/**`WHATSAPP_TOKEN`/`UPI_GATEWAY_API_KEY`/`UPI_WEBHOOK_SECRET`/`UPI_VPA`** empty. Do not paste secret values into any LLM context.

---

## 7. Known Breaks & Risks (verified, file:line)

1. **`api/routers/audio.py:7`** — `async def synthesize_audio(req: SynthesizeRequest):` empty body → **SyntaxError on import** → crashes v2_advanced startup (not caught by `except ImportError`).
2. **Mixed import paths** (`from backend...` + `from sahayak_ai_v3.backend...`) — boot depends on both repo root and `sahayak_ai_v3/` being importable; legacy `backend/` must stay.
3. **`sahayak_ai_v3/render.yaml`** start cmd `src.main:app` does not exist; real app `backend/api/main.py`. Root render.yaml correct.
4. **Rerank/grade runtime break:** `core/models.py:43` (`reranker_score`) has no `model_config` allowing extras, but `fusion.py` assigns it → `ValidationError`; and `sparse.py:89-96` reads `content/source/modality` from the wrong dict level built at `sparse.py:39-42` → BM25 docs come back with empty content. CRAG loop is not end-to-end functional yet despite modules existing.
5. **v3 features not yet in production UI:**
   - **Streaming tokens (SSE):** zero streaming code anywhere in `sahayak-ui` (frontend does full JSON exchange only; backend has no SSE route mounted either).
   - **PDF bbox citations:** `CitationViewer.jsx` + `ChatMessage.jsx` exist and chain `ChatMessage → CitationViewer` (react-pdf + bbox overlay) but **nothing imports `ChatInterface`/`ChatMessage`** — dead chain; `SearchChat.jsx` only renders text sources, no bbox/PDF canvas.
   - **Neo4j force-graph modal:** `KnowledgeGraphViewer.jsx` (react-force-graph-2d) exists but is **not imported** by any page.
   - **Groq-Whisper voice chat:** `useVoice.js` (browser Web Speech API) is live in `SearchChat`; `VoiceRecorder.jsx` (axios → `/api/v2/chat/voice-stream`) is **dead** and has stale axios deps.
   → migration checklist items 2–4 in §3 are effectively NOT started in code.
6. **Frontend↔backend surface mismatch:** `sahayak-ui` targets legacy unversioned legacy endpoints; the v3 app exposes `/api/v2|v3/*`. Merging them needs either legacy-compat routes on the v3 app or an `api/client.js` rewrite.
7. **Dead code in `sahayak-ui`:** `App.css` orphaned; `components/{ChatInterface, ChatMessage, CitationViewer, KnowledgeGraphViewer, VoiceRecorder}` ≈ 500 LOC unreachable; `dist/` stale vs source.
8. **Global CORS:** v3 `allow_origins=["*"]` + `allow_credentials=True` — invalid combo for real browsers; must whitelist deploy origins.
9. **`.env` vs `render.yaml`:** pipeline version `legacy` (local) vs `v2_advanced` (Render) — different feature surface locally vs prod.
10. **`gpu_test.py` imports `torch`** — dev-only; must never enter a deploy path.
11. Two near-duplicate supervisors (`agents/supervisor.py` vs `services/agents/supervisor.py`) and two `AgentState`/`GraphState` pairs — risk of drift.

---

## 8. Verification & Run Commands

### Backend (v3 orchestrator — repo root)
```bash
# everything (legacy + v2_advanced, requires sahayak_ai_v3/requirements.txt installed)
uvicorn sahayak_ai_v3.backend.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# health probes
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v2/health/ping

# v2 chat (requires pipeline version = v2_advanced in env, plus 1 boot-fix for audio.py)
curl -X POST http://127.0.0.1:8000/api/v2/chat/orchestrate \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is RAG?\", \"user_id\": \"guest_user\"}"
```

### Legacy backend (serves production UI `sahayak-ui`)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Frontend
```bash
# Production UI (React 19 + Vite 8)
cd sahayak-ui && npm install && npm run dev          # dev
npm run build && npm run preview                     # build + preview dist

# Classic UI (Streamlit)
cd frontend && streamlit run app.py

# Experimental v3 SPA (currently broken — missing src/api/v2_client)
cd sahayak_ai_v3/frontend && npm install && npm run build
```

### Tests
```bash
# v3 app + backend unit suites
python -m pytest sahayak_ai_v3/tests sahayak_ai_v3/backend/tests

# legacy suites (7 of 14 currently fail per 26-Aug-2026 report)
python -m pytest backend/tests

# load tests (50-VU Locust; k6 CI benchmark with threshold gates)
python -m pytest  # n/a — see sahayak_ai_v3/tests/load/locustfile.py and k6_benchmark.js
```

### OpenAPI / schema verification
```bash
# live export
curl -s http://127.0.0.1:8000/openapi.json -o openapi.json
# compare against docs/openapi.json (legacy app) — they document DIFFERENT apps; don't diff for equality
```
``