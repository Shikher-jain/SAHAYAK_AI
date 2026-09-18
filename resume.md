# Shikher Jain — AI / ML Backend Engineer (Generative AI)

**GitHub:** github.com/Shikherjain · **LinkedIn:** linkedin.com/in/shikher-jain-428b42255 · **Email:** [EMAIL_ADDRESS] · **Location:** India

---

## Professional Summary

AI/ML Backend Engineer specializing in **Generative AI, RAG/CRAG/GraphRAG systems, LangGraph multi-agent orchestration, and production FastAPI**. Designed and shipped **Sahayak AI**, a full-stack multimodal AI platform — hybrid retrieval, LLM-based grading/faithfulness checks, serverless semantic caching, real-time voice (STT/TTS), and Stripe/UPI monetization — deployed to **Render free tier under a strict 512 MB RAM ceiling (measured idle ~65 MB, peak ~110 MB)** by offloading all heavy compute to managed APIs (Groq, Hugging Face Serverless, Qdrant Cloud, Neo4j AuraDB, Upstash). Published research paper, 88+ API endpoints, CI/CD with black/flake8/pytest gates, and k6/Locust load-testing with explicit latency SLAs. Building LLM systems that go beyond prototypes: guarded, measured, and production-safe.

## Core Skills

- **Generative AI / LLM:** RAG pipelines, Corrective RAG (CRAG), GraphRAG, query rewriting, LLM-as-a-judge grading, faithfulness/grounding verification, Retrieval-Augmented Generation, prompt engineering, structured output (JSON mode)
- **Agent Systems:** LangGraph state graphs, multi-agent supervisor routing, typed state contracts, recursion/circuit-breaker guards, crisis-safety triage
- **ML / NLP:** Hybrid retrieval (dense + sparse + graph), Reciprocal Rank Fusion, cross-encoder reranking, embeddings (BAAI/bge-m3, MiniLM), semantic caching, token-budget context packing
- **Backend / Python:** FastAPI, uvicorn, async/await, Pydantic v2 + settings, StreamingResponse, multipart uploads, BackgroundTasks, SQLAlchemy, SQLite, webhooks (HMAC signature verification), rate limiting (token bucket), PII scrubbing
- **Cloud & Services:** Groq (llama-3.3-70b, llama-3.1-8b, whisper-large-v3), Hugging Face Serverless Inference, Qdrant Cloud, Neo4j AuraDB, Upstash Vector & Redis, OpenAI, Gemini, edge-tts, Stripe
- **DevOps / Quality:** Render deployment (512 MB free tier), Docker, GitHub Actions CI/CD, pytest + pytest-asyncio (offline/mocked), Locust + k6 load benchmarking, structured JSON logging, correlation IDs, alerting
- **Frontend:** React (18/19), Vite, Tailwind CSS, react-force-graph-2d (knowledge-graph viz), react-pdf, Streamlit

## Featured Project — Sahayak AI (Multimodal Hybrid CRAG + GraphRAG Platform)

**Role:** Principal AI Backend Engineer & Architect · **Research paper:** Zenodo — https://zenodo.org/records/20682334

### Architecture & Agent Systems
- Orchestrated a **LangGraph multi-agent supervisor** routing queries to RAG / counseling / recommender agents, with a typed `GraphState` contract, `add_messages` reducers, iteration caps (`MAX_RETRIEVAL_ATTEMPTS`), and emergency short-circuit on distress (score ≥ 0.80 → national helplines) — zero silent failure loops.
- Built a **Corrective RAG (CRAG) loop** — retrieve → rerank → grade (confidence GOOD ≥ 0.85 / PARTIAL 0.60–0.84 / BAD < 0.60) → query rewrite + graph multi-hop or web-search fallback → grounded generation → **faithfulness re-check before dispatch**. Structured output via Groq JSON mode for LLM-as-a-judge grading.
- Added **GraphRAG** over Neo4j AuraDB (Cypher 1–2 hop genre/personalization queries) with SQLite embedded-KG fallback.

### Retrieval & Ranking
- **Hybrid retrieval with `asyncio.gather`:** Qdrant Cloud dense (BAAI/bge-m3 embeddings) + BM25 sparse + Neo4j graph, merged with **Reciprocal Rank Fusion (k=60)**; top-30 → **BAAI/bge-reranker-base via Hugging Face Serverless** → top-10, graceful fallback to RRF scores (zero local model weights).
- Query decomposition ("and/vs/compared to"), intent-based query routing, citation markers `[n]` + **visual citations (page + bbox)** for PDF/image chunks.

### Caching & Performance
- **Semantic cache on Upstash Vector:** server-side embeddings, cosine similarity ≥ 0.92 = hit, 48 h TTL, async background writes via FastAPI `BackgroundTasks`, crisis/personal-query bypass enforcement (`X-Cache-Status: HIT|MISS`). Defined load-test SLAs: **cache-hit p95 < 80 ms / p99 < 120 ms, miss p95 < 1500 ms, error < 0.1%**, validated with **Locust + k6** at 50 concurrent VUs over 10 min.
- **Token-bucket rate limiting via Upstash Redis (Lua)** with `Retry-After` 429s; LRU response cache (128 entries); 5-turn conversation memory; LRU/NFKC output sanitization.

### Voice & Real-Time
- **STT:** Groq `whisper-large-v3` transcription, fully in-memory (`io.BytesIO`, 10 MB cap — no disk writes).
- **TTS:** chunked `edge-tts` neural voice streaming via `StreamingResponse` (MP3, zero RAM accumulation < 15 MB active).
- End-to-end **voice-chat turn** (STT → supervisor graph → response) and Telegram/WhatsApp bots with text + voice + document flows.

### Backend, Security & Monetization
- Delivered **88+ endpoints across 21 routers** (auth/JWT-RBAC, ingestion for pdf/audio/video/image/url/code/csv/youtube, search, summaries, quizzes, courses, commerce, roadmaps, knowledge graph, multi-language STT/RAG/TTS in en/hi/es/fr/de).
- **PII scrubber** (zero-dep regex + Verhoeff checksum for Aadhaar validation; reversible masking) wired into structured JSON logging, middleware, and alerts.
- **Monetization:** Stripe checkout + HMAC webhook → premium tier; UPI QR + asynchronous UTR verification (HMAC-SHA256, SQLite persistence); Telegram/Discord error alerts with dedupe cooldown.
- Observability: correlation IDs, exception hierarchy → 502/503 mapping, health/deep-ping keep-alive (DB + cache checks) with cron monitor.

### Frontend (3 generations)
- React 19 + Vite + Tailwind SPA (13 pages, auth gate), React 18 citation viewer with **ResizeObserver-driven page-bbox highlighting** (pt→px), MediaRecorder voice recorder with explicit stream-track cleanup, and canvas **knowledge-graph visualization** (react-force-graph-2d); legacy Streamlit multilingual UI.

### Deployment, CI/CD & Quality
- Render free tier, `--workers 1`, env-gated pipeline version; **CI/CD GitHub Actions** enforcing black + flake8 + full pytest suite; Docker Compose (Qdrant + API + UI).
- 40+ tests (offline, network-free via SDK-shim mocking): semantic cache hit/miss/bypass, rate limiter Lua logic, PII/Verhoeff, LangGraph state reducers, RRF math, ASGI endpoint & production-SSL-cert smoke tests.

## In Development / Roadmap
- Completing v2 multimodal ingestion: async PyMuPDF layout/table parsing and Groq-Whisper audio/video ingestion into the canonical chunk schema.
- Extending graph retrieval to full CRAG multi-hop traversal with entity mapping across dense + sparse + graph.
- Adding LangSmith-style production tracing/evals and CI-embedded k6 threshold gates for the 10-min load burn.
- Expanding OpenAI/Gemini fallback chains and multi-region cache warm-up for the semantic cache.

## Education & Research
- **B.Tech / B.E.** (or relevant degree) — Institution name · Year
- **Published Research Paper:** "Sahayak AI" — multimodal learning platform with hybrid RAG (Zenodo, https://zenodo.org/records/20682334).

## Certifications & Coursework (optional section)
- [Cert name] · [Issuer] · [Year] — e.g., DeepLearning.AI GenAI, FastAPI, LangGraph, MLOps.