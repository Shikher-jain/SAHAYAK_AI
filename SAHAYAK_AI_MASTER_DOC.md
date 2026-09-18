# Sahayak AI - Master Project Documentation

## 1. PROJECT OVERVIEW

**Sahayak AI** is a highly advanced, full-stack enterprise multimodal virtual assistant. Engineered to run efficiently under constrained environments (e.g., Render Free Tier 512MB RAM), the system leverages a completely serverless, zero-local-compute architecture. 

**Core Problem Solved:** 
Modern enterprise AI systems often require heavy local infrastructure and struggle with multimodal inputs or empathetic user routing. Sahayak AI solves this by offloading heavy ML inference to managed APIs (Groq, Hugging Face Serverless, Edge-TTS) and utilizing a sophisticated LangGraph-based Multi-Agent routing system. This ensures the assistant can handle factual documentation queries, provide empathetic counseling, or make recommendations seamlessly, without breaking memory constraints.

**Primary Features:**
*   **Multi-Agent Routing:** A LangGraph Supervisor dynamically routes user queries to specialized sub-agents (RAG, Counseling, Recommender).
*   **Hybrid CRAG + GraphRAG:** Corrective Retrieval-Augmented Generation using Qdrant (Dense/Sparse) backed by Neo4j (Graph relationships).
*   **Multimodal Input:** Processes voice (via Groq Whisper), PDFs (with bounding-box citations), and standard text.
*   **Zero Local Compute:** All LLM and embedding tasks are strictly offloaded via async REST/HTTP clients.
*   **Omnichannel Support:** Integrated with a React SPA frontend, Telegram Bot Webhooks, and WhatsApp Meta Cloud APIs.

---

## 2. TECH STACK & ARCHITECTURE

### Backend
*   **Framework:** FastAPI (Python 3.10+), Uvicorn.
*   **Agent Orchestration:** LangGraph, LangChain Core.
*   **LLM Engine:** Groq Cloud API (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`), OpenAI (fallback).
*   **Speech Services:** Groq Whisper (STT), Microsoft Edge-TTS (TTS).
*   **Network:** `httpx` for all async external API calls.

### Frontend
*   **Framework:** React 19 (via Vite 8).
*   **Styling:** TailwindCSS, Vanilla CSS (`index.css`).
*   **State & API:** Context API (`AppContext.jsx`), custom `fetch`-based client (`api/client.js`).
*   **Visualization:** `react-force-graph-2d` for Knowledge Graphs, `react-pdf` for document citations.

### Database & Storage
*   **Relational DB:** PostgreSQL (via SQLAlchemy) for Auth, Users, and Payments.
*   **Vector DB:** Qdrant Cloud (Dense and Sparse embeddings).
*   **Graph DB:** Neo4j AuraDB (Semantic property graphs).

### Data Flow Architecture
1.  **Ingress:** User inputs text, voice, or files via React UI, Telegram, or WhatsApp.
2.  **Input Guardrail:** Input is validated via an LLM (llama-3/gpt-oss-20b) for safety/malice.
3.  **Supervisor Router:** LangGraph supervisor evaluates intent and routes to one of three agents (RAG, Counseling, Recommender).
4.  **Specialized Agent Execution:**
    *   *RAG Agent:* Extracts dense vectors via HuggingFace Serverless, searches Qdrant, reranks via BAAI/bge-reranker, and synthesizes an answer via Groq.
    *   *Counseling Agent:* Performs an emergency triage check. If safe, provides empathetic response; if in crisis, returns de-escalation text and helpline numbers.
    *   *Recommender Agent:* Extracts entities via Groq, queries Neo4j, and formats subgraph relations.
5.  **Output Guardrail:** (For RAG) The final draft is checked against retrieved context to prevent hallucinations.
6.  **Egress:** Final synthesized response, visual citations (bounding boxes), and TTS streams are returned to the client.

---

## 3. DIRECTORY STRUCTURE

```text
d:\shikher sih\SAHAYAK_AI\
├── backend/                       # Legacy backend / API endpoints
│   ├── auth_system/               # Auth routes, SQLAlchemy models, DB init
│   ├── ingestion/                 # Legacy document parsers (PDF, image, audio)
│   ├── routers/                   # Core FastAPI routers (search, voice, admin)
│   └── main.py                    # Primary FastAPI application entry point
├── sahayak_ai_v3/                 # V3 Multi-Agent Engine (Current Core)
│   ├── backend/
│   │   ├── agents/                # LangGraph Agents (Supervisor, RAG, Counseling, etc.)
│   │   ├── api/                   # V3 specific API routes (voice, payments, health)
│   │   ├── core/                  # Core config, exceptions, schema definitions
│   │   └── webhooks/              # Omnichannel Webhooks (Telegram, WhatsApp)
├── sahayak-ui/                    # Production React 19 + Vite 8 SPA
│   ├── src/
│   │   ├── api/                   # Fetch client for backend communication
│   │   ├── components/            # Reusable UI components
│   │   ├── context/               # React Context (Auth, state)
│   │   ├── pages/                 # Full screen views (Dashboard, Chat, KnowledgeGraph)
│   │   ├── App.jsx                # Main React Router/Layout wrapper
│   │   └── index.css              # Global styles
├── AGENTS.md                      # strict multi-agent and system guardrail rules
├── PROJECT_STATUS.md              # Live single source of truth for repository state
├── requirements.txt               # Python dependencies for the backend
└── .env                           # API Keys, DB URIs, Config flags
```

**Critical Files:**
*   `sahayak_ai_v3/backend/agents/supervisor.py`: The brain of the LangGraph network, routing requests.
*   `sahayak_ai_v3/backend/agents/state.py`: The single source of truth `TypedDict` that flows through all LangGraph nodes.
*   `sahayak-ui/src/api/client.js`: Universal fetch wrapper handling auth headers and backend base URLs.
*   `backend/main.py`: The FastAPI server initialization, middleware setup, and router inclusion.

---

## 4. BACKEND IMPLEMENTATION & AGENTS

### Multi-Agent System (LangGraph)
The system operates as a state machine where `GraphState` is passed sequentially.
*   **Supervisor (`supervisor.py`):** Uses `ChatGroq(llama-3.3-70b)` with `with_structured_output` to yield a `RouteDecision`. It includes circuit breakers for infinite loops and a hard-coded distress keyword override.
*   **Counseling Agent (`counseling_agent.py`):** Calculates a `distress_score`. If > 0.8, it enters emergency mode, softening the system prompt and appending a static `CRISIS_HELPLINES` string.
*   **Recommender Agent (`recommender_agent.py`):** Extracts an entity from the prompt, queries Neo4j for 1-to-2 hop relationships, and uses Groq to synthesize the graph data into a bulleted list.
*   **RAG Agent (`rag_agent.py`):** The primary knowledge engine.

### Hybrid RAG Pipeline Implementation
1.  **Embeddings:** Calls Hugging Face Serverless API (`BAAI/bge-m3`) via `httpx` to get dense query embeddings.
2.  **Retrieval:** Queries `AsyncQdrantClient`. Falls back to a scroll/sparse search if the dense vector fails.
3.  **Reranking:** Sends the top candidates to a Hugging Face Serverless Reranker (`BAAI/bge-reranker-base`) via `httpx`.
4.  **Citations:** Caps to top 3 chunks, extracting the `bbox` (bounding box) and `page_number` for the frontend to highlight PDFs.
5.  **Synthesis:** Groq creates the final answer based purely on the formatted context string.

### Guardrails (`guardrails.py`)
Implemented via the `GuardrailManager` class:
*   **Input Guardrail (`check_input_safety`):** Evaluates user input against `SAFE_AND_RELEVANT`, `MALICIOUS`, or `OUT_OF_SCOPE` classifications.
*   **Output Guardrail (`check_factual_consistency`):** Evaluates the LLM's draft answer against the retrieved chunks to guarantee 100% grounding and identify hallucinations before the user sees the output.

---

## 5. FRONTEND IMPLEMENTATION

### Framework & State
*   **React 19:** Utilized exclusively in `sahayak-ui`. The legacy Streamlit `frontend/` directory is deprecated.
*   **State Management:** `AppContext.jsx` acts as the global state, holding `authToken`, `currentPage`, and user session data. Navigation is handled by conditionally rendering components in `App.jsx` based on `currentPage`.

### Client-API Interaction
*   **`client.js`:** A robust fetch wrapper that dynamically pulls the JWT and API keys from `localStorage`, appends them to headers, formats payloads (handling both `application/json` and `FormData`), and standardizes error formats for the UI.

### Multimodal Handling
*   **Audio Inputs:** Handled by capturing MediaStream blobs in the browser, packing them into `FormData`, and POSTing to `/api/v3/voice/transcribe` or `/api/v3/voice/process-voice-chat`.
*   **Document Highlighting:** `react-pdf` renders the underlying PDF, and absolute-positioned HTML `div` overlays are drawn using the `bbox` coordinates returned by the RAG Agent.
*   **Knowledge Graphs:** `react-force-graph-2d` is used in `KnowledgeGraph.jsx` to render the nodes and links returned by the Recommender Agent.

---

## 6. DATABASE SCHEMAS

### PostgreSQL (Relational - `backend/auth_system/models.py`)

**Users Table (`users`)**
```python
id = Integer (PK, Auto-increment)
username = String(64) (Unique, Not Null)
email = String(128) (Unique, Not Null)
hashed_password = String(256) (Not Null)
role = String(20) (Default: "student")
full_name = String(128)
is_active = Boolean (Default: True)
tier = String(20) (Default: "free")
created_at = DateTime
updated_at = DateTime
```

**Orders Table (`orders`)** - *Used for UPI Payments*
```python
id = String(64) (PK - UUID)
user_id = Integer (Not Null, Indexed)
amount = Integer (Not Null)
status = String(20) (Default: "PENDING")
utr_number = String(20) (Unique, Nullable)
created_at = DateTime
```

### Qdrant Vector Schema (RAG - `backend/core/models.py`)
Stored as payloads attached to vector points. Represented in code via `RetrievedDocument` extending `EvidenceUnit`:
```json
{
  "id": "uuid",
  "document_id": "doc-uuid",
  "modality": "text/image/audio",
  "content": "The actual raw text chunk...",
  "page_number": 1,
  "bbox": [x0, y0, x1, y1], // For visual citations
  "dense_score": 0.95,
  "metadata": { ... }
}
```

---

## 7. SETUP & DEPLOYMENT INSTRUCTIONS

### Step 1: Environment Variables
Create a `.env` file in the root directory:
```env
# AI APIs
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_hf_key
OPENAI_API_KEY=your_openai_key # Optional fallback

# Databases
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Authentication
JWT_SECRET=super_secret_string
```

### Step 2: Backend Setup
1. Open a terminal in `d:\shikher sih\SAHAYAK_AI\`.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server (Render Free Tier configuration):
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
   ```

### Step 3: Frontend Setup
1. Open a new terminal in `d:\shikher sih\SAHAYAK_AI\sahayak-ui`.
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

### Step 4: Access the Application
*   **Web App:** Navigate to `http://localhost:5173` (or the port Vite provides) in your browser.
*   **API Docs:** Navigate to `http://127.0.0.1:8000/docs` to view the Swagger documentation.
