# AGENTS.md

**Repository Operating Rules and Multi-Agent Architecture for Sahayak AI**

This document serves a dual purpose:
1. It defines the **Multi-Agent Architecture** (the runtime agents operating within the LangGraph CRAG loop).
2. It sets the **Repository Operating Rules** for AI Coding Assistants (Cursor, Windsurf, Copilot, Devin, Antigravity) to ensure automated code changes never violate system constraints.

---

## Part 1: System Architecture & Agent Topology

Sahayak AI utilizes a multi-agent graph orchestrated via LangGraph, running a highly optimized Multimodal Hybrid CRAG (Corrective RAG) + GraphRAG pipeline.

### 1. Router / Query Understanding Agent
- **Responsibilities:** Modality classification (text, image, audio, video), entity recognition, and query decomposition.
- **Behavior:** Parses the incoming query to determine which retrieval pipelines to activate and whether the query requires complex multi-hop graph traversal.

### 2. Retrieval Coordination Node
- **Responsibilities:** Manages parallel async dispatch across multiple retrieval systems.
- **Data Sources:** 
  - **Dense:** Qdrant Cloud (Vector search)
  - **Sparse:** BM25 (Keyword search)
  - **Graph:** Neo4j AuraDB (Relationship traversal)
- **Aggregation:** Merges results from all pipelines using Reciprocal Rank Fusion (RRF).

### 3. Reranker Integration
- **Responsibilities:** Refines the initial candidate pool (Top 30-100 chunks) to the most relevant top 10-20 matches.
- **Specification:** Sends the candidate pool to the Hugging Face Serverless Inference API running `BAAI/bge-reranker-base`.

### 4. Relevance Grader Agent (CRAG)
- **Responsibilities:** Evaluates query-document alignment using a fast LLM-as-a-judge.
- **Threshold Logic:**
  - **High Confidence (>= 0.85):** Proceed directly to synthesis.
  - **Ambiguous (0.60 <= Confidence < 0.85):** Trigger Query Rewriter + Graph multi-hop retrieval.
  - **Low Confidence (< 0.60):** Trigger External Fallback Search (e.g., Tavily/DDG API).

### 5. Context Engine & Synthesizer Agent
- **Responsibilities:** Prepares the final payload for the LLM. 
- **Tasks:** Context deduplication, parent-child chunk expansion, token budget packing, and grounded generation with strict citation markers.

### 6. Faithfulness / Grounding Checker Agent
- **Responsibilities:** A post-generation LLM-as-a-judge node that verifies whether output claims are 100% supported by the retrieved evidence before dispatching the response to the client.

---

## Part 2: Agent State Schema (LangGraph Contract)

All nodes in the CRAG loop communicate via a shared typed state (`CRAGState`) defined in `sahayak_ai_v3/backend/core/graph_state.py`. Do NOT rename or restructure these fields — every node reads and writes through them:

```python
class CRAGState(TypedDict):
    # Query Understanding
    original_query: str
    active_query: str                 # current query (original or rewritten)
    intent: Optional[str]
    detected_modality: Optional[str]  # text | image | audio | video
    extracted_entities: List[str]

    # Retrieval & Reranking
    candidate_pool: List[RetrievedDocument]  # top 30-100 after RRF (Qdrant + BM25 + Neo4j)
    refined_pool: List[RetrievedDocument]    # top 10-20 after BAAI/bge-reranker-base

    # CRAG Grading
    relevance_score: Optional[float]
    confidence_category: Optional[str]       # "GOOD" (>=0.85) | "PARTIAL" (0.60-0.84) | "BAD" (<0.60)

    # Control Flow (prevents infinite LangGraph loops)
    rewrite_count: int                       # capped by MAX_RETRIEVAL_ATTEMPTS
    web_search_invoked: bool

    # Context Engine
    compressed_context: Optional[str]

    # Generation & Faithfulness
    draft_response: Optional[str]
    is_faithful: Optional[bool]
    final_response: Optional[str]

    # Metadata
    filters: Optional[Dict[str, Any]]
    sources: Optional[List[Dict[str, str]]]
    learning_mode: Optional[str]
    user_mode: Optional[str]
```

Tune loop ceilings via `config.settings.MAX_RETRIEVAL_ATTEMPTS` (default 2) — never via a new counter in state.

---

## Part 3: Non-Negotiable Rules for AI Coding Agents

**ATTENTION ALL AI CODING ASSISTANTS:** You must strictly obey the following rules when modifying this repository.

### Rule 1: Memory Guardrail (Zero Local Heavy Compute)
**Context:** This application is deployed on a Render Free Tier (~512MB RAM constraint).
- **Requirement:** You MUST reject any PR, code snippet, or plan that introduces `import torch`, `sentence_transformers`, or local heavy model initializations. 
- **Enforcement:** All heavy compute (Whisper, OCR, Vectors, LLMs, Reranking) MUST be offloaded to managed APIs (Groq, Hugging Face Inference API, Gemini, Qdrant Cloud, Neo4j AuraDB). Use lightweight async HTTP clients (`httpx`, `aiohttp`).

### Rule 2: Pipeline Isolation & Backward Compatibility
- **Requirement:** Legacy routes (`/api/v1/...` and existing `backend/`) MUST remain intact. Never delete or break existing flows.
- **Enforcement:** All new v2/v3 features are housed in `sahayak_ai_v3/` and are conditionally loaded via `config.settings.SAHAYAK_PIPELINE_VERSION` (e.g., `legacy` vs `v2_advanced`).

### Rule 3: Canonical Schema Adherence
- **Requirement:** All multimodal ingestion modules must output to the unified canonical chunk schema.
- **Enforcement:** Document chunks must map to standard fields (`chunk_id`, `document_id`, `modality`, `bbox`, `entities`, `content`, `metadata`).

### Rule 4: Error Handling & Rate Limits
- **Requirement:** The system relies entirely on external APIs, meaning network failures and rate limits are guaranteed.
- **Enforcement:** All external API calls (Groq, Hugging Face, Qdrant Cloud, Neo4j) MUST include backoff retries and graceful degradation fallbacks. If the Reranker fails, fall back to the initial RRF scores. If Graph search fails, fall back to Dense search.

### Rule 5: Reranker Specification
- **Requirement:** The Reranker MUST be `BAAI/bge-reranker-base`.
- **Enforcement:** It must ONLY be called via the Hugging Face Serverless Inference API (or equivalent external HTTP endpoint). NEVER download the local weights for this model.
