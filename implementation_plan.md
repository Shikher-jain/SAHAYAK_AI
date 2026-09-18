# Implementation Plan: Autonomous Monetization, Agents & UX

This document outlines the approach for expanding the `v2_advanced` architecture to support a Multi-Agent Supervisor with memory, a Stripe Payment Gateway, and advanced UX features (Streaming Voice & Visual Citations), all while strictly adhering to the 512MB Render RAM ceiling.

## User Review Required

> [!IMPORTANT]
> - **Stripe Setup:** You will need to add `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET`, and `STRIPE_PRICE_ID` to your `.env` file.
> - **Database:** The payment module will require a lightweight DB connection. We will use the existing `backend.auth_system.database` (SQLite by default, which complies with 512MB RAM constraints).
> - **Neo4j AuraDB:** Ensure your `.env` has valid `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` credentials to execute the Recommender Agent graph queries.
> - **LangGraph Checkpointer:** We will use `SqliteSaver` (or an in-memory saver) for LangGraph state persistence, which keeps memory overhead low compared to Redis.

## Proposed Changes

---

### Task 1: Stripe Payment Gateway (FastAPI)

#### [NEW] `sahayak_ai_v3/backend/api/payments.py`
- Create the Stripe integration router.
- **`POST /api/v2/payments/create-checkout-session`**: Generates a Stripe checkout URL for the premium tier.
- **`POST /api/v2/payments/webhook`**: Securely parses the Stripe signature (`stripe.Webhook.construct_event`) and updates the user's tier in the auth database.
- **Dependency `verify_premium_tier`**: A FastAPI dependency function to gate specific agent capabilities (like the Recommender).

#### [MODIFY] `backend/auth_system/database.py` (or similar DB model)
- Ensure the user model has a `tier` or `subscription_status` column (e.g., "free", "premium") that the webhook can update.

---

### Task 2: Multi-Agent Supervisor Updates (LangGraph)

#### [MODIFY] `sahayak_ai_v3/backend/agents/supervisor.py`
- **Checkpointer:** Integrate `langgraph.checkpoint.memory.MemorySaver` or `langgraph.checkpoint.sqlite.SqliteSaver` to maintain long-term session memory for the Counseling Agent.
- **HITL Distress Detection:** Enhance the `counseling_agent_node` with a strict interrupt. If the `distress_level` $\ge 0.80$, the graph pauses and returns an immediate emergency fallback response without recursive generation.
- **Recommender Graph Traversal:** Implement the Neo4j AuraDB Cypher query within the `recommender_agent_node` (`MATCH (u:User)-[:LIKES]->(g:Genre)-[:HAS_ITEM]->(i) RETURN i`), merging results with Qdrant vector metadata.

---

### Task 3: Unbeatable UX Capabilities (Streaming & Citations)

#### [MODIFY] `sahayak_ai_v3/backend/api/chat.py`
- **`WebSocket /api/v2/chat/voice-stream`**: Add a new WebSocket endpoint that receives 5-10 second audio blobs.
- Implement an async offload to the **Groq Whisper API** for near real-time voice-to-text. The transcribed text is then piped into the Supervisor graph, and responses are streamed back to the client.

#### [MODIFY] `sahayak_ai_v3/backend/core/graph_state.py` & Generation Logic
- **Visual Citations:** Update the `RetrievedDocument` schema and `CRAGState` to explicitly carry `bbox` (bounding box) and `page_number` properties. 
- Ensure the final generation payload returns these exact coordinates alongside the text citations so the frontend can render cropped document images.

---

### Dependencies

#### [MODIFY] `requirements.txt`
- Add `stripe>=8.0.0`
- Add `neo4j>=5.14.0` (for AuraDB connection)

## Verification Plan

### Automated Tests
- We will mock the `stripe.Webhook.construct_event` to verify database tier updates.
- We will mock the Groq Whisper API to test the WebSocket audio blob parsing.

### Manual Verification
1. Run `uvicorn backend.main:app` and verify RAM usage remains under 100MB idle.
2. Trigger the Stripe webhook endpoint manually using the Stripe CLI to ensure subscription upgrades work locally.
3. Test the WebSocket endpoint using a simple HTML/JS snippet to send audio chunks.
