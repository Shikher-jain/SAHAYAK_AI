"""Multi-Agent Supervisor built with LangGraph.

Routes each user message to one of three workers:
  - rag_agent          : grounded, evidence-constrained answers over retrieved chunks
  - counseling_agent   : empathetic responses with an emergency / HITL short-circuit
  - recommender_agent  : Neo4j genre traversal (User -[:LIKES]-> Genre -[:CONTAINS]-> Item)

All inference goes through the Groq API (async, zero local model load — Render 512MB
safe). Routing uses Groq's JSON mode instead of langchain-groq/ChatGroq: with_structured_output
needs langchain-core>=0.3, which is incompatible with the repo-pinned langchain==0.0.350
(AGENTS.md Rule 2). Plain Groq JSON keeps the legacy env untouched.
"""
import asyncio
import json
import logging
import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph

from backend.core.config import settings

logger = logging.getLogger(__name__)

DISTRESS_THRESHOLD = 0.80

COUNSELING_EMERGENCY = (
    "It sounds like you are going through an overwhelming moment. You are not alone. "
    "Please connect directly with compassionate support:\n\n"
    "\u2022 Vandrevala Foundation Helpline (India): +91 9999 666 555 (24x7 Free)\n"
    "\u2022 Tele-MANAS: 14416 or 1800 891 4416\n"
    "\u2022 Emergency Services: 112\n\n"
    "If you are in immediate danger, please reach out to someone you trust or a professional."
)

NEO4J_GENRE_CYPHER = (
    "MATCH (u:User {id: $uid})-[:LIKES]->(g:Genre)-[:CONTAINS]->(e) "
    "WITH e, count(g) AS affinity ORDER BY affinity DESC RETURN e LIMIT 10"
)


# =====================================================================
# 1. State Definition
# =====================================================================
class AgentState(TypedDict):
    """Unified state flowing across the supervisor and specialized sub-agents."""
    messages: Annotated[List[Dict[str, str]], operator.add]
    next_agent: str
    user_id: str
    user_tier: str                      # "free" | "premium"
    distress_level: float               # 0.0 - 1.0 (safety guard for counseling)
    emergency_flag: bool                # HITL interrupt trigger
    final_output: str
    citations: List[dict]               # visual citations from the RAG agent


class RouteDecision(BaseModel):
    """Structured routing decision returned by the supervisor."""
    next_agent: Literal["rag_agent", "counseling_agent", "recommender_agent", "FINISH"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


# =====================================================================
# 2. Lightweight Groq client (Zero local RAM)
# =====================================================================
_client = None


def _groq():
    global _client
    if _client is None:
        import groq
        _client = groq.AsyncGroq(
            api_key=settings.GROQ_API_KEY,
            max_retries=2,          # Rule 4: bounded retry on transient API failures
            timeout=30.0,
        )
    return _client


def _model() -> str:
    return settings.GROQ_MODEL or "llama-3.3-70b-versatile"


async def _chat(
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = False,
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    """One Groq call. Returns '' on failure so every node can degrade gracefully."""
    kwargs = dict(
        model=_model(),
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = await _groq().chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "") or ""
    except Exception as e:
        logger.warning("Groq call failed: %s", e)
        return ""


# =====================================================================
# 3. Supervisor Node
# =====================================================================
SUPERVISOR_SYSTEM_PROMPT = (
    "You are the Central Supervisor for Sahayak AI.\n"
    "Classify the user's latest message and route it to exactly ONE worker:\n"
    "- 'counseling_agent': emotional support, mental wellness, life guidance, venting.\n"
    "- 'recommender_agent': suggestions for books, movies, courses, tech stacks, preferences.\n"
    "- 'rag_agent': factual, technical, documentation, project, or document-analysis queries.\n"
    "- 'FINISH': only for pure pleasantries such as 'bye' or 'thank you'.\n"
    "Respond with JSON only: {\"next_agent\": <worker>, \"confidence\": 0.0-1.0, \"rationale\": \"<one line>\"}"
)


def _fallback_route(text: str) -> str:
    """No-network keyword heuristic used when the Groq routing call fails."""
    t = text.lower()
    distress_words = ("sad", "anxious", "anxiety", "stress", "stressed", "depressed",
                      "lonely", "suicide", "cry", "crying", "panic", "overwhelm", "tired")
    if any(w in t for w in distress_words):
        return "counseling_agent"
    recommend_words = ("recommend", "suggestion", "suggest", "movie", "book", "course",
                       "recipe", "best", "top 5", "playlist", "series")
    if any(w in t for w in recommend_words):
        return "recommender_agent"
    return "rag_agent"


async def supervisor_node(state: AgentState) -> dict:
    """Intent classifier: routes the latest message to one worker agent."""
    raw = await _chat(
        [{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}, *state["messages"]],
        json_mode=True,
        max_tokens=60,
    )
    decision = None
    if raw:
        try:
            decision = RouteDecision.model_validate_json(raw)
        except Exception:
            logger.warning("Supervisor returned invalid JSON: %r", raw)
    if decision is None:
        decision = RouteDecision(
            next_agent=_fallback_route(state["messages"][-1]["content"]),
            confidence=0.5,
            rationale="keyword fallback (Groq routing unavailable)",
        )
    return {"next_agent": decision.next_agent}


# =====================================================================
# 4. Sub-Agent Nodes
# =====================================================================
def parse_distress(raw: Optional[str]) -> float:
    """Parse a distress score from a model response; anything unparseable = 0.0."""
    if not raw:
        return 0.0
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return 0.0


async def counseling_agent_node(state: AgentState) -> dict:
    """Empathetic counseling with an emergency (HITL) short-circuit."""
    last = state["messages"][-1]["content"]

    triage_raw = await _chat(
        [
            {"role": "system", "content": (
                "Analyze the user text for severe mental distress, self-harm, or crisis. "
                'Reply ONLY with JSON: {"score": <float 0.0 calm to 1.0 immediate crisis>}')},
            {"role": "user", "content": last},
        ],
        json_mode=True,
        max_tokens=40,
        temperature=0.0,
    )
    distress = 0.0
    if triage_raw:
        try:
            distress = max(0.0, min(1.0, float(json.loads(triage_raw).get("score", 0.0))))
        except Exception:
            distress = parse_distress(triage_raw)

    # Safety Guard: no further LLM generation past this point (HITL lane).
    if distress >= DISTRESS_THRESHOLD:
        return {"distress_level": distress, "emergency_flag": True, "final_output": COUNSELING_EMERGENCY}

    reply = await _chat(
        [
            {"role": "system", "content": (
                "You are Sahayak's Empathetic Guide. Provide a warm, supportive, grounded "
                "response. Practice active listening, validate feelings, avoid clinical jargon.")},
            *state["messages"],
        ],
        max_tokens=500,
    )
    if not reply:
        reply = "I'm here with you. Take your time — tell me more about what you're going through."
    return {
        "distress_level": distress,
        "emergency_flag": False,
        "final_output": reply,
        "messages": [{"role": "assistant", "content": reply}],
    }


# ponytail: List[str] genre vocab; extends when Canonical Schema gains a `genre`
# metadata key at ingest.
_GENRE_WORDS = (
    "fiction", "romance", "thriller", "history", "self-help", "fantasy", "science",
    "tech", "business", "mystery", "poetry", "biography", "comedy", "adventure",
)


def _qdrant_genre_entities(text: str) -> list:
    """Qdrant metadata-filter merge: docs whose metadata.genre matches the message."""
    from backend.vector_store.qdrant_store import qdrant_store  # noqa: E402 (legacy singleton)

    if not qdrant_store.is_available:
        return []
    genres = [w for w in _GENRE_WORDS if w in text.lower()]
    if not genres:
        return []
    points, _ = qdrant_store._client.scroll(
        collection_name=qdrant_store.collection_name,
        scroll_filter={
            "must": [{"key": "metadata.genre", "match": {"any": genres}}]
        },
        limit=10,
    )
    return [p.payload or {} for p in points]


def _recommender_context(state: AgentState) -> str:
    """Candidate catalog: Neo4j genre traversal first, then Qdrant genre metadata."""
    # ponytail: graph UID is the session user_id, not a real neo4j User node id;
    # alias a user_id->neo4j User node mapping when auth lands.
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        try:
            with driver.session() as session:
                recs = session.run(NEO4J_GENRE_CYPHER, uid=state["user_id"])
                hits = [dict(r["e"]) for r in recs]
        finally:
            driver.close()
        return "\n".join(str(h) for h in hits[:10])
    except Exception as e:
        logger.warning("Neo4j traversal unavailable (%s); falling back to Qdrant genre filter.", e)
        try:
            return "\n".join(str(h) for h in _qdrant_genre_entities(state["messages"][-1]["content"]))
        except Exception:
            return ""


async def recommender_agent_node(state: AgentState) -> dict:
    """Personalized recommendations grounded in the graph/genre catalog."""
    ctx = _recommender_context(state)
    sysmsg = (
        "You are Sahayak's Curated Recommender. Return 3 structured recommendations "
        "with concise justifications. When a candidate catalog is provided, ground "
        "your picks in it and cite candidates by number."
    )
    last = state["messages"][-1]["content"]
    user_payload = f"Candidate catalog:\n{ctx}\n\nUser request: {last}" if ctx else last
    rec = await _chat(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": user_payload}],
        max_tokens=600,
    )
    if not rec:
        rec = "I couldn't find matching items this time. Tell me a genre or mood and I'll refine the picks."
    return {"final_output": rec, "messages": [{"role": "assistant", "content": rec}]}


async def _retrieve_safely(query: str, user_id: str) -> list:
    """Parallel hybrid retrieval; never raises — any failure returns []."""
    try:
        from backend.services.retrieval.v3_orchestrator import _get_aggregator

        aggregator, router = _get_aggregator()
        strategy = router.route_query(query)
        tasks = {}
        if strategy.get("dense", True):
            tasks["dense"] = aggregator.dense_retriever.retrieve(query, top_k=5, filters={"user_id": user_id})
        if strategy.get("bm25", True):
            tasks["sparse"] = aggregator.sparse_retriever.retrieve(query, top_k=5, filters={"user_id": user_id})
        if strategy.get("graph", False) and aggregator.graph_retriever:
            tasks["graph"] = aggregator.graph_retriever.retrieve(query, top_k=5, filters={"user_id": user_id})
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        docs = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Retrieval worker failed: %s", r)
            elif r:
                docs.extend(r)
        return docs[:10]
    except Exception as e:
        logger.warning("Retrieval unavailable for rag_agent: %s", e)
        return []


def _visual_citations(docs: list) -> List[dict]:
    """Task 3: surfaces page + bbox for image/pdf chunks so the frontend can crop the citation."""
    out = []
    for d in docs:
        modality = d.modality or (d.metadata or {}).get("modality", "text")
        if modality not in ("image", "pdf"):
            continue
        out.append({
            "chunk_id": getattr(d, "chunk_id", None),
            "document_id": getattr(d, "document_id", None),
            "source": getattr(d, "source", None),
            "page": getattr(d, "page", None),
            "bbox": getattr(d, "bbox", None),
            "modality": modality,
            "snippet": (d.content or "")[:200],
        })
    return out[:8]


async def rag_agent_node(state: AgentState) -> dict:
    """Grounded generation over retrieved chunks with citation markers + visual citations."""
    query = state["messages"][-1]["content"]
    docs = await _retrieve_safely(query, state["user_id"])

    evidence = "\n\n".join(f"[{i + 1}] {d.content}" for i, d in enumerate(docs))
    citations = _visual_citations(docs)

    answer = await _chat(
        [
            {"role": "system", "content": (
                "You are Sahayak's Precision RAG Engine. Answer using ONLY the evidence. "
                "Mark every claim with its [n] citation. If the evidence is insufficient, "
                "say so rather than guessing.")},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {query}"},
        ],
        max_tokens=700,
    )
    if not answer:
        answer = "I don't have enough retrieved evidence to answer that confidently."
    return {
        "final_output": answer,
        "citations": citations,
        "messages": [{"role": "assistant", "content": answer}],
    }


# =====================================================================
# 5. Routing & Graph Assembly
# =====================================================================
def route_next(state: AgentState) -> str:
    """Routes on supervisor decision or short-circuits on the emergency/HITL flag."""
    if state.get("emergency_flag"):
        return END
    target = state.get("next_agent")
    if target in {"rag_agent", "counseling_agent", "recommender_agent"}:
        return target
    return END


def build_supervisor_graph():
    """Compiles the LangGraph multi-agent network."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("counseling_agent", counseling_agent_node)
    workflow.add_node("recommender_agent", recommender_agent_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "counseling_agent": "counseling_agent",
            "recommender_agent": "recommender_agent",
            "rag_agent": "rag_agent",
            END: END,
        },
    )
    workflow.add_edge("counseling_agent", END)
    workflow.add_edge("recommender_agent", END)
    workflow.add_edge("rag_agent", END)

    return workflow.compile()


# Singleton instance (importing this module builds the graph once).
supervisor_app = build_supervisor_graph()