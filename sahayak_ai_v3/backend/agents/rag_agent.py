import os
import logging
from typing import Any, Optional
import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from sahayak_ai_v3.backend.agents.graph_state import GraphState

logger = logging.getLogger("sahayak.rag_agent")

# Environment Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "sahayak_documents")

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN", ""))
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"
HF_RERANK_URL = "https://api-inference.huggingface.co/models/BAAI/bge-reranker-base"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


async def _embed_query_serverless(query: str, client: httpx.AsyncClient) -> Optional[list[float]]:
    """Generate dense query embeddings via Hugging Face Serverless API (Zero local RAM)."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not configured; skipping dense vector generation.")
        return None
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": query, "options": {"wait_for_model": True}}
    
    try:
        response = await client.post(HF_EMBED_URL, headers=headers, json=payload, timeout=6.0)
        if response.status_code == 200:
            embedding = response.json()
            # If batched response [1, dim], unpack to [dim]
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                return embedding[0]
            return embedding
        logger.error("HF Embed API returned status %d: %s", response.status_code, response.text)
    except Exception as exc:
        logger.warning("HF Serverless Embeddings failed: %s", exc)
    return None


async def _rerank_chunks_serverless(
    query: str, 
    candidates: list[dict[str, Any]], 
    client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    """
    Rerank candidates using BAAI/bge-reranker-base via Hugging Face REST.
    Falls back to initial Qdrant scores if the service fails.
    """
    if not HF_TOKEN or not candidates:
        return candidates

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    pairs = [[query, c.get("text", "")] for c in candidates]
    payload = {"inputs": pairs, "options": {"wait_for_model": True}}

    try:
        response = await client.post(HF_RERANK_URL, headers=headers, json=payload, timeout=6.0)
        if response.status_code == 200:
            scores = response.json()
            # If list of float scores returned
            if isinstance(scores, list) and len(scores) == len(candidates):
                for idx, score in enumerate(scores):
                    candidates[idx]["rerank_score"] = float(score if isinstance(score, (int, float)) else score.get("score", 0.0))
                return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception as exc:
        logger.warning("HF Serverless Rerank failed; falling back to vector score: %s", exc)
    
    return candidates


async def rag_agent(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node: Hybrid document retrieval with serverless reranking 
    and PDF bounding-box citation extraction.
    """
    # 1. Extract the latest user query from messages
    query_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            query_text = str(msg.content)
            break

    if not query_text.strip():
        return {
            "messages": [AIMessage(content="I could not find a valid question to answer.")],
            "visual_citations": [],
            "error": "EMPTY_QUERY"
        }

    retrieved_chunks: list[dict[str, Any]] = []
    retrieval_error: Optional[str] = None

    # 2. Async Qdrant Retrieval with Circuit Breaker
    if QDRANT_URL:
        async with httpx.AsyncClient() as http_client:
            try:
                dense_vector = await _embed_query_serverless(query_text, http_client)
                
                qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5.0)
                
                # Execute search: use dense vector if available, fallback to match text
                if dense_vector:
                    search_results = await qdrant.search(
                        collection_name=QDRANT_COLLECTION,
                        query_vector=dense_vector,
                        limit=6,
                        with_payload=True
                    )
                else:
                    # Sparse/scroll fallback when no vector could be computed
                    scroll_result, _ = await qdrant.scroll(
                        collection_name=QDRANT_COLLECTION,
                        limit=6,
                        with_payload=True
                    )
                    search_results = scroll_result

                await qdrant.close()

                for point in search_results:
                    payload = point.payload or {}
                    retrieved_chunks.append({
                        "id": str(point.id),
                        "score": getattr(point, "score", 0.0),
                        "text": payload.get("text", payload.get("content", "")),
                        "doc_id": payload.get("doc_id", "unknown_doc"),
                        "document_url": payload.get("document_url", ""),
                        "page_number": int(payload.get("page_number", 1)),
                        "bbox": payload.get("bbox", [0.0, 0.0, 0.0, 0.0])  # [x0, y0, x1, y1]
                    })

                # 3. Serverless Cross-Encoder Reranking
                if retrieved_chunks:
                    retrieved_chunks = await _rerank_chunks_serverless(query_text, retrieved_chunks, http_client)

            except Exception as exc:
                logger.error("Qdrant retrieval pipeline error: %s", exc)
                retrieval_error = f"RETRIEVAL_FAILED: {str(exc)}"
    else:
        retrieval_error = "QDRANT_NOT_CONFIGURED"

    # 4. Cap to Top 3 Chunks & Build Visual Citations
    top_chunks = retrieved_chunks[:3]
    visual_citations = []
    context_parts = []

    for chunk in top_chunks:
        context_parts.append(f"[{chunk['doc_id']} p.{chunk['page_number']}]: {chunk['text']}")
        visual_citations.append({
            "doc_id": chunk["doc_id"],
            "document_url": chunk["document_url"],
            "page_number": chunk["page_number"],
            "bbox": chunk["bbox"],
            "snippet": chunk["text"][:160] + "..." if len(chunk["text"]) > 160 else chunk["text"]
        })

    # 5. Formulate Context & LLM Synthesis
    if top_chunks:
        formatted_context = "\n\n".join(context_parts)
        system_prompt = (
            "You are the RAG Agent for Sahayak AI. Answer the user's question accurately using ONLY "
            "the provided reference context. Always cite your sources in brackets like [doc_id p.X]. "
            "If the context is insufficient, acknowledge the limitation clearly.\n\n"
            f"Context:\n{formatted_context}"
        )
    else:
        system_prompt = (
            "You are Sahayak AI. The document retrieval service is currently unavailable or returned "
            "no relevant excerpts. Answer the user's question concisely using your general knowledge, "
            "and explicitly notify them that primary document citations could not be verified."
        )

    # 6. Call Groq Cloud API (Zero local compute)
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_retries=2,
            groq_api_key=GROQ_API_KEY
        )
        llm_response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text}
        ])
        final_answer = str(llm_response.content)
    except Exception as exc:
        logger.error("Groq generation failed in RAG agent: %s", exc)
        final_answer = (
            "I encountered an issue synthesizing a response from our knowledge base. "
            "Please try again shortly."
        )
        retrieval_error = f"GROQ_GENERATION_FAILED: {str(exc)}"

    return {
        "messages": [AIMessage(content=final_answer)],
        "visual_citations": visual_citations,
        "error": retrieval_error
    }
