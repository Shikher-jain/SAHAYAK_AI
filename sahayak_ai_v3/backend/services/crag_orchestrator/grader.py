import logging
import os
from typing import Tuple, List
from sahayak_ai_v3.backend.core.models import RetrievedDocument

logger = logging.getLogger(__name__)

def grade_retrieval(query: str, refined_pool: List[RetrievedDocument]) -> Tuple[float, str]:
    """
    CRAG Relevance Grader.
    Evaluates the top reranked chunks against the query using a fast LLM.
    Returns: (confidence_score, category["GOOD", "PARTIAL", "BAD"])
    """
    logger.info("Evaluating retrieval quality using CRAG Grader...")
    if not refined_pool:
        return 0.0, "BAD"

    # Combine top chunks for evaluation
    context = "\n\n".join([f"Doc {i+1}: {doc.text}" for i, doc in enumerate(refined_pool[:3])])
    
    prompt = (
        "You are a strict grading system. Determine if the following retrieved documents contain the answer to the user's question.\n"
        "Return exactly one of the following words: GOOD, PARTIAL, BAD.\n\n"
        "- GOOD: The documents fully answer the question.\n"
        "- PARTIAL: The documents provide some relevant context, but not a complete answer.\n"
        "- BAD: The documents are irrelevant to the question.\n\n"
        f"Question: {query}\n\n"
        f"Documents:\n{context}\n\n"
        "Grade:"
    )

    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        grade_text = (resp.choices[0].message.content or "").strip().upper()
        
        if "GOOD" in grade_text:
            return 0.9, "GOOD"
        elif "PARTIAL" in grade_text:
            return 0.65, "PARTIAL"
        else:
            return 0.1, "BAD"
    except Exception as e:
        logger.warning(f"CRAG Grader LLM failed: {e}. Falling back to default heuristics.")
        # Fallback heuristic: If we have documents with high rerank scores, assume GOOD.
        best_score = max((doc.rerank_score or 0) for doc in refined_pool)
        if best_score > 0.8:
            return 0.9, "GOOD"
        elif best_score > 0.5:
            return 0.65, "PARTIAL"
        return 0.1, "BAD"
