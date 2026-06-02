from .db import build_faiss_index
# Deprecated: use backend.common.embedder as the unified embedding source.
from backend.common.embedder import embed_text
from .rag_base import answer_question_with_store


def answer_question(question: str, top_k: int = 5) -> str:
    return answer_question_with_store(
        question,
        embed_text=embed_text,
        build_faiss_index=build_faiss_index,
        top_k=top_k,
    )
