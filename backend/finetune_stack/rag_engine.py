from backend.local_stack.rag_base import answer_question_with_store

from .db import build_faiss_index
from .embedder import embed_text


def answer_question(question: str, top_k: int = 5) -> str:
    return answer_question_with_store(
        question,
        embed_text=embed_text,
        build_faiss_index=build_faiss_index,
        top_k=top_k,
    )
