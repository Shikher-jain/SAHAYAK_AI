from __future__ import annotations

from typing import Callable

import numpy as np


def answer_question_with_store(
    question: str,
    *,
    embed_text: Callable[[str], np.ndarray],
    build_faiss_index: Callable[[], tuple[object, list[str]]],
    top_k: int = 5,
) -> str:
    try:
        query_vec = embed_text(question)
        index, texts = build_faiss_index()

        if getattr(index, "ntotal", 0) == 0:
            return "No documents uploaded yet. Please upload PDF/image files first."

        distances, indices = index.search(np.array([query_vec], dtype="float32"), top_k)
        retrieved_chunks = [texts[i] for i in indices[0] if i < len(texts)]
        retrieved_chunks = [chunk for chunk in retrieved_chunks if chunk and chunk.strip()]

        if not retrieved_chunks:
            return "No relevant information found in uploaded documents."

        context = "\n\n".join(retrieved_chunks)
        return (
            "Based on the uploaded documents:\n\n"
            f"{context}\n\n"
            f"Relevant to your question: {question}"
        )
    except Exception as exc:
        return f"Error processing question: {exc}"
