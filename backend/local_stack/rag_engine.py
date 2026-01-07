from .embedder import embed_text
from .db import build_faiss_index
import numpy as np

def answer_question(question, top_k=5):
    try:
        query_vec = embed_text(question)
        index, texts = build_faiss_index()

        if index.ntotal == 0:
            return "No documents uploaded yet. Please upload PDF/image files first."

        distances, indices = index.search(np.array([query_vec], dtype="float32"), top_k)
        retrieved_chunks = [texts[i] for i in indices[0] if i < len(texts)]
        retrieved_chunks = [chunk for chunk in retrieved_chunks if chunk and chunk.strip()]

        if not retrieved_chunks:
            return "No relevant information found in uploaded documents."

        context = "\n\n".join(retrieved_chunks)
        response = f"""Based on the uploaded documents:

{context}

Relevant to your question: {question}"""

        return response
    except Exception as e:
        return f"Error processing question: {str(e)}"
