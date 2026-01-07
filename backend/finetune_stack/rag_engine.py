from .embedder import embed_text
from .db import build_faiss_index
import numpy as np

def answer_question(question, top_k=5):
    try:
        query_vec = embed_text(question)
        index, texts = build_faiss_index()
        
        if index.ntotal == 0:
            return "No documents uploaded yet. Please upload PDF/image files first."
        
        # Search using FAISS
        D, I = index.search(np.array([query_vec], dtype='float32'), top_k)
        retrieved_chunks = [texts[i] for i in I[0] if i < len(texts)]
        
        if not retrieved_chunks:
            return "No relevant information found in uploaded documents."
        
        # Build context from retrieved chunks
        context = "\n\n".join(retrieved_chunks)
        
        # For now, return the context as answer (you can integrate a local LLM later)
        response = f"""Based on the uploaded documents:

{context}

Relevant to your question: {question}"""
        
        return response
    except Exception as e:
        return f"Error processing question: {str(e)}"
