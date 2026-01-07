def rewrite_query(query: str) -> str:
    # Placeholder: implement query rewriting logic
    return query
def rewrite_query(query: str) -> str:
    # Placeholder: implement query rewriting logic
    return query
# backend/rag/query_rewrite.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QueryRewriter:
    def __init__(self):
        # Small embedding model to compute query similarity
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def expand_query(self, query, related_phrases=None, top_k=3):
        """
        Expand query using related phrases or semantic similarity
        - query: original query string
        - related_phrases: optional list of related phrases for expansion
        Returns: expanded query string
        """
        if not related_phrases or len(related_phrases) == 0:
            return query  # no expansion

        # Compute embeddings
        query_vec = self.model.encode([query])[0]
        phrases_vec = self.model.encode(related_phrases)

        # Compute similarity and pick top_k
        sims = cosine_similarity([query_vec], phrases_vec)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        top_phrases = [related_phrases[i] for i in top_indices]

        # Combine original query with top related phrases
        expanded_query = query + " " + " ".join(top_phrases)
        return expanded_query
