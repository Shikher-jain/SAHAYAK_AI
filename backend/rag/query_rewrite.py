from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.common.embedder import embed_text, embed_texts


def rewrite_query(query: str) -> str:
    return query.strip()

class QueryRewriter:
    def __init__(self):
        pass

    def expand_query(self, query, related_phrases=None, top_k=3):
        if not related_phrases or len(related_phrases) == 0:
            return query

        query_vec = embed_text(query)
        phrases_vec = embed_texts(related_phrases)

        sims = cosine_similarity([query_vec], phrases_vec)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        top_phrases = [related_phrases[i] for i in top_indices]

        expanded_query = query + " " + " ".join(top_phrases)
        return expanded_query
