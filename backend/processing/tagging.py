from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tag_text(text: str, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ["news", "sports", "finance", "technology", "health"]
    lowered = text.lower()
    ranked = sorted(candidate_labels, key=lambda label: lowered.count(label.lower()), reverse=True)
    return ranked

class Tagger:
    def __init__(self, top_k=5):
        self.top_k = top_k

    def extract_keywords(self, text):
        """
        Extract top-k keywords using TF-IDF
        """
        if not text or len(text.strip()) == 0:
            return []

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        top_keywords = feature_array[tfidf_sorting][:self.top_k]
        return top_keywords.tolist()

    def tag_chunks(self, chunks):
        """
        Extract tags for each chunk of text
        """
        return [self.extract_keywords(chunk) for chunk in chunks]
