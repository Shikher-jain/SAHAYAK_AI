from transformers import pipeline

classifier = pipeline('zero-shot-classification')

from transformers import pipeline

classifier = pipeline('zero-shot-classification')

def tag_text(text: str, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ['news', 'sports', 'finance', 'technology', 'health']
    result = classifier(text, candidate_labels)
    return result['labels']
    if candidate_labels is None:
        candidate_labels = ['news', 'sports', 'finance', 'technology', 'health']
    result = classifier(text, candidate_labels)
    return result['labels']# backend/processing/tagging.py

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Tagger:
    def __init__(self, top_k=5):
        self.top_k = top_k  # Number of tags to extract

    def extract_keywords(self, text):
        """
        Extract top-k keywords using TF-IDF
        """
        if not text or len(text.strip()) == 0:
            return []

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        top_keywords = feature_array[tfidf_sorting][:self.top_k]
        return top_keywords.tolist()

    def tag_chunks(self, chunks):
        """
        Extract tags for each chunk of text
        """
        all_tags = []
        for chunk in chunks:
            tags = self.extract_keywords(chunk)
            all_tags.append(tags)
        return all_tags
