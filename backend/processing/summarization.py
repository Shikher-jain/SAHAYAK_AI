from transformers import pipeline

summarizer = pipeline('summarization')

from transformers import pipeline

summarizer = pipeline('summarization')

def summarize_text(text: str) -> str:
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']# backend/processing/summarization.py

from transformers import pipeline

class Summarizer:
    def __init__(self):
        # Load HuggingFace summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_text(self, text, max_length=150, min_length=50):
        """
        Summarize input text.
        - max_length: max tokens in summary
        - min_length: minimum tokens in summary
        """
        if len(text.strip()) == 0:
            return ""

        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    def summarize_chunks(self, chunks, max_length=150, min_length=50):
        """Summarize a list of text chunks"""
        summaries = []
        for chunk in chunks:
            summaries.append(self.summarize_text(chunk, max_length=max_length, min_length=min_length))
        return summaries
