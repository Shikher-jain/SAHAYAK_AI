"""
Diagnostic script — empirically tests HF Inference API for Summarization,
QA, and NER task-names/URLs, same approach used earlier for embeddings.

Run: python test_hf_tasks.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
BASE = "https://router.huggingface.co/hf-inference/models"

if not TOKEN:
    print("ERROR: HUGGINGFACEHUB_API_TOKEN not set in .env")
    exit(1)

headers = {"Authorization": f"Bearer {TOKEN}"}

TESTS = [
    {
        "label": "Summarization",
        "model": "facebook/bart-large-cnn",
        "task": "summarization",
        "payload": {"inputs": "The quick brown fox jumps over the lazy dog. This is a longer piece of text written specifically to test whether the summarization endpoint works correctly with the Hugging Face router API. It needs to be long enough for a summarizer to meaningfully condense."},
    },
    {
        "label": "Question Answering",
        "model": "deepset/roberta-base-squad2",
        "task": "question-answering",
        "payload": {"inputs": {"question": "What is the capital of France?", "context": "France is a country in Europe. Its capital city is Paris, which is also its largest city."}},
    },
    {
        "label": "NER (Token Classification)",
        "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "task": "token-classification",
        "payload": {"inputs": "My name is Sarah and I work at Google in New York.", "parameters": {"aggregation_strategy": "simple"}},
    },
]

for t in TESTS:
    url = f"{BASE}/{t['model']}/pipeline/{t['task']}"
    print(f"--- {t['label']} ---")
    print(f"URL: {url}")
    try:
        resp = requests.post(url, headers=headers, json=t["payload"], timeout=30)
        print(f"Status: {resp.status_code}")
        print(f"Body (first 400 chars): {resp.text[:400]}")
        if resp.status_code == 200:
            print(">>> WORKS <<<")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    print()

print("Done.")