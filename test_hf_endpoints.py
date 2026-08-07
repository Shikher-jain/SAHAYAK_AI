"""
Diagnostic script — tests multiple HF Inference API endpoint URL patterns
to find which one currently works from this machine/network.

Run: python test_hf_endpoints.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
# load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
# print(TOKEN)    

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
if not TOKEN:
    print("ERROR: HUGGINGFACEHUB_API_TOKEN not set in .env")
    exit(1)

# Candidate URL patterns — HF has restructured their Inference API endpoints
# multiple times, so we test several known/plausible patterns in one shot
# instead of guessing one at a time.
CANDIDATES = [
    # ("api-inference (pipeline path)", f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL}"),
    # ("api-inference (models path)", f"https://api-inference.huggingface.co/models/{MODEL}"),
    # ("router.huggingface.co (models path)", f"https://router.huggingface.co/hf-inference/models/{MODEL}"),

    ("router.huggingface.co (hf-inference)", f"https://router.huggingface.co/hf-inference/models/{MODEL}/pipeline/feature-extraction"),
]

headers = {"Authorization": f"Bearer {TOKEN}"}
# headers = {"Authorization": f"Bearer {TOKEN}"}
payload = {"inputs": "This is a test sentence."}

print(f"Testing {len(CANDIDATES)} endpoint patterns for model: {MODEL}\n")

for label, url in CANDIDATES:
    print(f"--- {label} ---")
    print(f"URL: {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"Status: {resp.status_code}")
        print(f"Body (first 300 chars): {resp.text[:300]}")
        if resp.status_code == 200:
            print(">>> THIS ONE WORKS <<<")
    except requests.exceptions.ConnectionError as e:
        print(f"CONNECTION FAILED (DNS/network issue): {e}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    print()

print("Done. Use whichever URL pattern returned status 200 above.")