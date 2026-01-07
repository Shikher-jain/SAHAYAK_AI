from sentence_transformers import SentenceTransformer
import numpy as np

# Lazy loading - model loaded ONLY on first use
_model = None

def get_model():
    """Get or initialize the embedding model (lazy loading)"""
    global _model
    if _model is None:
        print("🔄 Loading embedding model (first time only)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully")
    return _model

def embed_text(text):
    """Generate embeddings for text using sentence-transformers"""
    model = get_model()
    return model.encode(text)
