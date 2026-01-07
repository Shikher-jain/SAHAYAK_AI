
from sentence_transformers import SentenceTransformer

# Use a local embedding model, e.g., 'all-MiniLM-L6-v2'
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text)
