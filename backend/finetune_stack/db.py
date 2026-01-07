import sqlite3
import pickle
import faiss
import numpy as np
import os

DB_PATH = "pdf_memory.db"
EMBED_DIM = 384  # for 'all-MiniLM-L6-v2'

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            text_chunk TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

# Add chunk with embedding
def add_chunk(filename, chunk_text, embedding):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    emb_blob = pickle.dumps(embedding)
    cur.execute("INSERT INTO pdfs (filename, text_chunk, embedding) VALUES (?, ?, ?)",
                (filename, chunk_text, emb_blob))
    conn.commit()
    conn.close()
# Get all embeddings and texts
def get_all_chunks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT text_chunk, embedding FROM pdfs")
    rows = cur.fetchall()
    texts, embeddings = [], []
    for t, e in rows:
        texts.append(t)
        embeddings.append(pickle.loads(e))
    conn.close()
    return texts, np.array(embeddings, dtype='float32')

# Build FAISS index from DB
def build_faiss_index():
    texts, embeddings = get_all_chunks()
    if len(embeddings) == 0:
        index = faiss.IndexFlatL2(EMBED_DIM)
    else:
        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(embeddings)
    return index, texts

