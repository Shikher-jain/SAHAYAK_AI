import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import sqlite3

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sahayak_09_02"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "pdf_memory.db"
EMBED_DIM = 384  # for 'all-MiniLM-L6-v2'


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            text_chunk TEXT,
            embedding BLOB
        )
        """
    )
    conn.commit()
    conn.close()


def add_chunk(filename: str, chunk_text: str, embedding: np.ndarray) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    emb_blob = pickle.dumps(embedding)
    cur.execute(
        "INSERT INTO pdfs (filename, text_chunk, embedding) VALUES (?, ?, ?)",
        (filename, chunk_text, emb_blob),
    )
    conn.commit()
    conn.close()


def get_all_chunks() -> Tuple[List[str], np.ndarray]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT text_chunk, embedding FROM pdfs")
    rows = cur.fetchall()
    texts: List[str] = []
    embeddings: List[np.ndarray] = []
    for text_chunk, emb in rows:
        texts.append(text_chunk)
        embeddings.append(pickle.loads(emb))
    conn.close()
    if embeddings:
        return texts, np.array(embeddings, dtype="float32")
    return texts, np.zeros((0, EMBED_DIM), dtype="float32")


def build_faiss_index() -> Tuple[faiss.IndexFlatL2, List[str]]:
    texts, embeddings = get_all_chunks()
    index = faiss.IndexFlatL2(EMBED_DIM)
    if len(embeddings):
        index.add(embeddings)
    return index, texts

