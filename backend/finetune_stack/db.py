from typing import List, Tuple

import faiss
import numpy as np

from backend.common.data_paths import get_finetune_db_path
from backend.local_stack.store_base import SQLiteFaissStore

DB_PATH = get_finetune_db_path()
EMBED_DIM = 384

_store = SQLiteFaissStore(DB_PATH, embed_dim=EMBED_DIM)

def init_db() -> None:
    _store.init_db()

def add_chunk(filename: str, chunk_text: str, embedding: np.ndarray) -> None:
    _store.add_chunk(filename, chunk_text, embedding)

def get_all_chunks() -> Tuple[List[str], np.ndarray]:
    return _store.get_all_chunks()


def build_faiss_index() -> Tuple[faiss.IndexFlatL2, List[str]]:
    return _store.build_faiss_index()

