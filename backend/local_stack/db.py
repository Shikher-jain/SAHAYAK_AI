from typing import Dict, List, Tuple

import faiss
import numpy as np

from backend.common.data_paths import get_local_db_path
from backend.local_stack.store_base import SQLiteFaissStore

DB_PATH = get_local_db_path()
EMBED_DIM = 384

_store = SQLiteFaissStore(DB_PATH, embed_dim=EMBED_DIM)


def init_db() -> None:
    _store.init_db()


def add_chunk(
    filename: str,
    chunk_text: str,
    embedding: np.ndarray,
    metadata: Dict[str, str] | None = None,
) -> None:
    _store.add_chunk(filename, chunk_text, embedding, metadata=metadata)


def add_chunk_with_metadata(
    filename: str,
    chunk_text: str,
    embedding: np.ndarray,
    metadata: Dict[str, str] | None = None,
) -> None:
    _store.add_chunk(filename, chunk_text, embedding, metadata=metadata)


def get_all_chunks() -> Tuple[List[str], np.ndarray]:
    return _store.get_all_chunks()


def get_all_records() -> Tuple[List[str], np.ndarray, List[Dict[str, str]]]:
    return _store.get_all_records()


def build_faiss_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    return _store.build_faiss_index()


def build_faiss_index_with_metadata() -> Tuple[faiss.IndexFlatIP, List[str], List[Dict[str, str]]]:
    return _store.build_faiss_index_with_metadata()

