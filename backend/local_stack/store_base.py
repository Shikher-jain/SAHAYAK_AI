from __future__ import annotations

import json
import pickle
import sqlite3
import threading
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import faiss
import numpy as np


class SQLiteFaissStore:
    def __init__(self, db_path: Path, embed_dim: int = 384) -> None:
        self.db_path = db_path
        self.embed_dim = embed_dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # TASK 24: thread-local connection pool — each thread reuses its own connection.
        self._local = threading.local()

        self._cache_lock = Lock()
        self._cached_signature: tuple[int, int] | None = None
        self._cached_index: faiss.IndexFlatIP | None = None
        self._cached_texts: List[str] | None = None
        self._cached_metadata: List[Dict[str, str]] | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """TASK 24: return a per-thread reusable SQLite connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def init_db(self) -> None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pdfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                text_chunk TEXT,
                embedding BLOB,
                metadata TEXT
            )
            """
        )
        # BUG 2 FIX: ensure legacy databases include a metadata column.
        self._ensure_metadata_column(conn)
        conn.commit()

    def _ensure_metadata_column(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(pdfs)")
        columns = {row[1] for row in cur.fetchall()}
        if "metadata" not in columns:
            cur.execute("ALTER TABLE pdfs ADD COLUMN metadata TEXT")

    def add_chunk(
        self,
        filename: str,
        chunk_text: str,
        embedding: np.ndarray,
        metadata: Dict[str, str] | None = None,
    ) -> None:
        conn = self._get_conn()
        cur = conn.cursor()
        emb_blob = pickle.dumps(np.asarray(embedding, dtype="float32"))
        meta_blob = json.dumps(metadata or {}, ensure_ascii=False)
        cur.execute(
            "INSERT INTO pdfs (filename, text_chunk, embedding, metadata) VALUES (?, ?, ?, ?)",
            (filename, chunk_text, emb_blob, meta_blob),
        )
        conn.commit()

        with self._cache_lock:
            self._cached_signature = None
            self._cached_index = None
            self._cached_texts = None
            self._cached_metadata = None

    def get_all_chunks(self) -> Tuple[List[str], np.ndarray]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT text_chunk, embedding FROM pdfs ORDER BY id ASC")
        rows = cur.fetchall()

        texts: List[str] = []
        embeddings: List[np.ndarray] = []
        for text_chunk, emb_blob in rows:
            texts.append(text_chunk)
            embeddings.append(np.asarray(pickle.loads(emb_blob), dtype="float32"))

        if not embeddings:
            return texts, np.zeros((0, self.embed_dim), dtype="float32")
        return texts, np.vstack(embeddings).astype("float32")

    def get_all_records(self) -> Tuple[List[str], np.ndarray, List[Dict[str, str]]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT text_chunk, embedding, metadata FROM pdfs ORDER BY id ASC")
        rows = cur.fetchall()

        texts: List[str] = []
        embeddings: List[np.ndarray] = []
        metadatas: List[Dict[str, str]] = []
        for text_chunk, emb_blob, metadata_blob in rows:
            texts.append(text_chunk)
            embeddings.append(np.asarray(pickle.loads(emb_blob), dtype="float32"))
            try:
                metadatas.append(json.loads(metadata_blob or "{}"))
            except json.JSONDecodeError:
                metadatas.append({})

        if not embeddings:
            return texts, np.zeros((0, self.embed_dim), dtype="float32"), metadatas
        return texts, np.vstack(embeddings).astype("float32"), metadatas

    def _current_signature(self) -> tuple[int, int]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM pdfs")
        count, max_id = cur.fetchone()
        return int(count), int(max_id)

    def build_faiss_index(self) -> Tuple[faiss.IndexFlatIP, List[str]]:
        index, texts, _ = self.build_faiss_index_with_metadata()
        return index, texts

    def build_faiss_index_with_metadata(self) -> Tuple[faiss.IndexFlatIP, List[str], List[Dict[str, str]]]:
        signature = self._current_signature()
        with self._cache_lock:
            if (
                self._cached_signature == signature
                and self._cached_index is not None
                and self._cached_texts is not None
                and self._cached_metadata is not None
            ):
                return self._cached_index, self._cached_texts, self._cached_metadata

        texts, embeddings, metadatas = self.get_all_records()
        # BUG 3 FIX: use inner product index (embeddings are normalized at source).
        index = faiss.IndexFlatIP(self.embed_dim)
        if len(embeddings):
            index.add(embeddings)

        with self._cache_lock:
            self._cached_signature = signature
            self._cached_index = index
            self._cached_texts = texts
            self._cached_metadata = metadatas
        return index, texts, metadatas
