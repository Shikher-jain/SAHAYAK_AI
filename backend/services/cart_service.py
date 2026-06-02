"""Cart service — shopping cart for e-commerce features (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "commerce"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "commerce.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_commerce_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cart_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            quantity INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS demo_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            email TEXT DEFAULT '',
            plan TEXT DEFAULT 'Pro',
            message TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_commerce_db()


# --- Cart ---

def add_to_cart(user_id: str, product_id: str, product_name: str, price: float, quantity: int = 1) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO cart_items (user_id, product_id, product_name, price, quantity) VALUES (?, ?, ?, ?, ?)",
        (user_id, product_id, product_name, price, quantity),
    )
    item_id = cur.lastrowid
    conn.commit()
    conn.close()
    return item_id


def get_cart(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM cart_items WHERE user_id = ? ORDER BY added_at DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def remove_from_cart(item_id: int, user_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM cart_items WHERE id = ? AND user_id = ?", (item_id, user_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def cart_total(user_id: str) -> float:
    items = get_cart(user_id)
    return sum(item["price"] * item["quantity"] for item in items)


def clear_cart(user_id: str) -> int:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM cart_items WHERE user_id = ?", (user_id,))
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


# --- Demo Requests ---

def request_demo(user_id: str, email: str, plan: str = "Pro", message: str = "") -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO demo_requests (user_id, email, plan, message) VALUES (?, ?, ?, ?)",
        (user_id, email, plan, message),
    )
    demo_id = cur.lastrowid
    conn.commit()
    conn.close()
    return demo_id
