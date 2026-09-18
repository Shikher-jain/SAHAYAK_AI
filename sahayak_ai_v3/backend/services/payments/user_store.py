"""Tier + chat-history + UPI order persistence for the v2 monetization features.

Stdlib sqlite3 only: zero new deps, survives restarts via a file on disk.
Suitable for Render's single web service.
"""
import os
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional

from backend.core.config import settings
from backend.services.payments.upi import UTR_RE

_DB_PATH = Path(settings.SAHAYAK_DB_PATH).absolute()


class UserStore:
    # ponytail: single-file sqlite. If you outgrow one Render instance, swap
    # this for Neon/Postgres (SQLAlchemy is already a repo dep) and add a
    # unique constraint on orders.utr there instead of the app-side check.
    def __init__(self, path: Path = _DB_PATH):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                " user_id TEXT PRIMARY KEY, tier TEXT NOT NULL DEFAULT 'free')"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " user_id TEXT, role TEXT, content TEXT, ts INTEGER)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id, ts)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS orders ("
                " order_id TEXT PRIMARY KEY,"
                " user_id TEXT NOT NULL,"
                " amount TEXT NOT NULL,"
                " status TEXT NOT NULL DEFAULT 'pending',"
                " utr TEXT UNIQUE,"
                " created_at INTEGER NOT NULL,"
                " updated_at INTEGER NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get_tier(self, user_id: str) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT tier FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        return row[0] if row else "free"

    def set_tier(self, user_id: str, tier: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO users(user_id, tier) VALUES(?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET tier=excluded.tier",
                (user_id, tier),
            )

    def is_premium(self, user_id: str) -> bool:
        return self.get_tier(user_id) == "premium"

    def assign_premium_on_payment(self, user_id: str) -> None:
        """Conditionally promotes to premium (already granted elsewhere)."""
        self.set_tier(user_id, "premium")

    def get_history(self, user_id: str, limit: int = 20) -> List[Dict[str, str]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content FROM messages WHERE user_id=? "
                "ORDER BY ts DESC, rowid DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"role": r, "content": m} for r, m in reversed(rows)]

    def append_message(self, user_id: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages(user_id, role, content, ts) VALUES(?,?,?,?)",
                (user_id, role, content, int(time.time())),
            )

    # ---- UPI orders ----

    def create_order(self, user_id: str, amount: str) -> str:
        order_id = f"ord_{int(time.time())}_{os.urandom(3).hex()}"
        now = int(time.time())
        with self._conn() as c:
            self._ensure_user_row(c, user_id)
            c.execute(
                "INSERT INTO orders(order_id, user_id, amount, status, created_at, updated_at) "
                "VALUES(?,?,?, 'pending', ?, ?)",
                (order_id, user_id, amount, now, now),
            )
        return order_id

    def get_order(self, order_id: str) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT order_id, user_id, amount, status, utr, created_at "
                "FROM orders WHERE order_id=?",
                (order_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "order_id": row[0],
            "user_id": row[1],
            "amount": row[2],
            "status": row[3],
            "utr": row[4],
            "created_at": row[5],
        }

    def reachable_orders(self, user_id: str) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT order_id, amount, status, utr, created_at "
                "FROM orders WHERE user_id=? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "order_id": r[0],
                "amount": r[1],
                "status": r[2],
                "utr": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def promote_with_utr(self, order_id: str, utr: str) -> bool:
        """Atomic pay-capture by manual UTR. Returns False if UTR already used or order lost."""
        with self._conn() as c:
            try:
                c.execute("BEGIN IMMEDIATE")
                existing = c.execute(
                    "SELECT order_id FROM orders WHERE utr=?", (utr,)
                ).fetchone()
                if existing:
                    c.execute("ROLLBACK")
                    return False
                cur = c.execute(
                    "UPDATE orders SET utr=?, status='paid', updated_at=? "
                    "WHERE order_id=? AND status='pending'",
                    (utr, int(time.time()), order_id),
                )
                if cur.rowcount != 1:
                    c.execute("ROLLBACK")
                    return False
                user_row = c.execute(
                    "SELECT user_id FROM orders WHERE order_id=?", (order_id,)
                ).fetchone()
                c.execute(
                    "INSERT OR REPLACE INTO users(user_id, tier) VALUES(?, 'premium')",
                    (user_row[0],),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                return False
        return True

    def _ensure_user_row(self, c, user_id: str) -> None:
        c.execute(
            "INSERT OR IGNORE INTO users(user_id, tier) VALUES(?, 'free')",
            (user_id,),
        )


user_store = UserStore()
