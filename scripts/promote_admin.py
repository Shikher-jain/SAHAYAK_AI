"""One-time bootstrap: promote an existing user to admin.

Run this manually (not part of the live API — no HTTP endpoint does this,
by design, to close the self-service role-escalation hole). Use it once to
create your first admin account, then use PATCH /auth/users/{id}/role
(admin-only) for any further role changes.

Usage:
    1. Register a normal account first via POST /auth/register (or the
       Streamlit signup page) — it will be created as "student".
    2. Run: python -m scripts.promote_admin <username>
"""
from __future__ import annotations

import sys

from backend.auth_system.database import SessionLocal
from backend.auth_system.models import User


def promote_to_admin(username: str) -> None:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"No user found with username '{username}'. Register the account first.")
            return
        if user.role == "admin":
            print(f"'{username}' is already an admin.")
            return
        user.role = "admin"
        db.commit()
        print(f"'{username}' promoted to admin.")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.promote_admin <username>")
        sys.exit(1)
    promote_to_admin(sys.argv[1])