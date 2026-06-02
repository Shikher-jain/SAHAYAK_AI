"""Core authentication service — JWT tokens, password hashing, user CRUD."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import hashlib

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.auth_system.database import get_db
from backend.auth_system.models import User
from backend.auth_system.schemas import TokenData, UserCreate, UserUpdate

# --- Configuration ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "sahayak-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24h default

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# --- Password helpers (direct bcrypt with SHA256 pre-hash) ---

def _sha256_digest(password: str) -> bytes:
    """SHA256-hash the password to produce a fixed 32-byte input for bcrypt."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("utf-8")


def hash_password(password: str) -> str:
    """Hash password using bcrypt with SHA256 pre-hashing."""
    digest = _sha256_digest(password)
    return bcrypt.hashpw(digest, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against a stored hash."""
    digest = _sha256_digest(plain)
    try:
        return bcrypt.checkpw(digest, hashed.encode("utf-8"))
    except Exception:
        return False


# --- JWT helpers ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = dict(data)
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode a JWT token. Returns None if token is missing/invalid."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "student")
        if username is None:
            return None
        return TokenData(username=username, role=role)
    except JWTError:
        return None


# --- User CRUD ---

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def register_user(db: Session, user_in: UserCreate) -> User:
    """Register a new user. Raises HTTPException on duplicate username/email."""
    if get_user_by_username(db, user_in.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if get_user_by_email(db, user_in.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        role=user_in.role,
        full_name=user_in.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Return user if credentials are valid, else None."""
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


def update_user_profile(db: Session, user: User, updates: UserUpdate) -> User:
    """Update mutable profile fields."""
    if updates.email is not None:
        existing = get_user_by_email(db, updates.email)
        if existing and existing.id != user.id:
            raise HTTPException(status_code=409, detail="Email already in use")
        user.email = updates.email
    if updates.full_name is not None:
        user.full_name = updates.full_name
    if updates.role is not None and updates.role in {"student", "teacher", "admin"}:
        user.role = updates.role
    db.commit()
    db.refresh(user)
    return user


# --- FastAPI dependencies ---

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Return the current user from the JWT token, or None if no token."""
    if token is None:
        return None
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        return None
    user = get_user_by_username(db, token_data.username)
    return user


async def require_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Require a valid authenticated user — raises 401 if missing/invalid."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_username(db, token_data.username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")
    return user


async def require_admin(user: User = Depends(require_user)) -> User:
    """Require the authenticated user to have admin role."""
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user
