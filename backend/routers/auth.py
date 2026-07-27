"""Auth router — register, login, profile, and admin user management."""
from __future__ import annotations

from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.common.rate_limit import limiter

from backend.auth_system.auth_service import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    admin_set_user_role,
    authenticate_user,
    create_access_token,
    get_current_user,
    register_user,
    require_admin,
    require_user,
    update_user_profile,
)
from backend.auth_system.database import get_db
from backend.auth_system.models import User
from backend.auth_system.schemas import AdminRoleUpdate, Token, UserCreate, UserLogin, UserProfile, UserUpdate

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    user = register_user(db, user_in)
    return user


@router.post("/login", response_model=Token)
@limiter.limit("5/minute")
def login(request: Request, form: UserLogin, db: Session = Depends(get_db)):
    """Authenticate and receive a JWT access token."""
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, role=user.role, username=user.username)


@router.get("/me", response_model=UserProfile)
def read_profile(current_user: User = Depends(require_user)):
    """Return the current authenticated user's profile."""
    return current_user


@router.put("/profile", response_model=UserProfile)
def update_profile(
    updates: UserUpdate,
    current_user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Update the current user's profile fields."""
    return update_user_profile(db, current_user, updates)


@router.post("/logout")
def logout():
    """Stateless logout — client should discard the token."""
    return {"detail": "Logged out successfully. Discard your access token."}


# --- Admin endpoints ---

@router.get("/users", response_model=List[UserProfile])
def list_users(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Admin: list all users."""
    return db.query(User).offset(skip).limit(limit).all()


@router.get("/stats")
def auth_stats(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """Admin: return user counts by role."""
    total = db.query(User).count()
    students = db.query(User).filter(User.role == "student").count()
    teachers = db.query(User).filter(User.role == "teacher").count()
    admins = db.query(User).filter(User.role == "admin").count()
    return {
        "total_users": total,
        "students": students,
        "teachers": teachers,
        "admins": admins,
    }


@router.patch("/users/{user_id}/role", response_model=UserProfile)
def set_user_role(
    user_id: int,
    payload: AdminRoleUpdate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Admin-only: change a user's role. This is the ONLY way to promote a
    user to teacher/admin — role can no longer be set via /register or
    /profile (that was a self-service-admin vulnerability, now closed)."""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    return admin_set_user_role(db, target_user, payload.role)