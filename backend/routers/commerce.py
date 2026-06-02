"""Commerce router — cart, pricing, checkout, demo requests."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.auth_system.auth_service import require_user
from backend.auth_system.models import User
from backend.services import cart_service

router = APIRouter(prefix="/commerce", tags=["commerce"])


class CartAddRequest(BaseModel):
    product_id: str
    product_name: str
    price: float
    quantity: int = 1


class DemoRequest(BaseModel):
    email: str
    plan: str = "Pro"
    message: str = ""


@router.post("/cart/add")
def add_to_cart(req: CartAddRequest, user: User = Depends(require_user)):
    """Add an item to the shopping cart."""
    item_id = cart_service.add_to_cart(str(user.id), req.product_id, req.product_name, req.price, req.quantity)
    return {"id": item_id, "status": "ok"}


@router.get("/cart")
def get_cart(user: User = Depends(require_user)):
    """Get the current user's shopping cart."""
    items = cart_service.get_cart(str(user.id))
    total = cart_service.cart_total(str(user.id))
    return {"items": items, "total": total, "currency": "INR"}


@router.delete("/cart/{item_id}")
def remove_cart_item(item_id: int, user: User = Depends(require_user)):
    """Remove an item from the cart."""
    removed = cart_service.remove_from_cart(item_id, str(user.id))
    return {"removed": removed}


@router.post("/cart/checkout")
def checkout(user: User = Depends(require_user)):
    """Process checkout (placeholder — integrates with payment gateway)."""
    total = cart_service.cart_total(str(user.id))
    items = cart_service.get_cart(str(user.id))
    count = cart_service.clear_cart(str(user.id))
    return {
        "status": "success",
        "total": total,
        "items_purchased": count,
        "message": f"Checkout successful! {count} items purchased for INR {total:.2f}",
    }


@router.post("/demo/request")
def request_demo(req: DemoRequest, user: User = Depends(require_user)):
    """Request a free demo of the Pro/Enterprise plan."""
    demo_id = cart_service.request_demo(str(user.id), req.email, req.plan, req.message)
    return {"demo_id": demo_id, "status": "received", "message": "We'll contact you within 24 hours."}
