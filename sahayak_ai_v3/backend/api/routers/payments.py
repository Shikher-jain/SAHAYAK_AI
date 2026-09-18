"""Stripe billing for the v2 Premium tier.

Endpoints:
  POST /api/v2/payments/create-checkout-session
  POST /api/v2/payments/webhook
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.core.config import settings
from backend.services.payments.user_store import user_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/payments", tags=["v2 Payments"])


class CheckoutRequest(BaseModel):
    user_id: str
    price_id: Optional[str] = None
    customer_email: Optional[str] = None


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


@router.post("/create-checkout-session", response_model=CheckoutResponse)
async def create_checkout_session(req: CheckoutRequest) -> CheckoutResponse:
    """Creates a Subscription checkout session and returns the hosted URL."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured (STRIPE_SECRET_KEY is empty).",
        )
    import stripe

    stripe.api_key = settings.STRIPE_SECRET_KEY
    # Rule 4: Stripe SDK already retries transient network errors; keep it modest.
    stripe.max_network_retries = 3

    kwargs = {}
    if req.customer_email:
        kwargs["customer_email"] = req.customer_email

    session = await run_in_threadpool(
        stripe.checkout.Session.create,
        mode="subscription",
        client_reference_id=req.user_id,
        line_items=[
            {
                "price": req.price_id or settings.STRIPE_PRICE_ID,
                "quantity": 1,
            }
        ],
        success_url=settings.STRIPE_SUCCESS_URL,
        cancel_url=settings.STRIPE_CANCEL_URL,
        metadata={"user_id": req.user_id},
        **kwargs,
    )
    return CheckoutResponse(checkout_url=session.url, session_id=session.id)


@router.post("/webhook")
async def stripe_webhook(request: Request) -> dict:
    """Verifies the Stripe-Signature and flips a user to Premium on completion."""
    import stripe

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    if not settings.STRIPE_WEBHOOK_SECRET:
        # ponytail: fail-open for local dev only. Set STRIPE_WEBHOOK_SECRET in
        # production — otherwise no user is ever upgraded and billing breaks.
        logger.warning("STRIPE_WEBHOOK_SECRET not configured; skipping signature check.")
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, settings.STRIPE_WEBHOOK_SECRET
            )
        except (ValueError, stripe.error.SignatureVerificationError):
            raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    if event["type"] == "checkout.session.completed":
        obj = event["data"]["object"]
        user_id = obj.get("client_reference_id") or obj.get("metadata", {}).get("user_id")
        if user_id:
            user_store.set_tier(user_id, "premium")
            logger.info("Upgraded user %s to premium.", user_id)
    return {"received": True}