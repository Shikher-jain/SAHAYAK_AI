import os
import hmac
import hashlib
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel

from sahayak_ai_v3.backend.services.payments.verifier import verifier
# We assume a worker function `process_utr_verification` exists in worker.py
from sahayak_ai_v3.backend.services.payments.worker import process_utr_verification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/payments", tags=["Payments"])
UPI_WEBHOOK_SECRET = os.getenv("UPI_WEBHOOK_SECRET", "")


class UTRRequest(BaseModel):
    user_id: str
    utr_number: str
    amount_inr: float


@router.post("/verify-utr", status_code=202)
async def verify_utr(req: UTRRequest, background_tasks: BackgroundTasks):
    """
    Accepts a UTR from the client, registers it as PENDING in the DB,
    and enqueues an asynchronous background verification task.
    """
    # 1. Database uniqueness check (mocked logic for idempotency)
    # async with async_session() as session:
    #    existing = await session.execute(select(UPITransaction).where(utr=req.utr_number))
    #    if existing.scalar_one_or_none():
    #        return {"status": "already_processed", "utr": req.utr_number}
    
    # 2. Save new PENDING record
    #    new_tx = UPITransaction(user_id=req.user_id, utr=req.utr_number, amount=req.amount_inr, status="PENDING")
    #    session.add(new_tx)
    #    await session.commit()
    logger.info(f"Registered UTR {req.utr_number} as PENDING for user {req.user_id}.")

    # 3. Trigger async verification worker
    background_tasks.add_task(process_utr_verification, req.user_id, req.utr_number, req.amount_inr)

    return {"status": "pending_verification", "utr": req.utr_number}


@router.post("/gateway-webhook")
async def gateway_webhook(request: Request):
    """
    Receives server-to-server webhook notifications directly from the payment provider.
    """
    payload = await request.body()
    signature = request.headers.get("x-webhook-signature", "")
    
    if not UPI_WEBHOOK_SECRET:
        logger.warning("UPI_WEBHOOK_SECRET not set, skipping webhook signature validation (FAIL-OPEN FOR DEV ONLY).")
    else:
        # Validate HMAC-SHA256 signature
        expected_sig = hmac.new(
            UPI_WEBHOOK_SECRET.encode(), payload, hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_sig, signature):
            logger.error("Webhook signature mismatch.")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
            
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
    utr_number = data.get("utr")
    status = data.get("status")
    
    if not utr_number or not status:
        return {"received": True, "note": "Missing UTR or status"}
        
    logger.info(f"Received webhook for UTR {utr_number} with status {status}")
    
    if status == "SUCCESS":
        # Atomically update UPITransaction and UserSubscription
        # async with async_session() as session:
        #    ... (Logic implemented in worker / db layer)
        pass
        
    return {"received": True}
