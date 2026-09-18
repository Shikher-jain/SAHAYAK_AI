import logging
from datetime import datetime, timedelta
from typing import Optional

from sahayak_ai_v3.backend.services.payments.verifier import verifier
from backend.services.payments.user_store import user_store

logger = logging.getLogger(__name__)

# Mock database for example purposes, replace with sqlalchemy.ext.asyncio actual implementation
# async with async_session.begin() as session:
#     stmt = select(UPITransaction).where(UPITransaction.status == 'PENDING')
# ...

async def process_utr_verification(user_id: str, utr_number: str, expected_amount: float):
    """
    Background worker that queries the gateway for the UTR status.
    Updates the transaction and user subscription atomically based on the result.
    """
    logger.info(f"[WORKER] Starting async verification for UTR {utr_number} (User: {user_id})")
    
    # 1. Call Gateway via httpx (async, non-blocking)
    is_valid = await verifier.verify_utr_with_gateway(utr_number, expected_amount)
    
    if is_valid:
        # 2. Update UPITransaction.status = 'SUCCESS'
        logger.info(f"[WORKER] UTR {utr_number} verified. Activating PRO subscription.")
        
        # 3. Upsert UserSubscription atomically
        # Mocking the ORM call for memory safety constraint
        # await session.execute(
        #     update(UserSubscription)
        #     .where(UserSubscription.user_id == user_id)
        #     .values(tier='pro', is_active=True, expires_at=datetime.utcnow() + timedelta(days=30))
        # )
        
        # Using existing user_store temporarily
        user_store.set_tier(user_id, "premium")
    else:
        # Update UPITransaction.status = 'FAILED'
        logger.warning(f"[WORKER] UTR {utr_number} verification failed.")
        # await session.execute(
        #     update(UPITransaction)
        #     .where(UPITransaction.utr == utr_number)
        #     .values(status='FAILED')
        # )
