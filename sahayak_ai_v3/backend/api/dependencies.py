from fastapi import Header, HTTPException, Depends
from backend.core.config import settings
from backend.services.payments.user_store import user_store

def get_pipeline_version() -> str:
    """
    Dependency to determine which pipeline version to route to.
    This enables Strategy/Factory pattern in the routers.
    """
    return settings.SAHAYAK_PIPELINE_VERSION

def verify_premium_tier(
    user_id: str = Header(..., alias="X-User-Id"),
) -> str:
    """
    Paywall gate for premium agents (Counseling / Recommender).
    Requires an authenticated X-User-Id header; panics 403 for free users.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id header.")
    if not user_store.is_premium(user_id):
        raise HTTPException(status_code=403, detail="Premium subscription required.")
    return user_id

# Example Strategy Interface
class RetrievalStrategy:
    async def retrieve_and_generate(self, query: str):
        pass

class AdvancedRetrievalStrategy(RetrievalStrategy):
    pass

class LegacyRetrievalStrategy(RetrievalStrategy):
    pass

def get_retrieval_strategy(version: str = Depends(get_pipeline_version)) -> RetrievalStrategy:
    if version == "v2_advanced":
        return AdvancedRetrievalStrategy()
    else:
        return LegacyRetrievalStrategy()
