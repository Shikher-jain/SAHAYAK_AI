import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

UPI_GATEWAY_API_URL = os.getenv("UPI_GATEWAY_API_URL", "https://api.paymentgateway.com/v1")
UPI_GATEWAY_API_KEY = os.getenv("UPI_GATEWAY_API_KEY", "")

class PaymentVerifier:
    """Asynchronous payment verifier using httpx to prevent blocking the event loop."""
    
    def __init__(self):
        self.api_url = UPI_GATEWAY_API_URL
        self.api_key = UPI_GATEWAY_API_KEY

    async def verify_utr_with_gateway(self, utr_number: str, expected_amount: float) -> bool:
        """
        Queries the UPI payment gateway asynchronously to verify the UTR.
        Validates transaction status, UTR match, and minimum payment amount.
        """
        if not self.api_key:
            logger.warning("UPI_GATEWAY_API_KEY not set. Mocking verification for dev.")
            return True  # Fail-open for local dev if missing keys, or return False in prod
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Example API call to a mock aggregator (e.g., Razorpay/Cashfree)
                response = await client.get(
                    f"{self.api_url}/payments/utr/{utr_number}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Gateway returned {response.status_code} for UTR {utr_number}")
                    return False
                    
                data = response.json()
                
                status = data.get("status")
                actual_utr = data.get("utr")
                amount = float(data.get("amount", 0.0))
                
                if status == "SUCCESS" and actual_utr == utr_number and amount >= expected_amount:
                    logger.info(f"Successfully verified UTR {utr_number} for {amount} INR.")
                    return True
                    
                logger.warning(f"Validation failed for UTR {utr_number}. Status: {status}, Amount: {amount}")
                return False
                
            except httpx.RequestError as e:
                logger.error(f"Network error during UTR verification for {utr_number}: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error verifying UTR {utr_number}: {e}")
                return False

verifier = PaymentVerifier()
