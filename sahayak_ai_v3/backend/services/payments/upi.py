"""UPI payment primitives (zero new deps, zero local compute).

- UPI deep-link URI construction (native UPI apps read this directly)
- Public QR render URL (frontend shows it; qrcode lib NOT needed)
- Exact-amount + UTR (12-digit) validation
- HMAC-SHA256 webhook signature verify (Razorpay-style X-Razorpay-Signature)
"""
import hashlib
import hmac
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional
from urllib.parse import quote

# ponytail: regex-backed; a national-Treasury-grade check needs a live PSP
# reconcile job — add one when UTR disputes actually occur.
UTR_RE = re.compile(r"^\d{12}$")
AMOUNT_DECIMALS = Decimal("0.01")


def parse_amount(value: str) -> Optional[Decimal]:
    """Parses '499' / '499.00' / '₹ 499.50' into a Decimal (2dp), else None."""
    if value is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(value).strip())
    if not cleaned:
        return None
    try:
        return Decimal(cleaned).quantize(AMOUNT_DECIMALS, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def build_upi_uri(vpa: str, payee_name: str, amount: Decimal, order_id: str) -> str:
    """Builds a UPI deep-link URI the user can tap to pay (UPI 2.0 spec)."""
    q = {
        "pa": vpa.strip(),
        "pn": payee_name.strip(),
        "am": f"{amount:.2f}",
        "cu": "INR",
        "tn": f"Upgrade {order_id}",
        "tid": order_id,
    }
    # ponytail: qrps= up to custom frontends; omitted (dynamic QR needs a UPI
    # app that honours it). Use the /qr-image URL and native apps for real QR.
    return "upi://pay?" + "&".join(f"{k}={quote(v)}" for k, v in q.items())


def build_qr_image_url(upi_uri: str, size: int = 480) -> str:
    """Public QR-render URL. Keeps QR generation off Render (no qrcode/Pillow)."""
    return (
        "https://api.qrserver.com/v1/create-qr-code/"
        + (
            f"?size={size}x{size}"
            f"&data={quote(upi_uri)}"
            "&margin=8&qzone=1&format=png&color=000000&bgcolor=ffffff"
        )
    )


def validate_utr(utr: str) -> bool:
    """A valid UTR is exactly 12 digits (NPCI reference-number format)."""
    return bool(UTR_RE.match((utr or "").strip()))


def verify_webhook_signature(
    body: bytes, signature: str, secret: str, header: str = "x-razorpay-signature"
) -> bool:
    """Constant-time HMAC-SHA256 check of the raw body (Razorpay contract)."""
    if not secret or not signature:
        return False
    header = (header or "").lower().strip().lstrip("x-").replace("-", "_")
    key = "X-Razorpay-Signature" if header == "razorpay_signature" else header
    name = "x-razorpay-signature" if key == "X-Razorpay-Signature" else key
    expected = hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    provided = signature.strip()
    return hmac.compare_digest(expected, provided)


def is_exact_amount(paid: Optional[Decimal], expected: Optional[Decimal]) -> bool:
    if paid is None or expected is None:
        return False
    return paid == expected
