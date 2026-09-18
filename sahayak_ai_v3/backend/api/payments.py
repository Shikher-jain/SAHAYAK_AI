from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import uuid
import qrcode
import io
import base64

from backend.auth_system.database import get_db
from backend.auth_system.models import User, Order
from backend.auth_system.auth_service import get_current_user

router = APIRouter(prefix="/api/v3/payments", tags=["payments"])

def verify_user_tier(required_tier: str = "pro"):
    def tier_checker(current_user: User = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if required_tier == "pro" and current_user.tier != "pro":
            raise HTTPException(status_code=403, detail=f"Upgrade to {required_tier} required")
        return current_user
    return tier_checker

class CreateOrderRequest(BaseModel):
    amount: int

class VerifyUTRRequest(BaseModel):
    order_id: str
    utr_number: str

@router.post("/create-order")
async def create_order(request: CreateOrderRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    order_id = str(uuid.uuid4())
    vpa = "sahayakai@ybl" # Placeholder VPA
    amount_str = f"{request.amount}.00"
    
    # Generate UPI URI
    upi_uri = f"upi://pay?pa={vpa}&pn=SahayakAI&am={amount_str}&cu=INR&tn=Order_{order_id}"
    
    # Generate QR code PNG in-memory
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(upi_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Save Order in PostgreSQL (or SQLite via database.py)
    new_order = Order(
        id=order_id,
        user_id=current_user.id,
        amount=request.amount,
        status="PENDING"
    )
    db.add(new_order)
    db.commit()
    db.refresh(new_order)
    
    return {
        "order_id": order_id,
        "upi_uri": upi_uri,
        "qr_code_base64": qr_base64
    }

@router.post("/verify-utr")
async def verify_utr(request: VerifyUTRRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    order = db.query(Order).filter(Order.id == request.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
        
    if order.status == "SUCCESS":
        raise HTTPException(status_code=400, detail="Order already processed")
        
    # Check for duplicate UTR
    existing_utr = db.query(Order).filter(Order.utr_number == request.utr_number).first()
    if existing_utr:
        raise HTTPException(status_code=400, detail="Duplicate UTR number")
        
    if len(request.utr_number) != 12:
        raise HTTPException(status_code=400, detail="Invalid UTR number length (must be 12 digits)")
        
    order.utr_number = request.utr_number
    order.status = "SUCCESS"
    
    # Upgrade user tier to pro
    current_user.tier = "pro"
    
    db.commit()
    db.refresh(order)
    db.refresh(current_user)
    
    return {
        "status": "success",
        "message": "Payment verified and tier upgraded to pro",
        "order_id": order.id
    }
