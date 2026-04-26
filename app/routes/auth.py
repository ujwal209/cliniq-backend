from fastapi import APIRouter, HTTPException, status, Depends
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token
from app.utils.email import generate_otp, send_otp_email
from app.models.auth import (
    SignupRequest, OTPVerifyRequest, ResendOTPRequest, LoginRequest, 
    ForgotPasswordRequest, ResetPasswordRequest, OnboardingRequest
)
from app.core.deps import get_current_user  # <-- THE BOUNCER
from datetime import datetime

router = APIRouter()

@router.post("/signup")
async def signup(user_data: SignupRequest):
    db = get_db()
    
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    otp = await generate_otp()
    new_user = {
        "full_name": user_data.full_name,
        "email": user_data.email,
        "password": get_password_hash(user_data.password),
        "is_verified": False,
        "otp_code": otp,
        "role": "unassigned", 
        "created_at": datetime.utcnow()
    }
    await db.users.insert_one(new_user)
    await send_otp_email(user_data.email, otp)
    
    return {"message": "Signup successful. Please check your email for the OTP."}


@router.post("/verify-otp")
async def verify_otp(data: OTPVerifyRequest):
    db = get_db()
    user = await db.users.find_one({"email": data.email})
    
    if not user or user.get("otp_code") != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    await db.users.update_one(
        {"email": data.email},
        {"$set": {"is_verified": True, "otp_code": None}}
    )
    
    token = create_access_token({"sub": data.email})
    return {"message": "Email verified successfully!", "access_token": token}


@router.post("/resend-otp")
async def resend_otp(data: ResendOTPRequest):
    db = get_db()
    user = await db.users.find_one({"email": data.email})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user.get("is_verified"):
        raise HTTPException(status_code=400, detail="User is already verified")
        
    otp = await generate_otp()
    await db.users.update_one({"email": data.email}, {"$set": {"otp_code": otp}})
    await send_otp_email(data.email, otp)
    
    return {"message": "OTP resent successfully!"}


@router.post("/login")
async def login(data: LoginRequest):
    db = get_db()
    user = await db.users.find_one({"email": data.email})
    
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    if not user.get("is_verified"):
        raise HTTPException(status_code=403, detail="Please verify your email first")

    token = create_access_token(
        {"sub": data.email, "role": user.get("role")}
    )
    return {
        "access_token": token, 
        "token_type": "bearer",
        "role": user.get("role"),
        "needs_onboarding": user.get("role") == "unassigned"
    }


@router.post("/forgot-password")
async def forgot_password(data: ForgotPasswordRequest):
    db = get_db()
    user = await db.users.find_one({"email": data.email})
    if not user:
        return {"message": "If that email exists, an OTP has been sent."}
        
    otp = await generate_otp()
    await db.users.update_one({"email": data.email}, {"$set": {"otp_code": otp}})
    await send_otp_email(data.email, otp)
    
    return {"message": "If that email exists, an OTP has been sent."}


@router.post("/reset-password")
async def reset_password(data: ResetPasswordRequest):
    db = get_db()
    user = await db.users.find_one({"email": data.email})
    
    if not user or user.get("otp_code") != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    new_hash = get_password_hash(data.new_password)
    await db.users.update_one(
        {"email": data.email},
        {"$set": {"password": new_hash, "otp_code": None}}
    )
    return {"message": "Password reset successful. You can now log in."}


@router.post("/onboarding")
async def onboarding(
    data: OnboardingRequest, 
    current_user: dict = Depends(get_current_user) # <-- REQUIRES VALID JWT
):
    db = get_db()
    
    # 🔒 SECURE: Get email directly from the verified token, NOT the request body!
    secure_email = current_user.get("sub")
    
    profile_data = {"role": data.role}
    if data.avatar_url:
        profile_data["avatar_url"] = data.avatar_url
    
    if data.role == "doctor":
        profile_data.update({
            "specialty": data.specialty,
            "license_number": data.license_number,
            "clinic_name": data.clinic_name
        })
    elif data.role == "patient":
        profile_data.update({
            "age": data.age,
            "gender": data.gender,
            "pre_existing_conditions": data.pre_existing_conditions or [],
            "allergies": data.allergies or [],
            "emergency_contact": data.emergency_contact,
            "health_goals": data.health_goals or ""
        })
    else:
        raise HTTPException(status_code=400, detail="Invalid role selected")

    result = await db.users.update_one(
        {"email": secure_email},
        {"$set": profile_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {"message": f"Successfully onboarded as {data.role}!"}