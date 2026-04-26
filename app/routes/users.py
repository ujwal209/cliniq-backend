from fastapi import APIRouter, Depends, HTTPException
from app.core.database import get_db
from app.core.deps import get_current_user
from pydantic import BaseModel

router = APIRouter()

class ProfileUpdate(BaseModel):
    full_name: str | None = None
    clinic_name: str | None = None
    specialty: str | None = None
    license_number: str | None = None
    avatar_url: str | None = None

@router.get("/me")
async def get_my_profile(current_user: dict = Depends(get_current_user)):
    db = get_db()
    user = await db.users.find_one({"email": current_user.get("sub")}, {"_id": 0, "password": 0, "otp_code": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/me")
async def update_my_profile(data: ProfileUpdate, current_user: dict = Depends(get_current_user)):
    db = get_db()
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided")
        
    await db.users.update_one(
        {"email": current_user.get("sub")},
        {"$set": update_data}
    )
    
    updated_user = await db.users.find_one({"email": current_user.get("sub")}, {"_id": 0, "password": 0, "otp_code": 0})
    return updated_user

@router.get("/progress")
async def get_my_progress(current_user: dict = Depends(get_current_user)):
    db = get_db()
    
    # Fallback to DB role check if token is stale (e.g., right after onboarding)
    role = current_user.get("role")
    if role != "patient":
        user_db = await db.users.find_one({"email": current_user.get("sub")})
        if user_db and user_db.get("role") == "patient":
            role = "patient"
            
    if role != "patient":
        raise HTTPException(status_code=403, detail="Only patients have progress logs")
        
    cursor = db.patient_progress.find({"patient_email": current_user.get("sub")}).sort("date", -1).limit(15)
    records = await cursor.to_list(length=15)
    return [{"log": r.get("log"), "date": r.get("date").isoformat()} for r in records]
