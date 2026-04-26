from fastapi import APIRouter, Depends, HTTPException
from app.core.database import get_db
from app.core.deps import get_current_user
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter()

class AppointmentCreate(BaseModel):
    patient_id: str
    patient_name: str
    title: str
    start_time: datetime
    end_time: datetime
    type: str # e.g. "Checkup", "Follow-up"
    status: str = "Scheduled"

class AppointmentUpdate(BaseModel):
    title: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    type: Optional[str] = None
    status: Optional[str] = None

@router.get("/")
async def get_appointments(current_user: dict = Depends(get_current_user)):
    db = get_db()
    # In a real app, filter by doctor ID. We assume single-doctor context for now.
    cursor = db.appointments.find().sort("start_time", 1)
    appointments = await cursor.to_list(length=100)
    for a in appointments:
        a["id"] = str(a["_id"])
        del a["_id"]
    return appointments

@router.post("/")
async def create_appointment(data: AppointmentCreate, current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = data.model_dump()
    doc["created_at"] = datetime.utcnow()
    result = await db.appointments.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    if "_id" in doc:
        del doc["_id"]
    return doc

@router.put("/{app_id}")
async def update_appointment(app_id: str, data: AppointmentUpdate, current_user: dict = Depends(get_current_user)):
    db = get_db()
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields")
        
    result = await db.appointments.update_one(
        {"_id": ObjectId(app_id)},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
        
    app_doc = await db.appointments.find_one({"_id": ObjectId(app_id)})
    app_doc["id"] = str(app_doc["_id"])
    del app_doc["_id"]
    return app_doc

@router.delete("/{app_id}")
async def delete_appointment(app_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    result = await db.appointments.delete_one({"_id": ObjectId(app_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"message": "Deleted"}
