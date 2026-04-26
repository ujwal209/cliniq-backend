from fastapi import APIRouter, Depends, HTTPException, status
from app.core.database import get_db
from app.models.patient import PatientModel
from bson import ObjectId
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from app.core.deps import get_current_user

router = APIRouter()

class PatientResponse(PatientModel):
    id: str

class PatientUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    age: int | None = None
    chief_complaint: str | None = None
    medical_history: List[str] | None = None
    current_medications: List[str] | None = None
    allergies: List[str] | None = None
    triage_status: str | None = None
    active_issues: List[str] | None = None
    is_archived: bool | None = None

@router.get("/", response_model=List[PatientResponse])
async def get_patients(current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    query = {"is_archived": {"$ne": True}}
    if current_user.get("role") == "doctor":
        query["doctor_email"] = current_user.get("sub")
        
    patients_cursor = db["patients"].find(query)
    patients = await patients_cursor.to_list(length=100)
    for p in patients:
        p["id"] = str(p["_id"])
        del p["_id"]
    return patients

@router.post("/", response_model=PatientResponse)
async def create_patient(patient: PatientModel, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    doc = patient.model_dump()
    if current_user.get("role") == "doctor":
        doc["doctor_email"] = current_user.get("sub")
        
    result = await db["patients"].insert_one(doc)
    doc["id"] = str(result.inserted_id)
    if "_id" in doc:
        del doc["_id"]
    return doc

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str, db=Depends(get_db)):
    patient = await db["patients"].find_one({"_id": ObjectId(patient_id)})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient["id"] = str(patient["_id"])
    del patient["_id"]
    return patient

@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: str, update_data: PatientUpdate, db=Depends(get_db)):
    update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields provided for update")
        
    result = await db["patients"].update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": update_dict}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    updated_patient = await db["patients"].find_one({"_id": ObjectId(patient_id)})
    updated_patient["id"] = str(updated_patient["_id"])
    del updated_patient["_id"]
    return updated_patient
