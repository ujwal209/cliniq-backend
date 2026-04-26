from fastapi import APIRouter, Depends
from app.core.database import get_db
from app.models.patient import PatientModel

router = APIRouter()

@router.post("/db-test")
async def test_db_insert(patient: PatientModel):
    db = get_db()
    # Insert into 'patients' collection
    result = await db.patients.insert_one(patient.model_dump())
    return {"inserted_id": str(result.inserted_id), "status": "Data saved in Atlas!"}