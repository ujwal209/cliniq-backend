from fastapi import APIRouter, Depends, HTTPException
from app.core.database import get_db
from app.core.deps import get_current_user
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

router = APIRouter()

class NoteUpdate(BaseModel):
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None

@router.get("/")
async def get_notes(patient_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    db = get_db()
    query = {}
    if patient_id:
        query["patient_id"] = patient_id
        
    cursor = db.clinical_notes.find(query).sort("created_at", -1)
    notes = await cursor.to_list(length=100)
    for n in notes:
        n["id"] = str(n["_id"])
        del n["_id"]
    return notes

@router.get("/{note_id}")
async def get_note(note_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    note = await db.clinical_notes.find_one({"_id": ObjectId(note_id)})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    note["id"] = str(note["_id"])
    del note["_id"]
    return note

@router.put("/{note_id}")
async def update_note(note_id: str, data: NoteUpdate, current_user: dict = Depends(get_current_user)):
    db = get_db()
    note = await db.clinical_notes.find_one({"_id": ObjectId(note_id)})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
        
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    
    if update_data:
        # update the nested soap_note object
        for k, v in update_data.items():
            note["soap_note"][k] = v
            
        await db.clinical_notes.update_one(
            {"_id": ObjectId(note_id)},
            {"$set": {"soap_note": note["soap_note"]}}
        )
        
    note["id"] = str(note["_id"])
    del note["_id"]
    return note

@router.delete("/{note_id}")
async def delete_note(note_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    result = await db.clinical_notes.delete_one({"_id": ObjectId(note_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"message": "Note deleted"}
