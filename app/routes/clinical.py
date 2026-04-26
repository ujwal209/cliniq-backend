from fastapi import APIRouter, Depends, HTTPException, status
from app.core.database import get_db
from app.models.clinical import (
    ClinicalNoteCreate, ClinicalNoteResponse, SOAPNote,
    TreatmentPlanCreate, TreatmentPlanResponse,
    OCRLabResultRequest, LabResultResponse,
    DiagnosticImageRequest, DiagnosticImageResponse,
    PreVisitChatRequest, MergePatientsRequest
)
from app.services import clinical_service
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel
from typing import List

router = APIRouter()

# CREATE: Clinical Notes (Ambient Scribe)
@router.post("/clinical-notes", response_model=ClinicalNoteResponse)
async def create_clinical_note(request: ClinicalNoteCreate, db=Depends(get_db)):
    ai_result = await clinical_service.generate_clinical_note(request.consultation_transcript)
    
    note_doc = {
        "patient_id": request.patient_id,
        "soap_note": ai_result.get("soap_note", {}),
        "billing_codes": ai_result.get("billing_codes", []),
        "created_at": datetime.utcnow()
    }
    result = await db["clinical_notes"].insert_one(note_doc)
    
    return ClinicalNoteResponse(
        id=str(result.inserted_id),
        patient_id=request.patient_id,
        soap_note=SOAPNote(**ai_result.get("soap_note", {})),
        billing_codes=ai_result.get("billing_codes", []),
        created_at=note_doc["created_at"]
    )

# CREATE: Patient Summaries
@router.get("/patients/{patient_id}/summary")
async def get_patient_summary(patient_id: str, db=Depends(get_db)):
    patient = await db["patients"].find_one({"_id": ObjectId(patient_id)})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    history = patient.get("medical_history", [])
    summary_points = await clinical_service.generate_patient_summary(history)
    return {"patient_id": patient_id, "active_issues_summary": summary_points}

# CREATE: Treatment Plans
@router.post("/treatment-plans", response_model=TreatmentPlanResponse)
async def create_treatment_plan(request: TreatmentPlanCreate, db=Depends(get_db)):
    ai_result = await clinical_service.generate_treatment_plan(request.diagnosis)
    
    plan_doc = {
        "patient_id": request.patient_id,
        "diagnosis": request.diagnosis,
        "diet": ai_result.get("diet", ""),
        "exercise": ai_result.get("exercise", ""),
        "follow_up_schedule": ai_result.get("follow_up_schedule", ""),
        "created_at": datetime.utcnow()
    }
    result = await db["treatment_plans"].insert_one(plan_doc)
    plan_doc["id"] = str(result.inserted_id)
    return TreatmentPlanResponse(**plan_doc)

# READ: Vector Search (Semantic Search & RAG)
class SearchRequest(BaseModel):
    query: str
    patient_id: str

@router.post("/search")
async def search_clinical_records(request: SearchRequest, db=Depends(get_db)):
    # Fetch all notes and history for the patient to simulate RAG retrieval
    notes = await db["clinical_notes"].find({"patient_id": request.patient_id}).to_list(length=100)
    documents = [str(n.get("soap_note", "")) for n in notes]
    
    patient = await db["patients"].find_one({"_id": ObjectId(request.patient_id)})
    if patient:
        documents.extend(patient.get("medical_history", []))
        
    relevant_sentences = await clinical_service.vector_search(request.query, documents)
    return {"results": relevant_sentences}

# READ: Cross-Reference (Drug Interactions)
class PrescriptionRequest(BaseModel):
    patient_id: str
    new_medications: List[str]

@router.post("/prescriptions/check")
async def check_prescriptions(request: PrescriptionRequest, db=Depends(get_db)):
    patient = await db["patients"].find_one({"_id": ObjectId(request.patient_id)})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    current_meds = patient.get("active_medications", [])
    all_meds = current_meds + request.new_medications
    warning = await clinical_service.check_drug_interactions(all_meds)
    return {"warning": warning}

# READ: Diagnostic Imaging
@router.post("/imaging/analyze", response_model=DiagnosticImageResponse)
async def analyze_imaging(request: DiagnosticImageRequest):
    ai_result = await clinical_service.analyze_diagnostic_image(request.image_base64)
    return DiagnosticImageResponse(**ai_result)

# UPDATE: Data Enrichment via OCR
@router.post("/lab-results/ocr", response_model=LabResultResponse)
async def process_lab_report(request: OCRLabResultRequest, db=Depends(get_db)):
    ai_result = await clinical_service.process_ocr(request.image_base64)
    
    doc = {
        "patient_id": request.patient_id,
        "extracted_text": ai_result.get("extracted_text", ""),
        "structured_data": ai_result.get("structured_data", {}),
        "created_at": datetime.utcnow()
    }
    result = await db["lab_results"].insert_one(doc)
    doc["id"] = str(result.inserted_id)
    
    # Update Risk Status (Predictive Triage) as a background/inline step
    # E.g., if structured_data has abnormal values, update triage
    if "Urgent" in str(doc["structured_data"]): # Simplified logic
        await db["patients"].update_one(
            {"_id": ObjectId(request.patient_id)},
            {"$set": {"triage_status": "Urgent"}}
        )
        
    return LabResultResponse(**doc)

# UPDATE: Pre-Visit State
@router.post("/patients/pre-visit")
async def update_pre_visit(request: PreVisitChatRequest, db=Depends(get_db)):
    structured_terms = await clinical_service.extract_structured_terms(request.raw_text)
    
    await db["patients"].update_one(
        {"_id": ObjectId(request.patient_id)},
        {"$addToSet": {"active_issues": {"$each": structured_terms}}}
    )
    return {"message": "Patient profile updated", "structured_terms": structured_terms}

# DELETE: Merge Duplicates
@router.post("/patients/merge")
async def merge_patients(request: MergePatientsRequest, db=Depends(get_db)):
    primary = await db["patients"].find_one({"_id": ObjectId(request.primary_patient_id)})
    duplicate = await db["patients"].find_one({"_id": ObjectId(request.duplicate_patient_id)})
    
    if not primary or not duplicate:
        raise HTTPException(status_code=404, detail="One or both patients not found")
        
    # Merge logic: append history, delete duplicate
    merged_history = list(set(primary.get("medical_history", []) + duplicate.get("medical_history", [])))
    
    await db["patients"].update_one(
        {"_id": ObjectId(request.primary_patient_id)},
        {"$set": {"medical_history": merged_history}}
    )
    await db["patients"].delete_one({"_id": ObjectId(request.duplicate_patient_id)})
    return {"message": "Patients merged successfully"}

# DELETE: PII Redaction
class RedactRequest(BaseModel):
    text: str

@router.post("/redact")
async def redact_record(request: RedactRequest):
    redacted_text = await clinical_service.redact_pii(request.text)
    return {"redacted_text": redacted_text}

# DELETE: Archiving Old Context
@router.put("/patients/{patient_id}/archive")
async def archive_patient_context(patient_id: str, db=Depends(get_db)):
    await db["patients"].update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": {"is_archived": True}}
    )
    return {"message": "Patient context archived successfully"}
