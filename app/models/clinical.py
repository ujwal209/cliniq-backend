from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class SOAPNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str

class ClinicalNoteCreate(BaseModel):
    patient_id: str
    consultation_transcript: str

class ClinicalNoteResponse(BaseModel):
    id: str
    patient_id: str
    soap_note: SOAPNote
    billing_codes: List[str]
    created_at: datetime

class PatientSummaryResponse(BaseModel):
    patient_id: str
    active_issues_summary: List[str]

class TreatmentPlanCreate(BaseModel):
    patient_id: str
    diagnosis: str

class TreatmentPlanResponse(BaseModel):
    id: str
    patient_id: str
    diagnosis: str
    diet: str
    exercise: str
    follow_up_schedule: str
    created_at: datetime

class OCRLabResultRequest(BaseModel):
    patient_id: str
    image_base64: str

class LabResultResponse(BaseModel):
    id: str
    patient_id: str
    extracted_text: str
    structured_data: dict
    created_at: datetime

class DiagnosticImageRequest(BaseModel):
    patient_id: str
    image_base64: str

class DiagnosticImageResponse(BaseModel):
    anomalies_detected: List[str]
    analysis_notes: str

class PreVisitChatRequest(BaseModel):
    patient_id: str
    raw_text: str

class MergePatientsRequest(BaseModel):
    primary_patient_id: str
    duplicate_patient_id: str
