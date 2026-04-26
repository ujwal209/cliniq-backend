from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional, List

class PatientModel(BaseModel):
    name: str
    email: EmailStr
    age: int
    chief_complaint: Optional[str] = None
    medical_history: List[str] = []
    current_medications: List[str] = []
    allergies: List[str] = []
    triage_status: str = Field(default="Normal")
    active_issues: List[str] = []
    is_archived: bool = False
    doctor_email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "chief_complaint": "Persistent headache for 3 days",
                "medical_history": ["High blood pressure", "Asthma"],
                "current_medications": ["Lisinopril", "Albuterol"],
                "allergies": ["Penicillin", "Peanuts"],
                "triage_status": "Normal",
                "active_issues": ["Occasional shortness of breath"],
                "is_archived": False
            }
        }