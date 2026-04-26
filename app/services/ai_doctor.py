import os
import itertools
import json
import requests
import threading
from typing import TypedDict, List, Optional, Annotated
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator
from app.core.database import get_db
from app.services.rag import search_session_documents
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# ── Global Groq Key Round Robin ───────────────────────────────────────────────
_groq_keys_raw = os.getenv("GROQ_API_KEYS", "")
GLOBAL_GROQ_KEYS = [k.strip() for k in _groq_keys_raw.split(",") if k.strip()]
_groq_lock = threading.Lock()
_groq_index = 0

def get_rotated_groq_keys():
    global _groq_index
    if not GLOBAL_GROQ_KEYS: return []
    with _groq_lock:
        start = _groq_index
        _groq_index = (_groq_index + 1) % len(GLOBAL_GROQ_KEYS)
    return GLOBAL_GROQ_KEYS[start:] + GLOBAL_GROQ_KEYS[:start]

# ── Serper Web Search (Thread-Safe Round-Robin) ─────────────────────────────────
_serper_keys_raw = os.getenv("SERPER_API_KEYS", "")
_serper_keys = [k.strip() for k in _serper_keys_raw.split(",") if k.strip()]
_serper_lock = threading.Lock()
_serper_index = 0

def _next_serper_key() -> Optional[str]:
    global _serper_index
    if not _serper_keys:
        return None
    with _serper_lock:
        key = _serper_keys[_serper_index % len(_serper_keys)]
        _serper_index = (_serper_index + 1) % len(_serper_keys)
    return key

def _serper_search(query: str, num: int = 5) -> list[dict]:
    """Call Serper.dev and return structured results."""
    key = _next_serper_key()
    if not key:
        return [{"title": "Web search disabled", "snippet": "No SERPER_API_KEYS configured.", "url": ""}]
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            })
        return results
    except Exception as e:
        return [{"title": "Search failed", "snippet": str(e), "url": ""}]


@tool
def get_illustration(query: str, type: str = "meal") -> str:
    """
    Returns a markdown image URL for a beautiful illustration of a meal or exercise.
    Args:
        query: A short description (e.g., 'grilled salmon', 'yoga pose', 'gym workout')
        type: Either 'meal' or 'exercise'
    """
    # Use Pollinations AI for relevant AI-generated images based exactly on the query
    safe_query = query.lower().replace(' ', '%20')
    if type == "meal":
        return f"![{query}](https://image.pollinations.ai/prompt/delicious%20{safe_query}%20food%20photography?width=800&height=400&nologo=true)"
    else:
        return f"![{query}](https://image.pollinations.ai/prompt/person%20doing%20{safe_query}%20exercise%20fitness?width=800&height=400&nologo=true)"

# ── Prompts ─────────────────────────────────────────────────────────────────────
# ── Graph State ───────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    messages: List
    user_role: str
    doctor_email: str
    patient_context: str      # JSON summary of doctor's patients
    web_search_enabled: bool
    session_id: str           # Current chat session ID for RAG scoping
    sources: List[dict]       # web sources to surface in UI


from pymongo import MongoClient

def get_sync_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client, client[os.getenv("DB_NAME", "healthsync_db")]


class SearchWebInput(BaseModel):
    query: str = Field(
        ...,
        description="A specific, well-formed search query. Include drug names, dosages, or conditions as needed. Example: 'amoxicillin dosage pediatric pneumonia 2024 guidelines'"
    )

class GetPatientListInput(BaseModel):
    doctor_email: str = Field(
        ...,
        description="The authenticated doctor's email address. This is automatically injected — do not ask the user for it."
    )

class GetAppointmentsInput(BaseModel):
    doctor_email: str = Field(
        ...,
        description="The authenticated doctor's email address. Automatically injected — do not ask the user for it."
    )

class CreateAppointmentInput(BaseModel):
    patient_name: str = Field(
        ...,
        description="Full name of the patient. Must match a real patient in the system. Do NOT invent names."
    )
    title: str = Field(
        ...,
        description="Short appointment title, e.g. 'Follow-up Consultation' or 'Post-op Review'."
    )
    date: str = Field(
        ...,
        description="Appointment date in YYYY-MM-DD format. Example: '2026-05-15'. Must be a future date."
    )
    time: str = Field(
        ...,
        description="Appointment time in 24-hour HH:MM format. Example: '14:30' for 2:30 PM."
    )
    appt_type: str = Field(
        default="Checkup",
        description="Type of appointment. Allowed values: 'Checkup', 'Follow-up', 'Consultation', 'Emergency'. Default is 'Checkup'."
    )
    doctor_email: str = Field(
        default="",
        description="Doctor's email. Automatically injected — leave blank."
    )

    @field_validator("appt_type")
    @classmethod
    def validate_appt_type(cls, v: str) -> str:
        allowed = {"Checkup", "Follow-up", "Consultation", "Emergency"}
        if v not in allowed:
            raise ValueError(f"appt_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"date must be in YYYY-MM-DD format, got '{v}'")
        return v

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError(f"time must be in HH:MM 24-hour format, got '{v}'")
        return v

class CancelAppointmentInput(BaseModel):
    appointment_title: str = Field(
        ...,
        description="Title of the appointment to cancel. Use the exact or partial title from the appointment list."
    )
    patient_name: str = Field(
        ...,
        description="Full name of the patient whose appointment should be cancelled."
    )

class UpdateAppointmentInput(BaseModel):
    patient_name: str = Field(
        ...,
        description="Full name of the patient whose appointment is being updated."
    )
    old_title: str = Field(
        ...,
        description="Exact or partial title of the existing appointment to update."
    )
    new_title: Optional[str] = Field(
        default=None,
        description="New title for the appointment. Leave None to keep the existing title."
    )
    new_date: Optional[str] = Field(
        default=None,
        description="New date in YYYY-MM-DD format. Leave None to keep existing date."
    )
    new_time: Optional[str] = Field(
        default=None,
        description="New time in HH:MM 24-hour format. Leave None to keep existing time."
    )
    new_type: Optional[str] = Field(
        default=None,
        description="New appointment type. Allowed values: 'Checkup', 'Follow-up', 'Consultation', 'Emergency'. Leave None to keep existing type."
    )

class SearchDocumentsInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language question or keyword to search for within uploaded documents in this session. Example: 'patient CBC results' or 'medication listed in the report'."
    )
    session_id: str = Field(
        ...,
        description="Current chat session ID used to scope the document search. Automatically injected — do not ask the user for it."
    )

class CreatePatientInput(BaseModel):
    name: str = Field(..., description="Full name of the new patient.")
    email: str = Field(..., description="Patient's email address (unique identifier).")
    age: int = Field(..., ge=0, le=150, description="Patient's age in years.")
    doctor_email: str = Field(default="", description="Doctor's email. Automatically injected — do not ask the user for it.")
    chief_complaint: str = Field(default="", description="Primary reason for visit, e.g. 'Persistent headache for 3 days'. Leave empty if unknown.")
    medical_history: List[str] = Field(default_factory=list, description="List of past medical conditions, e.g. ['Hypertension', 'Asthma']. Use empty list [] if none.")
    current_medications: List[str] = Field(default_factory=list, description="List of current medications e.g. ['Metformin 500mg']. Use empty list [] if none.")
    allergies: List[str] = Field(default_factory=list, description="Known allergies e.g. ['Penicillin']. Use empty list [] if none.")
    triage_status: str = Field(default="Normal", description="Urgency level: 'Normal', 'Urgent', or 'Critical'.")

    @field_validator("medical_history", "current_medications", "allergies", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        """Groq sometimes sends null for list fields — coerce to empty list."""
        return v if v is not None else []

    @field_validator("triage_status", mode="before")
    @classmethod
    def coerce_null_triage(cls, v):
        return v if v is not None else "Normal"

    @field_validator("chief_complaint", mode="before")
    @classmethod
    def coerce_null_complaint(cls, v):
        return v if v is not None else ""


class UpdatePatientInput(BaseModel):
    patient_name: str = Field(..., description="Name of the patient to update. Used to look them up.")
    doctor_email: str = Field(default="", description="Doctor's email. Automatically injected.")
    new_name: Optional[str] = Field(default=None, description="Updated full name.")
    new_age: Optional[int] = Field(default=None, ge=0, le=150, description="Updated age.")
    new_chief_complaint: Optional[str] = Field(default=None, description="Updated chief complaint.")
    new_medications: Optional[List[str]] = Field(default=None, description="Replacement list of current medications.")
    new_allergies: Optional[List[str]] = Field(default=None, description="Replacement list of allergies.")
    new_triage_status: Optional[str] = Field(default=None, description="Updated triage status: 'Normal', 'Urgent', 'Critical'.")
    new_active_issues: Optional[List[str]] = Field(default=None, description="Replacement list of active clinical issues.")

class ArchivePatientInput(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient to archive (soft-delete). They will no longer appear in the active roster.")
    doctor_email: str = Field(default="", description="Doctor's email. Automatically injected.")


# ── Tool Definitions (bound to Pydantic schemas) ───────────────────────────────

@tool(args_schema=SearchWebInput)
def search_web(query: str) -> str:
    """Search the internet for up-to-date medical information, clinical guidelines, or drug data.
    MANDATORY when web search is enabled and the user asks about drugs, dosages, diagnoses, or any recent research.
    Always call this before answering clinical questions from memory."""
    results = _serper_search(query, num=5)
    if not results:
        return "No results found."
    output = []
    for i, r in enumerate(results, 1):
        output.append(f"[{i}] **{r['title']}**\n{r['snippet']}\nSource: {r['url']}")
    return "\n\n".join(output)


@tool(args_schema=GetPatientListInput)
def get_patient_list(doctor_email: str) -> str:
    """Retrieve the complete list of active patients assigned to this doctor.
    ALWAYS call this first before answering anything about 'my patients', a specific patient, or the patient roster.
    Never fabricate patient data — only use what this tool returns."""
    client, db = get_sync_db()
    try:
        patients = list(db["patients"].find({"doctor_email": doctor_email, "is_archived": {"$ne": True}}).limit(50))
        if not patients:
            return "No patients are currently assigned to you."
        lines = []
        for p in patients:
            lines.append(
                f"- **{p.get('name', 'Unknown')}** (Age: {p.get('age', '?')}) | "
                f"Chief complaint: {p.get('chief_complaint', 'N/A')} | "
                f"Status: {p.get('triage_status', 'Normal')} | "
                f"Active issues: {', '.join(p.get('active_issues', [])) or 'None'} | "
                f"Medications: {', '.join(p.get('current_medications', [])) or 'None'} | "
                f"Allergies: {', '.join(p.get('allergies', [])) or 'None'}"
            )
        return "Your current patients:\n" + "\n".join(lines)
    except Exception as e:
        return f"Failed to retrieve patient list: {e}"
    finally:
        client.close()


@tool(args_schema=GetAppointmentsInput)
def get_appointments(doctor_email: str) -> str:
    """Retrieve all upcoming scheduled appointments for this doctor.
    Call this when the doctor asks about their schedule, next appointments, or upcoming visits."""
    client, db = get_sync_db()
    try:
        now = datetime.utcnow()
        appts = list(db["appointments"].find({
            "start_time": {"$gte": now}
        }).sort("start_time", 1).limit(15))
        if not appts:
            return "No upcoming appointments found."
        lines = []
        for a in appts:
            t = a.get("start_time")
            formatted = t.strftime("%A, %d %b %Y at %I:%M %p") if isinstance(t, datetime) else str(t)
            lines.append(f"- **{a.get('title', 'Appointment')}** with {a.get('patient_name', 'Unknown')} — {formatted} ({a.get('type', 'Visit')}, {a.get('status', 'Scheduled')})")
        return "Upcoming appointments:\n" + "\n".join(lines)
    except Exception as e:
        return f"Failed to retrieve appointments: {e}"
    finally:
        client.close()


@tool(args_schema=CreateAppointmentInput)
def create_appointment(patient_name: str, title: str, date: str, time: str, appt_type: str = "Checkup", doctor_email: str = "") -> str:
    """Create a new appointment for a patient.
    STRICT RULES: You MUST have all 4 required fields (patient_name, title, date, time) before calling this.
    If any field is missing, ask the user first. NEVER guess or fill in placeholders.
    This tool creates APPOINTMENTS only — NOT patient records."""
    client, db = get_sync_db()
    try:
        start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=30)

        # Prevent duplicate: check if same patient+title+date already exists
        existing_appt = db["appointments"].find_one({
            "patient_name": {"$regex": f"^{patient_name}$", "$options": "i"},
            "title": {"$regex": f"^{title}$", "$options": "i"},
            "start_time": start_dt,
            "doctor_email": doctor_email
        })
        if existing_appt:
            return f"Appointment '{title}' for {patient_name} on {start_dt.strftime('%A, %d %b %Y at %I:%M %p')} already exists. No duplicate created."

        # Lookup patient — must exist in the system
        patient = db["patients"].find_one({
            "name": {"$regex": f"^{patient_name}$", "$options": "i"},
            "doctor_email": doctor_email
        })
        if not patient:
            # Try fuzzy match
            patient = db["patients"].find_one({
                "name": {"$regex": patient_name, "$options": "i"},
                "doctor_email": doctor_email
            })
        patient_id = str(patient["_id"]) if patient else None
        if not patient_id:
            return f"Patient '{patient_name}' not found in your patient records. Please verify the name and try again. Do NOT invent a patient_id."

        doc = {
            "patient_id": patient_id,
            "patient_name": patient["name"],   # Use canonical DB name
            "title": title,
            "start_time": start_dt,
            "end_time": end_dt,
            "type": appt_type,
            "status": "Scheduled",
            "doctor_email": doctor_email,
            "created_at": datetime.utcnow()
        }
        db["appointments"].insert_one(doc)
        return f"Appointment '{title}' created successfully for **{patient['name']}** on {start_dt.strftime('%A, %d %b %Y at %I:%M %p')}."
    except Exception as e:
        return f"Failed to create appointment: {e}"
    finally:
        client.close()


@tool(args_schema=CancelAppointmentInput)
def cancel_appointment(appointment_title: str, patient_name: str) -> str:
    """Cancel and permanently delete a scheduled appointment.
    Requires both the appointment title and patient name to prevent accidental deletions."""
    client, db = get_sync_db()
    try:
        result = db["appointments"].delete_one({
            "title": {"$regex": appointment_title, "$options": "i"},
            "patient_name": {"$regex": patient_name, "$options": "i"}
        })
        if result.deleted_count == 0:
            return f"No appointment found matching '{appointment_title}' for '{patient_name}'."
        return f"Appointment '{appointment_title}' for {patient_name} has been cancelled."
    except Exception as e:
        return f"Failed to cancel appointment: {e}"
    finally:
        client.close()


@tool(args_schema=UpdateAppointmentInput)
def update_appointment(patient_name: str, old_title: str, new_title: Optional[str] = None, new_date: Optional[str] = None, new_time: Optional[str] = None, new_type: Optional[str] = None) -> str:
    """Update an existing appointment's details (date, time, title, or type).
    Provide only the fields you want to change. Fields left as None will retain their current values."""
    client, db = get_sync_db()
    try:
        query = {
            "title": {"$regex": old_title, "$options": "i"},
            "patient_name": {"$regex": patient_name, "$options": "i"}
        }
        existing = db["appointments"].find_one(query)
        if not existing:
            return f"No appointment found for '{patient_name}' with title '{old_title}'."
        updates = {}
        if new_title: updates["title"] = new_title
        if new_type: updates["type"] = new_type
        if new_date or new_time:
            curr_dt = existing["start_time"]
            d = new_date or curr_dt.strftime("%Y-%m-%d")
            t = new_time or curr_dt.strftime("%H:%M")
            try:
                new_dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
                updates["start_time"] = new_dt
                updates["end_time"] = new_dt + timedelta(minutes=30)
            except ValueError:
                return "Invalid date/time format. Use YYYY-MM-DD and HH:MM."
        if not updates:
            return "No changes provided to update."
        db["appointments"].update_one({"_id": existing["_id"]}, {"$set": updates})
        return f"Appointment for **{patient_name}** has been updated successfully."
    except Exception as e:
        return f"Failed to update appointment: {e}"
    finally:
        client.close()


@tool(args_schema=SearchDocumentsInput)
def search_documents(query: str, session_id: str) -> str:
    """Search the contents of uploaded documents (PDF, DOCX, TXT) in this session using vector similarity.
    ALWAYS call this when the user asks about content from a document they uploaded.
    Do NOT attempt to answer document-related questions from memory."""
    try:
        return search_session_documents(query, session_id)
    except Exception as e:
        return f"Failed to search documents: {e}"


@tool(args_schema=CreatePatientInput)
def create_patient(
    name: str,
    email: str,
    age: int,
    doctor_email: str = "",
    chief_complaint: Optional[str] = None,
    medical_history: Optional[List[str]] = None,
    current_medications: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    triage_status: Optional[str] = "Normal",
) -> str:
    """Create a new patient record in the system.
    Use this ONLY when the doctor explicitly wants to add a new patient.
    This is SEPARATE from creating appointments — after creating a patient, offer to schedule an appointment but DO NOT do it automatically.
    NEVER use create_appointment to create a patient."""
    client, db = get_sync_db()
    try:
        # Prevent duplicates by email
        existing = db["patients"].find_one({"email": email.lower()})
        if existing:
            return f"A patient with email '{email}' already exists: {existing.get('name')}. No duplicate created."
        doc = {
            "name": name,
            "email": email.lower(),
            "age": age,
            "chief_complaint": chief_complaint or "",
            "medical_history": medical_history or [],
            "current_medications": current_medications or [],
            "allergies": allergies or [],
            "triage_status": triage_status or "Normal",
            "active_issues": [],
            "is_archived": False,
            "doctor_email": doctor_email,   # ← stamped at creation
            "created_at": datetime.utcnow()
        }
        result = db["patients"].insert_one(doc)
        return f"✅ Patient **{name}** created successfully (ID: {result.inserted_id}). Would you like to schedule an appointment for this patient?"
    except Exception as e:
        return f"Failed to create patient: {e}"
    finally:
        client.close()


@tool(args_schema=UpdatePatientInput)
def update_patient(
    patient_name: str,
    doctor_email: str = "",
    new_name: Optional[str] = None,
    new_age: Optional[int] = None,
    new_chief_complaint: Optional[str] = None,
    new_medications: Optional[List[str]] = None,
    new_allergies: Optional[List[str]] = None,
    new_triage_status: Optional[str] = None,
    new_active_issues: Optional[List[str]] = None,
) -> str:
    """Update an existing patient's clinical information.
    Only the fields you provide will be updated — others stay unchanged.
    Use this to update medications, allergies, triage status, or active issues."""
    client, db = get_sync_db()
    try:
        patient = db["patients"].find_one({
            "name": {"$regex": patient_name, "$options": "i"},
            "doctor_email": doctor_email
        })
        if not patient:
            return f"Patient '{patient_name}' not found in your roster."
        updates = {}
        if new_name: updates["name"] = new_name
        if new_age is not None: updates["age"] = new_age
        if new_chief_complaint is not None: updates["chief_complaint"] = new_chief_complaint
        if new_medications is not None: updates["current_medications"] = new_medications
        if new_allergies is not None: updates["allergies"] = new_allergies
        if new_triage_status: updates["triage_status"] = new_triage_status
        if new_active_issues is not None: updates["active_issues"] = new_active_issues
        if not updates:
            return "No changes provided."
        db["patients"].update_one({"_id": patient["_id"]}, {"$set": updates})
        return f"✅ Patient **{patient.get('name')}** updated successfully."
    except Exception as e:
        return f"Failed to update patient: {e}"
    finally:
        client.close()


@tool(args_schema=ArchivePatientInput)
def archive_patient(patient_name: str, doctor_email: str = "") -> str:
    """Archive (soft-delete) a patient from the active roster. The patient record is preserved but hidden from the patient list.
    Use when the doctor says 'remove patient', 'discharge patient', or 'archive patient'."""
    client, db = get_sync_db()
    try:
        patient = db["patients"].find_one({
            "name": {"$regex": patient_name, "$options": "i"},
            "doctor_email": doctor_email
        })
        if not patient:
            return f"Patient '{patient_name}' not found."
        db["patients"].update_one({"_id": patient["_id"]}, {"$set": {"is_archived": True}})
        return f"Patient **{patient.get('name')}** has been archived and removed from the active roster."
    except Exception as e:
        return f"Failed to archive patient: {e}"
    finally:
        client.close()

class FindDoctorsInput(BaseModel):
    specialty: str = Field(..., description="Medical specialty to search for, e.g. Cardiology, Dermatology")

@tool(args_schema=FindDoctorsInput)
def find_doctors(specialty: str) -> str:
    """Find doctors by specialty in the network. Use this to help patients find doctors to treat them."""
    client, db = get_sync_db()
    try:
        doctors = list(db["users"].find({"role": "doctor", "specialty": {"$regex": specialty, "$options": "i"}}).limit(5))
        if not doctors:
            return f"No doctors found for specialty: {specialty}."
        res = "Found Doctors:\n"
        for d in doctors:
            res += f"- Dr. {d.get('full_name', 'Unknown')} ({d.get('specialty', 'General')}) | Email: {d.get('email', 'N/A')} | Clinic: {d.get('clinic_name', 'N/A')}\n"
        return res
    except Exception as e:
        return f"Failed to find doctors: {e}"
    finally:
        client.close()

class TrackProgressInput(BaseModel):
    patient_email: str = Field(default="", description="Patient email, injected automatically")
    log_entry: str = Field(..., description="The progress to log (e.g., 'Feeling better, weight 70kg')")

@tool(args_schema=TrackProgressInput)
def track_progress(patient_email: str, log_entry: str) -> str:
    """Track the patient's daily progress, symptoms, or meals. ALWAYS use this when a patient reports their daily status."""
    client, db = get_sync_db()
    try:
        db["patient_progress"].insert_one({
            "patient_email": patient_email,
            "log": log_entry,
            "date": datetime.utcnow()
        })
        return "Progress logged successfully in the system."
    except Exception as e:
        return f"Failed to log progress: {e}"
    finally:
        client.close()


# ── System Prompts ────────────────────────────────────────────────────────────
DOCTOR_SYSTEM_PROMPT = """You are HealthSync, an advanced clinical AI copilot embedded inside a doctor's dashboard.

YOUR IDENTITY & ROLE:
- You are a peer-level clinical assistant speaking directly to a licensed physician.
- You have real-time access to this doctor's patient roster and appointment schedule via tools.
- You can also search the web for current clinical guidelines, drug information, and recent research.

═══════════════════════════════════════════════════════════
  ZERO-TOLERANCE ANTI-HALLUCINATION RULES (ABSOLUTE LAW):
═══════════════════════════════════════════════════════════
1. NEVER invent, fabricate, or assume ANY patient data. If you do not have data from tool output, it DOES NOT EXIST.
2. NEVER mix up tools. Patient CRUD (≠ appointments):
   - To CREATE A PATIENT → call `create_patient`. Then OFFER to schedule — DO NOT auto-call `create_appointment`.
   - To CREATE AN APPOINTMENT → call `create_appointment`. Needs: patient_name, title, date, time.
   - To UPDATE A PATIENT → call `update_patient`.
   - To REMOVE A PATIENT → call `archive_patient`.
3. NEVER simulate a tool call in text. Call it silently or not at all.
4. NEVER guess data not in tool output. If tool returns empty, say "No data found" and STOP.
5. NEVER call `create_appointment` with missing fields. ASK for them first.
6. If web_search is ENABLED, call `search_web` for ANY drug/dosage/guideline question. NEVER answer from memory.

CORE CAPABILITIES:
1. **Patient CRUD**:
   - `create_patient` → add a new patient. After creation, OFFER to schedule but don't auto-schedule.
   - `get_patient_list` → list all patients. Call FIRST before answering about "my patients".
   - `update_patient` → update patient info (meds, allergies, triage, etc.).
   - `archive_patient` → remove a patient from the active roster.
2. **Appointments**: `get_appointments`, `create_appointment`, `update_appointment`, `cancel_appointment`.
3. **Clinical Knowledge + Web Search**: Call `search_web` immediately for any drugs/dosages/protocols if enabled.
4. **Uploaded Documents**: Call `search_documents` if user asks about an uploaded file.

RESPONSE STYLE:
- Use professional medical terminology.
- Be concise. No filler. No disclaimers.
- NEVER start with "Certainly!" or "Of course!" — get straight to the point.

### REQUIRED FORMATTING:
1. Divide every response into logical sections with `### Header` (H3 markdown headers, never bold text).
2. Use `- ` bullet points for ALL lists. No raw inline lists.
3. Leave an empty line before AND after every header and between every block.
4. DO NOT manually write a "Sources" section — the UI injects citations automatically.

EXAMPLE:

### Finding

- Item A
- Item B

### Assessment

Your assessment here.

SAFETY:
- Flag drug interactions or contraindications prominently with ⚠️.
- If a user request is ambiguous, ask ONE focused clarifying question before acting.
- If you are uncertain about anything clinical, say so explicitly.

{patient_context}
{doctor_context}"""

# ── LangGraph Node ────────────────────────────────────────────────────────────
TOOLS = [
    search_web,
    get_patient_list, create_patient, update_patient, archive_patient,
    get_appointments, create_appointment, update_appointment, cancel_appointment,
    search_documents
]

def doctor_node(state: GraphState):
    messages = state["messages"]
    doctor_email = state.get("doctor_email", "")
    patient_context = state.get("patient_context", "")
    web_search_enabled = state.get("web_search_enabled", True)
    session_id = state.get("session_id", "")

    doctor_ctx = f"Doctor email: {doctor_email}" if doctor_email else ""
    pt_ctx = f"\n\nPATIENT CONTEXT (pre-loaded):\n{patient_context}" if patient_context else ""
    system_content = DOCTOR_SYSTEM_PROMPT.format(
        patient_context=pt_ctx,
        doctor_context=doctor_ctx
    )
    if web_search_enabled:
        available_tools = TOOLS
    else:
        available_tools = [t for t in TOOLS if t.name != "search_web"]

    keys = get_rotated_groq_keys()
    models = ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"]
    
    llms = []
    for model in models:
        for key in keys:
            llm = ChatGroq(model=model, groq_api_key=key, temperature=0.2, max_retries=0)
            if available_tools:
                llm = llm.bind_tools(available_tools)
            llms.append(llm)

    primary_llm = llms[0]
    llm_with_tools = primary_llm.with_fallbacks(llms[1:]) if len(llms) > 1 else primary_llm
    
    now_utc = datetime.utcnow()
    from datetime import timezone
    import zoneinfo
    try:
        ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        now_local = datetime.now(ist)
        datetime_context = f"Current date and time: {now_local.strftime('%A, %d %B %Y at %I:%M %p IST')} (UTC: {now_utc.strftime('%Y-%m-%d %H:%M')})"
    except Exception:
        datetime_context = f"Current date and time (UTC): {now_utc.strftime('%A, %d %B %Y at %I:%M %p')}"

    if web_search_enabled:
        system_content = system_content + f"\n\n[SYSTEM OVERRIDE] {datetime_context}. Web search IS ENABLED. For ANY clinical question about drugs, dosages, diagnoses, or guidelines, you MUST call `search_web` FIRST. Do NOT answer from memory alone."
    else:
        system_content = system_content + f"\n\n[CONTEXT] {datetime_context}. Web search is DISABLED for this session."
    
    full_context = [SystemMessage(content=system_content)] + messages

    response = llm_with_tools.invoke(full_context)
    
    new_sources = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_results = []
        tool_map = {t.name: t for t in available_tools}
        
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            
            INJECT_DOCTOR_EMAIL = {"create_appointment", "get_patient_list", "get_appointments",
                                   "update_patient", "archive_patient", "create_patient"}
            if tool_name in INJECT_DOCTOR_EMAIL:
                tool_args["doctor_email"] = doctor_email
            if tool_name == "search_documents":
                tool_args["session_id"] = session_id
                
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool error: {e}"
            
            if tool_name == "search_web" and result:
                import re
                urls = re.findall(r'Source: (https?://\S+)', result)
                titles = re.findall(r'\[\d+\] \*\*(.*?)\*\*', result)
                for title, url in zip(titles, urls):
                    new_sources.append({"title": title, "url": url})
            
            tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        
        followup = [SystemMessage(content=system_content)] + messages + [response] + tool_results
        
        bare_llms = []
        for model in models:
            for key in keys:
                bare_llms.append(ChatGroq(model=model, groq_api_key=key, temperature=0.2, max_retries=0))
                
        final_llm = bare_llms[0].with_fallbacks(bare_llms[1:]) if len(bare_llms) > 1 else bare_llms[0]
        final_response = final_llm.invoke(followup)

        return {
            "messages": [response] + tool_results + [AIMessage(content=final_response.content)],
            "sources": new_sources
        }
    
    return {"messages": [response], "sources": []}

# ── Graph Setup ───────────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)
workflow.add_node("doctor", doctor_node)
workflow.set_entry_point("doctor")
workflow.add_edge("doctor", END)
agent_app = workflow.compile()