from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.core.database import get_db
from app.core.deps import get_current_user
import requests
import os
import logging
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from app.services.rag import index_medical_report, search_patient_reports, get_embedding_model, cosine_similarity
from app.routes.vision import get_round_robin_keys

router = APIRouter()

class ReportUploadRequest(BaseModel):
    title: str
    file_url: str
    file_type: str # 'pdf', 'image', 'docx'
    category: Optional[str] = "General"

class ChatRequest(BaseModel):
    message: str
    report_id: Optional[str] = None
    session_id: Optional[str] = None

@router.post("/upload")
async def upload_report(
    req: ReportUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    patient_email = current_user.get("sub")
    
    report_id = str(uuid.uuid4())
    report_doc = {
        "id": report_id,
        "patient_email": patient_email,
        "title": req.title,
        "file_url": req.file_url,
        "file_type": req.file_type,
        "category": req.category,
        "created_at": datetime.utcnow(),
        "status": "indexing"
    }
    
    await db.medical_reports.insert_one(report_doc)
    
    # Trigger background indexing
    background_tasks.add_task(
        index_and_update_status, 
        req.file_url, 
        patient_email, 
        report_id, 
        req.file_type
    )
    
    return {"message": "Report uploaded and indexing started", "report_id": report_id}

async def index_and_update_status(file_url: str, email: str, report_id: str, file_type: str):
    await index_medical_report(file_url, email, report_id, file_type)
    db = get_db()
    await db.medical_reports.update_one({"id": report_id}, {"$set": {"status": "ready"}})

@router.get("/")
async def get_reports(current_user: dict = Depends(get_current_user)):
    db = get_db()
    patient_email = current_user.get("sub")
    cursor = db.medical_reports.find({"patient_email": patient_email}).sort("created_at", -1)
    reports = await cursor.to_list(length=100)
    for r in reports:
        r["_id"] = str(r["_id"])
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
    return reports

@router.post("/sessions")
async def create_session(current_user: dict = Depends(get_current_user)):
    db = get_db()
    session_id = str(uuid.uuid4())
    session_doc = {
        "id": session_id,
        "patient_email": current_user.get("sub"),
        "title": "New Report Analysis",
        "created_at": datetime.utcnow(),
        "messages": []
    }
    await db.report_sessions.insert_one(session_doc)
    return {"session_id": session_id}

@router.get("/sessions")
async def get_sessions(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db.report_sessions.find({"patient_email": current_user.get("sub")}).sort("created_at", -1)
    sessions = await cursor.to_list(length=50)
    for s in sessions:
        s["_id"] = str(s["_id"])
        if isinstance(s.get("created_at"), datetime):
            s["created_at"] = s.get("created_at").isoformat()
    return sessions

@router.get("/sessions/{session_id}")
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    session = await db.report_sessions.find_one({"id": session_id, "patient_email": current_user.get("sub")})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session["_id"] = str(session["_id"])
    if isinstance(session.get("created_at"), datetime):
        session["created_at"] = session.get("created_at").isoformat()
    return session

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    await db.report_sessions.delete_one({"id": session_id, "patient_email": current_user.get("sub")})
    return {"message": "Session deleted"}

@router.post("/chat")
async def chat_with_reports(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    patient_email = current_user.get("sub")
    db = get_db()
    logger = logging.getLogger(__name__)
    
    try:
        # 0. Get History if session_id provided
        history = []
        if req.session_id:
            session = await db.report_sessions.find_one({"id": req.session_id})
            if session:
                history = session.get("messages", [])

        # 1. Retrieve relevant context
        if req.report_id:
            cursor = db.medical_report_embeddings.find({
                "patient_email": patient_email,
                "report_id": req.report_id
            })
            chunks = await cursor.to_list(length=100)
            
            embedder = get_embedding_model()
            query_emb = embedder.embed_query(req.message)
            
            scored_chunks = []
            for c in chunks:
                try:
                    emb = c.get("embedding")
                    if not emb: continue
                    score = cosine_similarity(query_emb, emb)
                    scored_chunks.append((score, c.get("text", "")))
                except Exception as e:
                    logger.error(f"Similarity error: {e}")
                    continue
                
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_results = scored_chunks[:10]
            context = "\n\n".join([t for s, t in top_results if s > 0.5])
            if not context:
                context = "No highly relevant information found specifically in this report."
        else:
            context = await search_patient_reports(req.message, patient_email)
        
        # 2. Call Groq for analysis
        from app.services.ai_doctor import get_rotated_groq_keys
        keys = get_rotated_groq_keys()
        if not keys:
            raise HTTPException(status_code=500, detail="Groq API Key missing")
            
        system_prompt = (
            "You are HealthSync Report Agent, an elite medical document analyzer. "
            "Use the provided context from the patient's medical reports to answer their questions.\n\n"
            "STRICT OUTPUT FORMATTING RULES:\n"
            "1. Use clear, professional markdown sections with headings (###).\n"
            "2. Always structure your analysis into logical parts: ### Executive Summary, ### Key Findings, ### Clinical Impression, and ### Next Steps.\n"
            "3. Use bolding for critical values or abnormal findings.\n"
            "4. Use bullet points for lists of symptoms or observations.\n"
            "5. Maintain a professional, clinical yet accessible tone.\n"
            "6. Always include a short disclaimer at the end about consulting a primary physician.\n\n"
            f"PATIENT CONTEXT FROM REPORTS:\n{context}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        # Add history (last 10 messages)
        for m in history[-10:]:
            role = "assistant" if m["role"] == "ai" else m["role"]
            messages.append({"role": role, "content": m["text"]})
        
        messages.append({"role": "user", "content": req.message})
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.2
        }
        
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {keys[0]}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        answer = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 3. Save to history
        if req.session_id:
            await db.report_sessions.update_one(
                {"id": req.session_id},
                {
                    "$push": {
                        "messages": {
                            "$each": [
                                {"role": "user", "text": req.message, "timestamp": datetime.utcnow().isoformat()},
                                {"role": "ai", "text": answer, "timestamp": datetime.utcnow().isoformat()}
                            ]
                        }
                    },
                    "$set": {"title": req.message[:30] + "..." if len(history) == 0 else history[0].get("title", "Report Analysis")}
                }
            )
        
        return {
            "answer": answer,
            "context_used": context != "No medical reports have been indexed yet."
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{report_id}")
async def delete_report(report_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    patient_email = current_user.get("sub")
    
    await db.medical_reports.delete_one({"id": report_id, "patient_email": patient_email})
    await db.medical_report_embeddings.delete_many({"report_id": report_id, "patient_email": patient_email})
    

@router.delete("/")
async def delete_all_reports(current_user: dict = Depends(get_current_user)):
    db = get_db()
    patient_email = current_user.get("sub")
    
    await db.medical_reports.delete_many({"patient_email": patient_email})
    await db.medical_report_embeddings.delete_many({"patient_email": patient_email})
    await db.report_sessions.delete_many({"patient_email": patient_email})
    
    return {"message": "All reports and sessions cleared"}
