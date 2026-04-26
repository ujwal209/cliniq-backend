from fastapi import APIRouter, HTTPException, Depends
from app.core.database import get_db
from app.models.chat import ChatSession, ChatMessage
from app.services.ai_doctor import agent_app as doctor_agent_app
from app.services.patient_agent import patient_agent_app
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from fastapi.concurrency import run_in_threadpool
from app.core.deps import get_current_user
from app.services.rag import index_document
from pydantic import BaseModel
import re

CONTEXT_WINDOW_CHARS = 80_000   # ~100k tokens @ ~1.3 chars/token — warn before hitting 128K limit

class DocumentUploadRequest(BaseModel):
    doc_url: str

class MessageSendRequest(BaseModel):
    content: str
    web_search: bool = False

class RenameSessionRequest(BaseModel):
    title: str

router = APIRouter()

@router.post("/session/start")
async def start_session(
    patient_email: str | None = None,
    current_user: dict = Depends(get_current_user)
):
    # If the user is not explicitly a doctor, force them to act as a patient 
    # for their own email, ignoring any passed patient_email.
    if current_user.get("role") != "doctor":
        patient_email = current_user.get("sub")
    elif not patient_email:
        raise HTTPException(status_code=400, detail="patient_email is required for doctors")
        
    if current_user.get("role") != "doctor" and current_user.get("sub") != patient_email:
        raise HTTPException(status_code=403, detail="You can only start a session for your own account.")

    db = get_db()
    
    # ── Context continuity: Pull history from previous session ──
    last_session = await db.chat_sessions.find_one(
        {"patient_email": patient_email}, 
        sort=[("created_at", -1)]
    )
    
    initial_messages = []
    if last_session and last_session.get("messages"):
        last_msgs = last_session["messages"][-20:] # Pull up to last 20 interactions
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_msgs])
        
        sys_msg = ChatMessage(
            role="system",
            content=f"PREVIOUS SESSION CONTEXT:\nThe following is a transcript of the user's most recent consultation. The Llama model has a 128K context window, so this is provided for seamless continuity. Use this context if the user refers to past discussions:\n\n{history_text}"
        )
        initial_messages.append(sys_msg.model_dump())

    new_session = ChatSession(
        patient_email=patient_email,
        messages=initial_messages 
    )
    
    await db.chat_sessions.insert_one(new_session.model_dump())
    return {"session_id": new_session.session_id, "message": "Session started successfully"}


@router.post("/session/{session_id}/document")
async def upload_session_document(
    session_id: str,
    req: DocumentUploadRequest,
    current_user: dict = Depends(get_current_user)
):
    """Downloads a document, extracts text, embeds it, and stores in the session's vector store."""
    db = get_db()
    session_data = await db.chat_sessions.find_one({"session_id": session_id})
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied")
        
    try:
        chunks_inserted = await index_document(req.doc_url, session_id)
        return {"message": "Document indexed successfully", "chunks": chunks_inserted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/message")
async def send_message(
    session_id: str, 
    content: str,
    web_search: bool = False,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    session_data = await db.chat_sessions.find_one({"session_id": session_id})
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied to this chat session")

    # 1. Save user message
    user_msg = ChatMessage(role="user", content=content)
    session_data["messages"].append(user_msg.model_dump())

    # 2. Format history for LangChain
    lc_messages = []
    for msg in session_data["messages"]:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # 3. Build patient context for doctor (pre-load roster for personalization)
    doctor_email = current_user.get("sub", "")
    patient_context_str = ""
    if current_user.get("role") == "doctor":
        try:
            cursor = db["patients"].find({"doctor_email": doctor_email, "is_archived": {"$ne": True}})
            patients = await cursor.to_list(length=20)
            if patients:
                summaries = []
                for p in patients:
                    summaries.append(
                        f"- {p.get('name', 'Unknown')} (Age {p.get('age', '?')}): "
                        f"{p.get('chief_complaint', 'No complaint')} | "
                        f"Status: {p.get('triage_status', 'Normal')} | "
                        f"Meds: {', '.join(p.get('current_medications', [])) or 'None'}"
                    )
                patient_context_str = "\n".join(summaries)
        except Exception:
            patient_context_str = ""
    elif current_user.get("role") != "doctor":
        try:
            patient_profile = await db["users"].find_one({"email": current_user.get("sub")})
            if patient_profile:
                patient_context_str = (
                    f"Patient Name: {patient_profile.get('full_name', 'Unknown')}\n"
                    f"Age: {patient_profile.get('age', 'Unknown')}\n"
                    f"Gender: {patient_profile.get('gender', 'Unknown')}\n"
                    f"Health Goals: {patient_profile.get('health_goals', 'Not specified')}\n"
                    f"Pre-existing Conditions: {', '.join(patient_profile.get('pre_existing_conditions', [])) or 'None'}\n"
                    f"Allergies: {', '.join(patient_profile.get('allergies', [])) or 'None'}\n"
                )
        except Exception:
            patient_context_str = ""

    # 4. Context-window check — warn doctor before hitting Llama's 128K limit
    total_chars = sum(len(m.get("content", "")) for m in session_data["messages"])
    context_warning = None
    if total_chars > CONTEXT_WINDOW_CHARS:
        context_warning = (
            "\u26a0\ufe0f This chat session is approaching the context window limit. "
            "To maintain AI quality, please summarise this conversation and start a new chat session."
        )

    # 5. Run LangGraph agent in a worker thread to prevent event loop blocking
    if current_user.get("role") == "doctor":
        target_app = doctor_agent_app
        invoke_args = {
            "messages": lc_messages,
            "doctor_email": doctor_email,
            "patient_context": patient_context_str,
            "web_search_enabled": web_search,
            "session_id": session_id,
            "sources": []
        }
    else:
        target_app = patient_agent_app
        invoke_args = {
            "messages": lc_messages,
            "patient_email": session_data["patient_email"],
            "patient_context": patient_context_str,
            "web_search_enabled": web_search,
            "session_id": session_id,
            "sources": []
        }

    result = await run_in_threadpool(
        target_app.invoke,
        invoke_args
    )
    
    raw_content = result["messages"][-1].content
    sources = result.get("sources", [])

    # Handle multimodal content (list of dicts) vs plain string
    if isinstance(raw_content, list):
        ai_response_content = ""
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                ai_response_content += block.get("text", "")
            elif isinstance(block, str):
                ai_response_content += block
        if not ai_response_content:
            ai_response_content = str(raw_content)
    else:
        ai_response_content = str(raw_content)

    # 5. Save AI response
    ai_msg = ChatMessage(role="assistant", content=ai_response_content, sources=sources)
    session_data["messages"].append(ai_msg.model_dump())

    await db.chat_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"messages": session_data["messages"], "updated_at": datetime.utcnow()}}
    )

    return {
        "reply": ai_response_content,
        "message_id": ai_msg.id,
        "sources": sources,
        "context_warning": context_warning   # None unless limit approaching
    }



@router.post("/session/{session_id}/feedback/{message_id}")
async def submit_feedback(
    session_id: str, 
    message_id: str, 
    feedback: str,
    current_user: dict = Depends(get_current_user) # <-- LOCK APPLIED
):
    if feedback not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'like' or 'dislike'")

    db = get_db()
    session_data = await db.chat_sessions.find_one({"session_id": session_id})
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # 🔒 SECURE: Check ownership
    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await db.chat_sessions.update_one(
        {"session_id": session_id, "messages.id": message_id},
        {"$set": {"messages.$.feedback": feedback}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
        
    return {"message": f"Feedback '{feedback}' recorded successfully"}


@router.get("/session/{session_id}/history")
async def get_history(
    session_id: str,
    current_user: dict = Depends(get_current_user) # <-- LOCK APPLIED
):
    db = get_db()
    session_data = await db.chat_sessions.find_one(
        {"session_id": session_id},
        {"_id": 0}
    )
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # 🔒 SECURE: Check ownership
    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied")
        
    return session_data

@router.get("/session")
async def get_sessions(current_user: dict = Depends(get_current_user)):
    db = get_db()
    
    # 🔒 SECURE: Doctors can see all, patients see only theirs. Or better, doctors see sessions they started or all sessions. Let's just filter by email for patients, and no filter for doctors (or maybe doctor email). Wait, start_session uses patient_email. If doctor is starting it, maybe we should track doctor_email too. For now, if role is doctor, return all sessions or sessions where patient_email is the doctor's email. Actually, the frontend passed doctor_email as patient_email in startSession.
    
    query = {}
    if current_user.get("role") != "doctor":
        query = {"patient_email": current_user.get("sub")}
        
    cursor = db.chat_sessions.find(
        query,
        {"_id": 0, "session_id": 1, "patient_email": 1, "updated_at": 1,
         "is_archived": 1, "custom_title": 1, "messages": {"$slice": 3}}
    ).sort("updated_at", -1)
    sessions = await cursor.to_list(length=100)

    history_list = []
    for s in sessions:
        title = s.get("custom_title")
        if not title:
            for m in s.get("messages", []):
                if m.get("role") == "user":
                    raw = m.get("content", "")
                    title = (raw[:50] + "…") if len(raw) > 50 else raw
                    break
        if not title:
            title = "New Chat"
        history_list.append({
            "id": s["session_id"],
            "title": title,
            "date": s.get("updated_at", datetime.utcnow()).timestamp() * 1000,
            "is_archived": s.get("is_archived", False),
            "has_custom_title": bool(s.get("custom_title"))
        })
    return history_list

@router.post("/session/{session_id}/archive")
async def toggle_archive(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    session_data = await db.chat_sessions.find_one({"session_id": session_id})
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied")
        
    new_status = not session_data.get("is_archived", False)
    await db.chat_sessions.update_one({"session_id": session_id}, {"$set": {"is_archived": new_status}})
    return {"message": "Chat archived" if new_status else "Chat unarchived", "is_archived": new_status}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Permanently delete a chat session and all its document embeddings."""
    db = get_db()
    session_data = await db.chat_sessions.find_one({"session_id": session_id})
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied")

    await db.chat_sessions.delete_one({"session_id": session_id})
    # Also clean up any RAG document embeddings for this session
    await db.document_embeddings.delete_many({"session_id": session_id})
    return {"message": "Chat session deleted successfully"}


@router.patch("/session/{session_id}/rename")
async def rename_session(session_id: str, req: RenameSessionRequest, current_user: dict = Depends(get_current_user)):
    """Rename a chat session by setting a custom title."""
    db = get_db()
    session_data = await db.chat_sessions.find_one({"session_id": session_id})
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if current_user.get("role") != "doctor" and session_data["patient_email"] != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Access denied")

    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if len(title) > 80:
        raise HTTPException(status_code=400, detail="Title too long (max 80 chars)")

    await db.chat_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"custom_title": title, "updated_at": datetime.utcnow()}}
    )
    return {"message": "Session renamed", "title": title}


@router.post("/suggestions")
async def get_suggestions(
    partial: str = "",
    current_user: dict = Depends(get_current_user)
):
    """Generate prompt suggestions using Serper Autocomplete."""
    import os, requests as req
    serper_keys = [k.strip() for k in os.getenv("SERPER_API_KEYS", "").split(",") if k.strip()]
    if not serper_keys or len(partial) < 2:
        return {"suggestions": []}
    
    key = serper_keys[0]
    
    try:
        resp = req.post(
            "https://google.serper.dev/autocomplete",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": partial},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            suggestions = [s["value"] for s in data.get("suggestions", [])[:5]]
            return {"suggestions": suggestions}
    except Exception as e:
        print("Serper autocomplete error:", e)
        
    return {"suggestions": []}