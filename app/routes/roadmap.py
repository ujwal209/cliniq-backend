from fastapi import APIRouter, Depends, HTTPException
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.roadmap import (
    RoadmapGenerateRequest, 
    TaskCompletionRequest, 
    RoadmapModifyRequest
)
from app.services.roadmap_agent import generate_roadmap_plan, modify_roadmap_plan
from datetime import datetime
from bson.objectid import ObjectId

router = APIRouter()


@router.post("/generate")
async def generate_patient_roadmap(
    req: RoadmapGenerateRequest,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    email = current_user.get("sub")
    
    user_data = await db.users.find_one({"email": email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
        
    patient_context_str = (
        f"Age: {user_data.get('age', 'Unknown')}\n"
        f"Gender: {user_data.get('gender', 'Unknown')}\n"
        f"Pre-existing Conditions: {', '.join(user_data.get('pre_existing_conditions', [])) or 'None'}\n"
        f"Allergies: {', '.join(user_data.get('allergies', [])) or 'None'}\n"
    )
    
    plan = generate_roadmap_plan(req.disease, req.goals, req.symptoms, patient_context_str, req.duration_days)
    
    roadmap_doc = {
        "patient_email": email,
        "disease": req.disease,
        "goals": req.goals,
        "symptoms": req.symptoms,
        "duration_days": req.duration_days,
        "plan": plan,
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    result = await db.roadmaps.insert_one(roadmap_doc)
    return {"id": str(result.inserted_id), "plan": plan}


# ─── IMPORTANT: /all MUST be registered BEFORE /{roadmap_id} ──────────────────
@router.get("/all")
async def get_all_roadmaps(current_user: dict = Depends(get_current_user)):
    db = get_db()
    email = current_user.get("sub")
    
    cursor = db.roadmaps.find({"patient_email": email}).sort("created_at", -1)
    roadmaps = await cursor.to_list(length=100)
    
    formatted = []
    for r in roadmaps:
        formatted.append({
            "id": str(r["_id"]),
            "disease": r["disease"],
            "duration_days": r.get("duration_days", 7),
            "created_at": r["created_at"].isoformat() if hasattr(r.get("created_at"), "isoformat") else str(r.get("created_at", "")),
            "summary": r.get("plan", {}).get("summary", "Clinical Protocol"),
            "status": r.get("status", "active")
        })
        
    return {"roadmaps": formatted}


# ─── Single roadmap (wildcard — must come AFTER all static routes) ─────────────
@router.get("/{roadmap_id}")
async def get_roadmap(roadmap_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(roadmap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Protocol ID format")
        
    roadmap = await db.roadmaps.find_one({"_id": oid, "patient_email": current_user.get("sub")})
    if not roadmap:
        raise HTTPException(status_code=404, detail="Protocol not found")
        
    return {
        "id": str(roadmap["_id"]),
        "has_roadmap": True, 
        "roadmap": roadmap["plan"], 
        "disease": roadmap["disease"],
        "duration_days": roadmap.get("duration_days", 7),
        "created_at": roadmap["created_at"].isoformat() if hasattr(roadmap.get("created_at"), "isoformat") else str(roadmap.get("created_at", ""))
    }


@router.post("/{roadmap_id}/tasks/complete")
async def complete_roadmap_task(
    roadmap_id: str,
    payload: TaskCompletionRequest,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    try:
        oid = ObjectId(roadmap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Protocol ID format")
        
    roadmap = await db.roadmaps.find_one({"_id": oid, "patient_email": current_user.get("sub")})
    if not roadmap:
        raise HTTPException(status_code=404, detail="Protocol not found")

    plan = roadmap.get("plan", {})
    routines = plan.get("routines", [])
    
    task_found = False
    
    for routine in routines:
        if routine.get("day") == payload.day:
            for task in routine.get("tasks", []):
                if task.get("task_id") == payload.task_id:
                    task["is_completed"] = True
                    task["completed_at"] = datetime.utcnow().isoformat()
                    task_found = True
                    break
        if task_found:
            break
            
    if not task_found:
        raise HTTPException(status_code=404, detail="Task identifier not found in protocol")

    await db.roadmaps.update_one(
        {"_id": oid},
        {"$set": {"plan": plan, "last_updated": datetime.utcnow()}}
    )

    return {"status": "success", "message": "Task confirmed", "plan": plan}


@router.post("/{roadmap_id}/modify")
async def modify_roadmap_state(
    roadmap_id: str,
    req: RoadmapModifyRequest,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    try:
        oid = ObjectId(roadmap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Protocol ID format")
        
    email = current_user.get("sub")
    
    roadmap = await db.roadmaps.find_one({"_id": oid, "patient_email": email})
    if not roadmap:
        raise HTTPException(status_code=404, detail="Protocol not found")
        
    user_data = await db.users.find_one({"email": email})
    patient_context_str = (
        f"Age: {user_data.get('age', 'Unknown')}\n"
        f"Gender: {user_data.get('gender', 'Unknown')}\n"
        f"Pre-existing Conditions: {', '.join(user_data.get('pre_existing_conditions', [])) or 'None'}\n"
        f"Allergies: {', '.join(user_data.get('allergies', [])) or 'None'}\n"
    ) if user_data else "No additional patient context found."
    
    current_plan = roadmap.get("plan", {})
    
    updated_plan = modify_roadmap_plan(
        current_plan=current_plan, 
        requested_changes=req.requested_changes, 
        patient_context=patient_context_str
    )
    
    await db.roadmaps.update_one(
        {"_id": oid},
        {
            "$set": {
                "plan": updated_plan, 
                "last_updated": datetime.utcnow()
            },
            "$push": {
                "modification_history": {
                    "requested_changes": req.requested_changes,
                    "timestamp": datetime.utcnow()
                }
            }
        }
    )

    return {
        "status": "success", 
        "message": "Protocol successfully mutated.", 
        "plan": updated_plan
    }


@router.delete("/{roadmap_id}")
async def delete_roadmap(roadmap_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(roadmap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Protocol ID format")
    
    result = await db.roadmaps.delete_one({"_id": oid, "patient_email": current_user.get("sub")})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Protocol not found")
    
    return {"status": "success", "message": "Protocol deleted"}