from fastapi import APIRouter, Depends, HTTPException
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.fitness import FitnessPlanGenerateRequest
from app.services.fitness_agent import generate_fitness_plan, get_explore_workouts
from datetime import datetime
from bson.objectid import ObjectId

router = APIRouter()
@router.get("/explore")
async def explore_workouts(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db.explore_fitness.find({})
    workouts = await cursor.to_list(length=10)
    for w in workouts:
        w["id"] = str(w["_id"])
        del w["_id"]
    return workouts

@router.get("/explore/{explore_id}")
async def get_explore_workout(
    explore_id: str,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    try:
        oid = ObjectId(explore_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Explore ID format")
        
    workout = await db.explore_fitness.find_one({"_id": oid})
    if not workout:
        raise HTTPException(status_code=404, detail="Explore workout not found")
        
    workout["id"] = str(workout["_id"])
    del workout["_id"]
    return workout

@router.post("/explore/{explore_id}/add")
async def add_explore_to_my_plans(
    explore_id: str,
    current_user: dict = Depends(get_current_user)
):
    db = get_db()
    try:
        oid = ObjectId(explore_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Explore ID format")
        
    explore_plan = await db.explore_fitness.find_one({"_id": oid})
    if not explore_plan:
        raise HTTPException(status_code=404, detail="Explore plan not found")
        
    email = current_user.get("sub")
    
    # Copy the plan to the user's fitness_plans collection
    new_plan = {
        "patient_email": email,
        "goal": explore_plan["goal"],
        "level": explore_plan["level"],
        "preferences": "Added from Explore",
        "duration_weeks": 4,
        "plan": explore_plan["plan"],
        "banner_image": explore_plan.get("banner_image"),
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    result = await db.fitness_plans.insert_one(new_plan)
    return {"status": "success", "id": str(result.inserted_id)}

@router.post("/explore/{explore_id}/video/refresh")
async def refresh_workout_video(
    explore_id: str,
    exercise_name: str,
    current_user: dict = Depends(get_current_user)
):
    from app.services.fitness_agent import search_workout_videos
    db = get_db()
    try:
        oid = ObjectId(explore_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Explore ID format")
        
    workout_doc = await db.explore_fitness.find_one({"_id": oid})
    if not workout_doc:
        raise HTTPException(status_code=404, detail="Explore workout not found")
        
    # Find the exercise and refresh its videos
    updated = False
    new_video_list = []
    if "plan" in workout_doc and "workouts" in workout_doc["plan"]:
        for day in workout_doc["plan"]["workouts"]:
            for ex in day.get("exercises", []):
                if ex["name"] == exercise_name:
                    # Slightly vary the query to get different results
                    query = ex.get("video_query", ex["name"] + " proper form tutorial")
                    new_video_list = search_workout_videos(query)
                    ex["videos"] = new_video_list
                    updated = True
                    break
            if updated: break
            
    if updated:
        await db.explore_fitness.replace_one({"_id": oid}, workout_doc)
        return {"status": "success", "videos": new_video_list}
    
    raise HTTPException(status_code=404, detail="Exercise not found in workout")

@router.post("/generate")
async def generate_patient_fitness_plan(
    req: FitnessPlanGenerateRequest,
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
    
    plan = generate_fitness_plan(
        goal=req.goal, 
        level=req.level, 
        preferences=req.preferences, 
        patient_context=patient_context_str,
        duration_weeks=req.duration_weeks
    )
    
    fitness_doc = {
        "patient_email": email,
        "goal": req.goal,
        "level": req.level,
        "preferences": req.preferences,
        "duration_weeks": req.duration_weeks,
        "plan": plan,
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    result = await db.fitness_plans.insert_one(fitness_doc)
    return {"id": str(result.inserted_id), "plan": plan}

@router.get("/all")
async def get_all_fitness_plans(current_user: dict = Depends(get_current_user)):
    db = get_db()
    email = current_user.get("sub")
    
    cursor = db.fitness_plans.find({"patient_email": email}).sort("created_at", -1)
    plans = await cursor.to_list(length=100)
    
    formatted = []
    for p in plans:
        formatted.append({
            "id": str(p["_id"]),
            "goal": p["goal"],
            "level": p["level"],
            "created_at": p["created_at"].isoformat() if hasattr(p.get("created_at"), "isoformat") else str(p.get("created_at", "")),
            "title": p.get("plan", {}).get("title", "Fitness Plan"),
            "banner_image": p.get("banner_image"),
            "status": p.get("status", "active")
        })
        
    return {"plans": formatted}

@router.get("/{plan_id}")
async def get_fitness_plan(plan_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(plan_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Plan ID format")
        
    plan = await db.fitness_plans.find_one({"_id": oid, "patient_email": current_user.get("sub")})
    if not plan:
        raise HTTPException(status_code=404, detail="Fitness plan not found")
        
    return {
        "id": str(plan["_id"]),
        "plan": plan["plan"], 
        "goal": plan["goal"],
        "level": plan["level"],
        "created_at": plan["created_at"].isoformat() if hasattr(plan.get("created_at"), "isoformat") else str(plan.get("created_at", ""))
    }

@router.delete("/{plan_id}")
async def delete_fitness_plan(plan_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(plan_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Plan ID format")
    
    result = await db.fitness_plans.delete_one({"_id": oid, "patient_email": current_user.get("sub")})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Fitness plan not found")
    
    return {"status": "success", "message": "Fitness plan deleted"}
