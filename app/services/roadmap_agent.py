import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import random
from app.models.roadmap import RoadmapData

def get_groq_key():
    keys = os.getenv("GROQ_API_KEYS", "").split(",")
    valid_keys = [k.strip() for k in keys if k.strip()]
    if not valid_keys:
        return ""
    return random.choice(valid_keys)

def generate_roadmap_plan(disease: str, goals: str, symptoms: str, patient_context: str, duration_days: int = 7) -> dict:
    key = get_groq_key()
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=key, temperature=0.2)
    
    # Use Pydantic Structured Output to guarantee the complex tracking schema
    structured_llm = llm.with_structured_output(RoadmapData)
    
    sys_prompt = """You are a board-certified clinical AI specialist creating a precise, evidence-based recovery protocol.

CRITICAL RULES — violating any of these is unacceptable:
1. MEDICATIONS: Always name the SPECIFIC drug (e.g., "Ibuprofen 400mg", "Paracetamol 500mg", "Amoxicillin 500mg"). NEVER write "Take prescribed medication". Include: drug name, dose, route (oral/topical), timing constraints (with food, 8hrs apart), and what it targets.
2. MEALS: Always list SPECIFIC ingredients with quantities (e.g., "2 boiled eggs, 1 slice whole-wheat toast, 200ml orange juice"). NEVER write "Eat a healthy meal". Include preparation method and clinical rationale.
3. EXERCISE: Always specify exact duration, reps/sets, intensity (e.g., "3 sets of 15 resistance band squats at moderate intensity"). Include the physiological benefit for the condition.
4. DAILY TIPS: Must reference the specific medications and activities of THAT day. Never give generic advice like "stay hydrated".
5. DESCRIPTIONS: Must be 3-5 detailed clinical sentences covering how, why, precautions, and expected outcome.
6. TASK IDs: Must be unique across all days, e.g., 'd1-morning-ibuprofen', 'd2-afternoon-walk'.
7. TIME BLOCKS: Distribute tasks across Morning, Afternoon, Evening, Night appropriately."""

    human_prompt = f"""
Patient Profile:
{patient_context}

Condition: {disease}
Current Symptoms: {symptoms}
Treatment Goals: {goals}
Protocol Duration: {duration_days} days

Generate a complete {duration_days}-day clinical protocol. Remember:
- Name every medication specifically with dose
- List every meal with exact ingredients and quantities  
- Specify every exercise with duration/reps
- Make each day's tip reference that day's specific treatments
"""
    
    try:
        response: RoadmapData = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ])
        return response.model_dump()
    except Exception as e:
        print(f"Protocol synthesis error: {e}")
        return {"summary": "Failed to synthesize protocol.", "duration_days": duration_days, "routines": []}

def modify_roadmap_plan(current_plan: dict, requested_changes: str, patient_context: str) -> dict:
    key = get_groq_key()
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=key, temperature=0.2)
    
    # Force strict schema adherence for the mutated state
    structured_llm = llm.with_structured_output(RoadmapData)
    
    sys_prompt = """You are an elite, enterprise-grade clinical AI orchestrator.
Your objective is to MODIFY an existing chronological recovery protocol based on new patient instructions.
You MUST keep the unaffected parts of the protocol exactly the same. Only mutate the tasks, meals, or schedules that directly align with the 'Requested Changes'.
Ensure time-based categorization (Morning, Afternoon, Evening, Night) and exact 'unsplash_keyword' tracking are preserved. 
Output strictly according to the defined schema."""

    human_prompt = f"""
Patient Profile Context:
{patient_context}

=== CURRENT PROTOCOL STATE ===
{json.dumps(current_plan, indent=2)}
==============================

Requested Changes: {requested_changes}

Synthesize the updated protocol.
"""
    
    try:
        response: RoadmapData = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ])
        return response.model_dump()
    except Exception as e:
        print(f"Protocol mutation error: {e}")
        return current_plan # Fallback to existing state if generation fails