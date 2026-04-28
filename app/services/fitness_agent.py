import json
import os
import random
import requests
from groq import Groq
from typing import List, Dict
from app.models.fitness import FitnessPlanData

# Load API Keys helpers
def get_groq_client():
    groq_keys = os.getenv("GROQ_API_KEYS", "").split(",")
    key = random.choice([k.strip() for k in groq_keys if k.strip()])
    return Groq(api_key=key)

def get_serper_key():
    serper_keys = os.getenv("SERPER_API_KEYS", "").split(",")
    valid_keys = [k.strip() for k in serper_keys if k.strip()]
    if not valid_keys:
        # Fallback for search_fitness_image if keys are missing
        return None
    return random.choice(valid_keys)

def search_workout_videos(query: str) -> List[Dict]:
    """Search for workout videos using Serper API."""
    url = "https://google.serper.dev/videos"
    payload = json.dumps({"q": query})
    key = get_serper_key()
    if not key:
        return []
    headers = {
        'X-API-KEY': key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()
        return results.get("videos", [])[:3]  # Return top 3 videos
    except Exception as e:
        print(f"Serper error: {e}")
        return []

def search_fitness_image(query: str) -> str:
    """Search for a fitness banner image using Serper Image API."""
    url = "https://google.serper.dev/images"
    payload = json.dumps({"q": query})
    key = get_serper_key()
    if not key:
        return "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1470&auto=format&fit=crop"
    headers = {
        'X-API-KEY': key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()
        images = results.get("images", [])
        if images:
            return images[0].get("imageUrl")
    except Exception as e:
        print(f"Serper image error: {e}")
    return "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1470&auto=format&fit=crop"

def generate_fitness_plan(goal: str, level: str, preferences: str, patient_context: str, duration_weeks: int = 4) -> Dict:
    client = get_groq_client()
    
    prompt = f"""
    You are an expert Fitness Coach and Personal Trainer.
    Generate a highly personalized fitness plan for a patient with the following profile:
    
    PATIENT CONTEXT:
    {patient_context}
    
    FITNESS GOAL: {goal}
    FITNESS LEVEL: {level}
    PREFERENCES: {preferences}
    DURATION: {duration_weeks} weeks
    
    The response MUST be a JSON object strictly following this structure:
    {{
        "title": "Title of the plan",
        "summary": "3-5 sentences summary",
        "diet_suggestions": ["tip 1", "tip 2", ...],
        "workouts": [
            {{
                "day": "Day 1: [Workout Name]",
                "focus": "Focus area",
                "exercises": [
                    {{
                        "name": "Exercise Name",
                        "sets": "e.g. 3",
                        "reps": "e.g. 10-12",
                        "rest": "e.g. 60 seconds",
                        "instructions": "Brief coaching cue",
                        "video_query": "A specific search query for a YouTube video of this exercise"
                    }},
                    ...
                ]
            }},
            ... (provide 5-7 days of a weekly routine)
        ]
    }}
    
    Ensure exercises are safe given the patient's context (pre-existing conditions).
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional fitness coach assistant. You output strictly JSON."},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"}
    )
    
    plan_data = json.loads(chat_completion.choices[0].message.content)
    
    # Enrich with video results
    for day in plan_data.get("workouts", []):
        for exercise in day.get("exercises", []):
            exercise["videos"] = search_workout_videos(exercise["video_query"])
            
    return plan_data

def get_explore_workouts() -> List[Dict]:
    """Return 5 predefined workout types for the explore page."""
    explore_configs = [
        {
            "title": "Full Body Blast",
            "goal": "Strength",
            "level": "Intermediate",
            "exercises": [
                {"name": "Pushups", "sets": "3", "reps": "15-20", "rest": "60s", "instructions": "Keep your core tight and back straight.", "video_query": "proper pushup form"},
                {"name": "Bodyweight Squats", "sets": "3", "reps": "20", "rest": "60s", "instructions": "Drive through your heels.", "video_query": "bodyweight squat tutorial"},
                {"name": "Plank", "sets": "3", "reps": "45s", "rest": "45s", "instructions": "Maintain a straight line from head to heels.", "video_query": "how to do a perfect plank"}
            ]
        },
        {
            "title": "Leg Day Hero",
            "goal": "Muscle Gain",
            "level": "Advanced",
            "exercises": [
                {"name": "Lunges", "sets": "4", "reps": "12 each leg", "rest": "60s", "instructions": "Keep your torso upright.", "video_query": "alternating lunges exercise"},
                {"name": "Glute Bridges", "sets": "3", "reps": "15", "rest": "60s", "instructions": "Squeeze your glutes at the top.", "video_query": "glute bridge form"},
                {"name": "Calf Raises", "sets": "4", "reps": "20", "rest": "45s", "instructions": "Full range of motion.", "video_query": "standing calf raises"}
            ]
        },
        {
            "title": "Core Crusher",
            "goal": "Endurance",
            "level": "Beginner",
            "exercises": [
                {"name": "Crunches", "sets": "3", "reps": "15", "rest": "30s", "instructions": "Don't pull on your neck.", "video_query": "abdominal crunches technique"},
                {"name": "Leg Raises", "sets": "3", "reps": "12", "rest": "45s", "instructions": "Keep your lower back flat on the floor.", "video_query": "lying leg raises"},
                {"name": "Russian Twists", "sets": "3", "reps": "20 total", "rest": "45s", "instructions": "Rotate your entire torso.", "video_query": "russian twists exercise"}
            ]
        },
        {
            "title": "Upper Body Pump",
            "goal": "Muscle Gain",
            "level": "Intermediate",
            "exercises": [
                {"name": "Dips (Chair or Bench)", "sets": "3", "reps": "12", "rest": "60s", "instructions": "Keep your elbows tucked in.", "video_query": "bench dips tutorial"},
                {"name": "Diamond Pushups", "sets": "3", "reps": "10", "rest": "60s", "instructions": "Target your triceps.", "video_query": "diamond pushup form"},
                {"name": "Superman", "sets": "3", "reps": "12", "rest": "45s", "instructions": "Lift your chest and legs off the floor.", "video_query": "superman back exercise"}
            ]
        },
        {
            "title": "Fat Burn HIIT",
            "goal": "Weight Loss",
            "level": "Intermediate",
            "exercises": [
                {"name": "Burpees", "sets": "4", "reps": "10", "rest": "30s", "instructions": "Explosive movement.", "video_query": "how to do burpees"},
                {"name": "Mountain Climbers", "sets": "4", "reps": "30s", "rest": "30s", "instructions": "Fast pace.", "video_query": "mountain climbers exercise"},
                {"name": "Jumping Jacks", "sets": "4", "reps": "45s", "rest": "15s", "instructions": "Maintain high intensity.", "video_query": "proper jumping jacks"}
            ]
        }
    ]

    for workout in explore_configs:
        for exercise in workout["exercises"]:
            exercise["videos"] = search_workout_videos(exercise["video_query"])
            
    return explore_configs
