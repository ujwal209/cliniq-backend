import asyncio
import os
import json
from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
from app.services.fitness_agent import search_fitness_image, search_workout_videos

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "healthsync")

explore_configs = [
    {
        "title": "Full Body Blast",
        "goal": "Strength",
        "level": "Intermediate",
        "image_query": "full body workout gym",
        "summary": "A comprehensive routine targeting all major muscle groups to build functional strength and balance.",
        "diet_suggestions": [
            "Consume 1.6g-2.2g of protein per kg of body weight.",
            "Stay hydrated with at least 3 liters of water daily.",
            "Post-workout meal should include complex carbs and fast-absorbing protein."
        ],
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
        "image_query": "leg day squats heavy",
        "summary": "Focus on the lower body's largest muscle groups for maximum metabolic impact and explosive power.",
        "diet_suggestions": [
            "Increase caloric intake by 200-300 kcal on leg days.",
            "Focus on potassium-rich foods to prevent muscle cramps.",
            "Creatine monohydrate can help with explosive power."
        ],
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
        "image_query": "abs core workout six pack",
        "summary": "Build a rock-solid midsection with this high-repetition core circuit designed for stability.",
        "diet_suggestions": [
            "Reduce sodium intake to minimize bloating.",
            "Fiber-rich foods help maintain a lean midsection.",
            "Focus on anti-inflammatory fats like Omega-3s."
        ],
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
        "image_query": "upper body chest triceps workout",
        "summary": "Target the chest, shoulders, and triceps with high-volume movements for that classic aesthetic look.",
        "diet_suggestions": [
            "Prioritize protein within 30 minutes post-workout.",
            "Stay away from sugary processed snacks.",
            "Good quality sleep is essential for muscle repair."
        ],
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
        "image_query": "hiit cardio workout sweat",
        "summary": "Maximum intensity interval training designed to spike your heart rate and torch calories.",
        "diet_suggestions": [
            "Try training in a fasted state for maximum fat oxidation.",
            "Black coffee can boost metabolic rate during HIIT.",
            "Stay consistent with your caloric deficit."
        ],
        "exercises": [
            {"name": "Burpees", "sets": "4", "reps": "10", "rest": "30s", "instructions": "Explosive movement.", "video_query": "how to do burpees"},
            {"name": "Mountain Climbers", "sets": "4", "reps": "30s", "rest": "30s", "instructions": "Fast pace.", "video_query": "mountain climbers exercise"},
            {"name": "Jumping Jacks", "sets": "4", "reps": "45s", "rest": "15s", "instructions": "Maintain high intensity.", "video_query": "proper jumping jacks"}
        ]
    }
]

async def seed():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    print("Clearing existing explore workouts...")
    await db.explore_fitness.delete_many({})
    
    print("Generating new explore workouts with images and videos...")
    
    final_workouts = []
    for config in explore_configs:
        print(f"Processing: {config['title']}")
        
        # Search for banner image
        banner_url = search_fitness_image(config["image_query"])
        
        # Search for videos for each exercise
        for ex in config["exercises"]:
            ex["videos"] = search_workout_videos(ex["video_query"])
            
        workout_doc = {
            "title": config["title"],
            "goal": config["goal"],
            "level": config["level"],
            "banner_image": banner_url,
            "plan": {
                "title": config["title"],
                "summary": config["summary"],
                "diet_suggestions": config["diet_suggestions"],
                "workouts": [
                    {
                        "day": "Day 1: " + config["title"],
                        "focus": config["goal"],
                        "exercises": config["exercises"]
                    }
                ]
            }
        }
        final_workouts.append(workout_doc)
        
    if final_workouts:
        result = await db.explore_fitness.insert_many(final_workouts)
        print(f"Successfully seeded {len(result.inserted_ids)} explore workouts.")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(seed())
