from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class RoadmapGenerateRequest(BaseModel):
    disease: str = Field(..., description="The main disease or condition to generate a plan for.")
    goals: str = Field(..., description="The primary health goals of the patient.")
    symptoms: str = Field(..., description="The current symptoms the patient is experiencing.")
    duration_days: int = Field(default=7, description="Number of days for the protocol.")

class TaskItem(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task, e.g., 'd1-morning-ibuprofen'")
    time_of_day: str = Field(..., description="Chronological time block: 'Morning', 'Afternoon', 'Evening', or 'Night'")
    time_display: str = Field(..., description="Suggested exact time, e.g., '08:00 AM'")
    title: str = Field(
        ..., 
        description=(
            "Highly specific, actionable title. "
            "For Medication: include drug name and dose, e.g., 'Ibuprofen 400mg with food'. "
            "For Meal: include actual dish, e.g., 'Oatmeal with banana and honey'. "
            "For Exercise: include reps/duration, e.g., '20-min brisk walk'. "
            "NEVER use vague phrases like 'Take prescribed medication'."
        )
    )
    description: str = Field(
        ..., 
        description=(
            "Rich clinical instructions (3-5 sentences). "
            "For Medication: state exact drug name, dose, frequency, and what it treats (e.g., 'Take Ibuprofen 400mg orally with a full glass of water to reduce inflammation and fever. Do not exceed 1200mg/day. Avoid on empty stomach to prevent gastric irritation.'). "
            "For Meal: list exact ingredients, quantities, preparation method, and nutritional rationale (e.g., 'Prepare a bowl of rolled oats cooked with 200ml low-fat milk, topped with 1 sliced banana and 1 tsp honey. Rich in complex carbs and potassium to support recovery.'). "
            "For Exercise: describe exact movements, duration, sets/reps, intensity, and clinical benefit."
        )
    )
    category: str = Field(..., description="One of: 'Meal', 'Exercise', 'Medication', or 'Lifestyle'")
    unsplash_keyword: str = Field(..., description="A single, highly relevant noun for image search (e.g., 'ibuprofen', 'oatmeal', 'walking', 'water').")
    is_completed: bool = Field(default=False)
    completed_at: Optional[str] = None

class DailyRoutineItem(BaseModel):
    day: int = Field(..., description="The day integer, e.g., 1")
    daily_tip: str = Field(
        ..., 
        description=(
            "Specific clinical insight for the day (2-3 sentences). "
            "Should reference the day's specific medications or activities. "
            "NEVER generic advice like 'Stay hydrated'. "
            "Example: 'Day 3 focuses on inflammation control — the combination of Ibuprofen and turmeric-rich meals works synergistically. Monitor for any gastric discomfort from NSAIDs and take with food.'"
        )
    )
    tasks: List[TaskItem] = Field(..., description="Chronological tasks scheduled throughout the day.")

class RoadmapData(BaseModel):
    summary: str = Field(..., description="Detailed clinical summary of the protocol (3-5 sentences), mentioning key medications, dietary strategy, and exercise approach.")
    duration_days: int = Field(..., description="Total duration of the roadmap in days.")
    routines: List[DailyRoutineItem] = Field(..., description="Detailed day-by-day scheduled routines.")

class TaskCompletionRequest(BaseModel):
    day: int
    task_id: str

class RoadmapModifyRequest(BaseModel):
    requested_changes: str = Field(..., description="The natural language instructions from the user detailing what needs to be changed in the current protocol.")