from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class FitnessPlanGenerateRequest(BaseModel):
    goal: str = Field(..., description="Fitness goal, e.g., 'Weight Loss', 'Muscle Gain', 'Endurance'")
    level: str = Field(..., description="Fitness level: 'Beginner', 'Intermediate', 'Advanced'")
    preferences: str = Field(..., description="User preferences, e.g., 'Home workout', 'Gym', 'No equipment'")
    duration_weeks: int = Field(default=4, description="Duration of the plan in weeks")

class ExerciseItem(BaseModel):
    name: str = Field(..., description="Name of the exercise")
    sets: str = Field(..., description="Number of sets")
    reps: str = Field(..., description="Number of reps or duration")
    rest: str = Field(..., description="Rest time between sets")
    instructions: str = Field(..., description="Brief instructions on how to perform the exercise")
    video_query: str = Field(..., description="Query for searching a workout video")

class WorkoutDay(BaseModel):
    day: str = Field(..., description="Day name, e.g., 'Day 1: Full Body'")
    focus: str = Field(..., description="Main focus of the workout")
    exercises: List[ExerciseItem] = Field(..., description="List of exercises for the day")

class FitnessPlanData(BaseModel):
    title: str = Field(..., description="Title of the fitness plan")
    summary: str = Field(..., description="Brief summary of the plan and its benefits")
    diet_suggestions: List[str] = Field(..., description="Dietary tips to complement the fitness plan")
    workouts: List[WorkoutDay] = Field(..., description="Weekly workout routine")

class FitnessPlanDocument(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    patient_email: str
    goal: str
    level: str
    preferences: str
    duration_weeks: int
    plan: FitnessPlanData
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "active"
