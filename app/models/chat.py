from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime
import uuid

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    feedback: Optional[Literal["like", "dislike"]] = None  # The ChatGPT thumbs up/down
    sources: Optional[List[dict]] = None

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_email: str
    messages: List[ChatMessage] = []
    is_archived: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)