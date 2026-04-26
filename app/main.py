from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.database import connect_to_mongo, close_mongo_connection
from app.routes import test
from app.routes import auth
from app.routes import chat
from app.routes import clinical
from app.routes import patients
from app.routes import users
from app.routes import notes
from app.routes import schedule
from app.routes import vision
from app.routes import voice
from app.routes import roadmap

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await connect_to_mongo()
    yield
    # Shutdown logic
    await close_mongo_connection()

app = FastAPI(title="HealthSync AI", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your routers
app.include_router(test.router, prefix="/api/test")
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(clinical.router, prefix="/api/clinical", tags=["Clinical"])
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(roadmap.router, prefix="/api/roadmap", tags=["Roadmap"])
app.include_router(notes.router, prefix="/api/notes", tags=["Notes"])
app.include_router(schedule.router, prefix="/api/schedule", tags=["Schedule"])
app.include_router(vision.router, prefix="/api/vision", tags=["Vision"])
app.include_router(voice.router, prefix="/api/voice", tags=["Voice"])

from app.routes import images
app.include_router(images.router, prefix="/api/images", tags=["Images"])

@app.get("/")
async def root():
    return {"message": "API is live"}