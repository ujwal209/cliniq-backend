from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

db_manager = MongoDB()

async def connect_to_mongo():
    """Create database connection pool."""
    db_manager.client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db_manager.db = db_manager.client[os.getenv("DB_NAME", "healthsync_db")]
    print("Connected to MongoDB Atlas! 🍃")

async def close_mongo_connection():
    """Close database connection pool."""
    db_manager.client.close()
    print("MongoDB connection closed.")

def get_db():
    """Dependency to get the database instance."""
    return db_manager.db