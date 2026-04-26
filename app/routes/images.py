import os
import requests
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.core.deps import get_current_user

router = APIRouter()

SERPER_INDEX = 0

def get_serper_key():
    global SERPER_INDEX
    keys_str = os.getenv("SERPER_API_KEYS", "")
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not keys:
        return None
    # Round-robin load balancing
    key = keys[SERPER_INDEX % len(keys)]
    SERPER_INDEX += 1
    return key

@router.get("/search")
async def search_image(q: str): # We omit get_current_user to allow easy fetching, or we can keep it.
    key = get_serper_key()
    if not key:
        raise HTTPException(status_code=500, detail="Serper API keys not configured in backend")
        
    url = "https://google.serper.dev/images"
    payload = {"q": q}
    headers = {
        'X-API-KEY': key,
        'Content-Type': 'application/json'
    }
    
    def fetch():
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
        
    try:
        data = await asyncio.to_thread(fetch)
        images = data.get("images", [])
        if not images:
            return {"url": "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=400&q=80"}
            
        return {"url": images[0].get("imageUrl")}
    except Exception as e:
        print(f"Serper API error: {e}")
        return {"url": "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=400&q=80"}
