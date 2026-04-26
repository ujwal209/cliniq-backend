"""
Vision Analysis Route — Uses Gemini 1.5 Pro for X-ray / medical image and PDF analysis.
Accepts a Cloudinary URL, fetches the file, and returns clinical analysis.
"""
import os
import requests
import base64
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from app.core.deps import get_current_user
import logging

logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter()

import random
import threading

# Setup global round-robin keys
gemini_keys_raw = os.getenv("GEMINI_API_KEYS", "")
GLOBAL_KEYS = [k.strip() for k in gemini_keys_raw.split(",") if k.strip()]

_key_lock = threading.Lock()
_key_index = 0

def get_round_robin_keys():
    """Returns all keys but rotates the starting index for perfect load distribution."""
    global _key_index
    if not GLOBAL_KEYS:
        return []
    with _key_lock:
        start = _key_index
        _key_index = (_key_index + 1) % len(GLOBAL_KEYS)
    return GLOBAL_KEYS[start:] + GLOBAL_KEYS[:start]

class VisionRequest(BaseModel):
    image_url: str
    prompt: Optional[str] = "Analyze this medical image. Identify any abnormalities, structures, or findings. Provide a structured clinical assessment."


class PDFAnalysisRequest(BaseModel):
    pdf_url: str
    prompt: Optional[str] = "Analyze this medical document. Extract key findings, diagnoses, medications, and recommendations."

def fetch_and_encode(url: str) -> tuple[bytes, str]:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "")
        return response.content, mime_type
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch file from URL: {e}")

@router.post("/analyze-image")
async def analyze_image(
    req: VisionRequest,
    current_user: dict = Depends(get_current_user)
):
    keys = get_round_robin_keys()
    if not keys:
        logger.error("GEMINI_API_KEY is not configured.")
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY configured.")

    file_bytes, mime_type = fetch_and_encode(req.image_url)
    if not mime_type:
        mime_type = "image/jpeg"

    system_prompt = (
        "You are HealthSync Vision, an expert radiologist AI assistant. "
        "When analyzing medical images:\n"
        "1. Identify the type of image (X-ray, CT, MRI, ultrasound, etc.)\n"
        "2. Describe anatomical structures visible\n"
        "3. Note any abnormalities or pathological findings\n"
        "4. Provide a structured assessment with headers\n"
        "5. Use markdown formatting\n"
        "6. Include severity assessment if applicable\n"
        "IMPORTANT: Always include a disclaimer that this is AI-assisted and requires physician confirmation."
    )

    prompt = system_prompt + "\n\nUser request: " + req.prompt
    b64_data = base64.b64encode(file_bytes).decode('utf-8')
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}}
            ]
        }]
    }

    # Priority list of models to fallback across
    models_priority = [
        "gemini-2.5-flash",
        "gemini-3.1-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro"
    ]

    last_err = None
    
    for model in models_priority:
        for key in keys:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30
                )
                if resp.status_code == 429:
                    logger.warning(f"Model {model} on Key {key[:8]} hit 429 limit, trying next combination...")
                    last_err = resp.text
                    continue
                    
                resp.raise_for_status()
                
                analysis_text = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not analysis_text:
                    last_err = "Empty response or safety block"
                    continue

                return {
                    "analysis": analysis_text,
                    "model": model,
                    "image_url": req.image_url
                }
            except Exception as e:
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    last_err = e.response.text
                    continue
                last_err = str(e)
                # On non-429 errors (like 400 Bad Request), break to try next model or fail
                break
            
    raise HTTPException(status_code=500, detail=f"Image analysis failed on all keys. Last error: {last_err}")


@router.post("/analyze-pdf")
async def analyze_pdf(
    req: PDFAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    keys = get_round_robin_keys()
    if not keys:
        logger.error("GEMINI_API_KEY is not configured.")
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY configured.")

    file_bytes, mime_type = fetch_and_encode(req.pdf_url)
    if not mime_type or "pdf" not in mime_type.lower():
        mime_type = "application/pdf"

    system_prompt = (
        "You are HealthSync Document Analyst, an expert medical document reader. "
        "When analyzing medical documents:\n"
        "1. Extract patient information if available\n"
        "2. Identify diagnoses, procedures, and medications\n"
        "3. Summarize key findings\n"
        "4. Note any critical values or urgent findings\n"
        "5. Use structured markdown formatting\n"
        "IMPORTANT: Always state this is AI-assisted analysis."
    )

    prompt = system_prompt + "\n\nUser request: " + req.prompt
    b64_data = base64.b64encode(file_bytes).decode('utf-8')
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}}
            ]
        }]
    }

    # Priority list of models to fallback across
    models_priority = [
        "gemini-2.5-flash",
        "gemini-3.1-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro"
    ]

    last_err = None
    
    for model in models_priority:
        for key in keys:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30
                )
                if resp.status_code == 429:
                    logger.warning(f"Model {model} on Key {key[:8]} hit 429 limit, trying next combination...")
                    last_err = resp.text
                    continue
                    
                resp.raise_for_status()
                
                analysis_text = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not analysis_text:
                    last_err = "Empty response or safety block"
                    continue

                return {
                    "analysis": analysis_text,
                    "model": model,
                    "pdf_url": req.pdf_url
                }
            except Exception as e:
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    last_err = e.response.text
                    continue
                last_err = str(e)
                # On non-429 errors (like 400 Bad Request), break to try next model or fail
                break
            
    raise HTTPException(status_code=500, detail=f"PDF analysis failed on all keys. Last error: {last_err}")
