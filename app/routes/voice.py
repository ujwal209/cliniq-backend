import os
import requests
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from app.core.deps import get_current_user

load_dotenv()

router = APIRouter()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

class TTSRequest(BaseModel):
    text: str

@router.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Convert audio file to text using Deepgram."""
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured.")

    try:
        audio_data = await file.read()
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": file.content_type or "audio/webm"
        }
        
        # Deepgram Nova-2 model
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
        
        response = requests.post(url, headers=headers, data=audio_data, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        transcript = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

@router.post("/tts")
async def text_to_speech(
    req: TTSRequest,
    current_user: dict = Depends(get_current_user)
):
    """Convert text to speech audio URL/base64 using Deepgram Aura."""
    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY not set")
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured.")

    try:
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        
        text_to_speak = req.text
        if len(text_to_speak) > 1900:
            import os
            groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
            if groq_keys:
                print("Text too long. Summarizing with Groq for TTS...")
                groq_key = groq_keys[0]
                try:
                    summary_res = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={
                            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                            "messages": [
                                {
                                    "role": "system", 
                                    "content": "You are a concise medical voice assistant. Summarize the following clinical response to be spoken aloud. Keep it extremely conversational, omit bullet points/markdown, and MUST be under 1800 characters."
                                },
                                {"role": "user", "content": text_to_speak}
                            ],
                            "max_tokens": 400
                        },
                        timeout=15
                    )
                    if summary_res.ok:
                        text_to_speak = summary_res.json()["choices"][0]["message"]["content"].strip()
                        print(f"Summarized text length: {len(text_to_speak)}")
                    else:
                        text_to_speak = text_to_speak[:1900]
                except Exception as e:
                    print(f"Groq summarization failed: {e}")
                    text_to_speak = text_to_speak[:1900]
            else:
                text_to_speak = text_to_speak[:1900]

        payload = {"text": text_to_speak}
        print(f"Sending TTS request to Deepgram. Text length: {len(text_to_speak)}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if not response.ok:
            print(f"Deepgram TTS Error: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        import base64
        audio_b64 = base64.b64encode(response.content).decode("utf-8")
        
        return {"audio_base64": audio_b64, "content_type": "audio/mp3"}
    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")
