import os
import itertools
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# 1. Parse the keys from Kashyap
gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

if not gemini_keys or not groq_keys:
    raise ValueError("Bro, make sure both GEMINI_API_KEYS and GROQ_API_KEYS are in your .env!")

# 2. Build the lists of LLMs
# Primary Priority: Gemini 2.5 Flash
gemini_llms = [
    ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=key, 
        temperature=0.2
    ) for key in gemini_keys
]

# Versatile Fallback: Groq
groq_llms = [
    ChatGroq(
        temperature=0.2, 
        model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
        groq_api_key=key
    ) for key in groq_keys
]

# 3. Create global iterators for Round-Robin
# These stay in memory and rotate to the next LLM on every call
gemini_cycle = itertools.cycle(gemini_llms)
groq_cycle = itertools.cycle(groq_llms)

def get_fallback_llm():
    """
    Grabs the NEXT Gemini model in the cycle, backed up by the NEXT Groq model.
    Perfect load balancing, bro.
    """
    # Grab the next LLM in the rotation
    primary_llm = next(gemini_cycle)
    fallback_llm = next(groq_cycle)
    
    # Chain them using LangChain's native fallback feature
    robust_llm = primary_llm.with_fallbacks([fallback_llm])
    
    return robust_llm