import json
from langchain_core.prompts import ChatPromptTemplate
from app.core.llm_factory import get_fallback_llm
from langchain_core.output_parsers import JsonOutputParser

async def generate_clinical_note(transcript: str) -> dict:
    llm = get_fallback_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI medical scribe. Read the following consultation transcript and generate a structured JSON containing two keys: 'soap_note' (which is an object with 'subjective', 'objective', 'assessment', 'plan' string keys) and 'billing_codes' (which is a list of relevant ICD-10 codes). Return ONLY JSON, no markdown formatting."),
        ("user", "{transcript}")
    ])
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({"transcript": transcript})

async def generate_patient_summary(history: list) -> list:
    llm = get_fallback_llm()
    history_text = "\n".join(history)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical AI assistant. Analyze the following medical history and extract exactly 3 bullet points summarizing the most critical active issues. Return a JSON list of strings, e.g. [\"Issue 1\", \"Issue 2\", \"Issue 3\"]."),
        ("user", "{history}")
    ])
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({"history": history_text})

async def generate_treatment_plan(diagnosis: str) -> dict:
    llm = get_fallback_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical AI. Create a draft treatment plan for the following diagnosis. Return a JSON object with 'diet', 'exercise', and 'follow_up_schedule' keys."),
        ("user", "Diagnosis: {diagnosis}")
    ])
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({"diagnosis": diagnosis})

async def extract_structured_terms(text: str) -> list:
    llm = get_fallback_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical natural language processing AI. Extract structured medical terms (symptoms, conditions) from the raw patient text. Return a JSON list of strings."),
        ("user", "{text}")
    ])
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({"text": text})

async def process_ocr(image_base64: str) -> dict:
    llm = get_fallback_llm()
    # In a real scenario, we would pass the image to Gemini multimodal
    # For now, we simulate extraction if we only have text fallback, or use the image payload.
    # We will assume a prompt that extracts lab values.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract text and structured lab values from the provided image base64. Return JSON with 'extracted_text' and 'structured_data' (dict).")
        # Note: Langchain chat models might need special message formats for images,
        # but for this MVP, we simulate or pass appropriately if supported.
    ])
    # Placeholder for actual multimodal call
    return {"extracted_text": "Simulated OCR text", "structured_data": {"Hemoglobin": "14.2 g/dL"}}

async def analyze_diagnostic_image(image_base64: str) -> dict:
    # Placeholder for computer vision model
    return {"anomalies_detected": ["Slight shadow in lower lobe"], "analysis_notes": "Recommend further review."}

async def check_drug_interactions(medications: list) -> str:
    llm = get_fallback_llm()
    meds_str = ", ".join(medications)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical pharmacist AI. Check the following list of medications for severe interactions. Return a short summary of warnings or 'No severe interactions detected.'"),
        ("user", "Medications: {medications}")
    ])
    chain = prompt | llm
    res = await chain.ainvoke({"medications": meds_str})
    return res.content

async def redact_pii(text: str) -> str:
    llm = get_fallback_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a redaction AI. Mask all Personally Identifiable Information (Names, SSNs, exact addresses) in the text to ensure HIPAA compliance. Replace with [REDACTED]. Return only the redacted text."),
        ("user", "{text}")
    ])
    chain = prompt | llm
    res = await chain.ainvoke({"text": text})
    return res.content

async def vector_search(query: str, documents: list) -> list:
    llm = get_fallback_llm()
    docs_str = "\n".join([f"Doc {i}: {d}" for i, d in enumerate(documents)])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a semantic search AI. Find exact sentences from the provided documents that match the user's query. Return a JSON list of relevant sentences."),
        ("user", "Query: {query}\n\nDocuments:\n{documents}")
    ])
    chain = prompt | llm | JsonOutputParser()
    try:
        return await chain.ainvoke({"query": query, "documents": docs_str})
    except Exception:
        return []
