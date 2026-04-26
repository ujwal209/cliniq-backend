import os
import requests
import io
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.database import get_db
import threading

_gemini_keys_raw = os.getenv("GEMINI_API_KEYS", "")
GLOBAL_GEMINI_KEYS = [k.strip() for k in _gemini_keys_raw.split(",") if k.strip()]
_gemini_lock = threading.Lock()
_gemini_index = 0

def get_rotated_gemini_keys():
    global _gemini_index
    if not GLOBAL_GEMINI_KEYS: return []
    with _gemini_lock:
        start = _gemini_index
        _gemini_index = (_gemini_index + 1) % len(GLOBAL_GEMINI_KEYS)
    return GLOBAL_GEMINI_KEYS[start:] + GLOBAL_GEMINI_KEYS[:start]

def get_embedding_model():
    keys = get_rotated_gemini_keys()
    key = keys[0] if keys else os.getenv("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=key)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def index_document(doc_url: str, session_id: str):
    """Download doc, extract text, embed, and store in MongoDB for the session."""
    try:
        resp = requests.get(doc_url, timeout=15)
        resp.raise_for_status()
        file_bytes = resp.content
    except Exception as e:
        raise ValueError(f"Failed to download document: {e}")

    text = ""
    filename = doc_url.split("/")[-1].lower()

    if "pdf" in filename or doc_url.lower().endswith(".pdf"):
        import fitz  # PyMuPDF — lazy import
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
    elif "docx" in filename or doc_url.lower().endswith(".docx"):
        import docx  # python-docx — lazy import
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        # Fallback to plain text decoding
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Unsupported document format or unable to decode text.")

    if not text.strip():
        raise ValueError("No extractable text found in document.")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    if not chunks:
        return 0

    embedder = get_embedding_model()
    embeddings = embedder.embed_documents(chunks)

    db = get_db()
    docs_to_insert = []
    for chunk, emb in zip(chunks, embeddings):
        docs_to_insert.append({
            "session_id": session_id,
            "text": chunk,
            "embedding": emb,
            "source": doc_url
        })

    if docs_to_insert:
        await db.document_embeddings.insert_many(docs_to_insert)

    return len(docs_to_insert)

def search_session_documents(query: str, session_id: str, top_k: int = 3) -> str:
    """Vector search inside MongoDB using in-memory cosine similarity for zero-setup RAG."""
    from pymongo import MongoClient
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME", "healthsync_db")]
    
    try:
        # Fetch all chunks for this session
        chunks = list(db.document_embeddings.find({"session_id": session_id}).limit(1000))
        
        if not chunks:
            return "No documents have been uploaded to this session yet."

        embedder = get_embedding_model()
        query_emb = embedder.embed_query(query)

        scored_chunks = []
        for c in chunks:
            score = cosine_similarity(query_emb, c["embedding"])
            scored_chunks.append((score, c["text"], c["source"]))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        top_results = scored_chunks[:top_k]
        
        context = []
        for score, text, source in top_results:
            if score > 0.6:  # Relevance threshold
                context.append(f"[Source: {source}]\n...{text}...")
                
        if not context:
            return "No highly relevant information found in the uploaded documents."
            
        return "\n\n---\n\n".join(context)
    finally:
        client.close()
