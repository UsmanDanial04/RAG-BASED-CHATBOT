from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import logging
from dotenv import load_dotenv

from rag_engine import rag_engine

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG BASED CHATBOT")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allowed upload types
ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: list = []  # List of {"role": "user"|"model", "text": str}
    api_key: str = None
    model: str = "gemini-3.6-flash"
    system_instruction: str = None
    use_rag: bool = True   # Whether to retrieve context from knowledge base

class LoginRequest(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/api/login")
async def login(request: LoginRequest):
    valid_email    = os.getenv("LOGIN_EMAIL", "").strip()
    valid_password = os.getenv("LOGIN_PASSWORD", "").strip()
    if not valid_email or not valid_password:
        raise HTTPException(status_code=500, detail="Login credentials not configured on server.")
    if request.email.strip().lower() != valid_email.lower() or request.password != valid_password:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"status": "ok", "email": request.email}

@app.get("/login", response_class=HTMLResponse)
async def get_login():
    try:
        with open("login.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="login.html not found at project root")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found at project root")


# ------------------------------------------------------------------
# Document Management Endpoints
# ------------------------------------------------------------------
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a document into the RAG knowledge base."""
    # Validate file extension
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20 MB.")

    try:
        doc = rag_engine.add_document(
            filename=file.filename,
            raw_bytes=raw_bytes,
            file_type=file.content_type or "text/plain",
        )
        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunk_count": doc.chunk_count,
            "size_bytes": doc.size_bytes,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during document upload")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")


@app.get("/api/documents")
async def list_documents():
    """Return all documents currently in the knowledge base."""
    docs = rag_engine.list_documents()
    return {
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "file_type": d.file_type,
                "chunk_count": d.chunk_count,
                "size_bytes": d.size_bytes,
            }
            for d in docs
        ],
        "total": len(docs),
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    deleted = rag_engine.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "deleted", "doc_id": doc_id}


# ------------------------------------------------------------------
# Chat Endpoint (RAG-augmented)
# ------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Resolve API key
    api_key = (request.api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini API Key is missing. Please provide it in the Settings panel or configure it in the .env file.",
        )

    try:
        client = genai.Client(api_key=api_key)

        # ----------------------------------------------------------------
        # RAG: Retrieve relevant chunks if enabled and docs exist
        # ----------------------------------------------------------------
        rag_context = ""
        sources: list[dict] = []

        if request.use_rag and rag_engine.document_count() > 0:
            chunks = rag_engine.retrieve(request.message, k=5)
            # Only use chunks with reasonable relevance
            relevant = [c for c in chunks if c.score >= 0.25]
            if relevant:
                context_parts = []
                seen_sources: set[str] = set()
                for i, chunk in enumerate(relevant, 1):
                    context_parts.append(
                        f"[Source {i} — {chunk.filename}]\n{chunk.text}"
                    )
                    if chunk.doc_id not in seen_sources:
                        sources.append({
                            "doc_id": chunk.doc_id,
                            "filename": chunk.filename,
                            "score": chunk.score,
                        })
                        seen_sources.add(chunk.doc_id)

                rag_context = (
                    "Use the following retrieved document excerpts to answer the user's question. "
                    "If the answer isn't in the context, say so honestly and answer from general knowledge.\n\n"
                    + "\n\n---\n\n".join(context_parts)
                    + "\n\n---\n\nNow answer the user's question based on the above context."
                )

        # ----------------------------------------------------------------
        # Build conversation contents
        # ----------------------------------------------------------------
        contents = []
        for msg in request.history:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg.get("text", ""))])
            )

        # Augment user message with retrieved context (if any)
        user_text = request.message
        if rag_context:
            user_text = f"{rag_context}\n\nUser question: {request.message}"

        contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

        # ----------------------------------------------------------------
        # System instruction
        # ----------------------------------------------------------------
        config_kwargs = {}
        base_instruction = (request.system_instruction or "").strip()
        if rag_context:
            rag_note = "You are a RAG-powered assistant. Ground your answers in the provided document excerpts when relevant."
            system_text = f"{rag_note}\n\n{base_instruction}" if base_instruction else rag_note
        else:
            system_text = base_instruction

        if system_text:
            config_kwargs["system_instruction"] = system_text

        generate_config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # ----------------------------------------------------------------
        # Call Gemini
        # ----------------------------------------------------------------
        response = client.models.generate_content(
            model=request.model or "gemini-3.6-flash",
            contents=contents,
            config=generate_config,
        )

        return {
            "response": response.text,
            "model": request.model,
            "sources": sources,          # list of cited documents
            "rag_used": bool(sources),   # whether RAG context was injected
        }

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            error_msg = "Invalid Gemini API Key. Please verify your key in the settings panel."
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    print("\nGemini Echo RAG Chatbot is running at: http://127.0.0.1:8000\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
