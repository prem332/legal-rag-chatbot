from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import chat, documents

app = FastAPI(
    title="Legal RAG Chatbot API",
    description="Enterprise Legal RAG with Citations & Memory",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(
    documents.router, 
    prefix="/api/v1/documents", 
    tags=["Documents"]
)
app.include_router(
    chat.router, 
    prefix="/api/v1/chat", 
    tags=["Chat"]
)

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "service": "Legal RAG Chatbot"
    }