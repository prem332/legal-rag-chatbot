from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.rag.pipeline import query_rag
from src.memory.chat_history import clear_session
from typing import Optional

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ClearRequest(BaseModel):
    session_id: str

@router.post("/query")
async def chat_query(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    return query_rag(request.question, request.session_id)

@router.post("/clear")
async def clear_chat(request: ClearRequest):
    success = clear_session(request.session_id)
    return {
        "cleared": success, 
        "session_id": request.session_id
    }