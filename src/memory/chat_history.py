from langchain.memory import ConversationBufferWindowMemory
from typing import Dict
from src.config.settings import settings  # ← Add this!
import uuid

session_store: Dict[str, ConversationBufferWindowMemory] = {}

def get_or_create_session(session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in session_store:
        session_store[session_id] = ConversationBufferWindowMemory(
            k=settings.MAX_HISTORY,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return session_id, session_store[session_id]

def clear_session(session_id: str):
    if session_id in session_store:
        del session_store[session_id]
        return True
    return False