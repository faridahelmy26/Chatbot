from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel
import uuid
import time

from app.core.chatbot import ChatBot
from app.core.embeddings import get_embedding_engine
from app.core.preprocessing import TextPreprocessor
from app.models.database import Database

router = APIRouter(tags=["Chat"])

_bot = None
db = Database()

def get_bot() -> ChatBot:
    global _bot
    if _bot is None:
        _bot = ChatBot()
    return _bot

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

@router.post("/chat")
def chat(request: Request, chat_request: ChatRequest):
    # Validate question
    if not chat_request.question or len(chat_request.question.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="السؤال قصير جداً (أقل من 3 أحرف)"
        )
    
    # Get bot
    bot = get_bot()
    
    # Process directly (no model check needed)
    response = bot.ask(
        question=chat_request.question,
        session_id=chat_request.session_id
    )
    
    return response

@router.get("/session/{session_id}/history")
def get_session_history(request: Request, session_id: str):
    bot = get_bot()
    history = bot.get_session_history(session_id)
    
    return {
        "success": True,
        "session_id": session_id,
        "count": len(history),
        "data": history
    }
