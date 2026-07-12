from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel
from app.core.preprocessing import TextPreprocessor
import uuid
import time

from app.core.chatbot import ChatBot
from app.core.embeddings import get_embedding_engine
from app.models.database import Database

router = APIRouter(tags=["Chat"])

_bot = None
db = Database()

def get_bot() -> ChatBot:
    """Get or create ChatBot instance"""
    global _bot
    if _bot is None:
        _bot = ChatBot()
    return _bot

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None



@router.get("/session/{session_id}/history")
def get_session_history(request: Request, session_id: str):
    """Get chat history for a session"""
    bot = get_bot()
    history = bot.get_session_history(session_id)
    
    return {
        "success": True,
        "session_id": session_id,
        "count": len(history),
        "data": history
    }

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
    
    # Check if model is still loading
    if bot.engine.is_loading():
        language = TextPreprocessor.detect_language(chat_request.question)
        msg_ar = "⏳ جارٍ تحميل النموذج، يرجى الانتظار لبضع ثوانٍ ثم المحاولة مرة أخرى."
        msg_en = "⏳ Model is loading, please wait a few seconds and try again."
        
        return {
            "success": False,
            "answer": msg_ar if language == "ar" else msg_en,
            "similarity": 0,
            "language": language,
            "session_id": chat_request.session_id or str(uuid.uuid4()),
            "response_time": 0
        }
    
    # Check if model is ready
    if not bot.engine.is_ready():
        language = TextPreprocessor.detect_language(chat_request.question)
        msg_ar = "❌ عذراً، النموذج غير متاح حالياً. يرجى المحاولة لاحقاً."
        msg_en = "❌ Sorry, the model is currently unavailable. Please try again later."
        
        return {
            "success": False,
            "answer": msg_ar if language == "ar" else msg_en,
            "similarity": 0,
            "language": language,
            "session_id": chat_request.session_id or str(uuid.uuid4()),
            "response_time": 0
        }
    
    # Process normally
    response = bot.ask(
        question=chat_request.question,
        session_id=chat_request.session_id
    )
    
    return response
