import pytest
from app.chatbot import ChatBot
from app.database import Database


def test_arabic_question():
    """Test Arabic question"""
    bot = ChatBot()
    response = bot.ask("ما هو مأمن؟")
    assert response["status"] == True
    assert "language" in response
    assert response["language"] == "ar"


def test_english_question():
    """Test English question"""
    bot = ChatBot()
    response = bot.ask("What is Ma'man?")
    assert response["status"] == True
    assert response["language"] == "en"


def test_unknown_question():
    """Test unknown question"""
    bot = ChatBot()
    response = bot.ask("سؤال غير موجود")
    assert response["status"] == False
    assert "لم أتمكن" in response["answer"] or "Sorry" in response["answer"]


def test_session_id():
    """Test session ID generation"""
    bot = ChatBot()
    response = bot.ask("Hello")
    assert "session_id" in response
    assert response["session_id"] is not None


def test_database():
    """Test database connection"""
    db = Database()
    stats = db.statistics()
    assert "faq" in stats
    assert "unknown_questions" in stats
    assert "chat_logs" in stats
    