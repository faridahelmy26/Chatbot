import pytest
from app.models.database import Database


def test_arabic_question(bot):
    """Test Arabic question"""
    response = bot.ask("ما هو مأمن؟")
    assert response["success"] == True
    assert "language" in response
    assert response["language"] == "ar"


def test_english_question(bot):
    """Test English question"""
    response = bot.ask("What is Ma'man?")
    assert response["success"] == True
    assert response["language"] == "en"


def test_unknown_question(bot):
    """Test unknown question"""
    response = bot.ask("سؤال غير موجود")
    assert response["success"] == False
    assert "لم أتمكن" in response["answer"] or "Sorry" in response["answer"]


def test_session_id(bot):
    """Test session ID generation"""
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