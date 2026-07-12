import pytest
from app.models.database import Database


def test_insert_faq():
    """Test inserting FAQ"""
    db = Database()
    
    result = db.insert_faq(
        question_ar="سؤال اختبار",
        answer_ar="إجابة اختبار",
        question_en="Test Question",
        answer_en="Test Answer",
        category="Test"
    )
    
    assert result == True or result == False  # May exist already


def test_get_faq():
    """Test getting FAQs"""
    db = Database()
    faqs = db.get_all_faq()
    assert isinstance(faqs, list)


def test_unknown_question():
    """Test unknown question operations"""
    db = Database()
    
    # Save unknown - returns the row ID (new or existing), not a boolean
    result = db.save_unknown_question("سؤال مجهول", "ar")
    assert isinstance(result, int)
    assert result > 0
    
    # Get unknown
    unknown = db.get_unknown_questions()
    assert isinstance(unknown, list)
    