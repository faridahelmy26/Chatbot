from fastapi import APIRouter, HTTPException, status, Request, Depends
from pydantic import BaseModel

from app.models.database import Database
from app.core.embeddings import get_embedding_engine
from app.api.middleware.auth import verify_token

from app.api.middleware.auth import create_jwt_token

router = APIRouter(prefix="/admin", tags=["Admin"])
db = Database()

# =====================================================
# Models
# =====================================================

class FAQRequest(BaseModel):
    question_ar: str
    answer_ar: str
    question_en: str
    answer_en: str
    category: str = "General"
    unknown_id: int | None = None

class FAQAnswerRequest(BaseModel):
    question_ar: str | None = None
    question_en: str | None = None
    answer_ar: str
    answer_en: str | None = None
    category: str = "General"

# =====================================================
# Get Unknown Questions
# =====================================================

@router.get("/unknown")
def get_unknown_questions(
    request: Request,
    token: str = Depends(verify_token)
):
    """Get all pending unknown questions"""
    conn = db.get_connection_direct()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT *
        FROM unknown_questions
        WHERE status='Pending'
        ORDER BY frequency DESC
    """)
    
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    db.close()
    
    return {
        "success": True,
        "count": len(rows),
        "data": rows
    }

# =====================================================
# Add FAQ
# =====================================================

@router.post("/faq")
def add_faq(
    request: Request,
    data: FAQRequest,
    token: str = Depends(verify_token)
):
    """Add new FAQ"""
    inserted = db.insert_faq(
        question_ar=data.question_ar,
        answer_ar=data.answer_ar,
        question_en=data.question_en,
        answer_en=data.answer_en,
        category=data.category
    )
    
    if not inserted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FAQ already exists."
        )
    
    if data.unknown_id is not None:
        db.mark_unknown_as_answered(data.unknown_id)
    
    # Try to rebuild embeddings
    try:
        get_embedding_engine().build_embeddings()
    except Exception as e:
        print(f"⚠️ Could not rebuild embeddings: {e}")
    
    return {
        "success": True,
        "message": "FAQ Added Successfully"
    }

@router.delete("/faq/{faq_id}")
def delete_faq(
    request: Request,
    faq_id: int,
    token: str = Depends(verify_token)
):
    """Soft delete FAQ"""
    deleted = db.delete_faq(faq_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAQ not found."
        )
    
    # Try to rebuild embeddings
    try:
        get_embedding_engine().build_embeddings()
    except Exception as e:
        print(f"⚠️ Could not rebuild embeddings: {e}")
    
    return {
        "success": True,
        "message": "FAQ deleted successfully."
    }

@router.delete("/unknown/{unknown_id}")
def delete_unknown_question(
    request: Request,
    unknown_id: int,
    token: str = Depends(verify_token)
):
    """Delete unknown question"""
    deleted = db.delete_unknown_question(unknown_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown question not found."
        )
    
    return {
        "success": True,
        "message": "Unknown question deleted successfully."
    }

@router.post("/unknown/{unknown_id}/reply")
def reply_unknown_question(
    request: Request,
    unknown_id: int,
    data: FAQAnswerRequest,
    token: str = Depends(verify_token)
):
    """Reply to unknown question and save as FAQ"""
    unknown_question = db.get_unknown_question_by_id(unknown_id)
    
    if unknown_question is None or unknown_question["status"] != "Pending":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown question not found or already answered."
        )
    
    question_text = unknown_question["question"]
    question_language = unknown_question["language"] or "ar"
    
    question_ar = data.question_ar or question_text if question_language == "ar" else data.question_ar or question_text
    question_en = data.question_en or question_text if question_language == "en" else data.question_en or question_text
    
    if not question_ar:
        question_ar = question_text
    if not question_en:
        question_en = question_text
    
    # Use Arabic answer for English if not provided
    answer_en = data.answer_en or data.answer_ar
    
    inserted = db.insert_faq(
        question_ar=question_ar,
        answer_ar=data.answer_ar,
        question_en=question_en,
        answer_en=answer_en,
        category=data.category
    )
    
    if not inserted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FAQ could not be created; a duplicate already exists."
        )
    
    db.mark_unknown_as_answered(unknown_id)
    
    # Try to rebuild embeddings
    try:
        get_embedding_engine().build_embeddings()
    except Exception as e:
        print(f"⚠️ Could not rebuild embeddings: {e}")
    
    return {
        "success": True,
        "message": "Unknown question replied and saved as FAQ."
    }

# =====================================================
# Re-index Endpoint
# =====================================================

@router.post("/reindex")
def reindex_embeddings(
    request: Request,
    token: str = Depends(verify_token)
):
    """Rebuild embeddings from current FAQ data"""
    try:
        get_embedding_engine().build_embeddings()
        faq_count = len(get_embedding_engine().metadata)
        
        return {
            "success": True,
            "message": "Embeddings rebuilt successfully",
            "faq_count": faq_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rebuild embeddings: {str(e)}"
        )
    

# =====================================================
# Generate Test Token (For Development Only)
# =====================================================

@router.get("/generate-token")
def generate_test_token():
    """
    Generate a test JWT token for development.
    ⚠️ Remove this endpoint in production!
    """
    from app.api.middleware.auth import create_jwt_token, JWT_EXPIRATION_MINUTES
    
    token = create_jwt_token({"sub": "admin", "role": "admin"})
    return {
        "token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_MINUTES * 60,
        "example_usage": f"Authorization: Bearer {token}"
    }

# =====================================================
# Statistics
# =====================================================

@router.get("/statistics")
def statistics(
    request: Request,
    token: str = Depends(verify_token)
):
    """Get platform statistics"""
    return db.statistics()



# =====================================================
# Generate Test Token (For Development Only)
# =====================================================

@router.get("/generate-token")
def generate_test_token():
    """
    Generate a test JWT token for development.
    ⚠️ Remove this endpoint in production!
    """
    token = create_jwt_token({"sub": "admin", "role": "admin"})
    return {
        "token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_MINUTES * 60,
        "example_usage": f"Authorization: Bearer {token}"
    }