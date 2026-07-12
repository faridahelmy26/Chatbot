from fastapi import APIRouter, Request
from app.models.database import Database

router = APIRouter(prefix="/stats", tags=["Statistics"])
db = Database()

@router.get("/")
def get_statistics(request: Request):
    """Get platform statistics"""
    stats = db.statistics()
    
    return {
        "success": True,
        "data": stats
    }

@router.get("/faq/categories")
def get_category_stats(request: Request):
    """Get FAQ categories statistics"""
    stats = db.get_category_stats()
    
    return {
        "success": True,
        "data": stats
    }

@router.get("/chat/daily")
def get_daily_stats(request: Request, days: int = 7):
    """Get daily chat statistics"""
    stats = db.get_daily_chat_stats(days)
    
    return {
        "success": True,
        "data": stats
    }