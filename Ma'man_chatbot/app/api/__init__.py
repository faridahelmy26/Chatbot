from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import os
from dotenv import load_dotenv

from app.api.routes import admin, chat, stats
from app.models.database import Database

# Load environment variables
load_dotenv()

# ==========================
# App Configuration
# ==========================

app = FastAPI(
    title="Ma'man AI Chatbot",
    version="1.0",
    description="AI Chatbot for Ma'man Platform",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ==========================
# Security
# ==========================

security = HTTPBearer()

# ==========================
# CORS Middleware
# ==========================

# Get allowed origins from environment variable
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Include Routers
# ==========================

app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(stats.router)

# ==========================
# Database
# ==========================

db = Database()


@app.on_event("startup")
def startup_event():
    """Create tables on startup"""
    try:
        db.create_tables()
        print("✅ Database initialized successfully!")
        print(f"📊 Database path: {db._connection}")
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")


@app.on_event("shutdown")
def shutdown_event():
    """Close database connection on shutdown"""
    try:
        db.close()
        print("✅ Database connection closed successfully!")
    except Exception as e:
        print(f"⚠️ Error closing database: {e}")


# ==========================
# Public Endpoints
# ==========================

@app.get("/")
def home():
    """Root endpoint"""
    return {
        "message": "Ma'man Chatbot API Running",
        "version": "1.0",
        "status": "active",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring and uptime.
    
    Returns:
        - status: healthy/unhealthy
        - database: connected/error
        - faq_count: number of active FAQs
        - pending_unknown: number of pending unknown questions
        - uptime_seconds: service uptime in seconds
    """
    try:
        # Check database connection
        conn = db.get_connection_direct()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        # Get additional stats
        faq_count = db.get_active_faq_count()
        pending_unknown = db.get_pending_unknown_count()
        
        conn.close()
        db.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "faq_count": faq_count,
            "pending_unknown": pending_unknown,
            "message": "Service is operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "message": "Service is experiencing issues"
        }


@app.get("/status")
def status():
    """
    Detailed status endpoint with comprehensive service information.
    
    Returns:
        - Service status
        - Database statistics
        - Model information
    """
    try:
        from app.core.embeddings import get_embedding_engine
        engine = get_embedding_engine()
        
        stats_data = db.get_complete_statistics()
        
        return {
            "service": {
                "name": "Ma'man AI Chatbot",
                "version": "1.0",
                "status": "running"
            },
            "database": {
                "status": "connected",
                "faq_count": stats_data["faq"]["active"],
                "total_faq": stats_data["faq"]["total"],
                "categories": stats_data["faq"]["categories"],
                "pending_unknown": stats_data["unknown_questions"]["pending"],
                "total_unknown": stats_data["unknown_questions"]["total"],
                "chat_logs": stats_data["chat_logs"]["total"]
            },
            "model": {
                "loaded": engine.model_loaded if hasattr(engine, 'model_loaded') else False,
                "embeddings_count": len(engine.metadata) if hasattr(engine, 'metadata') else 0
            },
            "authentication": {
                "enabled": bool(os.getenv("ADMIN_TOKEN") or os.getenv("JWT_SECRET_KEY")),
                "type": "JWT with API Key fallback"
            }
        }
    except Exception as e:
        return {
            "service": {
                "name": "Ma'man AI Chatbot",
                "version": "1.0",
                "status": "error"
            },
            "error": str(e)
        }


@app.get("/info")
def info():
    """
    Public information endpoint for the .NET team.
    
    Returns:
        - Endpoint information
        - Authentication requirements
        - Contact information
    """
    return {
        "service": "Ma'man AI Chatbot",
        "version": "1.0",
        "endpoints": {
            "chat": {
                "method": "POST",
                "path": "/chat",
                "body": {
                    "question": "string (required)",
                    "session_id": "string (optional)"
                },
                "response": {
                    "status": "boolean",
                    "answer": "string",
                    "similarity": "float",
                    "language": "string",
                    "session_id": "string",
                    "response_time": "float",
                    "unknown_question_id": "integer (only when status: false)"
                }
            },
            "admin": {
                "authentication": "Bearer JWT or X-API-Key",
                "endpoints": [
                    "POST /admin/faq - Add FAQ",
                    "DELETE /admin/faq/{id} - Delete FAQ",
                    "GET /admin/unknown - Get pending unknown questions",
                    "DELETE /admin/unknown/{id} - Delete unknown question",
                    "POST /admin/unknown/{id}/reply - Reply to unknown question",
                    "POST /admin/reindex - Rebuild embeddings",
                    "GET /admin/statistics - Get statistics"
                ]
            },
            "public": {
                "GET /": "Home",
                "GET /health": "Health check",
                "GET /status": "Detailed status",
                "GET /docs": "Swagger UI",
                "GET /stats/faq/categories": "FAQ categories"
            }
        },
        "authentication": {
            "type": "Bearer JWT Token or X-API-Key",
            "header_example": "Authorization: Bearer <jwt-token>",
            "api_key_example": "X-API-Key: your-api-key"
        }
    }