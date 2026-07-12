import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.models.database import Database
from app.core.embeddings import get_embedding_engine
from app.core.preprocessing import TextPreprocessor
from app.config import SIMILARITY_THRESHOLD

class ChatBot:
    """Main chatbot class"""
    
    def __init__(self):
        self.db = Database()
        self.engine = get_embedding_engine()
        self.threshold = SIMILARITY_THRESHOLD
    
    def ask(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user question
        
        Args:
            question: The user's question
            session_id: Optional session ID for tracking
        
        Returns:
            Dictionary with response data including success, answer, and metadata
        """
        start = time.time()
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Detect language
        language = TextPreprocessor.detect_language(question)
        
        # Search for similar questions
        results = self.engine.search(question)
        
        # If no results or low similarity
        if len(results) == 0 or results[0]["similarity"] < self.threshold:
            return self.handle_unknown(
                question, language, session_id, start
            )
        
        # Get best match
        best = results[0]
        
        # Save chat log
        response_time = round(time.time() - start, 3)
        self.db.save_chat_log(
            session_id=session_id,
            question=question,
            answer=best["answer"],
            language=language,
            similarity=best["similarity"],
            response_time=response_time
        )
        
        return {
            "success": True,
            "answer": best["answer"],
            "similarity": round(best["similarity"], 3),
            "language": language,
            "session_id": session_id,
            "response_time": response_time
        }
    
    def handle_unknown(self, question: str, language: str, session_id: str, start: float) -> Dict[str, Any]:
        """
        Handle unknown question - saves it and returns appropriate response
        
        Args:
            question: The unknown question
            language: Detected language
            session_id: Session ID
            start: Start time for response time calculation
        
        Returns:
            Dictionary with unknown response including the unknown_question_id
        """
        # Save unknown question and get its ID
        unknown_id = self.db.save_unknown_question(question, language)
        
        response_time = round(time.time() - start, 3)
        
        # Save chat log with empty answer
        self.db.save_chat_log(
            session_id=session_id,
            question=question,
            answer="",
            language=language,
            similarity=0,
            response_time=response_time
        )
        
        # Get appropriate response based on language
        if language == "ar":
            answer = (
                "عذرًا، لم أتمكن من العثور على إجابة لهذا السؤال.\n"
                "تم إرسال سؤالك إلى الإدارة وسيتم الرد عليه قريبًا."
            )
        else:
            answer = (
                "Sorry, I couldn't find an answer.\n"
                "Your question has been sent to the admin."
            )
        
        return {
            "success": False,
            "answer": answer,
            "similarity": 0,
            "language": language,
            "session_id": session_id,
            "response_time": response_time,
            "unknown_question_id": unknown_id  # 👈 NEW: Return the ID
        }
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get chat history for a session
        
        Args:
            session_id: Session ID to fetch history for
        
        Returns:
            List of chat history entries
        """
        conn = self.db.get_connection_direct()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_question, bot_answer, language, similarity, created_at
            FROM chat_logs
            WHERE session_id=?
            ORDER BY created_at DESC
            LIMIT 50
        """, (session_id,))
        
        history = [dict(r) for r in cursor.fetchall()]
        conn.close()
        
        return history
    
    def get_unknown_question_by_id(self, unknown_id: int) -> Optional[Dict[str, Any]]:
        """
        Get unknown question details by ID
        
        Args:
            unknown_id: The unknown question ID
        
        Returns:
            Unknown question details or None if not found
        """
        return self.db.get_unknown_question_by_id(unknown_id)
    
    def mark_unknown_as_answered(self, unknown_id: int) -> bool:
        """
        Mark an unknown question as answered
        
        Args:
            unknown_id: The unknown question ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.db.mark_unknown_as_answered(unknown_id)
            return True
        except Exception:
            return False
    
    def get_pending_unknown_count(self) -> int:
        """
        Get count of pending unknown questions
        
        Returns:
            Number of pending unknown questions
        """
        return self.db.get_pending_unknown_count()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get complete chatbot statistics
        
        Returns:
            Dictionary with all statistics
        """
        return self.db.get_complete_statistics()