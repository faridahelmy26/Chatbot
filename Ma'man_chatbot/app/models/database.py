import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

# Database path
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "maman.db"

class Database:
    def __init__(self):
        self._connection = None
    
    def get_connection_direct(self):
        """Get direct database connection (for backward compatibility)"""
        if self._connection is None:
            DB_PATH.parent.mkdir(exist_ok=True)
            self._connection = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connection"""
        conn = None
        try:
            DB_PATH.parent.mkdir(exist_ok=True)
            conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()
    
    def close(self):
        """Close connection if open"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def create_tables(self):
        """Create all tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faq (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_ar TEXT NOT NULL,
                    answer_ar TEXT NOT NULL,
                    question_en TEXT NOT NULL,
                    answer_en TEXT NOT NULL,
                    category TEXT DEFAULT 'General',
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unknown_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE,
                    language TEXT,
                    frequency INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'Pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_question TEXT,
                    bot_answer TEXT,
                    language TEXT,
                    similarity REAL,
                    response_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            print("✅ Tables created successfully")
    
    def insert_faq(self, question_ar, answer_ar, question_en, answer_en, category="General"):
        """Insert new FAQ"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("""
                SELECT id FROM faq
                WHERE question_ar=? OR question_en=?
            """, (question_ar, question_en))
            
            if cursor.fetchone():
                return False
            
            # Insert
            cursor.execute("""
                INSERT INTO faq (question_ar, answer_ar, question_en, answer_en, category)
                VALUES (?, ?, ?, ?, ?)
            """, (question_ar, answer_ar, question_en, answer_en, category))
            
            conn.commit()
            return True
    
    def get_all_faq(self):
        """Get all active FAQs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM faq
                WHERE is_active=1
                ORDER BY id
            """)
            return cursor.fetchall()
    
    def get_faq_by_language(self, language):
        """Get FAQs by language"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if language == "ar":
                cursor.execute("""
                    SELECT id, question_ar AS question, answer_ar AS answer, category
                    FROM faq
                    WHERE is_active=1
                """)
            else:
                cursor.execute("""
                    SELECT id, question_en AS question, answer_en AS answer, category
                    FROM faq
                    WHERE is_active=1
                """)
            
            return cursor.fetchall()
    
    def get_faq_by_id(self, faq_id):
        """Get FAQ by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM faq WHERE id=? AND is_active=1",
                (faq_id,)
            )
            return cursor.fetchone()
    
    def update_faq(self, faq_id, **kwargs):
        """Update FAQ"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            for key, value in kwargs.items():
                if value is not None and key in [
                    'question_ar', 'answer_ar',
                    'question_en', 'answer_en',
                    'category', 'is_active'
                ]:
                    updates.append(f"{key}=?")
                    params.append(value)
            
            if not updates:
                return False
            
            updates.append("updated_at=CURRENT_TIMESTAMP")
            params.append(faq_id)
            
            query = f"UPDATE faq SET {', '.join(updates)} WHERE id=?"
            
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_faq(self, faq_id):
        """Soft delete FAQ"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE faq
                SET is_active=0, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (faq_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_unknown_questions(self, status="Pending"):
        """Get unknown questions by status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM unknown_questions
                WHERE status=?
                ORDER BY frequency DESC
            """, (status,))
            return cursor.fetchall()
    
    def get_unknown_question_by_id(self, unknown_id):
        """Get unknown question by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM unknown_questions WHERE id=?",
                (unknown_id,)
            )
            return cursor.fetchone()
    
    def mark_unknown_as_answered(self, unknown_id):
        """Mark unknown question as answered"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE unknown_questions
                SET status='Answered'
                WHERE id=?
            """, (unknown_id,))
            conn.commit()
    
    def delete_unknown_question(self, unknown_id):
        """Delete unknown question"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM unknown_questions WHERE id=?",
                (unknown_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def save_unknown_question(self, question, language):
        """
        Save unknown question and return its ID
        
        Args:
            question: The question text
            language: 'ar' or 'en'
        
        Returns:
            The ID of the unknown question (existing or new)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id FROM unknown_questions WHERE question=?",
                (question,)
            )
            row = cursor.fetchone()
            
            if row:
                # Question exists - increment frequency
                cursor.execute("""
                    UPDATE unknown_questions
                    SET frequency=frequency+1
                    WHERE id=?
                """, (row["id"],))
                conn.commit()
                return row["id"]  # Return existing ID
            else:
                # New question - insert
                cursor.execute("""
                    INSERT INTO unknown_questions (question, language)
                    VALUES (?, ?)
                """, (question, language))
                conn.commit()
                return cursor.lastrowid  # Return new ID
    
    def save_chat_log(self, session_id, question, answer, language, similarity, response_time):
        """Save chat log"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (
                    session_id, user_question, bot_answer,
                    language, similarity, response_time
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, question, answer, language, similarity, response_time))
            conn.commit()
    
    def statistics(self):
        """Get platform statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM faq")
            faq = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM unknown_questions")
            unknown = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            chats = cursor.fetchone()[0]
            
            return {
                "faq": faq,
                "unknown_questions": unknown,
                "chat_logs": chats
            }
    
    def get_category_stats(self):
        """Get FAQ category statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM faq
                WHERE is_active=1
                GROUP BY category
                ORDER BY count DESC
            """)
            rows = cursor.fetchall()
            return {r["category"]: r["count"] for r in rows}
    
    def get_daily_chat_stats(self, days=7):
        """Get daily chat statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as total_chats,
                    AVG(similarity) as avg_similarity,
                    AVG(response_time) as avg_response_time
                FROM chat_logs
                WHERE created_at >= DATE('now', ?)
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, (f"-{days} days",))
            return [dict(r) for r in cursor.fetchall()]
    
    def get_pending_unknown_count(self) -> int:
        """Get count of pending unknown questions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM unknown_questions
                WHERE status='Pending'
            """)
            return cursor.fetchone()[0]
    
    def get_answered_unknown_count(self) -> int:
        """Get count of answered unknown questions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM unknown_questions
                WHERE status='Answered'
            """)
            return cursor.fetchone()[0]
    
    def get_total_chats(self) -> int:
        """Get total number of chat logs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            return cursor.fetchone()[0]
    
    def get_active_faq_count(self) -> int:
        """Get count of active FAQs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM faq
                WHERE is_active=1
            """)
            return cursor.fetchone()[0]
    
    def get_complete_statistics(self) -> Dict[str, Any]:
        """
        Get complete platform statistics including all metrics
        
        Returns:
            Dictionary with all statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM faq")
            total_faq = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM faq
                WHERE is_active=1
            """)
            active_faq = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM unknown_questions")
            total_unknown = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM unknown_questions
                WHERE status='Pending'
            """)
            pending_unknown = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            total_chats = cursor.fetchone()[0]
            
            # Get category distribution
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM faq
                WHERE is_active=1
                GROUP BY category
                ORDER BY count DESC
            """)
            categories = {r["category"]: r["count"] for r in cursor.fetchall()}
            
            return {
                "faq": {
                    "total": total_faq,
                    "active": active_faq,
                    "categories": categories
                },
                "unknown_questions": {
                    "total": total_unknown,
                    "pending": pending_unknown,
                    "answered": total_unknown - pending_unknown
                },
                "chat_logs": {
                    "total": total_chats
                }
            }