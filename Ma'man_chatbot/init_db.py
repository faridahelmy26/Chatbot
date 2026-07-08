import sys
import os

# أضف المجلد الرئيسي إلى مسار Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.database import Database


def init_database():
    """Initialize database with tables"""
    db = Database()
    db.create_tables()
    db.close()
    
    print("✅ Database initialized successfully!")
    print(f"📁 Database location: data/maman.db")


if __name__ == "__main__":
    init_database()
