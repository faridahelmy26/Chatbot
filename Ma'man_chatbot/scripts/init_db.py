import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.models.database import Database

def main():
    print("🚀 Creating database...")
    db = Database()
    db.create_tables()
    db.close()
    print("✅ Database created successfully!")
    print(f"📁 Location: data/maman.db")

if __name__ == "__main__":
    main()