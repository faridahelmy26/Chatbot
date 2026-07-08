import os
import sys
import uvicorn

# أضف المجلد الرئيسي إلى مسار Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# هيّئ قاعدة البيانات عند البدء
from app.database import Database
db = Database()
db.create_tables()
db.close()
print("✅ Database initialized!")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting server on port: {port}")
    print(f"🌐 Host: 0.0.0.0")
    
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
