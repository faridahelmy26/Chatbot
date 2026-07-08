from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "maman.db"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"

FAQ_JSON_PATH = DATA_DIR / "faq_data.json"

# ==========================
# Environment Detection
# ==========================

IS_PRODUCTION = os.getenv("RAILWAY_ENVIRONMENT", "development") == "production"

# ==========================
# AI Model - Different modes for different environments
# ==========================

if IS_PRODUCTION:
    # ✅ على Railway: من غير موديل (مطابقة نصوص بسيطة)
    USE_AI_MODEL = False
    MODEL_NAME = None  # مش هنستخدم موديل
    SIMILARITY_THRESHOLD = 0.05
    TOP_K = 5
    print("⚡ Running in PRODUCTION mode (no AI model)")
else:
    # ✅ على جهازك: مع الموديل الكامل
    USE_AI_MODEL = True
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 👈 الموديل بتاعك
    SIMILARITY_THRESHOLD = 0.3
    TOP_K = 5
    print(f"🔄 Running in DEVELOPMENT mode (with AI model: {MODEL_NAME})")

# ==========================
# Rate Limiting
# ==========================

RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
ENABLE_RATE_LIMIT = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"

# ==========================
# CORS Settings
# ==========================

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ==========================
# Logging
# ==========================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
