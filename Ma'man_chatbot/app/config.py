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
# AI Model - Smaller model for Railway (saves memory)
# ==========================

# ✅ استخدام موديل أصغر (90 MB بدل 500 MB)
MODEL_NAME = "all-MiniLM-L6-v2"  # 👈 أصغر وأسرع

# ==========================
# Search Settings
# ==========================

SIMILARITY_THRESHOLD = 0.35  # مناسب للموديل الأصغر
TOP_K = 5

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
