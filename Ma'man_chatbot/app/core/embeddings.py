import json
import numpy as np
from pathlib import Path
import warnings
import os
import re

warnings.filterwarnings("ignore")

from app.models.database import Database
from app.config import DATA_DIR, TOP_K, SIMILARITY_THRESHOLD, USE_AI_MODEL

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
METADATA_FILE = DATA_DIR / "metadata.json"

_engine = None

def get_embedding_engine():
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine

class EmbeddingEngine:
    def __init__(self):
        self.db = Database()
        self.embeddings = None
        self.metadata = []
        self.model_loaded = False
        self.use_ai = USE_AI_MODEL
        
        # Load metadata from database
        self.load_metadata()
        
        # Try to load embeddings if exists
        if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
            self.load()
    
    def load_metadata(self):
        """Load FAQ metadata from database"""
        print("📚 Loading FAQ metadata from database...")
        self.metadata = []
        
        for lang in ["ar", "en"]:
            rows = self.db.get_faq_by_language(lang)
            
            for row in rows:
                question = row["question"]
                self.metadata.append({
                    "id": row["id"],
                    "question": question,
                    "answer": row["answer"],
                    "language": lang,
                    "category": row.get("category", "General")
                })
        
        print(f"✅ Loaded {len(self.metadata)} FAQs from database")
    
    def build_embeddings(self):
        """Simple rebuild - just reload metadata"""
        self.load_metadata()
        print("✅ FAQ data reloaded")
    
    def load(self):
        """Load embeddings from file if exists"""
        if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
            try:
                self.embeddings = np.load(EMBEDDINGS_FILE)
                print(f"✅ Embeddings loaded from file")
                return True
            except Exception as e:
                print(f"⚠️ Error loading embeddings: {e}")
                return False
        return False
    
    def is_ready(self):
        """Always ready in simple mode"""
        return True
    
    def search(self, query):
        """
        Simple text search without AI model
        Uses keyword matching and basic similarity
        """
        if not self.metadata:
            print("⚠️ No metadata available for search")
            return []
        
        from app.core.preprocessing import TextPreprocessor
        
        language = TextPreprocessor.detect_language(query)
        query_clean = TextPreprocessor.clean(query)
        query_words = set(query_clean.lower().split())
        
        if not query_words:
            return []
        
        results = []
        
        for item in self.metadata:
            if item["language"] != language:
                continue
            
            question_clean = TextPreprocessor.clean(item["question"])
            question_words = set(question_clean.lower().split())
            
            # Calculate simple similarity
            common_words = query_words.intersection(question_words)
            union_words = query_words.union(question_words)
            
            if not union_words:
                continue
            
            similarity = len(common_words) / len(union_words)
            
            # Check for phrase match
            if query_clean.lower() in question_clean.lower():
                similarity = max(similarity, 0.8)
            
            if similarity >= SIMILARITY_THRESHOLD:
                results.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "language": item["language"],
                    "similarity": round(similarity, 3)
                })
        
        # Sort by similarity and return top K
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:TOP_K]
