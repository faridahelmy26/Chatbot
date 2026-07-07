import json
import numpy as np
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore")

from app.models.database import Database
from app.config import DATA_DIR, TOP_K, SIMILARITY_THRESHOLD

_engine = None

def get_embedding_engine():
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine

class EmbeddingEngine:
    def __init__(self):
        self.db = Database()
        self.metadata = []
        self.model_loaded = True  # Always ready
        self.load_metadata()
    
    def load_metadata(self):
        """Load FAQ metadata from database"""
        print("📚 Loading FAQ metadata...")
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
        
        print(f"✅ Loaded {len(self.metadata)} FAQs")
    
    def build_embeddings(self):
        """Rebuild - just reload metadata"""
        self.load_metadata()
        print("✅ FAQ data reloaded")
    
    def load(self):
        """No-op for compatibility"""
        return True
    
    def is_ready(self):
        """Always ready"""
        return True
    
    def search(self, query):
        """Simple text search"""
        if not self.metadata:
            return []
        
        from app.core.preprocessing import TextPreprocessor
        
        language = TextPreprocessor.detect_language(query)
        query_clean = TextPreprocessor.clean(query).lower()
        query_words = set(query_clean.split())
        
        if not query_words:
            return []
        
        results = []
        
        for item in self.metadata:
            if item["language"] != language:
                continue
            
            question_clean = TextPreprocessor.clean(item["question"]).lower()
            
            # Exact match check
            if query_clean in question_clean or question_clean in query_clean:
                similarity = 0.9
            else:
                # Word overlap
                question_words = set(question_clean.split())
                common = query_words.intersection(question_words)
                union = query_words.union(question_words)
                similarity = len(common) / len(union) if union else 0
            
            if similarity >= SIMILARITY_THRESHOLD:
                results.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "language": item["language"],
                    "similarity": round(similarity, 3)
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:TOP_K]
