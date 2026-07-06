import json
import numpy as np
from pathlib import Path
import warnings
import os

# تجاهل التحذيرات
warnings.filterwarnings("ignore")

# محاولة استيراد sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"⚠️ Error importing sentence_transformers: {e}")
    print("📦 Please install: pip install sentence-transformers==2.2.2")
    raise

from sklearn.metrics.pairwise import cosine_similarity

from app.models.database import Database
from app.config import DATA_DIR, MODEL_NAME, TOP_K, SIMILARITY_THRESHOLD

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
        self.model = None
        self.embeddings = None
        self.metadata = []
        self.model_loaded = False
        
        # Try to load existing embeddings first
        if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
            self.load()
            if self.embeddings is not None:
                # Try to load model for search
                try:
                    print("Loading SentenceTransformer model for search...")
                    self.model = SentenceTransformer(MODEL_NAME)
                    self.model_loaded = True
                    print("✅ Model loaded successfully!")
                except Exception as e:
                    print(f"⚠️ Could not load model: {e}")
                    print("📌 Search will use fallback mode")
                    self.model_loaded = False
                return
        
        # If no embeddings, try to load model
        try:
            print("Loading SentenceTransformer model...")
            self.model = SentenceTransformer(MODEL_NAME)
            self.model_loaded = True
            self.build_embeddings()
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("📌 Using fallback mode - embeddings will be built when FAQ data is added")
            self.model_loaded = False
    
    def build_embeddings(self):
        if not self.model_loaded:
            print("⚠️ Model not loaded. Trying to load...")
            try:
                self.model = SentenceTransformer(MODEL_NAME)
                self.model_loaded = True
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return
        
        print("Building embeddings...")
        metadata = []
        sentences = []
        
        for lang in ["ar", "en"]:
            rows = self.db.get_faq_by_language(lang)
            
            for row in rows:
                question = row["question"]
                metadata.append({
                    "id": row["id"],
                    "question": question,
                    "answer": row["answer"],
                    "language": lang,
                    "category": row.get("category", "General")
                })
                sentences.append(question)
        
        if not sentences:
            print("No FAQs found to build embeddings")
            return
        
        print(f"Encoding {len(sentences)} sentences...")
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        np.save(EMBEDDINGS_FILE, embeddings)
        
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        self.embeddings = embeddings
        self.metadata = metadata
        print(f"✅ Embeddings built: {len(metadata)} items")
    
    def load(self):
        if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
            try:
                self.embeddings = np.load(EMBEDDINGS_FILE)
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"✅ Embeddings loaded: {len(self.metadata)} items")
                return True
            except Exception as e:
                print(f"⚠️ Error loading embeddings: {e}")
                return False
        return False
    
    def search(self, query):
        # Return empty if no embeddings or metadata
        if not self.metadata or self.embeddings is None or len(self.embeddings) == 0:
            print("⚠️ No embeddings available for search")
            return []
        
        # If model not loaded, try to load it once
        if not self.model_loaded or self.model is None:
            try:
                print("🔄 Loading model for search...")
                self.model = SentenceTransformer(MODEL_NAME)
                self.model_loaded = True
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Model not available for search: {e}")
                # Return fallback answer
                return []
        
        # Import here to avoid circular imports
        from app.core.preprocessing import TextPreprocessor
        
        # استيراد QueryExpansion من الملف الصحيح
        try:
            from app.query_expansion import QueryExpansion
        except ImportError:
            # Fallback if import fails
            class QueryExpansion:
                @staticmethod
                def expand(text, language):
                    return text
        
        language = TextPreprocessor.detect_language(query)
        query = QueryExpansion.expand(query, language)
        query = TextPreprocessor.clean(query)
        
        try:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        except Exception as e:
            print(f"⚠️ Error encoding query: {e}")
            return []
        
        indices = [
            i
            for i, item in enumerate(self.metadata)
            if item["language"] == language
        ]
        
        if not indices:
            return []
        
        lang_embeddings = self.embeddings[indices]
        similarities = cosine_similarity(query_embedding, lang_embeddings)[0]
        order = similarities.argsort()[::-1][:TOP_K]
        
        results = []
        for idx in order:
            item = self.metadata[indices[idx]]
            similarity = float(similarities[idx])
            
            if similarity < SIMILARITY_THRESHOLD:
                continue
            
            results.append({
                "question": item["question"],
                "answer": item["answer"],
                "language": item["language"],
                "similarity": similarity
            })
        
        return results