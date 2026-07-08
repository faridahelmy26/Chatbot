import json
import numpy as np
from pathlib import Path
import warnings
import os
import re

warnings.filterwarnings("ignore")

# محاولة استيراد sentence_transformers (لو موجود)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from sklearn.metrics.pairwise import cosine_similarity

from app.models.database import Database
from app.config import DATA_DIR, MODEL_NAME, TOP_K, SIMILARITY_THRESHOLD, USE_AI_MODEL

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
        self.use_ai = USE_AI_MODEL
        
        # Load metadata from database
        self.load_metadata()
        
        if self.use_ai:
            print("🔄 Loading AI model...")
            self._load_model()
        else:
            print("⚡ Running in lightweight mode (no AI model)")
            self.model_loaded = True  # Always ready for simple search
    
    def load_metadata(self):
        """Load FAQ metadata from database"""
        print("📚 Loading FAQ metadata from database...")
        self.metadata = []
        
        for lang in ["ar", "en"]:
            rows = self.db.get_faq_by_language(lang)
            
            for row in rows:
                row_dict = dict(row)
                
                question = row_dict.get("question", "")
                answer = row_dict.get("answer", "")
                category = row_dict.get("category") or "General"
                
                self.metadata.append({
                    "id": row_dict.get("id"),
                    "question": question,
                    "answer": answer,
                    "language": lang,
                    "category": category
                })
        
        print(f"✅ Loaded {len(self.metadata)} FAQs from database")
    
    def _load_model(self):
        """Load AI model"""
        try:
            if SentenceTransformer is None:
                print("⚠️ SentenceTransformer not available")
                self.model_loaded = False
                return
            
            print(f"🔄 Loading model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME)
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            
            # Build embeddings if needed
            if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
                self.build_embeddings()
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model_loaded = False
    
    def build_embeddings(self):
        """Build embeddings using AI model"""
        if not self.model_loaded or self.model is None:
            print("⚠️ Model not loaded. Cannot build embeddings.")
            return
        
        print("Building embeddings...")
        metadata = []
        sentences = []
        
        for lang in ["ar", "en"]:
            rows = self.db.get_faq_by_language(lang)
            
            for row in rows:
                row_dict = dict(row)
                
                question = row_dict.get("question", "")
                answer = row_dict.get("answer", "")
                category = row_dict.get("category") or "General"
                
                metadata.append({
                    "id": row_dict.get("id"),
                    "question": question,
                    "answer": answer,
                    "language": lang,
                    "category": category
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
        """Load embeddings from file"""
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
    
    def is_ready(self):
        """Check if ready for search"""
        if self.use_ai:
            return self.model_loaded and self.model is not None
        return True  # Always ready in simple mode
    
    def search(self, query):
        """Main search function - chooses mode based on config"""
        print(f"🔍 Searching for: {query}")
        print(f"📚 Metadata count: {len(self.metadata)}")
        
        if not self.metadata:
            print("⚠️ No metadata available")
            return []
        
        if self.use_ai and self.is_ready():
            return self._ai_search(query)
        else:
            return self._simple_search(query)
    
    def _simple_search(self, query):
        """Simple text matching without AI"""
        from app.core.preprocessing import TextPreprocessor
        
        language = TextPreprocessor.detect_language(query)
        query_clean = TextPreprocessor.clean(query).lower()
        query_words = set(query_clean.split())
        
        # Remove common stop words
        stop_words = {'و', 'في', 'من', 'على', 'الى', 'عن', 'مع', 'هو', 'هي', 'ما', 'اذا', 'ان', 'ال', 'هل', 'كيف'}
        query_words = query_words - stop_words
        
        if not query_words:
            return []
        
        results = []
        
        for item in self.metadata:
            if item["language"] != language:
                continue
            
            question_clean = TextPreprocessor.clean(item["question"]).lower()
            question_words = set(question_clean.split()) - stop_words
            
            # Check for exact match
            if query_clean in question_clean or question_clean in query_clean:
                similarity = 0.9
            else:
                # Word overlap
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
        print(f"✅ Simple search results: {len(results)}")
        return results[:TOP_K]
    
    def _ai_search(self, query):
        """AI-powered semantic search"""
        if not self.metadata or self.embeddings is None or len(self.embeddings) == 0:
            print("⚠️ No embeddings available for AI search")
            return []
        
        from app.core.preprocessing import TextPreprocessor
        
        try:
            from app.query_expansion import QueryExpansion
        except ImportError:
            class QueryExpansion:
                @staticmethod
                def expand(text, language):
                    return text
        
        language = TextPreprocessor.detect_language(query)
        print(f"🌐 Language: {language}")
        
        query = QueryExpansion.expand(query, language)
        query = TextPreprocessor.clean(query)
        print(f"🧹 Cleaned query: {query}")
        
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
            i for i, item in enumerate(self.metadata)
            if item["language"] == language
        ]
        
        if not indices:
            return []
        
        lang_embeddings = self.embeddings[indices]
        similarities = cosine_similarity(query_embedding, lang_embeddings)[0]
        
        print(f"📊 Top similarities:")
        for i, idx in enumerate(similarities.argsort()[::-1][:5]):
            item = self.metadata[indices[idx]]
            sim = float(similarities[idx])
            print(f"  {i+1}. {item['question'][:40]}... -> {sim:.3f}")
        
        order = similarities.argsort()[::-1][:TOP_K]
        
        results = []
        for idx in order:
            item = self.metadata[indices[idx]]
            similarity = float(similarities[idx])
            
            if similarity < SIMILARITY_THRESHOLD:
                print(f"⏭️ Skipping (below {SIMILARITY_THRESHOLD}): {similarity:.3f}")
                continue
            
            results.append({
                "question": item["question"],
                "answer": item["answer"],
                "language": item["language"],
                "similarity": similarity
            })
        
        print(f"✅ AI search results: {len(results)}")
        return results
