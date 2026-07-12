import json
import numpy as np
from pathlib import Path
import warnings
import os
import threading
import time

warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"⚠️ Error importing sentence_transformers: {e}")
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
        self._loading_thread = None
        self._loading_complete = False
        
        if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
            self.load()
        
        self._start_background_loading()
    
    def _start_background_loading(self):
        if self._loading_thread is None or not self._loading_thread.is_alive():
            self._loading_thread = threading.Thread(target=self._load_model_background, daemon=True)
            self._loading_thread.start()
            print("🔄 Started background thread for model loading...")
    
    def _load_model_background(self):
        try:
            print(f"🔄 Loading SentenceTransformer model: {MODEL_NAME}")
            start_time = time.time()
            
            self.model = SentenceTransformer(MODEL_NAME)
            self.model_loaded = True
            self._loading_complete = True
            
            elapsed = time.time() - start_time
            print(f"✅ Model loaded successfully in {elapsed:.2f} seconds!")
            
            self.build_embeddings()
        except Exception as e:
            print(f"⚠️ Could not load model in background: {e}")
            self.model_loaded = False
            self._loading_complete = True
    
    def build_embeddings(self):
        if not self.model_loaded or self.model is None:
            print("⚠️ Model not loaded yet. Will build embeddings when model is ready.")
            return
        
        print("Building embeddings...")
        metadata = []
        sentences = []
        
        for lang in ["ar", "en"]:
            rows = self.db.get_faq_by_language(lang)
            
            for row in rows:
                # Convert sqlite3.Row to dict to avoid issues
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
        return self.model_loaded and self.model is not None
    
    def is_loading(self):
        return not self._loading_complete
    
    def search(self, query):
        print(f"🔍 Searching for: {query}")
        print(f"📚 Metadata count: {len(self.metadata)}")
        
        if not self.metadata or self.embeddings is None or len(self.embeddings) == 0:
                print("⚠️ No embeddings available for search")
                return []
        
        if not self.is_ready():
                print("⚠️ Model not loaded yet. Search will return empty.")
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
                print(f"📊 Query embedding shape: {query_embedding.shape}")
        except Exception as e:
                print(f"⚠️ Error encoding query: {e}")
                return []
        
        indices = [
                i for i, item in enumerate(self.metadata)
                if item["language"] == language
        ]
        
        print(f"📊 Language indices: {len(indices)}")
        
        if not indices:
                return []
        
        lang_embeddings = self.embeddings[indices]
        similarities = cosine_similarity(query_embedding, lang_embeddings)[0]
        print(f"📊 Similarities: {similarities}")
        
        order = similarities.argsort()[::-1][:TOP_K]
        print(f"📊 Order: {order}")
        
        results = []
        for idx in order:
                item = self.metadata[indices[idx]]
                similarity = float(similarities[idx])
                print(f"📊 Item: {item['question'][:30]}... Similarity: {similarity}")
                
                if similarity < SIMILARITY_THRESHOLD:
                        print(f"⏭️ Skipping (below threshold {SIMILARITY_THRESHOLD})")
                        continue
                
                results.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "language": item["language"],
                        "similarity": similarity
                })
        
        print(f"✅ Results count: {len(results)}")
        return results