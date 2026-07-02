import sys
import os
from pathlib import Path

# Add the 'src' folder to Python path (this is where your code is)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pickle
import logging
from datetime import datetime
import uvicorn
import json
import hashlib

# Import from the 'src' folder
try:
    from DataLoader import DataLoader
    from DataCleaner import DataCleaner
    from ContentRecommender import ContentBased
    from Collaborative import CollaborativeRecommender
    from MLModel import MLModel
    from recommender import HybridRecommender
    logger_import = logging.getLogger(__name__)
    logger_import.info("✅ All modules imported successfully from src/")
except ImportError as e:
    # Fallback: try different naming
    try:
        from DataLoader import DataLoader
        from DataCleaner import DataCleaner
        from ContentRecommender import ContentRecommender
        from Collaborative import CollaborativeRecommender
        from MLModel import MLModel
        from recommender import HybridRecommender
    except ImportError as e2:
        # If still failing, create dummy classes
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Import error: {e2}")
        
        class DataLoader:
            def __init__(self, data_dir="data"):
                self.data_dir = data_dir
            def load_content(self, path):
                return pd.read_csv(path)
            def load_interactions(self, path):
                return pd.read_csv(path)
            def load_users(self, path):
                return pd.read_excel(path) if os.path.exists(path) else pd.DataFrame()
        
        class DataCleaner:
            def clean(self, df):
                return df
            def clean_interactions(self, df):
                return df
        
        class ContentRecommender:
            def __init__(self, df=None, text_cols=None, use_embeddings=False):
                self.df = df
            def recommend_by_title(self, title, k=5, **kwargs):
                return None
        
        class CollaborativeRecommender:
            def __init__(self, interactions=None, n_components=50):
                self.interactions = interactions
            def fit(self):
                pass
            def recommend(self, user_id, k=5):
                return []
        
        class MLModel:
            def __init__(self, **kwargs):
                pass
        
        class HybridRecommender:
            def __init__(self, **kwargs):
                pass
            def recommend(self, user_id, top_n=5, **kwargs):
                return []

# =========================
# Setup Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('cache', exist_ok=True)

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="Hybrid Recommendation System API",
    description="Advanced recommendation system with content-based and collaborative filtering",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models
# =========================
class HybridRequest(BaseModel):
    user_id: int = Field(..., description="User ID", gt=0)
    top_n: int = Field(5, description="Number of recommendations", ge=1, le=50)
    include_scores: bool = Field(True, description="Include scores")
    exclude_seen: bool = Field(True, description="Exclude seen items")
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_level: Optional[str] = Field(None, description="Filter by level")

class ContentRequest(BaseModel):
    title: str = Field(..., description="Course title", min_length=1)
    top_n: int = Field(5, description="Number of recommendations", ge=1, le=50)
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_level: Optional[str] = Field(None, description="Filter by level")

class RetrainRequest(BaseModel):
    interactions_path: Optional[str] = Field(None, description="Path to new interactions file")
    force_retrain: bool = Field(False, description="Force retrain")

# =========================
# Global State
# =========================
class AppState:
    def __init__(self):
        self.content_df = None
        self.interactions_df = None
        self.users_df = None
        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.ml_model = None
        self.is_loaded = False
        self.last_update = None
        self.cache = {}
        self.cache_timestamps = {}

state = AppState()

# =========================
# Cache Functions
# =========================
def get_cache_key(user_id: int, top_n: int, **kwargs) -> str:
    """Generate cache key."""
    key_dict = {'user_id': user_id, 'top_n': top_n, **kwargs}
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached(user_id: int, top_n: int, **kwargs):
    """Get cached recommendations."""
    key = get_cache_key(user_id, top_n, **kwargs)
    if key in state.cache:
        if (datetime.now() - state.cache_timestamps[key]).seconds < 3600:  # 1 hour TTL
            logger.info(f"✅ Cache hit for user {user_id}")
            return state.cache[key]
        else:
            del state.cache[key]
            del state.cache_timestamps[key]
    return None

def cache_result(user_id: int, top_n: int, result: Dict, **kwargs):
    """Cache recommendations."""
    key = get_cache_key(user_id, top_n, **kwargs)
    state.cache[key] = result
    state.cache_timestamps[key] = datetime.now()
    logger.info(f"💾 Cached for user {user_id}")

# =========================
# Load Models
# =========================
def load_models():
    """Load all models and data."""
    global state
    
    logger.info("🔄 Loading data and models...")
    logger.info(f"📁 Current directory: {os.getcwd()}")
    logger.info(f"📁 Files in src/: {os.listdir('src') if os.path.exists('src') else 'src not found'}")
    
    try:
        # 1. Load data using DataLoader from src/
        loader = DataLoader(data_dir="data")
        
        # Find data files
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        content_paths = [
            os.path.join(data_dir, "content.csv"),
            "data/content.csv",
            "content.csv"
        ]
        
        interactions_paths = [
            os.path.join(data_dir, "interactions.csv"),
            "data/interactions.csv",
            "interactions.csv"
        ]
        
        users_paths = [
            os.path.join(data_dir, "users.xlsx"),
            "data/users.xlsx",
            "users.xlsx"
        ]
        
        # Load content
        content_path = None
        for path in content_paths:
            if os.path.exists(path):
                content_path = path
                break
        
        if content_path:
            state.content_df = pd.read_csv(content_path)
            logger.info(f"✅ Loaded content from {content_path} ({len(state.content_df)} items)")
        else:
            logger.warning("⚠️ No content file found")
            state.content_df = pd.DataFrame()
        
        # Load interactions
        interactions_path = None
        for path in interactions_paths:
            if os.path.exists(path):
                interactions_path = path
                break
        
        if interactions_path:
            state.interactions_df = pd.read_csv(interactions_path)
            logger.info(f"✅ Loaded interactions from {interactions_path} ({len(state.interactions_df)} interactions)")
        else:
            logger.warning("⚠️ No interactions file found")
            state.interactions_df = pd.DataFrame()
        
        # Load users
        users_path = None
        for path in users_paths:
            if os.path.exists(path):
                users_path = path
                break
        
        if users_path:
            try:
                state.users_df = pd.read_excel(users_path)
                logger.info(f"✅ Loaded users from {users_path} ({len(state.users_df)} users)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load users: {e}")
                state.users_df = pd.DataFrame()
        else:
            state.users_df = pd.DataFrame()
        
        # Check if data is empty
        if state.content_df.empty:
            logger.warning("⚠️ Content data is empty!")
        if state.interactions_df.empty:
            logger.warning("⚠️ Interactions data is empty!")
        
        # 2. Clean data (if DataCleaner is available)
        if 'DataCleaner' in globals() and not state.content_df.empty:
            try:
                cleaner = DataCleaner()
                state.content_df = cleaner.clean(state.content_df)
                logger.info("✅ Data cleaned")
            except Exception as e:
                logger.warning(f"⚠️ Data cleaning failed: {e}")
        
        # 3. Initialize content recommender
        # في load_models
        if 'ContentBased' in globals() and not state.content_df.empty:
            try:
                       content_model_path = "models/content_model.pkl"
                       if os.path.exists(content_model_path):
                                 logger.info("📂 Loading content recommender from disk...")
                                 state.content_recommender = ContentBased(df=state.content_df)
                                 state.content_recommender.load("models/content_model.pkl")
                                 logger.info("✅ Content recommender loaded")
                       else:
                                 logger.info("🔄 Creating new content recommender...")
                                 state.content_recommender = ContentBased(
                                          df=state.content_df,
                                          text_cols=['title', 'category', 'level', 'description'],
                                          use_embeddings=True
                                 )
                                 os.makedirs("models", exist_ok=True)
                                 state.content_recommender.save("models/content_model.pkl")
                                 logger.info("✅ Content recommender created and saved")
            except Exception as e:
                       logger.warning(f"⚠️ Content recommender failed: {e}")

                
        # 4. Initialize collaborative recommender
        if 'CollaborativeRecommender' in globals() and not state.interactions_df.empty:
            try:
                collab_path = "models/collaborative_model.pkl"
                if os.path.exists(collab_path):
                    logger.info("📂 Loading collaborative recommender from disk...")
                    with open(collab_path, 'rb') as f:
                        state.collaborative_recommender = pickle.load(f)
                    logger.info("✅ Collaborative recommender loaded")
                else:
                    logger.info("🔄 Creating new collaborative recommender...")
                    state.collaborative_recommender = CollaborativeRecommender(
                        interactions=state.interactions_df,
                        n_components=50
                    )
                    state.collaborative_recommender.fit()
                    with open(collab_path, 'wb') as f:
                        pickle.dump(state.collaborative_recommender, f)
                    logger.info("✅ Collaborative recommender created and saved")
            except Exception as e:
                logger.warning(f"⚠️ Collaborative recommender failed: {e}")
        
        # 5. Create hybrid recommender
        if state.content_recommender and state.collaborative_recommender:
            try:
                state.hybrid_recommender = HybridRecommender(
                    content_recommender=state.content_recommender,
                    collaborative_recommender=state.collaborative_recommender,
                    content_weight=0.5,
                    collab_weight=0.5
                )
                logger.info("✅ Hybrid recommender ready")
            except Exception as e:
                logger.warning(f"⚠️ Hybrid recommender failed: {e}")
        
        state.is_loaded = True
        state.last_update = datetime.now()
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error(f"❌ Full error: {e.__class__.__name__}: {str(e)}")
        state.is_loaded = True  # Still allow API to run in fallback mode

# =========================
# Startup Event
# =========================
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("🚀 Starting API server...")
    try:
        load_models()
        logger.info("✅ API started successfully!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down API...")

# =========================
# API Endpoints
# =========================

@app.get("/")
async def root():
    return {
        "message": "Hybrid Recommendation System API",
        "status": "ready" if state.is_loaded else "loading",
        "version": "2.0.0",
        "endpoints": {
            "POST /recommend/hybrid": "Get hybrid recommendations for a user",
            "POST /recommend/content": "Get content-based recommendations by title",
            "GET /recommend/popular": "Get popular items",
            "POST /retrain": "Retrain collaborative model",
            "POST /reload": "Reload all models",
            "GET /health": "Health check",
            "GET /stats": "System statistics",
            "POST /cache/clear": "Clear cache"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if state.is_loaded else "loading",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": state.is_loaded,
        "cache_size": len(state.cache)
    }

@app.get("/stats")
async def stats():
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "content_items": len(state.content_df) if state.content_df is not None else 0,
        "interactions": len(state.interactions_df) if state.interactions_df is not None else 0,
        "users": state.interactions_df['user_id'].nunique() if state.interactions_df is not None and 'user_id' in state.interactions_df.columns else 0,
        "categories": state.content_df['category'].nunique() if state.content_df is not None and 'category' in state.content_df.columns else 0,
        "cache_size": len(state.cache),
        "last_update": state.last_update.isoformat() if state.last_update else None
    }

@app.post("/recommend/hybrid")
async def recommend_hybrid(request: HybridRequest):
    """Get hybrid recommendations for a user."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Check cache
    cached = get_cached(
        request.user_id, 
        request.top_n,
        exclude_seen=request.exclude_seen,
        filter_category=request.filter_category,
        filter_level=request.filter_level
    )
    if cached:
        return cached
    
    try:
        logger.info(f"🔄 Using content-based + popular for user {request.user_id}")
        
        recommendations = []
        seen_items = set()
        
        # 1. Get user's history
        user_history = []
        if state.interactions_df is not None and not state.interactions_df.empty:
            user_history = state.interactions_df[
                state.interactions_df['user_id'] == request.user_id
            ]['content_id'].tolist()
            logger.info(f"📚 User {request.user_id} has {len(user_history)} interactions")
        
        # 2. Get content-based recommendations based on user's history
       # 2. Get content-based recommendations based on user's history
        if user_history and state.content_recommender:
            try:
                        first_item_id = user_history[0]
                        logger.info(f"📚 Using item {first_item_id} for content-based recommendations")
                        
                        # 🔥 استخدام recommend بدلاً من recommend_similar_to_item
                        recs = state.content_recommender.recommend(
                                    query=first_item_id,
                                    k=request.top_n * 2,
                                    return_scores=True
                        )
                        
                        # Process recommendations
                        if recs:
                                    for rec in recs:
                                                if isinstance(rec, dict):
                                                            content_id = rec.get('content_id')
                                                            if content_id and content_id not in seen_items and content_id not in user_history:
                                                                        rec['score'] = rec.get('similarity_score', 0.5)
                                                                        recommendations.append(rec)
                                                                        seen_items.add(content_id)
                                    logger.info(f"📚 Got {len(recommendations)} content-based recommendations")
                        else:
                                    logger.warning("⚠️ No content-based recommendations found")
                                    
            except Exception as e:
                        logger.error(f"❌ Content-based error: {e}")
                        logger.error(traceback.format_exc())
        
        # 3. Get popular items to fill remaining slots
        if len(recommendations) < request.top_n:
            try:
                popular = state.interactions_df.groupby('content_id').size().sort_values(ascending=False)
                popular_items = popular.head(request.top_n * 3).index.tolist()
                
                for item_id in popular_items:
                    if len(recommendations) >= request.top_n:
                        break
                    if item_id not in seen_items and item_id not in user_history:
                        item = state.content_df[state.content_df['content_id'] == item_id]
                        if not item.empty:
                            recommendations.append({
                                'content_id': int(item_id),
                                'title': item.iloc[0].get('title', f'Item {item_id}'),
                                'category': item.iloc[0].get('category', ''),
                                'level': item.iloc[0].get('level', ''),
                                'score': 0.5
                            })
                            seen_items.add(item_id)
                logger.info(f"📚 Added popular items, total: {len(recommendations)}")
            except Exception as e:
                logger.error(f"❌ Popular items error: {e}")
        
        # 4. If still no recommendations, get random items
        if not recommendations:
            try:
                random_items = state.content_df.sample(min(request.top_n, len(state.content_df)))
                for _, row in random_items.iterrows():
                    item_id = row['content_id']
                    if item_id not in user_history:
                        recommendations.append({
                            'content_id': int(item_id),
                            'title': row.get('title', f'Item {item_id}'),
                            'category': row.get('category', ''),
                            'level': row.get('level', ''),
                            'score': 0.3
                        })
                logger.info(f"📚 Added {len(recommendations)} random items")
            except Exception as e:
                logger.error(f"❌ Random items error: {e}")
        
        # 5. Apply filters
        if request.filter_category:
            recommendations = [r for r in recommendations if r.get('category') == request.filter_category]
        if request.filter_level:
            recommendations = [r for r in recommendations if r.get('level') == request.filter_level]
        
        # 6. Sort by score
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        result = {
            "user_id": request.user_id,
            "recommendations": recommendations[:request.top_n],
            "timestamp": datetime.now().isoformat(),
            "user_type": "existing_fallback",
            "total_interactions": len(user_history)
        }
        
        # Cache result
        cache_result(
            request.user_id, 
            request.top_n, 
            result,
            exclude_seen=request.exclude_seen,
            filter_category=request.filter_category,
            filter_level=request.filter_level
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error recommending for user {request.user_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/content")
async def recommend_content(request: ContentRequest):
    """Get content-based recommendations."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if state.content_recommender is None:
            raise HTTPException(status_code=503, detail="Content recommender not available")
        
        # 🔥 Search for the title in the dataframe
        search_term = request.title.lower().strip()
        
        # Find matching items
        matches = state.content_df[
            state.content_df['title'].str.lower().str.contains(search_term, na=False)
        ]
        
        recommendations = []
        
        if not matches.empty:
            # Use the first match for recommendations
            first_match_id = matches.iloc[0]['content_id']
            
            # Get recommendations using the recommend method
            recs = state.content_recommender.recommend(
                query=first_match_id,
                k=request.top_n * 2,
                return_scores=True,
                exclude_self=True
            )
            
            # Process recommendations
            if recs:
                for rec in recs:
                    if isinstance(rec, dict):
                        # Apply filters
                        if request.filter_category and rec.get('category') != request.filter_category:
                            continue
                        if request.filter_level and rec.get('level') != request.filter_level:
                            continue
                        
                        recommendations.append({
                            'content_id': rec.get('content_id'),
                            'title': rec.get('title'),
                            'category': rec.get('category', ''),
                            'level': rec.get('level', ''),
                            'similarity_score': rec.get('similarity_score', 0.0)
                        })
        
        # 🔥 If no recommendations from content, use popular items
        if not recommendations:
            logger.info(f"📚 No content recommendations for '{request.title}', using popular items")
            
            # Get popular items
            popular = state.interactions_df.groupby('content_id').size().sort_values(ascending=False)
            popular_items = popular.head(request.top_n * 2).index.tolist()
            
            for item_id in popular_items:
                if len(recommendations) >= request.top_n:
                    break
                    
                item = state.content_df[state.content_df['content_id'] == item_id]
                if not item.empty:
                    # Apply filters
                    if request.filter_category and item.iloc[0].get('category') != request.filter_category:
                        continue
                    if request.filter_level and item.iloc[0].get('level') != request.filter_level:
                        continue
                    
                    recommendations.append({
                        'content_id': int(item_id),
                        'title': item.iloc[0].get('title', f'Item {item_id}'),
                        'category': item.iloc[0].get('category', ''),
                        'level': item.iloc[0].get('level', ''),
                        'similarity_score': 0.5
                    })
        
        # 🔥 If STILL no recommendations, get random items
        if not recommendations:
            random_items = state.content_df.sample(min(request.top_n, len(state.content_df)))
            for _, row in random_items.iterrows():
                # Apply filters
                if request.filter_category and row.get('category') != request.filter_category:
                    continue
                if request.filter_level and row.get('level') != request.filter_level:
                    continue
                    
                recommendations.append({
                    'content_id': int(row['content_id']),
                    'title': row.get('title', f'Item {row["content_id"]}'),
                    'category': row.get('category', ''),
                    'level': row.get('level', ''),
                    'similarity_score': 0.3
                })
        
        # Return results
        return {
            "input": request.title,
            "matched_course": matches.iloc[0]['title'] if not matches.empty else request.title,
            "recommendations": recommendations[:request.top_n],
            "timestamp": datetime.now().isoformat(),
            "total_found": len(recommendations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error recommending for title '{request.title}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/popular")
async def get_popular(
    top_n: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get most popular items."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if state.interactions_df is None or state.interactions_df.empty:
            return {"popular_items": [], "timestamp": datetime.now().isoformat()}
        
        popular = state.interactions_df.groupby('content_id')['rating'].agg(['count', 'mean'])
        popular = popular.sort_values(['count', 'mean'], ascending=False)
        
        result = popular.head(top_n).reset_index()
        
        if state.content_df is not None and 'content_id' in state.content_df.columns:
            result = result.merge(
                state.content_df[['content_id', 'title', 'category', 'level']], 
                on='content_id',
                how='left'
            )
        
        if category and 'category' in result.columns:
            result = result[result['category'] == category]
        
        return {
            "popular_items": result.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting popular items: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Retrain the collaborative model."""
    global state
    
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if request.interactions_path and os.path.exists(request.interactions_path):
            new_data = pd.read_csv(request.interactions_path)
            updated_interactions = pd.concat([state.interactions_df, new_data])
        else:
            if not request.force_retrain:
                raise HTTPException(
                    status_code=400,
                    detail="No new data provided. Use force_retrain=True to retrain with existing data"
                )
            updated_interactions = state.interactions_df
        
        if state.collaborative_recommender is None:
            raise HTTPException(status_code=503, detail="Collaborative recommender not available")
        
        # Retrain in background if large
        if len(updated_interactions) > 100000:
            background_tasks.add_task(retrain_background, updated_interactions)
            return {
                "message": "🔄 Retraining started in background",
                "status": "processing"
            }
        else:
            # Retrain synchronously
            state.collaborative_recommender.fit(updated_interactions)
            state.interactions_df = updated_interactions
            state.last_update = datetime.now()
            
            # Save model
            with open("models/collaborative_model.pkl", 'wb') as f:
                pickle.dump(state.collaborative_recommender, f)
            
            # Clear cache
            state.cache.clear()
            state.cache_timestamps.clear()
            
            return {
                "message": "✅ Model retrained successfully!",
                "interactions": len(updated_interactions),
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_background(updated_interactions):
    """Background retraining task."""
    global state
    try:
        logger.info("🔄 Background retraining started...")
        
        # 🔥 Create a new instance of CollaborativeRecommender
        from Collaborative import CollaborativeRecommender
        
        # Create new model with the updated interactions
        new_collab = CollaborativeRecommender(
            interactions=updated_interactions,
            n_components=50
        )
        
        # Fit the new model
        new_collab.fit(updated_interactions)
        
        # Save the new model
        new_collab.save("models/")
        
        # Update the global state
        state.collaborative_recommender = new_collab
        state.interactions_df = updated_interactions
        state.last_update = datetime.now()
        
        # Clear cache
        state.cache.clear()
        state.cache_timestamps.clear()
        
        logger.info("✅ Background retraining completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Background retraining failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
@app.post("/reload")
async def reload():
    """Reload all models."""
    try:
        state.cache.clear()
        state.cache_timestamps.clear()
        load_models()
        return {"message": "✅ Models reloaded successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"❌ Error reloading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached recommendations."""
    state.cache.clear()
    state.cache_timestamps.clear()
    return {"message": "✅ Cache cleared successfully", "timestamp": datetime.now().isoformat()}

# =========================
# Exception Handlers
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("DEBUG", "False").lower() == "true" else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )