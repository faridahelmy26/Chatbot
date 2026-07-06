import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    """
    
    def __init__(
        self,
        content_recommender,
        collaborative_recommender,
        content_weight: float = 0.5,
        collab_weight: float = 0.5
    ):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        
        logger.info(f"✅ HybridRecommender initialized with weights: content={content_weight}, collab={collab_weight}")
    
    def recommend(
        self,
        user_id: int,
        top_n: int = 5,
        exclude_seen: bool = True,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get hybrid recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations
            exclude_seen: Exclude items user has seen
            filter_category: Filter by category
            filter_level: Filter by level
            
        Returns:
            List of recommendations
        """
        try:
            # 1. Get collaborative recommendations
            collab_recs = []
            collab_scores = {}
            
            if self.collaborative_recommender:
                try:
                    # Get collaborative recommendations
                    collab_items = self.collaborative_recommender.recommend_for_user(
                        user_id=user_id,
                        k=top_n * 2
                    )
                    
                    # Get scores for each item
                    for item_id in collab_items:
                        score = self.collaborative_recommender.predict_rating(user_id, item_id)
                        collab_scores[item_id] = float(score) if score else 0.0
                    
                    logger.info(f"✅ Got {len(collab_items)} collaborative recommendations for user {user_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Collaborative recommendation failed: {e}")
            
            # 2. Get content-based recommendations
            content_recs = []
            content_scores = {}
            
            if self.content_recommender:
                try:
                    # Get user's history for content-based
                    if hasattr(self.content_recommender, 'df'):
                        content_df = self.content_recommender.df
                        # Get content-based recommendations
                        # Use the most popular items as fallback
                        recs = self.content_recommender.recommend_by_title(
                            title="",  # Fallback to popular
                            k=top_n * 2
                        )
                        
                        if recs:
                            content_recs = recs
                            # Extract scores if available
                            if isinstance(recs, list) and len(recs) > 0:
                                if isinstance(recs[0], dict):
                                    for rec in recs:
                                        if 'content_id' in rec:
                                            content_scores[rec['content_id']] = rec.get('score', 0.5)
                        logger.info(f"✅ Got {len(content_recs)} content recommendations")
                except Exception as e:
                    logger.warning(f"⚠️ Content recommendation failed: {e}")
            
            # 3. Combine recommendations
            all_item_ids = set()
            
            # Collect all item IDs from both sources
            if collab_recs:
                all_item_ids.update(collab_recs)
            if content_recs:
                for rec in content_recs:
                    if isinstance(rec, dict) and 'content_id' in rec:
                        all_item_ids.add(rec['content_id'])
                    elif isinstance(rec, int):
                        all_item_ids.add(rec)
            
            logger.info(f"📊 Total unique items: {len(all_item_ids)}")
            
            # If no items found, use popular items as fallback
            if not all_item_ids:
                logger.warning(f"⚠️ No recommendations found for user {user_id}, using popular items")
                all_item_ids = self._get_popular_items(top_n * 2)
            
            # 4. Build final recommendations
            recommendations = []
            
            for item_id in all_item_ids:
                # Get content score (if available)
                content_score = content_scores.get(item_id, 0.0)
                
                # Get collaborative score (if available)
                collab_score = collab_scores.get(item_id, 0.0)
                
                # If both scores are 0, use content similarity
                if content_score == 0 and collab_score == 0:
                    if self.content_recommender and hasattr(self.content_recommender, 'df'):
                        # Check if item exists in content_df
                        if 'content_id' in self.content_recommender.df.columns:
                            item_row = self.content_recommender.df[
                                self.content_recommender.df['content_id'] == item_id
                            ]
                            if not item_row.empty:
                                # Use a small default score
                                content_score = 0.3
                
                # Hybrid score
                hybrid_score = (
                    self.content_weight * content_score +
                    self.collab_weight * collab_score
                )
                
                # Get item details
                item_details = self._get_item_details(item_id)
                
                if item_details:
                    # Apply filters
                    if filter_category and item_details.get('category') != filter_category:
                        continue
                    if filter_level and item_details.get('level') != filter_level:
                        continue
                    
                    recommendations.append({
                        'content_id': int(item_id),
                        'title': item_details.get('title', f'Item {item_id}'),
                        'category': item_details.get('category', ''),
                        'level': item_details.get('level', ''),
                        'score': float(hybrid_score),
                        'content_score': float(content_score),
                        'collab_score': float(collab_score)
                    })
            
            # Sort by score (descending)
            recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
            
            # Exclude seen items
            if exclude_seen and self.collaborative_recommender:
                try:
                    seen_items = self._get_user_history(user_id)
                    recommendations = [
                        r for r in recommendations 
                        if r['content_id'] not in seen_items
                    ]
                    logger.info(f"📍 Excluded {len(seen_items)} seen items")
                except Exception as e:
                    logger.warning(f"⚠️ Could not exclude seen items: {e}")
            
            # Return top N
            final_recs = recommendations[:top_n]
            logger.info(f"✅ Returning {len(final_recs)} recommendations for user {user_id}")
            
            return final_recs
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid recommendation: {e}")
            return []
    
    def _get_user_history(self, user_id: int) -> List[int]:
        """Get user's interaction history."""
        if hasattr(self.collaborative_recommender, 'interactions'):
            interactions = self.collaborative_recommender.interactions
            if interactions is not None and 'content_id' in interactions.columns:
                history = interactions[
                    interactions['user_id'] == user_id
                ]['content_id'].tolist()
                return history
        return []
    
    def _get_item_details(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get item details from content recommender."""
        if self.content_recommender and hasattr(self.content_recommender, 'df'):
            df = self.content_recommender.df
            if 'content_id' in df.columns:
                item = df[df['content_id'] == item_id]
                if not item.empty:
                    row = item.iloc[0]
                    return {
                        'content_id': int(row['content_id']),
                        'title': row.get('title', f'Item {item_id}'),
                        'category': row.get('category', ''),
                        'level': row.get('level', '')
                    }
        return None
    
    def _get_popular_items(self, n: int = 10) -> List[int]:
        """Get popular items as fallback."""
        if self.collaborative_recommender and hasattr(self.collaborative_recommender, 'interactions'):
            interactions = self.collaborative_recommender.interactions
            if interactions is not None:
                popular = interactions.groupby('content_id').size().sort_values(ascending=False)
                return popular.head(n).index.tolist()
        
        # If still no items, get from content recommender
        if self.content_recommender and hasattr(self.content_recommender, 'df'):
            df = self.content_recommender.df
            if 'content_id' in df.columns:
                return df['content_id'].head(n).tolist()
        
        return []