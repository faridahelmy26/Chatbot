import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union, Dict, Any
import pickle
import os
import re


class ContentBased:
    """
    Content-based recommender using TF-IDF or Sentence Transformers.
    
    Supports multiple text columns, customizable weights, and model persistence.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_cols: Optional[List[str]] = None,
        use_embeddings: bool = False,
        embedding_model: str = 'all-MiniLM-L6-v2',
        tfidf_params: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the content-based recommender.
        
        Args:
            df: DataFrame containing content data
            text_cols: Columns to use for text representation
            use_embeddings: Use Sentence Transformers instead of TF-IDF
            embedding_model: Sentence Transformer model name
            tfidf_params: Parameters for TfidfVectorizer
            weights: Custom weights for each text column
        """
        self.df = df.copy()
        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model
        
        # Default text columns
        if text_cols is None:
            text_cols = ['title', 'category', 'level', 'description']
        
        # Filter existing columns
        self.text_cols = [col for col in text_cols if col in self.df.columns]
        
        # Set weights (default: equal weights)
        if weights is None:
            self.weights = {col: 1.0 for col in self.text_cols}
        else:
            self.weights = {col: weights.get(col, 1.0) for col in self.text_cols}
        
        # Fill NaN and convert to string
        for col in self.text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)
        
        # Create combined text with weights
        self._prepare_text()
        
        # Initialize model
        if use_embeddings:
            self._init_embedding_model()
        else:
            self._init_tfidf_model(tfidf_params)
        
        print(f"✅ Content-based model initialized!")
        print(f"   Text columns: {self.text_cols}")
        print(f"   Method: {'Embeddings' if use_embeddings else 'TF-IDF'}")
        print(f"   Items: {len(self.df)}")
    
    def _prepare_text(self):
        """Prepare combined text with column weights."""
        weighted_texts = []
        
        for idx, row in self.df.iterrows():
            parts = []
            for col in self.text_cols:
                weight = self.weights.get(col, 1.0)
                text = str(row[col])
                # Repeat text based on weight (simple weighting approach)
                if weight > 1:
                    text = (text + " ") * int(weight)
                parts.append(text)
            weighted_texts.append(" ".join(parts))
        
        self.df['combined_text'] = weighted_texts
    
    def _init_tfidf_model(self, tfidf_params: Optional[Dict[str, Any]] = None):
        """Initialize TF-IDF vectorizer."""
        default_params = {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.9
        }
        
        if tfidf_params:
            default_params.update(tfidf_params)
        
        self.vectorizer = TfidfVectorizer(**default_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        self.model_type = 'tfidf'
    
    def _init_embedding_model(self):
        """Initialize Sentence Transformer model."""
        print(f"🔄 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_matrix = self.embedding_model.encode(
            self.df['combined_text'].tolist(),
            show_progress_bar=True
        )
        self.model_type = 'embeddings'
    
    def recommend(
        self,
        query: Union[str, int],
        k: int = 5,
        return_scores: bool = False,
        exclude_self: bool = True,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Get content-based recommendations.
        
        Args:
            query: Title (str) or content_id (int)
            k: Number of recommendations
            return_scores: Include similarity scores in results
            exclude_self: Exclude the item itself from recommendations
            filter_condition: Filter items before recommendation
                Example: {'category': 'Programming', 'level': 'Beginner'}
            
        Returns:
            List of recommended titles or list of dictionaries with details
        """
        # Get query vector
        query_vec = self._get_query_vector(query)
        
        if query_vec is None:
            return []
        
        # Get similarity scores
        if self.model_type == 'tfidf':
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:  # embeddings
            similarities = cosine_similarity(query_vec, self.embedding_matrix).flatten()
        
        # Get index of query item (if query is content_id)
        query_idx = None
        if isinstance(query, int):
            query_idx = self.df[self.df['content_id'] == query].index[0] if query in self.df['content_id'].values else None
        
        # Apply filters
        indices_to_include = self._apply_filters(filter_condition)
        
        # Create results
        results = []
        for idx, score in enumerate(similarities):
            # Skip if filtered out
            if indices_to_include is not None and idx not in indices_to_include:
                continue
            
            # Skip self
            if exclude_self and query_idx is not None and idx == query_idx:
                continue
            
            # Skip if title is same (for text queries)
            if exclude_self and isinstance(query, str):
                if self.df.iloc[idx]['title'].lower() == query.lower():
                    continue
            
            # Get item details
            item = {
                'content_id': int(self.df.iloc[idx]['content_id']),
                'title': self.df.iloc[idx]['title'],
                'category': self.df.iloc[idx].get('category', ''),
                'level': self.df.iloc[idx].get('level', ''),
                'similarity_score': float(score)
            }
            results.append(item)
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top-k
        top_results = results[:k]
        
        if return_scores:
            return top_results
        else:
            return [item['title'] for item in top_results]
        

    def recommend_by_title(
        self,
        title: str,
        k: int = 5,
        return_scores: bool = False,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Get recommendations based on a title with fuzzy matching.
        
        Args:
               title: Title to search for
               k: Number of recommendations
               return_scores: Include similarity scores
               filter_category: Filter by category
               filter_level: Filter by level
               
        Returns:
               List of recommendations
        """
        # 🔥 Use fuzzy search
        return self.search_by_text_fuzzy(
               text=title,
               k=k,
               return_scores=return_scores,
               filter_category=filter_category,
               filter_level=filter_level
        )
    
    def _get_query_vector(self, query: Union[str, int]):
        """Convert query to vector."""
        # If query is content_id
        if isinstance(query, int):
            idx = self.df[self.df['content_id'] == query].index
            if len(idx) == 0:
                return None
            if self.model_type == 'tfidf':
                return self.tfidf_matrix[idx[0]]
            else:
                return self.embedding_matrix[idx[0]].reshape(1, -1)
        
        # If query is text
        query_text = str(query).lower().strip()
        if not query_text:
            return None
        
        if self.model_type == 'tfidf':
            return self.vectorizer.transform([query_text])
        else:
            return self.embedding_model.encode([query_text])
    
    def _apply_filters(self, filter_condition: Optional[Dict[str, Any]] = None):
        """Apply filters to items."""
        if filter_condition is None:
            return None
        
        indices = self.df.index
        for col, value in filter_condition.items():
            if col in self.df.columns:
                if isinstance(value, list):
                    mask = self.df[col].isin(value)
                else:
                    mask = self.df[col] == value
                indices = indices[mask]
        
        return indices.tolist()
    
    def recommend_similar_to_item(
        self,
        content_id: int,
        k: int = 5,
        return_scores: bool = False
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Get recommendations similar to a specific content item.
        
        Args:
            content_id: ID of the content item
            k: Number of recommendations
            return_scores: Include similarity scores
            
        Returns:
            List of recommended items
        """
        return self.recommend(
            query=content_id,
            k=k,
            return_scores=return_scores,
            exclude_self=True
        )

    def search_by_text_fuzzy(
        self,
        text: str,
        k: int = 5,
        return_scores: bool = False,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
) -> List[Dict[str, Any]]:
        """
        Search for items matching text query with fuzzy matching.
        
        Args:
              text: Search query
              k: Number of results
              return_scores: Include similarity scores
              filter_category: Filter by category
              filter_level: Filter by level
              
        Returns:
              List of matched items
        """
        text = text.lower().strip()
        results = []
        
        # Apply filters
        df_filtered = self.df.copy()
        if filter_category:
              df_filtered = df_filtered[df_filtered['category'] == filter_category]
        if filter_level:
              df_filtered = df_filtered[df_filtered['level'] == filter_level]
        
        if df_filtered.empty:
              return []
        
        # Method 1: Exact or partial match on title
        matches = df_filtered[
              df_filtered['title'].str.lower().str.contains(text, na=False)
        ]
        
        if not matches.empty:
              # Get recommendations for each match
              for _, row in matches.head(3).iterrows():
                      similar = self.recommend(
                            query=row['content_id'],
                            k=k,
                            return_scores=True,
                            exclude_self=True
                      )
                      if similar:
                            results.extend(similar)
        
        # Method 2: If no matches, use text search
        if not results:
              # Use the existing search_by_text
              results = self.search_by_text(
                      text=text,
                      k=k,
                      return_scores=True
              )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for item in results:
              if isinstance(item, dict):
                      item_id = item.get('content_id')
              elif isinstance(item, int):
                      item_id = item
              else:
                      continue
                      
              if item_id not in seen:
                      seen.add(item_id)
                      unique_results.append(item)
        
        return unique_results[:k]    
    
    def get_similar_items(
        self,
        content_id: int,
        k: int = 5
    ) -> List[int]:
        """
        Get similar item IDs.
        
        Args:
            content_id: Content ID
            k: Number of similar items
            
        Returns:
            List of content IDs
        """
        results = self.recommend_similar_to_item(content_id, k, return_scores=True)
        return [item['content_id'] for item in results]
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Directory path to save model files
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model data
        model_data = {
            'df': self.df,
            'text_cols': self.text_cols,
            'weights': self.weights,
            'use_embeddings': self.use_embeddings,
            'embedding_model_name': self.embedding_model_name,
            'model_type': self.model_type
        }
        
        # Save TF-IDF model
        if self.model_type == 'tfidf':
            model_data['tfidf_matrix'] = self.tfidf_matrix
            with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            model_data['embedding_matrix'] = self.embedding_matrix
        
        # Save main model
        with open(os.path.join(path, 'content_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Content-based model saved to {path}")
    
    def load(self, path: str) -> 'ContentBased':
        """
        Load a model from disk.
        
        Args:
            path: Directory path containing model files
            
        Returns:
            self: Loaded model instance
        """
        with open(os.path.join(path, 'content_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore attributes
        self.df = model_data['df']
        self.text_cols = model_data['text_cols']
        self.weights = model_data['weights']
        self.use_embeddings = model_data['use_embeddings']
        self.embedding_model_name = model_data['embedding_model_name']
        self.model_type = model_data['model_type']
        
        # Load model components
        if self.model_type == 'tfidf':
            with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.tfidf_matrix = model_data.get('tfidf_matrix')
        else:
            self.embedding_matrix = model_data['embedding_matrix']
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        print(f"✅ Content-based model loaded from {path}")
        return self
    
    def add_item(self, item: Dict[str, Any]):
        """
        Add a new item to the model (incremental update).
        
        Args:
            item: Dictionary with item data
        """
        # Add to DataFrame
        new_row = pd.DataFrame([item])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Prepare text
        for col in self.text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)
        
        self._prepare_text()
        
        # Update model
        if self.model_type == 'tfidf':
            # For TF-IDF, need to retrain (or use incremental learning)
            self.vectorizer.fit(self.df['combined_text'])
            self.tfidf_matrix = self.vectorizer.transform(self.df['combined_text'])
        else:
            # For embeddings, recompute
            self.embedding_matrix = self.embedding_model.encode(
                self.df['combined_text'].tolist(),
                show_progress_bar=True
            )
        
        print(f"✅ Item added and model updated!")
    
    def get_item_count(self) -> int:
        """Get number of items in the model."""
        return len(self.df)
    
    def get_item_details(self, content_id: int) -> Optional[Dict[str, Any]]:
        """Get details for a specific item."""
        item = self.df[self.df['content_id'] == content_id]
        if len(item) == 0:
            return None
        return item.iloc[0].to_dict()


# =========================
# Usage Example
# =========================

if __name__ == "__main__":
    # Load data
    content_df = pd.read_csv("data/content.csv")
    
    # Example 1: Using TF-IDF
    print("\n🔹 Using TF-IDF:")
    cb_tfidf = ContentBased(
        content_df,
        text_cols=['title', 'category', 'level', 'description'],
        use_embeddings=False,
        weights={'title': 2.0, 'category': 1.0, 'level': 1.0, 'description': 1.5}
    )
    
    recommendations = cb_tfidf.recommend("Python", k=5, return_scores=True)
    print(f"\nRecommendations: {recommendations}")
    
    # Example 2: Using Sentence Embeddings
    print("\n🔹 Using Sentence Embeddings:")
    cb_emb = ContentBased(
        content_df,
        text_cols=['title', 'category', 'level', 'description'],
        use_embeddings=True,
        embedding_model='all-MiniLM-L6-v2'
    )
    
    recommendations_emb = cb_emb.recommend("Machine Learning", k=5, return_scores=True)
    print(f"\nRecommendations: {recommendations_emb}")
    
    # Example 3: Search by text
    print("\n🔹 Search by text:")
    search_results = cb_tfidf.search_by_text("data science", k=3, return_scores=True)
    print(f"\nSearch results: {search_results}")
    
    # Example 4: Filtered recommendations
    print("\n🔹 Filtered recommendations:")
    filtered = cb_tfidf.recommend(
        "Python",
        k=5,
        filter_condition={'category': 'Programming', 'level': 'Beginner'},
        return_scores=True
    )
    print(f"\nFiltered results: {filtered}")
    
    # Save and load
    cb_tfidf.save("models/content_based/")
    
    cb_loaded = ContentBased(content_df)
    cb_loaded.load("models/content_based/")
    
    # Get similar items
    similar = cb_loaded.get_similar_items(content_id=10, k=3)
    print(f"\nSimilar items to content 10: {similar}")