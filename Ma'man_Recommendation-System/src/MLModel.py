import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLModel:
    """
    Advanced ML model for ranking and re-ranking recommendations.
    
    Supports multiple algorithms with feature engineering and evaluation.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        task: str = 'regression',  # 'regression' or 'classification'
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize the ML model.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'linear')
            task: 'regression' or 'classification'
            hyperparameters: Custom hyperparameters for the model
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.task = task
        self.random_state = random_state
        self.hyperparameters = hyperparameters or {}
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
        
        # Initialize model
        self._init_model()
        
        logger.info(f"✅ MLModel initialized: {model_type} ({task})")
    
    def _init_model(self):
        """Initialize the underlying ML model."""
        if self.model_type == 'random_forest':
            if self.task == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(
                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                    max_depth=self.hyperparameters.get('max_depth', None),
                    min_samples_split=self.hyperparameters.get('min_samples_split', 2),
                    min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                    max_depth=self.hyperparameters.get('max_depth', None),
                    min_samples_split=self.hyperparameters.get('min_samples_split', 2),
                    min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
                    random_state=self.random_state,
                    n_jobs=-1
                )
        
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                if self.task == 'regression':
                    self.model = xgb.XGBRegressor(
                        n_estimators=self.hyperparameters.get('n_estimators', 100),
                        max_depth=self.hyperparameters.get('max_depth', 6),
                        learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    self.model = xgb.XGBClassifier(
                        n_estimators=self.hyperparameters.get('n_estimators', 100),
                        max_depth=self.hyperparameters.get('max_depth', 6),
                        learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                        random_state=self.random_state,
                        n_jobs=-1
                    )
            except ImportError:
                logger.warning("⚠️ XGBoost not installed, falling back to Random Forest")
                self.model_type = 'random_forest'
                self._init_model()
        
        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                if self.task == 'regression':
                    self.model = lgb.LGBMRegressor(
                        n_estimators=self.hyperparameters.get('n_estimators', 100),
                        max_depth=self.hyperparameters.get('max_depth', -1),
                        learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    self.model = lgb.LGBMClassifier(
                        n_estimators=self.hyperparameters.get('n_estimators', 100),
                        max_depth=self.hyperparameters.get('max_depth', -1),
                        learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                        random_state=self.random_state,
                        n_jobs=-1
                    )
            except ImportError:
                logger.warning("⚠️ LightGBM not installed, falling back to Random Forest")
                self.model_type = 'random_forest'
                self._init_model()
        
        elif self.model_type == 'linear':
            if self.task == 'regression':
                from sklearn.linear_model import Ridge
                self.model = Ridge(
                    alpha=self.hyperparameters.get('alpha', 1.0),
                    random_state=self.random_state
                )
            else:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(
                    C=self.hyperparameters.get('C', 1.0),
                    random_state=self.random_state,
                    max_iter=1000
                )
        
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        categorical_features: Optional[List[str]] = None,
        eval_set: Optional[Tuple] = None
    ) -> 'MLModel':
        """
        Train the model with feature engineering.
        
        Args:
            X: Feature matrix
            y: Target values
            categorical_features: List of categorical column names
            eval_set: Tuple of (X_val, y_val) for early stopping
            
        Returns:
            self: Trained model instance
        """
        logger.info("🔄 Training ML model...")
        
        # Convert to DataFrame if numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
        
        # Handle categorical features
        if categorical_features:
            X = self._encode_categorical(X, categorical_features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"✅ Model trained successfully!")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("❌ Model not fitted. Call fit() first.")
        
        # Convert to DataFrame if numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure columns match training
        if hasattr(X, 'columns') and self.feature_columns:
            # Add missing columns
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            # Reorder columns
            X = X[self.feature_columns]
        
        # Encode categorical features
        X = self._encode_categorical(X, list(self.label_encoders.keys()))
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities (for classification).
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array of prediction probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(self._preprocess(X))
        else:
            return self.model.predict(self._preprocess(X))
    
    def _preprocess(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Preprocess features before prediction."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Ensure columns match
        if hasattr(X, 'columns') and self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]
        
        # Encode categorical
        X = self._encode_categorical(X, list(self.label_encoders.keys()))
        
        # Scale
        return self.scaler.transform(X)
    
    def _encode_categorical(
        self,
        X: pd.DataFrame,
        categorical_features: List[str]
    ) -> pd.DataFrame:
        """Encode categorical features."""
        X = X.copy()
        
        for col in categorical_features:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Transform new data
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        X[col] = X[col].astype(str).apply(
                            lambda x: self.label_encoders[col].transform([x])[0]
                            if x in self.label_encoders[col].classes_
                            else -1
                        )
        
        return X
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {}
        
        if self.task == 'regression':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = 1 - (metrics['mse'] / np.var(y))
        else:
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1_macro'] = f1_score(y, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted')
        
        logger.info(f"📊 Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Directory path to save model files
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'task': self.task,
            'random_state': self.random_state,
            'hyperparameters': self.hyperparameters,
            'is_fitted': self.is_fitted
        }
        
        with open(os.path.join(path, 'ml_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to {path}")
    
    def load(self, path: str) -> 'MLModel':
        """
        Load a model from disk.
        
        Args:
            path: Directory path containing model files
            
        Returns:
            self: Loaded model instance
        """
        with open(os.path.join(path, 'ml_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore attributes
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.random_state = model_data['random_state']
        self.hyperparameters = model_data['hyperparameters']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"✅ Model loaded from {path}")
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            return None
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return None
        
        # Create DataFrame
        if self.feature_columns:
            return pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return None
    
    def rank_items(
        self,
        items: pd.DataFrame,
        user_features: Optional[pd.DataFrame] = None,
        context_features: Optional[pd.DataFrame] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank items for a user.
        
        Args:
            items: DataFrame with item features
            user_features: User features (optional)
            context_features: Contextual features (optional)
            top_n: Number of top items to return
            
        Returns:
            DataFrame with ranked items
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Prepare features
        X = items.copy()
        
        # Add user features if provided
        if user_features is not None:
            for col in user_features.columns:
                if col not in X.columns:
                    X[col] = user_features.iloc[0][col] if len(user_features) > 0 else 0
        
        # Add context features if provided
        if context_features is not None:
            for col in context_features.columns:
                if col not in X.columns:
                    X[col] = context_features.iloc[0][col] if len(context_features) > 0 else 0
        
        # Predict scores
        scores = self.predict(X)
        
        # Add scores to DataFrame
        result = items.copy()
        result['score'] = scores
        
        # Sort and return top N
        return result.sort_values('score', ascending=False).head(top_n)
    
    def rerank_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        user_features: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank existing recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            user_features: User features for context
            
        Returns:
            Re-ranked list of recommendations
        """
        if not recommendations:
            return []
        
        # Convert to DataFrame
        recs_df = pd.DataFrame(recommendations)
        
        # Add user features
        if user_features is not None:
            for col in user_features.columns:
                recs_df[col] = user_features.iloc[0][col] if len(user_features) > 0 else 0
        
        # Predict scores
        scores = self.predict(recs_df)
        
        # Add scores and re-rank
        recs_df['ml_score'] = scores
        recs_df = recs_df.sort_values('ml_score', ascending=False)
        
        # Convert back to list of dicts
        return recs_df.to_dict('records')
    
    def create_features(
        self,
        interactions: pd.DataFrame,
        content_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create features for training.
        
        Args:
            interactions: Interaction data
            content_df: Content metadata
            users_df: User metadata
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Creating features...")
        
        # Merge interactions with content
        features = interactions.merge(content_df, on='content_id')
        
        # Add user features if available
        if users_df is not None and 'user_id' in users_df.columns:
            features = features.merge(users_df, on='user_id')
        
        # Feature engineering
        # 1. Content popularity
        content_popularity = interactions.groupby('content_id')['rating'].agg(['count', 'mean'])
        content_popularity.columns = ['interaction_count', 'avg_rating']
        features = features.merge(content_popularity, on='content_id', how='left')
        
        # 2. User activity
        user_activity = interactions.groupby('user_id')['rating'].agg(['count', 'mean'])
        user_activity.columns = ['user_interaction_count', 'user_avg_rating']
        features = features.merge(user_activity, on='user_id', how='left')
        
        # 3. Category features
        if 'category' in features.columns:
            category_popularity = interactions.merge(content_df[['content_id', 'category']], on='content_id')
            category_popularity = category_popularity.groupby('category')['rating'].mean()
            features['category_avg_rating'] = features['category'].map(category_popularity)
        
        # 4. Time-based features
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['hour'] = features['timestamp'].dt.hour
            features['month'] = features['timestamp'].dt.month
        
        # 5. Interaction features
        features['user_content_interaction'] = features.groupby(['user_id', 'content_id']).cumcount() + 1
        
        # Fill missing values
        features = features.fillna(0)
        
        logger.info(f"✅ Created {features.shape[1]} features for {len(features)} samples")
        
        return features


# =========================
# Usage Example
# =========================

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Sample interactions
    interactions = pd.DataFrame({
        'user_id': np.random.randint(1, 50, 1000),
        'content_id': np.random.randint(1, 100, 1000),
        'rating': np.random.randint(1, 6, 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    
    # Sample content
    content = pd.DataFrame({
        'content_id': range(1, 101),
        'title': [f'Course {i}' for i in range(1, 101)],
        'category': np.random.choice(['Programming', 'Data Science', 'Design'], 100),
        'level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 100),
        'description': [f'Description {i}' for i in range(1, 101)]
    })
    
    # Sample users
    users = pd.DataFrame({
        'user_id': range(1, 51),
        'age': np.random.randint(18, 65, 50),
        'country': np.random.choice(['USA', 'UK', 'Egypt', 'UAE'], 50)
    })
    
    # Example 1: Training a model
    print("\n🔹 Training ML Model:")
    
    # Create ML model
    ml_model = MLModel(
        model_type='random_forest',
        task='regression',
        hyperparameters={'n_estimators': 50, 'max_depth': 10}
    )
    
    # Create features
    features = ml_model.create_features(interactions, content, users)
    
    # Prepare training data
    X = features.drop(['rating'], axis=1)
    y = features['rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    ml_model.fit(
        X_train, y_train,
        categorical_features=['category', 'level', 'country', 'day_of_week']
    )
    
    # Evaluate
    metrics = ml_model.evaluate(X_test, y_test)
    print(f"Metrics: {metrics}")
    
    # Feature importance
    importance = ml_model.get_feature_importance()
    if importance is not None:
        print(f"\nTop features:\n{importance.head(5)}")
    
    # Example 2: Re-ranking recommendations
    print("\n🔹 Re-ranking recommendations:")
    
    # Create some recommendations
    recommendations = [
        {'content_id': 1, 'title': 'Course 1', 'category': 'Programming'},
        {'content_id': 2, 'title': 'Course 2', 'category': 'Data Science'},
        {'content_id': 3, 'title': 'Course 3', 'category': 'Design'},
        {'content_id': 4, 'title': 'Course 4', 'category': 'Programming'},
        {'content_id': 5, 'title': 'Course 5', 'category': 'Data Science'}
    ]
    
    # Re-rank
    user_features = pd.DataFrame({'user_id': [1], 'age': [25], 'country': ['USA']})
    reranked = ml_model.rerank_recommendations(recommendations, user_features)
    
    for rec in reranked:
        print(f"  - {rec['title']} (score: {rec.get('ml_score', 'N/A'):.2f})")
    
    # Example 3: Save and load model
    print("\n🔹 Saving and loading model:")
    ml_model.save('models/')
    
    loaded_model = MLModel()
    loaded_model.load('models/')
    print("Model loaded successfully!")