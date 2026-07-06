import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Advanced data loader for recommendation system datasets.
    
    Supports loading from CSV, Excel, JSON, and Parquet formats.
    Includes data validation, caching, and error handling.
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = "data",
        use_cache: bool = True,
        validate_columns: bool = True
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing data files
            use_cache: Enable caching for faster repeated loads
            validate_columns: Validate required columns exist
        """
        self.data_dir = Path(data_dir)
        self.use_cache = use_cache
        self.validate_columns = validate_columns
        self._cache = {}
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📂 DataLoader initialized with directory: {self.data_dir}")
    
    # =========================
    # Main Loading Methods
    # =========================
    
    def load_content(
        self,
        path: Optional[Union[str, Path]] = None,
        required_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load content metadata.
        
        Args:
            path: Path to content file (default: data/content.csv)
            required_cols: List of required columns
            **kwargs: Additional arguments for pd.read_*
            
        Returns:
            DataFrame with content metadata
        """
        if path is None:
            path = self.data_dir / "content.csv"
        else:
            path = Path(path)
        
        default_required = ['content_id', 'title', 'category', 'level']
        required_cols = required_cols or default_required
        
        return self._load_file(
            path,
            required_cols=required_cols,
            file_type='content',
            **kwargs
        )
    
    def load_interactions(
        self,
        path: Optional[Union[str, Path]] = None,
        required_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load user-item interactions.
        
        Args:
            path: Path to interactions file (default: data/interactions.csv)
            required_cols: List of required columns
            **kwargs: Additional arguments for pd.read_*
            
        Returns:
            DataFrame with interactions
        """
        if path is None:
            path = self.data_dir / "interactions.csv"
        else:
            path = Path(path)
        
        default_required = ['user_id', 'content_id', 'rating']
        required_cols = required_cols or default_required
        
        return self._load_file(
            path,
            required_cols=required_cols,
            file_type='interactions',
            **kwargs
        )
    
    def load_users(
        self,
        path: Optional[Union[str, Path]] = None,
        required_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load user metadata.
        
        Args:
            path: Path to users file (default: data/users.xlsx)
            required_cols: List of required columns
            **kwargs: Additional arguments for pd.read_*
            
        Returns:
            DataFrame with user metadata
        """
        if path is None:
            # Try multiple formats
            possible_paths = [
                self.data_dir / "users.xlsx",
                self.data_dir / "users.csv",
                self.data_dir / "users.json"
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
            
            if path is None:
                raise FileNotFoundError(f"No users file found in {self.data_dir}")
        else:
            path = Path(path)
        
        default_required = ['user_id']
        required_cols = required_cols or default_required
        
        return self._load_file(
            path,
            required_cols=required_cols,
            file_type='users',
            **kwargs
        )
    
    def load_all(
        self,
        content_path: Optional[Union[str, Path]] = None,
        interactions_path: Optional[Union[str, Path]] = None,
        users_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets at once.
        
        Returns:
            Dictionary with 'content', 'interactions', 'users' DataFrames
        """
        logger.info("🔄 Loading all datasets...")
        
        data = {
            'content': self.load_content(content_path),
            'interactions': self.load_interactions(interactions_path),
        }
        
        try:
            data['users'] = self.load_users(users_path)
        except FileNotFoundError:
            logger.warning("⚠️ Users file not found, proceeding without it")
            data['users'] = pd.DataFrame()
        
        logger.info(f"✅ Loaded {len(data['content'])} content items, "
                   f"{len(data['interactions'])} interactions, "
                   f"{len(data['users'])} users")
        
        return data
    
    # =========================
    # Generic File Loader
    # =========================
    
    def _load_file(
        self,
        path: Path,
        required_cols: Optional[List[str]] = None,
        file_type: str = 'data',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generic file loader with caching and validation.
        
        Args:
            path: Path to file
            required_cols: Required columns to validate
            file_type: Type of file (for caching)
            **kwargs: Additional arguments for pd.read_*
            
        Returns:
            Loaded DataFrame
        """
        # Check cache
        cache_key = f"{file_type}_{str(path)}_{str(kwargs)}"
        if self.use_cache and cache_key in self._cache:
            logger.info(f"📦 Loading {file_type} from cache")
            return self._cache[cache_key].copy()
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        logger.info(f"📂 Loading {file_type} from: {path}")
        
        # Determine file format and load
        try:
            df = self._load_by_extension(path, **kwargs)
        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")
            raise
        
        # Basic validation
        if len(df) == 0:
            logger.warning(f"⚠️ {file_type} file is empty: {path}")
        
        # Validate columns
        if self.validate_columns and required_cols:
            self._validate_columns(df, required_cols, file_type)
        
        # Clean data
        df = self._clean_dataframe(df, file_type)
        
        # Cache
        if self.use_cache:
            self._cache[cache_key] = df.copy()
        
        logger.info(f"✅ Loaded {len(df)} rows from {file_type}")
        
        return df
    
    def _load_by_extension(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load file based on its extension."""
        extension = path.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(path, **kwargs)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(path, **kwargs)
        elif extension == '.json':
            return pd.read_json(path, **kwargs)
        elif extension == '.parquet':
            return pd.read_parquet(path, **kwargs)
        else:
            # Try CSV as default
            logger.warning(f"Unknown extension {extension}, trying CSV...")
            return pd.read_csv(path, **kwargs)
    
    # =========================
    # Validation Methods
    # =========================
    
    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], file_type: str):
        """Validate that required columns exist."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"❌ {file_type} file missing required columns: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Check for null values in required columns
        for col in required_cols:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                logger.warning(f"⚠️ {file_type}: {col} has {null_count} null values")
    
    def _clean_dataframe(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Clean and prepare DataFrame."""
        df = df.copy()
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Convert IDs to int if possible
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        for col in id_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            except:
                pass
        
        # Convert rating to float
        if 'rating' in df.columns:
            try:
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
                # Clip ratings to 1-5 range if they exist
                if df['rating'].min() >= 1 and df['rating'].max() <= 5:
                    df['rating'] = df['rating'].clip(1, 5)
            except:
                pass
        
        return df
    
    # =========================
    # Utility Methods
    # =========================
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data."""
        info = {}
        
        for key, df in self._cache.items():
            if isinstance(df, pd.DataFrame):
                info[key] = {
                    'rows': len(df),
                    'columns': df.columns.tolist(),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                    'null_count': df.isnull().sum().sum()
                }
        
        return info
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        logger.info("🗑️ Cache cleared")
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ):
        """
        Save a DataFrame to file.
        
        Args:
            df: DataFrame to save
            path: Output path
            format: Output format (csv, excel, json, parquet)
            **kwargs: Additional arguments for save function
        """
        path = Path(path)
        
        if format is None:
            format = path.suffix.lower().replace('.', '')
        
        if format == 'csv':
            df.to_csv(path, index=False, **kwargs)
        elif format in ['xlsx', 'xls']:
            df.to_excel(path, index=False, **kwargs)
        elif format == 'json':
            df.to_json(path, orient='records', **kwargs)
        elif format == 'parquet':
            df.to_parquet(path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Saved DataFrame to {path}")
    
    def create_sample_data(self, n_content: int = 100, n_interactions: int = 1000):
        """
        Create sample data for testing.
        
        Args:
            n_content: Number of content items
            n_interactions: Number of interactions
        """
        logger.info(f"🔧 Creating sample data with {n_content} items and {n_interactions} interactions")
        
        # Sample content
        categories = ['Programming', 'Data Science', 'Design', 'Business', 'Marketing']
        levels = ['Beginner', 'Intermediate', 'Advanced']
        
        content_data = {
            'content_id': range(1, n_content + 1),
            'title': [f'Course {i}' for i in range(1, n_content + 1)],
            'category': np.random.choice(categories, n_content),
            'level': np.random.choice(levels, n_content),
            'description': [f'Description for course {i}' for i in range(1, n_content + 1)]
        }
        content_df = pd.DataFrame(content_data)
        
        # Sample interactions
        interactions_data = {
            'user_id': np.random.randint(1, 100, n_interactions),
            'content_id': np.random.randint(1, n_content + 1, n_interactions),
            'rating': np.random.randint(1, 6, n_interactions),
            'timestamp': pd.date_range(
                start='2024-01-01',
                periods=n_interactions,
                freq='D'
            )[:n_interactions]
        }
        interactions_df = pd.DataFrame(interactions_data)
        
        # Sample users
        users_data = {
            'user_id': range(1, 101),
            'name': [f'User {i}' for i in range(1, 101)],
            'age': np.random.randint(18, 65, 100),
            'country': np.random.choice(['USA', 'UK', 'Egypt', 'UAE', 'Saudi'], 100)
        }
        users_df = pd.DataFrame(users_data)
        
        # Save to files
        self.save_dataframe(content_df, self.data_dir / 'content.csv')
        self.save_dataframe(interactions_df, self.data_dir / 'interactions.csv')
        self.save_dataframe(users_df, self.data_dir / 'users.xlsx')
        
        logger.info("✅ Sample data created successfully!")
        
        return {
            'content': content_df,
            'interactions': interactions_df,
            'users': users_df
        }
    
    def get_stats(self, df: pd.DataFrame, name: str = 'DataFrame') -> Dict[str, Any]:
        """
        Get statistics for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            name: Name for display
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'null_count': df.isnull().sum().sum(),
            'duplicate_count': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.to_dict()
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Category statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical_stats'] = {
                col: df[col].value_counts().head(5).to_dict()
                for col in categorical_cols[:3]  # Limit to 3 columns
            }
        
        return stats


# =========================
# Helper Functions (Backward Compatible)
# =========================

def load_content(
    path: Union[str, Path] = "data/content.csv",
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load content metadata (backward compatible).
    
    Args:
        path: Path to content file
        data_dir: Data directory (overrides path)
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with content metadata
    """
    if data_dir is not None:
        path = Path(data_dir) / Path(path).name
    
    loader = DataLoader(data_dir=Path(path).parent if data_dir is None else data_dir)
    return loader.load_content(path, **kwargs)


def load_interactions(
    path: Union[str, Path] = "data/interactions.csv",
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load interactions (backward compatible).
    
    Args:
        path: Path to interactions file
        data_dir: Data directory (overrides path)
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with interactions
    """
    if data_dir is not None:
        path = Path(data_dir) / Path(path).name
    
    loader = DataLoader(data_dir=Path(path).parent if data_dir is None else data_dir)
    return loader.load_interactions(path, **kwargs)


# =========================
# Usage Example
# =========================

if __name__ == "__main__":
    # Example 1: Basic usage (backward compatible)
    print("\n🔹 Basic usage:")
    content = load_content("data/content.csv")
    interactions = load_interactions("data/interactions.csv")
    print(f"Content: {len(content)} items")
    print(f"Interactions: {len(interactions)} interactions")
    
    # Example 2: Advanced usage with DataLoader class
    print("\n🔹 Advanced usage:")
    loader = DataLoader(data_dir="data", use_cache=True)
    
    # Load all data
    data = loader.load_all()
    print(f"Content: {len(data['content'])} items")
    print(f"Interactions: {len(data['interactions'])} interactions")
    print(f"Users: {len(data['users'])} users")
    
    # Get statistics
    stats = loader.get_stats(data['content'], "Content")
    print(f"\nStats: {json.dumps(stats, indent=2, default=str)[:500]}...")
    
    # Example 3: Create sample data
    print("\n🔹 Creating sample data:")
    loader.create_sample_data(n_content=50, n_interactions=500)
    
    # Example 4: Get data info
    print("\n🔹 Data info:")
    info = loader.get_data_info()
    print(json.dumps(info, indent=2, default=str))