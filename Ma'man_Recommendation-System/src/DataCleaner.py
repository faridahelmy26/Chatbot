import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Advanced data cleaning and preprocessing for recommendation systems.
    
    Handles missing values, outliers, encoding, scaling, and more.
    """
    
    def __init__(
        self,
        handle_missing: str = 'auto',  # 'auto', 'drop', 'fill'
        handle_outliers: str = 'auto',  # 'auto', 'clip', 'remove', 'none'
        scaling: str = 'none',  # 'none', 'standard', 'minmax'
        encode_categorical: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the DataCleaner.
        
        Args:
            handle_missing: Strategy for missing values
            handle_outliers: Strategy for outliers
            scaling: Scaling method for numeric features
            encode_categorical: Whether to encode categorical columns
            verbose: Print cleaning reports
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.scaling = scaling
        self.encode_categorical = encode_categorical
        self.verbose = verbose
        
        # Storage for transformers
        self.scalers = {}
        self.label_encoders = {}
        self.fitted = False
        
        # Cleaning reports
        self.cleaning_report = {}
        
        logger.info("✅ DataCleaner initialized")
    
    # =========================
    # Main Cleaning Method
    # =========================
    
    def clean(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Clean the entire DataFrame.
        
        Args:
            df: Input DataFrame
            config: Column-specific configuration
                Example: {
                    'user_id': {'type': 'id', 'handle_missing': 'none'},
                    'rating': {'type': 'numeric', 'handle_outliers': 'clip', 'min': 1, 'max': 5},
                    'category': {'type': 'categorical', 'fill_value': 'unknown'}
                }
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("🔄 Starting data cleaning...")
        
        self.cleaning_report = {
            'original_shape': df.shape,
            'original_missing': df.isnull().sum().sum(),
            'operations': []
        }
        
        df = df.copy()
        
        # Step 1: Remove duplicate rows
        df = self._remove_duplicates(df)
        
        # Step 2: Clean each column based on its type
        for col in df.columns:
            if config and col in config:
                col_config = config[col]
            else:
                col_config = self._infer_column_config(df[col])
            
            df[col] = self._clean_column(df[col], col, col_config)
        
        # Step 3: Remove rows with too many missing values
        df = self._remove_incomplete_rows(df)
        
        # Step 4: Encode categorical columns
        if self.encode_categorical:
            df = self._encode_categorical_columns(df)
        
        # Step 5: Scale numeric columns
        if self.scaling != 'none':
            df = self._scale_numeric_columns(df)
        
        # Step 6: Reset index
        df = df.reset_index(drop=True)
        
        # Update report
        self.cleaning_report['final_shape'] = df.shape
        self.cleaning_report['final_missing'] = df.isnull().sum().sum()
        self.cleaning_report['rows_removed'] = self.cleaning_report['original_shape'][0] - df.shape[0]
        
        self.fitted = True
        
        if self.verbose:
            self._print_report()
        
        logger.info(f"✅ Data cleaning complete! {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    # =========================
    # Column Cleaning Methods
    # =========================
    
    def _clean_column(
        self,
        series: pd.Series,
        col_name: str,
        config: Dict[str, Any]
    ) -> pd.Series:
        """Clean a single column."""
        
        # Handle missing values
        if config.get('handle_missing', self.handle_missing) != 'none':
            series = self._handle_missing(series, col_name, config)
        
        # Handle outliers for numeric columns
        if config.get('type') == 'numeric' and config.get('handle_outliers', self.handle_outliers) != 'none':
            series = self._handle_outliers(series, col_name, config)
        
        # Clean text columns
        if config.get('type') == 'text':
            series = self._clean_text(series, config)
        
        # Validate ranges
        if config.get('min') is not None:
            series = series.clip(lower=config['min'])
        if config.get('max') is not None:
            series = series.clip(upper=config['max'])
        
        return series
    
    def _infer_column_config(self, series: pd.Series) -> Dict[str, Any]:
        """Infer configuration for a column based on its data."""
        config = {'type': 'unknown'}
        
        # Check if ID column
        if series.name and ('id' in series.name.lower() or series.name == 'index'):
            config['type'] = 'id'
            config['handle_missing'] = 'none'
            return config
        
        # Check if date/time
        if pd.api.types.is_datetime64_any_dtype(series):
            config['type'] = 'datetime'
            config['handle_missing'] = 'fill'
            return config
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            config['type'] = 'numeric'
            
            # Check if rating (1-5)
            if series.min() >= 1 and series.max() <= 5:
                config['min'] = 1
                config['max'] = 5
                config['handle_outliers'] = 'clip'
            
            return config
        
        # Check if categorical (object with few unique values)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.5 and series.nunique() < 50:
                config['type'] = 'categorical'
                config['fill_value'] = 'unknown'
            else:
                config['type'] = 'text'
                config['fill_value'] = 'unknown'
            
            return config
        
        return config
    
    # =========================
    # Missing Values Handling
    # =========================
    
    def _handle_missing(
        self,
        series: pd.Series,
        col_name: str,
        config: Dict[str, Any]
    ) -> pd.Series:
        """Handle missing values based on column type."""
        
        missing_count = series.isnull().sum()
        if missing_count == 0:
            return series
        
        strategy = config.get('handle_missing', self.handle_missing)
        
        if strategy == 'drop':
            # Will be handled later
            return series
        
        elif strategy == 'fill':
            if config.get('type') == 'numeric':
                # Fill with median or mean
                fill_value = config.get('fill_value')
                if fill_value is None:
                    if series.skew() > 1 or series.skew() < -1:
                        fill_value = series.median()
                    else:
                        fill_value = series.mean()
                series = series.fillna(fill_value)
            
            elif config.get('type') in ['categorical', 'text']:
                # Fill with mode or custom value
                fill_value = config.get('fill_value', 'unknown')
                if fill_value == 'mode':
                    fill_value = series.mode()[0] if not series.mode().empty else 'unknown'
                series = series.fillna(fill_value)
            
            elif config.get('type') == 'datetime':
                # Fill with most frequent date
                fill_value = config.get('fill_value', series.mode()[0] if not series.mode().empty else pd.Timestamp.now())
                series = series.fillna(fill_value)
            
            else:
                # Default: forward fill
                series = series.fillna(method='ffill').fillna(method='bfill')
        
        elif strategy == 'auto':
            # Use smart strategy
            if config.get('type') == 'numeric':
                return self._handle_missing(series, col_name, {**config, 'handle_missing': 'fill'})
            else:
                return self._handle_missing(series, col_name, {**config, 'handle_missing': 'fill'})
        
        self.cleaning_report['operations'].append({
            'column': col_name,
            'operation': f'filled {missing_count} missing values'
        })
        
        return series
    
    # =========================
    # Outliers Handling
    # =========================
    
    def _handle_outliers(
        self,
        series: pd.Series,
        col_name: str,
        config: Dict[str, Any]
    ) -> pd.Series:
        """Handle outliers in numeric columns."""
        
        if not pd.api.types.is_numeric_dtype(series):
            return series
        
        strategy = config.get('handle_outliers', self.handle_outliers)
        
        if strategy == 'none':
            return series
        
        # Detect outliers using IQR or Z-score
        method = config.get('outlier_method', 'iqr')
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        else:  # z-score
            mean = series.mean()
            std = series.std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        
        # Apply bounds from config
        lower_bound = config.get('min', lower_bound)
        upper_bound = config.get('max', upper_bound)
        
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        
        if strategy == 'clip':
            series = series.clip(lower=lower_bound, upper=upper_bound)
            operation = f'clipped {outlier_count} outliers'
        
        elif strategy == 'remove':
            mask = (series >= lower_bound) & (series <= upper_bound)
            series = series[mask]
            operation = f'removed {outlier_count} outliers'
        
        else:
            return series
        
        self.cleaning_report['operations'].append({
            'column': col_name,
            'operation': operation
        })
        
        return series
    
    # =========================
    # Text Cleaning
    # =========================
    
    def _clean_text(
        self,
        series: pd.Series,
        config: Dict[str, Any]
    ) -> pd.Series:
        """Clean text columns."""
        
        series = series.astype(str)
        
        # Strip whitespace
        series = series.str.strip()
        
        # Convert to lowercase
        if config.get('lowercase', True):
            series = series.str.lower()
        
        # Remove special characters
        if config.get('remove_special', True):
            series = series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        
        # Remove multiple spaces
        if config.get('clean_whitespace', True):
            series = series.str.replace(r'\s+', ' ', regex=True)
        
        return series
    
    # =========================
    # Categorical Encoding
    # =========================
    
    def _encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns."""
        
        df = df.copy()
        
        for col in df.columns:
            # Skip if numeric or already encoded
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Skip if too many unique values (likely free text)
            if df[col].nunique() > 50:
                continue
            
            # Encode
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    unique_values = df[col].unique()
                    known_classes = self.label_encoders[col].classes_
                    df[col] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in known_classes
                        else -1
                    )
            
            self.cleaning_report['operations'].append({
                'column': col,
                'operation': f'encoded {df[col].nunique()} categories'
            })
        
        return df
    
    # =========================
    # Scaling
    # =========================
    
    def _scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns."""
        
        df = df.copy()
        
        # Identify numeric columns (excluding IDs and target)
        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Skip if ID column
                if 'id' in col.lower() or col == 'index':
                    continue
                # Skip if binary
                if df[col].nunique() <= 2:
                    continue
                numeric_cols.append(col)
        
        if not numeric_cols:
            return df
        
        # Select scaler
        if self.scaling == 'standard':
            scaler = StandardScaler()
        elif self.scaling == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df
        
        # Fit and transform
        if not self.fitted:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['numeric'] = scaler
        else:
            df[numeric_cols] = self.scalers['numeric'].transform(df[numeric_cols])
        
        self.cleaning_report['operations'].append({
            'columns': numeric_cols,
            'operation': f'scaled using {self.scaling}'
        })
        
        return df
    
    # =========================
    # Utility Methods
    # =========================
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        
        if before != after:
            self.cleaning_report['operations'].append({
                'operation': f'removed {before - after} duplicate rows'
            })
        
        return df
    
    def _remove_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with too many missing values."""
        
        # Remove rows with more than 50% missing values
        threshold = len(df.columns) * 0.5
        before = len(df)
        df = df.dropna(thresh=threshold)
        after = len(df)
        
        if before != after:
            self.cleaning_report['operations'].append({
                'operation': f'removed {before - after} incomplete rows'
            })
        
        return df
    
    # =========================
    # Report Generation
    # =========================
    
    def _print_report(self):
        """Print cleaning report."""
        
        print("\n" + "="*60)
        print("📊 DATA CLEANING REPORT")
        print("="*60)
        
        print(f"\n📐 Original shape: {self.cleaning_report['original_shape']}")
        print(f"📐 Final shape: {self.cleaning_report['final_shape']}")
        print(f"🗑️  Rows removed: {self.cleaning_report['rows_removed']}")
        
        print(f"\n🔍 Missing values:")
        print(f"   Before: {self.cleaning_report['original_missing']}")
        print(f"   After: {self.cleaning_report['final_missing']}")
        
        print(f"\n⚙️  Operations performed:")
        for i, op in enumerate(self.cleaning_report['operations'], 1):
            if 'column' in op:
                print(f"   {i}. {op['column']}: {op['operation']}")
            else:
                print(f"   {i}. {op['operation']}")
        
        print("\n" + "="*60 + "\n")
    
    def get_report(self) -> Dict[str, Any]:
        """Get the cleaning report as a dictionary."""
        return self.cleaning_report
    
    # =========================
    # Specific Cleaning Functions
    # =========================
    
    def clean_ratings(
        self,
        ratings: pd.Series,
        min_rating: int = 1,
        max_rating: int = 5
    ) -> pd.Series:
        """Clean ratings column."""
        
        ratings = ratings.copy()
        
        # Convert to numeric
        ratings = pd.to_numeric(ratings, errors='coerce')
        
        # Clip to range
        ratings = ratings.clip(min_rating, max_rating)
        
        # Round to nearest integer if likely integer ratings
        if ratings.dtype == float and (ratings % 1).sum() < len(ratings) * 0.1:
            ratings = ratings.round()
        
        return ratings
    
    def clean_date(
        self,
        dates: pd.Series,
        format: Optional[str] = None
    ) -> pd.Series:
        """Clean date/time column."""
        
        dates = dates.copy()
        
        if format:
            dates = pd.to_datetime(dates, format=format, errors='coerce')
        else:
            dates = pd.to_datetime(dates, errors='coerce')
        
        return dates
    
    def clean_categorical(
        self,
        categories: pd.Series,
        allowed_values: Optional[List[Any]] = None,
        unknown_value: str = 'unknown'
    ) -> pd.Series:
        """Clean categorical column."""
        
        categories = categories.copy()
        
        # Convert to string
        categories = categories.astype(str).str.strip()
        
        # Replace invalid values
        if allowed_values:
            categories = categories.where(categories.isin(allowed_values), unknown_value)
        
        return categories


# =========================
# Backward Compatible Function
# =========================

def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Clean data with backward compatibility.
    
    Args:
        df: Input DataFrame
        **kwargs: Additional arguments for DataCleaner
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(**kwargs)
    return cleaner.clean(df)


# =========================
# Usage Example
# =========================

if __name__ == "__main__":
    # Create sample data with issues
    np.random.seed(42)
    
    data = {
        'user_id': range(1, 101),
        'content_id': np.random.randint(1, 50, 100),
        'rating': np.random.choice([1, 2, 3, 4, 5, np.nan, 6, 0], 100),
        'category': np.random.choice(['Programming', 'Data Science', 'Design', 'Unknown', np.nan], 100),
        'level': np.random.choice(['Beginner', 'Intermediate', 'Advanced', np.nan], 100),
        'age': np.random.choice([18, 25, 30, 35, 40, 100, 200], 100),
        'country': ['USA', 'UK', 'Egypt', 'UAE', 'Unknown', np.nan] * 17,
        'title': [f'Course {i}' for i in range(1, 101)],
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Add duplicates
    duplicate = df.iloc[0:5].copy()
    df = pd.concat([df, duplicate], ignore_index=True)
    
    print("🔹 Original DataFrame:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(df.head())
    
    # Example 1: Basic cleaning
    print("\n" + "="*60)
    print("Example 1: Basic Cleaning")
    print("="*60)
    
    cleaner = DataCleaner(
        handle_missing='auto',
        handle_outliers='clip',
        scaling='standard',
        encode_categorical=True,
        verbose=True
    )
    
    cleaned_df = cleaner.clean(df)
    
    print("\n🔹 Cleaned DataFrame:")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Missing values: {cleaned_df.isnull().sum().sum()}")
    print(cleaned_df.head())
    
    # Example 2: Custom configuration
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)
    
    config = {
        'user_id': {'type': 'id', 'handle_missing': 'none'},
        'content_id': {'type': 'id', 'handle_missing': 'none'},
        'rating': {
            'type': 'numeric',
            'handle_missing': 'fill',
            'fill_value': 3,
            'handle_outliers': 'clip',
            'min': 1,
            'max': 5
        },
        'category': {
            'type': 'categorical',
            'handle_missing': 'fill',
            'fill_value': 'unknown'
        },
        'level': {
            'type': 'categorical',
            'handle_missing': 'fill',
            'fill_value': 'Beginner'
        },
        'age': {
            'type': 'numeric',
            'handle_outliers': 'clip',
            'min': 18,
            'max': 65
        },
        'country': {
            'type': 'categorical',
            'handle_missing': 'fill',
            'fill_value': 'unknown'
        }
    }
    
    custom_cleaner = DataCleaner(
        handle_missing='none',
        handle_outliers='none',
        scaling='none',
        encode_categorical=False,
        verbose=True
    )
    
    custom_df = custom_cleaner.clean(df, config=config)
    
    # Example 3: Specialized cleaning
    print("\n" + "="*60)
    print("Example 3: Specialized Cleaning")
    print("="*60)
    
    # Clean ratings separately
    ratings = df['rating']
    cleaned_ratings = cleaner.clean_ratings(ratings, min_rating=1, max_rating=5)
    print(f"Original ratings: {ratings.value_counts().sort_index().to_dict()}")
    print(f"Cleaned ratings: {cleaned_ratings.value_counts().sort_index().to_dict()}")
    
    # Example 4: Backward compatible function
    print("\n" + "="*60)
    print("Example 4: Backward Compatible")
    print("="*60)
    
    simple_cleaned = clean_data(df)
    print(f"Shape: {simple_cleaned.shape}")