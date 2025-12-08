"""
Data Preprocessor for ML Models

Handles data loading, feature encoding, scaling, and repository-level train/test splitting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses API endpoint data for ML model training.
    
    Handles:
    - Repository-level train/test splitting (prevents data leakage)
    - One-hot encoding for categorical features
    - Min-max scaling for continuous features
    - Label encoding for target variable
    """
    
    # Columns to drop (identifiers, redundant, or non-features)
    DROP_COLUMNS = [
        'endpoint_id',
        'endpoint_path',
        'source_file',
        'repo_path',
        'method',  # Redundant with http_method (already encoded as is_get, is_post, etc.)
        'http_method',  # Already encoded
        'framework',  # Already encoded as is_express, is_nestjs, is_fastify
    ]
    
    # Target column
    TARGET_COLUMN = 'schema_class'
    
    # Group column for repository-level splitting
    GROUP_COLUMN = 'repo_name'
    
    # Categorical columns that need encoding
    CATEGORICAL_COLUMNS = [
        'validator_libraries',
        'orm_type',
    ]
    
    # Boolean columns (already encoded, keep as-is)
    BOOLEAN_COLUMNS = [
        # Path/method features
        'has_path_params', 'has_version', 'has_api_prefix',
        'is_get', 'is_post', 'is_put', 'is_delete', 'is_patch',
        'is_safe_method', 'is_idempotent', 'modifies_data',
        # Schema detection features
        'has_openapi_spec', 'has_validators', 'has_type_defs',
        'has_multiple_schema_types', 'has_related_schema',
        'has_db_models', 'has_comments',
        # Repo-level features
        'is_typescript', 'is_express', 'is_nestjs', 'is_fastify',
        'has_orm', 'has_testing', 'has_linting',
        'has_readme', 'has_docs_folder', 'has_api_docs', 'has_swagger',
        # Original context features
        'has_middleware', 'has_auth', 'has_validation',
        'has_async', 'has_try_catch', 'has_response_types',
        'has_query_params', 'has_body_params', 'has_file_upload',
        # NEW: Validator-specific patterns
        'has_body_decorator', 'has_query_decorator', 'has_param_decorator',
        'has_validation_pipe', 'has_validate_call',
        'has_joi_validate', 'has_zod_parse', 'has_yup_validate',
        'has_express_validator',
        # NEW: OpenAPI-specific patterns
        'has_api_body_decorator', 'has_api_response_decorator',
        'has_api_operation_decorator', 'has_api_property_decorator',
        'has_api_tags_decorator', 'has_swagger_comment',
        # NEW: TypeDef-specific patterns
        'has_dto_reference', 'has_interface_cast', 'has_type_annotation',
        'has_generic_type', 'has_return_type',
        # NEW: File import features
        'file_imports_class_validator', 'file_imports_class_transformer',
        'file_imports_joi', 'file_imports_zod', 'file_imports_yup',
        'file_imports_swagger', 'file_imports_express_validator',
        'file_imports_dto', 'file_imports_types',
    ]
    
    # Continuous columns that need scaling
    CONTINUOUS_COLUMNS = [
        'line_number', 'path_depth', 'path_param_count', 'path_length',
        'openapi_file_count', 'validator_file_count', 'typedef_file_count',
        'related_schema_count',
        'js_file_count', 'ts_file_count', 'jsx_file_count', 'tsx_file_count',
        'total_source_files', 'typescript_ratio',
        'total_files', 'total_lines_of_code', 'avg_lines_per_file',
        'dependency_count', 'dev_dependency_count', 'total_dependency_count',
        'readme_length', 'endpoint_count', 'endpoints_per_file', 'endpoints_per_1000_lines',
        # Original context features
        'function_length', 'query_param_count', 'body_param_count', 'total_param_count',
        # NEW: Aggregate signal counts
        'validator_signal_count', 'openapi_signal_count', 'typedef_signal_count',
    ]
    
    # Class labels
    CLASS_LABELS = ['openapi', 'validator', 'typedef']
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.is_fitted = False
        
        # Track which validator libraries and ORM types we've seen
        self.validator_library_values: List[str] = []
        self.orm_type_values: List[str] = []
    
    def load_data(self, data_path: Path) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            data_path: Path to dataset CSV
            
        Returns:
            DataFrame with loaded data
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from {data_path}")
        logger.info(f"Columns: {len(df.columns)}")
        
        return df
    
    def create_repo_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create repository-level train/test split.
        
        All endpoints from the same repository stay together in either
        train or test set to prevent data leakage.
        
        Args:
            df: Full dataset
            test_size: Proportion of repositories for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.GROUP_COLUMN not in df.columns:
            raise ValueError(f"Group column '{self.GROUP_COLUMN}' not found in dataset")
        
        # Get unique repositories
        repos = df[self.GROUP_COLUMN].unique()
        logger.info(f"Total repositories: {len(repos)}")
        
        # Use GroupShuffleSplit for repository-level splitting
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        
        groups = df[self.GROUP_COLUMN]
        train_idx, test_idx = next(gss.split(df, groups=groups))
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Log split statistics
        train_repos = train_df[self.GROUP_COLUMN].nunique()
        test_repos = test_df[self.GROUP_COLUMN].nunique()
        
        logger.info(f"Train set: {len(train_df)} endpoints from {train_repos} repos")
        logger.info(f"Test set: {len(test_df)} endpoints from {test_repos} repos")
        
        # Log class distribution
        for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
            if self.TARGET_COLUMN in split_df.columns:
                dist = split_df[self.TARGET_COLUMN].value_counts(normalize=True)
                logger.info(f"{split_name} class distribution: {dist.to_dict()}")
        
        return train_df, test_df
    
    def _extract_validator_libraries(self, value: str) -> List[str]:
        """Extract individual validator libraries from comma-separated string."""
        if pd.isna(value) or value == '' or value == 'nan':
            return []
        # Handle quoted strings like '"joi, validator"'
        value = str(value).strip('"').strip("'")
        return [lib.strip().lower() for lib in value.split(',') if lib.strip()]
    
    def _encode_validator_libraries(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode validator_libraries column (multi-value field).
        
        Args:
            df: DataFrame with validator_libraries column
            fit: If True, learn the unique values; if False, use existing
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        if 'validator_libraries' not in df.columns:
            return df
        
        df = df.copy()
        
        # Extract all unique libraries
        if fit:
            all_libs = set()
            for value in df['validator_libraries']:
                all_libs.update(self._extract_validator_libraries(value))
            self.validator_library_values = sorted(all_libs)
            logger.info(f"Validator libraries found: {self.validator_library_values}")
        
        # Create one-hot columns
        for lib in self.validator_library_values:
            col_name = f'has_lib_{lib}'
            df[col_name] = df['validator_libraries'].apply(
                lambda x: 1 if lib in self._extract_validator_libraries(x) else 0
            )
        
        # Drop original column
        df = df.drop(columns=['validator_libraries'])
        
        return df
    
    def _encode_orm_type(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode orm_type column.
        
        Args:
            df: DataFrame with orm_type column
            fit: If True, learn the unique values; if False, use existing
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        if 'orm_type' not in df.columns:
            return df
        
        df = df.copy()
        
        # Clean and extract unique values
        df['orm_type'] = df['orm_type'].fillna('').astype(str).str.strip().str.lower()
        
        if fit:
            self.orm_type_values = [v for v in df['orm_type'].unique() if v and v != 'nan']
            logger.info(f"ORM types found: {self.orm_type_values}")
        
        # Create one-hot columns
        for orm in self.orm_type_values:
            col_name = f'orm_{orm}'
            df[col_name] = (df['orm_type'] == orm).astype(int)
        
        # Drop original column
        df = df.drop(columns=['orm_type'])
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Prepare feature matrix by encoding and cleaning.
        
        Args:
            df: Raw DataFrame
            fit: If True, fit encoders; if False, transform only
            
        Returns:
            Processed DataFrame ready for scaling
        """
        df = df.copy()
        
        # Drop identifier columns
        cols_to_drop = [col for col in self.DROP_COLUMNS if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Also drop repo_name (used for grouping, not as feature)
        if self.GROUP_COLUMN in df.columns:
            df = df.drop(columns=[self.GROUP_COLUMN])
        
        # Encode categorical columns
        df = self._encode_validator_libraries(df, fit=fit)
        df = self._encode_orm_type(df, fit=fit)
        
        # Convert boolean columns to int
        for col in self.BOOLEAN_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if x in [True, 'True', 1, '1'] else 0)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def fit_transform(
        self,
        train_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit preprocessor on training data and transform.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Tuple of (X_train, y_train, feature_names)
        """
        # Extract target
        if self.TARGET_COLUMN not in train_df.columns:
            raise ValueError(f"Target column '{self.TARGET_COLUMN}' not found")
        
        y_train = train_df[self.TARGET_COLUMN].values
        
        # Fit label encoder on actual data (handles unknown classes dynamically)
        self.label_encoder.fit(y_train)
        self.CLASS_LABELS = list(self.label_encoder.classes_)
        logger.info(f"Classes found: {self.CLASS_LABELS}")
        y_train = self.label_encoder.transform(y_train)
        
        # Prepare features
        df_features = self._prepare_features(train_df.drop(columns=[self.TARGET_COLUMN]), fit=True)
        
        # Store feature columns
        self.feature_columns = list(df_features.columns)
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        # Identify continuous columns present in data
        continuous_cols = [col for col in self.CONTINUOUS_COLUMNS if col in df_features.columns]
        
        # Fit and transform continuous features with scaler
        if continuous_cols:
            df_features[continuous_cols] = self.scaler.fit_transform(df_features[continuous_cols])
        
        self.is_fitted = True
        
        X_train = df_features.values.astype(np.float32)
        
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        return X_train, y_train, self.feature_columns
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Tuple of (X, y)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Extract target
        y = None
        if self.TARGET_COLUMN in df.columns:
            y = self.label_encoder.transform(df[self.TARGET_COLUMN].values)
            df = df.drop(columns=[self.TARGET_COLUMN])
        
        # Prepare features (don't refit)
        df_features = self._prepare_features(df, fit=False)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Keep only training columns in same order
        df_features = df_features[self.feature_columns]
        
        # Scale continuous features
        continuous_cols = [col for col in self.CONTINUOUS_COLUMNS if col in df_features.columns]
        if continuous_cols:
            df_features[continuous_cols] = self.scaler.transform(df_features[continuous_cols])
        
        X = df_features.values.astype(np.float32)
        
        logger.info(f"Transformed data shape: X={X.shape}")
        
        return X, y
    
    def get_class_labels(self) -> List[str]:
        """Get ordered class labels."""
        return list(self.label_encoder.classes_)
    
    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original class names."""
        return self.label_encoder.inverse_transform(y)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing."""
        return self.feature_columns.copy()
    
    def prepare_data(
        self,
        data_path: Path,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Loads data, creates repo-level split, and preprocesses.
        
        Args:
            data_path: Path to dataset CSV
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary with X_train, y_train, X_test, y_test, feature_names
        """
        # Load data
        df = self.load_data(data_path)
        
        # Create repository-level split
        train_df, test_df = self.create_repo_split(df, test_size, random_state)
        
        # Fit on training data
        X_train, y_train, feature_names = self.fit_transform(train_df)
        
        # Transform test data
        X_test, y_test = self.transform(test_df)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names,
            'class_labels': self.get_class_labels(),
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = DataPreprocessor()
    
    data_path = Path("data/final/dataset.csv")
    if data_path.exists():
        data = preprocessor.prepare_data(data_path)
        
        print(f"\nData prepared successfully:")
        print(f"  X_train shape: {data['X_train'].shape}")
        print(f"  y_train shape: {data['y_train'].shape}")
        print(f"  X_test shape: {data['X_test'].shape}")
        print(f"  y_test shape: {data['y_test'].shape}")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Classes: {data['class_labels']}")
    else:
        print(f"Dataset not found at {data_path}")

