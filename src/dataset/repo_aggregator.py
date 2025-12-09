"""
Repository-Level Dataset Aggregator

Converts endpoint-level dataset to repository-level for training models
that use only repo-level features (obtainable via GitHub API without cloning).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RepoAggregator:
    """
    Aggregates endpoint-level dataset to repository-level.
    
    Since all endpoints from a repo share the same repo-level features and label,
    this essentially deduplicates by repo_name and keeps only repo-level columns.
    """
    
    # Columns that are truly repo-level (same for all endpoints in a repo)
    REPO_LEVEL_COLUMNS = [
        # Identifiers
        'repo_name',
        
        # Target
        'schema_class',
        
        # Framework detection (from package.json)
        'is_express', 'is_nestjs', 'is_fastify',
        'framework',  # Keep for reference
        
        # ORM detection (from package.json)
        'has_orm', 'orm_type',
        
        # Schema detection signals (from file tree scanning)
        'has_openapi_spec', 'openapi_file_count',
        'has_validators', 'validator_file_count',
        'has_type_defs', 'typedef_file_count',
        'validator_libraries',
        'has_multiple_schema_types',
        
        # Language/file stats (from file tree)
        'js_file_count', 'ts_file_count', 'jsx_file_count', 'tsx_file_count',
        'total_source_files', 'is_typescript', 'typescript_ratio',
        
        # Codebase metrics (from file tree + package.json)
        'total_files', 'total_lines_of_code', 'avg_lines_per_file',
        'dependency_count', 'dev_dependency_count', 'total_dependency_count',
        
        # Quality indicators (from file tree + package.json)
        'has_testing', 'has_linting', 'has_typescript',
        'has_readme', 'readme_length',
        'has_docs_folder', 'has_api_docs', 'has_swagger',
    ]
    
    # Columns to EXCLUDE (require endpoint detection or file content analysis)
    EXCLUDED_COLUMNS = [
        # Endpoint identifiers
        'endpoint_id', 'endpoint_path', 'http_method', 'method',
        'source_file', 'line_number',
        
        # Path/method features (per-endpoint)
        'path_depth', 'path_param_count', 'has_path_params', 'path_length',
        'has_version', 'has_api_prefix',
        'is_get', 'is_post', 'is_put', 'is_delete', 'is_patch',
        'is_safe_method', 'is_idempotent', 'modifies_data',
        
        # Code context features (require file content)
        'has_middleware', 'has_auth', 'has_validation',
        'has_async', 'has_try_catch', 'has_response_types',
        'function_length',
        
        # Parameter features (require file content)
        'query_param_count', 'body_param_count', 'total_param_count',
        'has_query_params', 'has_body_params', 'has_file_upload',
        
        # Import patterns (require file content)
        'file_imports_class_validator', 'file_imports_class_transformer',
        'file_imports_joi', 'file_imports_zod', 'file_imports_yup',
        'file_imports_swagger', 'file_imports_express_validator',
        'file_imports_dto', 'file_imports_types',
        
        # Validator-specific patterns (require file content)
        'has_body_decorator', 'has_query_decorator', 'has_param_decorator',
        'has_validation_pipe', 'has_validate_call',
        'has_joi_validate', 'has_zod_parse', 'has_yup_validate',
        'has_express_validator',
        
        # OpenAPI patterns (require file content)
        'has_api_body_decorator', 'has_api_response_decorator',
        'has_api_operation_decorator', 'has_api_property_decorator',
        'has_api_tags_decorator', 'has_swagger_comment',
        
        # TypeDef patterns (require file content)
        'has_dto_reference', 'has_interface_cast', 'has_type_annotation',
        'has_generic_type', 'has_return_type',
        
        # Signal counts (aggregated from file content patterns)
        'validator_signal_count', 'openapi_signal_count', 'typedef_signal_count',
        
        # Schema linking features
        'related_schema_count', 'has_related_schema', 'has_db_models', 'has_comments',
        
        # Endpoint density (requires endpoint detection)
        'endpoint_count', 'endpoints_per_file', 'endpoints_per_1000_lines',
        
        # Path references
        'repo_path',
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize repo aggregator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def aggregate_to_repo_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert endpoint-level DataFrame to repo-level.
        
        Since all endpoints from a repo share the same repo-level features,
        we simply take the first row per repo and keep only repo-level columns.
        
        Args:
            df: Endpoint-level DataFrame
            
        Returns:
            Repo-level DataFrame (one row per repository)
        """
        logger.info(f"Aggregating {len(df)} endpoints to repo level...")
        
        # Identify which repo-level columns exist in the data
        available_columns = [col for col in self.REPO_LEVEL_COLUMNS if col in df.columns]
        missing_columns = [col for col in self.REPO_LEVEL_COLUMNS if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing expected repo-level columns: {missing_columns}")
        
        logger.info(f"Using {len(available_columns)} repo-level features")
        
        # Group by repo and take first row (all repo-level features are identical)
        repo_df = df.groupby('repo_name').first().reset_index()
        
        # Keep only repo-level columns
        repo_df = repo_df[available_columns]
        
        logger.info(f"Aggregated to {len(repo_df)} repositories")
        
        # Log class distribution
        if 'schema_class' in repo_df.columns:
            logger.info("Repo-level class distribution:")
            for cls, count in repo_df['schema_class'].value_counts().items():
                logger.info(f"  {cls}: {count} ({count/len(repo_df)*100:.1f}%)")
        
        return repo_df
    
    def save_repo_dataset(
        self, 
        repo_df: pd.DataFrame, 
        output_path: Path,
        create_splits: bool = True,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Save repo-level dataset to file(s).
        
        Args:
            repo_df: Repo-level DataFrame
            output_path: Path to save main dataset
            create_splits: Whether to create train/test splits
            test_size: Proportion for test set
            random_state: Random seed
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        repo_df.to_csv(output_path, index=False)
        logger.info(f"Saved repo-level dataset to: {output_path}")
        
        if create_splits:
            from sklearn.model_selection import train_test_split
            
            # Stratified split by schema_class
            stratify = repo_df['schema_class'] if 'schema_class' in repo_df.columns else None
            
            train_df, test_df = train_test_split(
                repo_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # Save splits
            train_path = output_path.parent / 'repo_train.csv'
            test_path = output_path.parent / 'repo_test.csv'
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Train set: {len(train_df)} repos -> {train_path}")
            logger.info(f"Test set: {len(test_df)} repos -> {test_path}")


def create_repo_dataset(
    endpoint_dataset_path: str = "data/final/dataset.csv",
    output_path: str = "data/final/repo_dataset.csv"
) -> pd.DataFrame:
    """
    Convenience function to create repo-level dataset from endpoint-level dataset.
    
    Args:
        endpoint_dataset_path: Path to endpoint-level dataset
        output_path: Path to save repo-level dataset
        
    Returns:
        Repo-level DataFrame
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load endpoint-level dataset
    endpoint_df = pd.read_csv(endpoint_dataset_path)
    logger.info(f"Loaded endpoint dataset: {len(endpoint_df)} rows, {len(endpoint_df.columns)} columns")
    
    # Aggregate to repo level
    aggregator = RepoAggregator()
    repo_df = aggregator.aggregate_to_repo_level(endpoint_df)
    
    # Save
    aggregator.save_repo_dataset(repo_df, Path(output_path))
    
    return repo_df


if __name__ == "__main__":
    create_repo_dataset()
