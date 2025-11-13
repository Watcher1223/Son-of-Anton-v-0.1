"""
Dataset Assembler

Assembles final dataset from extracted features and saves to CSV.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DatasetAssembler:
    """Assembles and exports the final dataset"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dataset assembler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.target_endpoints = self.config.get('target_endpoints', 600)
    
    def combine_features(
        self,
        endpoint_features: Dict,
        repo_features: Dict,
        schema_features: Dict
    ) -> Dict:
        """
        Combine endpoint, repository, and schema features
        
        Args:
            endpoint_features: Endpoint-level features
            repo_features: Repository-level features
            schema_features: Schema-source features
            
        Returns:
            Combined feature dictionary
        """
        combined = {
            **endpoint_features,
            **repo_features,
            **schema_features,
        }
        
        return combined
    
    def validate_record(self, record: Dict) -> bool:
        """
        Validate a single record for completeness
        
        Args:
            record: Feature record dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'endpoint_id',
            'endpoint_path',
            'http_method',
            'schema_class',
            'framework',
        ]
        
        for field in required_fields:
            if field not in record or record[field] is None:
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    def deduplicate_endpoints(self, records: List[Dict]) -> List[Dict]:
        """
        Remove duplicate endpoints
        
        Args:
            records: List of feature records
            
        Returns:
            Deduplicated list
        """
        seen = set()
        deduplicated = []
        
        for record in records:
            key = (
                record.get('repo_name'),
                record.get('http_method'),
                record.get('endpoint_path')
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(record)
            else:
                logger.debug(f"Duplicate endpoint: {key}")
        
        logger.info(f"Deduplicated: {len(records)} -> {len(deduplicated)} records")
        
        return deduplicated
    
    def check_class_balance(self, records: List[Dict]) -> Dict:
        """
        Check balance of schema classes
        
        Args:
            records: List of feature records
            
        Returns:
            Dictionary with class counts
        """
        class_counts = {}
        
        for record in records:
            schema_class = record.get('schema_class', 'unknown')
            class_counts[schema_class] = class_counts.get(schema_class, 0) + 1
        
        logger.info("Class distribution:")
        for schema_class, count in class_counts.items():
            percentage = (count / len(records) * 100) if len(records) > 0 else 0
            logger.info(f"  {schema_class}: {count} ({percentage:.1f}%)")
        
        return class_counts
    
    def assemble_dataset(
        self,
        records: List[Dict],
        output_path: Path,
        output_format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Assemble final dataset and save to file
        
        Args:
            records: List of feature records
            output_path: Path to save dataset
            output_format: Output format ('csv' or 'json')
            
        Returns:
            Pandas DataFrame of the dataset
        """
        logger.info(f"Assembling dataset with {len(records)} records")
        
        # Validate records
        valid_records = [r for r in records if self.validate_record(r)]
        logger.info(f"Valid records: {len(valid_records)}/{len(records)}")
        
        # Deduplicate
        records = self.deduplicate_endpoints(valid_records)
        
        # Check class balance
        self.check_class_balance(records)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Reorder columns: put important fields first
        priority_columns = [
            'endpoint_id',
            'repo_name',
            'schema_class',
            'http_method',
            'endpoint_path',
            'framework',
        ]
        
        # Get remaining columns
        remaining_columns = [col for col in df.columns if col not in priority_columns]
        
        # Reorder
        column_order = [col for col in priority_columns if col in df.columns] + remaining_columns
        df = df[column_order]
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'csv':
            df.to_csv(output_path, index=False)
            logger.info(f"Saved dataset to CSV: {output_path}")
        elif output_format == 'json':
            df.to_json(output_path, orient='records', indent=2)
            logger.info(f"Saved dataset to JSON: {output_path}")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Print summary statistics
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the dataset"""
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"\nFeatures: {', '.join(df.columns[:10])}...")
        
        # Schema class distribution
        if 'schema_class' in df.columns:
            logger.info("\nSchema Class Distribution:")
            for schema_class, count in df['schema_class'].value_counts().items():
                logger.info(f"  {schema_class}: {count} ({count/len(df)*100:.1f}%)")
        
        # Framework distribution
        if 'framework' in df.columns:
            logger.info("\nFramework Distribution:")
            for framework, count in df['framework'].value_counts().items():
                logger.info(f"  {framework}: {count} ({count/len(df)*100:.1f}%)")
        
        # HTTP method distribution
        if 'http_method' in df.columns:
            logger.info("\nHTTP Method Distribution:")
            for method, count in df['http_method'].value_counts().items():
                logger.info(f"  {method}: {count} ({count/len(df)*100:.1f}%)")
        
        logger.info("="*60 + "\n")
    
    def load_intermediate_data(self, intermediate_dir: Path) -> List[Dict]:
        """
        Load intermediate JSON data from extraction phase
        
        Args:
            intermediate_dir: Directory containing intermediate JSON files
            
        Returns:
            List of all records
        """
        intermediate_dir = Path(intermediate_dir)
        all_records = []
        
        for json_file in intermediate_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        all_records.extend(data)
                    elif isinstance(data, dict):
                        all_records.append(data)
                
                logger.info(f"Loaded {json_file.name}")
            
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(all_records)} records from {intermediate_dir}")
        
        return all_records
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Create train/test split and save separately
        
        Args:
            df: Complete dataset DataFrame
            output_dir: Directory to save splits
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
        """
        from sklearn.model_selection import train_test_split
        
        # Stratify by schema_class if available
        stratify_column = df['schema_class'] if 'schema_class' in df.columns else None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_column
        )
        
        # Save splits
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train.csv'
        test_path = output_dir / 'test.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Train set: {len(train_df)} records -> {train_path}")
        logger.info(f"Test set: {len(test_df)} records -> {test_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    assembler = DatasetAssembler()
    
    # Example: load intermediate data and assemble
    intermediate_dir = Path("data/intermediate")
    if intermediate_dir.exists():
        records = assembler.load_intermediate_data(intermediate_dir)
        
        if records:
            df = assembler.assemble_dataset(
                records,
                Path("data/final/dataset.csv"),
                output_format='csv'
            )
            
            # Create train/test split
            assembler.create_train_test_split(df, Path("data/final"))

