"""
Schema Linker

Links endpoints to their schema sources and extracts schema-source features.
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SchemaLinker:
    """Links endpoints to schema definitions"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize schema linker
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def find_related_schema_files(
        self,
        endpoint: Dict,
        schema_files: List[Path],
        repo_path: Path
    ) -> List[Path]:
        """
        Find schema files related to an endpoint
        
        Args:
            endpoint: Endpoint dictionary
            schema_files: List of all schema files in repo
            repo_path: Path to repository
            
        Returns:
            List of related schema file paths
        """
        related = []
        
        # Extract path segments from endpoint
        path = endpoint.get('path', '')
        segments = [s.lower() for s in path.strip('/').split('/') if s and not s.startswith(':') and '{' not in s]
        
        # Extract file segments from endpoint source file
        source_file = Path(endpoint.get('file', ''))
        file_segments = [p.lower() for p in source_file.parts if p not in ['src', 'routes', 'controllers', 'api']]
        
        # Combine all search terms
        search_terms = set(segments + file_segments)
        
        # Find schema files matching search terms
        for schema_file in schema_files:
            schema_str = str(schema_file).lower()
            
            # Check if any search term appears in schema file path
            if any(term in schema_str for term in search_terms):
                related.append(schema_file)
        
        return related
    
    def extract_schema_source_features(
        self,
        endpoint: Dict,
        schema_type: str,
        schema_details: Dict,
        repo_path: Path
    ) -> Dict:
        """
        Extract features related to schema sources
        
        Args:
            endpoint: Endpoint dictionary
            schema_type: Primary schema type for repo
            schema_details: Schema detection details
            repo_path: Path to repository
            
        Returns:
            Dictionary of schema source features
        """
        features = {
            'schema_class': schema_type,  # Label for classification
            'has_openapi_spec': schema_details.get('openapi_count', 0) > 0,
            'has_validators': schema_details.get('validator_count', 0) > 0,
            'has_type_defs': schema_details.get('typedef_count', 0) > 0,
            'openapi_file_count': schema_details.get('openapi_count', 0),
            'validator_file_count': schema_details.get('validator_count', 0),
            'typedef_file_count': schema_details.get('typedef_count', 0),
            'validator_libraries': ', '.join(schema_details.get('validator_libraries', [])),
            'has_multiple_schema_types': sum([
                schema_details.get('openapi_count', 0) > 0,
                schema_details.get('validator_count', 0) > 0,
                schema_details.get('typedef_count', 0) > 0,
            ]) > 1,
        }
        
        # Try to find related schema files for this specific endpoint
        all_schema_files = []
        
        if schema_details.get('openapi_files'):
            all_schema_files.extend([
                repo_path / f for f in schema_details['openapi_files']
            ])
        
        if schema_details.get('validator_files'):
            all_schema_files.extend([
                repo_path / f for f in schema_details['validator_files']
            ])
        
        if schema_details.get('typedef_files'):
            all_schema_files.extend([
                repo_path / f for f in schema_details['typedef_files']
            ])
        
        related_files = self.find_related_schema_files(endpoint, all_schema_files, repo_path)
        
        features['related_schema_count'] = len(related_files)
        features['has_related_schema'] = len(related_files) > 0
        
        # Check for database models
        features['has_db_models'] = self._check_for_db_models(repo_path)
        
        # Check for comments/documentation near endpoint
        # The endpoint file path needs to be joined with repo_path
        endpoint_file = endpoint.get('file', '')
        if endpoint_file:
            # Handle both absolute and relative paths
            file_path = Path(endpoint_file)
            if not file_path.is_absolute():
                file_path = repo_path / endpoint_file
            features['has_comments'] = self._check_for_comments(
                file_path,
                endpoint.get('line', 0)
            )
        else:
            features['has_comments'] = False
        
        return features
    
    def _check_for_db_models(self, repo_path: Path) -> bool:
        """Check if repository has database models"""
        model_dirs = ['models', 'entities', 'schemas']
        
        for dir_name in model_dirs:
            dir_path = repo_path / dir_name
            if dir_path.exists() and any(dir_path.iterdir()):
                return True
        
        return False
    
    def _check_for_comments(self, file_path: Path, line_number: int, context_lines: int = 5) -> bool:
        """
        Check if there are comments near the endpoint definition
        
        Args:
            file_path: Path to source file
            line_number: Line number of endpoint
            context_lines: Number of lines to check before endpoint
            
        Returns:
            True if comments found
        """
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines)
            end = line_number
            
            context = ''.join(lines[start:end])
            
            # Check for different comment styles
            has_line_comment = '//' in context
            has_block_comment = '/*' in context or '*/' in context
            has_jsdoc = '/**' in context
            
            return has_line_comment or has_block_comment or has_jsdoc
        
        except Exception as e:
            logger.warning(f"Error checking comments in {file_path}: {e}")
            return False
    
    def link_endpoint_to_schema(
        self,
        endpoint: Dict,
        schema_type: str,
        schema_details: Dict,
        repo_path: Path
    ) -> Dict:
        """
        Create complete linked record for endpoint with schema information
        
        Args:
            endpoint: Endpoint dictionary
            schema_type: Primary schema type
            schema_details: Schema detection details
            repo_path: Path to repository
            
        Returns:
            Complete linked record
        """
        linked_record = {
            **endpoint,
            **self.extract_schema_source_features(
                endpoint,
                schema_type,
                schema_details,
                repo_path
            )
        }
        
        return linked_record
    
    def link_all_endpoints(
        self,
        endpoints: List[Dict],
        schema_type: str,
        schema_details: Dict,
        repo_path: Path
    ) -> List[Dict]:
        """
        Link all endpoints to schema sources
        
        Args:
            endpoints: List of endpoint dictionaries
            schema_type: Primary schema type
            schema_details: Schema detection details
            repo_path: Path to repository
            
        Returns:
            List of linked records
        """
        linked_records = []
        
        for endpoint in endpoints:
            linked_record = self.link_endpoint_to_schema(
                endpoint,
                schema_type,
                schema_details,
                repo_path
            )
            linked_records.append(linked_record)
        
        logger.info(f"Linked {len(linked_records)} endpoints to schema sources")
        
        return linked_records


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    linker = SchemaLinker()
    
    # Example data
    test_endpoint = {
        'method': 'GET',
        'path': '/api/users/:id',
        'file': 'src/routes/users.js',
        'line': 15
    }
    
    test_schema_details = {
        'openapi_count': 1,
        'validator_count': 5,
        'typedef_count': 10,
        'validator_libraries': ['joi'],
        'openapi_files': ['docs/openapi.yaml'],
        'validator_files': ['validators/users.validator.js'],
        'typedef_files': ['types/user.types.ts'],
    }
    
    linked = linker.link_endpoint_to_schema(
        test_endpoint,
        'validator',
        test_schema_details,
        Path('data/raw/example_repo')
    )
    
    print("\nLinked record:")
    for key, value in linked.items():
        print(f"  {key}: {value}")

