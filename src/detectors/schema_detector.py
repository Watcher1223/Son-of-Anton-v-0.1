"""
Schema Detector

Detects and classifies schema definition styles in repositories.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import yaml
import json

logger = logging.getLogger(__name__)


class SchemaDetector:
    """Detects schema types and locations in repositories"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize schema detector
        
        Args:
            config: Configuration dictionary with detection patterns
        """
        self.config = config or {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load detection patterns from config"""
        schema_patterns = self.config.get('schema_patterns', {})
        
        self.openapi_patterns = schema_patterns.get('openapi', {
            'files': ['openapi.yaml', 'openapi.json', 'swagger.yaml', 'swagger.json'],
            'directories': ['docs/openapi', 'api-docs']
        })
        
        self.validator_patterns = schema_patterns.get('validator', {
            'files': ['*.validator.js', '*.validator.ts', '*.schema.js', '*.schema.ts'],
            'directories': ['validators', 'validation', 'schemas', 'dto'],
            'libraries': ['joi', 'zod', 'class-validator', 'yup', 'ajv']
        })
        
        self.typedef_patterns = schema_patterns.get('typedef', {
            'files': ['*.dto.ts', '*.types.ts', '*.interface.ts'],
            'directories': ['types', 'interfaces', 'models', 'dto']
        })
    
    def detect_openapi_files(self, repo_path: Path) -> List[Path]:
        """
        Detect OpenAPI/Swagger specification files
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of paths to OpenAPI files
        """
        openapi_files = []
        
        # Check for specific filenames
        for filename in self.openapi_patterns['files']:
            for file_path in repo_path.rglob(filename):
                if 'node_modules' not in str(file_path):
                    openapi_files.append(file_path)
        
        # Check in specific directories
        for dirname in self.openapi_patterns.get('directories', []):
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.yaml'):
                    openapi_files.append(file_path)
                for file_path in dir_path.rglob('*.json'):
                    openapi_files.append(file_path)
        
        logger.info(f"Found {len(openapi_files)} OpenAPI files in {repo_path.name}")
        return openapi_files
    
    def detect_validator_files(self, repo_path: Path) -> List[Path]:
        """
        Detect validator schema files
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of paths to validator files
        """
        validator_files = []
        
        # Search for files matching patterns
        patterns = ['*.validator.js', '*.validator.ts', '*.schema.js', '*.schema.ts']
        
        for pattern in patterns:
            for file_path in repo_path.rglob(pattern):
                if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                    validator_files.append(file_path)
        
        # Check in validator directories
        for dirname in self.validator_patterns.get('directories', []):
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.js'):
                    if file_path not in validator_files:
                        validator_files.append(file_path)
                for file_path in dir_path.rglob('*.ts'):
                    if file_path not in validator_files:
                        validator_files.append(file_path)
        
        logger.info(f"Found {len(validator_files)} validator files in {repo_path.name}")
        return validator_files
    
    def detect_typedef_files(self, repo_path: Path) -> List[Path]:
        """
        Detect TypeScript type definition files
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of paths to typedef files
        """
        typedef_files = []
        
        # Search for TypeScript definition files
        patterns = ['*.dto.ts', '*.types.ts', '*.interface.ts', '*.d.ts']
        
        for pattern in patterns:
            for file_path in repo_path.rglob(pattern):
                if 'node_modules' not in str(file_path):
                    typedef_files.append(file_path)
        
        # Check in type directories
        for dirname in self.typedef_patterns.get('directories', []):
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.ts'):
                    if file_path not in typedef_files:
                        typedef_files.append(file_path)
        
        logger.info(f"Found {len(typedef_files)} typedef files in {repo_path.name}")
        return typedef_files
    
    def detect_validator_libraries(self, repo_path: Path) -> List[str]:
        """
        Detect which validator libraries are used
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of detected validator library names
        """
        package_json = repo_path / "package.json"
        
        if not package_json.exists():
            return []
        
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                
                detected = []
                for lib in self.validator_patterns.get('libraries', []):
                    if lib in all_deps:
                        detected.append(lib)
                
                return detected
        except Exception as e:
            logger.error(f"Error reading package.json: {e}")
            return []
    
    def classify_schema_type(self, repo_path: Path) -> Tuple[str, Dict]:
        """
        Classify the primary schema type used in repository
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Tuple of (primary_schema_type, detection_details)
        """
        repo_path = Path(repo_path)
        
        # Detect all schema types
        openapi_files = self.detect_openapi_files(repo_path)
        validator_files = self.detect_validator_files(repo_path)
        typedef_files = self.detect_typedef_files(repo_path)
        validator_libs = self.detect_validator_libraries(repo_path)
        
        details = {
            'openapi_count': len(openapi_files),
            'validator_count': len(validator_files),
            'typedef_count': len(typedef_files),
            'validator_libraries': validator_libs,
            'openapi_files': [str(f.relative_to(repo_path)) for f in openapi_files],
            'validator_files': [str(f.relative_to(repo_path)) for f in validator_files[:10]],  # Limit output
            'typedef_files': [str(f.relative_to(repo_path)) for f in typedef_files[:10]],
        }
        
        # Determine primary schema type using heuristics
        scores = {
            'openapi': 0,
            'validator': 0,
            'typedef': 0
        }
        
        # OpenAPI files are strong indicators
        if openapi_files:
            scores['openapi'] += len(openapi_files) * 10
        
        # Validator libraries and files
        if validator_libs:
            scores['validator'] += len(validator_libs) * 5
        if validator_files:
            scores['validator'] += len(validator_files)
        
        # TypeScript type files
        if typedef_files:
            scores['typedef'] += len(typedef_files) * 0.5
        
        # Determine winner
        primary_type = max(scores, key=scores.get)
        
        # If all scores are 0, classify as 'unknown'
        if scores[primary_type] == 0:
            primary_type = 'unknown'
        
        details['scores'] = scores
        details['primary_type'] = primary_type
        
        logger.info(f"Classified {repo_path.name} as: {primary_type}")
        logger.debug(f"  Scores: {scores}")
        
        return primary_type, details
    
    def find_schema_for_endpoint(
        self, 
        repo_path: Path, 
        endpoint_path: str, 
        method: str
    ) -> Optional[Dict]:
        """
        Find schema definition for a specific endpoint
        
        Args:
            repo_path: Path to repository
            endpoint_path: API endpoint path (e.g., '/api/users')
            method: HTTP method
            
        Returns:
            Dictionary with schema information or None
        """
        # This is a simplified heuristic-based approach
        # A full implementation would parse OpenAPI specs and code
        
        schema_type, details = self.classify_schema_type(repo_path)
        
        result = {
            'endpoint': endpoint_path,
            'method': method,
            'schema_type': schema_type,
            'schema_files': []
        }
        
        # Try to find related schema files
        path_segments = endpoint_path.strip('/').split('/')
        
        if schema_type == 'openapi' and details['openapi_files']:
            result['schema_files'] = details['openapi_files']
        
        elif schema_type == 'validator':
            # Look for validator files matching endpoint path
            for vfile in details['validator_files']:
                if any(seg in vfile.lower() for seg in path_segments):
                    result['schema_files'].append(vfile)
        
        elif schema_type == 'typedef':
            # Look for typedef files matching endpoint path
            for tfile in details['typedef_files']:
                if any(seg in tfile.lower() for seg in path_segments):
                    result['schema_files'].append(tfile)
        
        return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    detector = SchemaDetector()
    
    test_path = Path("data/raw/example_repo")
    if test_path.exists():
        schema_type, details = detector.classify_schema_type(test_path)
        print(f"\nSchema Type: {schema_type}")
        print(f"Details: {json.dumps(details, indent=2)}")

