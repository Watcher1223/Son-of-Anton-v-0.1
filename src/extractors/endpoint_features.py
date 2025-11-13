"""
Endpoint Feature Extractor

Extracts features from individual API endpoints.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EndpointFeatureExtractor:
    """Extracts features from API endpoints"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize endpoint feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def extract_path_features(self, path: str) -> Dict:
        """
        Extract features from endpoint path
        
        Args:
            path: API endpoint path (e.g., '/api/users/:id')
            
        Returns:
            Dictionary of path-related features
        """
        # Clean path
        path = path.strip()
        if not path.startswith('/'):
            path = '/' + path
        
        # Split path into segments
        segments = [s for s in path.split('/') if s]
        
        # Count different parameter types
        path_params = len([s for s in segments if s.startswith(':') or '{' in s])
        
        features = {
            'path_depth': len(segments),
            'path_param_count': path_params,
            'has_path_params': path_params > 0,
            'path_length': len(path),
            'has_version': bool(re.search(r'v\d+', path.lower())),
            'has_api_prefix': path.lower().startswith('/api'),
        }
        
        return features
    
    def extract_method_features(self, method: str) -> Dict:
        """
        Extract features from HTTP method
        
        Args:
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            Dictionary of method-related features
        """
        method = method.upper()
        
        features = {
            'method': method,
            'is_get': method == 'GET',
            'is_post': method == 'POST',
            'is_put': method == 'PUT',
            'is_delete': method == 'DELETE',
            'is_patch': method == 'PATCH',
            'is_safe_method': method in ['GET', 'HEAD', 'OPTIONS'],
            'is_idempotent': method in ['GET', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'],
            'modifies_data': method in ['POST', 'PUT', 'PATCH', 'DELETE'],
        }
        
        return features
    
    def extract_code_context_features(self, file_path: Path, line_number: int) -> Dict:
        """
        Extract features from code surrounding the endpoint definition
        
        Args:
            file_path: Path to source file
            line_number: Line number of endpoint definition
            
        Returns:
            Dictionary of code context features
        """
        features = {
            'has_middleware': False,
            'has_auth': False,
            'has_validation': False,
            'has_async': False,
            'has_try_catch': False,
            'has_response_types': False,
            'function_length': 0,
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Get context around the endpoint (±20 lines)
            start = max(0, line_number - 20)
            end = min(len(lines), line_number + 20)
            context = ''.join(lines[start:end])
            
            # Detect middleware
            middleware_patterns = [
                r'\buse\s*\(',
                r'\bmiddleware\b',
                r'\.use\(',
                r'@UseGuards',
                r'@UseInterceptors',
            ]
            features['has_middleware'] = any(re.search(p, context) for p in middleware_patterns)
            
            # Detect authentication
            auth_patterns = [
                r'\bauth\b',
                r'\bauthenticate\b',
                r'\bverifyToken\b',
                r'\bjwt\b',
                r'@UseGuards',
                r'@Auth',
                r'requireAuth',
            ]
            features['has_auth'] = any(re.search(p, context, re.IGNORECASE) for p in auth_patterns)
            
            # Detect validation
            validation_patterns = [
                r'\bvalidate\b',
                r'\bschema\b',
                r'\.validate\(',
                r'@Body\(',
                r'@Param\(',
                r'@Query\(',
                r'Joi\.',
                r'zod\.',
                r'class-validator',
            ]
            features['has_validation'] = any(re.search(p, context) for p in validation_patterns)
            
            # Detect async
            features['has_async'] = bool(re.search(r'\basync\b', context))
            
            # Detect error handling
            features['has_try_catch'] = bool(re.search(r'\btry\s*\{', context))
            
            # Detect response type annotations (TypeScript)
            features['has_response_types'] = bool(re.search(r':\s*Promise<\w+>', context))
            
            # Estimate function length
            function_match = re.search(r'\{', context[context.find(str(line_number)):])
            if function_match:
                # Count lines in function (simple heuristic)
                remaining = context[context.find(str(line_number)):]
                features['function_length'] = remaining.count('\n')
        
        except Exception as e:
            logger.warning(f"Error extracting code context from {file_path}:{line_number}: {e}")
        
        return features
    
    def extract_parameter_features(self, file_path: Path, line_number: int, path: str) -> Dict:
        """
        Extract parameter-related features
        
        Args:
            file_path: Path to source file
            line_number: Line number of endpoint
            path: Endpoint path
            
        Returns:
            Dictionary of parameter features
        """
        features = {
            'query_param_count': 0,
            'body_param_count': 0,
            'path_param_count': 0,
            'total_param_count': 0,
            'has_query_params': False,
            'has_body_params': False,
            'has_file_upload': False,
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start = max(0, line_number - 5)
            end = min(len(lines), line_number + 30)
            context = ''.join(lines[start:end])
            
            # Count query parameters
            query_patterns = [
                r'req\.query',
                r'@Query\(',
                r'query\s*\.\s*\w+',
            ]
            features['query_param_count'] = sum(len(re.findall(p, context)) for p in query_patterns)
            features['has_query_params'] = features['query_param_count'] > 0
            
            # Count body parameters
            body_patterns = [
                r'req\.body',
                r'@Body\(',
                r'body\s*\.\s*\w+',
            ]
            features['body_param_count'] = sum(len(re.findall(p, context)) for p in body_patterns)
            features['has_body_params'] = features['body_param_count'] > 0
            
            # Count path parameters
            path_params = len(re.findall(r':[a-zA-Z_]\w*|\{[a-zA-Z_]\w*\}', path))
            features['path_param_count'] = path_params
            
            # Detect file uploads
            upload_patterns = [r'upload', r'multer', r'multipart']
            features['has_file_upload'] = any(re.search(p, context, re.IGNORECASE) for p in upload_patterns)
            
            # Total parameters
            features['total_param_count'] = (
                features['query_param_count'] +
                features['body_param_count'] +
                features['path_param_count']
            )
        
        except Exception as e:
            logger.warning(f"Error extracting parameters from {file_path}:{line_number}: {e}")
        
        return features
    
    def extract_all_features(self, endpoint: Dict, repo_path: Path) -> Dict:
        """
        Extract all features for an endpoint
        
        Args:
            endpoint: Endpoint dictionary from detector
            repo_path: Path to repository
            
        Returns:
            Complete feature dictionary
        """
        features = {
            'endpoint_id': f"{endpoint['method']}_{endpoint['path'].replace('/', '_')}",
            'endpoint_path': endpoint['path'],
            'http_method': endpoint['method'],
            'framework': endpoint.get('framework', 'unknown'),
            'source_file': endpoint['file'],
            'line_number': endpoint.get('line', 0),
        }
        
        # Extract path features
        path_features = self.extract_path_features(endpoint['path'])
        features.update(path_features)
        
        # Extract method features
        method_features = self.extract_method_features(endpoint['method'])
        features.update(method_features)
        
        # Extract code context features
        file_path = Path(endpoint['file'])
        if not file_path.is_absolute():
            file_path = repo_path / file_path
        
        if file_path.exists():
            context_features = self.extract_code_context_features(
                file_path, 
                endpoint.get('line', 0)
            )
            features.update(context_features)
            
            # Extract parameter features
            param_features = self.extract_parameter_features(
                file_path,
                endpoint.get('line', 0),
                endpoint['path']
            )
            features.update(param_features)
        
        return features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    extractor = EndpointFeatureExtractor()
    
    # Example endpoint
    test_endpoint = {
        'method': 'GET',
        'path': '/api/users/:id',
        'framework': 'express',
        'file': 'src/routes/users.js',
        'line': 15
    }
    
    features = extractor.extract_all_features(test_endpoint, Path("data/raw/example_repo"))
    
    print("\nExtracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

