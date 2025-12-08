"""
Endpoint Feature Extractor

Extracts features from individual API endpoints.
Enhanced with validation-specific, OpenAPI, and TypeDef pattern detection.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# PATTERN DEFINITIONS - Endpoint-level feature detection
# =============================================================================

# Validation-specific patterns (distinguish between validation libraries)
VALIDATOR_PATTERNS = {
    'has_body_decorator': r'@Body\s*\(',
    'has_query_decorator': r'@Query\s*\(',
    'has_param_decorator': r'@Param\s*\(',
    'has_validation_pipe': r'@UsePipes\s*\(|ValidationPipe',
    'has_validate_call': r'\.validate\s*\(',
    'has_joi_validate': r'Joi\.validate|schema\.validate|\.validateAsync\s*\(',
    'has_zod_parse': r'z\.\w+\(|\.parse\s*\(|\.safeParse\s*\(',
    'has_yup_validate': r'yup\.|validationSchema',
    'has_express_validator': r'\bbody\s*\(\s*[\'"]|query\s*\(\s*[\'"]|param\s*\(\s*[\'"]|check\s*\(\s*[\'"]',
}

# OpenAPI/Swagger patterns
OPENAPI_PATTERNS = {
    'has_api_body_decorator': r'@ApiBody\s*\(',
    'has_api_response_decorator': r'@ApiResponse\s*\(',
    'has_api_operation_decorator': r'@ApiOperation\s*\(',
    'has_api_property_decorator': r'@ApiProperty\s*\(',
    'has_api_tags_decorator': r'@ApiTags\s*\(',
    'has_swagger_comment': r'@swagger|@openapi|\*\s*@api\s',
}

# TypeDef patterns (TypeScript interfaces/types)
TYPEDEF_PATTERNS = {
    'has_dto_reference': r'\b[A-Z]\w*Dto\b',
    'has_interface_cast': r'as\s+[A-Z]\w*(?:Dto|Interface|Type|Request|Response)\b',
    'has_type_annotation': r':\s*[A-Z]\w*(?:Dto|Interface|Type)\b',
    'has_generic_type': r'<[A-Z]\w*(?:Dto|Interface|Type)>',
    'has_return_type': r'\)\s*:\s*Promise<[A-Z]\w+>|\)\s*:\s*[A-Z]\w+\s*\{',
}

# File-level import patterns
IMPORT_PATTERNS = {
    'file_imports_class_validator': r"from\s+['\"]class-validator['\"]",
    'file_imports_class_transformer': r"from\s+['\"]class-transformer['\"]",
    'file_imports_joi': r"from\s+['\"]@?hapi/joi['\"]|from\s+['\"]joi['\"]|require\s*\(\s*['\"]joi['\"]\s*\)",
    'file_imports_zod': r"from\s+['\"]zod['\"]|require\s*\(\s*['\"]zod['\"]\s*\)",
    'file_imports_yup': r"from\s+['\"]yup['\"]",
    'file_imports_swagger': r"from\s+['\"]@nestjs/swagger['\"]",
    'file_imports_express_validator': r"from\s+['\"]express-validator['\"]",
    'file_imports_dto': r"from\s+['\"][^'\"]*\.dto['\"]|from\s+['\"][^'\"]*dto['\"]",
    'file_imports_types': r"from\s+['\"][^'\"]*\.types?['\"]|from\s+['\"][^'\"]*types?['\"]",
}


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
    
    def extract_file_imports(self, file_path: Path) -> Dict:
        """
        Extract import-based features from file header.
        
        Analyzes the first 50 lines of the file to detect which
        validation/schema libraries are imported.
        
        Args:
            file_path: Path to source file
            
        Returns:
            Dictionary of import-based boolean features
        """
        features = {k: False for k in IMPORT_PATTERNS.keys()}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 50 lines (import section)
                header_lines = []
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    header_lines.append(line)
                header = ''.join(header_lines)
            
            for name, pattern in IMPORT_PATTERNS.items():
                features[name] = bool(re.search(pattern, header))
                
        except Exception as e:
            logger.warning(f"Error reading imports from {file_path}: {e}")
        
        return features
    
    def extract_handler_context(self, file_path: Path, line_number: int) -> str:
        """
        Get handler function context (fixed 50-line window after endpoint).
        
        Args:
            file_path: Path to source file
            line_number: Line number of endpoint declaration
            
        Returns:
            String containing handler context
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Start from endpoint line, read forward 50 lines
            start = max(0, line_number - 1)
            end = min(len(lines), line_number + 50)
            return ''.join(lines[start:end])
            
        except Exception as e:
            logger.warning(f"Error reading handler context from {file_path}:{line_number}: {e}")
            return ""
    
    def extract_validation_features(self, context: str) -> Dict:
        """
        Extract validation-specific features from handler context.
        
        Detects specific validation library patterns, OpenAPI decorators,
        and TypeScript type definition patterns.
        
        Args:
            context: Handler code context string
            
        Returns:
            Dictionary of validation-related boolean features
        """
        features = {}
        
        # Check validator patterns
        for name, pattern in VALIDATOR_PATTERNS.items():
            features[name] = bool(re.search(pattern, context))
        
        # Check OpenAPI patterns
        for name, pattern in OPENAPI_PATTERNS.items():
            features[name] = bool(re.search(pattern, context))
        
        # Check TypeDef patterns
        for name, pattern in TYPEDEF_PATTERNS.items():
            features[name] = bool(re.search(pattern, context))
        
        return features
    
    def compute_signal_counts(self, features: Dict) -> Dict:
        """
        Compute aggregate signal counts per schema category.
        
        These aggregate features help the model by combining multiple
        individual signals into a single strength indicator.
        
        Args:
            features: Dictionary containing individual feature values
            
        Returns:
            Dictionary with signal count features
        """
        return {
            # Count of validator-related signals
            'validator_signal_count': sum([
                features.get('has_body_decorator', False),
                features.get('has_query_decorator', False),
                features.get('has_param_decorator', False),
                features.get('has_validation_pipe', False),
                features.get('has_validate_call', False),
                features.get('has_joi_validate', False),
                features.get('has_zod_parse', False),
                features.get('has_yup_validate', False),
                features.get('has_express_validator', False),
                features.get('file_imports_class_validator', False),
                features.get('file_imports_joi', False),
                features.get('file_imports_zod', False),
                features.get('file_imports_yup', False),
                features.get('file_imports_express_validator', False),
            ]),
            
            # Count of OpenAPI-related signals
            'openapi_signal_count': sum([
                features.get('has_api_body_decorator', False),
                features.get('has_api_response_decorator', False),
                features.get('has_api_operation_decorator', False),
                features.get('has_api_property_decorator', False),
                features.get('has_api_tags_decorator', False),
                features.get('has_swagger_comment', False),
                features.get('file_imports_swagger', False),
            ]),
            
            # Count of TypeDef-related signals
            'typedef_signal_count': sum([
                features.get('has_dto_reference', False),
                features.get('has_interface_cast', False),
                features.get('has_type_annotation', False),
                features.get('has_generic_type', False),
                features.get('has_return_type', False),
                features.get('file_imports_dto', False),
                features.get('file_imports_types', False),
            ]),
        }
    
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
            
            # Detect validation (general)
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
        # Handle paths that might already include repo_path (from detector)
        if not file_path.is_absolute():
            # Check if file exists as-is (path already includes repo_path)
            if not file_path.exists():
                # Try prepending repo_path
                file_path = repo_path / file_path
        
        if file_path.exists():
            # Original context features
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
            
            # NEW: Extract file-level import features
            import_features = self.extract_file_imports(file_path)
            features.update(import_features)
            
            # NEW: Extract handler context and validation-specific features
            handler_context = self.extract_handler_context(
                file_path,
                endpoint.get('line', 0)
            )
            validation_features = self.extract_validation_features(handler_context)
            features.update(validation_features)
            
            # NEW: Compute aggregate signal counts
            signal_counts = self.compute_signal_counts(features)
            features.update(signal_counts)
        
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
    for key, value in sorted(features.items()):
        print(f"  {key}: {value}")
