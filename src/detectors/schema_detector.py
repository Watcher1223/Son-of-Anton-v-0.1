"""
Schema Detector

Detects and classifies schema definition styles in repositories.
Enhanced with content-based analysis for improved accuracy.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
import json

logger = logging.getLogger(__name__)


class SchemaDetector:
    """Detects schema types and locations in repositories with content analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize schema detector
        
        Args:
            config: Configuration dictionary with detection patterns
        """
        self.config = config or {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load enhanced detection patterns"""
        
        # OpenAPI file patterns - expanded
        self.openapi_file_patterns = [
            'openapi.yaml', 'openapi.json', 'openapi.yml',
            'swagger.yaml', 'swagger.json', 'swagger.yml',
            'api.yaml', 'api.json', 'api.yml',
            'api-spec.yaml', 'api-spec.json', 'api-spec.yml',
            'spec.yaml', 'spec.json',
        ]
        
        self.openapi_directories = [
            'docs', 'api-docs', 'openapi', 'swagger', 
            'docs/api', 'docs/openapi', 'specs', 'api/docs'
        ]
        
        # Content patterns to verify OpenAPI files
        self.openapi_content_patterns = [
            r'"openapi"\s*:\s*"[23]\.',
            r'openapi:\s*["\']?[23]\.',
            r'"swagger"\s*:\s*"2\.',
            r'swagger:\s*["\']?2\.',
            r'"paths"\s*:',
            r'paths:',
        ]
        
        # Validator libraries to check in package.json
        self.validator_libraries = [
            'joi', 'zod', 'yup', 'ajv', 'class-validator',
            'express-validator', 'validator', 'io-ts', 'superstruct',
            'fastest-validator', 'vine', '@sinclair/typebox', 'valibot'
        ]
        
        # Validator code patterns to search in source files
        self.validator_code_patterns = [
            # Joi patterns
            r'Joi\.(object|string|number|array|boolean|date|any)\s*\(',
            r'Joi\.validate\s*\(',
            r'import\s+.*\bJoi\b.*from\s+[\'"]joi[\'"]',
            r'require\s*\(\s*[\'"]joi[\'"]\s*\)',
            
            # Zod patterns
            r'z\.(object|string|number|array|boolean|date|any|enum|union)\s*\(',
            r'z\.parse\s*\(',
            r'z\.safeParse\s*\(',
            r'import\s+.*\bz\b.*from\s+[\'"]zod[\'"]',
            r'import\s+\{\s*z\s*\}.*from\s+[\'"]zod[\'"]',
            
            # Yup patterns
            r'yup\.(object|string|number|array|boolean|date|mixed)\s*\(',
            r'import\s+.*\byup\b.*from\s+[\'"]yup[\'"]',
            
            # Class-validator patterns (decorators)
            r'@IsString\s*\(',
            r'@IsNumber\s*\(',
            r'@IsEmail\s*\(',
            r'@IsNotEmpty\s*\(',
            r'@IsOptional\s*\(',
            r'@IsArray\s*\(',
            r'@IsBoolean\s*\(',
            r'@ValidateNested\s*\(',
            r'@IsInt\s*\(',
            r'@Min\s*\(',
            r'@Max\s*\(',
            r'import\s+.*from\s+[\'"]class-validator[\'"]',
            
            # Express-validator patterns
            r'body\s*\(\s*[\'"].*[\'"]\s*\)\s*\.',
            r'param\s*\(\s*[\'"].*[\'"]\s*\)\s*\.',
            r'query\s*\(\s*[\'"].*[\'"]\s*\)\s*\.',
            r'import\s+.*from\s+[\'"]express-validator[\'"]',
            
            # AJV patterns
            r'ajv\.compile\s*\(',
            r'new\s+Ajv\s*\(',
            
            # TypeBox patterns
            r'Type\.(Object|String|Number|Array|Boolean|Literal)\s*\(',
            r'import\s+.*from\s+[\'"]@sinclair/typebox[\'"]',
        ]
        
        # Validator file patterns
        self.validator_file_patterns = [
            '*.validator.js', '*.validator.ts',
            '*.schema.js', '*.schema.ts',
            '*.validation.js', '*.validation.ts',
        ]
        
        self.validator_directories = [
            'validators', 'validation', 'schemas', 'dto',
            'src/validators', 'src/validation', 'src/schemas',
        ]
        
        # TypeScript/TypeDef patterns
        self.typedef_file_patterns = [
            '*.dto.ts', '*.types.ts', '*.interface.ts',
            '*.entity.ts', '*.model.ts', '*.type.ts',
        ]
        
        self.typedef_directories = [
            'types', 'interfaces', 'models', 'dto', 'dtos',
            'entities', 'src/types', 'src/interfaces', 'src/dto',
            'src/entities', 'src/models',
        ]
        
        # TypeScript content patterns
        self.typedef_content_patterns = [
            r'export\s+interface\s+\w+',
            r'export\s+type\s+\w+\s*=',
            r'interface\s+\w+\s*\{',
            r'type\s+\w+\s*=\s*\{',
            r'export\s+class\s+\w+Dto',
            r'export\s+class\s+\w+Entity',
            r'@ApiProperty\s*\(',  # NestJS Swagger decorator
            r'@Column\s*\(',       # TypeORM
            r'@Entity\s*\(',       # TypeORM
        ]
        
        # Legacy patterns for backwards compatibility
        self.openapi_patterns = {
            'files': self.openapi_file_patterns,
            'directories': self.openapi_directories
        }
        self.validator_patterns = {
            'files': self.validator_file_patterns,
            'directories': self.validator_directories,
            'libraries': self.validator_libraries
        }
        self.typedef_patterns = {
            'files': self.typedef_file_patterns,
            'directories': self.typedef_directories
        }
    
    def _scan_file_content(self, file_path: Path, patterns: List[str], max_size: int = 100000) -> int:
        """
        Scan file content for patterns
        
        Args:
            file_path: Path to file
            patterns: List of regex patterns to search
            max_size: Maximum file size to scan (bytes)
            
        Returns:
            Number of pattern matches found
        """
        try:
            if file_path.stat().st_size > max_size:
                return 0
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, content, re.IGNORECASE))
            
            return matches
        except Exception as e:
            logger.debug(f"Error scanning {file_path}: {e}")
            return 0
    
    def _is_valid_openapi_file(self, file_path: Path) -> bool:
        """Check if file is a valid OpenAPI specification"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            for pattern in self.openapi_content_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            return False
        except Exception:
            return False
    
    def detect_openapi_files(self, repo_path: Path) -> List[Path]:
        """
        Detect OpenAPI/Swagger specification files with validation
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of paths to OpenAPI files
        """
        openapi_files = []
        self._openapi_confidence = 0
        
        # Check for specific filenames
        for filename in self.openapi_file_patterns:
            for file_path in repo_path.rglob(filename):
                if 'node_modules' not in str(file_path):
                    if self._is_valid_openapi_file(file_path):
                        openapi_files.append(file_path)
                        self._openapi_confidence += 15
                    else:
                        # File exists but might not be valid OpenAPI
                        openapi_files.append(file_path)
                        self._openapi_confidence += 5
        
        # Check in specific directories
        for dirname in self.openapi_directories:
            dir_path = repo_path / dirname
            if dir_path.exists():
                for ext in ['*.yaml', '*.yml', '*.json']:
                    for file_path in dir_path.rglob(ext):
                        if file_path not in openapi_files and 'node_modules' not in str(file_path):
                            if self._is_valid_openapi_file(file_path):
                                openapi_files.append(file_path)
                                self._openapi_confidence += 10
        
        # Deduplicate
        openapi_files = list(set(openapi_files))
        
        logger.info(f"Found {len(openapi_files)} OpenAPI files in {repo_path.name}")
        return openapi_files
    
    def detect_validator_files(self, repo_path: Path) -> List[Path]:
        """
        Detect validator schema files with content-based analysis
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of paths to validator files
        """
        validator_files = []
        self._validator_confidence = 0
        
        # Check package.json for validator libraries first
        validator_libs = self.detect_validator_libraries(repo_path)
        if validator_libs:
            self._validator_confidence += len(validator_libs) * 10
        
        # Search for files matching patterns
        for pattern in self.validator_file_patterns:
            for file_path in repo_path.rglob(pattern):
                if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                    validator_files.append(file_path)
                    self._validator_confidence += 5
        
        # Check in validator directories
        for dirname in self.validator_directories:
            dir_path = repo_path / dirname
            if dir_path.exists():
                for ext in ['*.js', '*.ts']:
                    for file_path in dir_path.rglob(ext):
                        if file_path not in validator_files and 'node_modules' not in str(file_path):
                            validator_files.append(file_path)
                            self._validator_confidence += 3
        
        # Scan source files for validator patterns (content analysis)
        source_files = []
        for ext in ['*.js', '*.ts']:
            for file_path in repo_path.rglob(ext):
                if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                    source_files.append(file_path)
        
        # Sample up to 50 source files for content scanning
        sample_files = source_files[:50]
        
        for file_path in sample_files:
            matches = self._scan_file_content(file_path, self.validator_code_patterns)
            if matches > 0:
                self._validator_confidence += min(matches * 2, 10)
                if file_path not in validator_files:
                    validator_files.append(file_path)
        
        # Deduplicate
        validator_files = list(set(validator_files))
        
        logger.info(f"Found {len(validator_files)} validator files in {repo_path.name}")
        return validator_files
    
    def detect_typedef_files(self, repo_path: Path) -> List[Path]:
        """
        Detect TypeScript type definition files with content validation
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of paths to typedef files
        """
        typedef_files = []
        self._typedef_confidence = 0
        
        # Check for TypeScript config first
        tsconfig = repo_path / "tsconfig.json"
        if tsconfig.exists():
            self._typedef_confidence += 5
        
        # Search for TypeScript definition files
        for pattern in self.typedef_file_patterns:
            for file_path in repo_path.rglob(pattern):
                if 'node_modules' not in str(file_path):
                    typedef_files.append(file_path)
                    self._typedef_confidence += 5
        
        # Also check for .d.ts files (declaration files)
        for file_path in repo_path.rglob('*.d.ts'):
            if 'node_modules' not in str(file_path) and file_path not in typedef_files:
                typedef_files.append(file_path)
                self._typedef_confidence += 3
        
        # Check in type directories
        for dirname in self.typedef_directories:
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.ts'):
                    if file_path not in typedef_files and 'node_modules' not in str(file_path):
                        typedef_files.append(file_path)
                        self._typedef_confidence += 2
        
        # Scan .ts files for interface/type definitions (content analysis)
        ts_files = []
        for file_path in repo_path.rglob('*.ts'):
            if 'node_modules' not in str(file_path) and file_path not in typedef_files:
                ts_files.append(file_path)
        
        # Sample up to 30 TypeScript files
        sample_files = ts_files[:30]
        
        for file_path in sample_files:
            matches = self._scan_file_content(file_path, self.typedef_content_patterns)
            if matches >= 2:  # At least 2 interface/type definitions
                typedef_files.append(file_path)
                self._typedef_confidence += 3
        
        # Deduplicate
        typedef_files = list(set(typedef_files))
        
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
    
    def _count_validator_code_patterns(self, repo_path: Path) -> int:
        """
        Count actual validator code pattern matches across source files
        
        This is a more thorough check specifically for classification decisions.
        Returns the total count of validator patterns found.
        """
        total_matches = 0
        source_files = []
        
        for ext in ['*.js', '*.ts']:
            for file_path in repo_path.rglob(ext):
                if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                    source_files.append(file_path)
        
        # Sample more files for this classification check
        sample_files = source_files[:100]
        
        for file_path in sample_files:
            matches = self._scan_file_content(file_path, self.validator_code_patterns)
            total_matches += matches
        
        return total_matches
    
    def _get_classification_reason(
        self, 
        primary_type: str, 
        openapi_files: List[Path], 
        validator_libs: List[str],
        validator_code_matches: int
    ) -> str:
        """Generate a human-readable explanation for the classification"""
        if primary_type == 'openapi':
            return f"Valid OpenAPI/Swagger spec files found ({len(openapi_files)} files)"
        elif primary_type == 'validator':
            if validator_libs:
                return f"Validator library in package.json ({', '.join(validator_libs)}) with {validator_code_matches} code patterns"
            else:
                return f"Validator file patterns found with {validator_code_matches} code patterns"
        elif primary_type == 'typedef':
            return "TypeScript types/interfaces found but no runtime validation"
        else:
            return "No schema definition patterns detected"
    
    def classify_schema_type(self, repo_path: Path) -> Tuple[str, Dict]:
        """
        Classify the primary schema type using smart priority-based logic
        
        Priority order:
        1. OpenAPI - if valid OpenAPI/Swagger files exist (strongest signal)
        2. Validator - if validator library in package.json AND code patterns found
        3. TypeDef - if TypeScript types/interfaces exist but no runtime validation
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Tuple of (primary_schema_type, detection_details)
        """
        repo_path = Path(repo_path)
        
        # Initialize confidence scores
        self._openapi_confidence = 0
        self._validator_confidence = 0
        self._typedef_confidence = 0
        self._validator_code_matches = 0  # Track actual code pattern matches
        
        # Detect all schema types (these methods now update confidence scores)
        openapi_files = self.detect_openapi_files(repo_path)
        validator_libs = self.detect_validator_libraries(repo_path)
        validator_files = self.detect_validator_files(repo_path)
        typedef_files = self.detect_typedef_files(repo_path)
        
        # Count actual validator code pattern matches for smarter classification
        validator_code_matches = self._count_validator_code_patterns(repo_path)
        
        details = {
            'openapi_count': len(openapi_files),
            'openapi_confidence': self._openapi_confidence,
            'validator_count': len(validator_files),
            'validator_confidence': self._validator_confidence,
            'validator_code_matches': validator_code_matches,
            'typedef_count': len(typedef_files),
            'typedef_confidence': self._typedef_confidence,
            'validator_libraries': validator_libs,
            'openapi_files': [str(f.relative_to(repo_path)) for f in openapi_files[:10]],
            'validator_files': [str(f.relative_to(repo_path)) for f in validator_files[:10]],
            'typedef_files': [str(f.relative_to(repo_path)) for f in typedef_files[:10]],
        }
        
        # SMART CLASSIFICATION LOGIC (priority-based)
        primary_type = 'unknown'
        
        # Priority 1: OpenAPI wins if valid spec files exist
        # OpenAPI is a strong, explicit signal - they deliberately wrote an API spec
        if openapi_files and self._openapi_confidence >= 10:
            primary_type = 'openapi'
        
        # Priority 2: Validator wins if library is present AND has meaningful code patterns
        # This catches repos that use Joi, Zod, class-validator even if they have lots of TS files
        elif validator_libs and (validator_code_matches >= 3 or len(validator_files) >= 2):
            primary_type = 'validator'
        
        # Priority 3: Validator also wins if strong validator confidence (file patterns)
        # Even without library in package.json, if they have validation files/patterns
        elif self._validator_confidence >= 30:
            primary_type = 'validator'
        
        # Priority 4: TypeDef - only if no runtime validation detected
        # TypeScript types alone don't help API testing - need actual schemas
        elif self._typedef_confidence >= 10:
            primary_type = 'typedef'
        
        # Fallback: use highest confidence if none of the above
        elif max(self._openapi_confidence, self._validator_confidence, self._typedef_confidence) >= 5:
            scores = {
                'openapi': self._openapi_confidence,
                'validator': self._validator_confidence,
                'typedef': self._typedef_confidence
            }
            primary_type = max(scores, key=scores.get)
        
        # Record scores for debugging
        scores = {
            'openapi': self._openapi_confidence,
            'validator': self._validator_confidence,
            'typedef': self._typedef_confidence
        }
        details['scores'] = scores
        details['primary_type'] = primary_type
        details['classification_reason'] = self._get_classification_reason(
            primary_type, openapi_files, validator_libs, validator_code_matches
        )
        
        logger.info(f"Classified {repo_path.name} as: {primary_type}")
        logger.info(f"  Scores: openapi={scores['openapi']}, validator={scores['validator']}, typedef={scores['typedef']}")
        
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

