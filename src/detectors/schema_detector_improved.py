"""
Improved Schema Detector

Enhanced detection of schema definition styles with content-based analysis.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
import json

logger = logging.getLogger(__name__)


class ImprovedSchemaDetector:
    """Enhanced schema type detection with content analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize improved schema detector
        
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
            '*.openapi.yaml', '*.openapi.json', '*.openapi.yml',
            '*.swagger.yaml', '*.swagger.json', '*.swagger.yml',
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
            r'\.validateAsync\s*\(',
            r'import\s+.*\bJoi\b.*from\s+[\'"]joi[\'"]',
            r'require\s*\(\s*[\'"]joi[\'"]\s*\)',
            
            # Zod patterns
            r'z\.(object|string|number|array|boolean|date|any|enum|union)\s*\(',
            r'z\.parse\s*\(',
            r'z\.safeParse\s*\(',
            r'import\s+.*\bz\b.*from\s+[\'"]zod[\'"]',
            r'import\s+\{\s*z\s*\}.*from\s+[\'"]zod[\'"]',
            r'require\s*\(\s*[\'"]zod[\'"]\s*\)',
            
            # Yup patterns
            r'yup\.(object|string|number|array|boolean|date|mixed)\s*\(',
            r'\.validate\s*\(',
            r'import\s+.*\byup\b.*from\s+[\'"]yup[\'"]',
            r'require\s*\(\s*[\'"]yup[\'"]\s*\)',
            
            # Class-validator patterns (decorators)
            r'@IsString\s*\(',
            r'@IsNumber\s*\(',
            r'@IsEmail\s*\(',
            r'@IsNotEmpty\s*\(',
            r'@IsOptional\s*\(',
            r'@IsArray\s*\(',
            r'@IsBoolean\s*\(',
            r'@IsDate\s*\(',
            r'@ValidateNested\s*\(',
            r'@IsInt\s*\(',
            r'@Min\s*\(',
            r'@Max\s*\(',
            r'@Length\s*\(',
            r'@Matches\s*\(',
            r'import\s+.*from\s+[\'"]class-validator[\'"]',
            
            # Express-validator patterns
            r'body\s*\(\s*[\'"]',
            r'param\s*\(\s*[\'"]',
            r'query\s*\(\s*[\'"]',
            r'check\s*\(\s*[\'"]',
            r'\.isEmail\s*\(',
            r'\.isLength\s*\(',
            r'\.notEmpty\s*\(',
            r'import\s+.*from\s+[\'"]express-validator[\'"]',
            
            # AJV patterns
            r'ajv\.compile\s*\(',
            r'ajv\.validate\s*\(',
            r'new\s+Ajv\s*\(',
            r'import\s+.*\bAjv\b.*from\s+[\'"]ajv[\'"]',
            
            # TypeBox patterns
            r'Type\.(Object|String|Number|Array|Boolean|Literal)\s*\(',
            r'import\s+.*from\s+[\'"]@sinclair/typebox[\'"]',
        ]
        
        # Validator file patterns
        self.validator_file_patterns = [
            '*.validator.js', '*.validator.ts',
            '*.schema.js', '*.schema.ts',
            '*.validation.js', '*.validation.ts',
            '*.validate.js', '*.validate.ts',
        ]
        
        self.validator_directories = [
            'validators', 'validation', 'schemas', 'dto',
            'src/validators', 'src/validation', 'src/schemas',
            'lib/validators', 'lib/validation',
        ]
        
        # TypeScript/TypeDef patterns
        self.typedef_file_patterns = [
            '*.dto.ts', '*.types.ts', '*.interface.ts',
            '*.entity.ts', '*.model.ts', '*.type.ts',
        ]
        
        self.typedef_directories = [
            'types', 'interfaces', 'models', 'dto', 'dtos',
            'entities', 'src/types', 'src/interfaces', 'src/dto',
            'src/entities', 'src/models', 'lib/types',
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
    
    def detect_openapi_files(self, repo_path: Path) -> Tuple[List[Path], int]:
        """
        Detect OpenAPI/Swagger specification files with validation
        
        Returns:
            Tuple of (list of OpenAPI files, confidence score)
        """
        openapi_files = []
        confidence = 0
        
        # Check for specific filenames
        for pattern in self.openapi_file_patterns:
            if '*' in pattern:
                for file_path in repo_path.rglob(pattern):
                    if 'node_modules' not in str(file_path) and self._is_valid_openapi_file(file_path):
                        openapi_files.append(file_path)
                        confidence += 10
            else:
                for file_path in repo_path.rglob(pattern):
                    if 'node_modules' not in str(file_path) and self._is_valid_openapi_file(file_path):
                        openapi_files.append(file_path)
                        confidence += 15  # Exact match = higher confidence
        
        # Check in specific directories
        for dirname in self.openapi_directories:
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.yaml'):
                    if file_path not in openapi_files and self._is_valid_openapi_file(file_path):
                        openapi_files.append(file_path)
                        confidence += 8
                for file_path in dir_path.rglob('*.yml'):
                    if file_path not in openapi_files and self._is_valid_openapi_file(file_path):
                        openapi_files.append(file_path)
                        confidence += 8
                for file_path in dir_path.rglob('*.json'):
                    if file_path not in openapi_files and self._is_valid_openapi_file(file_path):
                        openapi_files.append(file_path)
                        confidence += 8
        
        logger.info(f"Found {len(openapi_files)} validated OpenAPI files (confidence: {confidence})")
        return list(set(openapi_files)), confidence
    
    def detect_validator_usage(self, repo_path: Path) -> Tuple[List[str], List[Path], int]:
        """
        Detect validator library usage through package.json and code analysis
        
        Returns:
            Tuple of (libraries found, validator files, confidence score)
        """
        libraries_found = []
        validator_files = []
        confidence = 0
        
        # Check package.json
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                    
                    for lib in self.validator_libraries:
                        if lib in all_deps:
                            libraries_found.append(lib)
                            confidence += 10  # Package.json is strong indicator
            except Exception as e:
                logger.debug(f"Error reading package.json: {e}")
        
        # Find validator files by name pattern
        for pattern in self.validator_file_patterns:
            for file_path in repo_path.rglob(pattern):
                if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                    validator_files.append(file_path)
                    confidence += 3
        
        # Check in validator directories
        for dirname in self.validator_directories:
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.js'):
                    if file_path not in validator_files and 'node_modules' not in str(file_path):
                        validator_files.append(file_path)
                        confidence += 2
                for file_path in dir_path.rglob('*.ts'):
                    if file_path not in validator_files and 'node_modules' not in str(file_path):
                        validator_files.append(file_path)
                        confidence += 2
        
        # Scan source files for validator patterns (sampling)
        source_files = []
        for ext in ['*.js', '*.ts']:
            for file_path in repo_path.rglob(ext):
                if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                    source_files.append(file_path)
        
        # Sample up to 50 source files for content scanning
        sample_files = source_files[:50]
        code_matches = 0
        
        for file_path in sample_files:
            matches = self._scan_file_content(file_path, self.validator_code_patterns)
            if matches > 0:
                code_matches += matches
                if file_path not in validator_files:
                    validator_files.append(file_path)
        
        # Code pattern matches boost confidence significantly
        confidence += min(code_matches * 2, 30)  # Cap at 30
        
        logger.info(f"Found {len(libraries_found)} validator libs, {len(validator_files)} files (confidence: {confidence})")
        return libraries_found, list(set(validator_files)), confidence
    
    def detect_typedef_files(self, repo_path: Path) -> Tuple[List[Path], int]:
        """
        Detect TypeScript type definition files with content validation
        
        Returns:
            Tuple of (typedef files, confidence score)
        """
        typedef_files = []
        confidence = 0
        
        # Check for TypeScript config first
        tsconfig = repo_path / "tsconfig.json"
        if tsconfig.exists():
            confidence += 5
        
        # Find typedef files by name pattern
        for pattern in self.typedef_file_patterns:
            for file_path in repo_path.rglob(pattern):
                if 'node_modules' not in str(file_path):
                    typedef_files.append(file_path)
                    confidence += 3
        
        # Check in typedef directories
        for dirname in self.typedef_directories:
            dir_path = repo_path / dirname
            if dir_path.exists():
                for file_path in dir_path.rglob('*.ts'):
                    if file_path not in typedef_files and 'node_modules' not in str(file_path):
                        typedef_files.append(file_path)
                        confidence += 1
        
        # Scan .ts files for interface/type definitions
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
                confidence += 2
        
        logger.info(f"Found {len(typedef_files)} typedef files (confidence: {confidence})")
        return list(set(typedef_files)), confidence
    
    def classify_schema_type(self, repo_path: Path) -> Tuple[str, Dict]:
        """
        Classify the primary schema type with confidence-based scoring
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Tuple of (primary_schema_type, detection_details)
        """
        repo_path = Path(repo_path)
        
        # Detect all schema types with confidence scores
        openapi_files, openapi_confidence = self.detect_openapi_files(repo_path)
        validator_libs, validator_files, validator_confidence = self.detect_validator_usage(repo_path)
        typedef_files, typedef_confidence = self.detect_typedef_files(repo_path)
        
        details = {
            'openapi_count': len(openapi_files),
            'openapi_confidence': openapi_confidence,
            'validator_count': len(validator_files),
            'validator_confidence': validator_confidence,
            'validator_libraries': validator_libs,
            'typedef_count': len(typedef_files),
            'typedef_confidence': typedef_confidence,
            'openapi_files': [str(f.relative_to(repo_path)) for f in openapi_files[:10]],
            'validator_files': [str(f.relative_to(repo_path)) for f in validator_files[:10]],
            'typedef_files': [str(f.relative_to(repo_path)) for f in typedef_files[:10]],
        }
        
        # Use confidence scores for classification
        scores = {
            'openapi': openapi_confidence,
            'validator': validator_confidence,
            'typedef': typedef_confidence
        }
        
        details['scores'] = scores
        
        # Determine winner - require minimum confidence threshold
        max_score = max(scores.values())
        
        if max_score < 5:  # Minimum threshold
            primary_type = 'unknown'
        else:
            primary_type = max(scores, key=scores.get)
        
        details['primary_type'] = primary_type
        
        logger.info(f"Classified {repo_path.name} as: {primary_type}")
        logger.info(f"  Scores: openapi={scores['openapi']}, validator={scores['validator']}, typedef={scores['typedef']}")
        
        return primary_type, details


# Backwards compatible alias
SchemaDetector = ImprovedSchemaDetector


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    detector = ImprovedSchemaDetector()
    
    test_path = Path("data/raw/example_repo")
    if test_path.exists():
        schema_type, details = detector.classify_schema_type(test_path)
        print(f"\nSchema Type: {schema_type}")
        print(f"Details: {json.dumps(details, indent=2)}")

