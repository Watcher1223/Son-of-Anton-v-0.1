"""
Repository Feature Extractor

Extracts repository-level features.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RepositoryFeatureExtractor:
    """Extracts repository-level features"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize repository feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def extract_language_features(self, repo_path: Path) -> Dict:
        """
        Extract language-related features
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary of language features
        """
        js_files = len(list(repo_path.rglob('*.js')))
        ts_files = len(list(repo_path.rglob('*.ts')))
        jsx_files = len(list(repo_path.rglob('*.jsx')))
        tsx_files = len(list(repo_path.rglob('*.tsx')))
        
        # Exclude node_modules
        js_files = len([f for f in repo_path.rglob('*.js') if 'node_modules' not in str(f)])
        ts_files = len([f for f in repo_path.rglob('*.ts') if 'node_modules' not in str(f)])
        
        total_files = js_files + ts_files + jsx_files + tsx_files
        
        features = {
            'js_file_count': js_files,
            'ts_file_count': ts_files,
            'jsx_file_count': jsx_files,
            'tsx_file_count': tsx_files,
            'total_source_files': total_files,
            'is_typescript': ts_files > js_files,
            'typescript_ratio': ts_files / total_files if total_files > 0 else 0,
        }
        
        return features
    
    def extract_framework_features(self, repo_path: Path) -> Dict:
        """
        Extract framework-related features
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary of framework features
        """
        features = {
            'framework': 'unknown',
            'is_express': False,
            'is_nestjs': False,
            'is_fastify': False,
            'has_orm': False,
            'orm_type': None,
        }
        
        package_json = repo_path / 'package.json'
        
        if not package_json.exists():
            return features
        
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
            
            # Detect framework
            if '@nestjs/core' in all_deps or '@nestjs/common' in all_deps:
                features['framework'] = 'nestjs'
                features['is_nestjs'] = True
            elif 'fastify' in all_deps:
                features['framework'] = 'fastify'
                features['is_fastify'] = True
            elif 'express' in all_deps:
                features['framework'] = 'express'
                features['is_express'] = True
            
            # Detect ORM
            if 'typeorm' in all_deps:
                features['has_orm'] = True
                features['orm_type'] = 'typeorm'
            elif 'sequelize' in all_deps:
                features['has_orm'] = True
                features['orm_type'] = 'sequelize'
            elif 'mongoose' in all_deps:
                features['has_orm'] = True
                features['orm_type'] = 'mongoose'
            elif 'prisma' in all_deps or '@prisma/client' in all_deps:
                features['has_orm'] = True
                features['orm_type'] = 'prisma'
        
        except Exception as e:
            logger.error(f"Error extracting framework features: {e}")
        
        return features
    
    def extract_codebase_size_features(self, repo_path: Path) -> Dict:
        """
        Extract codebase size features
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary of size features
        """
        total_lines = 0
        total_files = 0
        
        extensions = ['.js', '.ts', '.jsx', '.tsx']
        
        for ext in extensions:
            for file_path in repo_path.rglob(f'*{ext}'):
                if 'node_modules' not in str(file_path) and '.git' not in str(file_path):
                    total_files += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for _ in f)
                    except Exception:
                        pass
        
        features = {
            'total_files': total_files,
            'total_lines_of_code': total_lines,
            'avg_lines_per_file': total_lines / total_files if total_files > 0 else 0,
        }
        
        return features
    
    def extract_dependency_features(self, repo_path: Path) -> Dict:
        """
        Extract dependency-related features
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary of dependency features
        """
        features = {
            'dependency_count': 0,
            'dev_dependency_count': 0,
            'total_dependency_count': 0,
            'has_testing': False,
            'has_linting': False,
            'has_typescript': False,
        }
        
        package_json = repo_path / 'package.json'
        
        if not package_json.exists():
            return features
        
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            deps = data.get('dependencies', {})
            dev_deps = data.get('devDependencies', {})
            
            features['dependency_count'] = len(deps)
            features['dev_dependency_count'] = len(dev_deps)
            features['total_dependency_count'] = len(deps) + len(dev_deps)
            
            # Check for testing frameworks
            testing_libs = ['jest', 'mocha', 'chai', 'jasmine', 'ava', 'tape']
            features['has_testing'] = any(lib in {**deps, **dev_deps} for lib in testing_libs)
            
            # Check for linting
            linting_libs = ['eslint', 'tslint', 'prettier']
            features['has_linting'] = any(lib in {**deps, **dev_deps} for lib in linting_libs)
            
            # Check for TypeScript
            features['has_typescript'] = 'typescript' in {**deps, **dev_deps}
        
        except Exception as e:
            logger.error(f"Error extracting dependency features: {e}")
        
        return features
    
    def extract_documentation_features(self, repo_path: Path) -> Dict:
        """
        Extract documentation-related features
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary of documentation features
        """
        features = {
            'has_readme': False,
            'readme_length': 0,
            'has_docs_folder': False,
            'has_api_docs': False,
            'has_swagger': False,
        }
        
        # Check for README
        readme_files = ['README.md', 'readme.md', 'README.MD', 'README.txt']
        for readme in readme_files:
            readme_path = repo_path / readme
            if readme_path.exists():
                features['has_readme'] = True
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        features['readme_length'] = len(f.read())
                except Exception:
                    pass
                break
        
        # Check for docs folder
        docs_paths = ['docs', 'documentation', 'doc']
        for docs in docs_paths:
            if (repo_path / docs).exists():
                features['has_docs_folder'] = True
                break
        
        # Check for API documentation
        api_doc_files = ['openapi.yaml', 'openapi.json', 'swagger.yaml', 'swagger.json']
        for doc_file in api_doc_files:
            if len(list(repo_path.rglob(doc_file))) > 0:
                features['has_api_docs'] = True
                features['has_swagger'] = True
                break
        
        return features
    
    def extract_endpoint_density_features(self, repo_path: Path, endpoint_count: int) -> Dict:
        """
        Extract endpoint density features
        
        Args:
            repo_path: Path to repository
            endpoint_count: Number of endpoints detected
            
        Returns:
            Dictionary of endpoint density features
        """
        size_features = self.extract_codebase_size_features(repo_path)
        
        features = {
            'endpoint_count': endpoint_count,
            'endpoints_per_file': endpoint_count / size_features['total_files'] if size_features['total_files'] > 0 else 0,
            'endpoints_per_1000_lines': (endpoint_count / size_features['total_lines_of_code'] * 1000) if size_features['total_lines_of_code'] > 0 else 0,
        }
        
        return features
    
    def extract_all_features(self, repo_path: Path, endpoint_count: int = 0) -> Dict:
        """
        Extract all repository-level features
        
        Args:
            repo_path: Path to repository
            endpoint_count: Number of endpoints detected (optional)
            
        Returns:
            Complete feature dictionary
        """
        repo_path = Path(repo_path)
        
        features = {
            'repo_name': repo_path.name,
            'repo_path': str(repo_path),
        }
        
        # Extract all feature categories
        features.update(self.extract_language_features(repo_path))
        features.update(self.extract_framework_features(repo_path))
        features.update(self.extract_codebase_size_features(repo_path))
        features.update(self.extract_dependency_features(repo_path))
        features.update(self.extract_documentation_features(repo_path))
        
        if endpoint_count > 0:
            features.update(self.extract_endpoint_density_features(repo_path, endpoint_count))
        
        logger.info(f"Extracted repository features for: {repo_path.name}")
        
        return features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    extractor = RepositoryFeatureExtractor()
    
    test_path = Path("data/raw/example_repo")
    if test_path.exists():
        features = extractor.extract_all_features(test_path)
        
        print("\nRepository features:")
        for key, value in features.items():
            print(f"  {key}: {value}")

