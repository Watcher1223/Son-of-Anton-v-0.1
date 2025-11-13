"""
Repository Filter

Filters and validates repositories to ensure they meet project requirements.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RepositoryFilter:
    """Filters repositories based on quality criteria"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize repository filter
        
        Args:
            config: Configuration dictionary with filtering criteria
        """
        self.config = config or {}
        self.min_endpoints = self.config.get('min_endpoints_per_repo', 5)
    
    def has_package_json(self, repo_path: Path) -> bool:
        """Check if repository has package.json"""
        package_json = repo_path / "package.json"
        return package_json.exists()
    
    def get_dependencies(self, repo_path: Path) -> Dict:
        """
        Extract dependencies from package.json
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary with dependencies and devDependencies
        """
        package_json = repo_path / "package.json"
        
        if not package_json.exists():
            return {'dependencies': {}, 'devDependencies': {}}
        
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'dependencies': data.get('dependencies', {}),
                    'devDependencies': data.get('devDependencies', {})
                }
        except Exception as e:
            logger.error(f"Error reading package.json in {repo_path}: {e}")
            return {'dependencies': {}, 'devDependencies': {}}
    
    def detect_framework(self, repo_path: Path) -> Optional[str]:
        """
        Detect web framework used in repository
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Framework name ('express', 'nestjs', 'fastify', 'other', or None)
        """
        deps = self.get_dependencies(repo_path)
        all_deps = {**deps['dependencies'], **deps['devDependencies']}
        
        if '@nestjs/core' in all_deps or '@nestjs/common' in all_deps:
            return 'nestjs'
        elif 'fastify' in all_deps:
            return 'fastify'
        elif 'express' in all_deps:
            return 'express'
        elif any('koa' in dep for dep in all_deps):
            return 'koa'
        elif 'hapi' in all_deps or '@hapi/hapi' in all_deps:
            return 'hapi'
        
        return None
    
    def count_files(self, repo_path: Path, extensions: List[str] = None) -> int:
        """
        Count files in repository by extension
        
        Args:
            repo_path: Path to repository
            extensions: List of file extensions to count (e.g., ['.js', '.ts'])
            
        Returns:
            Number of matching files
        """
        if extensions is None:
            extensions = ['.js', '.ts', '.jsx', '.tsx']
        
        count = 0
        for ext in extensions:
            count += len(list(repo_path.rglob(f"*{ext}")))
        
        return count
    
    def count_lines_of_code(self, repo_path: Path) -> int:
        """
        Count total lines of code in repository
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Total line count
        """
        extensions = ['.js', '.ts', '.jsx', '.tsx']
        total_lines = 0
        
        for ext in extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                # Skip node_modules and other common excluded directories
                if 'node_modules' in str(file_path) or '.git' in str(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += sum(1 for _ in f)
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        return total_lines
    
    def validate_repository(self, repo_path: Path, schema_type: str) -> Dict:
        """
        Validate repository meets quality criteria
        
        Args:
            repo_path: Path to repository
            schema_type: Expected schema type
            
        Returns:
            Dictionary with validation results
        """
        repo_path = Path(repo_path)
        
        validation = {
            'path': str(repo_path),
            'valid': False,
            'has_package_json': False,
            'framework': None,
            'file_count': 0,
            'line_count': 0,
            'issues': []
        }
        
        # Check package.json
        if not self.has_package_json(repo_path):
            validation['issues'].append("Missing package.json")
            return validation
        
        validation['has_package_json'] = True
        
        # Detect framework
        framework = self.detect_framework(repo_path)
        validation['framework'] = framework
        
        if not framework:
            validation['issues'].append("No supported framework detected")
        
        # Count files and lines
        validation['file_count'] = self.count_files(repo_path)
        validation['line_count'] = self.count_lines_of_code(repo_path)
        
        if validation['file_count'] < 5:
            validation['issues'].append(f"Too few files: {validation['file_count']}")
        
        if validation['line_count'] < 100:
            validation['issues'].append(f"Too few lines of code: {validation['line_count']}")
        
        # Repository is valid if no critical issues
        validation['valid'] = len(validation['issues']) == 0
        
        return validation
    
    def filter_repositories(self, repo_paths: List[Path], schema_types: List[str]) -> List[Dict]:
        """
        Filter a list of repositories
        
        Args:
            repo_paths: List of repository paths
            schema_types: Corresponding schema types for each repo
            
        Returns:
            List of valid repositories with metadata
        """
        valid_repos = []
        
        for repo_path, schema_type in zip(repo_paths, schema_types):
            logger.info(f"Validating: {repo_path}")
            validation = self.validate_repository(repo_path, schema_type)
            
            if validation['valid']:
                logger.info(f"  ✓ Valid repository")
                valid_repos.append(validation)
            else:
                logger.warning(f"  ✗ Invalid: {', '.join(validation['issues'])}")
        
        logger.info(f"\nFiltered: {len(valid_repos)}/{len(repo_paths)} valid repositories")
        return valid_repos


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    filter_tool = RepositoryFilter()
    
    # Example: validate a single repository
    test_path = Path("data/raw/example_repo")
    if test_path.exists():
        result = filter_tool.validate_repository(test_path, "openapi")
        print(json.dumps(result, indent=2))

