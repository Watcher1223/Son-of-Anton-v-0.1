"""
Framework Detector Utility

Detects web frameworks used in repositories.
"""

import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def detect_framework(repo_path: Path) -> Optional[str]:
    """
    Detect web framework used in repository
    
    Args:
        repo_path: Path to repository
        
    Returns:
        Framework name or None
    """
    package_json = repo_path / "package.json"
    
    if not package_json.exists():
        return None
    
    try:
        with open(package_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
        
        # Check for frameworks in priority order
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
        
    except Exception as e:
        logger.error(f"Error detecting framework: {e}")
        return None


def is_typescript_project(repo_path: Path) -> bool:
    """
    Check if repository is primarily TypeScript
    
    Args:
        repo_path: Path to repository
        
    Returns:
        True if TypeScript project
    """
    # Check for tsconfig.json
    if (repo_path / "tsconfig.json").exists():
        return True
    
    # Count TS vs JS files
    ts_count = len([f for f in repo_path.rglob("*.ts") if 'node_modules' not in str(f)])
    js_count = len([f for f in repo_path.rglob("*.js") if 'node_modules' not in str(f)])
    
    return ts_count > js_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_path = Path("data/raw/example_repo")
    if test_path.exists():
        framework = detect_framework(test_path)
        is_ts = is_typescript_project(test_path)
        
        print(f"Framework: {framework}")
        print(f"TypeScript: {is_ts}")

