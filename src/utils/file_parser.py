"""
File Parser Utility

Utilities for parsing JavaScript/TypeScript files.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def extract_imports(file_path: Path) -> List[str]:
    """
    Extract import statements from a file
    
    Args:
        file_path: Path to source file
        
    Returns:
        List of imported module names
    """
    imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Match ES6 imports
        es6_pattern = r"import\s+(?:[\w{},\s*]+\s+from\s+)?['\"]([^'\"]+)['\"]"
        imports.extend(re.findall(es6_pattern, content))
        
        # Match require statements
        require_pattern = r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        imports.extend(re.findall(require_pattern, content))
    
    except Exception as e:
        logger.error(f"Error extracting imports from {file_path}: {e}")
    
    return imports


def extract_exports(file_path: Path) -> List[str]:
    """
    Extract export statements from a file
    
    Args:
        file_path: Path to source file
        
    Returns:
        List of exported names
    """
    exports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Match named exports
        named_pattern = r"export\s+(?:const|let|var|function|class)\s+(\w+)"
        exports.extend(re.findall(named_pattern, content))
        
        # Match export { ... }
        braces_pattern = r"export\s+\{([^}]+)\}"
        matches = re.findall(braces_pattern, content)
        for match in matches:
            names = [n.strip().split(' as ')[0] for n in match.split(',')]
            exports.extend(names)
    
    except Exception as e:
        logger.error(f"Error extracting exports from {file_path}: {e}")
    
    return exports


def count_functions(file_path: Path) -> int:
    """
    Count functions in a file
    
    Args:
        file_path: Path to source file
        
    Returns:
        Number of functions
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Count function declarations
        function_pattern = r"\bfunction\s+\w+"
        arrow_pattern = r"\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"
        method_pattern = r"\w+\s*\([^)]*\)\s*\{"
        
        count = (
            len(re.findall(function_pattern, content)) +
            len(re.findall(arrow_pattern, content)) +
            len(re.findall(method_pattern, content))
        )
        
        return count
    
    except Exception as e:
        logger.error(f"Error counting functions in {file_path}: {e}")
        return 0


def extract_comments(file_path: Path) -> List[str]:
    """
    Extract comments from a file
    
    Args:
        file_path: Path to source file
        
    Returns:
        List of comment strings
    """
    comments = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Match single-line comments
        single_line_pattern = r"//(.*)$"
        comments.extend(re.findall(single_line_pattern, content, re.MULTILINE))
        
        # Match multi-line comments
        multi_line_pattern = r"/\*(.*?)\*/"
        comments.extend(re.findall(multi_line_pattern, content, re.DOTALL))
    
    except Exception as e:
        logger.error(f"Error extracting comments from {file_path}: {e}")
    
    return comments


def is_route_file(file_path: Path) -> bool:
    """
    Check if file is likely a route/controller file
    
    Args:
        file_path: Path to source file
        
    Returns:
        True if likely a route file
    """
    file_str = str(file_path).lower()
    
    indicators = [
        'route', 'routes',
        'controller', 'controllers',
        'api', 'endpoint', 'endpoints',
        'handler', 'handlers'
    ]
    
    return any(indicator in file_str for indicator in indicators)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_file = Path("data/raw/example_repo/src/routes/users.js")
    if test_file.exists():
        imports = extract_imports(test_file)
        exports = extract_exports(test_file)
        func_count = count_functions(test_file)
        is_route = is_route_file(test_file)
        
        print(f"Imports: {imports}")
        print(f"Exports: {exports}")
        print(f"Functions: {func_count}")
        print(f"Is route file: {is_route}")

