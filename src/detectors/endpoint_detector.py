"""
Endpoint Detector

Detects API endpoints in Node.js codebases across different frameworks.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EndpointDetector:
    """Detects API endpoints in Node.js repositories"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize endpoint detector
        
        Args:
            config: Configuration dictionary with framework patterns
        """
        self.config = config or {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load regex patterns for endpoint detection"""
        
        # Express.js patterns
        self.express_patterns = [
            r"(router|app)\.(get|post|put|delete|patch|options|head)\s*\(\s*['\"]([^'\"]+)['\"]",
            r"(router|app)\.route\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\.(get|post|put|delete|patch)",
        ]
        
        # NestJS patterns (decorator-based)
        self.nestjs_patterns = [
            r"@(Get|Post|Put|Delete|Patch)\s*\(\s*['\"]([^'\"]*)['\"]",
            r"@(Get|Post|Put|Delete|Patch)\s*\(\s*\)",  # Empty decorator
        ]
        
        # Fastify patterns
        self.fastify_patterns = [
            r"fastify\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
            r"fastify\.route\s*\(\s*\{[^}]*method:\s*['\"]([^'\"]+)['\"][^}]*url:\s*['\"]([^'\"]+)['\"]",
        ]
    
    def detect_express_endpoints(self, file_path: Path) -> List[Dict]:
        """
        Detect Express.js endpoints in a file
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for pattern in self.express_patterns:
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        groups = match.groups()
                        
                        if len(groups) >= 3:
                            method = groups[1].upper()
                            path = groups[2]
                        else:
                            continue
                        
                        endpoints.append({
                            'file': str(file_path),
                            'method': method,
                            'path': path,
                            'framework': 'express',
                            'line': content[:match.start()].count('\n') + 1
                        })
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return endpoints
    
    def detect_nestjs_endpoints(self, file_path: Path) -> List[Dict]:
        """
        Detect NestJS endpoints in a file
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Find controller decorator for base path
                controller_match = re.search(r"@Controller\s*\(\s*['\"]([^'\"]*)['\"]", content)
                base_path = controller_match.group(1) if controller_match else ''
                
                for pattern in self.nestjs_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    
                    for match in matches:
                        groups = match.groups()
                        method = groups[0].upper()
                        
                        # Handle empty decorator (uses base path only)
                        if len(groups) >= 2 and groups[1]:
                            route_path = groups[1]
                        else:
                            route_path = ''
                        
                        # Combine base path and route path
                        full_path = f"/{base_path}/{route_path}".replace('//', '/')
                        
                        endpoints.append({
                            'file': str(file_path),
                            'method': method,
                            'path': full_path,
                            'framework': 'nestjs',
                            'line': content[:match.start()].count('\n') + 1
                        })
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return endpoints
    
    def detect_fastify_endpoints(self, file_path: Path) -> List[Dict]:
        """
        Detect Fastify endpoints in a file
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for pattern in self.fastify_patterns:
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        groups = match.groups()
                        
                        if len(groups) >= 2:
                            method = groups[0].upper() if groups[0] else groups[1].upper()
                            path = groups[1] if len(groups) == 2 else groups[2]
                        else:
                            continue
                        
                        endpoints.append({
                            'file': str(file_path),
                            'method': method,
                            'path': path,
                            'framework': 'fastify',
                            'line': content[:match.start()].count('\n') + 1
                        })
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return endpoints
    
    def detect_endpoints_in_file(self, file_path: Path, framework: str = None) -> List[Dict]:
        """
        Detect endpoints in a file using appropriate framework detector
        
        Args:
            file_path: Path to source file
            framework: Framework to use ('express', 'nestjs', 'fastify', or None for auto)
            
        Returns:
            List of endpoint dictionaries
        """
        if not file_path.exists():
            return []
        
        endpoints = []
        
        # Try all detectors if framework not specified
        if framework is None or framework == 'express':
            endpoints.extend(self.detect_express_endpoints(file_path))
        
        if framework is None or framework == 'nestjs':
            endpoints.extend(self.detect_nestjs_endpoints(file_path))
        
        if framework is None or framework == 'fastify':
            endpoints.extend(self.detect_fastify_endpoints(file_path))
        
        return endpoints
    
    def detect_endpoints_in_repo(self, repo_path: Path, framework: str = None) -> List[Dict]:
        """
        Detect all endpoints in a repository
        
        Args:
            repo_path: Path to repository
            framework: Framework to use (or None for auto-detect)
            
        Returns:
            List of all detected endpoints
        """
        repo_path = Path(repo_path)
        all_endpoints = []
        
        # Search for route files
        extensions = ['.js', '.ts']
        route_dirs = ['routes', 'controllers', 'api']
        
        files_to_check = set()
        
        # Collect files matching route patterns
        for dir_name in route_dirs:
            for ext in extensions:
                for file_path in repo_path.rglob(f"{dir_name}/**/*{ext}"):
                    if 'node_modules' not in str(file_path) and 'test' not in str(file_path).lower():
                        files_to_check.add(file_path)
        
        # Also check all JS/TS files in src directories
        for ext in extensions:
            for file_path in repo_path.rglob(f"src/**/*{ext}"):
                if 'node_modules' not in str(file_path):
                    files_to_check.add(file_path)
        
        logger.info(f"Checking {len(files_to_check)} files for endpoints in {repo_path.name}")
        
        # Detect endpoints in each file
        for file_path in files_to_check:
            endpoints = self.detect_endpoints_in_file(file_path, framework)
            all_endpoints.extend(endpoints)
        
        # Deduplicate endpoints
        unique_endpoints = []
        seen = set()
        
        for ep in all_endpoints:
            key = (ep['method'], ep['path'])
            if key not in seen:
                seen.add(key)
                unique_endpoints.append(ep)
        
        logger.info(f"Found {len(unique_endpoints)} unique endpoints in {repo_path.name}")
        
        return unique_endpoints


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    detector = EndpointDetector()
    
    test_path = Path("data/raw/example_repo")
    if test_path.exists():
        endpoints = detector.detect_endpoints_in_repo(test_path)
        
        print(f"\nFound {len(endpoints)} endpoints:")
        for ep in endpoints[:10]:  # Show first 10
            print(f"  {ep['method']:6} {ep['path']:30} ({ep['framework']})")

