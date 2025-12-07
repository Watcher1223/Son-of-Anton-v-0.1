"""
Endpoint Detector

Detects API endpoints in Node.js codebases across different frameworks.
Supports: Express.js, NestJS, Fastify, Hapi.js, Koa.js
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
        
        # Express.js patterns (including server alias and template literals)
        self.express_patterns = [
            # Standard patterns with quotes
            r"(router|app|server)\.(get|post|put|delete|patch|options|head)\s*\(\s*['\"]([^'\"]+)['\"]",
            # Template literals
            r"(router|app|server)\.(get|post|put|delete|patch|options|head)\s*\(\s*`([^`]+)`",
            # Route method chaining
            r"(router|app|server)\.route\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\.(get|post|put|delete|patch|options|head)",
            r"(router|app|server)\.route\s*\(\s*`([^`]+)`\s*\)\s*\.(get|post|put|delete|patch|options|head)",
        ]
        
        # NestJS patterns (decorator-based) - expanded with All, Options, Head
        self.nestjs_patterns = [
            # Standard decorators with quotes
            r"@(Get|Post|Put|Delete|Patch|All|Options|Head)\s*\(\s*['\"]([^'\"]*)['\"]",
            # Template literals
            r"@(Get|Post|Put|Delete|Patch|All|Options|Head)\s*\(\s*`([^`]*)`",
            # Empty decorator (uses base path only)
            r"@(Get|Post|Put|Delete|Patch|All|Options|Head)\s*\(\s*\)",
        ]
        
        # Fastify patterns (including app/server aliases)
        self.fastify_patterns = [
            # Standard patterns
            r"(fastify|app|server)\.(get|post|put|delete|patch|options|head)\s*\(\s*['\"]([^'\"]+)['\"]",
            # Template literals
            r"(fastify|app|server)\.(get|post|put|delete|patch|options|head)\s*\(\s*`([^`]+)`",
            # Route object pattern
            r"(fastify|app|server)\.route\s*\(\s*\{[^}]*method:\s*['\"]([^'\"]+)['\"][^}]*url:\s*['\"]([^'\"]+)['\"]",
            # Reverse order (url before method)
            r"(fastify|app|server)\.route\s*\(\s*\{[^}]*url:\s*['\"]([^'\"]+)['\"][^}]*method:\s*['\"]([^'\"]+)['\"]",
        ]
        
        # Hapi.js patterns - more specific to avoid false positives
        self.hapi_patterns = [
            # server.route with object - require handler property nearby
            r"server\.route\s*\(\s*\{[^}]*method:\s*['\"]([A-Z]+)['\"][^}]*path:\s*['\"]([^'\"]+)['\"][^}]*handler:",
            r"server\.route\s*\(\s*\{[^}]*path:\s*['\"]([^'\"]+)['\"][^}]*method:\s*['\"]([A-Z]+)['\"][^}]*handler:",
        ]
        
        # Koa.js patterns (koa-router) - require koa import check in detector
        self.koa_patterns = [
            # router.method patterns - only match if it looks like a route definition
            r"router\.(get|post|put|delete|patch)\s*\(\s*['\"](/[^'\"]*)['\"]",
            r"router\.(get|post|put|delete|patch)\s*\(\s*`(/[^`]*)`",
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
                            # Check if it's route chaining pattern (path comes before method)
                            # Use '.route' to avoid matching 'router'
                            if '.route' in pattern:
                                method = groups[2].upper()
                                path = groups[1]
                            else:
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
                
                # Find controller decorator for base path (quotes or template literals)
                controller_match = re.search(r"@Controller\s*\(\s*['\"`]([^'\"`]*)['\"`]", content)
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
                    matches = re.finditer(pattern, content, re.DOTALL)
                    
                    for match in matches:
                        groups = match.groups()
                        
                        if 'route' in pattern and len(groups) >= 3:
                            # Route object pattern - need to figure out order
                            if 'url' in pattern.split('method')[0]:
                                # url before method
                                path = groups[1]
                                method = groups[2].upper()
                            else:
                                # method before url
                                method = groups[1].upper()
                                path = groups[2]
                        elif len(groups) >= 3:
                            # Standard method pattern
                            method = groups[1].upper()
                            path = groups[2]
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
    
    def detect_hapi_endpoints(self, file_path: Path) -> List[Dict]:
        """
        Detect Hapi.js endpoints in a file
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Check if this looks like a Hapi file
                if '@hapi' not in content and 'hapi' not in content.lower():
                    return endpoints
                
                for pattern in self.hapi_patterns:
                    matches = re.finditer(pattern, content, re.DOTALL)
                    
                    for match in matches:
                        groups = match.groups()
                        
                        if len(groups) >= 2:
                            # Determine order based on pattern
                            if 'path' in pattern.split('method')[0]:
                                # path before method
                                path = groups[0]
                                method = groups[1].upper()
                            else:
                                # method before path
                                method = groups[0].upper()
                                path = groups[1]
                        else:
                            continue
                        
                        endpoints.append({
                            'file': str(file_path),
                            'method': method,
                            'path': path,
                            'framework': 'hapi',
                            'line': content[:match.start()].count('\n') + 1
                        })
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return endpoints
    
    def detect_koa_endpoints(self, file_path: Path) -> List[Dict]:
        """
        Detect Koa.js endpoints in a file
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Check if this is actually a Koa file (must import koa)
                # Be strict to avoid false positives with Express router
                if 'koa' not in content.lower():
                    return endpoints
                if "require('koa" not in content and 'from "koa' not in content and "from 'koa" not in content:
                    return endpoints
                
                for pattern in self.koa_patterns:
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        groups = match.groups()
                        
                        if len(groups) >= 2:
                            method = groups[0].upper()
                            path = groups[1]
                        else:
                            continue
                        
                        endpoints.append({
                            'file': str(file_path),
                            'method': method,
                            'path': path,
                            'framework': 'koa',
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
            framework: Framework to use ('express', 'nestjs', 'fastify', 'hapi', 'koa', or None for auto)
            
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
        
        if framework is None or framework == 'hapi':
            endpoints.extend(self.detect_hapi_endpoints(file_path))
        
        if framework is None or framework == 'koa':
            endpoints.extend(self.detect_koa_endpoints(file_path))
        
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
        
        # Expanded file extensions (including ES modules)
        extensions = ['.js', '.ts', '.mjs', '.cjs']
        
        # Expanded search directories
        route_dirs = ['routes', 'controllers', 'api', 'lib', 'modules', 'endpoints', 'handlers']
        
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
        
        # Check root-level files (for small projects)
        for ext in extensions:
            for file_path in repo_path.glob(f"*{ext}"):
                if 'node_modules' not in str(file_path) and file_path.is_file():
                    files_to_check.add(file_path)
        
        # Check app/ directory (common in some frameworks)
        for ext in extensions:
            for file_path in repo_path.rglob(f"app/**/*{ext}"):
                if 'node_modules' not in str(file_path):
                    files_to_check.add(file_path)
        
        logger.info(f"Checking {len(files_to_check)} files for endpoints in {repo_path.name}")
        
        # Detect endpoints in each file
        for file_path in files_to_check:
            endpoints = self.detect_endpoints_in_file(file_path, framework)
            all_endpoints.extend(endpoints)
        
        # Deduplicate endpoints (include file to allow same path in different files)
        unique_endpoints = []
        seen = set()
        
        for ep in all_endpoints:
            # Use method, path, and file basename for dedup
            key = (ep['method'], ep['path'], Path(ep['file']).name)
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
