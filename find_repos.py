#!/usr/bin/env python3
"""
Enhanced GitHub Repository Finder for API Endpoints
Searches for JavaScript/TypeScript repositories with Express, NestJS, or Fastify endpoints
Uses Git Tree API for efficient file discovery and returns actual endpoint/schema files
"""

import os
import sys
import time
import json
import re
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class GitHubRepoFinder:
    """Finds GitHub repositories containing API endpoints with enhanced detection"""
    
    def __init__(self, github_token: str):
        """Initialize with GitHub Personal Access Token"""
        self.token = github_token
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
        self.repos_data = {}  # Dict to store unique repos
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
    def check_rate_limit(self):
        """Check and display current rate limit status"""
        url = f"{self.base_url}/rate_limit"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            core = data['resources']['core']
            search = data['resources']['search']
            print(f"Rate Limits - Core: {core['remaining']}/{core['limit']}, Search: {search['remaining']}/{search['limit']}")
            return search['remaining']
        return None
    
    def wait_for_rate_limit(self, retry_after: Optional[int] = None):
        """Wait if rate limit is exceeded"""
        if retry_after:
            wait_time = retry_after + 5
        elif self.rate_limit_reset:
            wait_time = max(self.rate_limit_reset - time.time() + 5, 0)
        else:
            wait_time = 60
        
        print(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
        time.sleep(wait_time)
    
    def search_code(self, query: str, max_pages: int = 10) -> List[Dict]:
        """
        Search GitHub code with a specific query
        Returns list of repository information
        """
        results = []
        page = 1
        
        while page <= max_pages:
            url = f"{self.base_url}/search/code"
            params = {
                "q": query,
                "per_page": 100,
                "page": page
            }
            
            print(f"  Searching page {page} for: {query[:60]}...")
            
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 403:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        self.wait_for_rate_limit(int(retry_after))
                        continue
                    else:
                        self.wait_for_rate_limit()
                        continue
                
                if response.status_code == 422:
                    print(f"  Query validation error: {query}")
                    break
                
                if response.status_code != 200:
                    print(f"  Error: Status {response.status_code}")
                    break
                
                data = response.json()
                items = data.get('items', [])
                
                if not items:
                    print(f"  No more results on page {page}")
                    break
                
                results.extend(items)
                print(f"  Found {len(items)} code matches")
                
                # Check if there are more pages
                if len(items) < 100:
                    break
                
                page += 1
                time.sleep(3)  # Respect rate limits
                
            except requests.exceptions.RequestException as e:
                print(f"  Network error: {e}")
                time.sleep(5)
                continue
        
        return results
    
    def get_repo_details(self, owner: str, repo: str) -> Optional[Dict]:
        """Get detailed repository information"""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 403:
                self.wait_for_rate_limit()
                response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  Error fetching repo details: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  Network error fetching repo: {e}")
            return None
    
    def get_repo_tree(self, owner: str, repo: str, default_branch: str = None) -> Optional[List[Dict]]:
        """
        Fetch entire repository file tree using Git Tree API
        Returns list of all files in one API call
        """
        # Get default branch if not provided
        if not default_branch:
            repo_details = self.get_repo_details(owner, repo)
            if not repo_details:
                return None
            default_branch = repo_details.get('default_branch', 'main')
        
        url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{default_branch}"
        params = {"recursive": "1"}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=20)
            
            if response.status_code == 403:
                self.wait_for_rate_limit()
                response = requests.get(url, headers=self.headers, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('tree', [])
            else:
                print(f"  Error fetching tree: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  Network error fetching tree: {e}")
            return None
    
    def parse_endpoint_files(self, tree: List[Dict]) -> List[str]:
        """
        Identify files likely to contain API endpoints from git tree
        Returns list of file paths
        """
        endpoint_files = []
        
        # Patterns for endpoint files
        patterns = [
            # Route/Router files
            (r'/routes?/', 'route directory'),
            (r'/routers?/', 'router directory'),
            # Controller files
            (r'/controllers?/', 'controller directory'),
            (r'\.controller\.(js|ts)$', 'controller file'),
            # API directories
            (r'/api/', 'api directory'),
            (r'/src/api/', 'src api directory'),
            (r'/server/api/', 'server api directory'),
            # NestJS patterns
            (r'\.module\.(ts)$', 'nestjs module'),
            # Entry points
            (r'^(app|server|index)\.(js|ts)$', 'entry point'),
            (r'^src/(app|server|index)\.(js|ts)$', 'src entry point'),
        ]
        
        for item in tree:
            if item['type'] != 'blob':  # Only files
                continue
            
            path = item['path']
            
            # Skip node_modules, tests, and other non-source directories
            if any(skip in path.lower() for skip in ['node_modules', 'test', 'dist', 'build', '.git']):
                continue
            
            # Check patterns
            for pattern, _ in patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    endpoint_files.append(path)
                    break
        
        return endpoint_files
    
    def parse_schema_files(self, tree: List[Dict]) -> Tuple[List[str], Set[str]]:
        """
        Identify schema-related files from git tree
        Returns tuple of (file_paths, schema_types)
        """
        schema_files = []
        schema_types = set()
        
        for item in tree:
            if item['type'] != 'blob':
                continue
            
            path = item['path'].lower()
            
            # Skip node_modules
            if 'node_modules' in path:
                continue
            
            # OpenAPI/Swagger files
            if any(name in path for name in ['openapi', 'swagger']):
                if path.endswith(('.json', '.yaml', '.yml')):
                    schema_files.append(item['path'])
                    schema_types.add('OpenAPI')
                    continue
            
            # JSON Schema files
            if path.endswith('.schema.json') or '/schemas/' in path:
                schema_files.append(item['path'])
                schema_types.add('JSON Schema')
                continue
            
            # Validator files
            if '/validators/' in path or '/validation/' in path or 'validator' in path:
                if path.endswith(('.js', '.ts')):
                    schema_files.append(item['path'])
                    schema_types.add('Validators')
                    continue
            
            # TypeScript definition files
            if path.endswith('.d.ts') or '/types/' in path or '/interfaces/' in path:
                schema_files.append(item['path'])
                schema_types.add('TypeScript')
                continue
            
            # GraphQL schemas
            if path.endswith(('.graphql', '.gql')) or 'schema.graphql' in path:
                schema_files.append(item['path'])
                schema_types.add('GraphQL')
                continue
            
            # Prisma schema
            if path.endswith('schema.prisma'):
                schema_files.append(item['path'])
                schema_types.add('Prisma')
                continue
            
            # Drizzle config
            if 'drizzle.config' in path:
                schema_files.append(item['path'])
                schema_types.add('Drizzle')
                continue
        
        return schema_files, schema_types
    
    def get_package_json(self, owner: str, repo: str) -> Optional[Dict]:
        """Fetch and parse package.json if it exists"""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/package.json"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('encoding') == 'base64':
                    import base64
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return json.loads(content)
        except:
            pass
        return None
    
    def verify_framework_in_package(self, package_json: Optional[Dict]) -> Dict[str, bool]:
        """Verify framework presence in package.json dependencies"""
        frameworks = {
            'express': False,
            'nestjs': False,
            'fastify': False
        }
        
        if not package_json:
            return frameworks
        
        all_deps = {}
        all_deps.update(package_json.get('dependencies', {}))
        all_deps.update(package_json.get('devDependencies', {}))
        
        # Check for frameworks
        if 'express' in all_deps:
            frameworks['express'] = True
        if '@nestjs/core' in all_deps or '@nestjs/common' in all_deps:
            frameworks['nestjs'] = True
        if 'fastify' in all_deps:
            frameworks['fastify'] = True
        
        return frameworks
    
    def analyze_package_json(self, package_json: Optional[Dict]) -> Dict:
        """Deep analysis of package.json for validators, tools, etc."""
        analysis = {
            'validator_libraries': [],
            'openapi_tools': [],
            'has_typescript': False,
            'has_graphql': False,
        }
        
        if not package_json:
            return analysis
        
        all_deps = {}
        all_deps.update(package_json.get('dependencies', {}))
        all_deps.update(package_json.get('devDependencies', {}))
        
        # Validator libraries
        validators = ['joi', 'zod', 'class-validator', 'yup', 'ajv', 'io-ts', 'superstruct', 'valibot', 'validator']
        analysis['validator_libraries'] = [v for v in validators if v in all_deps]
        
        # OpenAPI tools
        openapi_tools = ['swagger-jsdoc', '@nestjs/swagger', 'fastify-swagger', '@openapitools/openapi-generator-cli']
        analysis['openapi_tools'] = [t for t in openapi_tools if t in all_deps]
        
        # TypeScript
        analysis['has_typescript'] = 'typescript' in all_deps
        
        # GraphQL
        graphql_libs = ['graphql', 'apollo-server', '@nestjs/graphql', 'apollo-server-express']
        analysis['has_graphql'] = any(lib in all_deps for lib in graphql_libs)
        
        return analysis
    
    def check_schema_indicators(self, tree: List[Dict], package_json: Optional[Dict]) -> Dict:
        """Enhanced schema detection using tree and package.json"""
        indicators = {
            'has_openapi': False,
            'has_swagger': False,
            'has_validators': False,
            'has_typescript': False,
            'has_api_docs_folder': False,
            'has_tests': False,
            'schema_files': [],
            'schema_types': set(),
        }
        
        # Parse schema files from tree
        schema_files, schema_types = self.parse_schema_files(tree)
        indicators['schema_files'] = schema_files
        indicators['schema_types'] = schema_types
        
        # Set boolean flags
        indicators['has_openapi'] = 'OpenAPI' in schema_types
        indicators['has_swagger'] = 'OpenAPI' in schema_types  # Same as OpenAPI
        indicators['has_typescript'] = 'TypeScript' in schema_types
        
        # Check for API docs folder
        for item in tree:
            if item['type'] == 'tree':
                path = item['path'].lower()
                if any(doc in path for doc in ['/docs', '/api-docs', '/swagger', '/documentation']):
                    indicators['has_api_docs_folder'] = True
                    break
        
        # Check for test files
        for item in tree:
            if item['type'] == 'blob':
                path = item['path'].lower()
                if any(test in path for test in ['.test.', '.spec.', '/tests/', '/test/']):
                    indicators['has_tests'] = True
                    break
        
        # Package.json analysis
        pkg_analysis = self.analyze_package_json(package_json)
        if pkg_analysis['validator_libraries']:
            indicators['has_validators'] = True
        
        # Override TypeScript if in package.json
        if pkg_analysis['has_typescript']:
            indicators['has_typescript'] = True
        
        return indicators
    
    def calculate_quality_score(self, repo_details: Dict, schema_indicators: Dict, 
                                 package_analysis: Dict, framework_verified: Dict) -> int:
        """
        Calculate quality score (0-100) based on multiple factors
        """
        score = 0
        
        # Repository size and stars (0-20 points)
        stars = repo_details.get('stargazers_count', 0)
        size_kb = repo_details.get('size', 0)
        
        if stars > 1000:
            score += 15
        elif stars > 100:
            score += 10
        elif stars > 10:
            score += 5
        
        if size_kb > 1000:
            score += 5
        elif size_kb > 500:
            score += 3
        
        # Schema indicators present (0-30 points)
        if schema_indicators.get('has_openapi'):
            score += 10
        if schema_indicators.get('has_validators'):
            score += 10
        if len(schema_indicators.get('schema_types', [])) > 2:
            score += 10
        elif len(schema_indicators.get('schema_types', [])) > 0:
            score += 5
        
        # Framework verified in package.json (0-20 points)
        if any(framework_verified.values()):
            score += 20
        
        # TypeScript usage (0-15 points)
        if schema_indicators.get('has_typescript'):
            score += 15
        
        # Documentation/tests present (0-15 points)
        if schema_indicators.get('has_api_docs_folder'):
            score += 8
        if schema_indicators.get('has_tests'):
            score += 7
        
        return min(score, 100)  # Cap at 100
    
    def process_search_results(self, results: List[Dict], query: str, framework: str):
        """Process search results and extract repository information with enhanced detection"""
        for item in results:
            repo_info = item.get('repository', {})
            repo_full_name = repo_info.get('full_name', '')
            
            if not repo_full_name:
                continue
            
            # Initialize repo entry if not exists
            if repo_full_name not in self.repos_data:
                owner, repo_name = repo_full_name.split('/')
                
                # Get detailed repo info
                print(f"  Fetching details for: {repo_full_name}")
                repo_details = self.get_repo_details(owner, repo_name)
                
                if not repo_details:
                    continue
                
                # Filter by minimum size (100 KB = 100,000 bytes)
                size_kb = repo_details.get('size', 0)
                if size_kb < 100:
                    print(f"  Skipping {repo_full_name} (too small: {size_kb} KB)")
                    continue
                
                # Get repository tree
                print(f"  Fetching file tree...")
                default_branch = repo_details.get('default_branch', 'main')
                tree = self.get_repo_tree(owner, repo_name, default_branch)
                
                if not tree:
                    print(f"  Could not fetch tree for {repo_full_name}")
                    continue
                
                # Parse endpoint and schema files
                endpoint_files = self.parse_endpoint_files(tree)
                
                # Get package.json
                package_json = self.get_package_json(owner, repo_name)
                
                # Verify frameworks
                framework_verified = self.verify_framework_in_package(package_json)
                
                # Analyze package.json
                package_analysis = self.analyze_package_json(package_json)
                
                # Check schema indicators
                schema_indicators = self.check_schema_indicators(tree, package_json)
                
                # Calculate quality score
                quality_score = self.calculate_quality_score(
                    repo_details, schema_indicators, package_analysis, framework_verified
                )
                
                self.repos_data[repo_full_name] = {
                    'repo_url': repo_details['html_url'],
                    'repo_name': repo_name,
                    'owner': owner,
                    'full_name': repo_full_name,
                    'stars': repo_details.get('stargazers_count', 0),
                    'language': repo_details.get('language', 'Unknown'),
                    'description': repo_details.get('description', ''),
                    'last_updated': repo_details.get('updated_at', ''),
                    'size_kb': size_kb,
                    'frameworks': set(),
                    'matched_patterns': [],
                    'api_url': repo_details['url'],
                    'endpoint_files': endpoint_files,
                    'endpoint_file_count': len(endpoint_files),
                    'schema_files': schema_indicators['schema_files'],
                    'schema_file_count': len(schema_indicators['schema_files']),
                    'has_openapi': schema_indicators['has_openapi'],
                    'has_swagger': schema_indicators['has_swagger'],
                    'has_validators': schema_indicators['has_validators'],
                    'validator_libraries': ','.join(package_analysis['validator_libraries']),
                    'validator_count': len(package_analysis['validator_libraries']),
                    'has_typescript': schema_indicators['has_typescript'],
                    'has_api_docs_folder': schema_indicators['has_api_docs_folder'],
                    'has_tests': schema_indicators['has_tests'],
                    'schema_types': ','.join(sorted(schema_indicators['schema_types'])),
                    'framework_in_package': any(framework_verified.values()),
                    'quality_score': quality_score,
                }
                
                time.sleep(2)  # Rate limiting
            
            # Add framework and pattern info
            self.repos_data[repo_full_name]['frameworks'].add(framework)
            self.repos_data[repo_full_name]['matched_patterns'].append(query)
    
    def search_all_patterns(self):
        """Execute all search queries for different frameworks with expanded patterns"""
        
        # Expanded search patterns covering all HTTP methods
        patterns = {
            'Express': [
                'language:JavaScript "app.get("',
                'language:JavaScript "app.post("',
                'language:JavaScript "app.put("',
                'language:JavaScript "app.delete("',
                'language:JavaScript "router.get("',
                'language:JavaScript "router.post("',
                'language:JavaScript "router.put("',
                'language:JavaScript "router.delete("',
                'language:TypeScript "app.get("',
                'language:TypeScript "router.post("',
            ],
            'NestJS': [
                'language:TypeScript "@Get("',
                'language:TypeScript "@Post("',
                'language:TypeScript "@Put("',
                'language:TypeScript "@Delete("',
                'language:TypeScript "@Patch("',
                'language:TypeScript "@Controller("',
                'language:TypeScript "@ApiTags("',
                'language:TypeScript "@ApiOperation("',
            ],
            'Fastify': [
                'language:JavaScript "fastify.get("',
                'language:JavaScript "fastify.post("',
                'language:JavaScript "fastify.put("',
                'language:JavaScript "fastify.delete("',
                'language:JavaScript "fastify.route("',
                'language:TypeScript "fastify.get("',
            ]
        }
        
        print("Starting GitHub repository search with enhanced detection...")
        print("=" * 60)
        
        # Check initial rate limit
        self.check_rate_limit()
        
        for framework, queries in patterns.items():
            print(f"\n🔍 Searching {framework} repositories...")
            print("-" * 60)
            
            for query in queries:
                results = self.search_code(query, max_pages=10)
                print(f"  Processing {len(results)} code matches...")
                self.process_search_results(results, query, framework)
                print(f"  Total unique repos so far: {len(self.repos_data)}")
                
                # Save checkpoint
                self.save_checkpoint()
        
        print("\n" + "=" * 60)
        print(f"✅ Search complete! Found {len(self.repos_data)} unique repositories")
    
    def save_checkpoint(self):
        """Save current results to a checkpoint file"""
        if self.repos_data:
            checkpoint_file = 'checkpoint.json'
            checkpoint_data = {}
            for key, value in self.repos_data.items():
                checkpoint_data[key] = value.copy()
                checkpoint_data[key]['frameworks'] = list(checkpoint_data[key]['frameworks'])
                checkpoint_data[key]['schema_types'] = list(checkpoint_data[key].get('schema_types', set()))
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self):
        """Load results from checkpoint file if exists"""
        checkpoint_file = 'checkpoint.json'
        if os.path.exists(checkpoint_file):
            print("Loading from checkpoint...")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                for key, value in checkpoint_data.items():
                    self.repos_data[key] = value
                    self.repos_data[key]['frameworks'] = set(value['frameworks'])
            print(f"Loaded {len(self.repos_data)} repos from checkpoint")
    
    def export_to_csv(self, output_file: str = 'github_repos.csv'):
        """Export collected repository data to CSV with enhanced columns"""
        if not self.repos_data:
            print("No repositories to export!")
            return
        
        # Prepare data for CSV
        rows = []
        for repo in self.repos_data.values():
            row = {
                'repo_url': repo['repo_url'],
                'repo_name': repo['repo_name'],
                'owner': repo['owner'],
                'full_name': repo['full_name'],
                'stars': repo['stars'],
                'size_kb': repo['size_kb'],
                'language': repo['language'],
                'description': repo['description'],
                'last_updated': repo['last_updated'],
                'frameworks': ','.join(sorted(repo['frameworks'])),
                'matched_patterns': ' | '.join(repo['matched_patterns']),
                'api_url': repo['api_url'],
                'endpoint_files': ' | '.join(repo.get('endpoint_files', [])),
                'endpoint_file_count': repo.get('endpoint_file_count', 0),
                'schema_files': ' | '.join(repo.get('schema_files', [])),
                'schema_file_count': repo.get('schema_file_count', 0),
                'quality_score': repo.get('quality_score', 0),
                'has_openapi': repo['has_openapi'],
                'has_swagger': repo['has_swagger'],
                'has_validators': repo['has_validators'],
                'validator_libraries': repo['validator_libraries'],
                'validator_count': repo.get('validator_count', 0),
                'has_typescript': repo['has_typescript'],
                'has_api_docs_folder': repo.get('has_api_docs_folder', False),
                'has_tests': repo.get('has_tests', False),
                'schema_types': repo.get('schema_types', ''),
                'framework_in_package': repo.get('framework_in_package', False),
            }
            rows.append(row)
        
        # Create DataFrame and sort by quality score, then stars
        df = pd.DataFrame(rows)
        df = df.sort_values(['quality_score', 'stars'], ascending=False)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        print(f"\n✅ Exported {len(rows)} repositories to {output_file}")
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"  Total repositories: {len(rows)}")
        print(f"  Average stars: {df['stars'].mean():.1f}")
        print(f"  Average quality score: {df['quality_score'].mean():.1f}")
        print(f"  Average endpoint files per repo: {df['endpoint_file_count'].mean():.1f}")
        print(f"  Average schema files per repo: {df['schema_file_count'].mean():.1f}")
        print(f"\n  By Framework:")
        for fw in ['Express', 'NestJS', 'Fastify']:
            count = df['frameworks'].str.contains(fw).sum()
            print(f"    {fw}: {count}")
        print(f"\n  Schema Indicators:")
        print(f"    Has OpenAPI: {df['has_openapi'].sum()}")
        print(f"    Has Validators: {df['has_validators'].sum()}")
        print(f"    Has TypeScript: {df['has_typescript'].sum()}")
        print(f"    Has API Docs Folder: {df['has_api_docs_folder'].sum()}")
        print(f"    Has Tests: {df['has_tests'].sum()}")
        print(f"\n  Quality Scores:")
        print(f"    High (80-100): {(df['quality_score'] >= 80).sum()}")
        print(f"    Medium (50-79): {((df['quality_score'] >= 50) & (df['quality_score'] < 80)).sum()}")
        print(f"    Low (0-49): {(df['quality_score'] < 50).sum()}")


def main():
    """Main execution function"""
    
    # Get GitHub token from environment
    github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        print("❌ Error: GITHUB_TOKEN not found in environment variables")
        print("Please create a .env file with your GitHub Personal Access Token")
        print("See .env.example for template")
        sys.exit(1)
    
    # Initialize finder
    finder = GitHubRepoFinder(github_token)
    
    # Load checkpoint if exists
    finder.load_checkpoint()
    
    # Search for repositories
    try:
        finder.search_all_patterns()
    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'github_repos_{timestamp}.csv'
    finder.export_to_csv(output_file)


if __name__ == '__main__':
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed, reading from environment only")
    
    main()
