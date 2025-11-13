"""
GitHub Repository Scraper

Searches for and clones Node.js repositories containing API schema definitions.
"""

import os
import time
from typing import List, Dict, Optional
from pathlib import Path

from github import Github, Repository
from git import Repo
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GitHubScraper:
    """Scrapes GitHub for Node.js repositories with API schemas"""
    
    def __init__(self, github_token: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize GitHub scraper
        
        Args:
            github_token: GitHub personal access token
            config: Configuration dictionary with scraping parameters
        """
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN env var or pass token.")
        
        self.github = Github(self.token)
        self.config = config or {}
        self.min_stars = self.config.get('min_stars', 10)
        self.min_size_kb = self.config.get('min_repo_size_kb', 100)
        self.max_size_mb = self.config.get('max_repo_size_mb', 100)
        self.languages = self.config.get('languages', ['JavaScript', 'TypeScript'])
        self.api_delay = float(os.getenv("GITHUB_API_DELAY", "1.0"))
        
        logger.info(f"GitHubScraper initialized with token: {self.token[:10]}...")
    
    def search_repositories(
        self, 
        schema_type: str, 
        max_results: int = 50
    ) -> List[Repository.Repository]:
        """
        Search GitHub for repositories matching schema type criteria
        
        Args:
            schema_type: Type of schema to search for ('openapi', 'validator', 'typedef')
            max_results: Maximum number of repositories to return
            
        Returns:
            List of GitHub Repository objects
        """
        query_map = {
            'openapi': 'openapi OR swagger language:JavaScript OR language:TypeScript',
            'validator': 'joi OR zod OR class-validator language:JavaScript OR language:TypeScript',
            'typedef': 'dto OR interface language:TypeScript'
        }
        
        if schema_type not in query_map:
            raise ValueError(f"Invalid schema_type: {schema_type}")
        
        query = query_map[schema_type]
        query += f" stars:>={self.min_stars}"
        query += f" size:{self.min_size_kb}..{self.max_size_mb * 1024}"
        
        logger.info(f"Searching repositories with query: {query}")
        
        try:
            repositories = self.github.search_repositories(
                query=query,
                sort='stars',
                order='desc'
            )
            
            results = []
            for repo in repositories[:max_results]:
                logger.info(f"Found: {repo.full_name} (stars: {repo.stargazers_count}, size: {repo.size}KB)")
                results.append(repo)
                time.sleep(self.api_delay)  # Rate limiting
                
                if len(results) >= max_results:
                    break
            
            logger.info(f"Found {len(results)} repositories for schema type: {schema_type}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []
    
    def clone_repository(
        self, 
        repo: Repository.Repository, 
        target_dir: Path
    ) -> Optional[Path]:
        """
        Clone a GitHub repository to local filesystem
        
        Args:
            repo: GitHub Repository object
            target_dir: Base directory to clone into
            
        Returns:
            Path to cloned repository or None if failed
        """
        repo_name = repo.full_name.replace('/', '_')
        clone_path = target_dir / repo_name
        
        if clone_path.exists():
            logger.warning(f"Repository already exists: {clone_path}")
            return clone_path
        
        try:
            logger.info(f"Cloning {repo.full_name} to {clone_path}")
            Repo.clone_from(
                repo.clone_url,
                clone_path,
                depth=self.config.get('clone_depth', 1)
            )
            logger.info(f"Successfully cloned: {repo.full_name}")
            return clone_path
            
        except Exception as e:
            logger.error(f"Error cloning {repo.full_name}: {e}")
            return None
    
    def get_repository_metadata(self, repo: Repository.Repository) -> Dict:
        """
        Extract metadata from GitHub repository
        
        Args:
            repo: GitHub Repository object
            
        Returns:
            Dictionary of repository metadata
        """
        try:
            return {
                'full_name': repo.full_name,
                'name': repo.name,
                'description': repo.description,
                'language': repo.language,
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'size_kb': repo.size,
                'created_at': repo.created_at.isoformat(),
                'updated_at': repo.updated_at.isoformat(),
                'has_wiki': repo.has_wiki,
                'has_pages': repo.has_pages,
                'open_issues': repo.open_issues_count,
                'default_branch': repo.default_branch,
                'topics': repo.get_topics(),
                'clone_url': repo.clone_url,
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {repo.full_name}: {e}")
            return {}
    
    def collect_repositories(
        self,
        target_count: int = 12,
        output_dir: Path = Path("data/raw")
    ) -> List[Dict]:
        """
        Collect repositories across all schema types
        
        Args:
            target_count: Total number of repositories to collect
            output_dir: Directory to clone repositories into
            
        Returns:
            List of collected repository metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        schema_types = ['openapi', 'validator', 'typedef']
        repos_per_type = target_count // len(schema_types)
        
        collected = []
        
        for schema_type in schema_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting {repos_per_type} repositories for: {schema_type}")
            logger.info(f"{'='*60}")
            
            repos = self.search_repositories(schema_type, max_results=repos_per_type * 2)
            
            for repo in repos[:repos_per_type]:
                clone_path = self.clone_repository(repo, output_dir)
                if clone_path:
                    metadata = self.get_repository_metadata(repo)
                    metadata['schema_type'] = schema_type
                    metadata['local_path'] = str(clone_path)
                    collected.append(metadata)
                
                if len([r for r in collected if r.get('schema_type') == schema_type]) >= repos_per_type:
                    break
        
        logger.info(f"\nSuccessfully collected {len(collected)} repositories")
        return collected


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    scraper = GitHubScraper()
    repos = scraper.collect_repositories(target_count=12)
    
    print(f"\nCollected {len(repos)} repositories:")
    for repo in repos:
        print(f"  - {repo['full_name']} ({repo['schema_type']})")

