#!/usr/bin/env python3
"""
Main Pipeline Script for API Schema Extraction - Milestone 1

This script orchestrates the full data collection and feature extraction pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
import yaml

from src.scrapers.github_scraper import GitHubScraper
from src.scrapers.repo_filter import RepositoryFilter
from src.detectors.schema_detector import SchemaDetector
from src.detectors.endpoint_detector import EndpointDetector
from src.extractors.endpoint_features import EndpointFeatureExtractor
from src.extractors.repo_features import RepositoryFeatureExtractor
from src.extractors.schema_linker import SchemaLinker
from src.dataset.assembler import DatasetAssembler


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: Path = Path("config/config.yaml")) -> Dict:
    """Load configuration from YAML file"""
    if not config_path.exists():
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def collect_repositories(config: Dict, target_count: int = 12) -> List[Dict]:
    """
    Collect GitHub repositories
    
    Args:
        config: Configuration dictionary
        target_count: Number of repositories to collect
        
    Returns:
        List of repository metadata
    """
    logging.info(f"\n{'='*70}")
    logging.info("PHASE 1: COLLECTING REPOSITORIES")
    logging.info(f"{'='*70}\n")
    
    scraper = GitHubScraper(config=config.get('github', {}))
    repos = scraper.collect_repositories(
        target_count=target_count,
        output_dir=Path("data/raw")
    )
    
    # Save repository metadata
    metadata_path = Path("data/intermediate/repo_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(repos, f, indent=2)
    
    logging.info(f"Saved repository metadata to: {metadata_path}")
    
    return repos


def extract_features(config: Dict, input_dir: Path = Path("data/raw")) -> List[Dict]:
    """
    Extract features from collected repositories
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing cloned repositories
        
    Returns:
        List of feature records
    """
    logging.info(f"\n{'='*70}")
    logging.info("PHASE 2: EXTRACTING FEATURES")
    logging.info(f"{'='*70}\n")
    
    # Initialize extractors
    schema_detector = SchemaDetector(config=config)
    endpoint_detector = EndpointDetector(config=config)
    endpoint_extractor = EndpointFeatureExtractor(config=config)
    repo_extractor = RepositoryFeatureExtractor(config=config)
    schema_linker = SchemaLinker(config=config)
    repo_filter = RepositoryFilter(config=config.get('dataset', {}))
    
    all_records = []
    
    # Process each repository
    repo_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    logging.info(f"Found {len(repo_dirs)} repositories to process")
    
    for repo_path in repo_dirs:
        logging.info(f"\n{'─'*70}")
        logging.info(f"Processing: {repo_path.name}")
        logging.info(f"{'─'*70}")
        
        try:
            # Validate repository
            validation = repo_filter.validate_repository(repo_path, "unknown")
            if not validation['valid']:
                logging.warning(f"Skipping invalid repository: {', '.join(validation['issues'])}")
                continue
            
            # Detect schema type
            schema_type, schema_details = schema_detector.classify_schema_type(repo_path)
            logging.info(f"Schema type: {schema_type}")
            
            # Detect framework
            framework = validation.get('framework')
            
            # Detect endpoints
            endpoints = endpoint_detector.detect_endpoints_in_repo(repo_path, framework)
            logging.info(f"Found {len(endpoints)} endpoints")
            
            if len(endpoints) == 0:
                logging.warning("No endpoints found, skipping repository")
                continue
            
            # Extract repository features
            repo_features = repo_extractor.extract_all_features(repo_path, len(endpoints))
            
            # Extract features for each endpoint
            for endpoint in endpoints:
                # Extract endpoint features
                endpoint_features = endpoint_extractor.extract_all_features(endpoint, repo_path)
                
                # Link to schema
                linked_record = schema_linker.link_endpoint_to_schema(
                    endpoint_features,
                    schema_type,
                    schema_details,
                    repo_path
                )
                
                # Combine with repo features
                full_record = {**linked_record, **repo_features}
                all_records.append(full_record)
            
            # Save intermediate results for this repo
            intermediate_path = Path(f"data/intermediate/{repo_path.name}_features.json")
            with open(intermediate_path, 'w') as f:
                json.dump([{k: str(v) if isinstance(v, Path) else v for k, v in r.items()} 
                          for r in all_records[-len(endpoints):]], f, indent=2)
            
            logging.info(f"Extracted {len(endpoints)} endpoint records")
            logging.info(f"Total records so far: {len(all_records)}")
        
        except Exception as e:
            logging.error(f"Error processing {repo_path.name}: {e}", exc_info=True)
            continue
    
    logging.info(f"\n{'='*70}")
    logging.info(f"FEATURE EXTRACTION COMPLETE")
    logging.info(f"Total records extracted: {len(all_records)}")
    logging.info(f"{'='*70}\n")
    
    return all_records


def assemble_dataset(config: Dict, output_path: Path = Path("data/final/dataset.csv")) -> None:
    """
    Assemble final dataset from intermediate data
    
    Args:
        config: Configuration dictionary
        output_path: Path to save final dataset
    """
    logging.info(f"\n{'='*70}")
    logging.info("PHASE 3: ASSEMBLING DATASET")
    logging.info(f"{'='*70}\n")
    
    assembler = DatasetAssembler(config=config.get('dataset', {}))
    
    # Load intermediate data
    records = assembler.load_intermediate_data(Path("data/intermediate"))
    
    if not records:
        logging.error("No records found to assemble!")
        return
    
    # Assemble and save dataset
    df = assembler.assemble_dataset(records, output_path, output_format='csv')
    
    # Create train/test split
    test_size = config.get('dataset', {}).get('train_test_split', 0.8)
    test_size = 1 - test_size  # Convert to test proportion
    assembler.create_train_test_split(df, output_path.parent, test_size=test_size)
    
    logging.info(f"\n{'='*70}")
    logging.info("DATASET ASSEMBLY COMPLETE")
    logging.info(f"Dataset saved to: {output_path}")
    logging.info(f"{'='*70}\n")


def run_full_pipeline(config: Dict, target_repos: int = 12) -> None:
    """
    Run the complete pipeline: collect, extract, assemble
    
    Args:
        config: Configuration dictionary
        target_repos: Number of repositories to collect
    """
    logging.info(f"\n{'#'*70}")
    logging.info("# API SCHEMA EXTRACTION - MILESTONE 1 PIPELINE")
    logging.info(f"# Target: {target_repos} repositories, 600+ endpoints")
    logging.info(f"{'#'*70}\n")
    
    # Phase 1: Collect repositories
    repos = collect_repositories(config, target_repos)
    
    # Phase 2: Extract features
    records = extract_features(config)
    
    # Phase 3: Assemble dataset
    assemble_dataset(config)
    
    logging.info(f"\n{'#'*70}")
    logging.info("# PIPELINE COMPLETE!")
    logging.info(f"{'#'*70}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="API Schema Extraction - Milestone 1 Pipeline"
    )
    
    parser.add_argument(
        '--mode',
        choices=['collect', 'extract', 'assemble', 'full'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--target',
        type=int,
        default=12,
        help='Target number of repositories to collect'
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/raw'),
        help='Input directory for extraction mode'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/final/dataset.csv'),
        help='Output path for dataset'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/config.yaml'),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    config = load_config(args.config)
    
    # Run appropriate mode
    try:
        if args.mode == 'collect':
            collect_repositories(config, args.target)
        
        elif args.mode == 'extract':
            extract_features(config, args.input)
        
        elif args.mode == 'assemble':
            assemble_dataset(config, args.output)
        
        elif args.mode == 'full':
            run_full_pipeline(config, args.target)
        
        logging.info("\n✓ Success!")
        
    except KeyboardInterrupt:
        logging.warning("\n\nInterrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logging.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

