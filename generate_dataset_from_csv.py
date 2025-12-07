#!/usr/bin/env python3
"""
Generate labeled dataset from existing repository CSV
"""

import pandas as pd
import sys
import logging
from pathlib import Path
from git import Repo

from src.detectors.schema_detector import SchemaDetector
from src.detectors.endpoint_detector import EndpointDetector
from src.extractors.endpoint_features import EndpointFeatureExtractor
from src.extractors.repo_features import RepositoryFeatureExtractor
from src.extractors.schema_linker import SchemaLinker
from src.dataset.assembler import DatasetAssembler
from src.scrapers.repo_filter import RepositoryFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clone_and_process_repos(csv_file: str, target_count: int = 12, output_dir: Path = Path("data/raw")):
    """Clone repositories from CSV and process them"""
    
    # Read CSV
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} repositories from {csv_file}")
    
    # Filter for repos with Express/NestJS/Fastify
    df = df[df['frameworks'].notna()]
    
    # Take top repositories by stars
    df = df.sort_values('stars', ascending=False).head(target_count * 3)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    schema_detector = SchemaDetector()
    endpoint_detector = EndpointDetector()
    endpoint_extractor = EndpointFeatureExtractor()
    repo_extractor = RepositoryFeatureExtractor()
    schema_linker = SchemaLinker()
    repo_filter = RepositoryFilter()
    
    all_records = []
    successful_repos = 0
    
    for idx, row in df.iterrows():
        if successful_repos >= target_count:
            break
        
        repo_url = row['repo_url']
        repo_name = row['repo_name'].replace('/', '_')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {row['full_name']} (stars: {row['stars']})")
        logger.info(f"{'='*70}")
        
        clone_path = output_dir / repo_name
        
        try:
            # Clone if not exists
            if not clone_path.exists():
                logger.info(f"Cloning {repo_url}...")
                Repo.clone_from(repo_url, clone_path, depth=1)
                logger.info("✓ Cloned successfully")
            else:
                logger.info("✓ Repository already exists")
            
            # Validate repository
            validation = repo_filter.validate_repository(clone_path, "unknown")
            if not validation['valid']:
                logger.warning(f"Skipping: {', '.join(validation['issues'])}")
                continue
            
            # Detect schema type
            schema_type, schema_details = schema_detector.classify_schema_type(clone_path)
            logger.info(f"Schema type: {schema_type}")
            
            # Detect endpoints
            framework = validation.get('framework')
            endpoints = endpoint_detector.detect_endpoints_in_repo(clone_path, framework)
            logger.info(f"Found {len(endpoints)} endpoints")
            
            if len(endpoints) == 0:
                logger.warning("No endpoints found, skipping")
                continue
            
            # Cap endpoints per repo to prevent large repos from dominating
            MAX_ENDPOINTS_PER_REPO = 50
            if len(endpoints) > MAX_ENDPOINTS_PER_REPO:
                logger.info(f"Capping endpoints from {len(endpoints)} to {MAX_ENDPOINTS_PER_REPO}")
                endpoints = endpoints[:MAX_ENDPOINTS_PER_REPO]
            
            # Extract repository features
            repo_features = repo_extractor.extract_all_features(clone_path, len(endpoints))
            
            # Extract features for each endpoint
            for endpoint in endpoints:
                try:
                    # Extract endpoint features
                    endpoint_features = endpoint_extractor.extract_all_features(endpoint, clone_path)
                    
                    # Link to schema
                    linked_record = schema_linker.link_endpoint_to_schema(
                        endpoint_features,
                        schema_type,
                        schema_details,
                        clone_path
                    )
                    
                    # Combine with repo features
                    full_record = {**linked_record, **repo_features}
                    all_records.append(full_record)
                    
                except Exception as e:
                    logger.error(f"Error processing endpoint: {e}")
            
            logger.info(f"✓ Extracted {len(endpoints)} endpoints")
            logger.info(f"Total records so far: {len(all_records)}")
            
            successful_repos += 1
            
            # Save intermediate results
            import json
            intermediate_path = Path(f"data/intermediate/{repo_name}_features.json")
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)
            with open(intermediate_path, 'w') as f:
                json.dump([{k: str(v) if isinstance(v, Path) else v for k, v in r.items()} 
                          for r in all_records[-len(endpoints):]], f, indent=2)
        
        except Exception as e:
            logger.error(f"Error processing {repo_name}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"Repositories processed: {successful_repos}")
    logger.info(f"Total endpoints extracted: {len(all_records)}")
    logger.info(f"{'='*70}\n")
    
    return all_records


def main():
    # Accept CSV filename as argument, default to curated_repos.csv if it exists
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    elif Path("curated_repos.csv").exists():
        csv_file = "curated_repos.csv"
        logger.info("Using curated_repos.csv for better class balance")
    else:
        csv_file = "github_repos_20251113_024339.csv"
    
    if not Path(csv_file).exists():
        logger.error(f"CSV file not found: {csv_file}")
        logger.error("Usage: python generate_dataset_from_csv.py [csv_file]")
        sys.exit(1)
    
    # Clone and extract features - increased target count for better dataset
    logger.info("Phase 1: Cloning repositories and extracting features...")
    logger.info(f"Using CSV: {csv_file}")
    records = clone_and_process_repos(csv_file, target_count=40)
    
    if len(records) == 0:
        logger.error("No records extracted!")
        sys.exit(1)
    
    # Assemble dataset
    logger.info("\nPhase 2: Assembling final dataset...")
    assembler = DatasetAssembler()
    
    output_path = Path("data/final/dataset.csv")
    df = assembler.assemble_dataset(records, output_path, output_format='csv')
    
    # Create train/test splits
    assembler.create_train_test_split(df, Path("data/final"), test_size=0.2)
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ DATASET GENERATION COMPLETE!")
    logger.info(f"{'='*70}")
    logger.info(f"\nDataset saved to: {output_path}")
    logger.info(f"Total endpoints: {len(df)}")
    logger.info(f"\nClass distribution:")
    for cls, count in df['schema_class'].value_counts().items():
        logger.info(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
    logger.info(f"\nFiles created:")
    logger.info(f"  - data/final/dataset.csv (complete dataset)")
    logger.info(f"  - data/final/train.csv (training set)")
    logger.info(f"  - data/final/test.csv (test set)")
    logger.info(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

