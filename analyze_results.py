#!/usr/bin/env python3
"""
Enhanced Results Analyzer for GitHub Repository Finder
Analyzes the enhanced CSV output with endpoint and schema file information
"""

import pandas as pd
import sys
from pathlib import Path
import json


def analyze_csv(csv_file: str):
    """Analyze the enhanced repository CSV file"""
    
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        print("\nPlease run find_repos.py first to generate the CSV file")
        sys.exit(1)
    
    # Load the CSV
    df = pd.read_csv(csv_file)
    
    print("=" * 80)
    print(f"Enhanced Analysis of {csv_file}")
    print("=" * 80)
    
    # Basic statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total repositories: {len(df)}")
    print(f"   Average stars: {df['stars'].mean():.1f}")
    print(f"   Median stars: {df['stars'].median():.0f}")
    print(f"   Max stars: {df['stars'].max()}")
    print(f"   Average size: {df['size_kb'].mean():.0f} KB")
    print(f"   Average quality score: {df['quality_score'].mean():.1f}/100")
    
    # Quality score distribution
    print(f"\n⭐ QUALITY SCORE DISTRIBUTION")
    high_quality = df[df['quality_score'] >= 80]
    medium_quality = df[(df['quality_score'] >= 50) & (df['quality_score'] < 80)]
    low_quality = df[df['quality_score'] < 50]
    print(f"   High (80-100):   {len(high_quality):4d} repos ({len(high_quality)/len(df)*100:.1f}%)")
    print(f"   Medium (50-79):  {len(medium_quality):4d} repos ({len(medium_quality)/len(df)*100:.1f}%)")
    print(f"   Low (0-49):      {len(low_quality):4d} repos ({len(low_quality)/len(df)*100:.1f}%)")
    
    # Endpoint and schema file statistics
    print(f"\n📁 FILE DISCOVERY STATISTICS")
    print(f"   Total endpoint files found: {df['endpoint_file_count'].sum()}")
    print(f"   Avg endpoint files per repo: {df['endpoint_file_count'].mean():.1f}")
    print(f"   Repos with >5 endpoint files: {(df['endpoint_file_count'] > 5).sum()}")
    print(f"   Total schema files found: {df['schema_file_count'].sum()}")
    print(f"   Avg schema files per repo: {df['schema_file_count'].mean():.1f}")
    print(f"   Repos with >3 schema files: {(df['schema_file_count'] > 3).sum()}")
    
    # Framework breakdown
    print(f"\n🔧 BY FRAMEWORK")
    for framework in ['Express', 'NestJS', 'Fastify']:
        count = df['frameworks'].str.contains(framework, na=False).sum()
        pct = (count / len(df) * 100)
        verified = df[df['frameworks'].str.contains(framework, na=False)]['framework_in_package'].sum()
        print(f"   {framework:10s}: {count:4d} repos ({pct:.1f}%) - {verified} verified in package.json")
    
    # Language breakdown
    print(f"\n💻 BY LANGUAGE")
    lang_counts = df['language'].value_counts().head(5)
    for lang, count in lang_counts.items():
        pct = (count / len(df) * 100)
        print(f"   {lang:15s}: {count:4d} repos ({pct:.1f}%)")
    
    # Schema indicators
    print(f"\n📋 SCHEMA INDICATORS")
    print(f"   OpenAPI specs:      {df['has_openapi'].sum():4d} repos ({df['has_openapi'].sum()/len(df)*100:.1f}%)")
    print(f"   Validator libraries: {df['has_validators'].sum():4d} repos ({df['has_validators'].sum()/len(df)*100:.1f}%)")
    print(f"   TypeScript config:   {df['has_typescript'].sum():4d} repos ({df['has_typescript'].sum()/len(df)*100:.1f}%)")
    print(f"   API docs folder:     {df['has_api_docs_folder'].sum():4d} repos ({df['has_api_docs_folder'].sum()/len(df)*100:.1f}%)")
    print(f"   Test files:          {df['has_tests'].sum():4d} repos ({df['has_tests'].sum()/len(df)*100:.1f}%)")
    
    # Schema types breakdown
    print(f"\n🎯 SCHEMA TYPES DETECTED")
    all_schema_types = []
    for types in df['schema_types'].dropna():
        if types:
            all_schema_types.extend(types.split(','))
    
    if all_schema_types:
        from collections import Counter
        type_counts = Counter(all_schema_types)
        for schema_type, count in type_counts.most_common():
            print(f"   {schema_type:15s}: {count:4d} repos")
    
    # Top validator libraries
    print(f"\n🛡️  TOP VALIDATOR LIBRARIES")
    all_validators = []
    for libs in df['validator_libraries'].dropna():
        if libs:
            all_validators.extend(libs.split(','))
    
    if all_validators:
        from collections import Counter
        validator_counts = Counter(all_validators)
        for lib, count in validator_counts.most_common(10):
            print(f"   {lib:15s}: {count:4d} repos")
    else:
        print("   No validator libraries found")
    
    # High-value repositories (multiple schema indicators)
    print(f"\n⭐ HIGH-VALUE REPOSITORIES")
    high_value = df[
        (df['quality_score'] >= 70) & 
        (df['endpoint_file_count'] > 3)
    ]
    print(f"   High quality + multiple endpoints: {len(high_value)}")
    
    if len(high_value) > 0:
        print(f"\n   Top 10 by quality score:")
        for idx, row in high_value.nlargest(10, 'quality_score').iterrows():
            print(f"     🏆 Score {row['quality_score']:3.0f} - {row['full_name']:45s} ⭐{row['stars']:5d} [{row['frameworks']}]")
            print(f"        Endpoints: {row['endpoint_file_count']:2d} files | Schemas: {row['schema_file_count']:2d} files")
    
    # Repos with complete schema coverage
    complete_coverage = df[
        (df['has_openapi'] == True) & 
        (df['has_validators'] == True) & 
        (df['has_typescript'] == True)
    ]
    print(f"\n   Repos with COMPLETE schema coverage (OpenAPI + Validators + TypeScript): {len(complete_coverage)}")
    
    # ML Training Data Recommendations
    print(f"\n🎯 RECOMMENDATIONS FOR ML TRAINING DATA")
    
    # Best repos for each schema class
    openapi_repos = df[df['has_openapi'] == True].nlargest(5, 'quality_score')
    validator_repos = df[df['has_validators'] == True].nlargest(5, 'quality_score')
    typescript_repos = df[df['has_typescript'] == True].nlargest(5, 'quality_score')
    
    print(f"\n   1. OPENAPI SCHEMA CLASS - Top 5 candidates:")
    for _, row in openapi_repos.iterrows():
        print(f"      • {row['full_name']:45s} (Score: {row['quality_score']:.0f}, ⭐{row['stars']:4d}, {row['endpoint_file_count']} endpoints)")
    
    print(f"\n   2. VALIDATOR SCHEMA CLASS - Top 5 candidates:")
    for _, row in validator_repos.iterrows():
        libs = row['validator_libraries'] if row['validator_libraries'] else 'unknown'
        print(f"      • {row['full_name']:45s} (Score: {row['quality_score']:.0f}, ⭐{row['stars']:4d}, libs: {libs})")
    
    print(f"\n   3. TYPESCRIPT TYPEDEF CLASS - Top 5 candidates:")
    for _, row in typescript_repos.iterrows():
        print(f"      • {row['full_name']:45s} (Score: {row['quality_score']:.0f}, ⭐{row['stars']:4d}, {row['schema_file_count']} schema files)")
    
    # Save filtered datasets
    output_dir = Path("filtered_repos")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 SAVING FILTERED DATASETS to {output_dir}/")
    
    # High-quality repos
    high_quality_file = output_dir / "high_quality_repos.csv"
    high_quality.to_csv(high_quality_file, index=False)
    print(f"   ✓ High-quality repos (score ≥80): {high_quality_file} ({len(high_quality)} repos)")
    
    # OpenAPI class
    openapi_class_file = output_dir / "openapi_class_repos.csv"
    df[df['has_openapi'] == True].to_csv(openapi_class_file, index=False)
    print(f"   ✓ OpenAPI schema class: {openapi_class_file} ({df['has_openapi'].sum()} repos)")
    
    # Validator class
    validator_class_file = output_dir / "validator_class_repos.csv"
    df[df['has_validators'] == True].to_csv(validator_class_file, index=False)
    print(f"   ✓ Validator schema class: {validator_class_file} ({df['has_validators'].sum()} repos)")
    
    # TypeScript class
    typescript_class_file = output_dir / "typescript_class_repos.csv"
    df[df['has_typescript'] == True].to_csv(typescript_class_file, index=False)
    print(f"   ✓ TypeScript typedef class: {typescript_class_file} ({df['has_typescript'].sum()} repos)")
    
    # By framework
    for framework in ['Express', 'NestJS', 'Fastify']:
        fw_df = df[df['frameworks'].str.contains(framework, na=False)]
        fw_file = output_dir / f"{framework.lower()}_repos.csv"
        fw_df.to_csv(fw_file, index=False)
        print(f"   ✓ {framework} repos: {fw_file} ({len(fw_df)} repos)")
    
    # Repos with many endpoint files (good for extraction)
    rich_endpoints = df[df['endpoint_file_count'] > 5]
    rich_file = output_dir / "endpoint_rich_repos.csv"
    rich_endpoints.to_csv(rich_file, index=False)
    print(f"   ✓ Endpoint-rich repos (>5 files): {rich_file} ({len(rich_endpoints)} repos)")
    
    # Export endpoint and schema file mapping (JSON for programmatic use)
    file_mapping = {}
    for _, row in df.iterrows():
        if row['endpoint_file_count'] > 0 or row['schema_file_count'] > 0:
            file_mapping[row['full_name']] = {
                'repo_url': row['repo_url'],
                'api_url': row['api_url'],
                'endpoint_files': row['endpoint_files'].split(' | ') if pd.notna(row['endpoint_files']) and row['endpoint_files'] else [],
                'schema_files': row['schema_files'].split(' | ') if pd.notna(row['schema_files']) and row['schema_files'] else [],
                'quality_score': row['quality_score'],
                'frameworks': row['frameworks'],
            }
    
    mapping_file = output_dir / "file_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(file_mapping, f, indent=2)
    print(f"   ✓ Endpoint/schema file mapping: {mapping_file}")
    
    print(f"\n✅ Analysis complete!")
    print("=" * 80)
    
    # Next steps guidance
    print(f"\n📖 NEXT STEPS:")
    print(f"   1. Review filtered CSVs in '{output_dir}/' directory")
    print(f"   2. Use 'file_mapping.json' to programmatically access endpoint and schema files")
    print(f"   3. Clone high-quality repos for deeper analysis")
    print(f"   4. Use the GitHub API URLs to fetch specific files for your ML dataset")
    print(f"\n   Example: Fetch endpoint file from repo using 'api_url' + '/contents/' + endpoint_file_path")


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Find the most recent github_repos CSV file
        csv_files = list(Path('.').glob('github_repos_*.csv'))
        if not csv_files:
            print("Error: No github_repos_*.csv files found")
            print("\nUsage: python analyze_results.py [csv_file]")
            print("   or: python analyze_results.py")
            print("       (will use the most recent github_repos_*.csv file)")
            sys.exit(1)
        
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent CSV file: {csv_file}\n")
    
    analyze_csv(str(csv_file))


if __name__ == '__main__':
    main()

