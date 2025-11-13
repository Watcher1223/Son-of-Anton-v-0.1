#!/usr/bin/env python3
"""
Verification script for API Schema Extraction project setup
"""

import sys
from pathlib import Path

def check_structure():
    """Verify project structure"""
    print("Checking project structure...")
    
    required_dirs = [
        'src/scrapers',
        'src/detectors', 
        'src/extractors',
        'src/utils',
        'src/dataset',
        'data/raw',
        'data/intermediate',
        'data/final',
        'config',
        'tests',
        'logs'
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} MISSING")
            return False
    
    return True

def check_files():
    """Verify key files exist"""
    print("\nChecking key files...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'config/config.yaml',
        'src/scrapers/github_scraper.py',
        'src/detectors/schema_detector.py',
        'src/detectors/endpoint_detector.py',
        'src/extractors/endpoint_features.py',
        'src/extractors/repo_features.py',
        'src/dataset/assembler.py',
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} MISSING")
            return False
    
    return True

def check_imports():
    """Verify imports work"""
    print("\nChecking module imports...")
    
    try:
        from src.scrapers import github_scraper
        print("  ✓ github_scraper")
    except Exception as e:
        print(f"  ✗ github_scraper: {e}")
        return False
    
    try:
        from src.detectors import schema_detector
        print("  ✓ schema_detector")
    except Exception as e:
        print(f"  ✗ schema_detector: {e}")
        return False
    
    try:
        from src.extractors import endpoint_features
        print("  ✓ endpoint_features")
    except Exception as e:
        print(f"  ✗ endpoint_features: {e}")
        return False
    
    return True

def check_env():
    """Check for .env configuration"""
    print("\nChecking configuration...")
    
    if Path('.env').exists():
        print("  ✓ .env file exists")
        return True
    else:
        print("  ⚠ .env file not found")
        print("    Create with: echo 'GITHUB_TOKEN=your_token' > .env")
        return False

def main():
    print("="*60)
    print("API Schema Extraction - Setup Verification")
    print("="*60)
    print()
    
    checks = [
        check_structure(),
        check_files(),
        check_imports(),
    ]
    
    env_ok = check_env()
    
    print()
    print("="*60)
    
    if all(checks):
        print("✓ All checks passed!")
        if not env_ok:
            print("⚠ Remember to configure .env with your GitHub token")
        print("\nReady to run: python main.py --mode full --target 12")
        return 0
    else:
        print("✗ Some checks failed")
        print("Please review the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
