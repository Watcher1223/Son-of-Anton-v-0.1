# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- Git
- GitHub personal access token

## Quick Start

### 1. Clone and Setup Environment

```bash
cd Son-of-Anton-v-0.1

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# Or on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GitHub Token

Create a GitHub personal access token:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `public_repo` (minimum)
4. Copy the generated token

Create a `.env` file in the project root:

```bash
echo "GITHUB_TOKEN=your_token_here" > .env
```

Replace `your_token_here` with your actual token.

### 3. Verify Installation

```bash
python -c "import github, git, pandas; print('✓ All dependencies installed')"
```

## Running the Pipeline

### Full Pipeline (Recommended)

Collect 12 repositories, extract features, and assemble dataset:

```bash
python main.py --mode full --target 12
```

### Individual Phases

#### Phase 1: Collect Repositories Only

```bash
python main.py --mode collect --target 12
```

This will:
- Search GitHub for Node.js repositories
- Clone them to `data/raw/`
- Save metadata to `data/intermediate/repo_metadata.json`

#### Phase 2: Extract Features Only

```bash
python main.py --mode extract --input data/raw/
```

This will:
- Detect schema types in repositories
- Find API endpoints
- Extract endpoint and repository features
- Save to `data/intermediate/*.json`

#### Phase 3: Assemble Dataset Only

```bash
python main.py --mode assemble --output data/final/dataset.csv
```

This will:
- Load intermediate JSON files
- Validate and deduplicate records
- Create final CSV dataset
- Generate train/test splits

## Configuration

Edit `config/config.yaml` to customize:

- GitHub search parameters
- Schema detection patterns
- Framework routing patterns
- Feature extraction settings
- Dataset assembly options

## Expected Output

After running the full pipeline, you should have:

```
data/
├── raw/                      # Cloned repositories
│   ├── org_repo1/
│   ├── org_repo2/
│   └── ...
├── intermediate/             # Extracted features (JSON)
│   ├── repo_metadata.json
│   ├── org_repo1_features.json
│   └── ...
└── final/                    # Final dataset
    ├── dataset.csv           # Complete dataset
    ├── train.csv             # Training split
    └── test.csv              # Test split
```

## Troubleshooting

### GitHub Rate Limiting

If you hit rate limits:
1. Increase `GITHUB_API_DELAY` in `.env`
2. Wait for rate limit reset (check: https://api.github.com/rate_limit)
3. Use authenticated token (higher limits)

### Missing Dependencies

If imports fail:
```bash
pip install --upgrade -r requirements.txt
```

### Permission Errors

On macOS/Linux, ensure main.py is executable:
```bash
chmod +x main.py
```

## Data Quality Checks

To verify your dataset quality:

```python
import pandas as pd

df = pd.read_csv('data/final/dataset.csv')

print(f"Total records: {len(df)}")
print(f"\nSchema classes:")
print(df['schema_class'].value_counts())
print(f"\nFrameworks:")
print(df['framework'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum().sum())
```

## Next Steps

After collecting data for Milestone 1:

1. **Exploratory Data Analysis**: Use notebooks in `notebooks/` to explore
2. **Feature Engineering**: Refine features based on initial results
3. **Model Development**: Move to Milestone 2 (model training)

## Support

For issues or questions:
- Check logs in `logs/pipeline.log`
- Review configuration in `config/config.yaml`
- Verify GitHub token has required permissions

