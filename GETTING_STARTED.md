# Getting Started - API Schema Extraction

## 🚀 Quick Start (5 minutes)

### Step 1: Setup Environment

```bash
# Run the quick start script
./quickstart.sh
```

Or manually:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure GitHub Token

1. Get a token: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scope: `public_repo`
   - Copy the token

2. Create `.env` file:
```bash
echo "GITHUB_TOKEN=ghp_your_token_here" > .env
```

### Step 3: Run the Pipeline

```bash
# Collect 12 repos and extract features (may take 30-60 minutes)
python main.py --mode full --target 12
```

That's it! The pipeline will:
1. ✓ Search and clone 12 Node.js repositories (4 per schema class)
2. ✓ Detect API endpoints (targeting 600+)
3. ✓ Extract endpoint and repository features
4. ✓ Assemble dataset with train/test splits

## 📊 Expected Output

```
data/
├── raw/                           # 12 cloned repositories
│   ├── expressjs_express/
│   ├── nestjs_nest/
│   └── ...
├── intermediate/                  # Feature JSON files
│   ├── repo_metadata.json
│   ├── expressjs_express_features.json
│   └── ...
└── final/                         # Final dataset
    ├── dataset.csv                # Complete dataset (600+ rows)
    ├── train.csv                  # Training set (80%)
    └── test.csv                   # Test set (20%)
```

## 🔍 Verify Your Data

```bash
# Check row count
wc -l data/final/dataset.csv

# View first few rows
head -n 5 data/final/dataset.csv

# Python analysis
python3 << EOF
import pandas as pd
df = pd.read_csv('data/final/dataset.csv')
print(f"Total endpoints: {len(df)}")
print(f"\nSchema classes:\n{df['schema_class'].value_counts()}")
print(f"\nFrameworks:\n{df['framework'].value_counts()}")
EOF
```

## 📈 Exploratory Analysis

```bash
# Start Jupyter notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

This notebook includes:
- Class distribution plots
- Framework analysis
- Feature correlations
- Data quality checks

## 🎯 Running Individual Phases

### Collect Repositories Only
```bash
python main.py --mode collect --target 12
```

Output: `data/raw/` with cloned repos

### Extract Features Only
```bash
python main.py --mode extract --input data/raw/
```

Output: `data/intermediate/*.json` with features

### Assemble Dataset Only
```bash
python main.py --mode assemble --output data/final/dataset.csv
```

Output: `data/final/dataset.csv` and train/test splits

## ⚙️ Customization

Edit `config/config.yaml` to adjust:

```yaml
github:
  min_stars: 10              # Minimum GitHub stars
  min_repo_size_kb: 100      # Minimum repo size
  
dataset:
  target_endpoints: 600      # Target endpoint count
  train_test_split: 0.8      # Train/test ratio
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_schema_detector.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐛 Troubleshooting

### GitHub Rate Limiting
```bash
# Check your rate limit
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/rate_limit

# Increase delay in .env
echo "GITHUB_API_DELAY=2.0" >> .env
```

### No Endpoints Found
- Check that repositories have actual API endpoints
- Verify framework detection is working
- Review logs in `logs/pipeline.log`

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python3 -c "import github, git, pandas; print('✓ OK')"
```

### Low Endpoint Count
- Increase `--target` (more repos)
- Adjust `min_repo_size_kb` in config (larger repos)
- Check `min_endpoints_per_repo` setting

## 📝 Log Files

All pipeline activity is logged to `logs/pipeline.log`:

```bash
# View recent activity
tail -f logs/pipeline.log

# Search for errors
grep ERROR logs/pipeline.log

# Count endpoints found
grep "Found.*endpoints" logs/pipeline.log
```

## 🎓 Milestone 1 Checklist

- [ ] Setup complete (environment, dependencies, token)
- [ ] Pipeline runs without errors
- [ ] Collected 12 repositories (4 per schema class)
- [ ] Extracted 600+ endpoints
- [ ] Dataset has all three schema classes
- [ ] Train/test splits created
- [ ] Data quality verified (no missing values, balanced classes)
- [ ] Exploratory analysis completed

## 🔜 Next: Milestone 2

Once you have your dataset:

1. **Feature Engineering**: Refine features based on EDA
2. **Model Selection**: Logistic Regression, XGBoost, Neural Networks
3. **Training**: Use `train.csv` for model training
4. **Evaluation**: Test on `test.csv`
5. **Metrics**: Accuracy, Precision, Recall, F1, Cross-validation

## 📚 Documentation

- `README.md` - Project overview and structure
- `SETUP.md` - Detailed setup instructions
- `PROJECT_SUMMARY.md` - Complete technical documentation
- `config/config.yaml` - All configuration options

## 💡 Tips

1. **Start small**: Test with `--target 3` first to verify everything works
2. **Monitor progress**: Watch `logs/pipeline.log` in real-time
3. **Save intermediate data**: Don't delete `data/intermediate/` - useful for debugging
4. **Check class balance**: Ensure roughly equal OpenAPI/Validator/TypeDef counts
5. **Quality over quantity**: Better to have 400 quality endpoints than 800 noisy ones

## ❓ Questions?

Check the documentation or review:
- Logs: `logs/pipeline.log`
- Config: `config/config.yaml`
- Code: `src/` modules with inline documentation

---

**Ready to start?** Run `./quickstart.sh` and then `python main.py --mode full --target 12`

Good luck! 🎉

