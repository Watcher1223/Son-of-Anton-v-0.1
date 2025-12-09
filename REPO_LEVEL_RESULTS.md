# Repository-Level Schema Classification Results

This document summarizes the findings from training models on **repository-level features only**, which can be obtained via the GitHub API without cloning repositories.

## Motivation

The endpoint-level models require:
1. Cloning repositories locally
2. Scanning file contents for patterns (decorators, imports, etc.)
3. Detecting and analyzing individual endpoints

This is expensive in terms of time and disk space. The repo-level approach aims to:
- **Use endpoint-level analysis only for labeling** (training time)
- **Train models on features obtainable via GitHub API** (3-5 API calls per repo)
- **Enable cheap inference at scale** without cloning

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Repos in curated CSV | 100 |
| Repos cloned | 100 |
| Repos skipped (validation) | 41 |
| Repos with "unknown" schema | 17 |
| **Repos with valid labels** | **37** |
| Training Set (80%) | 29 repos |
| Test Set (20%) | 8 repos |
| Number of Features | 42 |

### Why Only 37 Repos?

Out of 100 curated repositories:
- **41 skipped**: Missing package.json, no framework detected, too few files
- **17 unknown**: No schema signals detected (all scores = 0) - plain Express apps without validation
- **5 no endpoints**: Valid schema but 0 endpoints detected
- **37 final**: Valid schema + endpoints found

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| validator | 26 | 70.3% |
| typedef | 6 | 16.2% |
| openapi | 5 | 13.5% |

**Note:** The dataset is heavily imbalanced. All models use class weighting to address this.

---

## Feature Categories

### Features Used (42 total)

These features can be obtained via **GitHub API without cloning**:

#### From Git Tree API (1 API call)
- `has_openapi_spec`, `openapi_file_count`
- `has_validators`, `validator_file_count`
- `has_type_defs`, `typedef_file_count`
- `has_multiple_schema_types`
- `js_file_count`, `ts_file_count`, `jsx_file_count`, `tsx_file_count`
- `total_source_files`, `is_typescript`, `typescript_ratio`
- `total_files`, `total_lines_of_code`, `avg_lines_per_file`
- `has_docs_folder`, `has_api_docs`, `has_swagger`
- `has_readme`, `readme_length`

#### From package.json (Contents API - 1 call)
- `is_express`, `is_nestjs`, `is_fastify`
- `has_orm`, `orm_type` (→ one-hot: `orm_prisma`, `orm_mongoose`, `orm_typeorm`, `orm_sequelize`)
- `validator_libraries` (→ one-hot: `has_lib_joi`, `has_lib_zod`, `has_lib_class-validator`, etc.)
- `dependency_count`, `dev_dependency_count`, `total_dependency_count`
- `has_testing`, `has_linting`, `has_typescript`

### Features Excluded

These require **file content analysis** and are NOT used:

- Endpoint-level patterns: `has_body_decorator`, `has_joi_validate`, `has_zod_parse`
- Import analysis: `file_imports_class_validator`, `file_imports_swagger`, etc.
- Signal counts: `validator_signal_count`, `openapi_signal_count`, `typedef_signal_count`
- Endpoint density: `endpoint_count`, `endpoints_per_file`, `endpoints_per_1000_lines`
- All per-endpoint path/method features

---

## Model Results

### Performance Comparison (Single 80/20 Split)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|----------|-------------------|----------------|------------|
| Logistic Regression | 87.5% | 50.0% | 66.7% | 55.6% |
| XGBoost | 87.5% | 61.9% | 66.7% | 64.1% |
| Neural Network | **100%** | **100%** | **100%** | **100%** |

### 5-Fold Cross-Validation (More Reliable)

With only 37 repos, single split results are unreliable. Cross-validation provides better estimates:

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 81.1% | ± 6.5% |
| F1 (macro) | 66.7% | ± 16.2% |

**Per-fold results (LogReg):**
- Fold 1: Acc=75.0%, F1=50.0%
- Fold 2: Acc=87.5%, F1=85.9%
- Fold 3: Acc=71.4%, F1=48.9%
- Fold 4: Acc=85.7%, F1=63.6%
- Fold 5: Acc=85.7%, F1=85.2%

The high variance (±16% on F1) shows results depend heavily on which repos are in test set.

### Confusion Matrices

#### Logistic Regression
```
              Predicted
              openapi  typedef  validator
Actual
openapi          0        1         0
typedef          0        1         0
validator        0        0         6
```
- Misclassified 1 openapi repo as typedef

#### XGBoost
```
              Predicted
              openapi  typedef  validator
Actual
openapi          1        0         0
typedef          0        0         1
validator        0        0         6
```
- Correctly identified openapi
- Misclassified 1 typedef repo as validator

#### Neural Network
```
              Predicted
              openapi  typedef  validator
Actual
openapi          1        0         0
typedef          0        1         0
validator        0        0         6
```
- Perfect classification on test set

---

## Feature Importance Analysis

### Top 10 Features (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `has_openapi_spec` | 17.4% |
| 2 | `openapi_file_count` | 14.7% |
| 3 | `has_validators` | 14.6% |
| 4 | `validator_file_count` | 12.1% |
| 5 | `has_multiple_schema_types` | 8.3% |
| 6 | `dependency_count` | 4.7% |
| 7 | `total_lines_of_code` | 3.6% |
| 8 | `readme_length` | 3.6% |
| 9 | `ts_file_count` | 3.0% |
| 10 | `typescript_ratio` | 3.0% |

### Top 10 Features (Logistic Regression - by coefficient magnitude)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `has_openapi_spec` | 0.813 |
| 2 | `has_validators` | 0.599 |
| 3 | `has_multiple_schema_types` | 0.541 |
| 4 | `has_docs_folder` | 0.321 |
| 5 | `validator_file_count` | 0.298 |
| 6 | `has_swagger` | 0.291 |
| 7 | `has_api_docs` | 0.291 |
| 8 | `has_lib_joi` | 0.269 |
| 9 | `has_type_defs` | 0.251 |
| 10 | `is_typescript` | 0.248 |

### Key Insights

1. **Schema detection signals are most important**: `has_openapi_spec`, `has_validators`, and their file counts dominate
2. **Presence matters more than count**: Boolean flags often rank higher than raw counts
3. **Specific validator libraries help**: `has_lib_joi` is a strong signal for the validator class
4. **TypeScript usage correlates with typedef**: `is_typescript`, `typescript_ratio` help identify typedef repos

---

## Comparison: Repo-Level vs Endpoint-Level Models

| Metric | Endpoint-Level (LogReg) | Repo-Level (LogReg) |
|--------|------------------------|---------------------|
| Samples | 676 train / 140 test | 29 train / 8 test |
| Features | 110 | 42 |
| Accuracy | 87.1% | 87.5% |
| F1 (macro) | 49.9% | 55.6% |

**Observations:**
- Similar accuracy despite far fewer samples and features
- Repo-level has slightly better F1 due to better handling of minority classes
- Much smaller dataset makes repo-level models more prone to variance

---

## Limitations and Caveats

### Small Dataset Size
- Only **37 repositories** with valid labels out of 100 curated
- **17 repos (17%)** had no detectable schema patterns ("unknown")
- High variance in CV results: F1 = 66.7% ± 16.2%
- The 100% accuracy of the NN is due to overfitting on 8 test samples

### Why So Few Valid Repos?
Many real-world Express/Node.js repos don't use structured validation:
- No OpenAPI specs
- No Joi/Zod/class-validator
- Just plain `req.body` access with manual checks

These repos get `schema_class=unknown` and can't be used for training.

### Class Imbalance
- `validator` class dominates (70%)
- Only 5-6 repos for `openapi` and `typedef`
- Need more data collection for minority classes

### Recommendations
1. **Use cross-validation results** (81% accuracy, 67% F1) as the true estimate
2. **Prefer LogReg or XGBoost** over NN for small datasets
3. **Collect more repositories** with clear openapi/typedef patterns
4. **Consider manual labeling** of the 17 "unknown" repos if possible

---

## Files Generated

```
models/repo_level/
├── repo_logreg_model.pkl              # Trained LogReg model + preprocessor
├── repo_logreg_results.json           # Metrics and feature importance
├── repo_logreg_confusion_matrix.png   # Visualization
├── repo_xgboost_model.pkl             # Trained XGBoost model + preprocessor
├── repo_xgboost_results.json          # Metrics and feature importance
├── repo_xgboost_confusion_matrix.png  # Visualization
├── repo_xgboost_feature_importance.png # Feature importance plot
├── repo_nn_model.pt                   # PyTorch model weights
├── repo_nn_preprocessor.pkl           # Preprocessor for NN
├── repo_nn_results.json               # Metrics and training history
├── repo_nn_confusion_matrix.png       # Visualization
└── repo_nn_training_history.png       # Loss/accuracy curves

data/final/
├── repo_dataset.csv                   # Full repo-level dataset
├── repo_train.csv                     # Training split
└── repo_test.csv                      # Test split
```

---

## Usage

### Training
```bash
# Generate repo-level dataset from endpoint data
python -m src.dataset.repo_aggregator

# Train models
python train_repo_logreg.py
python train_repo_xgboost.py
python train_repo_nn.py
```

### Inference (example)
```python
import pickle

# Load model
with open('models/repo_level/repo_xgboost_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    preprocessor = data['preprocessor']

# Prepare features (from GitHub API)
repo_features = {
    'has_openapi_spec': True,
    'openapi_file_count': 2,
    'has_validators': False,
    # ... other features
}

# Predict
# (would need to transform features through preprocessor first)
```

---

## Conclusion

Repository-level models provide a viable alternative to endpoint-level classification:

- **Similar accuracy** (87.5%) with far fewer features (42 vs 110)
- **Cheap inference**: Only 3-5 GitHub API calls vs full clone
- **Strong feature signals**: `has_openapi_spec` and `has_validators` are highly predictive
- **Needs more data**: Current dataset is too small for reliable conclusions

The approach is promising for **screening repositories at scale** before deciding which ones warrant deeper (endpoint-level) analysis.

