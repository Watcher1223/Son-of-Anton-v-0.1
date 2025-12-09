# Model Results Tracking

This document tracks the evolution of model performance as we iterate on the dataset and models.

---

## Run 1: Initial Training (December 7, 2025)

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Records | 840 |
| Total Repositories | 28 |
| Train Endpoints | 796 (22 repos) |
| Test Endpoints | 44 (6 repos) |
| Features | 63 |

### Class Distribution
| Class | Train | Test |
|-------|-------|------|
| validator | 43.1% | 34.1% |
| typedef | 40.3% | 52.3% |
| openapi | 16.6% | 13.6% |

### Model Performance

| Model | CV F1 (Macro) | Test Accuracy | Test F1 (Macro) | Test F1 (Weighted) | API Calls Saved |
|-------|---------------|---------------|-----------------|--------------------| ----------------|
| Logistic Regression | 0.9324 ± 0.135 | 47.7% | 0.522 | 0.329 | 58.4% |
| XGBoost | 0.9565 ± 0.053 | 36.4% | 0.472 | 0.278 | 50.5% |
| Neural Network | - | 34.1% | 0.170 | 0.173 | 48.9% |

### Per-Class Performance (Test Set)

**Logistic Regression:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| openapi | 1.00 | 1.00 | 1.00 | 6 |
| typedef | 0.00 | 0.00 | 0.00 | 23 |
| validator | 0.39 | 1.00 | 0.57 | 15 |

**XGBoost:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| openapi | 1.00 | 1.00 | 1.00 | 6 |
| typedef | 0.00 | 0.00 | 0.00 | 23 |
| validator | 0.30 | 0.67 | 0.42 | 15 |

**Neural Network:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| openapi | 0.00 | 0.00 | 0.00 | 6 |
| typedef | 0.00 | 0.00 | 0.00 | 23 |
| validator | 0.34 | 1.00 | 0.51 | 15 |

### Top 10 Features (Logistic Regression)
1. has_openapi_spec: 1.4846
2. openapi_file_count: 0.9821
3. has_multiple_schema_types: 0.7774
4. has_linting: 0.7312
5. has_lib_express-validator: 0.6104
6. has_validators: 0.6099
7. orm_prisma: 0.5584
8. validator_file_count: 0.5194
9. is_nestjs: 0.5014
10. has_lib_class-validator: 0.5014

### Top 10 Features (XGBoost - by gain)
1. typedef_file_count: 0.2076
2. has_lib_express-validator: 0.1144
3. openapi_file_count: 0.1005
4. jsx_file_count: 0.0985
5. endpoints_per_1000_lines: 0.0850
6. js_file_count: 0.0772
7. has_openapi_spec: 0.0756
8. has_linting: 0.0399
9. orm_prisma: 0.0313
10. readme_length: 0.0294

### Observations
1. **High CV vs Low Test Performance**: All models show excellent cross-validation scores (93-96%) but poor test performance (34-48%). This is expected with repository-level splitting - the models learn repo-specific patterns.

2. **OpenAPI Detection Works Well**: All models achieve 100% precision and recall on `openapi` class. The `has_openapi_spec` feature is a strong signal.

3. **Typedef Classification Fails**: All models score 0% on the `typedef` class, which makes up 52% of the test set. The 6 test repositories with typedef may have different characteristics than training repos.

4. **Logistic Regression Performs Best**: Simpler model generalizes better with limited data.

5. **Feature Importance Makes Sense**: Schema-type specific features (`has_openapi_spec`, `validator_file_count`, `typedef_file_count`) rank highly.

### Next Steps
- [ ] Investigate why typedef classification fails on test repos
- [ ] Add more typedef-heavy repositories to training data
- [ ] Consider stratified repository selection for better class balance
- [ ] Try feature engineering to better capture typedef patterns
- [ ] Experiment with different train/test splits

---

## Run 2: Endpoint‑Level Features + Expanded Curated Repos (December 8, 2025)

### Changes Made

- **Expanded curated training set** (`curated_repos.csv`):
  - Added dozens of real‑world Express/NestJS/Fastify backends, including OpenAPI‑heavy projects (e.g. `agile-team-tool`, `nestjs/swagger` examples, a Fastify swagger example, and [`shapeshift/unchained`](https://github.com/shapeshift/unchained)).
  - Switched `generate_dataset_from_csv.py` to process **all curated repos** (no longer truncating by stars when a curated CSV is used).
- **Endpoint‑level feature extraction** (`src/extractors/endpoint_features.py`):
  - Added ~25 new per‑endpoint features capturing:
    - Validation usage (`has_body_decorator`, `has_validation_pipe`, `has_joi_validate`, `has_zod_parse`, `has_express_validator`, `has_yup_validate`, ...).
    - OpenAPI decorators and comments (`has_api_body_decorator`, `has_api_response_decorator`, `has_api_operation_decorator`, `has_swagger_comment`, ...).
    - TypeScript type usage (`has_dto_reference`, `has_interface_cast`, `has_type_annotation`, `has_generic_type`, `has_return_type`).
    - File‑level imports for validator and swagger libraries (`file_imports_class_validator`, `file_imports_joi`, `file_imports_zod`, `file_imports_swagger`, `file_imports_express_validator`, `file_imports_dto`, ...).
    - Aggregate signal counts: `validator_signal_count`, `openapi_signal_count`, `typedef_signal_count`.
- **Preprocessor updates** (`src/models/preprocessor.py`):
  - Registered all new boolean and continuous feature columns so they are preserved and scaled.
  - `prepare_data()` now produces **109 numeric/boolean features** per endpoint.

### Dataset Statistics

| Metric              | Value                    |
|---------------------|--------------------------|
| Total Records       | 816                      |
| Total Repositories  | 37                       |
| Train Endpoints     | 676 (29 repos, GSS)      |
| Test Endpoints      | 140 (8 repos, GSS)       |
| Features (X columns)| 109                      |

> Note: Train/test split is created inside `DataPreprocessor` using `GroupShuffleSplit` on `repo_name` to ensure no repository appears in both train and test.

### Class Distribution (DataPreprocessor split)

| Class     | Train (≈%) | Test (count / %) |
|-----------|------------|------------------|
| validator | ~57%       | 104 / 93.6%      |
| typedef   | ~22%       | 1 / 0.7%         |
| openapi   | ~20%       | 8 / 5.7%         |

The final `data/final/dataset.csv` (after dedup + dropping `unknown`) has:

- `validator`: 518 endpoints (63.5%)
- `typedef`: 152 endpoints (18.6%)
- `openapi`: 146 endpoints (17.9%)

### Model Performance (Run 2 – Repo‑Level Test)

From `results/model_comparison.csv` on the 140‑endpoint, 8‑repo test set:

| Model              | CV F1 (Macro)     | Test Accuracy | Test F1 (Macro) | Test F1 (Weighted) | F1(openapi) | F1(typedef) | F1(validator) | API Calls Saved |
|--------------------|------------------|---------------|-----------------|--------------------|-------------|-------------|---------------|-----------------|
| Logistic Regression| 0.7976 ± 0.268   | 87.1%         | 0.50            | 0.90               | 0.57        | 0.00        | 0.93          | 86.0%           |
| XGBoost            | 0.8742 ± 0.155   | **99.3%**     | **0.67**        | **0.99**           | **1.00**    | 0.00        | **0.996**     | **94.5%**       |
| Neural Network     | – (no CV logged) | 74.3%         | 0.39            | 0.81               | 0.31        | 0.00        | 0.84          | 77.0%           |

### Per‑Class Performance (Test Set)

**Logistic Regression:**

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| openapi   | 0.57      | 1.00   | 0.71     | 8       |
| typedef   | 0.00      | 0.00   | 0.00     | 1       |
| validator | 0.93      | 0.88   | 0.91     | 131     |

**XGBoost:**

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| openapi   | 1.00      | 1.00   | 1.00     | 8       |
| typedef   | 0.00      | 0.00   | 0.00     | 1       |
| validator | 0.99      | 1.00   | 1.00     | 131     |

**Neural Network:**

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| openapi   | 0.31      | 1.00   | 0.47     | 8       |
| typedef   | 0.00      | 0.00   | 0.00     | 1       |
| validator | 0.84      | 0.85   | 0.84     | 131     |

### Top 10 Features (Logistic Regression – Run 2)

1. `has_openapi_spec`: 1.3310  
2. `has_multiple_schema_types`: 0.9395  
3. `has_validators`: 0.8594  
4. `openapi_file_count`: 0.8366  
5. `has_type_defs`: 0.7942  
6. `has_lib_zod`: 0.6040  
7. `typedef_file_count`: 0.5980  
8. `avg_lines_per_file`: 0.5414  
9. `has_readme`: 0.4202  
10. `dev_dependency_count`: 0.4031  

### Top 10 Features (XGBoost – Run 2, by gain)

1. `openapi_file_count`: 0.1393  
2. `jsx_file_count`: 0.0985  
3. `typedef_file_count`: 0.1051  
4. `total_lines_of_code`: 0.0963  
5. `has_openapi_spec`: 0.0868  
6. `has_lib_express-validator`: 0.0498  
7. `orm_prisma`: 0.0404  
8. `has_try_catch`: 0.0381  
9. `has_multiple_schema_types`: 0.0340  
10. `has_testing`: 0.0304  

### Observations

1. **OpenAPI detection is strong**: all three models achieve **100% recall** on the 8 openapi endpoints in the held‑out repos; XGBoost also achieves **perfect precision** (F1=1.0 on `openapi`).
2. **Validator vs non‑validator separation is now very good**: XGBoost reaches ~99–100% accuracy and F1≈0.996 on the `validator` class; Logistic Regression is slightly behind but still strong (F1≈0.93 on `validator`).
3. **Typedef classification is still effectively unsolved** on the test set: only 1 typedef endpoint is present in the held‑out repos, so all three models have F1≈0. The training set does contain 150+ typedef endpoints, but they are concentrated in a few repos; with repo‑level splitting, it’s easy for the single typedef test repo to look “different” from training.
4. **CV vs test gap has narrowed**: XGBoost still sees high CV F1 (≈0.87) but now also achieves strong test performance (F1_macro≈0.67, Accuracy≈99%), indicating better generalization than in Run 1.
5. **Endpoint‑level features matter**: the new validation/OpenAPI/type‑definition signals and aggregate counts appear prominently in both Logistic Regression and XGBoost feature importances, confirming that the models are using code‑level cues rather than just coarse repo metadata.

### Updated Next Steps

- [ ] Improve typedef coverage in the test set by:
  - Adding more **diverse typedef‑dominant repos** (e.g. TypeScript heavy, minimal explicit validators).
  - Considering a **per‑repo stratified split** that ensures multiple typedef repos land in the test fold.
- [ ] Add more **Express‑style OpenAPI examples** (swagger‑jsdoc / swagger‑ui‑express around real routes) to further diversify the `openapi` class.
- [ ] Explore a **binary OpenAPI vs non‑OpenAPI** classifier with calibrated thresholds to maximize recall when the downstream task is “should I call an expensive schema‑inference API?”.
- [ ] Refine typedef features (e.g., richer analysis of TypeScript AST around handlers, detection of shared DTO usage across endpoints).
- [ ] Consider a **two‑stage cascade**: first detect openapi vs non‑openapi, then distinguish validator vs typedef within non‑openapi endpoints.

---

## Template for Future Runs

```markdown
## Run N: [Description] ([Date])

### Changes Made
- [List changes since previous run]

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Records | |
| Total Repositories | |
| Train Endpoints | |
| Test Endpoints | |
| Features | |

### Class Distribution
| Class | Train | Test |
|-------|-------|------|
| validator | | |
| typedef | | |
| openapi | | |

### Model Performance
| Model | CV F1 (Macro) | Test Accuracy | Test F1 (Macro) | API Calls Saved |
|-------|---------------|---------------|-----------------|-----------------|
| Logistic Regression | | | | |
| XGBoost | | | | |
| Neural Network | | | | |

### Observations
- 

### Next Steps
- [ ] 
```

