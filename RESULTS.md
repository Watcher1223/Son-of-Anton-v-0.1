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

