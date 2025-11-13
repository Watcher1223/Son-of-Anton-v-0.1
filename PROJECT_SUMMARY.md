# Project Summary: API Schema Extraction - Milestone 1

## Overview

This Python project implements the data collection and feature engineering pipeline for **Milestone 1** of the Machine Learning final project on API schema extraction prediction.

## Project Goal

Predict which schema extraction strategy (OpenAPI, Validator-based, or TypeScript typedef) will be most successful for a given API endpoint, reducing extraction attempts from 15-30 API calls to ≤5 calls per endpoint.

## Project Structure

```
Son-of-Anton-v-0.1/
├── src/
│   ├── scrapers/
│   │   ├── github_scraper.py      # GitHub API integration & repository cloning
│   │   └── repo_filter.py         # Repository validation & quality checks
│   ├── detectors/
│   │   ├── schema_detector.py     # Schema type detection (OpenAPI/Validator/TypeDef)
│   │   └── endpoint_detector.py   # API endpoint discovery (Express/NestJS/Fastify)
│   ├── extractors/
│   │   ├── endpoint_features.py   # Endpoint-level feature extraction
│   │   ├── repo_features.py       # Repository-level feature extraction
│   │   └── schema_linker.py       # Link endpoints to schema sources
│   ├── utils/
│   │   ├── file_parser.py         # Parse JS/TS files
│   │   └── framework_detector.py  # Detect web frameworks
│   └── dataset/
│       └── assembler.py           # Assemble final dataset & train/test splits
├── data/
│   ├── raw/                       # Cloned repositories
│   ├── intermediate/              # Extracted features (JSON)
│   └── final/                     # Final CSV dataset
├── config/
│   └── config.yaml                # Configuration settings
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks for EDA
├── main.py                        # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── SETUP.md                       # Setup instructions
```

## Key Components

### 1. GitHub Scraper (`src/scrapers/github_scraper.py`)
- Searches GitHub for Node.js repositories with schema definitions
- Filters by language, size, stars, and schema indicators
- Clones repositories locally for analysis
- Extracts repository metadata

**Key Features:**
- Rate limiting handling
- Support for all three schema classes
- Shallow cloning for efficiency

### 2. Schema Detector (`src/detectors/schema_detector.py`)
- Detects OpenAPI/Swagger specification files
- Identifies validator libraries (Joi, Zod, class-validator, etc.)
- Finds TypeScript type definition files
- Classifies primary schema type using heuristics

**Detection Patterns:**
- OpenAPI: `openapi.yaml`, `swagger.json`, etc.
- Validators: `*.validator.{js,ts}`, validator directories
- TypeDefs: `*.dto.ts`, `*.types.ts`, `*.interface.ts`

### 3. Endpoint Detector (`src/detectors/endpoint_detector.py`)
- Detects API endpoints across multiple frameworks
- Regex-based pattern matching for route definitions
- Supports Express.js, NestJS (decorators), and Fastify

**Detected Information:**
- HTTP method (GET, POST, PUT, DELETE, etc.)
- Endpoint path
- Source file and line number
- Framework used

### 4. Feature Extractors

#### Endpoint Features (`src/extractors/endpoint_features.py`)
- **Path features:** depth, parameters, length, versioning
- **Method features:** HTTP method type, safety, idempotency
- **Code context:** middleware, auth, validation, async, error handling
- **Parameters:** query, body, path params, file uploads

#### Repository Features (`src/extractors/repo_features.py`)
- **Language:** JS/TS ratio, file counts
- **Framework:** Express, NestJS, Fastify, ORM type
- **Size:** LOC, file count, average file size
- **Dependencies:** count, testing/linting presence
- **Documentation:** README, docs folder, API documentation
- **Endpoint density:** endpoints per file/line

#### Schema Linking (`src/extractors/schema_linker.py`)
- Links endpoints to related schema files
- Extracts schema source features
- Detects database models, comments, documentation

### 5. Dataset Assembler (`src/dataset/assembler.py`)
- Combines all feature types into unified dataset
- Validates data quality and completeness
- Deduplicates endpoints
- Checks class balance
- Creates train/test splits
- Exports to CSV format

## Features Extracted

### Endpoint-Level (30+ features)
- HTTP method, path depth, parameter counts
- Middleware presence, authentication indicators
- Validation presence, async/await usage
- Response types, error handling
- Function length, code complexity

### Repository-Level (25+ features)
- Primary language, TypeScript ratio
- Framework type, ORM presence
- Total files, lines of code
- Dependency counts, testing/linting
- Documentation presence, README length
- Endpoint density metrics

### Schema-Source (15+ features)
- Schema class label (OpenAPI/Validator/TypeDef)
- Presence of each schema type
- File counts per type
- Validator libraries used
- Related schema file count
- Database models, comments

## Usage

### Quick Start
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure GitHub token
echo "GITHUB_TOKEN=your_token" > .env

# Run full pipeline
python main.py --mode full --target 12
```

### Individual Phases
```bash
# Collect repositories only
python main.py --mode collect --target 12

# Extract features only
python main.py --mode extract --input data/raw/

# Assemble dataset only
python main.py --mode assemble --output data/final/dataset.csv
```

## Configuration

Edit `config/config.yaml` to customize:
- GitHub search parameters (languages, stars, size)
- Schema detection patterns (file names, directories)
- Framework routing patterns
- Feature extraction settings
- Dataset assembly options

## Output Format

**Final Dataset:** `data/final/dataset.csv`

Each row represents one endpoint with:
- Identifier columns: `endpoint_id`, `repo_name`, `schema_class`
- Endpoint features: method, path, parameters, middleware, etc.
- Repository features: framework, size, dependencies, etc.
- Schema features: schema types present, validator libraries, etc.

**Train/Test Splits:**
- `data/final/train.csv` (80% of data)
- `data/final/test.csv` (20% of data)
- Stratified by `schema_class` for balanced splits

## Data Quality Checks

The pipeline includes:
- ✓ Repository validation (package.json, framework, minimum size)
- ✓ Endpoint deduplication
- ✓ Feature completeness validation
- ✓ Class balance reporting
- ✓ Missing value detection

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Current test coverage:
- Schema detector initialization and detection logic
- Endpoint feature extraction (path, method, parameters)
- Basic edge cases and error handling

## Exploratory Analysis

Use the provided Jupyter notebook:
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Includes visualizations for:
- Schema class distribution
- Framework distribution
- HTTP method distribution
- Feature correlations
- Missing values
- Repository statistics

## Milestone 1 Deliverables

✅ **Goal:** Collect 12 repositories, extract 600+ endpoints

**Implemented:**
1. GitHub repository scraper with filtering
2. Schema type detection system
3. Multi-framework endpoint detector
4. Comprehensive feature extraction pipeline
5. Dataset assembly with quality checks
6. Train/test split generation
7. Documentation and setup instructions
8. Unit tests and example notebooks

## Next Steps (Milestone 2)

1. **Run the pipeline** to collect actual data
2. **Exploratory analysis** on the collected dataset
3. **Feature engineering refinement** based on initial results
4. **Model development:**
   - Logistic Regression baseline
   - XGBoost classifier
   - Neural network
5. **Evaluation:** accuracy, precision, recall, F1, cross-validation
6. **Feature importance analysis:** SHAP values, permutation importance

## Technology Stack

- **Python 3.8+**
- **Data Collection:** PyGithub, GitPython
- **Data Processing:** pandas, numpy
- **Code Analysis:** regex, esprima (JS parser)
- **ML Preparation:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Development:** pytest, jupyter, black, flake8

## Team Responsibilities

- **Alspencer Omondi:** 6 repositories, 300+ endpoints
- **Ethan Sandoval:** 6 repositories, 300+ endpoints
- **Joint:** Pipeline development, dataset assembly, quality assurance

## Known Limitations & Challenges

1. **Mixed schema styles:** Some repos use multiple schema types
2. **Framework variations:** Different routing conventions (Express vs NestJS)
3. **Rate limiting:** GitHub API limits may slow scraping
4. **False positives:** Some repos have schema files not actively used
5. **Dynamic routing:** Complex routing patterns may be missed

## License

MIT License - Academic project for Machine Learning course

## Contact

- Alspencer Omondi
- Ethan Sandoval

---

**Last Updated:** November 13, 2025  
**Milestone:** 1 of 3  
**Status:** Ready for data collection

