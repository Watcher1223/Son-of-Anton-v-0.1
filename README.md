# API Schema Extraction Prediction - Milestone 1

Machine Learning project for predicting optimal schema-extraction methods for Node.js APIs.

## ⚡ Quick Start

```bash
# 1. Setup environment
./quickstart.sh

# 2. Add GitHub token (get from: https://github.com/settings/tokens)
echo "GITHUB_TOKEN=your_token_here" > .env

# 3. Run pipeline
python main.py --mode full --target 12
```

**That's it!** The pipeline will collect 12 repositories and extract 600+ API endpoints with features.

## Project Overview

This project aims to predict which schema extraction strategy (OpenAPI, Validator-based, or TypeScript typedef) will be most successful for a given API endpoint, reducing the number of extraction attempts from 15-30 API calls to ≤5 calls per endpoint.

### Three Schema Classes
1. **OpenAPI**: Swagger/OpenAPI specification files
2. **Validator-based**: Runtime validators (Joi, Zod, class-validator, Pydantic)
3. **TypeScript typedef/DTO-based**: TypeScript interfaces and Data Transfer Objects

## Milestone 1 Goals

- Collect 12 Node.js repositories (4 per schema class)
- Extract ≥600 labeled endpoints with features
- Build feature extraction pipeline
- Generate structured dataset for model training

## Project Structure

```
Son-of-Anton-v-0.1/
├── src/
│   ├── scrapers/
│   │   ├── github_scraper.py      # GitHub repo search and cloning
│   │   └── repo_filter.py         # Repository filtering logic
│   ├── detectors/
│   │   ├── schema_detector.py     # Detect schema types in repos
│   │   └── endpoint_detector.py   # Find API endpoints
│   ├── extractors/
│   │   ├── endpoint_features.py   # Extract endpoint-level features
│   │   ├── repo_features.py       # Extract repository-level features
│   │   └── schema_linker.py       # Link endpoints to schema sources
│   ├── utils/
│   │   ├── file_parser.py         # Parse JS/TS/YAML files
│   │   └── framework_detector.py  # Detect Express/NestJS/Fastify
│   └── dataset/
│       └── assembler.py           # Assemble final dataset CSV
├── data/
│   ├── raw/                       # Cloned repositories
│   ├── intermediate/              # Extracted features (JSON)
│   └── final/                     # Final CSV dataset
├── config/
│   └── config.yaml                # Configuration settings
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── main.py                        # Main execution script
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file with your GitHub token:
```
GITHUB_TOKEN=your_github_personal_access_token
```

2. Adjust settings in `config/config.yaml` as needed

## Usage

### Collect Repositories
```bash
python main.py --mode collect --target 12
```

### Extract Features
```bash
python main.py --mode extract --input data/raw/
```

### Assemble Dataset
```bash
python main.py --mode assemble --output data/final/dataset.csv
```

### Run Full Pipeline
```bash
python main.py --mode full
```

## Features Extracted

### Endpoint-Level Features
- HTTP method (GET, POST, PUT, DELETE, etc.)
- Path depth (number of segments)
- Parameter count and location (query/body/path)
- Presence of middleware
- Authentication indicators

### Repository-Level Features
- Primary language
- Framework (Express, NestJS, Fastify)
- Codebase size (lines of code, file count)
- Dependency list
- Endpoint density
- Documentation presence

### Extraction-Source Features
- Presence of OpenAPI spec
- Validator files detected
- TypeScript type definitions
- Database models
- Comments and docstrings

## Dataset Schema

The final CSV will contain columns for all features plus:
- `endpoint_id`: Unique identifier
- `repo_name`: Source repository
- `schema_class`: Label (openapi/validator/typedef)
- `endpoint_path`: API route path
- `http_method`: HTTP verb
- ... (all feature columns)

## Team

- **Alspencer Omondi**: 6 repositories, 300 endpoints
- **Ethan Sandoval**: 6 repositories, 300 endpoints

## Timeline

- **Nov 6-15**: Milestone 1 (Data Collection + Feature Engineering)
- **Nov 15-21**: Milestone 2 (Model Development)
- **Nov 22-25**: Milestone 3 (Optimization & Report)
- **Dec 9**: Final Deliverables

## License

Academic project for Machine Learning course. 

# If you like son of Anton V0.1 reach out

