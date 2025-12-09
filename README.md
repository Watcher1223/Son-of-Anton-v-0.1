## Son of Anton: GitHub Mining + Schema‑Style Classification

This project has two main parts:

- **Enhanced GitHub Repository Finder** – a fast GitHub miner that discovers real‑world Node.js / TypeScript API backends (Express, NestJS, Fastify) and their schema definitions (OpenAPI specs, validator files, TypeScript types).
- **ML Dataset & Models** – an end‑to‑end pipeline that clones curated repos, detects endpoints and schema styles, extracts ~110 features per endpoint, and trains classifiers (LogReg, XGBoost, Neural Net) to predict each endpoint’s primary **schema class**: `openapi`, `validator`, or `typedef`.

See [`RESULTS.md`](RESULTS.md) for run‑by‑run dataset and model performance.

---

## 🚀 GitHub Repository Finder (v2.0)

A powerful Python tool to discover GitHub repositories containing JavaScript/TypeScript API endpoints built with Express, NestJS, or Fastify frameworks. Uses GitHub's Git Tree API for efficient file discovery and returns actual endpoint and schema files for ML dataset creation.

### New mining features

- **Git Tree API Integration**: Scans entire repository structure in one API call (10x faster)
- **Actual File Discovery**: Returns lists of endpoint files and schema files with paths
- **Expanded Search Patterns**: All HTTP methods (GET, POST, PUT, DELETE, PATCH)
- **Quality Scoring System**: 0-100 score based on multiple indicators
- **Enhanced Schema Detection**: OpenAPI, GraphQL, Prisma, validators, TypeScript
- **Package.json Deep Analysis**: Verifies frameworks and detects validator libraries
- **Comprehensive Output**: 25+ columns including file lists, scores, and indicators

## Features (Mining)

### 🔍 Smart Repository Discovery

- **Framework-Specific Search**: Targets Express, NestJS, and Fastify patterns
- **All HTTP Methods**: GET, POST, PUT, DELETE, PATCH coverage
- **JavaScript & TypeScript**: Full support for both languages

### 📁 File-Level Discovery

Returns actual files likely to contain endpoints:
- Route files: `routes/**/*.{js,ts}`, `routers/**/*.{js,ts}`
- Controller files: `controllers/**/*.{js,ts}`, `*.controller.ts`
- API directories: `api/**/*.{js,ts}`
- Entry points: `app.js`, `server.js`, `index.js`

Returns actual schema files:
- OpenAPI/Swagger specs: `openapi.{json,yaml}`, `swagger.{json,yaml}`
- JSON schemas: `*.schema.json`, `schemas/**/*.json`
- Validator files: `validators/**/*.{js,ts}`
- TypeScript types: `*.d.ts`, `types/**/*.ts`, `interfaces/**/*.ts`
- GraphQL schemas: `*.graphql`, `*.gql`
- Database schemas: `schema.prisma`, `drizzle.config.*`

### 🎯 Quality Scoring (0-100)

Automatically calculates quality score based on:
- Repository size and stars (0-20 points)
- Schema indicators present (0-30 points)
- Framework verified in package.json (0-20 points)
- TypeScript usage (0-15 points)
- Documentation/tests present (0-15 points)

### 📊 Comprehensive Metadata

Each repository includes:
- Basic info: URL, name, owner, stars, size, language
- Frameworks detected and verified
- **Endpoint files**: List of files with API routes
- **Schema files**: List of spec/validation files
- Quality score and all indicators
- GitHub API URLs for programmatic access

## Requirements

- Python 3.7+
- GitHub Personal Access Token (PAT)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create a `.env` file** with your GitHub token:
```bash
GITHUB_TOKEN=your_actual_github_token_here
```

## Getting a GitHub Personal Access Token

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token" (classic)
3. Give it a descriptive name (e.g., "Repo Finder")
4. **No scopes are required** for public repository access
5. Click "Generate token"
6. Copy the token and add it to your `.env` file

## Usage

### Basic Usage

Run the script to start searching for repositories:

```bash
python find_repos.py
```

The script will:
1. Search GitHub for repositories with API endpoints (all HTTP methods)
2. Use Git Tree API to scan entire repository structure
3. Identify actual endpoint and schema files
4. Filter out small/toy projects (<100KB)
5. Calculate quality scores
6. Save results to timestamped CSV (e.g., `github_repos_20241113_143022.csv`)

### Resume from Checkpoint

If interrupted (Ctrl+C), the script saves progress to `checkpoint.json`. Run again to resume from where you left off.

## Output

### CSV Columns

The generated CSV includes 25+ columns:

**Basic Information:**
- `repo_url`: Full GitHub repository URL
- `repo_name`: Repository name
- `owner`: Repository owner/organization
- `full_name`: Full name (owner/repo)
- `stars`: Number of stars
- `size_kb`: Repository size in KB
- `language`: Primary programming language
- `description`: Repository description
- `last_updated`: Last update timestamp

**Framework Detection:**
- `frameworks`: Detected frameworks (Express, NestJS, Fastify)
- `framework_in_package`: Boolean - framework verified in package.json
- `matched_patterns`: Search queries that found this repo

**File Discovery (NEW):**
- `endpoint_files`: Pipe-separated list of files with endpoints
- `endpoint_file_count`: Number of endpoint files found
- `schema_files`: Pipe-separated list of schema/spec files
- `schema_file_count`: Number of schema files found

**Schema Indicators:**
- `has_openapi`: Boolean - OpenAPI specification present
- `has_swagger`: Boolean - Swagger specification present
- `has_validators`: Boolean - Validation libraries detected
- `validator_libraries`: Comma-separated list of validator libs
- `validator_count`: Number of different validator libraries
- `has_typescript`: Boolean - TypeScript configuration present
- `has_api_docs_folder`: Boolean - Documentation folder present
- `has_tests`: Boolean - Test files present
- `schema_types`: Types of schemas found (OpenAPI, GraphQL, etc.)

**Quality Metrics:**
- `quality_score`: Calculated score (0-100)
- `api_url`: GitHub API URL for programmatic access

### Example Output

```csv
repo_url,repo_name,owner,stars,endpoint_files,endpoint_file_count,schema_files,quality_score,...
https://github.com/user/api,api,user,245,src/routes/users.js | src/routes/posts.js,2,docs/openapi.yaml,85,...
```

## Analyze Results

After collecting repositories, analyze them with:

```bash
python analyze_results.py
```

This generates:
- Detailed statistics and breakdowns
- Filtered CSVs by schema class (OpenAPI, Validator, TypeScript)
- Framework-specific CSVs
- High-quality repository list
- `file_mapping.json` - Programmatic access to all endpoint/schema files

### Sample Analysis Output

```
📊 OVERALL STATISTICS
   Total repositories: 347
   Average quality score: 62.3/100
   Average endpoint files per repo: 8.5

📁 FILE DISCOVERY STATISTICS
   Total endpoint files found: 2,950
   Total schema files found: 1,234

⭐ HIGH-VALUE REPOSITORIES
   High quality + multiple endpoints: 89
   
🎯 RECOMMENDATIONS FOR ML TRAINING DATA
   1. OPENAPI SCHEMA CLASS - Top 5 candidates:
      • user/awesome-api (Score: 95, ⭐1234, 15 endpoints)
   ...
```

## Search Patterns

### Express.js (10 patterns)
- JavaScript: `app.get(`, `app.post(`, `app.put(`, `app.delete(`, `router.get(`, `router.post(`, `router.put(`, `router.delete(`
- TypeScript: `app.get(`, `router.post(`

### NestJS (8 patterns)
- Decorators: `@Get(`, `@Post(`, `@Put(`, `@Delete(`, `@Patch(`, `@Controller(`
- OpenAPI: `@ApiTags(`, `@ApiOperation(`

### Fastify (6 patterns)
- JavaScript: `fastify.get(`, `fastify.post(`, `fastify.put(`, `fastify.delete(`, `fastify.route(`
- TypeScript: `fastify.get(`

## Using the File Lists

The `endpoint_files` and `schema_files` columns contain actual file paths. To fetch them:

```python
import requests

# From CSV row
api_url = row['api_url']  # e.g., https://api.github.com/repos/user/repo
endpoint_file = row['endpoint_files'].split(' | ')[0]  # e.g., src/routes/users.js

# Fetch file content
file_url = f"{api_url}/contents/{endpoint_file}"
headers = {"Authorization": f"token {YOUR_TOKEN}"}
response = requests.get(file_url, headers=headers)

if response.status_code == 200:
    import base64
    content = base64.b64decode(response.json()['content']).decode('utf-8')
    print(content)  # The actual endpoint code!
```

Or use the `file_mapping.json` generated by `analyze_results.py` for easier access.

## Rate Limits

GitHub's API has the following rate limits:

- **Authenticated users**: 30 requests per minute for code search
- **Core API**: 5,000 requests per hour

The script automatically:
- Monitors rate limit status
- Waits when limits are reached
- Retries failed requests
- Saves checkpoints for resume capability

## Performance Improvements

**v2.0 vs v1.0:**
- **10x faster** repository scanning (Git Tree API vs individual file checks)
- **1 API call** to scan entire repo (vs 10+ calls)
- **More accurate** schema detection
- **Actual file paths** returned for immediate use

## Example Workflow for ML Dataset Creation

1. **Collect repositories**:
```bash
python find_repos.py
```

2. **Analyze and filter**:
```bash
python analyze_results.py
```

3. **Select repositories by schema class**:
   - Use `filtered_repos/openapi_class_repos.csv`
   - Use `filtered_repos/validator_class_repos.csv`
   - Use `filtered_repos/typescript_class_repos.csv`

4. **Fetch endpoint files**:
   - Use `file_mapping.json` for programmatic access
   - Or parse `endpoint_files` column from CSV
   - Fetch using GitHub API: `GET /repos/{owner}/{repo}/contents/{path}`

5. **Extract endpoint patterns**:
   - Parse fetched files for route definitions
   - Link to schema files from same repo
   - Build your ML training dataset!

## Project Structure

```
Son-of-Anton-v-0.1/
├── find_repos.py                 # GitHub miner for API/schema‑rich repos
├── analyze_results.py            # Analyze mined repos, build filtered CSVs
├── generate_dataset_from_csv.py  # Clone curated repos & build ML dataset (data/final/*.csv)
├── train_logreg.py               # Train Logistic Regression classifier
├── train_xgboost.py              # Train XGBoost classifier
├── train_nn.py                   # Train simple Neural Network classifier
├── evaluate_models.py            # Compare models (metrics, confusion matrices, PR curves)
├── RESULTS.md                    # Run‑by‑run dataset + model performance summary
├── requirements.txt              # Python dependencies
├── .env                          # GitHub token and other env (create this)
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── QUICKSTART.md                 # Quick start guide
├── checkpoint.json               # Auto-generated mining checkpoint
├── github_repos_*.csv            # Mined GitHub repo metadata
├── curated_repos.csv             # Hand‑curated training repos (Express/NestJS/Fastify)
├── data/
│   ├── raw/                      # Cloned Git repositories
│   ├── intermediate/             # Per‑repo JSON feature dumps
│   └── final/                    # dataset.csv, train.csv, test.csv
├── src/
│   ├── detectors/                # endpoint_detector.py, schema_detector.py
│   ├── extractors/               # endpoint_features.py, repo_features.py, schema_linker.py
│   ├── dataset/                  # assembler.py
│   └── models/                   # preprocessor.py, logistic_regression.py, xgboost_model.py, neural_network.py, evaluation.py, ...
└── filtered_repos/               # Filtered datasets (after analysis)
    ├── high_quality_repos.csv
    ├── openapi_class_repos.csv
    ├── validator_class_repos.csv
    ├── typescript_class_repos.csv
    ├── express_repos.csv
    ├── nestjs_repos.csv
    ├── fastify_repos.csv
    ├── endpoint_rich_repos.csv
    └── file_mapping.json
```

## Troubleshooting

### "GITHUB_TOKEN not found in environment variables"

Make sure you have:
1. Created a `.env` file in the project directory
2. Added your token: `GITHUB_TOKEN=your_token_here`
3. Not quoted the token value

### Rate Limit Errors

If you hit rate limits:
- The script automatically waits and retries
- Progress is saved in `checkpoint.json`
- Run the script again after rate limits reset (hourly)

### Interrupted Search

If you stop the script (Ctrl+C):
- Current results are exported to CSV automatically
- Progress saved in `checkpoint.json`
- Run again to resume from checkpoint

## Advanced Usage

### Custom Search Queries

Edit the `patterns` dictionary in `find_repos.py` to add custom search patterns:

```python
patterns = {
    'Express': [
        'language:JavaScript "app.get("',
        # Add your custom patterns
    ]
}
```

### Adjust Filtering

Modify size threshold in `process_search_results()`:

```python
if size_kb < 100:  # Change this value
    print(f"Skipping {repo_full_name} (too small: {size_kb} KB)")
    continue
```

## Contributing

This tool is designed for research and ML dataset creation. Please ensure you:
- Respect GitHub's Terms of Service and rate limits
- Only access public repositories
- Do not make live requests to discovered endpoints
- Properly attribute and cite repositories in research

## License

This project is provided as-is for research and educational purposes.

## Changelog

### v2.0 (Current)
- ✨ Git Tree API integration for 10x faster scanning
- ✨ Returns actual endpoint and schema file paths
- ✨ Expanded search patterns (all HTTP methods)
- ✨ Quality scoring system (0-100)
- ✨ Enhanced schema detection (GraphQL, Prisma, etc.)
- ✨ Package.json deep analysis
- ✨ 25+ CSV columns with comprehensive metadata
- ✨ Enhanced analyzer with file mapping export

### v1.0
- Basic repository search
- Express, NestJS, Fastify detection
- Simple schema indicators
- CSV export
```

---

## ML Dataset & Models

In addition to mining, this repo contains a full pipeline for building and evaluating a **schema‑style classifier** over real‑world Node.js/TypeScript API endpoints.

### Dataset generation pipeline

The script `generate_dataset_from_csv.py` orchestrates dataset creation:

- **Input**: `curated_repos.csv` – a hand‑curated list of ~100 Node/TypeScript backends spanning Express, NestJS and Fastify.
- **Clone & validate**: clones each repo into `data/raw/` and drops non‑Node or empty projects using `src/scrapers/repo_filter.py`.
- **Endpoint detection** (`src/detectors/endpoint_detector.py`):
  - Express: `app.get/post/put/delete/patch/options/head`, `router.*`, `.route().get()` with string and template literal paths.
  - NestJS: `@Controller()`, `@Get/@Post/@Put/@Delete/@Patch/@All/@Options/@Head`.
  - Fastify: `fastify/app/server.get/post/put/delete/patch`, `fastify.route({ method, url })`.
  - Hapi: `server.route({ method, path, handler })`.
  - Koa: `router.get/post/put/delete/patch` (including template literal paths).
- **Schema detection** (`src/detectors/schema_detector.py`):
  - Classifies each repo into `openapi`, `validator`, or `typedef` using file patterns and content (OpenAPI specs, validator imports, TS types, etc.).
- **Endpoint‑level features** (`src/extractors/endpoint_features.py`):
  - Path & method shape (`path_depth`, `path_length`, `path_param_count`, `is_get/is_post/...`, `modifies_data`, ...).
  - Code‑context (`has_middleware`, `has_auth`, `has_async`, `has_try_catch`, `has_response_types`).
  - Parameter usage (`has_query_params`, `has_body_params`, `has_file_upload`, `@Body/@Query/@Param`, `req.body`, `req.query`, ...).
  - **Validation signals**: `has_body_decorator`, `has_validation_pipe`, `has_validate_call`, `has_joi_validate`, `has_zod_parse`, `has_yup_validate`, `has_express_validator`, plus file‑level imports like `file_imports_class_validator`, `file_imports_joi`, `file_imports_zod`, `file_imports_express_validator`, etc.
  - **OpenAPI signals**: `has_api_body_decorator`, `has_api_response_decorator`, `has_api_operation_decorator`, `has_api_property_decorator`, `has_swagger_comment`, and `file_imports_swagger`, swagger‑jsdoc / `swagger-ui-express` imports.
  - **TypeDef signals**: `has_dto_reference`, `has_interface_cast`, `has_type_annotation`, `has_generic_type`, `has_return_type`, `file_imports_dto`, `file_imports_types`.
  - Aggregate counts: `validator_signal_count`, `openapi_signal_count`, `typedef_signal_count`.
- **Repository‑level features** (`src/extractors/repo_features.py`):
  - File counts (`js/ts/jsx/tsx`), total files & LOC, `endpoints_per_1000_lines`, `has_orm`, `has_testing`, `has_linting`, `has_readme`, `has_docs_folder`, `has_api_docs`, `has_swagger`, etc.
- **Schema linking & dataset assembly** (`src/extractors/schema_linker.py`, `src/dataset/assembler.py`):
  - Link endpoints to nearby schema files (OpenAPI specs, validator and type definition files).
  - Drop endpoints with `schema_class = 'unknown'`.
  - Deduplicate `(repo_name, http_method, endpoint_path)` combinations.
  - Save:
    - `data/final/dataset.csv` – full labeled feature table (110 columns: ID + 109 features/label)
    - `data/final/train.csv` / `data/final/test.csv` – simple random splits for quick inspection.

### Current dataset snapshot (Run 2 – 2025‑12‑08)

- **Total endpoints**: 816  
- **Total repositories**: 37 (Node.js/TypeScript backends)  
- **Feature columns**: 109  

Schema class distribution (`schema_class`):

| Class     | Count | Share |
|----------|-------|-------|
| validator| 518   | 63.5% |
| typedef  | 152   | 18.6% |
| openapi  | 146   | 17.9% |

Train/Test split (via `DataPreprocessor` `GroupShuffleSplit` on `repo_name`):

| Split | Endpoints | Repos | validator | typedef | openapi |
|-------|-----------|-------|-----------|---------|---------|
| Train | 676       | 29    | —         | —       | —       |
| Test  | 140       | 8     | 131       | 1       | 8       |

> The held‑out test set is **repo‑disjoint** from the training set to avoid leaking repo‑specific patterns.

OpenAPI endpoints now come from five distinct repos (e.g. `agile-team-tool`, several `@nestjs/swagger` examples, a Fastify swagger example, and [`shapeshift/unchained`](https://github.com/shapeshift/unchained)), so the model sees both decorator‑based and swagger‑jsdoc/tsoa‑generated specs.

### Training & evaluation

Training scripts:

```bash
python train_logreg.py  --data data/final/dataset.csv --log-level INFO
python train_xgboost.py --data data/final/dataset.csv --log-level INFO
python train_nn.py      --data data/final/dataset.csv --log-level INFO --epochs 50 --batch-size 64
```

Evaluation:

```bash
python evaluate_models.py --data data/final/dataset.csv --log-level INFO
```

`evaluate_models.py` reloads the trained models from `models/`, evaluates them on the 140‑endpoint held‑out set, and writes a summary table to `results/model_comparison.csv` plus plots in `results/`.

### Latest model performance (Run 2)

On the 140‑endpoint, 8‑repo test set:

| Model              | Accuracy | F1 (Macro) | F1 (Weighted) | F1(openapi) | F1(typedef) | F1(validator) | API calls saved |
|--------------------|----------|-----------:|--------------:|------------:|------------:|--------------:|----------------:|
| **XGBoost**        | 0.99     | 0.67       | **0.99**      | **1.00**    | 0.00        | 0.996         | **94.5%**       |
| **LogisticReg**    | 0.87     | 0.50       | 0.90          | 0.57        | 0.00        | 0.93          | 86.0%           |
| **Neural Network** | 0.74     | 0.39       | 0.81          | 0.31        | 0.00        | 0.84          | 77.0%           |

Highlights:

- **OpenAPI recall is 100% across all models** (all 8 openapi endpoints in the held‑out repos are correctly identified).
- **Validator vs non‑validator separation is strong**, especially for XGBoost (F1≈0.996 on `validator`).
- **Typedef remains under‑represented in the test set** (1 endpoint), so its F1 is still 0; addressing this will require more diverse typedef‑heavy repos and/or richer type‑usage features.
- In terms of **practical value**, XGBoost can avoid ~94.5% of schema‑inference API calls while keeping overall endpoint‑level accuracy at ~99%.

For more detailed, run‑by‑run metrics and rationale, see [`RESULTS.md`](RESULTS.md).

---

**Need Help?** Open an issue or check the [QUICKSTART.md](QUICKSTART.md) for a quick reference guide.
