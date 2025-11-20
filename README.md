# Enhanced GitHub Repository Finder for API Endpoints

A powerful Python tool to discover GitHub repositories containing JavaScript/TypeScript API endpoints built with Express, NestJS, or Fastify frameworks. Uses GitHub's Git Tree API for efficient file discovery and returns actual endpoint and schema files for ML dataset creation.

## 🚀 New Features (v2.0)

- **Git Tree API Integration**: Scans entire repository structure in one API call (10x faster)
- **Actual File Discovery**: Returns lists of endpoint files and schema files with paths
- **Expanded Search Patterns**: All HTTP methods (GET, POST, PUT, DELETE, PATCH)
- **Quality Scoring System**: 0-100 score based on multiple indicators
- **Enhanced Schema Detection**: OpenAPI, GraphQL, Prisma, validators, TypeScript
- **Package.json Deep Analysis**: Verifies frameworks and detects validator libraries
- **Comprehensive Output**: 25+ columns including file lists, scores, and indicators

## Features

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
├── find_repos.py              # Main enhanced script
├── analyze_results.py         # Results analyzer
├── requirements.txt           # Python dependencies
├── .env                       # GitHub token (create this)
├── .gitignore                # Git ignore rules
├── README.md                  # This file
├── QUICKSTART.md             # Quick start guide
├── checkpoint.json           # Auto-generated checkpoint
├── github_repos_*.csv        # Generated CSV files
└── filtered_repos/           # Filtered datasets (after analysis)
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

---

**Need Help?** Open an issue or check the [QUICKSTART.md](QUICKSTART.md) for a quick reference guide.

**ML Dataset Ready!** 🚀 This tool is specifically designed to help build machine learning datasets for API schema extraction and endpoint analysis.
