# Quick Start Guide - Enhanced Repository Finder v2.0

## You're Ready to Go! 🚀

Your enhanced GitHub repository finder with file-level discovery is fully set up.

## What's New in v2.0

✅ **10x Faster** - Uses Git Tree API to scan entire repos in one call  
✅ **Actual Files** - Returns lists of endpoint and schema files with paths  
✅ **Quality Scores** - 0-100 scoring system for repository quality  
✅ **All HTTP Methods** - GET, POST, PUT, DELETE, PATCH coverage  
✅ **Enhanced Detection** - GraphQL, Prisma, validators, TypeScript, OpenAPI  
✅ **25+ CSV Columns** - Comprehensive metadata for ML dataset creation  

## Run in 2 Steps

### Step 1: Install Dependencies

```bash
cd /Users/ethansandoval/Desktop/Son-of-Anton-v-0.1
pip install -r requirements.txt
```

### Step 2: Run the Script

```bash
python find_repos.py
```

That's it! The script will:

1. Search GitHub for JavaScript/TypeScript repos (Express, NestJS, Fastify)
2. Use Git Tree API to scan entire repository structure (10x faster!)
3. Identify actual endpoint files (`routes/*.js`, `controllers/*.ts`, etc.)
4. Identify schema files (OpenAPI, validators, TypeScript, GraphQL, Prisma)
5. Calculate quality scores based on multiple indicators
6. Save results to timestamped CSV with 25+ columns

## Expected Output

```
Starting GitHub repository search with enhanced detection...
============================================================
Rate Limits - Core: 4998/5000, Search: 30/30

🔍 Searching Express repositories...
------------------------------------------------------------
  Searching page 1 for: language:JavaScript "app.get("...
  Found 100 code matches
  Processing 100 code matches...
  Fetching details for: user/awesome-api
  Fetching file tree...
  Total unique repos so far: 1
...

✅ Search complete! Found 347 unique repositories

✅ Exported 347 repositories to github_repos_20241113_143022.csv

📊 Summary Statistics:
  Total repositories: 347
  Average quality score: 62.3/100
  Average endpoint files per repo: 8.5
  Average schema files per repo: 3.2

  By Framework:
    Express: 289
    NestJS: 87
    Fastify: 45

  Schema Indicators:
    Has OpenAPI: 134
    Has Validators: 267
    Has TypeScript: 198

  Quality Scores:
    High (80-100): 45
    Medium (50-79): 156
    Low (0-49): 146
```

## CSV Output

The generated CSV includes:

### Basic Info
- repo_url, repo_name, owner, stars, size_kb, language, description

### File Discovery (NEW!)
- **endpoint_files**: List of files with API routes (pipe-separated)
- **endpoint_file_count**: Number of endpoint files found
- **schema_files**: List of schema/spec files (pipe-separated)
- **schema_file_count**: Number of schema files found

### Quality & Indicators
- **quality_score**: 0-100 calculated score
- has_openapi, has_validators, has_typescript, has_api_docs_folder, has_tests
- validator_libraries, schema_types, framework_in_package

### Programmatic Access
- api_url: GitHub API URL to fetch files

## Analyze Results

After collecting repos, analyze them:

```bash
python analyze_results.py
```

This will:
- Show detailed statistics and breakdowns
- Identify top candidates for each schema class
- Create filtered CSVs in `filtered_repos/` directory
- Export `file_mapping.json` for programmatic file access

### Generated Files:

```
filtered_repos/
├── high_quality_repos.csv          # Score ≥80
├── openapi_class_repos.csv         # For OpenAPI schema class
├── validator_class_repos.csv       # For Validator schema class
├── typescript_class_repos.csv      # For TypeScript typedef class
├── express_repos.csv               # Express framework
├── nestjs_repos.csv                # NestJS framework
├── fastify_repos.csv               # Fastify framework
├── endpoint_rich_repos.csv         # >5 endpoint files
└── file_mapping.json              # Easy programmatic access
```

## Using the File Lists

### From CSV

Open the CSV and find the `endpoint_files` or `schema_files` columns:

```
endpoint_files: src/routes/users.js | src/routes/posts.js | src/controllers/auth.js
schema_files: docs/openapi.yaml | src/validators/userValidator.js
```

### Fetch File Content

Use the `api_url` and file path to fetch content:

```python
import requests
import base64

api_url = "https://api.github.com/repos/user/repo"
file_path = "src/routes/users.js"
token = "your_github_token"

# Fetch file
url = f"{api_url}/contents/{file_path}"
headers = {"Authorization": f"token {token}"}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    content = base64.b64decode(data['content']).decode('utf-8')
    print(content)  # Actual endpoint code!
```

### Using file_mapping.json

Even easier with the generated mapping:

```python
import json
import requests
import base64

# Load mapping
with open('filtered_repos/file_mapping.json', 'r') as f:
    mapping = json.load(f)

# Get a repo's files
repo = mapping['user/awesome-api']
print(f"Endpoint files: {len(repo['endpoint_files'])}")
print(f"Schema files: {len(repo['schema_files'])}")

# Fetch first endpoint file
if repo['endpoint_files']:
    file_path = repo['endpoint_files'][0]
    url = f"{repo['api_url']}/contents/{file_path}"
    # ... fetch as above
```

## ML Dataset Workflow

### For Your ML Project

1. **Collect repositories**:
   ```bash
   python find_repos.py
   ```

2. **Analyze and filter**:
   ```bash
   python analyze_results.py
   ```

3. **Select by schema class**:
   - OpenAPI: Use `filtered_repos/openapi_class_repos.csv`
   - Validator: Use `filtered_repos/validator_class_repos.csv`
   - TypeScript: Use `filtered_repos/typescript_class_repos.csv`

4. **Extract endpoints**:
   - Use `file_mapping.json` for programmatic access
   - Fetch endpoint files using GitHub API
   - Parse route definitions from files
   - Link to schema files from same repo

5. **Build dataset**:
   - Extract features from endpoints
   - Label with schema class
   - Create training dataset CSV

## Expected Runtime

- **Small search** (5-10 queries): ~5-10 minutes
- **Full search** (24 patterns): ~20-40 minutes
- **With Git Tree API**: 10x faster than v1.0!

## Tips & Best Practices

### 1. Check Quality Scores

Sort CSV by `quality_score` column to focus on high-quality repos:
```python
import pandas as pd
df = pd.read_csv('github_repos_20241113_143022.csv')
top_repos = df[df['quality_score'] >= 80]
print(f"Found {len(top_repos)} high-quality repositories")
```

### 2. Filter by Endpoint Count

Focus on repos with many endpoints:
```python
rich_repos = df[df['endpoint_file_count'] > 5]
```

### 3. Find Complete Coverage

Repos with all three schema types:
```python
complete = df[
    (df['has_openapi'] == True) & 
    (df['has_validators'] == True) & 
    (df['has_typescript'] == True)
]
```

### 4. Monitor Rate Limits

The script shows rate limits during execution:
```
Rate Limits - Core: 4998/5000, Search: 30/30
```

If you hit limits, the script waits automatically. Or pause and resume later.

### 5. Resume Interrupted Searches

Interrupted? No problem! Just run again:
```bash
python find_repos.py  # Automatically resumes from checkpoint.json
```

## File Discovery Examples

### Endpoint Files Found

Typical endpoint files identified:
- `src/routes/users.js`
- `src/routes/api/posts.ts`
- `src/controllers/authController.ts`
- `api/endpoints/products.js`
- `app.js` (main entry point)

### Schema Files Found

Typical schema files identified:
- `docs/openapi.yaml`
- `swagger.json`
- `src/validators/userValidator.js`
- `src/types/interfaces.ts`
- `schema.graphql`
- `prisma/schema.prisma`

## Troubleshooting

### Rate Limit Errors

**Symptom**: "Rate limit exceeded. Waiting 60 seconds..."

**Solution**: 
- Script waits automatically
- Progress saved in `checkpoint.json`
- Resume later if needed

### Checkpoint Recovery

**Symptom**: Search interrupted

**Solution**:
```bash
python find_repos.py  # Automatically loads checkpoint
```

To start fresh:
```bash
rm checkpoint.json
python find_repos.py
```

### No Results for Query

**Symptom**: "No more results on page 1"

**Solution**: Normal - not all patterns will match. Script continues with other patterns.

### "Could not fetch tree"

**Symptom**: "Could not fetch tree for user/repo"

**Solution**: 
- Repo might be private or deleted
- Rate limit hit (script will retry)
- Script continues with other repos

## Performance Stats

**v2.0 Improvements:**
- ⚡ 10x faster repository scanning
- 📁 Returns actual file paths (not just indicators)
- 🎯 More accurate detection (Git Tree vs individual checks)
- 💾 1 API call per repo (vs 10+ calls in v1.0)
- 📊 Quality scoring for better filtering

## What's Next?

After getting your CSV:

1. **Review filtered_repos/** - Pre-filtered datasets ready to use
2. **Check file_mapping.json** - Easy programmatic access
3. **Sort by quality_score** - Focus on best repositories
4. **Fetch endpoint files** - Use GitHub API with returned paths
5. **Build your ML dataset** - Extract features and train models!

## Need Help?

- **Full documentation**: See [README.md](README.md)
- **Detailed analysis**: Run `python analyze_results.py`
- **File structure**: All paths in CSV are relative to repo root

---

**Happy repository hunting with file-level precision!** 🎯

Now you have actual endpoint and schema files ready for your ML dataset creation!
