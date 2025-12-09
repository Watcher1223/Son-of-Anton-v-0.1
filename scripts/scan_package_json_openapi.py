#!/usr/bin/env python3
"""Scan GitHub repos from CSV and auto-flag OpenAPI-capable Node backends.

Usage:
  GITHUB_TOKEN=... python scripts/scan_package_json_openapi.py github_repos_20251113_024339.csv openapi_candidates.csv

This script:
  - Reads the scraped repos CSV
  - For each repo, uses the GitHub API to fetch package.json from likely backend dirs
  - Looks for both web frameworks (express/nest/fastify/hapi) and OpenAPI libs
  - Writes a CSV of candidate OpenAPI backends with detected libraries
"""

import csv
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

GITHUB_API = "https://api.github.com"

# Heuristics
FRAMEWORK_LIBS = {
    "express",
    "@nestjs/core",
    "fastify",
    "@fastify/fastify",
    "@hapi/hapi",
}

OPENAPI_LIBS = {
    "swagger-jsdoc",
    "swagger-ui-express",
    "@nestjs/swagger",
    "@fastify/swagger",
    "hapi-swagger",
    "tsoa",
    "express-openapi-validator",
}

BACKEND_DIR_CANDIDATES = [
    "",
    "server",
    "backend",
    "api",
    "node",
    "packages/api",
    "packages/server",
]


@dataclass
class RepoRecord:
    repo_url: str
    full_name: str
    stars: int
    language: str
    frameworks: str
    has_openapi_flag: bool


@dataclass
class ScanResult:
    full_name: str
    repo_url: str
    stars: int
    language: str
    frameworks: str
    backend_dir: Optional[str]
    found_package_json: bool
    has_framework_lib: bool
    has_openapi_lib: bool
    framework_libs_found: str
    openapi_libs_found: str


def get_token() -> Optional[str]:
    return os.environ.get("GITHUB_TOKEN")


def github_get(path: str):
    url = f"{GITHUB_API}{path}"
    headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_package_json(full_name: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Try to fetch package.json from several likely backend directories.

    Returns (package_json_dict, dir_prefix or None).
    """
    for prefix in BACKEND_DIR_CANDIDATES:
        path = f"/{full_name}/contents/{prefix + '/' if prefix else ''}package.json"
        data = github_get(path)
        if not data:
            continue
        if isinstance(data, dict) and data.get("encoding") == "base64":
            import base64, json

            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            try:
                pkg = json.loads(content)
                return pkg, prefix or "."
            except Exception:
                continue
    return None, None


def parse_deps(pkg: Dict) -> Dict[str, str]:
    deps: Dict[str, str] = {}
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        if isinstance(pkg.get(section), dict):
            deps.update(pkg[section])
    return deps


def analyze_repo(rec: RepoRecord) -> ScanResult:
    pkg, backend_dir = fetch_package_json(rec.full_name)
    if not pkg:
        return ScanResult(
            full_name=rec.full_name,
            repo_url=rec.repo_url,
            stars=rec.stars,
            language=rec.language,
            frameworks=rec.frameworks,
            backend_dir=None,
            found_package_json=False,
            has_framework_lib=False,
            has_openapi_lib=False,
            framework_libs_found="",
            openapi_libs_found="",
        )

    deps = {name.lower(): name for name in parse_deps(pkg).keys()}

    fw_found = sorted({orig for low, orig in deps.items() if low in FRAMEWORK_LIBS})
    oa_found = sorted({orig for low, orig in deps.items() if low in OPENAPI_LIBS})

    return ScanResult(
        full_name=rec.full_name,
        repo_url=rec.repo_url,
        stars=rec.stars,
        language=rec.language,
        frameworks=rec.frameworks,
        backend_dir=backend_dir,
        found_package_json=True,
        has_framework_lib=bool(fw_found),
        has_openapi_lib=bool(oa_found),
        framework_libs_found=";".join(fw_found),
        openapi_libs_found=";".join(oa_found),
    )


def load_repos(csv_path: Path) -> List[RepoRecord]:
    records: List[RepoRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stars = int(row.get("stars", 0) or 0)
            except ValueError:
                stars = 0
            records.append(
                RepoRecord(
                    repo_url=row.get("repo_url", ""),
                    full_name=row.get("full_name", ""),
                    stars=stars,
                    language=row.get("language", ""),
                    frameworks=row.get("frameworks", ""),
                    has_openapi_flag=str(row.get("has_openapi", "")).lower() == "true",
                )
            )
    return records


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        print("Usage: scan_package_json_openapi.py <input_csv> <output_csv>")
        sys.exit(1)

    in_csv = Path(argv[1])
    out_csv = Path(argv[2])

    repos = load_repos(in_csv)
    print(f"Loaded {len(repos)} repos from {in_csv}")

    results: List[ScanResult] = []
    for i, rec in enumerate(repos, 1):
        if not rec.full_name:
            continue
        # Optional: quick pre-filter to Node-ish languages/frameworks
        if rec.language not in {"JavaScript", "TypeScript"}:
            continue
        print(f"[{i}/{len(repos)}] Scanning {rec.full_name} (stars={rec.stars})")
        try:
            res = analyze_repo(rec)
            # Keep only ones where we found package.json and at least some relevant libs
            if res.found_package_json and (res.has_openapi_lib or rec.has_openapi_flag):
                results.append(res)
        except Exception as e:
            print(f"  Error scanning {rec.full_name}: {e}")

    print(f"\nFound {len(results)} candidate OpenAPI backends.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(results[0]).keys()) if results else [
                "full_name","repo_url","stars","language","frameworks",
                "backend_dir","found_package_json","has_framework_lib",
                "has_openapi_lib","framework_libs_found","openapi_libs_found",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main(sys.argv)
