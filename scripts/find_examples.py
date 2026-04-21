#!/usr/bin/env python3
"""
find_examples.py — Find example/tutorial/script files in a GitHub repository.

Ported from huggingface/ml-intern's github_find_examples.py. Uses fuzzy matching
against a curated list of "examples/", "scripts/", "tutorials/", etc. directory
patterns, optionally filtered by a keyword.

Use AFTER list_repos.py to pick a repo, BEFORE read_file.py to study the match.

Requires GITHUB_TOKEN in the environment.

Usage:
    find_examples.py --repo trl --keyword sft
    find_examples.py --repo transformers --keyword grpo --max-results 20
    find_examples.py --repo diffusers --org huggingface
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests
from thefuzz import fuzz

# In order of priority (lower index = higher priority)
EXAMPLE_PATTERNS = [
    "scripts",
    "examples", "example",
    "notebooks", "notebook",
    "tutorials", "tutorial",
    "quickstart", "walkthroughs", "walkthrough",
    "cookbook", "cookbooks", "recipes", "recipe",
    "demos", "demo", "samples", "sample",
    "guides", "guide",
    "getting-started", "getting_started",
    "playground", "howto", "how-to",
    "use-cases", "usecases", "use_cases",
    "sandbox", "showcase",
]


def _get_repo_tree(org: str, repo: str, token: str) -> tuple[list[dict[str, Any]], str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }
    full_repo = f"{org}/{repo}"
    try:
        response = requests.get(f"https://api.github.com/repos/{full_repo}", headers=headers, timeout=10)
        if response.status_code == 404:
            return [], "not_found"
        if response.status_code != 200:
            return [], f"API error: {response.status_code}"
        default_branch = response.json().get("default_branch", "main")
    except Exception as e:
        return [], f"Error fetching repo: {e}"

    try:
        response = requests.get(
            f"https://api.github.com/repos/{full_repo}/git/trees/{default_branch}",
            headers=headers, params={"recursive": "1"}, timeout=30,
        )
        if response.status_code != 200:
            return [], f"Error fetching tree: {response.status_code}"
        tree = response.json().get("tree", [])
        files = [
            {
                "path": item["path"],
                "ref": item["sha"],
                "size": item.get("size", 0),
                "url": f"https://github.com/{full_repo}/blob/{default_branch}/{item['path']}",
            }
            for item in tree
            if item["type"] == "blob"
        ]
        return files, ""
    except Exception as e:
        return [], f"Error processing tree: {e}"


def _search_similar_repos(org: str, repo: str, token: str) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }
    query = f"org:{org} {repo}"
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 10},
            timeout=30,
        )
        if response.status_code != 200:
            return []
        items = response.json().get("items", [])
        return [
            {
                "full_name": item.get("full_name"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count", 0),
                "url": item.get("html_url"),
            }
            for item in items
        ]
    except Exception:
        return []


def _score_example(path: str) -> int:
    return max((fuzz.token_set_ratio(p.lower(), path.lower()) for p in EXAMPLE_PATTERNS), default=0)


def _score_keyword(path: str, keyword: str) -> int:
    return max(
        fuzz.partial_ratio(keyword.lower(), path.lower()),
        fuzz.token_set_ratio(keyword.lower(), path.lower()),
    )


def _pattern_priority(path: str) -> tuple[int, int, int]:
    path_lower = path.lower()
    parts = path_lower.split("/")
    in_examples_dir = 0 if (parts[0] in ("examples", "example")) else 1
    best_priority = 999
    best_depth_at_match = -1
    for i, pattern in enumerate(EXAMPLE_PATTERNS):
        if pattern in parts:
            depth = len(parts) - 1 - parts[::-1].index(pattern)
            if depth > best_depth_at_match or (depth == best_depth_at_match and i < best_priority):
                best_priority = i
                best_depth_at_match = depth
    return (in_examples_dir, best_priority, len(parts))


def find_examples(
    keyword: str,
    repo: str,
    org: str = "huggingface",
    max_results: int = 10,
    min_score: int = 80,
) -> tuple[str, bool]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "ERROR: GITHUB_TOKEN environment variable is required.", False
    if not repo:
        return "ERROR: --repo is required.", False

    all_files, error = _get_repo_tree(org, repo, token)

    if error == "not_found":
        similar = _search_similar_repos(org, repo, token)
        if not similar:
            return f"Repository '{org}/{repo}' not found and no similar repositories found.", False
        lines = [f"**Repository '{org}/{repo}' not found. Similar repositories:**\n"]
        for i, r in enumerate(similar, 1):
            lines.append(f"{i}. **{r['full_name']}** (⭐ {r['stars']:,})")
            if r["description"]:
                d = r["description"][:100] + ("..." if len(r["description"]) > 100 else "")
                lines.append(f"   {d}")
            lines.append(f"   {r['url']}\n")
        return "\n".join(lines), False
    if error:
        return f"Error accessing repository '{org}/{repo}': {error}", False
    if not all_files:
        return f"No files found in repository '{org}/{repo}'.", True

    example_threshold = 60
    example_files = [
        {**f, "example_score": score}
        for f in all_files
        if (score := _score_example(f["path"])) >= example_threshold
    ]
    if not example_files:
        return (
            f"No example files found in {org}/{repo} (no files match example patterns "
            f"with score >= {example_threshold})."
        ), True

    if keyword:
        scored = [
            {**f, "score": s}
            for f in example_files
            if (s := _score_keyword(f["path"], keyword)) >= min_score
        ]
        if not scored:
            return (
                f"No files in {org}/{repo} match keyword '{keyword}' (min score: {min_score}) "
                f"among {len(example_files)} example files."
            ), True
        scored.sort(key=lambda x: x["score"], reverse=True)
    else:
        scored = []
        for f in example_files:
            in_ex, prio, depth = _pattern_priority(f["path"])
            scored.append({**f, "score": f["example_score"],
                           "in_examples_dir": in_ex, "pattern_priority": prio, "path_depth": depth})
        scored.sort(key=lambda x: (x["in_examples_dir"], x["pattern_priority"], x["path_depth"], x["path"]))

    results = scored[:max_results]
    kw_desc = f" matching '{keyword}'" if keyword else ""
    lines = [f"**Found {len(results)} example files in {org}/{repo}{kw_desc}:**"]
    if len(scored) > max_results:
        lines[0] += f" (showing {max_results} of {len(scored)})"
    lines.append("")
    for i, f in enumerate(results, 1):
        lines.append(f"{i}. **{f['path']}**")
        lines.append(f"   Size: {f['size']:,} bytes | Ref: {f['ref'][:7]}")
        lines.append(f"   URL: {f['url']}")
        lines.append(f"   read_file.py --repo {org}/{repo} --path '{f['path']}'")
        lines.append("")
    return "\n".join(lines), True


def main() -> int:
    p = argparse.ArgumentParser(prog="find_examples.py", description=__doc__)
    p.add_argument("--repo", required=True, help="Repository name (e.g. 'trl').")
    p.add_argument("--org", default="huggingface", help="GitHub org/user (default: huggingface).")
    p.add_argument("--keyword", default="", help="Fuzzy-match keyword (e.g. 'sft').")
    p.add_argument("--max-results", type=int, default=10)
    p.add_argument("--min-score", type=int, default=60,
                   help="Minimum fuzzy match score 0-100 (default: 60).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    out, ok = find_examples(
        keyword=args.keyword, repo=args.repo, org=args.org,
        max_results=args.max_results, min_score=args.min_score,
    )
    if args.json:
        print(json.dumps({"ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
