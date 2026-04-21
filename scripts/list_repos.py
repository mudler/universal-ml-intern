#!/usr/bin/env python3
"""
list_repos.py — List and sort GitHub repositories for a user or organization.

Ported from huggingface/ml-intern's github_list_repos.py.

Requires GITHUB_TOKEN in the environment.

Usage:
    list_repos.py --owner huggingface --sort stars --limit 10
    list_repos.py --owner karpathy --owner-type user --sort updated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


def list_repos(
    owner: str,
    owner_type: str = "org",
    sort: str = "stars",
    order: str = "desc",
    limit: int | None = 30,
) -> tuple[str, bool]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "ERROR: GITHUB_TOKEN environment variable is required.", False

    url = (
        f"https://api.github.com/orgs/{owner}/repos" if owner_type == "org"
        else f"https://api.github.com/users/{owner}/repos"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    api_sort_map = {"created": "created", "updated": "updated", "stars": None, "forks": None}
    api_sort = api_sort_map.get(sort)
    need_manual_sort = api_sort is None

    all_repos: list[dict[str, Any]] = []
    page = 1
    per_page = 100

    try:
        while True:
            params: dict[str, Any] = {"page": page, "per_page": per_page}
            if api_sort:
                params["sort"] = api_sort
                params["direction"] = order
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 403:
                data = response.json()
                return f"GitHub API rate limit / permission error: {data.get('message', 'Unknown')}", False
            if response.status_code != 200:
                msg = f"GitHub API error (status {response.status_code})"
                try:
                    msg += f": {response.json().get('message', '')}"
                except Exception:
                    pass
                return msg, False
            items = response.json()
            if not items:
                break
            for item in items:
                all_repos.append({
                    "name": item.get("name"),
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "html_url": item.get("html_url"),
                    "language": item.get("language"),
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "open_issues": item.get("open_issues_count", 0),
                    "topics": item.get("topics", []),
                    "updated_at": item.get("updated_at"),
                    "created_at": item.get("created_at"),
                })
            if len(items) < per_page:
                break
            if limit and len(all_repos) >= limit:
                break
            page += 1
    except requests.exceptions.RequestException as e:
        return f"Failed to connect to GitHub API: {e}", False

    if need_manual_sort and all_repos:
        reverse = order == "desc"
        all_repos.sort(key=lambda x: x[sort], reverse=reverse)
    if limit:
        all_repos = all_repos[:limit]

    if not all_repos:
        return f"No repositories found for {owner_type} '{owner}'", True

    lines = [f"**Found {len(all_repos)} repositories for {owner}:**\n"]
    for i, repo in enumerate(all_repos, 1):
        lines.append(f"{i}. **{repo['full_name']}**")
        lines.append(
            f"   ⭐ {repo['stars']:,} stars | 🍴 {repo['forks']:,} forks | "
            f"Language: {repo['language'] or 'N/A'}"
        )
        desc = repo["description"]
        if desc:
            if len(desc) > 100:
                desc = desc[:100] + "..."
            lines.append(f"   {desc}")
        lines.append(f"   URL: {repo['html_url']}")
        if repo["topics"]:
            lines.append(f"   Topics: {', '.join(repo['topics'][:5])}")
        lines.append(f"   find_examples.py --org {owner} --repo {repo['name']}")
        lines.append("")

    return "\n".join(lines), True


def main() -> int:
    p = argparse.ArgumentParser(prog="list_repos.py", description=__doc__)
    p.add_argument("--owner", required=True, help="GitHub username or org name.")
    p.add_argument("--owner-type", choices=["user", "org"], default="org")
    p.add_argument("--sort", choices=["stars", "forks", "updated", "created"], default="stars")
    p.add_argument("--order", choices=["asc", "desc"], default="desc")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--json", action="store_true", help="Emit JSON envelope.")
    args = p.parse_args()

    out, ok = list_repos(
        owner=args.owner, owner_type=args.owner_type,
        sort=args.sort, order=args.order, limit=args.limit,
    )
    if args.json:
        print(json.dumps({"ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
