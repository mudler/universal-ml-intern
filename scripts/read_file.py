#!/usr/bin/env python3
"""
read_file.py — Read file contents from a GitHub repository.

Ported from huggingface/ml-intern's github_read_file.py. Supports:
  - Arbitrary repos and refs (branch / tag / commit SHA).
  - Line-range extraction (--line-start, --line-end).
  - Automatic .ipynb → Markdown conversion (outputs stripped).

Requires GITHUB_TOKEN in the environment.

Usage:
    read_file.py --repo huggingface/trl --path examples/scripts/sft.py
    read_file.py --repo huggingface/transformers --path src/transformers/models/llama/modeling_llama.py \\
                 --line-start 100 --line-end 200
    read_file.py --repo huggingface/cookbook --path notebooks/en/rag.ipynb
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Any

import requests


def _convert_ipynb_to_markdown(content: str) -> str:
    try:
        import nbformat
        from nbconvert import MarkdownExporter
        from nbconvert.preprocessors import ClearOutputPreprocessor, TagRemovePreprocessor
    except ImportError:
        return content

    try:
        nb_dict = json.loads(content)
        if "cells" in nb_dict:
            for cell in nb_dict["cells"]:
                src = cell.get("source")
                if isinstance(src, list):
                    cell["source"] = "".join(src)
        nb = nbformat.reads(json.dumps(nb_dict), as_version=4)
        clear = ClearOutputPreprocessor()
        nb, _ = clear.preprocess(nb, {})
        remove = TagRemovePreprocessor(
            remove_cell_tags={"hide", "hidden", "remove"},
            remove_input_tags=set(),
            remove_all_outputs_tags=set(),
        )
        nb, _ = remove.preprocess(nb, {})
        exporter = MarkdownExporter()
        markdown, _ = exporter.from_notebook_node(nb)
        return markdown
    except (json.JSONDecodeError, Exception):
        return content


def read_file(
    repo: str,
    path: str,
    ref: str = "HEAD",
    line_start: int | None = None,
    line_end: int | None = None,
) -> tuple[str, bool]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "ERROR: GITHUB_TOKEN environment variable is required.", False
    if "/" not in repo:
        return "ERROR: --repo must be in format 'owner/repo'.", False

    owner, repo_name = repo.split("/", 1)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    params: dict[str, Any] = {}
    if ref and ref != "HEAD":
        params["ref"] = ref

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 404:
            return f"File not found: {path} in {repo} (ref: {ref})", False
        if response.status_code != 200:
            msg = f"GitHub API error (status {response.status_code})"
            try:
                msg += f": {response.json().get('message', '')}"
            except Exception:
                pass
            return msg, False

        data = response.json()
        if data.get("type") != "file":
            return f"Path {path} is not a file (type: {data.get('type')})", False

        content_b64 = data.get("content", "")
        if content_b64:
            content_b64 = content_b64.replace("\n", "").replace(" ", "")
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        else:
            raw_headers = {
                "Accept": "application/vnd.github.raw",
                "X-GitHub-Api-Version": "2022-11-28",
                "Authorization": f"Bearer {token}",
            }
            raw_resp = requests.get(url, headers=raw_headers, params=params, timeout=30)
            if raw_resp.status_code != 200:
                return "Failed to fetch file content (raw fallback).", False
            content = raw_resp.text

        if path.lower().endswith(".ipynb"):
            content = _convert_ipynb_to_markdown(content)

        lines = content.split("\n")
        total_lines = len(lines)
        truncated = False

        if line_start is None and line_end is None:
            if total_lines > 300:
                line_start = 1
                line_end = 300
                truncated = True
            else:
                line_start = 1
                line_end = total_lines
        else:
            if line_start is None:
                line_start = 1
            if line_end is None:
                line_end = total_lines
            line_start = max(1, line_start)
            line_end = min(total_lines, line_end)
            if line_start > line_end:
                return f"Invalid range: line_start ({line_start}) > line_end ({line_end})", False

        selected = lines[line_start - 1:line_end]
        selected_content = "\n".join(selected)

        out_lines = [f"**Reading file from repo: {repo}, path: {path}**"]
        if ref and ref != "HEAD":
            out_lines.append(f"Ref: {ref}")
        out_lines.append("\n**File content:**")
        out_lines.append("```")
        out_lines.append(selected_content)
        out_lines.append("```")
        if truncated:
            out_lines.append(
                f"Showing lines {line_start}-{line_end} of {total_lines}. "
                "Use --line-start / --line-end to view more."
            )
        return "\n".join(out_lines), True
    except requests.exceptions.RequestException as e:
        return f"Failed to connect to GitHub API: {e}", False


def main() -> int:
    p = argparse.ArgumentParser(prog="read_file.py", description=__doc__)
    p.add_argument("--repo", required=True, help="'owner/repo' format, e.g. 'huggingface/trl'.")
    p.add_argument("--path", required=True, help="Path inside the repo, e.g. 'examples/scripts/sft.py'.")
    p.add_argument("--ref", default="HEAD", help="Branch / tag / commit SHA (default: HEAD).")
    p.add_argument("--line-start", type=int)
    p.add_argument("--line-end", type=int)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    out, ok = read_file(
        repo=args.repo, path=args.path, ref=args.ref,
        line_start=args.line_start, line_end=args.line_end,
    )
    if args.json:
        print(json.dumps({"ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
