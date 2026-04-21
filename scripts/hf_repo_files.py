#!/usr/bin/env python3
"""
hf_repo_files.py — File operations on HuggingFace repositories.

Ported from huggingface/ml-intern's hf_repo_files_tool.py. Operates on HF repos
(model/dataset/space) via huggingface_hub.

Subcommands:
  list    — list files recursively, with sizes
  read    — read a text file (uses hf_hub_download)
  upload  — upload content to a repo (optionally as a PR)
  delete  — delete files/folders by pattern (supports wildcards)

Requires HF_TOKEN in the environment.

Usage:
    hf_repo_files.py list --repo-id gpt2
    hf_repo_files.py read --repo-id gpt2 --path config.json
    hf_repo_files.py upload --repo-id me/model --path README.md --content-file ./README.md
    hf_repo_files.py upload --repo-id me/model --path fix.py --content-stdin --create-pr
    hf_repo_files.py delete --repo-id me/model --pattern '*.tmp' --pattern 'logs/'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    print("ERROR: huggingface_hub not installed.", file=sys.stderr)
    sys.exit(2)


def _repo_url(repo_id: str, repo_type: str = "model") -> str:
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}"
    return f"https://huggingface.co/{repo_type}s/{repo_id}"


def _fmt_size(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}PB"


async def _async(func, *a, **kw):
    return await asyncio.to_thread(func, *a, **kw)


async def cmd_list(args, api: HfApi) -> tuple[str, bool]:
    try:
        items = list(await _async(
            api.list_repo_tree,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            path_in_repo=args.path or "",
            recursive=True,
        ))
    except RepositoryNotFoundError:
        return f"Repository not found: {args.repo_id}", False
    except Exception as e:
        return f"Error: {e}", False

    if not items:
        return f"No files in {args.repo_id}", True

    lines: list[str] = []
    total = 0
    for item in sorted(items, key=lambda x: x.path):
        size = getattr(item, "size", None)
        if size:
            total += size
            lines.append(f"{item.path} ({_fmt_size(size)})")
        else:
            lines.append(f"{item.path}/")

    url = _repo_url(args.repo_id, args.repo_type)
    header = (
        f"**{args.repo_id}** ({len(items)} files, {_fmt_size(total)})\n"
        f"{url}/tree/{args.revision}\n\n"
    )
    return header + "\n".join(lines), True


async def cmd_read(args, api: HfApi) -> tuple[str, bool]:
    try:
        path = await _async(
            hf_hub_download,
            repo_id=args.repo_id, filename=args.path,
            repo_type=args.repo_type, revision=args.revision,
            token=api.token,
        )
    except (RepositoryNotFoundError, EntryNotFoundError) as e:
        return f"Not found: {e}", False
    except Exception as e:
        return f"Error: {e}", False

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        size = os.path.getsize(path)
        return f"Binary file ({_fmt_size(size)})", True
    truncated = len(content) > args.max_chars
    if truncated:
        content = content[:args.max_chars]
    url = f"{_repo_url(args.repo_id, args.repo_type)}/blob/{args.revision}/{args.path}"
    return (
        f"**{args.path}**{' (truncated)' if truncated else ''}\n{url}\n\n```\n{content}\n```",
        True,
    )


async def cmd_upload(args, api: HfApi) -> tuple[str, bool]:
    if args.content_file:
        with open(args.content_file, "rb") as f:
            content = f.read()
    elif args.content_stdin:
        content = sys.stdin.buffer.read()
    elif args.content is not None:
        content = args.content.encode("utf-8")
    else:
        return "ERROR: provide --content, --content-file, or --content-stdin.", False

    commit_message = args.commit_message or f"Upload {args.path}"
    try:
        result = await _async(
            api.upload_file,
            path_or_fileobj=content,
            path_in_repo=args.path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=commit_message,
            create_pr=args.create_pr,
        )
    except Exception as e:
        return f"Upload failed: {e}", False

    url = _repo_url(args.repo_id, args.repo_type)
    if args.create_pr and hasattr(result, "pr_url"):
        return f"**Uploaded as PR:** {result.pr_url}", True
    return f"**Uploaded:** {args.path}\n{url}/blob/{args.revision}/{args.path}", True


async def cmd_delete(args, api: HfApi) -> tuple[str, bool]:
    patterns = args.pattern or []
    if not patterns:
        return "ERROR: at least one --pattern is required.", False
    commit_message = args.commit_message or f"Delete {', '.join(patterns)}"
    try:
        await _async(
            api.delete_files,
            repo_id=args.repo_id,
            delete_patterns=patterns,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=commit_message,
            create_pr=args.create_pr,
        )
    except Exception as e:
        return f"Delete failed: {e}", False
    return f"**Deleted:** {', '.join(patterns)} from {args.repo_id}", True


CMDS: dict[str, Any] = {"list": cmd_list, "read": cmd_read, "upload": cmd_upload, "delete": cmd_delete}


def _add_common(p: argparse.ArgumentParser, *, needs_path: bool = False) -> None:
    p.add_argument("--repo-id", required=True, help="Repo ID, e.g. 'gpt2' or 'username/my-model'.")
    p.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    p.add_argument("--revision", default="main", help="Branch / tag / commit (default: main).")
    if needs_path:
        p.add_argument("--path", required=True, help="Path inside the repo.")


def main() -> int:
    parser = argparse.ArgumentParser(prog="hf_repo_files.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="List files in a repo.")
    _add_common(pl)
    pl.add_argument("--path", help="Subpath to list (optional).")

    pr = sub.add_parser("read", help="Read a text file.")
    _add_common(pr, needs_path=True)
    pr.add_argument("--max-chars", type=int, default=50000)

    pu = sub.add_parser("upload", help="Upload content to a repo.")
    _add_common(pu, needs_path=True)
    pu.add_argument("--content", help="Inline content string.")
    pu.add_argument("--content-file", help="Path to a local file with the content.")
    pu.add_argument("--content-stdin", action="store_true", help="Read content from stdin.")
    pu.add_argument("--create-pr", action="store_true")
    pu.add_argument("--commit-message")

    pd = sub.add_parser("delete", help="Delete files / folders by pattern.")
    _add_common(pd)
    pd.add_argument("--pattern", action="append", help="Glob pattern (repeatable).")
    pd.add_argument("--create-pr", action="store_true")
    pd.add_argument("--commit-message")

    for p in (pl, pr, pu, pd):
        p.add_argument("--json", action="store_true")

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        return 2
    api = HfApi(token=token)

    handler = CMDS[args.cmd]
    try:
        out, ok = asyncio.run(handler(args, api))
    except Exception as e:
        out, ok = f"Error in {args.cmd}: {e}", False

    if getattr(args, "json", False):
        print(json.dumps({"cmd": args.cmd, "ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
