#!/usr/bin/env python3
"""
hf_repo_git.py — Git-like operations on HuggingFace repositories.

Ported from huggingface/ml-intern's hf_repo_git_tool.py. Wraps
huggingface_hub.HfApi branch/tag/PR/repo endpoints.

Subcommands (grouped):
  Branches:  create-branch, delete-branch, list-refs
  Tags:      create-tag, delete-tag
  PRs:       create-pr, list-prs, get-pr, merge-pr, close-pr, comment-pr, change-pr-status
  Repo:      create-repo, update-repo

PR workflow:
  1. create-pr  → creates a DRAFT PR (empty by default)
  2. Upload files via `hf_repo_files.py upload --revision refs/pr/N` to add commits
  3. change-pr-status --new-status open  (to publish the draft)
  4. merge-pr when ready

Destructive ops (delete-branch, delete-tag, merge-pr, create-repo, update-repo)
should be explicitly confirmed by the user — this script does NOT implement
extra safeguards; the operator is expected to gate them at the harness layer.

Requires HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("ERROR: huggingface_hub not installed.", file=sys.stderr)
    sys.exit(2)


def _repo_url(repo_id: str, repo_type: str = "model") -> str:
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}"
    return f"https://huggingface.co/{repo_type}s/{repo_id}"


async def _async(func, *a, **kw):
    return await asyncio.to_thread(func, *a, **kw)


# --- Branches ----------------------------------------------------------------

async def cmd_create_branch(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.create_branch, repo_id=args.repo_id, branch=args.branch,
                 revision=args.from_rev, repo_type=args.repo_type, exist_ok=args.exist_ok)
    return f"**Branch created:** {args.branch}\n{_repo_url(args.repo_id, args.repo_type)}/tree/{args.branch}", True


async def cmd_delete_branch(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.delete_branch, repo_id=args.repo_id, branch=args.branch, repo_type=args.repo_type)
    return f"**Branch deleted:** {args.branch}", True


async def cmd_list_refs(args, api: HfApi) -> tuple[str, bool]:
    refs = await _async(api.list_repo_refs, repo_id=args.repo_id, repo_type=args.repo_type)
    branches = [b.name for b in (refs.branches or [])]
    tags = [t.name for t in (getattr(refs, "tags", None) or [])]
    lines = [f"**{args.repo_id}**", _repo_url(args.repo_id, args.repo_type), ""]
    lines.append(f"**Branches ({len(branches)}):** " + (", ".join(branches) if branches else "none"))
    lines.append(f"**Tags ({len(tags)}):** " + (", ".join(tags) if tags else "none"))
    return "\n".join(lines), True


# --- Tags --------------------------------------------------------------------

async def cmd_create_tag(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.create_tag, repo_id=args.repo_id, tag=args.tag,
                 revision=args.revision, tag_message=args.tag_message or "",
                 repo_type=args.repo_type, exist_ok=args.exist_ok)
    return f"**Tag created:** {args.tag}\n{_repo_url(args.repo_id, args.repo_type)}/tree/{args.tag}", True


async def cmd_delete_tag(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.delete_tag, repo_id=args.repo_id, tag=args.tag, repo_type=args.repo_type)
    return f"**Tag deleted:** {args.tag}", True


# --- PRs ---------------------------------------------------------------------

async def cmd_create_pr(args, api: HfApi) -> tuple[str, bool]:
    res = await _async(api.create_pull_request, repo_id=args.repo_id, title=args.title,
                       description=args.description or "", repo_type=args.repo_type)
    url = f"{_repo_url(args.repo_id, args.repo_type)}/discussions/{res.num}"
    return (
        f"**Draft PR #{res.num} created:** {args.title}\n{url}\n\n"
        f"Add commits via hf_repo_files.py upload with --revision refs/pr/{res.num}",
        True,
    )


async def cmd_list_prs(args, api: HfApi) -> tuple[str, bool]:
    status = None if args.status == "all" else args.status
    discussions = list(api.get_repo_discussions(
        repo_id=args.repo_id, repo_type=args.repo_type, discussion_status=status,
    ))
    if not discussions:
        return f"No discussions in {args.repo_id}", True
    url = _repo_url(args.repo_id, args.repo_type)
    lines = [f"**{args.repo_id}** - {len(discussions)} discussions", f"{url}/discussions", ""]
    for d in discussions[:20]:
        tag = {"draft": "[DRAFT]", "open": "[OPEN]", "merged": "[MERGED]"}.get(d.status, "[CLOSED]")
        kind = "PR" if d.is_pull_request else "D"
        lines.append(f"{tag} #{d.num} [{kind}] {d.title}")
    return "\n".join(lines), True


async def cmd_get_pr(args, api: HfApi) -> tuple[str, bool]:
    pr = await _async(api.get_discussion_details, repo_id=args.repo_id,
                      discussion_num=args.pr_num, repo_type=args.repo_type)
    url = f"{_repo_url(args.repo_id, args.repo_type)}/discussions/{args.pr_num}"
    kind = "Pull Request" if pr.is_pull_request else "Discussion"
    lines = [f"**{kind} #{args.pr_num}:** {pr.title}", f"**Status:** {pr.status}",
             f"**Author:** {pr.author}", url]
    if pr.is_pull_request and pr.status in ("draft", "open"):
        lines.append(f"\nTo add commits: upload with --revision refs/pr/{args.pr_num}")
    return "\n".join(lines), True


async def cmd_merge_pr(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.merge_pull_request, repo_id=args.repo_id,
                 discussion_num=args.pr_num, comment=args.comment or "",
                 repo_type=args.repo_type)
    url = f"{_repo_url(args.repo_id, args.repo_type)}/discussions/{args.pr_num}"
    return f"**PR #{args.pr_num} merged**\n{url}", True


async def cmd_close_pr(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.change_discussion_status, repo_id=args.repo_id,
                 discussion_num=args.pr_num, new_status="closed",
                 comment=args.comment or "", repo_type=args.repo_type)
    return f"**Discussion #{args.pr_num} closed**", True


async def cmd_comment_pr(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.comment_discussion, repo_id=args.repo_id,
                 discussion_num=args.pr_num, comment=args.comment,
                 repo_type=args.repo_type)
    url = f"{_repo_url(args.repo_id, args.repo_type)}/discussions/{args.pr_num}"
    return f"**Comment added to #{args.pr_num}**\n{url}", True


async def cmd_change_pr_status(args, api: HfApi) -> tuple[str, bool]:
    await _async(api.change_discussion_status, repo_id=args.repo_id,
                 discussion_num=args.pr_num, new_status=args.new_status,
                 comment=args.comment or "", repo_type=args.repo_type)
    url = f"{_repo_url(args.repo_id, args.repo_type)}/discussions/{args.pr_num}"
    return f"**PR #{args.pr_num} status → {args.new_status}**\n{url}", True


# --- Repo management ---------------------------------------------------------

async def cmd_create_repo(args, api: HfApi) -> tuple[str, bool]:
    if args.repo_type == "space" and not args.space_sdk:
        return "ERROR: --space-sdk is required for spaces (gradio/streamlit/docker/static).", False
    kwargs: dict[str, Any] = {
        "repo_id": args.repo_id, "repo_type": args.repo_type,
        "private": args.private, "exist_ok": args.exist_ok,
    }
    if args.space_sdk:
        kwargs["space_sdk"] = args.space_sdk
    result = await _async(api.create_repo, **kwargs)
    return f"**Repository created:** {args.repo_id}\n**Private:** {args.private}\n{result}", True


async def cmd_update_repo(args, api: HfApi) -> tuple[str, bool]:
    if args.private is None and args.gated is None:
        return "ERROR: specify --private or --gated.", False
    kwargs: dict[str, Any] = {"repo_id": args.repo_id, "repo_type": args.repo_type}
    if args.private is not None:
        kwargs["private"] = args.private
    if args.gated is not None:
        # "false" string means disable gating
        kwargs["gated"] = False if args.gated == "false" else args.gated
    await _async(api.update_repo_settings, **kwargs)
    changes = []
    if args.private is not None:
        changes.append(f"private={args.private}")
    if args.gated is not None:
        changes.append(f"gated={args.gated}")
    return f"**Settings updated:** {', '.join(changes)}\n{_repo_url(args.repo_id, args.repo_type)}/settings", True


CMDS = {
    "create-branch": cmd_create_branch, "delete-branch": cmd_delete_branch, "list-refs": cmd_list_refs,
    "create-tag": cmd_create_tag, "delete-tag": cmd_delete_tag,
    "create-pr": cmd_create_pr, "list-prs": cmd_list_prs, "get-pr": cmd_get_pr,
    "merge-pr": cmd_merge_pr, "close-pr": cmd_close_pr, "comment-pr": cmd_comment_pr,
    "change-pr-status": cmd_change_pr_status,
    "create-repo": cmd_create_repo, "update-repo": cmd_update_repo,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="hf_repo_git.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Helpers to reduce repetition
    def repo_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-id", required=True)
        p.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")

    cb = sub.add_parser("create-branch"); repo_args(cb)
    cb.add_argument("--branch", required=True); cb.add_argument("--from-rev", default="main")
    cb.add_argument("--exist-ok", action="store_true")

    db = sub.add_parser("delete-branch"); repo_args(db)
    db.add_argument("--branch", required=True)

    lr = sub.add_parser("list-refs"); repo_args(lr)

    ct = sub.add_parser("create-tag"); repo_args(ct)
    ct.add_argument("--tag", required=True); ct.add_argument("--revision", default="main")
    ct.add_argument("--tag-message"); ct.add_argument("--exist-ok", action="store_true")

    dt = sub.add_parser("delete-tag"); repo_args(dt); dt.add_argument("--tag", required=True)

    cp = sub.add_parser("create-pr"); repo_args(cp)
    cp.add_argument("--title", required=True); cp.add_argument("--description")

    lp = sub.add_parser("list-prs"); repo_args(lp)
    lp.add_argument("--status", choices=["open", "closed", "all"], default="all")

    gp = sub.add_parser("get-pr"); repo_args(gp); gp.add_argument("--pr-num", type=int, required=True)

    mp = sub.add_parser("merge-pr"); repo_args(mp)
    mp.add_argument("--pr-num", type=int, required=True); mp.add_argument("--comment")

    clp = sub.add_parser("close-pr"); repo_args(clp)
    clp.add_argument("--pr-num", type=int, required=True); clp.add_argument("--comment")

    mtp = sub.add_parser("comment-pr"); repo_args(mtp)
    mtp.add_argument("--pr-num", type=int, required=True)
    mtp.add_argument("--comment", required=True)

    cps = sub.add_parser("change-pr-status"); repo_args(cps)
    cps.add_argument("--pr-num", type=int, required=True)
    cps.add_argument("--new-status", choices=["open", "closed"], required=True)
    cps.add_argument("--comment")

    cr = sub.add_parser("create-repo"); repo_args(cr)
    cr.add_argument("--private", action="store_true", default=True)
    cr.add_argument("--public", dest="private", action="store_false")
    cr.add_argument("--space-sdk", choices=["gradio", "streamlit", "docker", "static"])
    cr.add_argument("--exist-ok", action="store_true")

    ur = sub.add_parser("update-repo"); repo_args(ur)
    ur.add_argument("--private", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                    help="true / false")
    ur.add_argument("--gated", choices=["auto", "manual", "false"], default=None)

    for p in sub.choices.values():
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
    except RepositoryNotFoundError:
        out, ok = f"Repository not found: {args.repo_id}", False
    except Exception as e:
        out, ok = f"Error in {args.cmd}: {e}", False

    if getattr(args, "json", False):
        print(json.dumps({"cmd": args.cmd, "ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
