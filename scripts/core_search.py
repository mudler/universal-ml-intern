#!/usr/bin/env python3
"""
core_search.py — Search academic papers via CORE (core.ac.uk) with full-text body search.

CORE (https://core.ac.uk) aggregates ~287M metadata records from open-access
repositories worldwide, with ~49M of those having full text indexed. Unlike
HuggingFace Papers (curated, ML-focused) and Semantic Scholar (metadata +
citations), CORE lets you **search the actual body text of papers** — not
just titles and abstracts. This is the right tool when you need to find
specific claims, methods, or datasets mentioned deep inside papers that
wouldn't surface from a title/abstract search.

API: https://api.core.ac.uk/v3 — requires an API key (free, get one at
https://core.ac.uk/services/api — set via CORE_API_KEY env var).

Rate limits (free tier): ~10 requests/second per key.

Why this exists alongside `papers.py` and `openalex.py`:

- `papers.py` — HF + arXiv + Semantic Scholar. Use first for ML papers.
- `openalex.py` — broad catalog (~250M works), strong at PDF extraction.
- `core_search.py` — full-text BODY search. Use when you need to find a
  specific phrase / method / dataset name that probably only appears inside
  the paper body, not in its abstract.

Subcommands:

  search        Metadata search (title/abstract). Returns titles + abstracts + URLs.
  full-text     Body-text search. Same query syntax but ranks on full-text match.
                Returns snippet-ish excerpts + download URLs.
  get           Fetch a single paper by CORE ID, optionally with full text.

Output format matches the other scripts: markdown to stdout by default,
JSON envelope with --json.

Usage:
  core_search.py search --query "LLM quantization MoE" --limit 5
  core_search.py full-text --query "adaptive scales per expert" --limit 5
  core_search.py get --id 123456789 --with-full-text
  core_search.py search --query "GRPO" --save-dir ./core_texts --with-full-text
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

import httpx

CORE_API = "https://api.core.ac.uk/v3"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("CORE_API_KEY")
    if not key:
        print(
            "ERROR: CORE_API_KEY not set. Get a free key at "
            "https://core.ac.uk/services/api and export it as CORE_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(2)
    return {"Authorization": f"Bearer {key}"}


async def _get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict | None:
    """GET with retry on 429/5xx."""
    url = f"{CORE_API}{path}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
        except httpx.RequestError as e:
            wait = 3 * (attempt + 1)
            print(f"Request error ({e}); retrying in {wait}s...", file=sys.stderr)
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"CORE rate limited, retrying in {wait}s...", file=sys.stderr)
            await asyncio.sleep(wait)
            continue
        if resp.status_code >= 500 and attempt < MAX_RETRIES - 1:
            await asyncio.sleep(3)
            continue
        if resp.status_code != 200:
            print(f"CORE API error (status {resp.status_code}): {resp.text[:200]}", file=sys.stderr)
            return None
        return resp.json()
    return None


def _normalize(work: dict) -> dict:
    """Flatten a CORE /search/works result into the shape we use throughout."""
    authors = [a.get("name", "") for a in (work.get("authors") or []) if a.get("name")]
    urls: list[str] = []
    download_url = work.get("downloadUrl") or ""
    if download_url:
        urls.append(download_url)
    for link in work.get("links") or []:
        u = link.get("url")
        if u and u not in urls:
            urls.append(u)
    doi = work.get("doi") or ""
    abstract = work.get("abstract") or ""
    full_text = work.get("fullText") or ""
    return {
        "id": str(work.get("id", "")),
        "title": work.get("title") or "Untitled",
        "authors": authors,
        "abstract": abstract,
        "full_text": full_text,
        "year": work.get("yearPublished") or work.get("publishedDate", "")[:4] or "?",
        "doi": doi,
        "download_url": download_url,
        "landing_page": (work.get("sourceFulltextUrls") or [None])[0] or (urls[0] if urls else ""),
        "language": (work.get("language") or {}).get("code", "") if isinstance(work.get("language"), dict) else "",
    }


# --- Formatting --------------------------------------------------------------

def _format_paper(i: int, p: dict, snippet_from_full_text: bool = False) -> str:
    lines = [f"## {i}. {p['title']}"]
    meta = [f"**id:** {p['id']}", f"**year:** {p['year']}"]
    if p.get("doi"):
        meta.append(f"**doi:** {p['doi']}")
    lines.append(" | ".join(meta))
    if p.get("landing_page"):
        lines.append(p["landing_page"])
    if p.get("download_url") and p["download_url"] != p.get("landing_page"):
        lines.append(f"**PDF:** {p['download_url']}")
    if p.get("authors"):
        names = ", ".join(p["authors"][:10])
        if len(p["authors"]) > 10:
            names += f" (+{len(p['authors']) - 10} more)"
        lines.append(f"**Authors:** {names}")
    abstract = p.get("abstract") or ""
    if abstract:
        lines.append(f"\n**Abstract:** {abstract[:500]}{'...' if len(abstract) > 500 else ''}")
    full_text = p.get("full_text") or ""
    if snippet_from_full_text and full_text:
        lines.append(f"\n**Full-text excerpt (first 600 chars):**\n{full_text[:600]}...")
    return "\n".join(lines)


def _format_results(title: str, papers: list[dict], *, full_text_mode: bool = False) -> str:
    if not papers:
        return f"No CORE results."
    lines = [f"# {title}", f"Showing {len(papers)} paper(s)\n"]
    for i, p in enumerate(papers, 1):
        lines.append(_format_paper(i, p, snippet_from_full_text=full_text_mode))
        lines.append("")
    return "\n".join(lines)


# --- Saving ------------------------------------------------------------------

def _save(p: dict, save_dir: str, with_full_text: bool) -> str:
    os.makedirs(save_dir, exist_ok=True)
    safe_id = re.sub(r"[^\w.\-]", "_", p["id"])
    path = os.path.join(save_dir, f"core_{safe_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Title: {p['title']}\n")
        f.write(f"Authors: {', '.join(p['authors'])}\n")
        f.write(f"Year: {p['year']}\n")
        if p.get("doi"):
            f.write(f"DOI: {p['doi']}\n")
        if p.get("landing_page"):
            f.write(f"URL: {p['landing_page']}\n")
        if with_full_text and p.get("full_text"):
            f.write(f"\nFull Text:\n{p['full_text']}\n")
        else:
            f.write(f"\nAbstract:\n{p.get('abstract') or '(no abstract)'}\n")
    return path


# --- Subcommand handlers -----------------------------------------------------

async def cmd_search(args) -> int:
    """Metadata search — title/abstract match."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, "/search/works", {
            "q": args.query,
            "limit": args.limit,
            # Metadata search mode: do NOT require full-text presence
        })
        if not data:
            return 1
        results = data.get("results") or []
        papers = [_normalize(r) for r in results]

        saved: list[str] = []
        if args.save_dir:
            for p in papers:
                saved.append(_save(p, args.save_dir, with_full_text=args.with_full_text))

    if args.json:
        out = {"ok": bool(papers), "query": args.query, "count": len(papers), "papers": papers}
        if saved:
            out["saved"] = saved
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(_format_results(f"CORE search: '{args.query}'", papers))
        if saved:
            print(f"\nSaved {len(saved)} file(s) to {args.save_dir}.")
    return 0 if papers else 1


async def cmd_full_text(args) -> int:
    """Full-text body search — CORE filters to papers that have full text indexed
    and ranks by body relevance. We also request the fullText field back."""
    async with httpx.AsyncClient() as client:
        # Restrict to works that have full text indexed
        q = f"({args.query}) AND _exists_:fullText"
        data = await _get(client, "/search/works", {
            "q": q,
            "limit": args.limit,
        })
        if not data:
            return 1
        results = data.get("results") or []
        papers = [_normalize(r) for r in results]

        saved: list[str] = []
        if args.save_dir:
            for p in papers:
                saved.append(_save(p, args.save_dir, with_full_text=True))

    if args.json:
        out = {"ok": bool(papers), "query": args.query, "count": len(papers), "papers": papers}
        if saved:
            out["saved"] = saved
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(_format_results(f"CORE full-text search: '{args.query}'", papers, full_text_mode=True))
        if saved:
            print(f"\nSaved {len(saved)} file(s) to {args.save_dir}.")
    return 0 if papers else 1


async def cmd_get(args) -> int:
    """Fetch a single paper by its CORE ID."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"/works/{args.id}")
        if not data:
            return 1
        paper = _normalize(data)

        saved_path: str | None = None
        if args.save_dir:
            saved_path = _save(paper, args.save_dir, with_full_text=args.with_full_text)

    if args.json:
        out = {"ok": True, "paper": paper}
        if saved_path:
            out["saved"] = saved_path
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(_format_paper(1, paper, snippet_from_full_text=args.with_full_text))
        if saved_path:
            print(f"\nSaved to {saved_path}.")
    return 0


# --- Main --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        prog="core_search.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="Metadata (title/abstract) search.")
    ps.add_argument("--query", "-q", required=True)
    ps.add_argument("--limit", "-l", type=int, default=10)
    ps.add_argument("--save-dir", metavar="DIR",
                    help="Save each paper as a .txt to this directory.")
    ps.add_argument("--with-full-text", action="store_true",
                    help="Include the fullText field in saved files (when present).")
    ps.add_argument("--json", action="store_true")

    pf = sub.add_parser("full-text", help="Body-text search; only returns papers with full text.")
    pf.add_argument("--query", "-q", required=True)
    pf.add_argument("--limit", "-l", type=int, default=10)
    pf.add_argument("--save-dir", metavar="DIR")
    pf.add_argument("--json", action="store_true")

    pg = sub.add_parser("get", help="Fetch a single paper by CORE ID.")
    pg.add_argument("--id", required=True, help="CORE work ID (e.g. 123456789).")
    pg.add_argument("--save-dir", metavar="DIR")
    pg.add_argument("--with-full-text", action="store_true",
                    help="Include fullText in the saved file (when present).")
    pg.add_argument("--json", action="store_true")

    args = p.parse_args()

    handler = {"search": cmd_search, "full-text": cmd_full_text, "get": cmd_get}[args.cmd]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
