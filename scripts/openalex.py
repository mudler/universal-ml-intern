#!/usr/bin/env python3
"""
openalex.py — Search academic papers via OpenAlex, optionally download PDFs
and extract full text.

OpenAlex (https://openalex.org) is a free, open catalog of ~250M scholarly
works with no authentication required and generous public rate limits
(~10 req/s when you include a `mailto=` parameter for the "polite pool").

Why this exists alongside `papers.py`:

- `papers.py` is the rich entry point — HuggingFace's curated daily papers,
  arXiv HTML parsing, Semantic Scholar citation graphs, snippet search. Use
  it first.
- `openalex.py` is the broad-catalog fallback. It covers papers HF hasn't
  indexed, sidesteps Semantic Scholar rate limits, and can download PDFs +
  extract text inline — useful for older or long-tail papers.

This script is a near-verbatim port of autoresearch-quant's `paper_search.py`,
adapted to this repo's conventions (async httpx, markdown output, --json
envelope, shared flag style).

Usage:
  openalex.py --query "mixture of experts quantization" --limit 5
  openalex.py --query "LLM quantization" --pdf --save-dir ./texts
  openalex.py --query "GPQA" --include-closed --limit 3

Subcommands are single-action; everything fits in one invocation.

Output:
  By default, formatted markdown to stdout.
  With --save-dir, each paper is written as a .txt file (abstract or full
  text per --pdf) to that directory with a sanitized-ID filename.
  With --json, emits {"ok": bool, "papers": [{...}]} for easy piping.

Rate-limiting:
  OpenAlex honors a "polite pool" for requests that include a mailto=
  query parameter. We send a generic mailto automatically — set
  OPENALEX_MAILTO in the environment to use your own email address
  (raises your per-IP quota in practice).

PDF extraction:
  --pdf downloads the best open-access PDF and runs pypdf over it. pypdf is
  already in the repo's dependencies. If extraction fails (scanned PDF,
  missing OA copy, paywall), the abstract is used as a fallback and a warning
  is printed to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import sys
import time
from typing import Any

import httpx

OPENALEX_API = "https://api.openalex.org/works"
DEFAULT_MAILTO = os.environ.get("OPENALEX_MAILTO", "universal-ml-intern@users.noreply.github.com")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index (word → positions).
    Rebuild the linear text."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _extract_id(work: dict) -> str:
    """Pick a short, filename-safe ID — arXiv ID if we can find one in a
    landing page URL, otherwise the OpenAlex suffix (Wxxxxxxxxx)."""
    for loc in work.get("locations", []) or []:
        landing = (loc.get("landing_page_url") or "") or ""
        if "arxiv.org" in landing:
            m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", landing)
            if m:
                return m.group(0)
    openalex_id = (work.get("ids") or {}).get("openalex", "")
    if openalex_id:
        return openalex_id.rsplit("/", 1)[-1]
    return (work.get("id") or "unknown").rsplit("/", 1)[-1]


async def _search_works(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    oa_only: bool,
) -> list[dict]:
    """Hit OpenAlex /works with retries on transient errors."""
    filters = ["type:article"]
    if oa_only:
        filters.append("open_access.is_oa:true")

    params = {
        "search": query,
        "per_page": limit,
        "sort": "relevance_score:desc",
        "filter": ",".join(filters),
        "mailto": DEFAULT_MAILTO,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(OPENALEX_API, params=params, timeout=REQUEST_TIMEOUT)
        except httpx.RequestError as e:
            wait = 3 * (attempt + 1)
            print(f"Request failed ({e}), retrying in {wait}s...", file=sys.stderr)
            await asyncio.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = 3 * (attempt + 1)
            print(f"Rate limited by OpenAlex, retrying in {wait}s...", file=sys.stderr)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json().get("results", [])

    print("OpenAlex API failed after retries.", file=sys.stderr)
    return []


def _normalize_work(work: dict) -> dict:
    """Flatten OpenAlex's nested work object into the dict shape we use
    throughout this script."""
    oa_loc = work.get("best_oa_location") or {}
    pdf_url = oa_loc.get("pdf_url")
    url = oa_loc.get("landing_page_url") or work.get("doi") or work.get("id", "")
    authors = [
        (a.get("author") or {}).get("display_name", "Unknown")
        for a in (work.get("authorships") or [])
    ]
    return {
        "id": _extract_id(work),
        "title": work.get("display_name") or "Untitled",
        "authors": authors,
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")) or "No abstract available.",
        "url": url,
        "pdf_url": pdf_url,
        "date": work.get("publication_date") or "unknown",
        "cited_by_count": work.get("cited_by_count") or 0,
    }


async def _download_pdf(client: httpx.AsyncClient, pdf_url: str) -> bytes | None:
    """Download a PDF; return bytes if the response is actually a PDF, None otherwise."""
    try:
        resp = await client.get(
            pdf_url,
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": "universal-ml-intern/0.1"},
        )
        resp.raise_for_status()
        if b"%PDF" in resp.content[:1024]:
            return resp.content
        print(f"  Warning: URL did not return a PDF: {pdf_url}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"  Warning: PDF download failed ({e})", file=sys.stderr)
    return None


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Best-effort text extraction. pypdf ships in this repo's deps."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""
        if txt:
            pages.append(txt)
    return "\n\n".join(pages)


# --- Formatting --------------------------------------------------------------

def _format_paper(i: int, paper: dict) -> str:
    lines = [f"## {i}. {paper['title']}"]
    meta = [f"**id:** {paper['id']}"]
    if paper.get("cited_by_count"):
        meta.append(f"**cited_by:** {paper['cited_by_count']:,}")
    meta.append(f"**date:** {paper['date']}")
    lines.append(" | ".join(meta))
    if paper.get("url"):
        lines.append(paper["url"])
    if paper.get("pdf_url"):
        lines.append(f"**PDF:** {paper['pdf_url']}")
    if paper.get("authors"):
        names = ", ".join(paper["authors"][:10])
        if len(paper["authors"]) > 10:
            names += f" (+{len(paper['authors']) - 10} more)"
        lines.append(f"**Authors:** {names}")
    abstract = paper.get("abstract") or ""
    if abstract:
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(f"\n**Abstract:** {abstract}")
    return "\n".join(lines)


def _format_results(query: str, papers: list[dict]) -> str:
    if not papers:
        return f"No OpenAlex results for '{query}'."
    lines = [f"# OpenAlex results for '{query}'", f"Showing {len(papers)} paper(s)\n"]
    for i, p in enumerate(papers, 1):
        lines.append(_format_paper(i, p))
        lines.append("")
    return "\n".join(lines)


# --- Saving ------------------------------------------------------------------

def _save_paper(paper: dict, save_dir: str, full_text: str | None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    safe_id = re.sub(r"[^\w.\-]", "_", paper["id"])
    path = os.path.join(save_dir, f"{safe_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Title: {paper['title']}\n")
        f.write(f"Authors: {', '.join(paper['authors'])}\n")
        f.write(f"URL: {paper['url']}\n")
        f.write(f"Date: {paper['date']}\n")
        f.write(f"Cited by: {paper.get('cited_by_count', 0)}\n")
        if full_text:
            f.write(f"\nFull Text:\n{full_text}\n")
        else:
            f.write(f"\nAbstract:\n{paper['abstract']}\n")
    return path


# --- Main --------------------------------------------------------------------

async def _run(args) -> int:
    oa_only = not args.include_closed

    async with httpx.AsyncClient() as client:
        works = await _search_works(client, args.query, args.limit, oa_only)
        papers = [_normalize_work(w) for w in works]

        saved_paths: list[str] = []
        fulltext_by_id: dict[str, str] = {}

        if args.pdf or args.save_dir:
            for paper in papers:
                full_text: str | None = None
                if args.pdf and paper.get("pdf_url"):
                    print(f"  Downloading PDF: {paper['id']}...", file=sys.stderr)
                    pdf_bytes = await _download_pdf(client, paper["pdf_url"])
                    if pdf_bytes:
                        try:
                            full_text = _pdf_to_text(pdf_bytes)
                            fulltext_by_id[paper["id"]] = full_text
                            print(f"    extracted {len(full_text)} chars", file=sys.stderr)
                        except Exception as e:
                            print(f"    pypdf extraction failed: {e}", file=sys.stderr)
                    else:
                        print(f"    PDF unavailable, using abstract", file=sys.stderr)

                if args.save_dir:
                    path = _save_paper(paper, args.save_dir, full_text)
                    saved_paths.append(path)

    if args.json:
        out = {
            "ok": bool(papers),
            "query": args.query,
            "count": len(papers),
            "papers": papers,
        }
        if saved_paths:
            out["saved"] = saved_paths
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(_format_results(args.query, papers))
        if saved_paths:
            print(f"\nSaved {len(saved_paths)} file(s) to {args.save_dir}:")
            for p in saved_paths:
                print(f"  {p}")

    return 0 if papers else 1


def main() -> int:
    p = argparse.ArgumentParser(
        prog="openalex.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query", "-q", required=True,
                   help="Search query (free text).")
    p.add_argument("--limit", "-l", type=int, default=10,
                   help="Max results (default: 10, API cap: 200 per page).")
    p.add_argument("--include-closed", action="store_true",
                   help="Include non-open-access papers (default: OA only).")
    p.add_argument("--pdf", "-p", action="store_true",
                   help="Download best-OA PDFs and extract full text (requires pypdf).")
    p.add_argument("--save-dir", metavar="DIR",
                   help="Save each paper as a .txt in this directory "
                        "(with full text if --pdf, else abstract).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON envelope.")
    args = p.parse_args()

    if args.limit < 1 or args.limit > 200:
        print("ERROR: --limit must be between 1 and 200.", file=sys.stderr)
        return 2

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
