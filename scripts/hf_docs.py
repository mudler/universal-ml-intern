#!/usr/bin/env python3
"""
hf_docs.py — Explore and fetch HuggingFace library documentation.

Ported (simplified) from huggingface/ml-intern's docs_tools.py. Two subcommands:

  explore  — list pages in a doc set (e.g. 'trl', 'transformers', 'datasets'),
             optionally filtered by a keyword query.
  fetch    — fetch the full markdown content of a specific doc page URL.

Composite endpoints are supported (e.g. 'optimum' expands to optimum-habana,
optimum-neuron, optimum-intel, …; 'courses' expands to every HF course repo).

Search uses Whoosh full-text when installed (install via `pip install -e '.[docs]'`),
falling back to substring matching otherwise.

Requires HF_TOKEN in the environment.

Usage:
    hf_docs.py explore --endpoint trl --query sft
    hf_docs.py explore --endpoint transformers --query 'flash attention'
    hf_docs.py fetch --url https://huggingface.co/docs/trl/sft_trainer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx
from bs4 import BeautifulSoup

DEFAULT_MAX_RESULTS = 20
MAX_RESULTS_CAP = 50

COMPOSITE_ENDPOINTS: dict[str, list[str]] = {
    "optimum": [
        "optimum", "optimum-habana", "optimum-neuron",
        "optimum-intel", "optimum-executorch", "optimum-tpu",
    ],
    "courses": [
        "llm-course", "robotics-course", "mcp-course", "smol-course",
        "agents-course", "deep-rl-course", "computer-vision-course",
        "audio-course", "ml-games-course", "diffusion-course",
        "ml-for-3d-course", "cookbook",
    ],
}

_docs_cache: dict[str, list[dict[str, str]]] = {}
_cache_lock = asyncio.Lock()


async def _fetch_endpoint_docs(hf_token: str, endpoint: str) -> list[dict[str, str]]:
    url = f"https://huggingface.co/docs/{endpoint}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        sidebar = soup.find("nav", class_=lambda x: x and "flex-auto" in x)
        if not sidebar:
            raise ValueError(f"Could not find navigation sidebar for '{endpoint}'")

        nav_items = []
        for link in sidebar.find_all("a", href=True):
            href = link["href"]
            page_url = f"https://huggingface.co{href}" if href.startswith("/") else href
            nav_items.append({"title": link.get_text(strip=True), "url": page_url})
        if not nav_items:
            raise ValueError(f"No navigation links found for '{endpoint}'")

        async def fetch_page(item: dict[str, str]) -> dict[str, str]:
            md_url = f"{item['url']}.md"
            try:
                r = await client.get(md_url, headers=headers)
                r.raise_for_status()
                content = r.text.strip()
                glimpse = content[:200] + "..." if len(content) > 200 else content
            except Exception as e:
                content, glimpse = "", f"[Could not fetch: {str(e)[:50]}]"
            return {
                "title": item["title"],
                "url": item["url"],
                "md_url": md_url,
                "glimpse": glimpse,
                "content": content,
                "section": endpoint,
            }

        return list(await asyncio.gather(*[fetch_page(item) for item in nav_items]))


async def _get_docs(hf_token: str, endpoint: str) -> list[dict[str, str]]:
    async with _cache_lock:
        if endpoint in _docs_cache:
            return _docs_cache[endpoint]

    sub_endpoints = COMPOSITE_ENDPOINTS.get(endpoint, [endpoint])
    all_docs: list[dict[str, str]] = []

    for sub in sub_endpoints:
        async with _cache_lock:
            if sub in _docs_cache:
                all_docs.extend(_docs_cache[sub])
                continue
        docs = await _fetch_endpoint_docs(hf_token, sub)
        async with _cache_lock:
            _docs_cache[sub] = docs
        all_docs.extend(docs)

    async with _cache_lock:
        _docs_cache[endpoint] = all_docs
    return all_docs


def _whoosh_search(docs: list[dict[str, str]], query: str, limit: int) -> list[dict[str, Any]] | None:
    """Try Whoosh full-text search. Returns None if Whoosh unavailable."""
    try:
        from whoosh.analysis import StemmingAnalyzer
        from whoosh.fields import ID, TEXT, Schema
        from whoosh.filedb.filestore import RamStorage
        from whoosh.qparser import MultifieldParser, OrGroup
    except ImportError:
        return None

    analyzer = StemmingAnalyzer()
    schema = Schema(
        title=TEXT(stored=True, analyzer=analyzer),
        url=ID(stored=True, unique=True),
        md_url=ID(stored=True),
        section=ID(stored=True),
        glimpse=TEXT(stored=True, analyzer=analyzer),
        content=TEXT(stored=False, analyzer=analyzer),
    )
    storage = RamStorage()
    index = storage.create_index(schema)
    writer = index.writer()
    for d in docs:
        writer.add_document(
            title=d.get("title", ""),
            url=d.get("url", ""),
            md_url=d.get("md_url", ""),
            section=d.get("section", ""),
            glimpse=d.get("glimpse", ""),
            content=d.get("content", ""),
        )
    writer.commit()

    parser = MultifieldParser(
        ["title", "content"], schema=schema,
        fieldboosts={"title": 2.0, "content": 1.0},
        group=OrGroup,
    )
    try:
        q = parser.parse(query)
    except Exception:
        return []

    with index.searcher() as searcher:
        hits = searcher.search(q, limit=limit)
        return [
            {
                "title": h["title"], "url": h["url"],
                "md_url": h.get("md_url", ""), "section": h.get("section", ""),
                "glimpse": h["glimpse"], "score": round(h.score, 2),
            }
            for h in hits
        ]


def _substring_search(docs: list[dict[str, str]], query: str, limit: int) -> list[dict[str, Any]]:
    """Fallback search: rank by substring-hit count in title + content."""
    q_lower = query.lower()
    q_tokens = [t for t in q_lower.split() if t]
    scored = []
    for d in docs:
        title = d.get("title", "").lower()
        content = d.get("content", "").lower()
        glimpse = d.get("glimpse", "").lower()
        score = 0.0
        for tok in q_tokens:
            score += title.count(tok) * 3.0
            score += glimpse.count(tok) * 1.5
            score += content.count(tok) * 1.0
        if score > 0:
            scored.append({
                "title": d.get("title", ""), "url": d.get("url", ""),
                "md_url": d.get("md_url", ""), "section": d.get("section", ""),
                "glimpse": d.get("glimpse", ""), "score": round(score, 2),
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def _format_results(
    endpoint: str,
    items: list[dict[str, Any]],
    total: int,
    query: str | None = None,
    note: str | None = None,
) -> str:
    out = f"Documentation structure for: https://huggingface.co/docs/{endpoint}\n\n"
    if query:
        out += f"Query: '{query}' → showing {len(items)} result(s) out of {total} pages"
        if note:
            out += f" ({note})"
        out += "\n\n"
    else:
        out += f"Found {len(items)} page(s) (total available: {total}).\n"
        if note:
            out += f"({note})\n"
        out += "\n"
    for i, item in enumerate(items, 1):
        out += f"{i}. **{item['title']}**\n"
        out += f"   URL: {item['url']}\n"
        out += f"   Section: {item.get('section', endpoint)}\n"
        if query and "score" in item:
            out += f"   Relevance score: {item['score']:.2f}\n"
        out += f"   Glimpse: {item['glimpse']}\n\n"
    return out


async def cmd_explore(args) -> tuple[str, bool]:
    endpoint = args.endpoint.lstrip("/")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "ERROR: HF_TOKEN environment variable is required.", False

    try:
        docs = await _get_docs(hf_token, endpoint)
    except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
        return f"Error fetching docs for '{endpoint}': {e}", False
    total = len(docs)

    max_results = args.max_results
    if max_results is None:
        limit = DEFAULT_MAX_RESULTS
        note = f"Showing top {DEFAULT_MAX_RESULTS} results (use --max-results to adjust)."
    elif max_results > MAX_RESULTS_CAP:
        limit = MAX_RESULTS_CAP
        note = f"Requested {max_results} but showing top {MAX_RESULTS_CAP} (cap)."
    else:
        limit = max_results
        note = None

    query = args.query.strip() if args.query and args.query.strip() else None
    results: list[dict[str, Any]] = []
    if query:
        res = _whoosh_search(docs, query, limit)
        if res is None:
            note = (note + "; " if note else "") + "Whoosh not installed, using substring fallback"
            results = _substring_search(docs, query, limit)
        else:
            results = res
        if not results:
            results = docs[:limit]
            note = (note + "; " if note else "") + "No matches for query; showing default ordering"
    else:
        results = docs[:limit]

    return _format_results(endpoint, results, total, query, note), True


async def cmd_fetch(args) -> tuple[str, bool]:
    url = args.url
    if not url:
        return "ERROR: --url is required.", False
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "ERROR: HF_TOKEN environment variable is required.", False

    if not url.endswith(".md"):
        url = f"{url}.md"
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {hf_token}"})
            resp.raise_for_status()
        return f"Documentation from: {url}\n\n{resp.text}", True
    except httpx.HTTPStatusError as e:
        return f"HTTP error: {e.response.status_code} — {e.response.text[:200]}", False
    except httpx.RequestError as e:
        return f"Request error: {e}", False


def main() -> int:
    p = argparse.ArgumentParser(prog="hf_docs.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("explore", help="List doc pages, optionally filter by --query.")
    pe.add_argument("--endpoint", required=True,
                    help="Doc set name: trl, transformers, datasets, peft, accelerate, "
                         "trackio, vllm, optimum, courses, …")
    pe.add_argument("--query", help="Optional search query.")
    pe.add_argument("--max-results", type=int, help="Max results (default: 20, cap: 50).")
    pe.add_argument("--json", action="store_true")

    pf = sub.add_parser("fetch", help="Fetch full markdown of a documentation page.")
    pf.add_argument("--url", required=True,
                    help="Page URL (e.g. https://huggingface.co/docs/trl/sft_trainer).")
    pf.add_argument("--json", action="store_true")

    args = p.parse_args()
    handler = cmd_explore if args.cmd == "explore" else cmd_fetch
    out, ok = asyncio.run(handler(args))
    if args.json:
        print(json.dumps({"cmd": args.cmd, "ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
