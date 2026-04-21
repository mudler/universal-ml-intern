#!/usr/bin/env python3
"""
papers.py — Discover ML papers, read sections, trace citations, find linked resources.

Ported from huggingface/ml-intern's `agent/tools/papers_tool.py` to a standalone
CLI. Combines HuggingFace Papers API + arXiv HTML + Semantic Scholar.

Operations (use --op):
  trending           Trending daily papers on HF (optional --query keyword filter)
  search             Search papers (HF by default; triggers Semantic Scholar
                     when --date-from / --min-citations / --categories given)
  paper_details      Metadata, abstract, AI summary, GitHub link
  read_paper         Read paper: without --section → abstract + TOC; with → full section text
  citation_graph     References + citations with influence flags and intents
  snippet_search     Semantic search over 12M+ full-text passages
  recommend          Find similar papers (single --arxiv-id or --positive-ids)
  find_datasets      Datasets linked to a paper
  find_models        Models linked to a paper
  find_collections   Collections that include a paper
  find_all_resources Parallel fetch of datasets + models + collections

Env:
  S2_API_KEY (optional) — raises Semantic Scholar rate limits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag

HF_API = "https://huggingface.co/api"
ARXIV_HTML = "https://arxiv.org/html"
AR5IV_HTML = "https://ar5iv.labs.arxiv.org/html"

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
MAX_SUMMARY_LEN = 300
MAX_SECTION_PREVIEW_LEN = 280
MAX_SECTION_TEXT_LEN = 8000

SORT_MAP = {"downloads": "downloads", "likes": "likes", "trending": "trendingScore"}

S2_API = "https://api.semanticscholar.org"
S2_API_KEY = os.environ.get("S2_API_KEY")
S2_HEADERS: dict[str, str] = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
S2_TIMEOUT = 12
_s2_last_request: float = 0.0
_s2_cache: dict[str, Any] = {}
_S2_CACHE_MAX = 500


# --- Semantic Scholar plumbing -------------------------------------------------

def _s2_paper_id(arxiv_id: str) -> str:
    return f"ARXIV:{arxiv_id}"


def _s2_cache_key(path: str, params: dict | None) -> str:
    p = tuple(sorted((params or {}).items()))
    return f"{path}:{p}"


async def _s2_request(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> httpx.Response | None:
    global _s2_last_request
    url = f"{S2_API}{path}"
    kwargs.setdefault("headers", {}).update(S2_HEADERS)
    kwargs.setdefault("timeout", S2_TIMEOUT)

    for attempt in range(3):
        if S2_API_KEY:
            min_interval = 1.0 if "search" in path else 0.1
            elapsed = time.monotonic() - _s2_last_request
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        _s2_last_request = time.monotonic()

        try:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code == 429:
                if attempt < 2:
                    await asyncio.sleep(60)
                    continue
                return None
            if resp.status_code >= 500:
                if attempt < 2:
                    await asyncio.sleep(3)
                    continue
                return None
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt < 2:
                await asyncio.sleep(3)
                continue
            return None
    return None


async def _s2_get_json(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict | None:
    key = _s2_cache_key(path, params)
    if key in _s2_cache:
        return _s2_cache[key]

    resp = await _s2_request(client, "GET", path, params=params or {})
    if resp and resp.status_code == 200:
        data = resp.json()
        if len(_s2_cache) < _S2_CACHE_MAX:
            _s2_cache[key] = data
        return data
    return None


# --- HTML paper parsing --------------------------------------------------------

def _parse_paper_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("h1", class_="ltx_title")
    title = title_el.get_text(strip=True).removeprefix("Title:") if title_el else ""

    abstract_el = soup.find("div", class_="ltx_abstract")
    abstract = ""
    if abstract_el:
        for child in abstract_el.children:
            if isinstance(child, Tag) and child.name in ("h6", "h2", "h3", "p", "span"):
                if child.get_text(strip=True).lower() == "abstract":
                    continue
            if isinstance(child, Tag) and child.name == "p":
                abstract += child.get_text(separator=" ", strip=True) + " "
        abstract = abstract.strip()

    sections: list[dict[str, Any]] = []
    headings = soup.find_all(["h2", "h3"], class_=lambda c: c and "ltx_title" in c)

    for heading in headings:
        level = 2 if heading.name == "h2" else 3
        heading_text = heading.get_text(separator=" ", strip=True)

        text_parts: list[str] = []
        sibling = heading.find_next_sibling()
        while sibling:
            if isinstance(sibling, Tag):
                if sibling.name in ("h2", "h3") and "ltx_title" in (sibling.get("class") or []):
                    break
                if sibling.name == "h2" and level == 3:
                    break
                text_parts.append(sibling.get_text(separator=" ", strip=True))
            sibling = sibling.find_next_sibling()

        parent_section = heading.find_parent("section")
        if parent_section and not text_parts:
            for p in parent_section.find_all("p", recursive=False):
                text_parts.append(p.get_text(separator=" ", strip=True))

        section_text = "\n\n".join(t for t in text_parts if t)
        num_match = re.match(r"^([A-Z]?\d+(?:\.\d+)*)\s", heading_text)
        section_id = num_match.group(1) if num_match else ""

        sections.append({"id": section_id, "title": heading_text, "level": level, "text": section_text})

    return {"title": title, "abstract": abstract, "sections": sections}


def _find_section(sections: list[dict], query: str) -> dict | None:
    query_lower = query.lower().strip()
    for s in sections:
        if s["id"] == query_lower or s["id"] == query:
            return s
    for s in sections:
        if query_lower == s["title"].lower():
            return s
    for s in sections:
        if query_lower in s["title"].lower():
            return s
    for s in sections:
        if s["id"].startswith(query_lower + ".") or s["id"] == query_lower:
            return s
    return None


# --- Formatting helpers --------------------------------------------------------

def _clean_description(text: str) -> str:
    text = re.sub(r"[\t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_paper_list(papers: list, title: str, date: str | None = None, query: str | None = None) -> str:
    lines = [f"# {title}"]
    if date:
        lines[0] += f" ({date})"
    if query:
        lines.append(f"Filtered by: '{query}'")
    lines.append(f"Showing {len(papers)} paper(s)\n")

    for i, item in enumerate(papers, 1):
        paper = item.get("paper", item)
        arxiv_id = paper.get("id", "")
        paper_title = paper.get("title", "Unknown")
        upvotes = paper.get("upvotes", 0)
        summary = paper.get("ai_summary") or _truncate(paper.get("summary", ""), MAX_SUMMARY_LEN)
        keywords = paper.get("ai_keywords") or []
        github = paper.get("githubRepo") or ""
        stars = paper.get("githubStars") or 0

        lines.append(f"## {i}. {paper_title}")
        lines.append(f"**arxiv_id:** {arxiv_id} | **upvotes:** {upvotes}")
        lines.append(f"https://huggingface.co/papers/{arxiv_id}")
        if keywords:
            lines.append(f"**Keywords:** {', '.join(keywords[:5])}")
        if github:
            lines.append(f"**GitHub:** {github} ({stars} stars)")
        if summary:
            lines.append(f"**Summary:** {_truncate(summary, MAX_SUMMARY_LEN)}")
        lines.append("")

    return "\n".join(lines)


def _format_paper_detail(paper: dict, s2_data: dict | None = None) -> str:
    arxiv_id = paper.get("id", "")
    title = paper.get("title", "Unknown")
    upvotes = paper.get("upvotes", 0)
    ai_summary = paper.get("ai_summary") or ""
    summary = paper.get("summary", "")
    keywords = paper.get("ai_keywords") or []
    github = paper.get("githubRepo") or ""
    stars = paper.get("githubStars") or 0
    authors = paper.get("authors") or []

    lines = [f"# {title}"]
    meta_parts = [f"**arxiv_id:** {arxiv_id}", f"**upvotes:** {upvotes}"]
    if s2_data:
        cites = s2_data.get("citationCount", 0)
        influential = s2_data.get("influentialCitationCount", 0)
        meta_parts.append(f"**citations:** {cites} ({influential} influential)")
    lines.append(" | ".join(meta_parts))
    lines.append(f"https://huggingface.co/papers/{arxiv_id}")
    lines.append(f"https://arxiv.org/abs/{arxiv_id}")

    if authors:
        names = [a.get("name", "") for a in authors[:10]]
        author_str = ", ".join(n for n in names if n)
        if len(authors) > 10:
            author_str += f" (+{len(authors) - 10} more)"
        lines.append(f"**Authors:** {author_str}")

    if keywords:
        lines.append(f"**Keywords:** {', '.join(keywords)}")
    if s2_data and s2_data.get("s2FieldsOfStudy"):
        fields = [f["category"] for f in s2_data["s2FieldsOfStudy"] if f.get("category")]
        if fields:
            lines.append(f"**Fields:** {', '.join(fields)}")
    if s2_data and s2_data.get("venue"):
        lines.append(f"**Venue:** {s2_data['venue']}")
    if github:
        lines.append(f"**GitHub:** {github} ({stars} stars)")

    if s2_data and s2_data.get("tldr"):
        tldr_text = s2_data["tldr"].get("text", "")
        if tldr_text:
            lines.append(f"\n## TL;DR\n{tldr_text}")
    if ai_summary:
        lines.append(f"\n## AI Summary\n{ai_summary}")
    if summary:
        lines.append(f"\n## Abstract\n{_truncate(summary, 500)}")

    lines.append(
        "\n**Next:** read_paper to read specific sections, find_all_resources for linked datasets/models, "
        "or citation_graph to trace references and citations."
    )
    return "\n".join(lines)


def _format_read_paper_toc(parsed: dict[str, Any], arxiv_id: str) -> str:
    lines = [f"# {parsed['title']}"]
    lines.append(f"https://arxiv.org/abs/{arxiv_id}\n")
    if parsed["abstract"]:
        lines.append(f"## Abstract\n{parsed['abstract']}\n")
    lines.append("## Sections")
    for s in parsed["sections"]:
        prefix = "  " if s["level"] == 3 else ""
        preview = _truncate(s["text"], MAX_SECTION_PREVIEW_LEN) if s["text"] else "(empty)"
        lines.append(f"{prefix}- **{s['title']}**: {preview}")
    lines.append('\nCall read_paper with --section (e.g. --section 4 or --section Experiments) to read a specific section.')
    return "\n".join(lines)


def _format_read_paper_section(section: dict, arxiv_id: str) -> str:
    lines = [f"# {section['title']}"]
    lines.append(f"https://arxiv.org/abs/{arxiv_id}\n")
    text = section["text"]
    if len(text) > MAX_SECTION_TEXT_LEN:
        text = text[:MAX_SECTION_TEXT_LEN] + f"\n\n... (truncated at {MAX_SECTION_TEXT_LEN} chars)"
    lines.append(text if text else "(This section has no extractable text content.)")
    return "\n".join(lines)


def _format_datasets(datasets: list, arxiv_id: str, sort: str) -> str:
    lines = [f"# Datasets linked to paper {arxiv_id}"]
    lines.append(f"https://huggingface.co/papers/{arxiv_id}")
    lines.append(f"Showing {len(datasets)} dataset(s), sorted by {sort}\n")
    for i, ds in enumerate(datasets, 1):
        ds_id = ds.get("id", "unknown")
        downloads = ds.get("downloads", 0)
        likes = ds.get("likes", 0)
        desc = _truncate(_clean_description(ds.get("description") or ""), MAX_SUMMARY_LEN)
        tags = ds.get("tags") or []
        interesting = [t for t in tags if not t.startswith(("arxiv:", "region:"))][:5]
        lines.append(f"**{i}. [{ds_id}](https://huggingface.co/datasets/{ds_id})**")
        lines.append(f"   Downloads: {downloads:,} | Likes: {likes}")
        if interesting:
            lines.append(f"   Tags: {', '.join(interesting)}")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")
    if datasets:
        top = datasets[0].get("id", "")
        lines.append(f'**Inspect top dataset:** inspect_dataset.py --dataset "{top}"')
    return "\n".join(lines)


def _format_datasets_compact(datasets: list) -> str:
    if not datasets:
        return "## Datasets\nNone found"
    lines = [f"## Datasets ({len(datasets)})"]
    for ds in datasets:
        lines.append(f"- **{ds.get('id', '?')}** ({ds.get('downloads', 0):,} downloads)")
    return "\n".join(lines)


def _format_models(models: list, arxiv_id: str, sort: str) -> str:
    lines = [f"# Models linked to paper {arxiv_id}"]
    lines.append(f"https://huggingface.co/papers/{arxiv_id}")
    lines.append(f"Showing {len(models)} model(s), sorted by {sort}\n")
    for i, m in enumerate(models, 1):
        model_id = m.get("id", "unknown")
        downloads = m.get("downloads", 0)
        likes = m.get("likes", 0)
        pipeline = m.get("pipeline_tag") or ""
        library = m.get("library_name") or ""
        lines.append(f"**{i}. [{model_id}](https://huggingface.co/{model_id})**")
        meta = f"   Downloads: {downloads:,} | Likes: {likes}"
        if pipeline:
            meta += f" | Task: {pipeline}"
        if library:
            meta += f" | Library: {library}"
        lines.append(meta)
        lines.append("")
    return "\n".join(lines)


def _format_models_compact(models: list) -> str:
    if not models:
        return "## Models\nNone found"
    lines = [f"## Models ({len(models)})"]
    for m in models:
        pipeline = m.get("pipeline_tag") or ""
        suffix = f" ({pipeline})" if pipeline else ""
        lines.append(f"- **{m.get('id', '?')}** ({m.get('downloads', 0):,} downloads){suffix}")
    return "\n".join(lines)


def _format_collections(collections: list, arxiv_id: str) -> str:
    lines = [f"# Collections containing paper {arxiv_id}"]
    lines.append(f"Showing {len(collections)} collection(s)\n")
    for i, c in enumerate(collections, 1):
        slug = c.get("slug", "")
        title = c.get("title", "Untitled")
        upvotes = c.get("upvotes", 0)
        owner = c.get("owner", {}).get("name", "")
        desc = _truncate(c.get("description") or "", MAX_SUMMARY_LEN)
        num_items = len(c.get("items", []))
        lines.append(f"**{i}. {title}**")
        lines.append(f"   By: {owner} | Upvotes: {upvotes} | Items: {num_items}")
        lines.append(f"   https://huggingface.co/collections/{slug}")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")
    return "\n".join(lines)


def _format_collections_compact(collections: list) -> str:
    if not collections:
        return "## Collections\nNone found"
    lines = [f"## Collections ({len(collections)})"]
    for c in collections:
        title = c.get("title", "Untitled")
        owner = c.get("owner", {}).get("name", "")
        upvotes = c.get("upvotes", 0)
        lines.append(f"- **{title}** by {owner} ({upvotes} upvotes)")
    return "\n".join(lines)


def _format_s2_paper_list(papers: list[dict], title: str) -> str:
    lines = [f"# {title}"]
    lines.append(f"Showing {len(papers)} result(s)\n")
    for i, paper in enumerate(papers, 1):
        ptitle = paper.get("title") or "(untitled)"
        year = paper.get("year") or "?"
        cites = paper.get("citationCount", 0)
        venue = paper.get("venue") or ""
        ext_ids = paper.get("externalIds") or {}
        aid = ext_ids.get("ArXiv", "")
        tldr = (paper.get("tldr") or {}).get("text", "")

        lines.append(f"### {i}. {ptitle}")
        meta = [f"Year: {year}", f"Citations: {cites}"]
        if venue:
            meta.append(f"Venue: {venue}")
        if aid:
            meta.append(f"arxiv_id: {aid}")
        lines.append(" | ".join(meta))
        if aid:
            lines.append(f"https://arxiv.org/abs/{aid}")
        if tldr:
            lines.append(f"**TL;DR:** {tldr}")
        lines.append("")
    lines.append("Use paper_details with arxiv_id for full info, or read_paper to read sections.")
    return "\n".join(lines)


def _format_citation_entry(entry: dict, show_context: bool = False) -> str:
    paper = entry.get("citingPaper") or entry.get("citedPaper") or {}
    title = paper.get("title") or "(untitled)"
    year = paper.get("year") or "?"
    cites = paper.get("citationCount", 0)
    ext_ids = paper.get("externalIds") or {}
    aid = ext_ids.get("ArXiv", "")
    influential = " **[influential]**" if entry.get("isInfluential") else ""

    parts = [f"- **{title}** ({year}, {cites} cites){influential}"]
    if aid:
        parts[0] += f"  arxiv:{aid}"

    if show_context:
        intents = entry.get("intents") or []
        if intents:
            parts.append(f"  Intent: {', '.join(intents)}")
        contexts = entry.get("contexts") or []
        for ctx in contexts[:2]:
            if ctx:
                parts.append(f"  > {_truncate(ctx, 200)}")

    return "\n".join(parts)


def _format_citation_graph(arxiv_id: str, references: list[dict] | None, citations: list[dict] | None) -> str:
    lines = [f"# Citation Graph for {arxiv_id}"]
    lines.append(f"https://arxiv.org/abs/{arxiv_id}\n")
    if references is not None:
        lines.append(f"## References ({len(references)})")
        if references:
            for entry in references:
                lines.append(_format_citation_entry(entry))
        else:
            lines.append("No references found.")
        lines.append("")
    if citations is not None:
        lines.append(f"## Citations ({len(citations)})")
        if citations:
            for entry in citations:
                lines.append(_format_citation_entry(entry, show_context=True))
        else:
            lines.append("No citations found.")
        lines.append("")
    lines.append("**Tip:** Use paper_details with an arxiv_id from above to explore further.")
    return "\n".join(lines)


def _format_snippets(snippets: list[dict], query: str) -> str:
    lines = [f"# Snippet Search: '{query}'"]
    lines.append(f"Found {len(snippets)} matching passage(s)\n")
    for i, item in enumerate(snippets, 1):
        paper = item.get("paper") or {}
        ptitle = paper.get("title") or "(untitled)"
        year = paper.get("year") or "?"
        cites = paper.get("citationCount", 0)
        ext_ids = paper.get("externalIds") or {}
        aid = ext_ids.get("ArXiv", "")
        snippet = item.get("snippet") or {}
        text = snippet.get("text", "")
        section = snippet.get("section") or ""
        lines.append(f"### {i}. {ptitle} ({year}, {cites} cites)")
        if aid:
            lines.append(f"arxiv:{aid}")
        if section:
            lines.append(f"Section: {section}")
        if text:
            lines.append(f"> {_truncate(text, 400)}")
        lines.append("")
    lines.append("Use paper_details or read_paper with arxiv_id to explore a paper further.")
    return "\n".join(lines)


# --- Operation implementations ------------------------------------------------

async def op_trending(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    params: dict[str, Any] = {"limit": limit if not args.query else max(limit * 3, 30)}
    if args.date:
        params["date"] = args.date
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/daily_papers", params=params)
        resp.raise_for_status()
        papers = resp.json()

    if args.query:
        q = args.query.lower()
        papers = [
            p for p in papers
            if q in p.get("title", "").lower()
            or q in p.get("paper", {}).get("title", "").lower()
            or q in p.get("paper", {}).get("summary", "").lower()
            or any(q in kw.lower() for kw in (p.get("paper", {}).get("ai_keywords") or []))
        ]

    papers = papers[:limit]
    if not papers:
        msg = "No trending papers found"
        if args.query:
            msg += f" matching '{args.query}'"
        if args.date:
            msg += f" for {args.date}"
        return msg
    return _format_paper_list(papers, "Trending Papers", date=args.date, query=args.query)


async def _s2_bulk_search(args, limit: int) -> str | None:
    params: dict[str, Any] = {
        "query": args.query,
        "limit": limit,
        "fields": "title,externalIds,year,citationCount,tldr,venue,publicationDate",
    }
    if args.date_from or args.date_to:
        params["publicationDateOrYear"] = f"{args.date_from or ''}:{args.date_to or ''}"
    if args.categories:
        params["fieldsOfStudy"] = args.categories
    if args.min_citations:
        params["minCitationCount"] = str(args.min_citations)
    if args.sort_by and args.sort_by != "relevance":
        params["sort"] = f"{args.sort_by}:desc"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await _s2_request(client, "GET", "/graph/v1/paper/search/bulk", params=params)
        if not resp or resp.status_code != 200:
            return None
        data = resp.json()

    papers = data.get("data") or []
    if not papers:
        return f"No papers found for '{args.query}' with the given filters."
    return _format_s2_paper_list(papers[:limit], f"Papers matching '{args.query}' (Semantic Scholar)")


async def op_search(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.query:
        return "ERROR: --query is required for search."

    use_s2 = any([args.date_from, args.date_to, args.categories, args.min_citations, args.sort_by])
    if use_s2:
        result = await _s2_bulk_search(args, limit)
        if result is not None:
            return result

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/papers/search", params={"q": args.query, "limit": limit})
        resp.raise_for_status()
        papers = resp.json()
    if not papers:
        return f"No papers found for '{args.query}'"
    return _format_paper_list(papers, f"Papers matching '{args.query}'")


async def op_paper_details(args) -> str:
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/papers/{args.arxiv_id}")
        resp.raise_for_status()
        paper = resp.json()
        # Also grab S2 data for TL;DR and citation count
        s2_data = await _s2_get_json(
            client,
            f"/graph/v1/paper/{_s2_paper_id(args.arxiv_id)}",
            {"fields": "citationCount,influentialCitationCount,tldr,venue,s2FieldsOfStudy"},
        )
    return _format_paper_detail(paper, s2_data)


async def op_read_paper(args) -> str:
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."

    parsed = None
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for base_url in (ARXIV_HTML, AR5IV_HTML):
            try:
                resp = await client.get(f"{base_url}/{args.arxiv_id}")
                if resp.status_code == 200:
                    parsed = _parse_paper_html(resp.text)
                    if parsed["sections"]:
                        break
                    parsed = None
            except httpx.RequestError:
                continue

    if not parsed or not parsed["sections"]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{HF_API}/papers/{args.arxiv_id}")
                resp.raise_for_status()
                paper = resp.json()
            abstract = paper.get("summary", "")
            title = paper.get("title", "")
            msg = f"# {title}\nhttps://arxiv.org/abs/{args.arxiv_id}\n\n"
            msg += f"## Abstract\n{abstract}\n\n"
            msg += "HTML version not available for this paper. Only abstract shown.\n"
            msg += f"PDF: https://arxiv.org/pdf/{args.arxiv_id}"
            return msg
        except Exception:
            return f"Could not fetch paper {args.arxiv_id}. Check the arxiv ID is correct."

    if not args.section:
        return _format_read_paper_toc(parsed, args.arxiv_id)

    section = _find_section(parsed["sections"], args.section)
    if not section:
        available = "\n".join(f"- {s['title']}" for s in parsed["sections"])
        return f"Section '{args.section}' not found. Available sections:\n{available}"
    return _format_read_paper_section(section, args.arxiv_id)


async def op_citation_graph(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."
    direction = args.direction or "both"
    s2_id = _s2_paper_id(args.arxiv_id)
    fields = "title,externalIds,year,citationCount,influentialCitationCount,contexts,intents,isInfluential"
    params = {"fields": fields, "limit": limit}

    async with httpx.AsyncClient(timeout=15) as client:
        refs, cites = None, None
        coros = []
        if direction in ("references", "both"):
            coros.append(_s2_get_json(client, f"/graph/v1/paper/{s2_id}/references", params))
        if direction in ("citations", "both"):
            coros.append(_s2_get_json(client, f"/graph/v1/paper/{s2_id}/citations", params))
        results = await asyncio.gather(*coros, return_exceptions=True)
        idx = 0
        if direction in ("references", "both"):
            r = results[idx]
            if isinstance(r, dict):
                refs = r.get("data", [])
            idx += 1
        if direction in ("citations", "both"):
            r = results[idx]
            if isinstance(r, dict):
                cites = r.get("data", [])

    if refs is None and cites is None:
        return f"Could not fetch citation data for {args.arxiv_id}. Paper may not be indexed by Semantic Scholar."
    return _format_citation_graph(args.arxiv_id, refs, cites)


async def op_snippet_search(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.query:
        return "ERROR: --query is required for snippet_search."
    params: dict[str, Any] = {
        "query": args.query,
        "limit": limit,
        "fields": "title,externalIds,year,citationCount",
    }
    if args.date_from or args.date_to:
        params["publicationDateOrYear"] = f"{args.date_from or ''}:{args.date_to or ''}"
    if args.categories:
        params["fieldsOfStudy"] = args.categories
    if args.min_citations:
        params["minCitationCount"] = str(args.min_citations)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await _s2_request(client, "GET", "/graph/v1/snippet/search", params=params)
        if not resp or resp.status_code != 200:
            return "Snippet search failed. Semantic Scholar may be unavailable."
        data = resp.json()

    snippets = data.get("data") or []
    if not snippets:
        return f"No snippets found for '{args.query}'."
    return _format_snippets(snippets, args.query)


async def op_recommend(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.arxiv_id and not args.positive_ids:
        return "ERROR: --arxiv-id or --positive-ids is required for recommend."
    fields = "title,externalIds,year,citationCount,tldr,venue"

    async with httpx.AsyncClient(timeout=15) as client:
        if args.positive_ids and not args.arxiv_id:
            pos = [_s2_paper_id(pid.strip()) for pid in args.positive_ids.split(",") if pid.strip()]
            neg = (
                [_s2_paper_id(pid.strip()) for pid in args.negative_ids.split(",") if pid.strip()]
                if args.negative_ids else []
            )
            resp = await _s2_request(
                client, "POST", "/recommendations/v1/papers/",
                json={"positivePaperIds": pos, "negativePaperIds": neg},
                params={"fields": fields, "limit": limit},
            )
            if not resp or resp.status_code != 200:
                return "Recommendation request failed. Semantic Scholar may be unavailable."
            data = resp.json()
        else:
            data = await _s2_get_json(
                client,
                f"/recommendations/v1/papers/forpaper/{_s2_paper_id(args.arxiv_id)}",
                {"fields": fields, "limit": limit, "from": "recent"},
            )
            if not data:
                return "Recommendation request failed. Semantic Scholar may be unavailable."

    papers = data.get("recommendedPapers") or []
    if not papers:
        return "No recommendations found."
    title = f"Recommended papers based on {args.arxiv_id or args.positive_ids}"
    return _format_s2_paper_list(papers[:limit], title)


async def op_find_datasets(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."
    sort = args.sort or "downloads"
    sort_key = SORT_MAP.get(sort, "downloads")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HF_API}/datasets",
            params={"filter": f"arxiv:{args.arxiv_id}", "limit": limit, "sort": sort_key, "direction": -1},
        )
        resp.raise_for_status()
        datasets = resp.json()
    if not datasets:
        return f"No datasets found linked to paper {args.arxiv_id}.\nhttps://huggingface.co/papers/{args.arxiv_id}"
    return _format_datasets(datasets, args.arxiv_id, sort)


async def op_find_models(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."
    sort = args.sort or "downloads"
    sort_key = SORT_MAP.get(sort, "downloads")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HF_API}/models",
            params={"filter": f"arxiv:{args.arxiv_id}", "limit": limit, "sort": sort_key, "direction": -1},
        )
        resp.raise_for_status()
        models = resp.json()
    if not models:
        return f"No models found linked to paper {args.arxiv_id}.\nhttps://huggingface.co/papers/{args.arxiv_id}"
    return _format_models(models, args.arxiv_id, sort)


async def op_find_collections(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/collections", params={"paper": args.arxiv_id})
        resp.raise_for_status()
        collections = resp.json()
    if not collections:
        return f"No collections found containing paper {args.arxiv_id}.\nhttps://huggingface.co/papers/{args.arxiv_id}"
    collections = collections[:limit]
    return _format_collections(collections, args.arxiv_id)


async def op_find_all_resources(args) -> str:
    limit = min(args.limit, MAX_LIMIT)
    if not args.arxiv_id:
        return "ERROR: --arxiv-id is required."
    per_cat = min(limit, 10)

    async with httpx.AsyncClient(timeout=15) as client:
        results = await asyncio.gather(
            client.get(
                f"{HF_API}/datasets",
                params={"filter": f"arxiv:{args.arxiv_id}", "limit": per_cat, "sort": "downloads", "direction": -1},
            ),
            client.get(
                f"{HF_API}/models",
                params={"filter": f"arxiv:{args.arxiv_id}", "limit": per_cat, "sort": "downloads", "direction": -1},
            ),
            client.get(f"{HF_API}/collections", params={"paper": args.arxiv_id}),
            return_exceptions=True,
        )

    sections = []
    if isinstance(results[0], Exception):
        sections.append(f"## Datasets\nError: {results[0]}")
    else:
        datasets = results[0].json()
        sections.append(_format_datasets_compact(datasets[:per_cat]))
    if isinstance(results[1], Exception):
        sections.append(f"## Models\nError: {results[1]}")
    else:
        models = results[1].json()
        sections.append(_format_models_compact(models[:per_cat]))
    if isinstance(results[2], Exception):
        sections.append(f"## Collections\nError: {results[2]}")
    else:
        collections = results[2].json()
        sections.append(_format_collections_compact(collections[:per_cat]))

    header = f"# Resources linked to paper {args.arxiv_id}\nhttps://huggingface.co/papers/{args.arxiv_id}\n"
    return header + "\n\n".join(sections)


OPERATIONS = {
    "trending": op_trending,
    "search": op_search,
    "paper_details": op_paper_details,
    "read_paper": op_read_paper,
    "citation_graph": op_citation_graph,
    "snippet_search": op_snippet_search,
    "recommend": op_recommend,
    "find_datasets": op_find_datasets,
    "find_models": op_find_models,
    "find_collections": op_find_collections,
    "find_all_resources": op_find_all_resources,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="papers.py",
        description=__doc__.strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--op", required=True, choices=list(OPERATIONS.keys()),
                   help="Operation to run.")
    p.add_argument("--query", help="Search query (search, snippet_search, trending filter).")
    p.add_argument("--arxiv-id", help="ArXiv paper ID, e.g. 2305.18290.")
    p.add_argument("--section", help="Section number or name (read_paper only).")
    p.add_argument("--direction", choices=["citations", "references", "both"],
                   help="Direction for citation_graph (default: both).")
    p.add_argument("--date", help="YYYY-MM-DD for trending.")
    p.add_argument("--date-from", help="YYYY-MM-DD (triggers Semantic Scholar search).")
    p.add_argument("--date-to", help="YYYY-MM-DD (triggers Semantic Scholar search).")
    p.add_argument("--categories", help="Field of study, e.g. 'Computer Science'.")
    p.add_argument("--min-citations", type=int, help="Minimum citation count filter.")
    p.add_argument("--sort-by", choices=["relevance", "citationCount", "publicationDate"],
                   help="Sort for Semantic Scholar search.")
    p.add_argument("--positive-ids", help="Comma-separated arxiv IDs for multi-paper recommendations.")
    p.add_argument("--negative-ids", help="Comma-separated arxiv IDs as negative examples.")
    p.add_argument("--sort", choices=["downloads", "likes", "trending"],
                   help="Sort for find_datasets/find_models (default: downloads).")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help=f"Max results (default: {DEFAULT_LIMIT}, max: {MAX_LIMIT}).")
    p.add_argument("--json", action="store_true",
                   help="Emit a minimal JSON envelope around the formatted output.")
    return p


async def _run(args) -> int:
    handler = OPERATIONS[args.op]
    try:
        out = await handler(args)
    except httpx.HTTPStatusError as e:
        out = f"API error: {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        out = f"Request error: {e}"
    except Exception as e:
        out = f"Error in {args.op}: {e}"

    if args.json:
        print(json.dumps({"op": args.op, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
