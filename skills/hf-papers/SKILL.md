---
name: hf-papers
description: Discover ML research papers, read their contents by section, trace citations, find linked datasets/models — combining HuggingFace Papers, arXiv, and Semantic Scholar. Use whenever you need to research a method, find training recipes with published results, crawl a citation graph, or locate the datasets a paper used. This is your starting point for any literature-first workflow.
---

# hf-papers

Backed by `scripts/papers.py`. All subcommands call that script via Bash.

## When to use

- You need current literature on a task (your internal knowledge is outdated).
- You have an anchor paper and need to crawl its citation graph.
- You want the methodology section of a specific paper — full text, not abstract.
- You need to find datasets/models linked to a paper.
- You need to check if there's a more recent paper that beat a known result.

## Operations (via `--op`)

| Operation | What it does | Key flags |
|---|---|---|
| `trending` | Trending papers on HF daily, optionally filtered by keyword | `--query`, `--date YYYY-MM-DD` |
| `search` | Search papers (HF by default; Semantic Scholar when filters set) | `--query`, `--date-from`, `--date-to`, `--min-citations`, `--categories`, `--sort-by` |
| `paper_details` | Metadata, abstract, AI summary, GitHub link, citation counts | `--arxiv-id` |
| `read_paper` | Abstract + TOC (no `--section`), or full section text | `--arxiv-id`, `--section` |
| `citation_graph` | References + citations with *isInfluential* flag and intents | `--arxiv-id`, `--direction` |
| `snippet_search` | Semantic search across 12M+ full-text passages | `--query`, `--min-citations` |
| `recommend` | Similar papers (single seed or positive/negative lists) | `--arxiv-id` OR `--positive-ids`, `--negative-ids` |
| `find_datasets` | HF datasets linked to a paper | `--arxiv-id`, `--sort` |
| `find_models` | HF models linked to a paper | `--arxiv-id`, `--sort` |
| `find_collections` | HF collections containing a paper | `--arxiv-id` |
| `find_all_resources` | Datasets + models + collections in parallel | `--arxiv-id` |

All ops accept `--limit N` (default 10, max 50) and `--json` for a minimal JSON envelope.

## Canonical literature-crawl flow

```bash
# 1. Find the anchor paper
python scripts/papers.py --op search --query "GPQA graduate questions" --sort-by citationCount

# 2. Crawl citations downstream — who improved on it?
python scripts/papers.py --op citation_graph --arxiv-id 2311.12022 --direction citations

# 3. Read methodology of a promising downstream paper
python scripts/papers.py --op read_paper --arxiv-id 2404.01348                 # TOC
python scripts/papers.py --op read_paper --arxiv-id 2404.01348 --section 3     # Methodology
python scripts/papers.py --op read_paper --arxiv-id 2404.01348 --section 4     # Experiments

# 4. Find resources linked to it
python scripts/papers.py --op find_all_resources --arxiv-id 2404.01348

# 5. Inspect the dataset they used
python scripts/hf_docs.py explore --endpoint trl --query sft    # pivot to code once recipe is clear
```

## Environment

- `S2_API_KEY` (optional) — raises Semantic Scholar rate limits.

## Output contract

Every op prints markdown-formatted text to stdout. Exit code 0 on success, 1 on error. With `--json` you get `{"op": "...", "output": "..."}`.

## Related skills

- **`research-subagent`** — when the crawl is heavy, spawn a sub-agent and pass this skill's invocation pattern as its toolkit. Keeps the main context clean.
- **`hf-datasets`** — call right after `find_datasets` to verify the top result.
- **`github-code`** — pivot to here once you've extracted a training recipe, to find working implementation code.
