---
name: paper-search-alt
description: Alternative paper search backends — OpenAlex (~250M works, no auth, strong PDF extraction) and CORE (~49M full-text, BODY search). Use when hf-papers can't find a paper, when Semantic Scholar is rate-limited, when you need PDFs of older/long-tail papers, or when you need to search the actual body text of papers (not just titles/abstracts).
---

# paper-search-alt

Two scripts, each wrapping a different open catalog:

- `scripts/openalex.py` — OpenAlex-backed search + PDF download/extract.
- `scripts/core_search.py` — CORE-backed metadata + full-text body search.

This skill is the **fallback / broad-catalog** companion to `hf-papers`. You should almost always start with `hf-papers` (it has HF's curated ML-paper metadata, arXiv section parsing, and Semantic Scholar citation graphs). Come here when:

- A paper you're looking for isn't in HF Papers (common for older or non-ML papers).
- Semantic Scholar is rate-limiting you (HTTP 429 from `snippet_search` / `recommend` / filtered `search`).
- You need the full PDF text of an older paper that doesn't have an arXiv HTML version.
- You need to search the **body** of papers for a specific phrase/method/dataset name that probably isn't in any abstract.

## Backend comparison

| Need | Best tool |
|---|---|
| ML-specific trending / curated / AI summaries | `hf-papers --op trending / search / paper_details` |
| Parsed paper sections (methodology, experiments) | `hf-papers --op read_paper --section N` |
| Citation graph, TL;DRs, influential citations | `hf-papers --op citation_graph / recommend / snippet_search` |
| Broad metadata search (incl. non-ML, older papers) | `openalex.py --query ...` |
| Downloading PDFs + extracting text | `openalex.py --pdf --save-dir ...` |
| **Full-text body search** (phrase inside paper) | `core_search.py full-text --query ...` |
| Fetching a specific paper by CORE ID | `core_search.py get --id ...` |

## Environment

- `OPENALEX_MAILTO` (optional) — your email, gets you into OpenAlex's "polite pool" with higher rate limits. Otherwise a generic mailto is used and public limits apply (still fairly generous).
- `CORE_API_KEY` (required for `core_search.py`) — free, request at https://core.ac.uk/services/api. Without it the script exits with a clear error.

## Canonical fallback pattern

```bash
# 1. Try hf-papers first (rich, curated, ML-tuned)
./.venv/bin/python scripts/papers.py --op search --query "sparse mixture of experts quantization"

# 2. If the paper isn't there (or results are weak), go broad with OpenAlex
./.venv/bin/python scripts/openalex.py --query "sparse mixture of experts quantization" --limit 10

# 3. If an interesting result has a PDF but no arXiv HTML, grab full text:
./.venv/bin/python scripts/openalex.py --query "MoE expert pruning 2024" \
    --pdf --save-dir ./texts

# 4. If you need to find a specific method/dataset mentioned inside a paper body:
./.venv/bin/python scripts/core_search.py full-text \
    --query "adaptive scales per expert routing"
```

## `openalex.py` reference

Single-command script — one invocation per search.

```
openalex.py --query Q [--limit N] [--include-closed] [--pdf] [--save-dir DIR] [--json]
```

- `--query, -q` (required): search query.
- `--limit, -l`: results (default 10, max 200).
- `--include-closed`: include non-OA papers (default: OA-only).
- `--pdf, -p`: download the best-OA PDF for each result and extract text with pypdf. Best-effort — scanned PDFs or paywalled OA stubs fall back to the abstract and a warning is printed to stderr.
- `--save-dir DIR`: write each result as `DIR/<safe_id>.txt` (with full text if `--pdf`, else abstract).
- `--json`: emit `{"ok", "query", "count", "papers": [...], "saved": [...]}`.

ID extraction: if the paper's landing page URL contains an arXiv ID, that's used as `id`. Otherwise it's the OpenAlex suffix (e.g. `W2141902134`).

## `core_search.py` reference

Three subcommands. All require `CORE_API_KEY` in the environment.

```
core_search.py search --query Q [--limit N] [--save-dir DIR] [--with-full-text] [--json]
core_search.py full-text --query Q [--limit N] [--save-dir DIR] [--json]
core_search.py get --id ID [--save-dir DIR] [--with-full-text] [--json]
```

- `search`: metadata (title/abstract) search. Default for general lookups.
- `full-text`: BODY search — restricted to papers that have full text indexed (`_exists_:fullText`). Returns body excerpts in the output. Use when hunting a specific phrase.
- `get`: fetch a single paper by CORE ID. Useful to pull full text after finding an interesting ID via `search` or `full-text`.
- `--with-full-text` (save-dir only): write full body text into the saved `.txt` file instead of just the abstract. On `search`, only works if the result happens to include `fullText`; on `get`, always works if the paper has it.

## Tradeoffs to know

- OpenAlex is the biggest catalog but its search ranking is relevance-based on metadata; don't expect it to find papers by quoted phrases from the body. That's what CORE `full-text` is for.
- CORE's full-text index isn't uniform — not every paper CORE knows about has full text indexed. If `full-text` returns nothing, try `search` (metadata-only) or fall back to OpenAlex.
- Neither backend has citation graphs. For citations, always use `hf-papers --op citation_graph`.
- OpenAlex `--pdf` follows `best_oa_location.pdf_url`. If that's paywalled or doesn't exist, no text is extracted. Many papers have OA copies on institutional repositories that OpenAlex indexes.

## Related skills

- **`hf-papers`** — the primary research entry point. Start here.
- **`research-subagent`** — if you spawn a research sub-agent, add `scripts/openalex.py` and `scripts/core_search.py` to its allowed tools when the crawl risks exhausting Semantic Scholar's rate limit or when the domain is non-ML.
