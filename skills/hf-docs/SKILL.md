---
name: hf-docs
description: Explore and fetch HuggingFace library documentation (transformers, trl, datasets, peft, accelerate, trackio, vllm, optimum, courses, etc.). Use when you need current trainer arguments, config parameters, or API reference that your internal knowledge may have wrong.
---

# hf-docs

Backed by `scripts/hf_docs.py`. Two subcommands:

- `explore` — list pages in a doc set, optionally filtered by a keyword query.
- `fetch` — fetch the full markdown content of a specific page URL.

## When to use

- Before passing arguments to any HF library trainer / config — confirm they exist and have the expected meaning.
- When you see a deprecation warning or an `unexpected keyword argument` error.
- To discover an API you don't know the name of yet.

## Supported endpoints

Pass any of these to `--endpoint`:

- Core libraries: `transformers`, `datasets`, `tokenizers`, `accelerate`, `peft`, `trl`, `diffusers`, `trackio`, `safetensors`.
- Inference: `text-generation-inference`, `text-embeddings-inference`, `inference-endpoints`, `vllm`, `lighteval`.
- Data/hub: `hub`, `huggingface_hub`, `hub-docs`, `datasets-server`.
- Composite endpoints (expand to multiple sub-endpoints):
  - `optimum` → optimum + optimum-habana + optimum-neuron + optimum-intel + optimum-executorch + optimum-tpu.
  - `courses` → llm-course + robotics-course + mcp-course + smol-course + agents-course + deep-rl-course + computer-vision-course + audio-course + ml-games-course + diffusion-course + ml-for-3d-course + cookbook.

## Canonical pattern

```bash
# Find the page(s) matching your question
python scripts/hf_docs.py explore --endpoint trl --query sft
python scripts/hf_docs.py explore --endpoint transformers --query 'flash attention'

# Fetch the full markdown of the most relevant result
python scripts/hf_docs.py fetch --url https://huggingface.co/docs/trl/sft_trainer
```

## Search quality

- If `whoosh` is installed (via `pip install -e '.[docs]'`), search uses a proper stemming full-text index.
- Otherwise, falls back to substring-count ranking on title + content. Good enough for most queries.

## Environment

- `HF_TOKEN` — required.

## Flags

`explore`:
- `--endpoint` (required), `--query` (optional), `--max-results N` (default 20, cap 50).

`fetch`:
- `--url` (required) — can be the `.md` URL or the plain doc URL (`.md` is auto-appended).

Both accept `--json`.

## Related skills

- **`github-code`** — use together: docs tell you what parameters exist, example scripts show them in use.
- **`hf-papers`** — papers describe *what* to do; docs describe *how* the current API supports it.
