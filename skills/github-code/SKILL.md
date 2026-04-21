---
name: github-code
description: Discover GitHub repositories, find example scripts within them, and read source files with line-range support and automatic Jupyter notebook conversion. Use BEFORE writing any ML training, fine-tuning, or inference code — your knowledge of library APIs is outdated and example scripts show the current patterns.
---

# github-code

Three scripts:

- `scripts/list_repos.py` — list and sort a user/org's repositories.
- `scripts/find_examples.py` — find example/tutorial files in a specific repo using fuzzy directory matching (`examples/`, `scripts/`, `notebooks/`, `cookbook/`, …).
- `scripts/read_file.py` — read file contents with line ranges; `.ipynb` → Markdown conversion.

## When to use

MANDATORY before writing ML training / fine-tuning / inference code. Your internal knowledge WILL hallucinate imports, trainer arguments, and config parameters. Working examples show current APIs.

Also useful when:
- Exploring what libraries exist for a task (`list_repos` on `huggingface`, `pytorch`, etc.).
- Checking recently updated repos for latest features (`--sort updated`).
- Studying a specific implementation before adapting it.

## Canonical pattern

```bash
# 1. Which libraries does an org have?
python scripts/list_repos.py --owner huggingface --sort stars --limit 10

# 2. What examples exist in the chosen repo?
python scripts/find_examples.py --repo trl --keyword sft
python scripts/find_examples.py --repo transformers --keyword 'flash attention' --max-results 20

# 3. Read the implementation
python scripts/read_file.py --repo huggingface/trl --path examples/scripts/sft.py
python scripts/read_file.py --repo huggingface/transformers \
    --path src/transformers/models/llama/modeling_llama.py \
    --line-start 300 --line-end 450
```

## Flags cheat-sheet

`list_repos.py`:
- `--owner` (required), `--owner-type {user|org}`
- `--sort {stars|forks|updated|created}`, `--order {asc|desc}`, `--limit N`

`find_examples.py`:
- `--repo` (required), `--org` (default `huggingface`)
- `--keyword` (optional fuzzy match), `--max-results N`, `--min-score N` (0–100)

`read_file.py`:
- `--repo owner/repo` (required), `--path` (required)
- `--ref` (branch / tag / commit, default `HEAD`)
- `--line-start`, `--line-end` (1-indexed inclusive)

All accept `--json`.

## Environment

- `GITHUB_TOKEN` — required. A fine-grained token with public-repo read scope is enough.

## Notebook handling

`read_file.py` detects `.ipynb` automatically, parses the notebook JSON, strips outputs, removes cells tagged `hide`/`hidden`/`remove`, and converts code+markdown cells to Markdown. Install `nbformat` + `nbconvert` (already in `pyproject.toml`) or the raw JSON is returned untouched.

## Related skills

- **`hf-papers`** — use `find_datasets` / `find_models` on a paper, then jump here to find the implementation.
- **`hf-docs`** — use alongside: `github-code` gives you a working recipe, `hf-docs` gives you the authoritative API reference.
