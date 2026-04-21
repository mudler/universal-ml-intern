---
name: hf-datasets
description: Inspect a HuggingFace dataset in one call — status, configs/splits, schema, sample rows, parquet file listing, and detailed messages-column analysis (roles, keys, tool-calls). REQUIRED before submitting any training job to verify column format matches the training method.
---

# hf-datasets

Backed by `scripts/inspect_dataset.py`.

## When to use

**Always call this before training.** Training fails with `KeyError` if columns don't match the method. Specifically:

- **SFT** needs `messages`, `text`, or `prompt` + `completion`.
- **DPO** needs `prompt`, `chosen`, `rejected`.
- **GRPO** needs `prompt`.

Also use to:
- Understand a dataset before writing data-loading code.
- Check schema changes between dataset versions.
- Analyze the structure of chat-format datasets (which roles are used, whether tool-calls are present).
- See an example row before deciding if the dataset fits the task.

## Usage

```bash
# Full inspection — auto-detects config and split
python scripts/inspect_dataset.py --dataset stanfordnlp/imdb

# Pick a specific split + more sample rows
python scripts/inspect_dataset.py --dataset HuggingFaceH4/ultrachat_200k --split train_sft --sample-rows 5

# Private / gated dataset — HF_TOKEN must be set
HF_TOKEN=hf_... python scripts/inspect_dataset.py --dataset org/gated-dataset
```

## What you get

Output is markdown with sections:

1. **Status** — whether viewer / preview / search / filter / statistics are available.
2. **Structure** — table of configs × splits.
3. **Schema** — table of columns × types (ClassLabel values inlined if ≤5 classes).
4. **Sample Rows** — up to `--sample-rows` rows, long values truncated at 150 chars.
5. **Messages Column Format** (auto-included when a column named `messages` is present) — roles seen, which keys exist (`role`, `content`, `tool_calls`, `tool_call_id`, `name`, `function_call`), tool-call/result presence, and an example message structure.
6. **Files (Parquet)** — parquet file count + total size, grouped by `config/split`.

## Flags

- `--dataset` (required) — `org/name` format.
- `--config`, `--split` — auto-detected if omitted.
- `--sample-rows N` — default 3, cap 10.
- `--json` — wrap output in `{"ok": bool, "output": "..."}`.

## Environment

- `HF_TOKEN` — only needed for private / gated datasets.

## Related skills

- **`hf-papers`** → `find_datasets` surfaces datasets linked to a paper. Pipe the top hit into this skill.
- **`hf-repos`** → for deeper repo-level file access (e.g. read a custom `dataset_infos.json`).
