---
name: hf-jobs
description: Submit and manage HuggingFace compute jobs — run training / fine-tuning / inference / batch work on HF cloud infrastructure in Python or Docker mode. Supports one-shot and recurring (cron-style) jobs.
---

# hf-jobs

Backed by `scripts/hf_jobs.py`. Wraps `huggingface_hub.HfApi` job endpoints.

## Pre-flight — DO NOT SKIP

Before submitting any non-trivial job, confirm:

- [ ] **Reference implementation** — which working example did you base this script on? (Use `github-code` to find it.)
- [ ] **Dataset format verified** — columns match the training method. (Use `hf-datasets` to confirm.)
- [ ] **Model verified** — architecture / size / tokenizer correct. (Use `hf-repos` to list files.)
- [ ] **Persistence** — `push_to_hub=True` AND `hub_model_id=` set. Job storage is ephemeral; without these, the trained model is gone.
- [ ] **Timeout** — >= 2h for any training. Default 30m kills training mid-run.
- [ ] **Monitoring** — Trackio (or equivalent) included in the script; note the dashboard URL.

For batch / ablation runs: submit **one** job first, confirm it starts training correctly, then submit the rest. Do not submit them all at once — if there's a bug, they all fail for the same reason.

## Subcommands

| Subcommand | What it does |
|---|---|
| `run` | Run a one-shot job (Python script or Docker command) |
| `ps` | List jobs (default: running only; `--all` for everything) |
| `logs` | Fetch full logs for a job (after it completes) |
| `inspect` | JSON details for a job |
| `cancel` | Cancel a running job |
| `sched-run` | Create a scheduled (cron) job |
| `sched-ps` | List scheduled jobs |
| `sched-inspect` | Inspect a scheduled job |
| `sched-delete` | Delete a scheduled job |
| `sched-suspend` | Suspend (pause) a scheduled job |
| `sched-resume` | Resume a suspended scheduled job |

## Canonical patterns

**Python training job** (recommended: pass a local file or a URL, not inline code):

```bash
python scripts/hf_jobs.py run \
    --script ./train.py \
    --dep transformers --dep trl --dep torch --dep datasets --dep trackio --dep accelerate \
    --hardware-flavor a100-large \
    --timeout 8h \
    --env WANDB_DISABLED=true
```

**Docker one-shot:**

```bash
python scripts/hf_jobs.py run \
    --command duckdb -c 'select 1 + 2' \
    --image duckdb/duckdb \
    --hardware-flavor cpu-basic \
    --timeout 1h
```

**Inline Python via stdin** (rare — prefer a file):

```bash
cat << 'EOF' | python scripts/hf_jobs.py run --script-stdin --dep trl --hardware-flavor a10g-large --timeout 4h
# your script here
EOF
```

**Submit and return immediately** (don't block on log stream):

```bash
python scripts/hf_jobs.py run --script ./train.py --hardware-flavor a100-large --timeout 8h --no-wait
# → returns job ID
python scripts/hf_jobs.py ps
python scripts/hf_jobs.py logs --job-id <id>
```

**Scheduled:**

```bash
python scripts/hf_jobs.py sched-run \
    --script ./nightly_eval.py \
    --schedule '@daily' \
    --hardware-flavor t4-small \
    --timeout 2h
```

## Hardware reference

| Flavor | vCPU / RAM / GPU |
|---|---|
| `cpu-basic` | 2 / 16 GB |
| `cpu-upgrade` | 8 / 32 GB |
| `t4-small` | 4 / 15 GB / 16 GB |
| `a10g-small` | 4 / 15 GB / 24 GB |
| `a10g-large` | 12 / 46 GB / 24 GB |
| `a10g-largex2` | 24 / 92 GB / 48 GB |
| `a100-large` | 12 / 142 GB / 80 GB |
| `a100x4` | 48 / 568 GB / 320 GB |
| `a100x8` | 96 / 1136 GB / 640 GB |
| `l40sx4` | 48 / 382 GB / 192 GB |

Sizing: 1–3B → `a10g-largex2`, 7–13B → `a100-large`, 30B+ → `l40sx4` / `a100x4`, 70B+ → `a100x8`.

`a10g-small` and `a10g-large` share the same 24 GB GPU — only CPU/RAM differ.

## OOM recovery (do NOT change the training method)

1. Reduce `per_device_train_batch_size`, raise `gradient_accumulation_steps` proportionally (effective batch stays the same).
2. Enable `gradient_checkpointing=True`.
3. Upgrade GPU (a10gx4 → a100 → a100x4 → a100x8).

Do **not** switch SFT → LoRA or reduce `max_length`. Those change what the user asked for.

## Environment

- `HF_TOKEN` — required. Auto-injected into the job's `HF_TOKEN` + `HUGGINGFACE_HUB_TOKEN` secrets.
- `HF_NAMESPACE` — optional. Defaults to `whoami().name`.

## Related skills

- **`github-code`** — to find the reference implementation before writing the training script.
- **`hf-datasets`** — to verify dataset format before submission.
- **`hf-repos`** — to confirm the base model exists and has the right files.
