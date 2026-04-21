---
name: hf-repos
description: File and git-like operations on HuggingFace repositories (model / dataset / space). Covers list / read / upload / delete of files, branches, tags, pull requests, and repo management (create / update). Use to inspect repos, ship artifacts, and propose changes via PRs.
---

# hf-repos

Two scripts:

- `scripts/hf_repo_files.py` — **file** operations: list, read, upload, delete (with wildcard patterns).
- `scripts/hf_repo_git.py` — **git-like** operations: branches, tags, PRs, repo create/update.

## When to use

- **Read config files** (`config.json`, `README.md`, tokenizer configs) on a model/dataset repo.
- **Upload artifacts** — training scripts, results, fine-tuned checkpoints (use `hf-jobs` for the training run itself; this is for the push).
- **Open a PR** against someone else's repo to fix / contribute.
- **Manage your own repo** — branches for experiments, tags for versions, visibility changes.

## Canonical patterns

### `hf_repo_files.py` (file ops)

```bash
# Peek at a model repo
python scripts/hf_repo_files.py list --repo-id meta-llama/Llama-2-7b

# Read a config
python scripts/hf_repo_files.py read --repo-id gpt2 --path config.json

# Upload a README
python scripts/hf_repo_files.py upload \
    --repo-id me/my-model --path README.md \
    --content-file ./README.md \
    --commit-message "Initial README"

# Upload as a PR (propose, don't commit)
python scripts/hf_repo_files.py upload \
    --repo-id org/model --path fix.py \
    --content-file ./fix.py --create-pr

# Delete with wildcards
python scripts/hf_repo_files.py delete \
    --repo-id me/my-model --pattern '*.tmp' --pattern 'logs/'
```

Flags:
- `--repo-type {model|dataset|space}` (default `model`).
- `--revision` (branch / tag / commit, default `main`).
- Upload content via `--content-file` (file path), `--content-stdin` (pipe), or `--content` (inline string).

### `hf_repo_git.py` (git-like ops)

**Branch / tag flow:**

```bash
python scripts/hf_repo_git.py list-refs --repo-id me/my-model
python scripts/hf_repo_git.py create-branch --repo-id me/my-model --branch experiment-v2
python scripts/hf_repo_git.py create-tag --repo-id me/my-model --tag v1.0 --revision main
```

**PR flow (HF uses "discussions" as PRs):**

```bash
# 1. Create draft PR (empty by default)
python scripts/hf_repo_git.py create-pr --repo-id org/model --title "Fix tokenizer config"

# 2. Add commits via file upload, targeting the PR ref
python scripts/hf_repo_files.py upload \
    --repo-id org/model --path tokenizer_config.json \
    --content-file ./fix.json \
    --revision refs/pr/1

# 3. Publish the draft (draft → open)
python scripts/hf_repo_git.py change-pr-status --repo-id org/model --pr-num 1 --new-status open

# 4. Merge (for your own repo)
python scripts/hf_repo_git.py merge-pr --repo-id me/my-model --pr-num 1

# Comment or close
python scripts/hf_repo_git.py comment-pr --repo-id me/my-model --pr-num 1 --comment "LGTM"
python scripts/hf_repo_git.py close-pr --repo-id me/my-model --pr-num 1
```

**Repo management:**

```bash
python scripts/hf_repo_git.py create-repo --repo-id me/my-new-model --private
python scripts/hf_repo_git.py create-repo --repo-id me/my-space --repo-type space --space-sdk gradio
python scripts/hf_repo_git.py update-repo --repo-id me/my-model --gated auto
python scripts/hf_repo_git.py update-repo --repo-id me/my-model --private false
```

## Destructive ops — caution

These mutate published state and should be confirmed with the user:
- `delete-branch`, `delete-tag`, `merge-pr`, `create-repo`, `update-repo`, `hf_repo_files.py delete`, `hf_repo_files.py upload` (can overwrite files).

The scripts don't gate them — your harness should. When in auto mode, think twice.

## Environment

- `HF_TOKEN` — required.

## Related skills

- **`hf-jobs`** — the other half of the deploy loop: train on Jobs, push artifacts with this.
- **`hf-datasets`** — for deep dataset inspection, prefer `hf-datasets` over `hf_repo_files.py read` on raw parquet.
