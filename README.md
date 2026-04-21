# universal-ml-intern

A harness-agnostic port of Hugging Face's [ml-intern](https://github.com/huggingface/ml-intern) — the prompts, skills, and tools it uses to autonomously research, implement, and run ML experiments, packaged as **plain scripts and skill files** so any agentic CLI that honors the [AGENTS.md](https://agents.md) convention (Claude Code, Codex, Cursor, Aider, and others) can drive them.

## What this is and isn't

**Is:** AGENTS.md + skill definitions + ported Python scripts for paper search, citation graphs, GitHub code discovery, HF docs/datasets/jobs/repos. Designed to be cloned into *any* project and loaded by any harness that reads `AGENTS.md` at repo root. A `CLAUDE.md` symlink is included so Claude Code also picks it up automatically.

**Isn't:** an agent loop, a sandbox, a chat UI, or a bespoke LLM client. Those already exist in your harness.

## Layout

```
universal-ml-intern/
├── AGENTS.md                 # the contract — read this first
├── README.md
├── pyproject.toml            # Python deps (httpx, bs4, huggingface_hub, …)
├── bootstrap.sh              # uv sync + optional skills symlink (Claude Code: ~/.claude/skills)
├── CLAUDE.md → AGENTS.md     # symlink so Claude Code auto-loads the same contract
├── .env.example              # HF_TOKEN, GITHUB_TOKEN, S2_API_KEY
│
├── skills/                   # markdown skill definitions, one dir per skill
│   ├── hf-papers/SKILL.md
│   ├── github-code/SKILL.md
│   ├── hf-docs/SKILL.md
│   ├── hf-datasets/SKILL.md
│   ├── hf-jobs/SKILL.md
│   ├── hf-repos/SKILL.md
│   ├── research-subagent/SKILL.md
│   └── ml-research-loop/SKILL.md
│
└── scripts/                  # the actual tool implementations
    ├── papers.py             # 11 ops: trending, search, read_paper, citation_graph, …
    ├── find_examples.py      # fuzzy file-search in GitHub repos
    ├── read_file.py          # GitHub file read with line ranges + ipynb→md
    ├── list_repos.py         # list+sort user/org repos
    ├── hf_docs.py            # explore + fetch HF library docs
    ├── inspect_dataset.py    # one-shot HF dataset inspector
    ├── hf_jobs.py            # submit / monitor HF jobs
    ├── hf_repo_files.py      # list / read / upload / delete on HF repos
    └── hf_repo_git.py        # branches / tags / PRs / repo mgmt on HF repos
```

## The `program.md` convention

AGENTS.md defines the generic contract. Each specific experiment run defines its own `program.md` at the working directory root — goal, metrics, keep/discard criteria, output format. AGENTS.md tells the agent to honor it if present.

Minimal example:

```markdown
# program.md — quantization sweep, june-run-1

## Goal
Find a quantization scheme that beats Q4_K_M perplexity on wikitext-2
at ≤80% of its file size, on Qwen3.5-35B-A3B.

## Output
Append each experiment to `results.tsv` with columns:
commit  perplexity  size_mb  tokens_per_sec  quant_type  status  description
```

The agent reads AGENTS.md for the loop protocol and `program.md` for *this run's* task.

## Install

```bash
git clone git@github.com:mudler/universal-ml-intern.git
cd universal-ml-intern
bash bootstrap.sh                    # installs deps, creates .env, smoke-tests scripts
$EDITOR .env                         # fill in HF_TOKEN, GITHUB_TOKEN
```

Or skip manual setup and let the agent bootstrap itself — just open your agentic CLI in the cloned repo and ask it to read `AGENTS.md`. The `Bootstrap` section of AGENTS.md instructs the agent to detect a missing environment and run `bash bootstrap.sh` on its own. You'll still need to provide the tokens.

### Running against a task

```bash
# In your project root (wherever the experiment code lives):
#   1. Create a program.md describing the task, metrics, and output format.
#   2. Point your agent at the AGENTS.md contract:
@AGENTS.md at /path/to/universal-ml-intern/AGENTS.md
```

### Skills are used in-place — no install needed

The skill definitions live in this repo under `skills/<name>/SKILL.md`. AGENTS.md lists every skill with its purpose and a relative path, and the agent loads them on demand via its file-reading tool. **Nothing to install for this to work** — any harness that honors `AGENTS.md` is ready as soon as the repo is cloned and `bootstrap.sh` has been run.

If your harness auto-loads a per-project memory file, the `CLAUDE.md` symlink (pointing to `AGENTS.md`) means Claude Code picks up the contract without explicit instruction. Codex, Cursor, Aider, and similar harnesses read `AGENTS.md` directly.

### Optional: register skills for harness auto-discovery

Only useful if your harness has a separate *auto-discovery* mechanism for skills (Claude Code does: skills under `~/.claude/skills/` or `<project>/.claude/skills/` are auto-invoked by description match). This is a UX convenience, **not a requirement** — the agent can already find and use every skill from the repo tree.

Pick one scope:

```bash
# Per-project (recommended — scoped to one working directory):
bash bootstrap.sh --link-skills --target /path/to/your/project/.claude/skills

# User-wide (applies to every project):
bash bootstrap.sh --link-skills                  # → ~/.claude/skills
```

## Credit

All tool implementations are adapted from [huggingface/ml-intern](https://github.com/huggingface/ml-intern) (Apache 2.0). The AGENTS.md discipline section is distilled from ml-intern's `system_prompt_v3.yaml`. This repo removes the bespoke agent loop, MCP client, sandbox, and UI — any AGENTS.md-compatible harness provides those for free.

## License

MIT for the harness-agnostic repackaging. Upstream scripts retain ml-intern's Apache 2.0 license.
