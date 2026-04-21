#!/usr/bin/env bash
# bootstrap.sh — one-time setup for universal-ml-intern
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "=== universal-ml-intern bootstrap ==="
echo "Repo: $REPO_DIR"
echo ""

# --- 1. Check Python ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+." >&2
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PY_VERSION"

# --- 2. Install Python dependencies ---
if command -v uv &>/dev/null; then
    echo "Using uv to install dependencies..."
    uv sync --extra docs
else
    echo "uv not found — falling back to pip. Install uv for faster installs: https://docs.astral.sh/uv/"
    python3 -m pip install --upgrade pip
    python3 -m pip install -e ".[docs]"
fi
echo ""

# --- 3. Create .env if it doesn't exist ---
if [ ! -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    echo "Created .env from template — edit it to add your tokens."
else
    echo ".env already exists — not overwriting."
fi
echo ""

# --- 4. Optional: symlink skills into a harness's skills dir ---
# Skills already live in this repo under skills/<name>/SKILL.md and work
# directly via AGENTS.md references — this symlink step is only needed if
# your harness has a separate auto-discovery mechanism (e.g. Claude Code's
# Skill tool auto-invokes skills registered in known skill directories).
#
# Two common targets:
#   Per-project:   <your-project>/.claude/skills     (recommended for scoped use)
#   User-wide:     ~/.claude/skills                  (default — applies everywhere)
#
# Usage:
#   bash bootstrap.sh --link-skills                        # → ~/.claude/skills
#   bash bootstrap.sh --link-skills --target <path>        # → <path>
#   SKILLS_DIR=/path bash bootstrap.sh --link-skills       # env-var form
LINK_MODE=""
SKILLS_DIR_ARG=""
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --link-skills|--link-claude) LINK_MODE="1" ;;
        --target)
            i=$((i + 1))
            SKILLS_DIR_ARG="${!i}"
            ;;
    esac
    i=$((i + 1))
done
SKILLS_DIR="${SKILLS_DIR_ARG:-${SKILLS_DIR:-$HOME/.claude/skills}}"

if [ "$LINK_MODE" = "1" ] || [ "${LINK_SKILLS:-}" = "1" ] || [ "${LINK_CLAUDE_SKILLS:-}" = "1" ]; then
    mkdir -p "$SKILLS_DIR"
    echo "Linking skills into $SKILLS_DIR"
    for skill in "$REPO_DIR"/skills/*/; do
        skill_name=$(basename "$skill")
        target="$SKILLS_DIR/$skill_name"
        if [ -L "$target" ] || [ -e "$target" ]; then
            echo "  ⚠ $target already exists — skipping."
        else
            ln -s "$skill" "$target"
            echo "  ✓ Linked $skill_name → $target"
        fi
    done
else
    echo "Skipping skills symlink (optional — skills work via AGENTS.md references without it)."
    echo "To enable auto-discovery:"
    echo "  Per-project:   bash bootstrap.sh --link-skills --target <project>/.claude/skills"
    echo "  User-wide:     bash bootstrap.sh --link-skills   (→ $HOME/.claude/skills)"
fi
echo ""

# --- 5. Smoke-test scripts ---
echo "--- Smoke-testing scripts ---"
ALL_OK=true
for script in papers.py find_examples.py read_file.py list_repos.py hf_docs.py inspect_dataset.py hf_jobs.py hf_repo_files.py hf_repo_git.py; do
    path="$REPO_DIR/scripts/$script"
    if [ ! -f "$path" ]; then
        echo "  ⚠ $script missing (not yet ported)"
        continue
    fi
    if python3 "$path" --help &>/dev/null; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script failed --help"
        ALL_OK=false
    fi
done
echo ""

if [ "$ALL_OK" = true ]; then
    echo "=== Bootstrap complete. ==="
    echo "Next steps:"
    echo "  1. Edit .env with your tokens."
    echo "  2. Point your agent at $REPO_DIR/AGENTS.md"
    echo "  3. In the working directory of a specific run, add a program.md"
else
    echo "=== Bootstrap finished with warnings — see above. ==="
    exit 1
fi
