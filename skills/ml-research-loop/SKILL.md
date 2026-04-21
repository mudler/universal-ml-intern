---
name: ml-research-loop
description: The autonomous ML-research loop protocol. Use when driving a `program.md`-defined experiment session — this skill defines the LOOP FOREVER shape, pre-flight checks, keep/discard discipline, when to search papers, and when to spawn a research sub-agent. Pair this skill with the `/loop` harness command (or equivalent) for autonomous operation.
---

# ml-research-loop

This skill defines the **execution discipline** for an autonomous ML-research session. It is the runtime protocol side of AGENTS.md — AGENTS.md tells you *what to avoid* and *which skills exist*; this skill tells you *how to structure the loop*.

## When to use

- Operator starts an autonomous run (typically with `/loop` or equivalent).
- A `program.md` exists at the working directory root defining the task.
- You are expected to iterate without human input until the task is done or the time/iteration budget is exhausted.

## Required before starting

1. **Read `program.md`**. If absent, ask the user and offer to scaffold one. Do not start a loop without a specific task, metrics, and keep/discard criteria.
2. **Read `AGENTS.md`**. You will reference it repeatedly for pre-flight checks and error-recovery rules.
3. **Verify environment**: `HF_TOKEN`, `GITHUB_TOKEN` set; scripts in `scripts/` runnable (`python scripts/papers.py --help`).
4. **Initialize an output file** — whatever format `program.md` specifies (usually a TSV or a markdown log). Write the header row / opening heading before the first experiment.
5. **Pick a run tag / branch** — if `program.md` is iterating on a codebase, create a dedicated branch `autoresearch/<tag>` so experiments don't pollute `main`.

## The loop

```
WHILE time-budget remains AND task not complete:

    1. Look at current state
       - Read the output file (results so far)
       - Read AGENTS.md pre-flight checklist
       - Check task list (TaskCreate/TaskList in your harness) for what's next

    2. Decide the next experiment
       - Every ~5 iterations, or when stuck, delegate a literature pass to the
         research-subagent skill (keeps main context clean).
       - Otherwise: formulate a hypothesis rooted in prior results or a paper
         finding. Be specific: what change, what you predict, how you'll measure.

    3. Implement
       - Modify code / write script / prepare data.
       - For ML training jobs: pre-flight checklist (see AGENTS.md).
       - Commit the experiment's code change before running, so you can revert.

    4. Run
       - Local dev first where feasible. Use hf-jobs only when you need scale.
       - Stream logs. Capture all metrics.

    5. Evaluate
       - Extract the primary metric(s) defined in program.md.
       - Compare against the current best (not against the reference — each
         iteration competes with your own running best).

    6. Keep or discard
       - If result beats current best on primary metric: commit, update the
         output file, advance the branch.
       - Otherwise: revert the code change (git reset), log as discard.
       - Always log to the output file even for failures — "crash" and
         "build_fail" are valid outcomes worth tracking.

    7. Clean up
       - Remove temporary artifacts (quantized weights, log dumps, etc.) to
         keep disk usage bounded.
       - Mark the task completed in your task list.

    GO TO 1
```

## Discipline rules

- **Never stop working until the task is done or the budget is exhausted.** No "should I continue?" prompts. There is no human to answer.
- **Never respond with only text.** In autonomous mode, every response must include at least one tool call. A text-only response ends the loop permanently.
- **Never silently substitute** datasets, models, or methods when the requested ones fail. Log it, escalate.
- **Never change the user's stated goal** to avoid an error. Fix the error (see AGENTS.md error-recovery ladder).
- **When you run out of ideas:** go back to the literature. Crawl citation graphs deeper. Combine recipes from different papers. Re-read the training logs. There is always a paper you haven't read.
- **Reserve the last ~10% of the budget** for final evaluation and artifact saving.

## Research cadence

Good default: **every 5 experiments or whenever a streak of 3 no-improvements happens**, spawn a `research-subagent` with a focused task (e.g. "What are the best methods beating X on benchmark Y published since 2024-06? Return ranked recipe table").

Bad pattern: searching papers every single iteration. Noisy; burns context.

Worse pattern: never searching. You'll exhaust your own ideas and plateau.

## Budget awareness

Check the time budget periodically. If `program.md` specifies a wall-clock time, track it. If the harness provides a timer, read it.

When ~10% of the budget remains:
- Stop launching new experiments.
- Produce a final summary / leaderboard in the output file.
- Save any in-flight model artifacts to HF Hub (if applicable).
- Write a short retrospective: what worked, what didn't, what to try next run.

## Invocation

From the user's working directory with a `program.md`:

```
# Manual kick-off (user observes, intervenes if needed)
Read /path/to/universal-ml-intern/AGENTS.md, then read program.md, then kick off
the loop per the ml-research-loop skill.

# Fully autonomous (headless, time-boxed) — example for Claude Code:
/loop 30m — follow ml-research-loop on program.md
```

The harness's recurrence primitive (Claude Code's `/loop`, a cron job, a shell
`while` loop wrapping the CLI, or equivalent) handles *when* to iterate. This
skill defines the *shape* of each iteration.

## Related

- **AGENTS.md** — the stable contract (discipline + skills inventory + pre-flight details).
- **`program.md`** — the per-run task spec (not in this repo; lives in the user's working directory).
- **`research-subagent`** — the literature-crawl delegate for heavy research phases.
- **`hf-papers`**, **`github-code`**, **`hf-docs`**, **`hf-datasets`**, **`hf-jobs`**, **`hf-repos`** — the tool skills this loop calls.
