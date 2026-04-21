#!/usr/bin/env python3
"""
hf_jobs.py — Submit and manage HuggingFace compute jobs from the CLI.

Ported (minus sandbox/session plumbing) from huggingface/ml-intern's
jobs_tool.py. Wraps `huggingface_hub.HfApi` job endpoints.

Subcommands:
  run          Run a one-shot job (Python script or Docker command)
  ps           List running (or all) jobs
  logs         Fetch logs for a job
  inspect      Inspect job details
  cancel       Cancel a running job
  sched-run    Create a scheduled (recurring) job
  sched-ps     List scheduled jobs
  sched-inspect Inspect a scheduled job
  sched-delete Delete a scheduled job
  sched-suspend Suspend a scheduled job
  sched-resume Resume a scheduled job

Requires HF_TOKEN in the environment. Reads the user's namespace from
HF_NAMESPACE (or `whoami`).

Python script mode expects either:
  --script path-to-local-file
  --script https://example.com/script.py
  --script-stdin         (read inline Python from stdin)

Training/fine-tuning pre-flight (enforced by AGENTS.md, not this script):
  * Use github-code skill to find a working reference implementation.
  * Use hf-datasets skill to verify dataset columns.
  * Include push_to_hub=True and hub_model_id — job storage is EPHEMERAL.
  * Timeout: >= 2h for any training; default 30m kills mid-run.
  * Include Trackio monitoring and surface the dashboard URL.

Hardware (common picks):
  CPU: cpu-basic (2vCPU/16GB), cpu-upgrade (8vCPU/32GB)
  GPU: t4-small, a10g-small, a10g-large, a10g-largex2, a100-large, a100x4,
       a100x8, l40sx1, l40sx4, l40sx8
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from datetime import datetime
from typing import Any

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run `pip install huggingface-hub` or `bash bootstrap.sh`.", file=sys.stderr)
    sys.exit(2)

UV_DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"

_DEFAULT_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "UV_NO_PROGRESS": "1",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def _fmt_date(s: str | None) -> str:
    if not s:
        return "N/A"
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s


def _fmt_cmd(cmd: list[str] | None) -> str:
    return " ".join(cmd) if cmd else "N/A"


def _image_or_space(job: dict) -> str:
    return job.get("spaceId") or job.get("dockerImage") or "N/A"


def _job_info_to_dict(j) -> dict[str, Any]:
    return {
        "id": j.id,
        "status": {"stage": j.status.stage, "message": j.status.message},
        "command": j.command,
        "createdAt": j.created_at.isoformat() if j.created_at else None,
        "dockerImage": j.docker_image,
        "spaceId": j.space_id,
        "hardware_flavor": j.flavor,
        "owner": {"name": j.owner.name},
    }


def _scheduled_job_info_to_dict(s) -> dict[str, Any]:
    spec = s.job_spec
    last_run = None
    next_run = None
    if s.status:
        if s.status.last_job and s.status.last_job.created_at:
            lr = s.status.last_job.created_at
            last_run = lr.isoformat() if hasattr(lr, "isoformat") else str(lr)
        if s.status.next_job_run_at:
            nr = s.status.next_job_run_at
            next_run = nr.isoformat() if hasattr(nr, "isoformat") else str(nr)
    return {
        "id": s.id,
        "schedule": s.schedule,
        "suspend": s.suspend,
        "lastRun": last_run,
        "nextRun": next_run,
        "jobSpec": {
            "dockerImage": spec.docker_image,
            "spaceId": spec.space_id,
            "command": spec.command or [],
            "hardware_flavor": spec.flavor or "cpu-basic",
        },
    }


def _format_jobs_table(jobs: list[dict]) -> str:
    if not jobs:
        return "No jobs found."
    id_w = max(max(len(j["id"]) for j in jobs), len("JOB ID"))
    cols = {"id": id_w, "image": 20, "command": 30, "created": 19, "status": 12}
    header = (
        f"| {'JOB ID'.ljust(cols['id'])} "
        f"| {'IMAGE/SPACE'.ljust(cols['image'])} "
        f"| {'COMMAND'.ljust(cols['command'])} "
        f"| {'CREATED'.ljust(cols['created'])} "
        f"| {'STATUS'.ljust(cols['status'])} |"
    )
    sep = "|" + "|".join("-" * (cols[k] + 2) for k in ("id", "image", "command", "created", "status")) + "|"
    rows = []
    for j in jobs:
        rows.append(
            f"| {j['id'].ljust(cols['id'])} "
            f"| {_truncate(_image_or_space(j), cols['image']).ljust(cols['image'])} "
            f"| {_truncate(_fmt_cmd(j.get('command')), cols['command']).ljust(cols['command'])} "
            f"| {_truncate(_fmt_date(j.get('createdAt')), cols['created']).ljust(cols['created'])} "
            f"| {_truncate(j['status']['stage'], cols['status']).ljust(cols['status'])} |"
        )
    return "\n".join([header, sep, *rows])


def _format_scheduled_jobs_table(jobs: list[dict]) -> str:
    if not jobs:
        return "No scheduled jobs found."
    id_w = max(max(len(j["id"]) for j in jobs), len("ID"))
    cols = {"id": id_w, "schedule": 12, "image": 18, "command": 25, "lastRun": 19, "nextRun": 19, "suspend": 9}
    header = (
        f"| {'ID'.ljust(cols['id'])} "
        f"| {'SCHEDULE'.ljust(cols['schedule'])} "
        f"| {'IMAGE/SPACE'.ljust(cols['image'])} "
        f"| {'COMMAND'.ljust(cols['command'])} "
        f"| {'LAST RUN'.ljust(cols['lastRun'])} "
        f"| {'NEXT RUN'.ljust(cols['nextRun'])} "
        f"| {'SUSPENDED'.ljust(cols['suspend'])} |"
    )
    sep = "|" + "|".join(
        "-" * (cols[k] + 2)
        for k in ("id", "schedule", "image", "command", "lastRun", "nextRun", "suspend")
    ) + "|"
    rows = []
    for j in jobs:
        spec = j["jobSpec"]
        rows.append(
            f"| {j['id'].ljust(cols['id'])} "
            f"| {_truncate(j['schedule'], cols['schedule']).ljust(cols['schedule'])} "
            f"| {_truncate(_image_or_space(spec), cols['image']).ljust(cols['image'])} "
            f"| {_truncate(_fmt_cmd(spec.get('command')), cols['command']).ljust(cols['command'])} "
            f"| {_truncate(_fmt_date(j.get('lastRun')), cols['lastRun']).ljust(cols['lastRun'])} "
            f"| {_truncate(_fmt_date(j.get('nextRun')), cols['nextRun']).ljust(cols['nextRun'])} "
            f"| {('Yes' if j.get('suspend') else 'No').ljust(cols['suspend'])} |"
        )
    return "\n".join([header, sep, *rows])


def _add_default_env(params: dict | None) -> dict:
    result = dict(_DEFAULT_ENV)
    result.update(params or {})
    return result


def _add_secrets(params: dict | None, token: str | None) -> dict:
    result = dict(params or {})
    if result.get("HF_TOKEN", "").strip().startswith("$"):
        result.pop("HF_TOKEN", None)
    if token:
        result["HF_TOKEN"] = token
        result["HUGGINGFACE_HUB_TOKEN"] = token
    return result


def _ensure_hf_transfer(deps: list[str] | None) -> list[str]:
    if isinstance(deps, list):
        out = list(deps)
        if "hf-transfer" not in out:
            out.append("hf-transfer")
        return out
    return ["hf-transfer"]


def _build_uv_command(
    script: str,
    with_deps: list[str] | None,
    python: str | None,
    script_args: list[str] | None,
) -> list[str]:
    parts = ["uv", "run"]
    if with_deps:
        for d in with_deps:
            parts.extend(["--with", d])
    if python:
        parts.extend(["-p", python])
    parts.append(script)
    if script_args:
        parts.extend(script_args)
    return parts


def _wrap_inline_script(
    script: str,
    with_deps: list[str] | None,
    python: str | None,
    script_args: list[str] | None,
) -> str:
    encoded = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    cmd = " ".join(_build_uv_command("-", with_deps, python, script_args))
    return f'echo "{encoded}" | base64 -d | {cmd}'


def _resolve_uv_command(
    script: str,
    with_deps: list[str] | None,
    python: str | None,
    script_args: list[str] | None,
) -> list[str]:
    if script.startswith(("http://", "https://")):
        return _build_uv_command(script, with_deps, python, script_args)
    if "\n" in script:
        return ["/bin/sh", "-lc", _wrap_inline_script(script, with_deps, python, script_args)]
    return _build_uv_command(script, with_deps, python, script_args)


async def _async(func, *a, **kw):
    return await asyncio.to_thread(func, *a, **kw)


# --- Subcommand implementations ----------------------------------------------

async def cmd_run(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    script_content: str | None = None
    if args.script_stdin:
        script_content = sys.stdin.read()
    elif args.script:
        # if it's a local file (not URL), read it so HfApi gets its content
        if not args.script.startswith(("http://", "https://")):
            if os.path.isfile(args.script):
                with open(args.script) as f:
                    script_content = f.read()
            else:
                # treat as inline script (single line) or URL-ish
                script_content = args.script
        else:
            script_content = args.script  # URL — passed through

    command_list = args.command  # list or None

    if script_content and command_list:
        return "ERROR: --script/--script-stdin and --command are mutually exclusive.", False
    if not script_content and not command_list:
        return "ERROR: provide --script, --script-stdin, or --command.", False

    if script_content is not None:
        deps = _ensure_hf_transfer(args.dep)
        cmd = _resolve_uv_command(script_content, deps, args.python, args.script_args)
        image = args.image or UV_DEFAULT_IMAGE
        mode = "Python"
    else:
        cmd = list(command_list)
        image = args.image or "python:3.12"
        mode = "Docker"

    env = {}
    for kv in args.env or []:
        if "=" not in kv:
            return f"ERROR: invalid --env '{kv}' (expected KEY=VALUE).", False
        k, v = kv.split("=", 1)
        env[k] = v
    secrets = {}
    for kv in args.secret or []:
        if "=" not in kv:
            return f"ERROR: invalid --secret '{kv}' (expected KEY=VALUE).", False
        k, v = kv.split("=", 1)
        secrets[k] = v

    job = await _async(
        api.run_job,
        image=image,
        command=cmd,
        env=_add_default_env(env),
        secrets=_add_secrets(secrets, os.environ.get("HF_TOKEN")),
        flavor=args.hardware_flavor or "cpu-basic",
        timeout=args.timeout or "30m",
        namespace=namespace,
    )

    print(f"{mode} job started: {job.url}", file=sys.stderr)
    print("Streaming logs...", file=sys.stderr)

    all_logs: list[str] = []
    terminal = {"COMPLETED", "FAILED", "CANCELED", "ERROR"}

    if args.no_wait:
        return (
            f"{mode} job submitted.\n"
            f"**Job ID:** {job.id}\n"
            f"**View at:** {job.url}\n"
            f"Poll with: hf_jobs.py ps | hf_jobs.py logs --job-id {job.id}",
            True,
        )

    # blocking stream
    try:
        for line in api.fetch_job_logs(job_id=job.id, namespace=namespace):
            print(line)
            all_logs.append(line)
    except Exception as e:
        print(f"(log stream interrupted: {e})", file=sys.stderr)

    final_status = "UNKNOWN"
    for _ in range(6):
        info = await _async(api.inspect_job, job_id=job.id, namespace=namespace)
        final_status = info.status.stage
        if final_status in terminal:
            break
        await asyncio.sleep(2.5)

    text = _strip_ansi("\n".join(all_logs)) if all_logs else "(no logs)"
    out = (
        f"{mode} job completed.\n\n"
        f"**Job ID:** {job.id}\n"
        f"**Final Status:** {final_status}\n"
        f"**View at:** {job.url}\n\n"
        f"**Logs:**\n```\n{text}\n```"
    )
    return out, final_status == "COMPLETED"


async def cmd_ps(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    jobs = await _async(api.list_jobs, namespace=namespace)
    if not args.all:
        jobs = [j for j in jobs if j.status.stage == "RUNNING"]
    if args.status:
        sf = args.status.upper()
        jobs = [j for j in jobs if sf in j.status.stage]
    dicts = [_job_info_to_dict(j) for j in jobs]
    if not dicts:
        return ("No running jobs. Use --all to show all jobs." if not args.all else "No jobs found."), True
    return f"**Jobs ({len(dicts)}):**\n\n" + _format_jobs_table(dicts), True


async def cmd_logs(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.job_id:
        return "ERROR: --job-id is required.", False
    try:
        gen = api.fetch_job_logs(job_id=args.job_id, namespace=namespace)
        logs = await _async(list, gen)
    except Exception as e:
        return f"Failed to fetch logs: {e}", False
    if not logs:
        return f"No logs available for job {args.job_id}", True
    return f"**Logs for {args.job_id}:**\n\n```\n{_strip_ansi(chr(10).join(logs))}\n```", True


async def cmd_inspect(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.job_id:
        return "ERROR: --job-id is required.", False
    info = await _async(api.inspect_job, job_id=args.job_id, namespace=namespace)
    d = _job_info_to_dict(info)
    return f"**Job Details:**\n\n```json\n{json.dumps(d, indent=2)}\n```", True


async def cmd_cancel(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.job_id:
        return "ERROR: --job-id is required.", False
    await _async(api.cancel_job, job_id=args.job_id, namespace=namespace)
    return f"✓ Job {args.job_id} cancelled.", True


async def cmd_sched_run(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.schedule:
        return "ERROR: --schedule is required.", False
    # reuse cmd_run's logic for building command
    script_content: str | None = None
    if args.script_stdin:
        script_content = sys.stdin.read()
    elif args.script:
        if not args.script.startswith(("http://", "https://")) and os.path.isfile(args.script):
            with open(args.script) as f:
                script_content = f.read()
        else:
            script_content = args.script
    command_list = args.command
    if script_content and command_list:
        return "ERROR: --script and --command are mutually exclusive.", False
    if not script_content and not command_list:
        return "ERROR: provide --script, --script-stdin, or --command.", False

    if script_content is not None:
        deps = _ensure_hf_transfer(args.dep)
        cmd = _resolve_uv_command(script_content, deps, args.python, args.script_args)
        image = args.image or UV_DEFAULT_IMAGE
    else:
        cmd = list(command_list)
        image = args.image or "python:3.12"

    env = {}
    for kv in args.env or []:
        k, v = kv.split("=", 1)
        env[k] = v
    secrets = {}
    for kv in args.secret or []:
        k, v = kv.split("=", 1)
        secrets[k] = v

    sj = await _async(
        api.create_scheduled_job,
        image=image,
        command=cmd,
        schedule=args.schedule,
        env=_add_default_env(env),
        secrets=_add_secrets(secrets, os.environ.get("HF_TOKEN")),
        flavor=args.hardware_flavor or "cpu-basic",
        timeout=args.timeout or "30m",
        namespace=namespace,
    )
    d = _scheduled_job_info_to_dict(sj)
    return (
        f"✓ Scheduled job created.\n"
        f"**ID:** {d['id']}\n"
        f"**Schedule:** {d['schedule']}\n"
        f"**Next Run:** {d.get('nextRun', 'N/A')}\n"
        f"**Suspended:** {'Yes' if d['suspend'] else 'No'}"
    ), True


async def cmd_sched_ps(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    jobs = await _async(api.list_scheduled_jobs, namespace=namespace)
    if not args.all:
        jobs = [j for j in jobs if not j.suspend]
    dicts = [_scheduled_job_info_to_dict(j) for j in jobs]
    if not dicts:
        return ("No scheduled jobs (use --all for suspended)." if not args.all else "No scheduled jobs found."), True
    return f"**Scheduled Jobs ({len(dicts)}):**\n\n" + _format_scheduled_jobs_table(dicts), True


async def cmd_sched_inspect(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.scheduled_job_id:
        return "ERROR: --scheduled-job-id is required.", False
    sj = await _async(api.inspect_scheduled_job, scheduled_job_id=args.scheduled_job_id, namespace=namespace)
    d = _scheduled_job_info_to_dict(sj)
    return f"**Scheduled Job:**\n\n```json\n{json.dumps(d, indent=2)}\n```", True


async def cmd_sched_delete(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.scheduled_job_id:
        return "ERROR: --scheduled-job-id is required.", False
    await _async(api.delete_scheduled_job, scheduled_job_id=args.scheduled_job_id, namespace=namespace)
    return f"✓ Scheduled job {args.scheduled_job_id} deleted.", True


async def cmd_sched_suspend(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.scheduled_job_id:
        return "ERROR: --scheduled-job-id is required.", False
    await _async(api.suspend_scheduled_job, scheduled_job_id=args.scheduled_job_id, namespace=namespace)
    return f"✓ Scheduled job {args.scheduled_job_id} suspended.", True


async def cmd_sched_resume(args, api: HfApi, namespace: str | None) -> tuple[str, bool]:
    if not args.scheduled_job_id:
        return "ERROR: --scheduled-job-id is required.", False
    await _async(api.resume_scheduled_job, scheduled_job_id=args.scheduled_job_id, namespace=namespace)
    return f"✓ Scheduled job {args.scheduled_job_id} resumed.", True


CMDS = {
    "run": cmd_run, "ps": cmd_ps, "logs": cmd_logs, "inspect": cmd_inspect, "cancel": cmd_cancel,
    "sched-run": cmd_sched_run, "sched-ps": cmd_sched_ps, "sched-inspect": cmd_sched_inspect,
    "sched-delete": cmd_sched_delete, "sched-suspend": cmd_sched_suspend, "sched-resume": cmd_sched_resume,
}


def _add_run_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--script", help="Python script path / URL / inline code.")
    p.add_argument("--script-stdin", action="store_true", help="Read Python script from stdin.")
    p.add_argument("--command", nargs="+", help="Docker command (e.g. --command duckdb -c 'select 1').")
    p.add_argument("--image", help="Docker image (default: UV image for Python, python:3.12 for Docker).")
    p.add_argument("--dep", action="append", help="Pip dependency (repeatable).")
    p.add_argument("--python", help="Python version for UV (e.g. '3.12').")
    p.add_argument("--script-args", nargs="*", help="Extra args passed to the script.")
    p.add_argument("--hardware-flavor", help="Hardware flavor (default: cpu-basic).")
    p.add_argument("--timeout", help="Job timeout, e.g. '8h' (default: 30m — too short for training!).")
    p.add_argument("--env", action="append", help="Env var KEY=VALUE (repeatable).")
    p.add_argument("--secret", action="append", help="Secret KEY=VALUE (repeatable). HF_TOKEN auto-included.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hf_jobs.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a one-shot job.")
    _add_run_shared(pr)
    pr.add_argument("--no-wait", action="store_true", help="Submit and return without streaming logs.")

    pps = sub.add_parser("ps", help="List jobs.")
    pps.add_argument("--all", action="store_true", help="Include non-running jobs.")
    pps.add_argument("--status", help="Filter by status substring (e.g. FAILED).")

    pl = sub.add_parser("logs", help="Fetch logs for a job.")
    pl.add_argument("--job-id", required=True)

    pi = sub.add_parser("inspect", help="Inspect a job.")
    pi.add_argument("--job-id", required=True)

    pc = sub.add_parser("cancel", help="Cancel a running job.")
    pc.add_argument("--job-id", required=True)

    psr = sub.add_parser("sched-run", help="Create a scheduled (recurring) job.")
    _add_run_shared(psr)
    psr.add_argument("--schedule", required=True,
                     help="Cron expression or preset (@hourly, @daily, @weekly, @monthly).")

    sps = sub.add_parser("sched-ps", help="List scheduled jobs.")
    sps.add_argument("--all", action="store_true", help="Include suspended jobs.")

    for name in ("sched-inspect", "sched-delete", "sched-suspend", "sched-resume"):
        sp = sub.add_parser(name, help=f"{name} — operate on a scheduled job by ID.")
        sp.add_argument("--scheduled-job-id", required=True)

    for s in (pr, pps, pl, pi, pc, psr, sps) + tuple(
        [p_ for p_ in sub.choices.values() if p_.prog.endswith(("sched-inspect", "sched-delete", "sched-suspend", "sched-resume"))]
    ):
        s.add_argument("--json", action="store_true")

    return p


async def _run() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    namespace = os.environ.get("HF_NAMESPACE")
    if not namespace:
        try:
            namespace = api.whoami().get("name")
        except Exception as e:
            print(f"WARN: could not resolve namespace from whoami: {e}", file=sys.stderr)
            namespace = None

    handler = CMDS[args.cmd]
    try:
        out, ok = await handler(args, api, namespace)
    except Exception as e:
        out, ok = f"Error in {args.cmd}: {e}", False

    if getattr(args, "json", False):
        print(json.dumps({"cmd": args.cmd, "ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
