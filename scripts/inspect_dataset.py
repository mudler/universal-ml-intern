#!/usr/bin/env python3
"""
inspect_dataset.py — Comprehensive inspection of a HuggingFace dataset in one call.

Ported from huggingface/ml-intern's dataset_tools.py. Combines /is-valid, /splits,
/info, /first-rows, and /parquet endpoints of the HF datasets-server to give you:

  - Status (viewer / preview / search / filter / statistics availability)
  - Configs and splits
  - Schema (columns + types)
  - Sample rows
  - Parquet file listing (count + total size by config/split)
  - Special analysis for "messages" columns (roles, keys, tool-call presence)

REQUIRED before submitting any training job — verify column format matches the method:
  SFT  → messages, text, or prompt/completion
  DPO  → prompt, chosen, rejected
  GRPO → prompt

Supports private/gated datasets when HF_TOKEN is set.

Usage:
    inspect_dataset.py --dataset stanfordnlp/imdb
    inspect_dataset.py --dataset HuggingFaceH4/ultrachat_200k --split train_sft --sample-rows 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, TypedDict

import httpx

BASE_URL = "https://datasets-server.huggingface.co"
MAX_SAMPLE_VALUE_LEN = 150


class SplitConfig(TypedDict):
    name: str
    splits: list[str]


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _format_status(data: dict) -> str:
    available = [k for k in ("viewer", "preview", "search", "filter", "statistics") if data.get(k)]
    if available:
        return f"## Status\n✓ Valid ({', '.join(available)})"
    return "## Status\n✗ Dataset may have issues"


def _extract_configs(splits_data: dict) -> list[SplitConfig]:
    configs: dict[str, SplitConfig] = {}
    for s in splits_data.get("splits", []):
        cfg = s.get("config", "default")
        if cfg not in configs:
            configs[cfg] = {"name": cfg, "splits": []}
        configs[cfg]["splits"].append(s.get("split"))
    return list(configs.values())


def _format_structure(configs: list[SplitConfig], max_rows: int = 10) -> str:
    lines = ["## Structure (configs & splits)", "| Config | Split |", "|--------|-------|"]
    total = sum(len(c["splits"]) for c in configs)
    added = 0
    for cfg in configs:
        for split_name in cfg["splits"]:
            if added >= max_rows:
                break
            lines.append(f"| {cfg['name']} | {split_name} |")
            added += 1
        if added >= max_rows:
            break
    if total > added:
        lines.append(f"| ... | ... |  (_{added} of {total} rows_) |")
    return "\n".join(lines)


def _get_type_str(col_info: Any) -> str:
    # HF encodes nested sequence schemas as a list of sub-feature dicts,
    # e.g. messages: [{role: Value, content: Value}]. Describe as Sequence[{...}].
    if isinstance(col_info, list):
        if not col_info:
            return "Sequence[unknown]"
        first = col_info[0]
        if isinstance(first, dict):
            sub = ", ".join(f"{k}: {_get_type_str(v)}" for k, v in first.items())
            return f"Sequence[{{{sub}}}]"
        return f"Sequence[{_get_type_str(first)}]"
    if not isinstance(col_info, dict):
        return str(col_info)
    dtype = col_info.get("dtype") or col_info.get("_type", "unknown")
    if col_info.get("_type") == "ClassLabel":
        names = col_info.get("names", [])
        if names and len(names) <= 5:
            return f"ClassLabel ({', '.join(f'{n}={i}' for i, n in enumerate(names))})"
        return f"ClassLabel ({len(names)} classes)"
    if col_info.get("_type") == "Sequence":
        inner = col_info.get("feature", {})
        return f"Sequence[{_get_type_str(inner)}]"
    return str(dtype)


def _format_schema(info: dict, config: str) -> str:
    features = info.get("dataset_info", {}).get("features", {})
    lines = [f"## Schema ({config})", "| Column | Type |", "|--------|------|"]
    for col_name, col_info in features.items():
        lines.append(f"| {col_name} | {_get_type_str(col_info)} |")
    return "\n".join(lines)


def _format_messages_structure(messages_data: Any) -> str | None:
    if isinstance(messages_data, str):
        try:
            messages_data = json.loads(messages_data)
        except json.JSONDecodeError:
            return None
    if not isinstance(messages_data, list) or not messages_data:
        return None

    lines = ["## Messages Column Format"]
    roles_seen: set[str] = set()
    has_tool_calls = False
    has_tool_results = False
    message_keys: set[str] = set()

    for msg in messages_data:
        if not isinstance(msg, dict):
            continue
        message_keys.update(msg.keys())
        role = msg.get("role", "")
        if role:
            roles_seen.add(role)
        if "tool_calls" in msg or "function_call" in msg:
            has_tool_calls = True
        if role in ("tool", "function") or msg.get("tool_call_id"):
            has_tool_results = True

    lines.append(f"**Roles:** {', '.join(sorted(roles_seen)) if roles_seen else 'unknown'}")
    common = ["role", "content", "tool_calls", "tool_call_id", "name", "function_call"]
    status = [f"{k} ✓" if k in message_keys else f"{k} ✗" for k in common]
    lines.append(f"**Message keys:** {', '.join(status)}")
    if has_tool_calls:
        lines.append("**Tool calls:** ✓ Present")
    if has_tool_results:
        lines.append("**Tool results:** ✓ Present")

    # pick an illustrative example
    example = None
    fallback = None
    for msg in messages_data:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if msg.get("tool_calls") or msg.get("function_call"):
            example = msg
            break
        if role == "assistant" and example is None:
            example = msg
        elif role != "system" and fallback is None:
            fallback = msg
    if example is None:
        example = fallback

    if example:
        clean = {}
        for k, v in example.items():
            clean[k] = v[:100] + "..." if (k == "content" and isinstance(v, str) and len(v) > 100) else v
        lines.append("")
        lines.append("**Example message structure:**")
        lines.append("```json")
        lines.append(json.dumps(clean, indent=2, ensure_ascii=False))
        lines.append("```")
    return "\n".join(lines)


def _format_samples(rows_data: dict, config: str, split: str, limit: int) -> str:
    rows = rows_data.get("rows", [])[:limit]
    lines = [f"## Sample Rows ({config}/{split})"]
    messages_col_data = None
    for i, rw in enumerate(rows, 1):
        row = rw.get("row", {})
        lines.append(f"**Row {i}:**")
        for k, v in row.items():
            if k.lower() == "messages" and messages_col_data is None:
                messages_col_data = v
            val = str(v)
            if len(val) > MAX_SAMPLE_VALUE_LEN:
                val = val[:MAX_SAMPLE_VALUE_LEN] + "..."
            lines.append(f"- {k}: {val}")
    if messages_col_data is not None:
        block = _format_messages_structure(messages_col_data)
        if block:
            lines.append("")
            lines.append(block)
    return "\n".join(lines)


def _format_parquet(data: dict, max_rows: int = 10) -> str | None:
    files = data.get("parquet_files", [])
    if not files:
        return None
    groups: dict[str, dict] = {}
    for f in files:
        key = f"{f.get('config', 'default')}/{f.get('split', 'train')}"
        if key not in groups:
            groups[key] = {"count": 0, "size": 0}
        size = f.get("size") or 0
        if not isinstance(size, (int, float)):
            size = 0
        groups[key]["count"] += 1
        groups[key]["size"] += int(size)

    lines = ["## Files (Parquet)"]
    items = list(groups.items())
    shown = 0
    for key, info in items[:max_rows]:
        size_mb = info["size"] / (1024 * 1024)
        lines.append(f"- {key}: {info['count']} file(s) ({size_mb:.1f} MB)")
        shown += 1
    if len(items) > shown:
        lines.append(f"- ... (_showing {shown} of {len(items)} parquet groups_)")
    return "\n".join(lines)


async def inspect_dataset(
    dataset: str,
    config: str | None = None,
    split: str | None = None,
    sample_rows: int = 3,
    hf_token: str | None = None,
) -> tuple[str, bool]:
    headers = _auth_headers(hf_token)
    output_parts: list[str] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        is_valid_t = client.get(f"{BASE_URL}/is-valid", params={"dataset": dataset})
        splits_t = client.get(f"{BASE_URL}/splits", params={"dataset": dataset})
        parquet_t = client.get(f"{BASE_URL}/parquet", params={"dataset": dataset})

        results = await asyncio.gather(is_valid_t, splits_t, parquet_t, return_exceptions=True)

        if not isinstance(results[0], Exception):
            try:
                output_parts.append(_format_status(results[0].json()))
            except Exception as e:
                errors.append(f"is-valid: {e}")

        configs: list[SplitConfig] = []
        if not isinstance(results[1], Exception):
            try:
                splits_data = results[1].json()
                configs = _extract_configs(splits_data)
                if not config:
                    config = configs[0]["name"] if configs else "default"
                if not split:
                    split = configs[0]["splits"][0] if configs else "train"
                output_parts.append(_format_structure(configs))
            except Exception as e:
                errors.append(f"splits: {e}")

        if not config:
            config = "default"
        if not split:
            split = "train"

        parquet_section = None
        if not isinstance(results[2], Exception):
            try:
                parquet_section = _format_parquet(results[2].json())
            except Exception:
                pass

        info_t = client.get(f"{BASE_URL}/info", params={"dataset": dataset, "config": config})
        rows_t = client.get(
            f"{BASE_URL}/first-rows",
            params={"dataset": dataset, "config": config, "split": split},
            timeout=30,
        )
        content = await asyncio.gather(info_t, rows_t, return_exceptions=True)

        if not isinstance(content[0], Exception):
            try:
                output_parts.append(_format_schema(content[0].json(), config))
            except Exception as e:
                errors.append(f"info: {e}")
        if not isinstance(content[1], Exception):
            try:
                output_parts.append(_format_samples(content[1].json(), config, split, sample_rows))
            except Exception as e:
                errors.append(f"rows: {e}")
        if parquet_section:
            output_parts.append(parquet_section)

    formatted = f"# {dataset}\n\n" + "\n\n".join(output_parts)
    if errors:
        formatted += f"\n\n**Warnings:** {'; '.join(errors)}"
    return formatted, bool(output_parts)


def main() -> int:
    p = argparse.ArgumentParser(prog="inspect_dataset.py", description=__doc__)
    p.add_argument("--dataset", required=True, help="Dataset ID, e.g. 'stanfordnlp/imdb'.")
    p.add_argument("--config", help="Config / subset name. Auto-detected if omitted.")
    p.add_argument("--split", help="Split. Auto-detected if omitted.")
    p.add_argument("--sample-rows", type=int, default=3,
                   help="Number of sample rows to show (default: 3, max: 10).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    out, ok = asyncio.run(inspect_dataset(
        dataset=args.dataset, config=args.config, split=args.split,
        sample_rows=min(args.sample_rows, 10), hf_token=hf_token,
    ))
    if args.json:
        print(json.dumps({"ok": ok, "output": out}, ensure_ascii=False))
    else:
        print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
