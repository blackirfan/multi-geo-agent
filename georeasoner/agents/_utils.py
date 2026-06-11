"""Shared utilities for agent nodes."""

from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace_entry(agent: str, tool: str, args: dict, result: str) -> dict:
    return {
        "agent": agent,
        "tool": tool,
        "args": args,
        "result": result[:300],
        "ts": now_iso(),
    }
