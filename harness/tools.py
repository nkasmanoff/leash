"""Tool registry and bash execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from harness.parse import ToolCall

if TYPE_CHECKING:
    from harness.fake_tools import FakeToolContext


@dataclass
class ToolResult:
    ok: bool
    output: str


ToolHandler = Callable[[dict, str], ToolResult]


def run_bash(args: dict, cwd: str) -> ToolResult:
    command = args.get("command")
    if not command or not str(command).strip():
        return ToolResult(ok=False, output="error: bash requires 'command'")

    timeout_s = int(args.get("timeout", 120))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output=f"error: command timed out after {timeout_s}s")

    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(proc.stderr.rstrip())
    if proc.returncode != 0:
        parts.append(f"[exit {proc.returncode}]")
    body = "\n".join(parts) if parts else "(no output)"
    ok = proc.returncode == 0
    return ToolResult(ok=ok, output=body)


_REGISTRY: dict[str, ToolHandler] = {
    "bash": run_bash,
}


def register(name: str, handler: ToolHandler) -> None:
    _REGISTRY[name] = handler


def execute(
    call: ToolCall,
    cwd: str,
    *,
    fake: bool = False,
    ctx: FakeToolContext | None = None,
) -> ToolResult:
    if fake:
        from harness.fake_tools import FakeToolContext, execute_fake

        state = ctx or FakeToolContext(cwd=cwd)
        return execute_fake(call.name, call.arguments, cwd, state)
    handler = _REGISTRY.get(call.name)
    if handler is None:
        names = ", ".join(sorted(_REGISTRY))
        return ToolResult(
            ok=False,
            output=f"error: unknown tool {call.name!r}. Available: {names}",
        )
    return handler(call.arguments, cwd)


def tool_names(*, fake: bool = False) -> list[str]:
    if fake:
        from harness.fake_tools import fake_tool_names

        return fake_tool_names()
    return sorted(_REGISTRY)
