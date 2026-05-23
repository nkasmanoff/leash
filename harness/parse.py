"""Parse tool calls from Qwen / Leash model text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

THINKING_START = "<think>"
THINKING_END = "</think>"
IM_END = "<|im_end|>"


@dataclass
class ToolCall:
    name: str
    arguments: dict


LEASH_TOOL_RE = re.compile(
    r"```leash-tool\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

# Fallbacks for common Qwen / OpenCode-style hallucinations
TOOL_CODE_RE = re.compile(
    r"```tool_code\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

BACKTICK_CMD_RE = re.compile(
    r"\[opencode\]\s*Running\s+`([^`]+)`",
    re.IGNORECASE,
)

FAKE_CREATE_FILE_RE = re.compile(
    r'create_file\s*\(\s*filename\s*=\s*["\']([^"\']+)["\']\s*,\s*content\s*=\s*["\']([^"\']*)["\']\s*\)',
    re.IGNORECASE,
)


def strip_model_artifacts(text: str) -> str:
    """Remove thinking blocks and Qwen end markers for display."""
    out = text
    while THINKING_START in out:
        start = out.find(THINKING_START)
        end = out.find(THINKING_END, start)
        if end == -1:
            out = out[:start]
            break
        out = out[:start] + out[end + len(THINKING_END) :]
    return out.replace(IM_END, "").strip()


def visible_reply(text: str) -> str:
    """Text shown to user after stripping tools and thinking."""
    cleaned = strip_model_artifacts(text)
    cleaned = LEASH_TOOL_RE.sub("", cleaned)
    cleaned = TOOL_CODE_RE.sub("", cleaned)
    return cleaned.strip()


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []

    for match in LEASH_TOOL_RE.finditer(text):
        call = _parse_json_tool(match.group(1).strip())
        if call:
            calls.append(call)

    if calls:
        return calls

    for match in TOOL_CODE_RE.finditer(text):
        call = _parse_loose_tool_line(match.group(1).strip())
        if call:
            calls.append(call)

    if calls:
        return calls

    for match in FAKE_CREATE_FILE_RE.finditer(text):
        filename, content = match.group(1), match.group(2)
        calls.append(
            ToolCall(
                name="bash",
                arguments={
                    "command": _write_file_command(filename, content),
                    "description": f"create file {filename}",
                },
            )
        )

    if calls:
        return calls

    match = BACKTICK_CMD_RE.search(text)
    if match:
        calls.append(
            ToolCall(
                name="bash",
                arguments={
                    "command": match.group(1).strip(),
                    "description": "run shell command",
                },
            )
        )

    return calls


def _parse_json_tool(raw: str) -> ToolCall | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_loose_tool_line(raw)
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("tool")
    args = data.get("arguments") or data.get("args") or {}
    if not name or not isinstance(args, dict):
        return None
    return ToolCall(name=str(name), arguments=args)


def _write_file_command(filename: str, content: str) -> str:
    import shlex

    path = shlex.quote(filename)
    if not content:
        return f": > {path}"
    return f"cat > {path} << 'LEASH_EOF'\n{content}\nLEASH_EOF"


def _parse_loose_tool_line(line: str) -> ToolCall | None:
    """Parse create_file(...) or name(args) style lines."""
    m = FAKE_CREATE_FILE_RE.search(line)
    if m:
        filename, content = m.group(1), m.group(2)
        return ToolCall(
            name="bash",
            arguments={
                "command": _write_file_command(filename, content),
                "description": f"create file {filename}",
            },
        )
    # write("path", "content") style
    m2 = re.match(r"(\w+)\s*\((.*)\)\s*$", line, re.DOTALL)
    if not m2:
        return None
    name, inner = m2.group(1), m2.group(2)
    if name in {"bash", "run", "shell"}:
        cmd = inner.strip().strip('"').strip("'")
        return ToolCall(
            name="bash",
            arguments={"command": cmd, "description": "run shell command"},
        )
    return None
