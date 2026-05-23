"""No-op tool handlers for stress testing — plausible results, no side effects."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from harness.tools import ToolResult


@dataclass
class FakeToolContext:
    """In-memory workspace so multi-turn tool loops stay coherent."""

    cwd: str
    files: dict[str, str] = field(default_factory=dict)

    def resolve(self, path: str) -> str:
        p = PurePosixPath(path)
        if p.is_absolute():
            return str(p)
        return str(PurePosixPath(self.cwd) / p)

    def read(self, path: str) -> str | None:
        return self.files.get(self.resolve(path))

    def write(self, path: str, content: str) -> None:
        self.files[self.resolve(path)] = content

    def delete(self, path: str) -> None:
        self.files.pop(self.resolve(path), None)

    def list_dir(self, path: str = ".") -> list[str]:
        base = self.resolve(path)
        prefix = base.rstrip("/") + "/"
        names: set[str] = set()
        for key in self.files:
            if key == base:
                continue
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                names.add(rest.split("/")[0])
        return sorted(names)


def _ok(output: str) -> ToolResult:
    return ToolResult(ok=True, output=f"{output}\n[simulated — no side effects]")


def _err(msg: str) -> ToolResult:
    return ToolResult(ok=False, output=f"error: {msg}")


def _placeholder_file(path: str) -> str:
    name = PurePosixPath(path).name
    if name.endswith(".py"):
        return (
            f'"""Placeholder content for {name}."""\n\n'
            f"def main() -> None:\n"
            f'    print("hello from {name}")\n\n\n'
            f'if __name__ == "__main__":\n'
            f"    main()\n"
        )
    if name.endswith((".json", ".jsonl")):
        return "{}\n"
    if name.endswith((".md", ".txt")):
        return f"# {name}\n\nSample content for stress testing.\n"
    return f"// placeholder contents of {name}\n"


def fake_read(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    path = args.get("path")
    if not path:
        return _err("Read requires 'path'")
    resolved = ctx.resolve(str(path))
    if resolved in ctx.files:
        return _ok(ctx.files[resolved])
    return _ok(_placeholder_file(resolved))


def fake_write(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    path = args.get("path")
    contents = args.get("contents")
    if not path:
        return _err("Write requires 'path'")
    if contents is None:
        return _err("Write requires 'contents'")
    text = str(contents)
    ctx.write(str(path), text)
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return _ok(f"Wrote {len(text)} bytes ({lines} lines) to {path}")


def fake_str_replace(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    path = args.get("path")
    old = args.get("old_string")
    new = args.get("new_string", "")
    if not path or old is None:
        return _err("StrReplace requires 'path' and 'old_string'")
    resolved = ctx.resolve(str(path))
    current = ctx.read(str(path)) or _placeholder_file(resolved)
    count = args.get("replace_all", False)
    if count:
        updated = current.replace(str(old), str(new))
        n = current.count(str(old))
    else:
        updated = current.replace(str(old), str(new), 1)
        n = 1 if str(old) in current else 0
    if n == 0:
        return _err(f"old_string not found in {path}")
    ctx.write(str(path), updated)
    return _ok(f"Replaced {n} occurrence(s) in {path}")


def fake_grep(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    pattern = args.get("pattern")
    if not pattern:
        return _err("Grep requires 'pattern'")
    path = args.get("path")
    glob_pat = args.get("glob")
    try:
        rx = re.compile(str(pattern))
    except re.error as exc:
        return _err(f"invalid pattern: {exc}")

    targets: list[tuple[str, str]] = []
    if path:
        resolved = ctx.resolve(str(path))
        targets.append((resolved, ctx.read(str(path)) or _placeholder_file(resolved)))
    else:
        for key, content in sorted(ctx.files.items()):
            name = PurePosixPath(key).name
            if glob_pat and not PurePosixPath(name).match(str(glob_pat)):
                continue
            targets.append((key, content))

    if not targets and glob_pat:
        # Pretend a few repo files exist for open-ended searches.
        for fake in ("README.md", "main.py", "harness/agent.py"):
            targets.append((ctx.resolve(fake), _placeholder_file(fake)))

    lines_out: list[str] = []
    for file_path, content in targets:
        for i, line in enumerate(content.splitlines(), 1):
            if rx.search(line):
                lines_out.append(f"{file_path}:{i}:{line}")
    if not lines_out:
        return _ok("(no matches)")
    return _ok("\n".join(lines_out[:200]))


def fake_glob(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    pattern = args.get("glob_pattern") or args.get("pattern")
    if not pattern:
        return _err("Glob requires 'glob_pattern'")
    pat = str(pattern)
    known = sorted(ctx.files.keys())
    if known:
        matched = [p for p in known if PurePosixPath(p).match(pat)]
    else:
        root = PurePosixPath(ctx.cwd)
        matched = [str(root / "README.md"), str(root / "main.py")]
    return _ok("\n".join(matched) if matched else "(no matches)")


def fake_semantic_search(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    query = args.get("query")
    if not query:
        return _err("SemanticSearch requires 'query'")
    target = args.get("target_directories") or ["."]
    hits = [
        {
            "path": ctx.resolve("harness/agent.py"),
            "snippet": "def iter_user_message(...):  # agent loop with tool execution",
        },
        {
            "path": ctx.resolve("harness/tools.py"),
            "snippet": "def execute(call, cwd, *, fake=False): ...",
        },
    ]
    return _ok(json.dumps({"query": query, "target_directories": target, "results": hits}, indent=2))


def _simulate_bash(command: str, ctx: FakeToolContext) -> ToolResult:
    cmd = command.strip()
    if not cmd:
        return _ok("(no output)")

    if cmd.startswith("cd "):
        dest = cmd[3:].strip().strip('"').strip("'")
        ctx.cwd = ctx.resolve(dest)
        return _ok("")

    if cmd in {"pwd", "pwd -P"}:
        return _ok(ctx.cwd)

    if cmd.startswith("ls"):
        path = "."
        parts = shlex.split(cmd)
        for part in parts[1:]:
            if part.startswith("-"):
                continue
            path = part
            break
        entries = ctx.list_dir(path)
        if not entries:
            return _ok("total 0")
        lines = ["total 0"]
        for name in entries:
            lines.append(f"-rw-r--r--  1 user  staff  0 May 23 12:00 {name}")
        return _ok("\n".join(lines))

    if cmd.startswith("cat "):
        path = cmd[4:].strip().strip('"').strip("'")
        content = ctx.read(path) or _placeholder_file(path)
        return _ok(content.rstrip("\n"))

    if cmd.startswith("touch "):
        path = cmd[6:].strip().strip('"').strip("'")
        if ctx.read(path) is None:
            ctx.write(path, "")
        return _ok("(no output)")

    if cmd.startswith("mkdir "):
        return _ok("(no output)")

    if " > " in cmd or cmd.startswith("echo "):
        m = re.match(r"echo\s+(.+?)\s*>\s*(.+)$", cmd)
        if m:
            content = m.group(1).strip().strip('"').strip("'")
            ctx.write(m.group(2).strip(), content + "\n")
            return _ok("(no output)")

    if cmd.startswith("python ") or cmd.startswith("python3 "):
        return _ok("")

    if cmd.startswith("git "):
        if "status" in cmd:
            return _ok("On branch main\nnothing to commit, working tree clean")
        return _ok("(no output)")

    return _ok("(no output)")


def fake_bash(args: dict, _cwd: str, ctx: FakeToolContext) -> ToolResult:
    command = args.get("command")
    if not command or not str(command).strip():
        return _err("bash requires 'command'")
    return _simulate_bash(str(command), ctx)


FAKE_HANDLERS: dict[str, Any] = {
    "bash": fake_bash,
    "Read": fake_read,
    "Write": fake_write,
    "StrReplace": fake_str_replace,
    "Grep": fake_grep,
    "Glob": fake_glob,
    "SemanticSearch": fake_semantic_search,
}


def execute_fake(
    name: str,
    arguments: dict,
    cwd: str,
    ctx: FakeToolContext,
) -> ToolResult:
    handler = FAKE_HANDLERS.get(name)
    if handler is None:
        names = ", ".join(sorted(FAKE_HANDLERS))
        return _err(f"unknown tool {name!r}. Available: {names}")
    if ctx.cwd != cwd:
        ctx.cwd = cwd
    return handler(arguments, cwd, ctx)


def fake_tool_names() -> list[str]:
    return sorted(FAKE_HANDLERS)
