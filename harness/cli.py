"""Interactive CLI for the Leash agent harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from harness.agent import AgentConfig, AgentSession, RunResult
from harness.client import TokenChunk
from harness.tools import tool_names


def _load_dotenv() -> None:
    """Load repo-root .env if present (no extra dependency)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


def _colorize_projection(proj: float) -> str:
    if not sys.stdout.isatty():
        return f"{proj:+.1f}"
    t = max(0.0, min(1.0, (proj + 50.0) / 100.0))
    code = 196 if t < 0.5 else 46
    return f"\033[38;5;{code}m{proj:+.1f}\033[0m"


def _print_banner(cfg: AgentConfig) -> None:
    print("Leash Agent — interactive harness")
    print(f"  backend:  {cfg.leash_url}")
    print(f"  cwd:      {cfg.cwd}")
    print(f"  clamp:    {cfg.clamp}  thinking: {cfg.thinking}  fake tools: {cfg.fake_tools}")
    print(f"  tools:    {', '.join(tool_names(fake=cfg.fake_tools))}")
    if cfg.fake_tools:
        print("  mode:     fake tools (no side effects)")
    print("  commands: /help /exit /clamp /thinking /fake /clear /proj")
    print()


def _print_help() -> None:
    print("""
Commands:
  /help              show this help
  /exit, /quit       exit the REPL
  /clamp on|off      toggle activation capping on Leash requests
  /thinking on|off   toggle Qwen3 thinking mode
  /fake on|off       toggle no-op tools (stress testing)
  /clear             new session (keeps cwd, resets conversation)
  /proj on|off       show per-token projection while streaming

Anything else is sent to the agent.
Tools: use ```leash-tool JSON block (see system prompt).
Use --fake-tools for stress testing without real shell/file side effects.
""")


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()

    p = argparse.ArgumentParser(description="Leash agent harness CLI")
    p.add_argument(
        "--url",
        default=os.environ.get(
            "LEASH_URL",
            "https://nkasmanoff--leash-leash-chat-dev.modal.run",
        ),
        help="Leash /chat endpoint URL",
    )
    p.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory for bash commands",
    )
    p.add_argument("--clamp", action="store_true", default=_env_bool("LEASH_CLAMP"))
    p.add_argument(
        "--thinking",
        action="store_true",
        default=_env_bool("LEASH_AGENT_THINKING", default=False),
    )
    p.add_argument(
        "--fake-tools",
        action="store_true",
        default=_env_bool("LEASH_FAKE_TOOLS", default=False),
        help="Use no-op tools with plausible outputs (stress testing)",
    )
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--once", metavar="PROMPT", help="Run a single prompt and exit")
    args = p.parse_args(argv)

    trace_dir = Path(__file__).resolve().parent.parent / "traces" / "harness"

    cfg = AgentConfig(
        leash_url=args.url.rstrip("/"),
        cwd=str(Path(args.cwd).resolve()),
        clamp=args.clamp,
        thinking=args.thinking,
        fake_tools=args.fake_tools,
        max_turns=args.max_turns,
        trace_dir=trace_dir,
    )

    if args.once:
        session = AgentSession(cfg)
        turns_before = len(session.turns)
        result = session.run_user_message(args.once)
        new_turns = session.turns[turns_before:]
        for turn in new_turns:
            for tr in turn.tool_results:
                cmd = tr.get("arguments", {}).get("command") or tr.get("arguments", {}).get("path", "")
                print(f"$ {cmd}\n{tr['output']}")
        if result.reply:
            print(result.reply)
        _print_turn_footer(result, new_turns)
        return 0

    show_proj = False
    session = AgentSession(cfg)
    _print_banner(cfg)

    while True:
        try:
            line = input("leash> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        low = line.lower()
        if low in {"/exit", "/quit"}:
            break
        if low == "/help":
            _print_help()
            continue
        if low.startswith("/clamp"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                cfg.clamp = parts[1] == "on"
                print(f"clamp = {cfg.clamp}")
            else:
                print(f"clamp = {cfg.clamp}  (use: /clamp on|off)")
            continue
        if low.startswith("/thinking"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                cfg.thinking = parts[1] == "on"
                session = AgentSession(cfg)
                print(f"thinking = {cfg.thinking} (new session)")
            else:
                print(f"thinking = {cfg.thinking}  (use: /thinking on|off)")
            continue
        if low.startswith("/fake"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                cfg.fake_tools = parts[1] == "on"
                session = AgentSession(cfg)
                print(f"fake tools = {cfg.fake_tools} (new session)")
            else:
                print(f"fake tools = {cfg.fake_tools}  (use: /fake on|off)")
            continue
        if low == "/clear":
            session = AgentSession(cfg)
            print(f"new session {session.session_id}")
            continue
        if low.startswith("/proj"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in {"on", "off"}:
                show_proj = parts[1] == "on"
                print(f"projection stream = {show_proj}")
            else:
                print(f"projection stream = {show_proj}")
            continue

        print()

        def on_token(chunk: TokenChunk) -> None:
            if show_proj:
                sys.stdout.write(
                    f"\r  proj {_colorize_projection(chunk.projection)}  "
                )
                sys.stdout.flush()

        turns_before = len(session.turns)
        result = session.run_user_message(
            line, on_token=on_token if show_proj else None
        )
        if show_proj:
            print()

        new_turns = session.turns[turns_before:]
        for turn in new_turns:
            for tr in turn.tool_results:
                status = "ok" if tr["ok"] else "fail"
                args = tr.get("arguments", {})
                cmd = args.get("command") or args.get("path") or json.dumps(args)
                print(f"\n$ {cmd}")
                print(tr["output"])
                print(f"[tool {tr['name']} · {status}]")

        if result.reply:
            print(result.reply)

        _print_turn_footer(result, new_turns)
        print()

    return 0


def _print_turn_footer(result: RunResult, turns: list) -> None:
    if not turns:
        return
    total_tok = sum(int(t.stats["n_tokens"]) for t in turns)
    means = [t.stats["mean"] for t in turns if t.stats["n_tokens"]]
    mean = sum(means) / len(means) if means else 0.0
    tools = sum(len(t.tool_calls) for t in turns)
    duration = sum(t.duration_s for t in turns)
    print(
        f"\n[{len(turns)} model turn(s) · {duration:.1f}s · "
        f"{total_tok} tok · mean proj {mean:+.1f}"
        f"{'' if not tools else f' · {tools} tool call(s)'}]"
    )


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
