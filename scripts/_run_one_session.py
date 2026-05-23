"""Run a single agent session with explicit seed/max_new_tokens.

Used by run_hack_sweep.py as a subprocess so each run gets a clean Python
interpreter and the harness's session_id naming.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness.agent import AgentConfig, AgentSession  # noqa: E402


def _load_dotenv() -> None:
    env_path = REPO / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cwd", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--clamp", action="store_true")
    args = p.parse_args()

    _load_dotenv()
    leash_url = os.environ.get(
        "LEASH_URL",
        "https://nkasmanoff--leash-leash-chat-dev.modal.run",
    ).rstrip("/")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = AgentConfig(
        leash_url=leash_url,
        cwd=args.cwd,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        thinking=args.thinking,
        clamp=args.clamp,
        fake_tools=False,
        trace_dir=REPO / "traces" / "harness",
    )

    session = AgentSession(cfg)
    t0 = time.time()
    result = session.run_user_message(args.prompt)
    duration = time.time() - t0

    src = REPO / "traces" / "harness" / session.session_id
    if src.exists():
        shutil.copytree(src, out / "session", dirs_exist_ok=True)

    turn_files = sorted((out / "session").glob("turn-*.json")) if (out / "session").exists() else []
    tool_calls, per_turn, all_tokens = [], [], []
    for tf in turn_files:
        d = json.loads(tf.read_text())
        for tc in d.get("tool_calls", []):
            tool_calls.append({
                "turn": d.get("turn"),
                "name": tc.get("name"),
                "command": (tc.get("arguments", {}).get("command") or "")[:600],
            })
        s = d.get("stats", {})
        per_turn.append({
            "turn": d.get("turn"),
            "n_tokens": s.get("n_tokens", 0),
            "mean_proj": s.get("mean", 0.0),
            "min_proj": s.get("min", 0.0),
            "max_proj": s.get("max", 0.0),
            "duration_s": d.get("duration_s", 0.0),
        })
        for t in d.get("tokens", []):
            all_tokens.append(t.get("projection", 0.0))

    summary = {
        "session_id": session.session_id,
        "thinking": args.thinking,
        "clamp": args.clamp,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "duration_s": duration,
        "n_turns": len(turn_files),
        "n_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "per_turn_stats": per_turn,
    }
    if all_tokens:
        summary["all_tokens"] = {
            "n": len(all_tokens),
            "mean": sum(all_tokens) / len(all_tokens),
            "min": min(all_tokens),
            "max": max(all_tokens),
        }
    summary["final_reply"] = result.reply[:2000] if result.reply else ""
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
