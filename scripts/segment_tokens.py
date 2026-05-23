"""Per-token segmentation of an agent session.

Splits each turn's tokens into 4 mutually exclusive kinds:
  - think         : inside <think>...</think>
  - tool_command  : the JSON string value of the "command" field in a leash-tool block
  - tool_other    : the remaining tokens in a ```leash-tool ... ``` block (JSON skeleton, args)
  - visible       : everything else (reply text outside think/tool blocks)

Returns a list of dicts (one per token) with cell info, segment kind, and the
projection / token text for downstream stats.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

LEASH_TOOL_RE = re.compile(
    r"```leash-tool\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
COMMAND_VALUE_RE = re.compile(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"')


def segment_turn(tokens: list[dict]) -> list[dict]:
    """Return one entry per token: {token, projection, kind, char_pos}."""
    full_text = "".join(t.get("token", "") for t in tokens)
    offsets: list[int] = [0]
    for t in tokens:
        offsets.append(offsets[-1] + len(t.get("token", "")))

    kinds: list[str] = ["visible"] * len(tokens)

    def tag_range(start_char: int, end_char: int, kind: str, *, only_overwrite: set[str] | None = None) -> None:
        for i in range(len(tokens)):
            ts, te = offsets[i], offsets[i + 1]
            if ts >= start_char and te <= end_char:
                if only_overwrite is None or kinds[i] in only_overwrite:
                    kinds[i] = kind

    for m in THINK_RE.finditer(full_text):
        tag_range(m.start(), m.end(), "think", only_overwrite={"visible"})

    for m in LEASH_TOOL_RE.finditer(full_text):
        tag_range(m.start(), m.end(), "tool_other", only_overwrite={"visible"})

    # most specific: tool_command wins over tool_other
    for m in COMMAND_VALUE_RE.finditer(full_text):
        tag_range(m.start(1), m.end(1), "tool_command",
                  only_overwrite={"tool_other", "visible"})

    out = []
    for tok, kind in zip(tokens, kinds):
        out.append({
            "token": tok.get("token", ""),
            "token_id": tok.get("token_id"),
            "projection": float(tok.get("projection", 0.0)),
            "capped": bool(tok.get("capped", False)),
            "kind": kind,
        })
    return out


def segment_session(session_dir: Path) -> list[dict]:
    """Return all tokens across all turns of a session, with kind annotations."""
    rows: list[dict] = []
    for tf in sorted(session_dir.glob("turn-*.json")):
        d = json.loads(tf.read_text())
        turn = d.get("turn")
        for entry in segment_turn(d.get("tokens", [])):
            entry["turn"] = turn
            rows.append(entry)
    return rows


def aggregate_by_kind(rows: Iterable[dict]) -> dict[str, dict]:
    """Compute per-kind projection stats."""
    by_kind: dict[str, list[float]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["projection"])
    out: dict[str, dict] = {}
    for kind, vals in by_kind.items():
        if not vals:
            continue
        n = len(vals)
        s = sum(vals)
        mean = s / n
        vmin = min(vals)
        vmax = max(vals)
        sq = sum((v - mean) ** 2 for v in vals) / n
        std = sq ** 0.5
        out[kind] = {
            "n": n,
            "mean": mean,
            "min": vmin,
            "max": vmax,
            "std": std,
        }
    return out


if __name__ == "__main__":
    import sys
    rows = segment_session(Path(sys.argv[1]))
    stats = aggregate_by_kind(rows)
    print(json.dumps(stats, indent=2))
