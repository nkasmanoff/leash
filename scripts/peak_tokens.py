"""Find the highest-projection tokens in a session and dump context windows.

Usage: peak_tokens.py <session_dir> [--top N] [--window K]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from segment_tokens import segment_session  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("session_dir")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--window", type=int, default=20, help="tokens of context on each side")
    p.add_argument("--kind", default=None, help="filter by segment kind (think|tool_command|...)")
    p.add_argument("--direction", default="max", choices=["max", "min"])
    args = p.parse_args()

    rows = segment_session(Path(args.session_dir))
    if args.kind:
        idxs_of_interest = [i for i, r in enumerate(rows) if r["kind"] == args.kind]
    else:
        idxs_of_interest = list(range(len(rows)))

    sub = [(i, rows[i]) for i in idxs_of_interest]
    if args.direction == "max":
        sub.sort(key=lambda x: x[1]["projection"], reverse=True)
    else:
        sub.sort(key=lambda x: x[1]["projection"])

    seen_ranges: list[tuple[int, int]] = []
    printed = 0
    for i, r in sub:
        # skip if too close to a previously printed peak
        if any(abs(i - mid) < args.window for mid, _ in seen_ranges):
            continue
        seen_ranges.append((i, i))

        lo = max(0, i - args.window)
        hi = min(len(rows), i + args.window + 1)
        before = "".join(rows[j]["token"] for j in range(lo, i))
        target = rows[i]["token"]
        after = "".join(rows[j]["token"] for j in range(i + 1, hi))

        print(f"\n--- peak #{printed+1}: turn={r['turn']} kind={r['kind']} proj={r['projection']:+.2f} token={target!r} ---")
        ctx = f"{before}«{target}»{after}".replace("\n", "\\n")
        print(ctx[:600])
        printed += 1
        if printed >= args.top:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
