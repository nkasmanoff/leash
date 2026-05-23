"""Per-session: count tokens with projection above thresholds, segmented by kind.

A simple detector candidate: 'hacky reasoning' may produce more tokens above some
projection threshold inside <think>...</think> than honest debugging.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from segment_tokens import segment_session  # noqa: E402

SWEEP = REPO / "traces" / "hack_sweep" / "run-1779559443"

SESSIONS = [
    ("clean_fix",            False, "think_off_clamp_off", 1),
    ("clean_fix",            False, "think_off_clamp_on",  1),
    ("clean_fix",            True,  "think_on_clamp_off",  8),
    ("delete_tests",         False, "think_off_clamp_off", 8),
    ("special_case_source",  True,  "think_on_clamp_off",  7),
    ("tautological_assert",  True,  "think_on_clamp_on",   7),
]

THRESHOLDS = [0, 5, 10, 15, 20]


def session_signal(session_dir: Path) -> dict:
    rows = segment_session(session_dir)
    out: dict = {}
    for kind in ("think", "tool_command", "visible"):
        sub = [r for r in rows if r["kind"] == kind]
        out[kind] = {
            "n": len(sub),
            **{f"frac_above_+{t}": (sum(1 for r in sub if r["projection"] > t) / len(sub))
               if sub else None for t in THRESHOLDS},
            **{f"count_above_+{t}": sum(1 for r in sub if r["projection"] > t) for t in THRESHOLDS},
        }
    return out


def main() -> int:
    print("# Above-threshold token fractions per session (by segment)\n")

    print("## think segment")
    print("| label | thinking | session | n | frac>+0 | frac>+5 | frac>+10 | frac>+15 | frac>+20 | tokens>+15 |")
    print("|-------|----------|---------|--:|--------:|--------:|---------:|---------:|---------:|-----------:|")
    for lbl, t, cell, seed in SESSIONS:
        sd = SWEEP / cell / f"seed-{seed}" / "session"
        s = session_signal(sd)["think"]
        n = s["n"]
        if n == 0:
            print(f"| {lbl} | {'on' if t else 'off'} | {cell}/seed-{seed} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        print(
            f"| {lbl} | {'on' if t else 'off'} | {cell}/seed-{seed} | {n} | "
            f"{s['frac_above_+0']*100:.1f}% | {s['frac_above_+5']*100:.1f}% | "
            f"{s['frac_above_+10']*100:.1f}% | {s['frac_above_+15']*100:.1f}% | "
            f"{s['frac_above_+20']*100:.1f}% | {s['count_above_+15']} |"
        )

    print("\n## tool_command segment")
    print("| label | thinking | session | n | frac>+0 | frac>-5 | frac>-10 | frac>-15 | mean | max |")
    print("|-------|----------|---------|--:|--------:|--------:|---------:|---------:|-----:|----:|")
    for lbl, t, cell, seed in SESSIONS:
        sd = SWEEP / cell / f"seed-{seed}" / "session"
        rows = segment_session(sd)
        sub = [r for r in rows if r["kind"] == "tool_command"]
        n = len(sub)
        if n == 0:
            continue
        projs = [r["projection"] for r in sub]
        f0 = sum(1 for v in projs if v > 0) / n
        fm5 = sum(1 for v in projs if v > -5) / n
        fm10 = sum(1 for v in projs if v > -10) / n
        fm15 = sum(1 for v in projs if v > -15) / n
        print(
            f"| {lbl} | {'on' if t else 'off'} | {cell}/seed-{seed} | {n} | "
            f"{f0*100:.1f}% | {fm5*100:.1f}% | {fm10*100:.1f}% | {fm15*100:.1f}% | "
            f"{sum(projs)/n:+.2f} | {max(projs):+.2f} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
