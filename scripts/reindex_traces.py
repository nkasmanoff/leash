#!/usr/bin/env python3
"""Rebuild traces/harness/index.json from on-disk trace files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_store import TraceStore  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "traces" / "harness"
    store = TraceStore(root)
    turns = store.list_turns()
    store._write_index()
    print(f"Indexed {len(turns)} turn(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
