"""Re-run a hand-picked list of (cell, seed) pairs into the same sweep dir.

Safer defaults than the original sweep:
  - workers=2 (avoid OOM cascades)
  - max_new_tokens=2048 (faster, less KV-cache pressure)
  - timeout=1800s (more headroom for thinking mode)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.run_hack_sweep import CELLS, PROMPT, make_demo_dir  # noqa: E402

RUN_DIR = REPO / "traces" / "hack_sweep" / "run-1779559443"

FAILED = [
    ("think_on_clamp_off", 0),
    ("think_on_clamp_off", 1),
    ("think_on_clamp_off", 2),
    ("think_on_clamp_off", 3),
    ("think_on_clamp_off", 4),
    ("think_on_clamp_off", 5),
    ("think_on_clamp_off", 6),
    ("think_on_clamp_off", 8),
    ("think_on_clamp_off", 9),
    ("think_on_clamp_on", 5),
    ("think_on_clamp_on", 6),
]


def cell_by_name(name: str) -> dict:
    for c in CELLS:
        if c["name"] == name:
            return c
    raise KeyError(name)


def run_one(cell_name: str, seed: int, max_new_tokens: int = 2048) -> dict:
    cell = cell_by_name(cell_name)
    cell_dir = RUN_DIR / cell_name / f"seed-{seed}"

    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)

    print(f"[start] {cell_name} seed={seed}", flush=True)
    demo = make_demo_dir(cell_name, seed)
    prompt = PROMPT.format(cwd=str(demo))

    runner = REPO / "scripts" / "_run_one_session.py"
    args = [
        sys.executable, str(runner),
        "--cwd", str(demo),
        "--seed", str(seed),
        "--max-new-tokens", str(max_new_tokens),
        "--max-turns", "20",
        "--prompt", prompt,
        "--out", str(cell_dir),
    ]
    if cell["thinking"]:
        args.append("--thinking")
    if cell["clamp"]:
        args.append("--clamp")

    env = os.environ.copy()
    env["LEASH_FAKE_TOOLS"] = "false"

    t0 = time.time()
    try:
        proc = subprocess.run(
            args, env=env, cwd=str(REPO),
            capture_output=True, text=True, timeout=1800,
        )
        duration = time.time() - t0
        (cell_dir / "stdout.log").write_text(proc.stdout)
        (cell_dir / "stderr.log").write_text(proc.stderr)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        exit_code = "TIMEOUT_1800s"
        (cell_dir / "stderr.log").write_text("TIMEOUT after 1800s\n")

    pytest_proc = subprocess.run(
        ["pytest", "--tb=short", "-v"],
        cwd=str(demo),
        capture_output=True, text=True, timeout=60,
    )
    (cell_dir / "pytest.log").write_text(pytest_proc.stdout + "\n" + pytest_proc.stderr)

    snap = cell_dir / "demo_snapshot"
    snap.mkdir(exist_ok=True)
    for fname in ("discount.py", "test_discount.py", "README.md"):
        src = demo / fname
        if src.exists():
            shutil.copy2(src, snap / fname)

    print(f"[done ] {cell_name} seed={seed} duration={duration:.0f}s exit={exit_code} pytest_exit={pytest_proc.returncode}", flush=True)
    return {"cell": cell_name, "seed": seed, "exit": exit_code, "pytest_exit": pytest_proc.returncode}


def main() -> int:
    print(f"[rerun] {len(FAILED)} failed seeds, workers=2", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one, c, s) for c, s in FAILED]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as exc:
                print(f"[error] {exc}", flush=True)

    elapsed = time.time() - t0
    (RUN_DIR / "rerun_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[rerun] DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
