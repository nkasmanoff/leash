"""Project per-session decision-token activations onto every axis in the
role/trait vector library at the target layer.

Inputs:
  --activations-dir   directory of <cell>__seed-<n>.npz files
                      (output of scripts/extract_activations.py)
  --axes-dir          directory containing the qwen-3-32b/ subtree
                      (output of scripts/download_axes.py)
  --layer             axis row to use (must match extraction layer)

Outputs (under --out-dir, default = activations-dir/../projections):
  axes_index.json           list of {axis_name, axis_kind, file}
  projections_by_session.parquet
                            row per (cell, seed, axis_name)
                            columns: cell, seed, axis_kind, axis_name, n,
                                     mean, std, abs_mean, p10, p50, p90
  projections_by_kind.parquet
                            row per (cell, seed, axis_name, kind)
                            columns: cell, seed, axis_kind, axis_name, kind,
                                     n, mean, std

Falls back to CSV (gzipped) if pyarrow / fastparquet is not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AXES = Path.home() / ".cache" / "leash-axes" / "qwen-3-32b"
DEFAULT_ACT = REPO / "traces" / "hack_sweep" / "run-1779559443" / "activations"


def _try_torch_load(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def load_axes(axes_dir: Path, layer: int) -> tuple[np.ndarray, list[dict]]:
    """Return (axes [n_axes, hidden_dim] float32, index list)."""
    import torch

    entries: list[tuple[str, str, Path]] = []
    for kind_dir, kind in [("role_vectors", "role"), ("trait_vectors", "trait")]:
        for p in sorted((axes_dir / kind_dir).glob("*.pt")):
            entries.append((p.stem, kind, p))
    for special in ["assistant_axis", "default_vector"]:
        p = axes_dir / f"{special}.pt"
        if p.exists():
            entries.append((special, "special", p))

    print(f"loading {len(entries)} axes at layer {layer} ...")
    rows: list[np.ndarray] = []
    index: list[dict] = []
    t0 = time.time()
    for i, (name, kind, path) in enumerate(entries):
        t = _try_torch_load(path)
        if isinstance(t, dict):
            for v in t.values():
                if hasattr(v, "shape"):
                    t = v
                    break
        if not hasattr(t, "shape"):
            print(f"  skip {name}: unknown payload type {type(t).__name__}")
            continue
        if t.ndim != 2:
            print(f"  skip {name}: shape {tuple(t.shape)} (expected 2D)")
            continue
        if layer >= t.shape[0]:
            print(f"  skip {name}: shape {tuple(t.shape)} has no layer {layer}")
            continue
        v = t[layer].to(torch.float32).cpu().numpy()
        n = float(np.linalg.norm(v))
        if n == 0.0:
            print(f"  skip {name}: zero norm at layer {layer}")
            continue
        rows.append((v / n).astype(np.float32))
        index.append(
            {"axis_name": name, "axis_kind": kind, "file": str(path), "raw_norm": n}
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(entries)} loaded ({time.time() - t0:.1f}s)")
    if not rows:
        raise RuntimeError("no axes loaded")
    arr = np.stack(rows).astype(np.float32)
    print(f"axes matrix shape={arr.shape} (n_axes, hidden_dim)")
    return arr, index


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", default=str(DEFAULT_ACT))
    p.add_argument("--axes-dir", default=str(DEFAULT_AXES))
    p.add_argument("--layer", type=int, default=32)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    act_dir = Path(args.activations_dir)
    axes_dir = Path(args.axes_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else act_dir.parent / "projections"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    axes, index = load_axes(axes_dir, args.layer)
    (out_dir / "axes_index.json").write_text(json.dumps(index, indent=2))

    session_files = sorted(act_dir.glob("*.npz"))
    print(f"sessions: {len(session_files)}")

    by_session_rows: list[dict] = []
    by_kind_rows: list[dict] = []

    t0 = time.time()
    for i, sf in enumerate(session_files):
        d = np.load(sf, allow_pickle=False)
        a = d["activations"].astype(np.float32)  # [n_positions, hidden_dim]
        if a.shape[1] != axes.shape[1]:
            raise RuntimeError(
                f"hidden_dim mismatch: session={a.shape[1]} axes={axes.shape[1]}"
            )
        kinds_b = d["kind"]
        kinds = [
            (k.decode("utf-8") if isinstance(k, (bytes, np.bytes_)) else str(k))
            for k in kinds_b.tolist()
        ]
        proj = a @ axes.T  # [n_positions, n_axes]

        cell = str(d["cell"].item()) if d["cell"].shape == () else str(d["cell"])
        seed = int(d["seed"].item()) if d["seed"].shape == () else int(d["seed"])

        # per-session aggregate (over all decision tokens)
        mean = proj.mean(axis=0)
        std = proj.std(axis=0)
        absmean = np.abs(proj).mean(axis=0)
        p10 = np.percentile(proj, 10, axis=0)
        p50 = np.percentile(proj, 50, axis=0)
        p90 = np.percentile(proj, 90, axis=0)
        n_pos = proj.shape[0]

        for j, ax in enumerate(index):
            by_session_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "axis_kind": ax["axis_kind"],
                    "axis_name": ax["axis_name"],
                    "n": n_pos,
                    "mean": float(mean[j]),
                    "std": float(std[j]),
                    "abs_mean": float(absmean[j]),
                    "p10": float(p10[j]),
                    "p50": float(p50[j]),
                    "p90": float(p90[j]),
                }
            )

        # per-kind aggregate
        unique_kinds = sorted(set(kinds))
        for k in unique_kinds:
            mask = np.array([kk == k for kk in kinds])
            if not mask.any():
                continue
            sub = proj[mask]
            sm = sub.mean(axis=0)
            ss = sub.std(axis=0)
            n_k = int(mask.sum())
            for j, ax in enumerate(index):
                by_kind_rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "axis_kind": ax["axis_kind"],
                        "axis_name": ax["axis_name"],
                        "kind": k,
                        "n": n_k,
                        "mean": float(sm[j]),
                        "std": float(ss[j]),
                    }
                )

        if (i + 1) % 5 == 0 or i == len(session_files) - 1:
            print(
                f"  [{i + 1}/{len(session_files)}] {sf.name} "
                f"({n_pos} pos, {time.time() - t0:.1f}s)"
            )

    _write_table(by_session_rows, out_dir / "projections_by_session")
    _write_table(by_kind_rows, out_dir / "projections_by_kind")
    print(f"wrote outputs to {out_dir}")
    return 0


def _write_table(rows: list[dict], stem: Path) -> None:
    if not rows:
        print(f"  empty table -> skip {stem}")
        return
    try:
        import pandas as pd  # noqa

        df = pd.DataFrame(rows)
        try:
            df.to_parquet(stem.with_suffix(".parquet"), index=False)
            print(f"  wrote {stem}.parquet  ({len(df):,} rows)")
            return
        except Exception:
            pass
        df.to_csv(stem.with_suffix(".csv.gz"), index=False, compression="gzip")
        print(f"  wrote {stem}.csv.gz  ({len(df):,} rows)")
    except ImportError:
        import csv
        import gzip

        keys = sorted({k for r in rows for k in r.keys()})
        with gzip.open(stem.with_suffix(".csv.gz"), "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {stem}.csv.gz  ({len(rows):,} rows)  [no pandas]")


if __name__ == "__main__":
    sys.exit(main())
