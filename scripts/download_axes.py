"""Cache the full set of role/trait axis vectors from the Hugging Face dataset
`lu-christina/assistant-axis-vectors` (Qwen 3 32B subset).

Each axis is a (n_layers, hidden_dim) bf16 tensor saved as `.pt`. After this
script runs, every axis is on local disk and `project_axes.py` can load them
without hitting the network.

Output: <out_dir>/qwen-3-32b/{assistant_axis,default_vector,role_vectors/*,trait_vectors/*}.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUT = Path.home() / ".cache" / "leash-axes"
REPO = "lu-christina/assistant-axis-vectors"
SUBDIR = "qwen-3-32b"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument(
        "--include-traits",
        action="store_true",
        default=True,
        help="Download trait_vectors/*.pt (default True)",
    )
    p.add_argument(
        "--include-roles",
        action="store_true",
        default=True,
        help="Download role_vectors/*.pt (default True)",
    )
    p.add_argument(
        "--max-workers", type=int, default=8, help="Concurrent download workers"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files already present (default True)",
    )
    args = p.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"target: {out_root}/{SUBDIR}")

    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset")
    files = [f for f in all_files if f.startswith(f"{SUBDIR}/")]

    selected: list[str] = []
    for f in files:
        rest = f[len(f"{SUBDIR}/"):]
        if "/" not in rest:
            selected.append(f)
            continue
        kind = rest.split("/", 1)[0]
        if kind == "role_vectors" and args.include_roles:
            selected.append(f)
        elif kind == "trait_vectors" and args.include_traits:
            selected.append(f)

    n_total = len(selected)
    print(f"to download: {n_total} axis files (concurrency={args.max_workers})")

    def fetch(name: str) -> tuple[str, bool, str]:
        local_dir = out_root
        target = local_dir / name
        if args.skip_existing and target.exists() and target.stat().st_size > 0:
            return name, True, "cached"
        try:
            hf_hub_download(
                repo_id=REPO,
                filename=name,
                repo_type="dataset",
                local_dir=str(local_dir),
            )
            return name, True, "downloaded"
        except Exception as e:  # noqa: BLE001
            return name, False, f"{type(e).__name__}: {e}"

    n_done = 0
    n_fail = 0
    n_cached = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(fetch, f): f for f in selected}
        for fut in as_completed(futures):
            name, ok, status = fut.result()
            if not ok:
                n_fail += 1
                print(f"FAIL {name}: {status}")
                continue
            if status == "cached":
                n_cached += 1
            else:
                n_done += 1
            if (n_done + n_cached) % 50 == 0:
                print(f"  {n_done} fetched, {n_cached} cached, {n_fail} failed")

    print(
        f"done: {n_done} downloaded, {n_cached} cached, {n_fail} failed "
        f"(out of {n_total})"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
