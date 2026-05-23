"""
P0-2: Verify HF artifact download for the Qwen3-32B assistant axis + capping config.

Downloads the three relevant files from lu-christina/assistant-axis-vectors,
loads each with the assistant_axis library helpers (where applicable), and
prints shapes / dtypes / structural metadata so we can confirm our mental
model before touching Modal.

Run:
    python scripts/inspect_artifacts.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from assistant_axis import (
    get_config,
    load_axis,
    load_capping_config,
)


REPO_ID = "lu-christina/assistant-axis-vectors"
MODEL_NAME = "Qwen/Qwen3-32B"

FILES = [
    "qwen-3-32b/assistant_axis.pt",
    "qwen-3-32b/capping_config.pt",
    "qwen-3-32b/default_vector.pt",
]


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    hr("Config from assistant_axis.MODEL_CONFIGS")
    cfg = get_config(MODEL_NAME)
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    hr("Download")
    paths: dict[str, str] = {}
    for fn in FILES:
        path = hf_hub_download(repo_id=REPO_ID, filename=fn, repo_type="dataset")
        paths[fn] = path
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {fn}  ->  {path}  ({size_mb:.2f} MB)")

    # ---- assistant_axis.pt ----
    hr("assistant_axis.pt (monitoring axis)")
    axis = load_axis(paths["qwen-3-32b/assistant_axis.pt"])
    print(f"  type:  {type(axis).__name__}")
    print(f"  shape: {tuple(axis.shape)}")
    print(f"  dtype: {axis.dtype}")
    print(f"  device:{axis.device}")
    n_layers, hidden = axis.shape
    print(f"  -> n_layers={n_layers}, hidden_dim={hidden}")

    target = cfg["target_layer"]
    a = axis[target].float()
    print(f"  axis[{target}] norm: {a.norm().item():.4f}")
    print(f"  axis[{target}] first 5: {a[:5].tolist()}")

    # ---- default_vector.pt ----
    hr("default_vector.pt (mean default-Assistant activation)")
    raw = torch.load(
        paths["qwen-3-32b/default_vector.pt"],
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(raw, dict):
        print(f"  dict keys: {list(raw.keys())}")
        for k, v in raw.items():
            if torch.is_tensor(v):
                print(f"    {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v).__name__} = {v!r}")
    else:
        print(f"  tensor shape={tuple(raw.shape)} dtype={raw.dtype}")

    # ---- capping_config.pt ----
    hr("capping_config.pt (intervention recipes)")
    cc = load_capping_config(paths["qwen-3-32b/capping_config.pt"])
    print(f"  top-level keys: {list(cc.keys())}")
    print(f"  num vectors:     {len(cc['vectors'])}")
    print(f"  num experiments: {len(cc['experiments'])}")

    # Peek at one vector entry
    first_vec_name = next(iter(cc["vectors"]))
    vec_entry = cc["vectors"][first_vec_name]
    print(f"\n  Sample vector '{first_vec_name}':")
    if isinstance(vec_entry, dict):
        for k, v in vec_entry.items():
            if torch.is_tensor(v):
                print(f"    {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"    {k}: {v!r}")
    else:
        print(f"    (non-dict entry: {type(vec_entry).__name__})")

    # Find the recommended experiment
    recommended = cfg["capping_experiment"]
    exp = next((e for e in cc["experiments"] if e["id"] == recommended), None)
    print(f"\n  Recommended experiment id: {recommended}")
    if exp is None:
        print("  !! NOT FOUND in capping_config['experiments']")
    else:
        print(f"  experiment keys: {list(exp.keys())}")
        ivs = exp["interventions"]
        cap_ivs = [iv for iv in ivs if "cap" in iv]
        print(f"  total interventions: {len(ivs)}, capping: {len(cap_ivs)}")
        print(f"  first capping intervention: {cap_ivs[0] if cap_ivs else 'none'}")
        # Layer coverage
        layers_used = sorted({cc["vectors"][iv["vector"]]["layer"] for iv in cap_ivs})
        print(f"  layers touched by capping: {layers_used}")
        # Cap value distribution
        caps = [float(iv["cap"]) for iv in cap_ivs]
        if caps:
            print(
                f"  cap values: min={min(caps):.4f} "
                f"max={max(caps):.4f} mean={sum(caps)/len(caps):.4f}"
            )

    # ---- A few experiment ids to sanity-check the inventory ----
    hr("First 10 experiment IDs in capping_config")
    for e in cc["experiments"][:10]:
        n_cap = len([iv for iv in e["interventions"] if "cap" in iv])
        print(f"  {e['id']:35s}  caps={n_cap}")

    hr("Done")
    print("All three artifacts loaded successfully.")


if __name__ == "__main__":
    main()
