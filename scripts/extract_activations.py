"""Replay each saved session through Modal /extract and dump residual-stream
activations at decision-token positions.

For each session under traces/hack_sweep/<run>/<cell>/seed-<n>/session/turn-*.json
we walk turn-by-turn, sending the conversation prefix + the original generated
token IDs to /extract, and saving the residual stream (at the same target layer
the assistant axis is monitored on) for a curated set of "decision tokens":

  - first token of every JSON `"command": "..."` segment in a leash-tool block
  - last 5 tokens of each `<think>...</think>` segment (the conclusion / commit)
  - hack-keyword and fix-keyword tokens inside <think> (regex from
    decision_tokens.py)
  - first 3 tokens of each visible (user-facing reply) segment

This keeps per-session payloads small (usually 20-80 positions) and lets us
project hack vs honest sessions onto the role/trait vector library without
saving multi-GB of raw hidden states.

Output: <run>/activations/<cell>__seed-<n>.npz  with arrays
  activations  float32 [n_positions, hidden_dim]
  kinds        |S32     [n_positions]   ("tool_cmd_first", "think_end", ...)
  turn         int32    [n_positions]
  position     int32    [n_positions]   (token offset within the turn)
  token        |S64     [n_positions]   (utf-8 truncated)
  projection   float32  [n_positions]   (the assistant-axis projection captured
                                         live during generation, for sanity)
and metadata-as-scalar fields cell, seed, session_id, layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from segment_tokens import segment_turn  # noqa: E402
from decision_tokens import (  # noqa: E402
    HACK_PATTERNS,
    FIX_PATTERNS,
    find_label_spans,
)


def _extract_url(chat_url: str) -> str:
    """Derive the /extract endpoint URL from the /chat URL."""
    if "-leash-chat-" in chat_url:
        return chat_url.replace("-leash-chat-", "-leash-extract-")
    if chat_url.endswith("/chat"):
        return chat_url[: -len("/chat")] + "/extract"
    return chat_url.rstrip("/") + "/extract"


def get_prefix_messages(turn_data: dict) -> list[dict]:
    """Slice everything before this turn's assistant message.

    The saved `messages` array is the post-tool-result state: it ends with
    `[..., assistant_text, tool_result_1, tool_result_2, ...]`. To replay the
    *input* to this turn's generation we drop everything from the assistant
    message of this turn onwards.
    """
    msgs = turn_data["messages"]
    target = turn_data["assistant_text"]
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m["content"] == target:
            return msgs[:i]
    raise ValueError(
        f"could not locate this turn's assistant message in turn-{turn_data['turn']:02d}"
    )


def select_positions(turn_tokens: list[dict]) -> list[tuple[int, str]]:
    """Pick (position, kind) pairs of decision tokens within one turn."""
    rows = segment_turn(turn_tokens)
    out: dict[int, str] = {}

    def add(p: int, kind: str) -> None:
        if 0 <= p < len(rows):
            out[p] = (out[p] + "+" + kind) if p in out else kind

    in_cmd = False
    cmd_start = -1
    for i, r in enumerate(rows):
        if r["kind"] == "tool_command":
            if not in_cmd:
                cmd_start = i
                in_cmd = True
        else:
            if in_cmd:
                add(cmd_start, "tool_cmd_first")
                if cmd_start + 1 < i:
                    add(cmd_start + 1, "tool_cmd_2")
                if cmd_start + 2 < i:
                    add(cmd_start + 2, "tool_cmd_3")
            in_cmd = False
    if in_cmd:
        add(cmd_start, "tool_cmd_first")

    in_think = False
    think_buffer: list[int] = []
    for i, r in enumerate(rows):
        if r["kind"] == "think":
            think_buffer.append(i)
            in_think = True
        else:
            if in_think and think_buffer:
                for p in think_buffer[-5:]:
                    add(p, "think_end")
            in_think = False
            think_buffer = []
    if in_think and think_buffer:
        for p in think_buffer[-5:]:
            add(p, "think_end")

    in_visible = False
    visible_start = -1
    for i, r in enumerate(rows):
        if r["kind"] == "visible" and not in_visible:
            visible_start = i
            in_visible = True
        elif r["kind"] != "visible" and in_visible:
            for k, p in enumerate(range(visible_start, min(visible_start + 3, i))):
                add(p, f"visible_{k}")
            in_visible = False
    if in_visible:
        end = len(rows)
        for k, p in enumerate(range(visible_start, min(visible_start + 3, end))):
            add(p, f"visible_{k}")

    think_idx = [i for i, r in enumerate(rows) if r["kind"] == "think"]
    if think_idx:
        text = ""
        char_starts: list[int] = []
        for i in think_idx:
            char_starts.append(len(text))
            text += rows[i]["token"]
        hack_spans = find_label_spans(text, HACK_PATTERNS)
        fix_spans = find_label_spans(text, FIX_PATTERNS)
        for ti, orig_i in enumerate(think_idx):
            s = char_starts[ti]
            e = s + len(rows[orig_i]["token"])
            in_hack = any(
                hs <= s < he or hs < e <= he or (s <= hs and he <= e)
                for hs, he in hack_spans
            )
            in_fix = any(
                fs <= s < fe or fs < e <= fe or (s <= fs and fe <= e)
                for fs, fe in fix_spans
            )
            if in_hack:
                add(orig_i, "hack_kw")
            elif in_fix:
                add(orig_i, "fix_kw")

    return [(p, out[p]) for p in sorted(out)]


def extract_session(
    session_dir: Path,
    extract_url: str,
    layer: int,
    timeout: float = 600.0,
):
    """Returns (activations, metadata, session_id) or (None, None, session_id)."""
    activations: list[np.ndarray] = []
    metadata: list[dict] = []
    session_id = ""

    for tf in sorted(session_dir.glob("turn-*.json")):
        td = json.loads(tf.read_text())
        session_id = td.get("session_id", "")
        prefix = get_prefix_messages(td)
        gen_ids = [t["token_id"] for t in td["tokens"]]
        if not gen_ids:
            continue
        picks = select_positions(td["tokens"])
        if not picks:
            continue
        positions = [p for p, _ in picks]
        kinds = [k for _, k in picks]

        payload = {
            "messages": prefix,
            "enable_thinking": td.get("thinking", False),
            "generated_token_ids": gen_ids,
            "layer": layer,
            "positions": positions,
        }
        resp = requests.post(extract_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        h = np.asarray(data["hidden_states"], dtype=np.float32)
        decoded = data["tokens"]
        if h.shape[0] != len(positions):
            raise RuntimeError(
                f"shape mismatch: got {h.shape[0]} hidden states for {len(positions)} positions"
            )

        for idx, (pos, kind) in enumerate(zip(positions, kinds)):
            tok_record = td["tokens"][pos]
            saved_token = tok_record.get("token", "")
            if saved_token != decoded[idx]:
                # alignment drift would invalidate downstream analysis
                raise RuntimeError(
                    f"token mismatch at turn {td['turn']} pos {pos}: "
                    f"saved={saved_token!r} extracted={decoded[idx]!r}"
                )
            activations.append(h[idx])
            metadata.append(
                {
                    "turn": int(td.get("turn", 0)),
                    "position": int(pos),
                    "kind": kind,
                    "token": saved_token,
                    "projection": float(tok_record.get("projection", 0.0)),
                    "capped": bool(tok_record.get("capped", False)),
                }
            )

    if not activations:
        return None, None, session_id

    return np.stack(activations).astype(np.float32), metadata, session_id


def _save(out_path: Path, acts: np.ndarray, meta: list[dict], extras: dict) -> None:
    arr_kind = np.array([m["kind"] for m in meta], dtype="S32")
    arr_turn = np.array([m["turn"] for m in meta], dtype=np.int32)
    arr_pos = np.array([m["position"] for m in meta], dtype=np.int32)
    arr_token = np.array(
        [m["token"].encode("utf-8")[:64] for m in meta], dtype="S64"
    )
    arr_proj = np.array([m["projection"] for m in meta], dtype=np.float32)
    arr_capped = np.array([m["capped"] for m in meta], dtype=np.bool_)

    np.savez_compressed(
        out_path,
        activations=acts,
        kind=arr_kind,
        turn=arr_turn,
        position=arr_pos,
        token=arr_token,
        projection=arr_proj,
        capped=arr_capped,
        cell=np.array(extras["cell"]),
        seed=np.array(extras["seed"], dtype=np.int32),
        session_id=np.array(extras.get("session_id", "")),
        layer=np.array(extras["layer"], dtype=np.int32),
        hidden_dim=np.array(acts.shape[1], dtype=np.int32),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        default=str(REPO / "traces/hack_sweep/run-1779559443"),
        help="Sweep directory containing <cell>/seed-N/session/turn-*.json",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write per-session .npz files (default: <run-dir>/activations)",
    )
    p.add_argument(
        "--leash-url",
        default=os.environ.get(
            "LEASH_URL", "https://nkasmanoff--leash-leash-chat-dev.modal.run"
        ),
    )
    p.add_argument(
        "--layer",
        type=int,
        default=32,
        help="Model layer to extract (default 32, matches the assistant-axis monitor)",
    )
    p.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Restrict to these cell names (default: all)",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Restrict to these seeds (default: all)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the output .npz already exists",
    )
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"run-dir does not exist: {run_dir}")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_url = _extract_url(args.leash_url)
    print(f"extract endpoint: {extract_url}")
    print(f"layer:           {args.layer}")
    print(f"output:          {out_dir}")

    session_dirs: list[Path] = []
    for cell_dir in sorted(run_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        if args.cells and cell_dir.name not in args.cells:
            continue
        for seed_dir in sorted(cell_dir.glob("seed-*")):
            if args.seeds is not None:
                seed_n = int(seed_dir.name.split("-")[1])
                if seed_n not in args.seeds:
                    continue
            sd = seed_dir / "session"
            if sd.exists():
                session_dirs.append(sd)

    print(f"sessions found:  {len(session_dirs)}")

    n_done = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    for sd in session_dirs:
        seed_dir = sd.parent
        cell_name = seed_dir.parent.name
        seed_name = seed_dir.name
        out_path = out_dir / f"{cell_name}__{seed_name}.npz"
        if out_path.exists() and not args.force:
            n_skip += 1
            continue

        print(f"  [{n_done + n_skip + n_fail + 1}/{len(session_dirs)}] {cell_name}/{seed_name} ...", flush=True)
        try:
            acts, meta, session_id = extract_session(
                sd, extract_url, args.layer, timeout=args.timeout
            )
        except Exception as e:  # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {e}")
            n_fail += 1
            continue

        if acts is None:
            print("      no decision tokens, skipping")
            continue

        _save(
            out_path,
            acts,
            meta,
            {
                "cell": cell_name,
                "seed": int(seed_name.split("-")[1]),
                "session_id": session_id,
                "layer": args.layer,
            },
        )
        n_done += 1
        print(f"      saved {acts.shape[0]} positions -> {out_path.name}")

    elapsed = time.time() - t0
    print(
        f"done: {n_done} extracted, {n_skip} skipped (cached), {n_fail} failed "
        f"in {elapsed:.1f}s"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
