"""
P0-3: Local plumbing test for the projection hook.

Loads Qwen3-1.7B on MPS (or CPU), registers a forward hook on the middle
transformer block, projects the last-token residual onto a *random* unit
vector each step, and streams ~20 tokens with KV cache.

We don't care about the projection numbers being scientifically meaningful
here. We do care that:
  - the hook fires every generation step,
  - projections are finite (no NaN/Inf),
  - projections are non-zero,
  - projections actually vary token-to-token (proves we're reading the new
    token's residual, not a stale cached value).

Run:
    python scripts/local_plumbing.py
"""

from __future__ import annotations

import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-1.7B"
PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.7
SEED = 0


def pick_device() -> str:
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = pick_device()
    print(f"device = {device}")

    print(f"loading {MODEL_NAME} (first run downloads ~3.5 GB)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 2
    print(
        f"hidden_dim={hidden}, n_layers={n_layers}, "
        f"target_layer={target_layer}"
    )

    # Random unit-vector "axis" — values are meaningless, math is the same.
    torch.manual_seed(42)
    axis = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    axis = axis / axis.norm()
    print(f"axis: shape={tuple(axis.shape)}, norm={axis.norm().item():.4f}")

    # Mutable slot that the hook writes into each step.
    state: dict[str, float | None] = {"projection": None, "hook_calls": 0}

    def projection_hook(module, inputs, outputs):
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        last = hidden_states[:, -1, :]              # (batch, hidden)
        state["projection"] = float((last @ axis).item())
        state["hook_calls"] = (state["hook_calls"] or 0) + 1
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(projection_hook)

    messages = [{"role": "user", "content": PROMPT}]
    # Two-step: render template -> tokenize. Works on transformers 4.x and 5.x
    # (5.x changed apply_chat_template(return_tensors=...) to return a
    # BatchEncoding instead of a raw tensor).
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    print(f"prompt tokens = {input_ids.shape[-1]}")

    projections: list[float] = []
    tokens_decoded: list[str] = []
    past_kv = None
    torch.manual_seed(SEED)

    print()
    print(f"{'step':>4} {'proj':>10} {'token':<20}")
    print("-" * 38)

    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            step_input = input_ids if past_kv is None else input_ids[:, -1:]
            out = model(step_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

            logits = out.logits[:, -1, :].float() / max(TEMPERATURE, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            proj = state["projection"]
            assert proj is not None, "hook did not fire"
            projections.append(proj)
            decoded = tokenizer.decode(next_token[0])
            tokens_decoded.append(decoded)
            print(f"{i:>4d} {proj:>+10.4f} {decoded!r}")

            if next_token.item() == tokenizer.eos_token_id:
                print("  (eos reached)")
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    handle.remove()

    print()
    finite = all(math.isfinite(p) for p in projections)
    nonzero = any(abs(p) > 1e-6 for p in projections)
    distinct = len({round(p, 4) for p in projections})

    print(f"hook calls:        {state['hook_calls']}")
    print(f"all finite:        {finite}")
    print(f"any non-zero:      {nonzero}")
    print(f"distinct values:   {distinct} / {len(projections)}")
    print(f"min / max proj:    {min(projections):+.4f} / {max(projections):+.4f}")
    print(f"completion:        {''.join(tokens_decoded)!r}")

    assert finite, "FAIL: some projections are NaN/Inf"
    assert nonzero, "FAIL: all projections are ~0 (hook may not be capturing)"
    assert distinct > 1, "FAIL: all projections identical (stale state?)"
    print("\nP0-3 PASSED.")


if __name__ == "__main__":
    main()
