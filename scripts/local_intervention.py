"""
P0-4: Local intervention test — does ActivationSteering actually change downstream output?

Initial run revealed: the library steerer DOES modify the residual stream, but
output tokens on a high-confidence factual prompt can stay the same after sampling.
The right plumbing test is at the *logit* level (deterministic, no sampling noise).

Tests:
  (1) Logits at the unembedding differ between baseline and `addition`-steered
      forwards (proves the intervention reaches the final layer).
  (2) Library steerer is bit-identical to a hand-rolled manual hook
      (proves the library does what we expect).
  (3) Logits differ under `capping` too.
  (4) Bonus: with a deliberately huge coefficient, greedy-decoded tokens on an
      open-ended prompt actually change.

Run:
    python scripts/local_intervention.py
"""

from __future__ import annotations

import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistant_axis import ActivationSteering


MODEL_NAME = "Qwen/Qwen3-1.7B"
SHORT_PROMPT = "Hi"
OPEN_PROMPT = "Tell me a short story about a fox."
MAX_NEW_TOKENS = 20

ADDITION_COEFF = 30.0       # baseline residual norm ~50–100; this is a ~30% nudge
HUGE_COEFF = 200.0          # for the token-change bonus test
CAP_TAU = -50.0             # baseline projections are O(+a few); pull way down


def pick_device() -> str:
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode(tokenizer, prompt: str, device: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def greedy_generate(model, tokenizer, input_ids, max_new=MAX_NEW_TOKENS):
    """Pure greedy — no sampling — for deterministic comparison."""
    past = None
    out_ids: list[int] = []
    work = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new):
            step = work if past is None else work[:, -1:]
            o = model(step, past_key_values=past, use_cache=True)
            past = o.past_key_values
            tid = int(o.logits[0, -1, :].argmax().item())
            out_ids.append(tid)
            if tid == tokenizer.eos_token_id:
                break
            work = torch.cat([work, torch.tensor([[tid]], device=work.device)], dim=-1)
    return out_ids


def main() -> None:
    device = pick_device()
    print(f"device = {device}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    print(f"loaded {MODEL_NAME} in {time.time() - t0:.1f}s")

    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 2
    print(f"hidden_dim={hidden}, n_layers={n_layers}, target_layer={target_layer}")

    torch.manual_seed(42)
    axis = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    axis = axis / axis.norm()

    # ---- Test 1 + 2: logits delta + library==manual ----
    print("\n--- Tests 1 & 2: logits comparison on short prompt ---")
    ids_short = encode(tokenizer, SHORT_PROMPT, device)

    with torch.no_grad():
        baseline_logits = model(ids_short, use_cache=False).logits[0, -1, :].float()

    with ActivationSteering(
        model=model,
        steering_vectors=[axis],
        coefficients=[ADDITION_COEFF],
        layer_indices=[target_layer],
        intervention_type="addition",
        positions="all",
    ):
        with torch.no_grad():
            lib_logits = model(ids_short, use_cache=False).logits[0, -1, :].float()

    def manual_addition_hook(module, ins, out):
        if isinstance(out, tuple):
            return (out[0] + ADDITION_COEFF * axis,) + out[1:]
        return out + ADDITION_COEFF * axis

    h = model.model.layers[target_layer].register_forward_hook(manual_addition_hook)
    with torch.no_grad():
        manual_logits = model(ids_short, use_cache=False).logits[0, -1, :].float()
    h.remove()

    delta_addition = (baseline_logits - lib_logits).abs().max().item()
    library_vs_manual = (lib_logits - manual_logits).abs().max().item()
    print(f"  max |baseline - addition|: {delta_addition:.6f}  (must be > 0)")
    print(f"  max |library - manual|:    {library_vs_manual:.6f}  (must be 0 — library should match hand-rolled hook)")

    # ---- Test 3: capping changes logits too ----
    print("\n--- Test 3: capping changes logits ---")
    with ActivationSteering(
        model=model,
        steering_vectors=[axis],
        coefficients=[0.0],
        layer_indices=[target_layer],
        intervention_type="capping",
        cap_thresholds=[CAP_TAU],
        positions="all",
    ):
        with torch.no_grad():
            cap_logits = model(ids_short, use_cache=False).logits[0, -1, :].float()
    delta_cap = (baseline_logits - cap_logits).abs().max().item()
    print(f"  max |baseline - capping|: {delta_cap:.6f}  (must be > 0)")

    # ---- Test 4: huge coeff + open prompt + greedy => tokens change ----
    print("\n--- Test 4: greedy tokens change under huge coeff on open prompt ---")
    ids_open = encode(tokenizer, OPEN_PROMPT, device)
    base_tokens = greedy_generate(model, tokenizer, ids_open)
    print(f"  baseline (greedy): {tokenizer.decode(base_tokens)!r}")
    with ActivationSteering(
        model=model,
        steering_vectors=[axis],
        coefficients=[HUGE_COEFF],
        layer_indices=[target_layer],
        intervention_type="addition",
        positions="all",
    ):
        steered_tokens = greedy_generate(model, tokenizer, ids_open)
    print(f"  steered  (greedy): {tokenizer.decode(steered_tokens)!r}")
    tokens_changed = base_tokens != steered_tokens

    # ---- Pass/fail ----
    print()
    print(f"logits change under addition:  {delta_addition > 0.01}  (|delta|={delta_addition:.4f})")
    print(f"library == manual hook:        {math.isclose(library_vs_manual, 0.0, abs_tol=1e-6)}  (|delta|={library_vs_manual:.6f})")
    print(f"logits change under capping:   {delta_cap > 0.01}  (|delta|={delta_cap:.4f})")
    print(f"greedy tokens change @huge K:  {tokens_changed}")

    assert delta_addition > 0.01, "FAIL: addition produced no logit change"
    assert math.isclose(library_vs_manual, 0.0, abs_tol=1e-6), \
        "FAIL: library steerer differs from hand-rolled hook"
    assert delta_cap > 0.01, "FAIL: capping produced no logit change"
    assert tokens_changed, "FAIL: even at coeff=200 on open prompt, tokens unchanged"
    print("\nP0-4 PASSED.")


if __name__ == "__main__":
    main()
