"""Diagnostic: what does a Qwen3 decoder layer's forward return in transformers 5.x,
and does a forward hook's return value actually replace it?

Three checks:
  1. Print the type and structure of layer 14's output.
  2. Register a hook that ZEROS the output; check that the next layer sees zeros.
  3. Register a hook that returns the correct (modified,) tuple; check propagation.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DEVICE = "mps"
else:
    DEVICE = "cpu"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(DEVICE).eval()

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Hi"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

TARGET = 14
NEXT = 15

captured = {"out_14": None, "in_15": None}

def probe_out_14(module, inputs, outputs):
    captured["out_14"] = outputs
    return outputs

def probe_in_15(module, inputs, outputs):
    # The INPUT to layer 15 is the OUTPUT of layer 14, post any hook on layer 14.
    captured["in_15"] = inputs
    return outputs

h1 = model.model.layers[TARGET].register_forward_hook(probe_out_14)
h2 = model.model.layers[NEXT].register_forward_hook(probe_in_15)

with torch.no_grad():
    _ = model(input_ids, use_cache=False)

h1.remove(); h2.remove()

print("=" * 60)
print("CHECK 1: structure of layer 14 output")
print("=" * 60)
out = captured["out_14"]
print(f"type: {type(out).__name__}")
if isinstance(out, tuple):
    print(f"  len: {len(out)}")
    for i, x in enumerate(out):
        print(f"  [{i}]: type={type(x).__name__}", end="")
        if torch.is_tensor(x):
            print(f"  shape={tuple(x.shape)} dtype={x.dtype}")
        else:
            print(f"  value={x!r}")
elif torch.is_tensor(out):
    print(f"  shape: {tuple(out.shape)}, dtype: {out.dtype}")
else:
    print(f"  attrs: {[a for a in dir(out) if not a.startswith('_')]}")

print()
print("=" * 60)
print("CHECK 2: input to layer 15 (should equal layer-14 output)")
print("=" * 60)
in_15 = captured["in_15"]
print(f"type: {type(in_15).__name__}")
if isinstance(in_15, tuple):
    print(f"  len: {len(in_15)}")
    for i, x in enumerate(in_15):
        print(f"  [{i}]: type={type(x).__name__}", end="")
        if torch.is_tensor(x):
            print(f"  shape={tuple(x.shape)} dtype={x.dtype}")
        else:
            print(f"  value={x!r}")

# Confirm equality
out_14_tensor = out[0] if isinstance(out, tuple) else out
in_15_tensor = in_15[0]
print(f"\n  out_14 == in_15? {torch.equal(out_14_tensor, in_15_tensor)}")

print()
print("=" * 60)
print("CHECK 3: hook that zeros output — does layer 15 see zeros?")
print("=" * 60)

captured2 = {"in_15_after_zero": None}

def zero_out_14(module, inputs, outputs):
    if isinstance(outputs, tuple):
        zeroed = (torch.zeros_like(outputs[0]),) + outputs[1:]
        return zeroed
    return torch.zeros_like(outputs)

def probe_in_15_v2(module, inputs, outputs):
    captured2["in_15_after_zero"] = inputs
    return outputs

h3 = model.model.layers[TARGET].register_forward_hook(zero_out_14)
h4 = model.model.layers[NEXT].register_forward_hook(probe_in_15_v2)

with torch.no_grad():
    _ = model(input_ids, use_cache=False)

h3.remove(); h4.remove()

in_15z = captured2["in_15_after_zero"]
in_15z_tensor = in_15z[0]
all_zero = bool((in_15z_tensor == 0).all().item())
max_abs = float(in_15z_tensor.abs().max().item())
print(f"layer 15 input all-zero? {all_zero}, max|val|={max_abs}")
if all_zero:
    print("  ==> forward hook return value IS honored. Bug is elsewhere.")
else:
    print("  ==> forward hook return value is BEING IGNORED. transformers 5.x issue.")
