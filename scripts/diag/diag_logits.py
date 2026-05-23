"""Directly compare logits with vs without steering. If the steering hook
is actually modifying the residual, the logits MUST differ at the unembedding."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant_axis import ActivationSteering

if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DEVICE = "mps"
else:
    DEVICE = "cpu"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(DEVICE).eval()

TARGET = 14
hidden = model.config.hidden_size
torch.manual_seed(42)
axis = torch.randn(hidden, dtype=torch.bfloat16, device=DEVICE); axis = axis / axis.norm()

prompt_text = tok.apply_chat_template(
    [{"role": "user", "content": "Hi"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(DEVICE)

with torch.no_grad():
    out_baseline = model(input_ids, use_cache=False).logits[0, -1, :].float()

with ActivationSteering(
    model=model,
    steering_vectors=[axis],
    coefficients=[30.0],
    layer_indices=[TARGET],
    intervention_type="addition",
    positions="all",
):
    with torch.no_grad():
        out_steered_lib = model(input_ids, use_cache=False).logits[0, -1, :].float()

# Compare to a MANUAL steering hook that does the same thing
def manual_steer(module, ins, out):
    if isinstance(out, tuple):
        return (out[0] + 30.0 * axis,) + out[1:]
    return out + 30.0 * axis

h = model.model.layers[TARGET].register_forward_hook(manual_steer)
with torch.no_grad():
    out_steered_manual = model(input_ids, use_cache=False).logits[0, -1, :].float()
h.remove()

print(f"baseline logits[:5]: {out_baseline[:5].tolist()}")
print(f"library steered[:5]: {out_steered_lib[:5].tolist()}")
print(f"manual  steered[:5]: {out_steered_manual[:5].tolist()}")
print()
print(f"|baseline - library_steered|_max: {(out_baseline - out_steered_lib).abs().max().item():.6f}")
print(f"|baseline - manual_steered|_max:  {(out_baseline - out_steered_manual).abs().max().item():.6f}")
print(f"|library  - manual|_max:           {(out_steered_lib - out_steered_manual).abs().max().item():.6f}")
print()
print(f"argmax baseline:        {int(out_baseline.argmax())} = {tok.decode([int(out_baseline.argmax())])!r}")
print(f"argmax library steered: {int(out_steered_lib.argmax())} = {tok.decode([int(out_steered_lib.argmax())])!r}")
print(f"argmax manual  steered: {int(out_steered_manual.argmax())} = {tok.decode([int(out_steered_manual.argmax())])!r}")
