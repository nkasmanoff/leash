"""Diagnostic: does ActivationSteering work when it's the ONLY hook on the layer?
Suspect: monitor hook fires first, returns the original tensor, somehow
prevents the library steerer's return value from being honored."""
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
    [{"role": "user", "content": "What is the capital of France?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(DEVICE)

def generate(max_new=10, seed=0):
    torch.manual_seed(seed)
    past = None
    out_tokens = []
    work = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new):
            step = work if past is None else work[:, -1:]
            o = model(step, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[:, -1, :].float() / 0.7
            probs = torch.softmax(logits, dim=-1)
            t = torch.multinomial(probs, 1)
            out_tokens.append(int(t.item()))
            if int(t.item()) == tok.eos_token_id:
                break
            work = torch.cat([work, t], dim=-1)
    return out_tokens

# A) baseline, no hooks
ids_baseline = generate()
print(f"baseline:                  {tok.decode(ids_baseline)!r}")

# B) ONLY steering hook
with ActivationSteering(
    model=model,
    steering_vectors=[axis],
    coefficients=[30.0],
    layer_indices=[TARGET],
    intervention_type="addition",
    positions="all",
):
    ids_steer_only = generate()
print(f"steering only:             {tok.decode(ids_steer_only)!r}")
print(f"  changed vs baseline? {ids_steer_only != ids_baseline}")

# C) Monitor hook FIRST, then steering
def monitor(module, ins, out):
    return out  # explicit no-op return

mh = model.model.layers[TARGET].register_forward_hook(monitor)
with ActivationSteering(
    model=model,
    steering_vectors=[axis],
    coefficients=[30.0],
    layer_indices=[TARGET],
    intervention_type="addition",
    positions="all",
):
    ids_monitor_then_steer = generate()
mh.remove()
print(f"monitor (return out) + steering: {tok.decode(ids_monitor_then_steer)!r}")
print(f"  changed vs baseline? {ids_monitor_then_steer != ids_baseline}")

# D) Monitor hook that returns None
def monitor_none(module, ins, out):
    return None

mh2 = model.model.layers[TARGET].register_forward_hook(monitor_none)
with ActivationSteering(
    model=model,
    steering_vectors=[axis],
    coefficients=[30.0],
    layer_indices=[TARGET],
    intervention_type="addition",
    positions="all",
):
    ids_none_monitor = generate()
mh2.remove()
print(f"monitor (return None) + steering: {tok.decode(ids_none_monitor)!r}")
print(f"  changed vs baseline? {ids_none_monitor != ids_baseline}")
