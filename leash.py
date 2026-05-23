"""
leash.py — Modal backend for Leash.

Live assistant-axis monitoring + optional activation capping for Qwen3-32B.
Streams {token, token_id, projection, capped} per generation step.

Endpoints:
  POST /chat     — SSE stream. Request body:
                       {
                         "messages": [{"role": "system"|"user", "content": "..."}],
                         "max_new_tokens": 256,
                         "temperature": 0.7,
                         "seed": null | int,
                         "enable_thinking": false,
                         "clamp": false
                       }
                   Legacy: {"prompt": "...", "system": "...", "clamp": false}

  GET /health    — readiness + axis/capping metadata.

Run locally:  modal serve leash.py
Deploy:       modal deploy leash.py
"""

from __future__ import annotations

import modal


# --- Config ---------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-32B"
AXIS_REPO = "lu-christina/assistant-axis-vectors"
AXIS_FILE = "qwen-3-32b/assistant_axis.pt"
GPU = "H100"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

# --- Image ----------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.55.0",
        "accelerate==1.0.1",
        "huggingface_hub==0.34.4",
        "safetensors==0.4.5",
        "numpy<2",
        "scikit-learn",
        "plotly",
        "fastapi[standard]",
    )
    .run_commands(
        "pip install --no-deps git+https://github.com/safety-research/assistant-axis.git"
    )
)

app = modal.App("leash", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.cls(
    gpu=GPU,
    volumes={"/cache": hf_cache},
    timeout=3600,
    scaledown_window=600,
    min_containers=4,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Leash:

    @modal.enter()
    def load(self):
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from assistant_axis import get_config, load_axis, load_capping_config

        os.environ["HF_HOME"] = "/cache"
        os.environ["HF_HUB_CACHE"] = "/cache/hub"
        self.torch = torch

        config = get_config(MODEL_NAME)
        self.target_layer = config["target_layer"]
        self.capping_experiment = config["capping_experiment"]
        print(
            f"[leash] target layer {self.target_layer} "
            f"of {config['total_layers']}, "
            f"capping={self.capping_experiment}"
        )

        print(f"[leash] loading tokenizer {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir="/cache"
        )

        print(f"[leash] loading model {MODEL_NAME}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir="/cache",
        ).eval()
        print("[leash] model loaded")

        print(f"[leash] downloading axis: {AXIS_REPO}/{AXIS_FILE}")
        axis_path = hf_hub_download(
            repo_id=AXIS_REPO,
            filename=AXIS_FILE,
            repo_type="dataset",
            cache_dir="/cache",
        )
        full_axis = load_axis(axis_path)
        axis_at_layer = full_axis[self.target_layer].to("cuda", dtype=torch.bfloat16)
        self.axis_unit = axis_at_layer / axis_at_layer.norm()
        self.axis_raw_norm = float(axis_at_layer.norm().item())
        print(
            f"[leash] axis shape={tuple(full_axis.shape)} "
            f"layer[{self.target_layer}] norm={self.axis_raw_norm:.4f}"
        )

        print(f"[leash] downloading capping config: {config['capping_config']}")
        capping_path = hf_hub_download(
            repo_id=AXIS_REPO,
            filename=config["capping_config"],
            repo_type="dataset",
            cache_dir="/cache",
        )
        self.capping_config = load_capping_config(capping_path)
        n_exps = len(self.capping_config["experiments"])
        print(f"[leash] capping config loaded ({n_exps} experiments)")

        eos = set()
        if self.tokenizer.eos_token_id is not None:
            eos.add(int(self.tokenizer.eos_token_id))
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            eos.add(im_end)
        self.eos_ids = eos

        self._last_projection = 0.0
        target_block = self.model.model.layers[self.target_layer]
        target_block.register_forward_hook(self._monitor_hook)
        print(f"[leash] monitoring hook on layer {self.target_layer}")
        print("[leash] ready")

    def _monitor_hook(self, module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        last = hidden[:, -1, :]
        self._last_projection = float((last @ self.axis_unit).item())

    def _token_loop(self, input_ids, max_new_tokens, temperature, capped):
        torch = self.torch
        past_kv = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                step = input_ids if past_kv is None else input_ids[:, -1:]
                out = self.model(step, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values

                logits = out.logits[:, -1, :].float()
                if temperature > 0.0:
                    logits = logits / max(temperature, 1e-5)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                tid = int(next_token.item())
                yield {
                    "token": self.tokenizer.decode([tid]),
                    "token_id": tid,
                    "projection": self._last_projection,
                    "capped": capped,
                }
                if tid in self.eos_ids:
                    break
                input_ids = torch.cat([input_ids, next_token], dim=-1)

    def _generate(
        self,
        messages,
        max_new_tokens,
        temperature,
        seed,
        enable_thinking,
        clamp,
    ):
        from assistant_axis import build_capping_steerer

        torch = self.torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt"
        ).input_ids.to("cuda")

        if seed is not None:
            torch.manual_seed(int(seed))

        if clamp:
            steerer = build_capping_steerer(
                self.model,
                self.capping_config,
                self.capping_experiment,
            )
            with steerer:
                yield from self._token_loop(
                    input_ids, max_new_tokens, temperature, capped=True
                )
        else:
            yield from self._token_loop(
                input_ids, max_new_tokens, temperature, capped=False
            )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def chat(self, payload: dict):
        from fastapi.responses import StreamingResponse
        import json

        messages = payload.get("messages")
        if not messages:
            prompt = payload.get("prompt", "")
            system = payload.get("system")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        max_new = int(payload.get("max_new_tokens", 256))
        temperature = float(payload.get("temperature", 0.7))
        seed = payload.get("seed")
        enable_thinking = bool(payload.get("enable_thinking", False))
        clamp = bool(payload.get("clamp", False))

        def sse():
            try:
                for chunk in self._generate(
                    messages=messages,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    clamp=clamp,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:  # noqa: BLE001
                err = {"error": type(e).__name__, "message": str(e)}
                yield f"data: {json.dumps(err)}\n\n"

        headers = {**CORS_HEADERS, "X-Accel-Buffering": "no"}
        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        from fastapi.responses import JSONResponse

        body = {
            "status": "ok",
            "model": MODEL_NAME,
            "target_layer": self.target_layer,
            "axis_raw_norm": self.axis_raw_norm,
            "capping_experiment": self.capping_experiment,
            "n_layers": int(self.model.config.num_hidden_layers),
            "hidden_size": int(self.model.config.hidden_size),
            "eos_ids": sorted(self.eos_ids),
        }
        return JSONResponse(body, headers=CORS_HEADERS)
