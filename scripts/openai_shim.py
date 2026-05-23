"""
OpenAI-compatible proxy in front of the Leash /chat SSE endpoint.

OpenCode (and other coding agents) speak /v1/chat/completions. Leash speaks
custom SSE with per-token projection metadata. This shim translates between
them and logs rich harness traces for the dashboard.

Usage:
    export LEASH_URL="https://…--leash-leash-chat-dev.modal.run"
    python scripts/openai_shim.py

Dashboard (separate terminal):
    cd dashboard && npm run dev
    open http://localhost:5173 → Harness tab
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Allow importing trace_store and harness from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_store import TraceStore  # noqa: E402
from harness.agent import AgentConfig, AgentSession  # noqa: E402

LEASH_URL = os.environ.get(
    "LEASH_URL",
    "https://nkasmanoff--leash-leash-chat-dev.modal.run",
).rstrip("/")
HOST = os.environ.get("LEASH_SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("LEASH_SHIM_PORT", "8787"))
TRACE_DIR = Path(os.environ.get("LEASH_TRACE_DIR", "traces/harness"))

store = TraceStore(TRACE_DIR)
agent_sessions: dict[str, AgentSession] = {}
AGENT_CWD = os.environ.get("LEASH_AGENT_CWD", str(REPO_ROOT))
FAKE_TOOLS = os.environ.get("LEASH_FAKE_TOOLS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="Leash OpenAI shim", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _leash_payload(body: dict[str, Any], clamp: bool, thinking: bool) -> dict[str, Any]:
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 256
    temperature = float(body.get("temperature", 0.7))
    seed = body.get("seed")

    return {
        "messages": messages,
        "max_new_tokens": int(max_tokens),
        "temperature": temperature,
        "seed": seed,
        "enable_thinking": thinking,
        "clamp": clamp,
    }


def _iter_leash_chunks(url: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    with requests.post(url, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if not text.startswith("data: "):
                continue
            data = text[len("data: ") :]
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"{chunk['error']}: {chunk.get('message', '')}")
            yield chunk


def _openai_non_stream(model: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(c.get("token", "") for c in chunks)
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(chunks),
            "total_tokens": len(chunks),
        },
        "leash": {
            "projections": [c.get("projection") for c in chunks],
            "capped": any(c.get("capped") for c in chunks),
        },
    }


def _openai_stream_event(model: str, chunk: dict[str, Any] | None, *, done: bool = False) -> str:
    created = int(time.time())
    if done:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    else:
        assert chunk is not None
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.get("token", "")},
                    "finish_reason": None,
                }
            ],
            "leash": {
                "projection": chunk.get("projection"),
                "capped": chunk.get("capped"),
                "token_id": chunk.get("token_id"),
            },
        }
    return f"data: {json.dumps(payload)}\n\n"


def _save_turn(
    req_id: str,
    body: dict[str, Any],
    clamp: bool,
    thinking: bool,
    records: list[dict[str, Any]],
    t0: float,
    error: str | None = None,
) -> None:
    if not records and not error:
        return
    store.save_turn(
        req_id=req_id,
        body=body,
        clamp=clamp,
        thinking=thinking,
        records=records,
        duration_s=time.time() - t0,
        error=error,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "leash_url": LEASH_URL}


@app.get("/api/traces")
def api_list_traces() -> dict[str, Any]:
    turns = store.list_turns()
    sessions: dict[str, list[str]] = {}
    for t in turns:
        sessions.setdefault(t["session_id"], []).append(t["req_id"])
    return {
        "turns": turns,
        "sessions": [
            {
                "session_id": sid,
                "req_ids": ids,
                "turn_count": len(ids),
            }
            for sid, ids in sessions.items()
        ],
    }


@app.get("/api/traces/{req_id}")
def api_get_trace(req_id: str) -> dict[str, Any]:
    try:
        return store.load_turn(req_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str) -> dict[str, Any]:
    try:
        return store.load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _as_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def _agent_config(body: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        leash_url=LEASH_URL,
        cwd=str(body.get("cwd") or AGENT_CWD),
        clamp=_as_bool(body.get("clamp"), False),
        thinking=_as_bool(body.get("thinking"), False),
        fake_tools=_as_bool(body.get("fake_tools"), FAKE_TOOLS),
        max_turns=int(body.get("max_turns", 20)),
        max_new_tokens=int(body.get("max_new_tokens", 2048)),
        temperature=float(body.get("temperature", 0.7)),
        trace_dir=TRACE_DIR,
    )


def _get_agent_session(body: dict[str, Any]) -> AgentSession:
    cfg = _agent_config(body)
    sid = body.get("session_id")
    if sid and sid in agent_sessions:
        session = agent_sessions[sid]
        mode_changed = (
            session.config.fake_tools != cfg.fake_tools
            or session.config.thinking != cfg.thinking
        )
        session.config.clamp = cfg.clamp
        session.config.thinking = cfg.thinking
        session.config.fake_tools = cfg.fake_tools
        if mode_changed:
            session.sync_tool_mode()
        return session
    session = AgentSession(cfg)
    agent_sessions[session.session_id] = session
    return session


@app.post("/api/agent/chat", response_model=None)
def api_agent_chat(body: dict[str, Any]) -> StreamingResponse:
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    session = _get_agent_session(body)

    def sse() -> Iterator[str]:
        try:
            for event in session.iter_user_message(message):
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/api/agent/status")
def api_agent_status() -> dict[str, Any]:
    return {
        "fake_tools_default": FAKE_TOOLS,
        "cwd": AGENT_CWD,
        "active_sessions": len(agent_sessions),
    }


@app.post("/api/agent/reset")
def api_agent_reset(body: dict[str, Any] | None = None) -> dict[str, str]:
    payload = body or {}
    sid = payload.get("session_id")
    if sid and sid in agent_sessions:
        del agent_sessions[sid]
    session = AgentSession(_agent_config(payload))
    agent_sessions[session.session_id] = session
    return {"session_id": session.session_id}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-32b",
                "object": "model",
                "owned_by": "leash",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(
    body: dict[str, Any],
    x_leash_clamp: str | None = Header(default=None, alias="X-Leash-Clamp"),
    x_leash_thinking: str | None = Header(default=None, alias="X-Leash-Thinking"),
) -> JSONResponse | StreamingResponse:
    model = body.get("model", "qwen3-32b")
    stream = bool(body.get("stream", False))

    clamp = x_leash_clamp.lower() in {"1", "true", "yes", "on"} if x_leash_clamp else False
    thinking = (
        x_leash_thinking.lower() in {"1", "true", "yes", "on"}
        if x_leash_thinking
        else False
    )

    payload = _leash_payload(body, clamp=clamp, thinking=thinking)
    req_id = f"harness-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    if not stream:
        records: list[dict[str, Any]] = []
        err: str | None = None
        try:
            chunks = list(_iter_leash_chunks(LEASH_URL, payload))
            records = [
                {"i": i, **c, "ts": time.time() - t0, "req_id": req_id}
                for i, c in enumerate(chunks)
            ]
        except requests.RequestException as exc:
            err = str(exc)
            _save_turn(req_id, body, clamp, thinking, records, t0, error=err)
            raise HTTPException(status_code=502, detail=err) from exc
        except RuntimeError as exc:
            err = str(exc)
            _save_turn(req_id, body, clamp, thinking, records, t0, error=err)
            raise HTTPException(status_code=502, detail=err) from exc

        _save_turn(req_id, body, clamp, thinking, records, t0)
        return JSONResponse(_openai_non_stream(model, chunks))

    def sse() -> Iterator[str]:
        records: list[dict[str, Any]] = []
        err: str | None = None
        try:
            for i, chunk in enumerate(_iter_leash_chunks(LEASH_URL, payload)):
                records.append(
                    {"i": i, **chunk, "ts": time.time() - t0, "req_id": req_id}
                )
                yield _openai_stream_event(model, chunk)
            yield _openai_stream_event(model, None, done=True)
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            yield f"data: {json.dumps({'error': {'message': err, 'type': 'leash_shim_error'}})}\n\n"
        finally:
            _save_turn(req_id, body, clamp, thinking, records, t0, error=err)

    return StreamingResponse(sse(), media_type="text/event-stream")


def main() -> None:
    print(f"Leash shim -> {LEASH_URL}")
    print(f"OpenCode baseURL: http://{HOST}:{PORT}/v1")
    print(f"Trace API:        http://{HOST}:{PORT}/api/traces")
    print(f"Agent API:        http://{HOST}:{PORT}/api/agent/chat")
    print(f"Agent cwd:        {AGENT_CWD}")
    print(f"Fake tools:       {FAKE_TOOLS}")
    print(f"Traces dir:       {TRACE_DIR.resolve()}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
