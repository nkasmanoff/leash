"""Stream completions from the Leash /chat SSE endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests


@dataclass
class TokenChunk:
    token: str
    projection: float
    capped: bool = False
    token_id: int | None = None


@dataclass
class ChatResult:
    text: str
    tokens: list[TokenChunk] = field(default_factory=list)
    error: str | None = None


def stream_chat(
    url: str,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    seed: int | None = None,
    enable_thinking: bool = False,
    clamp: bool = False,
    timeout: int = 600,
) -> ChatResult:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "clamp": clamp,
    }
    if seed is not None:
        payload["seed"] = seed

    chunks: list[TokenChunk] = []
    parts: list[str] = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8")
                if not text.startswith("data: "):
                    continue
                data = text[6:]
                if data == "[DONE]":
                    break
                row = json.loads(data)
                if "error" in row:
                    return ChatResult(
                        text="".join(parts),
                        tokens=chunks,
                        error=f"{row['error']}: {row.get('message', '')}",
                    )
                tok = row.get("token", "")
                parts.append(tok)
                chunks.append(
                    TokenChunk(
                        token=tok,
                        projection=float(row.get("projection", 0)),
                        capped=bool(row.get("capped", False)),
                        token_id=row.get("token_id"),
                    )
                )
    except requests.RequestException as exc:
        return ChatResult(text="".join(parts), tokens=chunks, error=str(exc))

    return ChatResult(text="".join(parts), tokens=chunks)


def iter_chat_tokens(
    url: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> Iterator[TokenChunk]:
    """Yield tokens as they arrive (for live CLI display)."""
    payload: dict[str, Any] = {
        "messages": messages,
        "max_new_tokens": kwargs.get("max_new_tokens", 2048),
        "temperature": kwargs.get("temperature", 0.7),
        "enable_thinking": kwargs.get("enable_thinking", False),
        "clamp": kwargs.get("clamp", False),
    }
    if kwargs.get("seed") is not None:
        payload["seed"] = kwargs["seed"]

    timeout = kwargs.get("timeout", 600)
    with requests.post(
        url, json=payload, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if not text.startswith("data: "):
                continue
            data = text[6:]
            if data == "[DONE]":
                return
            row = json.loads(data)
            if "error" in row:
                raise RuntimeError(f"{row['error']}: {row.get('message', '')}")
            yield TokenChunk(
                token=row.get("token", ""),
                projection=float(row.get("projection", 0)),
                capped=bool(row.get("capped", False)),
                token_id=row.get("token_id"),
            )
