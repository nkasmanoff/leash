"""
Harness trace storage + parsing for the OpenAI shim and dashboard.

Each LLM request is saved as traces/harness/<req_id>/:
  meta.json   — request context, parsed prompts/tools, response summary
  tokens.jsonl — per-token projection stream
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces" / "harness"
SESSION_GAP_S = 30 * 60  # group turns within 30 minutes

THINKING_START = "<think>"
THINKING_END = "</think>"


def _msg_content(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
        return "\n".join(parts)
    return str(content)


def parse_request_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    input_tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = _msg_content(msg)
        entry: dict[str, Any] = {
            "index": i,
            "role": role,
            "content": content,
        }

        if role == "system":
            system_parts.append(content)
            entry["kind"] = "system"
        elif role == "user":
            entry["kind"] = "user"
        elif role == "assistant":
            entry["kind"] = "assistant"
            tcs = msg.get("tool_calls") or []
            if tcs:
                entry["tool_calls"] = tcs
                for tc in tcs:
                    fn = tc.get("function") or {}
                    input_tool_calls.append(
                        {
                            "message_index": i,
                            "id": tc.get("id"),
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        }
                    )
        elif role == "tool":
            entry["kind"] = "tool_result"
            entry["tool_call_id"] = msg.get("tool_call_id")
            entry["name"] = msg.get("name")
            tool_results.append(
                {
                    "message_index": i,
                    "tool_call_id": msg.get("tool_call_id"),
                    "name": msg.get("name"),
                    "content": content,
                }
            )
        else:
            entry["kind"] = role

        conversation.append(entry)

    return {
        "system_prompt": "\n\n".join(p for p in system_parts if p.strip()),
        "conversation": conversation,
        "input_tool_calls": input_tool_calls,
        "tool_results": tool_results,
        "message_count": len(messages),
    }


def parse_response_text(text: str) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        think_start = text.find(THINKING_START, pos)
        if think_start == -1:
            if pos < len(text):
                segments.append(
                    {"kind": "content", "start": pos, "end": len(text), "text": text[pos:]}
                )
            break
        if think_start > pos:
            segments.append(
                {
                    "kind": "content",
                    "start": pos,
                    "end": think_start,
                    "text": text[pos:think_start],
                }
            )
        think_end = text.find(THINKING_END, think_start)
        if think_end == -1:
            segments.append(
                {
                    "kind": "thinking",
                    "start": think_start,
                    "end": len(text),
                    "text": text[think_start:],
                }
            )
            break
        end = think_end + len(THINKING_END)
        segments.append(
            {
                "kind": "thinking",
                "start": think_start,
                "end": end,
                "text": text[think_start:end],
            }
        )
        pos = end

    # Heuristic: OpenCode-style tool narration in assistant text
    tool_markers: list[dict[str, Any]] = []
    for m in re.finditer(
        r"\[opencode\]\s*Running\s+`([^`]+)`",
        text,
        flags=re.IGNORECASE,
    ):
        tool_markers.append(
            {"kind": "tool_invocation", "command": m.group(1), "at": m.start()}
        )
    for m in re.finditer(r"```(?:bash|sh|shell)?\n(.*?)```", text, re.DOTALL):
        tool_markers.append(
            {"kind": "shell_block", "command": m.group(1).strip(), "at": m.start()}
        )

    return {"segments": segments, "tool_markers": tool_markers, "text": text}


def projection_stats(projections: list[float]) -> dict[str, float | int]:
    if not projections:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "n_tokens": 0}
    return {
        "min": min(projections),
        "max": max(projections),
        "mean": sum(projections) / len(projections),
        "n_tokens": len(projections),
    }


class TraceStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or TRACE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._last_turn_at: float | None = None
        self._session_id: str | None = None

    def _session_for_turn(self, now: float) -> str:
        if (
            self._session_id is None
            or self._last_turn_at is None
            or now - self._last_turn_at > SESSION_GAP_S
        ):
            self._session_id = f"session-{int(now)}-{uuid.uuid4().hex[:6]}"
        self._last_turn_at = now
        return self._session_id

    def save_turn(
        self,
        *,
        req_id: str,
        body: dict[str, Any],
        clamp: bool,
        thinking: bool,
        records: list[dict[str, Any]],
        duration_s: float,
        error: str | None = None,
    ) -> Path:
        turn_dir = self.root / req_id
        turn_dir.mkdir(parents=True, exist_ok=True)

        messages = body.get("messages") or []
        parsed_req = parse_request_messages(messages)
        response_text = "".join(r.get("token", "") for r in records)
        parsed_resp = parse_response_text(response_text)
        projections = [float(r.get("projection", 0)) for r in records]
        now = time.time()

        meta = {
            "req_id": req_id,
            "session_id": self._session_for_turn(now),
            "created_at": now - duration_s,
            "finished_at": now,
            "duration_s": round(duration_s, 3),
            "model": body.get("model", "qwen3-32b"),
            "clamp": clamp,
            "thinking": thinking,
            "stream": bool(body.get("stream", False)),
            "error": error,
            "request": {
                "temperature": body.get("temperature"),
                "max_tokens": body.get("max_tokens")
                or body.get("max_completion_tokens"),
                "tools": body.get("tools"),
                "tool_choice": body.get("tool_choice"),
                "messages": messages,
            },
            "parsed": {
                **parsed_req,
                "response": parsed_resp,
            },
            "response_text": response_text,
            "stats": projection_stats(projections),
        }

        with (turn_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with (turn_dir / "tokens.jsonl").open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")

        self._write_index()
        return turn_dir

    def _write_index(self) -> None:
        turns = self.list_turns()
        sessions: dict[str, list[str]] = {}
        for t in turns:
            sessions.setdefault(t["session_id"], []).append(t["req_id"])

        index = {
            "updated_at": time.time(),
            "turns": turns,
            "sessions": [
                {
                    "session_id": sid,
                    "req_ids": req_ids,
                    "turn_count": len(req_ids),
                    "started_at": min(
                        t["created_at"] for t in turns if t["req_id"] in req_ids
                    ),
                    "ended_at": max(
                        t["finished_at"] for t in turns if t["req_id"] in req_ids
                    ),
                }
                for sid, req_ids in sorted(
                    sessions.items(), key=lambda x: x[1][0], reverse=True
                )
            ],
        }
        with (self.root / "index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def list_turns(self) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []

        # OpenAI shim format: harness-*/meta.json
        for meta_path in self.root.glob("harness-*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                turns.append(self._turn_summary(meta))
            except (json.JSONDecodeError, OSError):
                continue

        # Agent harness format: agent-*/turn-*.json
        for turn_path in self.root.glob("agent-*/turn-*.json"):
            try:
                turns.append(self._agent_turn_summary(turn_path))
            except (json.JSONDecodeError, OSError):
                continue

        # Legacy: flat harness-*.jsonl at root
        for legacy in self.root.glob("harness-*.jsonl"):
            req_id = legacy.stem
            if (self.root / req_id / "meta.json").exists():
                continue
            try:
                turns.append(self._legacy_summary(req_id, legacy))
            except OSError:
                continue

        turns.sort(key=lambda t: t["created_at"], reverse=True)
        return turns

    def _turn_summary(self, meta: dict[str, Any]) -> dict[str, Any]:
        parsed = meta.get("parsed") or {}
        system = parsed.get("system_prompt") or ""
        conv = parsed.get("conversation") or []
        last_user = next(
            (c.get("content", "") for c in reversed(conv) if c.get("role") == "user"),
            "",
        )
        return {
            "req_id": meta["req_id"],
            "session_id": meta.get("session_id", "unknown"),
            "created_at": meta.get("created_at", 0),
            "finished_at": meta.get("finished_at", 0),
            "duration_s": meta.get("duration_s", 0),
            "model": meta.get("model"),
            "clamp": meta.get("clamp", False),
            "thinking": meta.get("thinking", False),
            "error": meta.get("error"),
            "stats": meta.get("stats", {}),
            "preview": {
                "system_chars": len(system),
                "last_user": last_user[:120],
                "input_tool_calls": len(parsed.get("input_tool_calls") or []),
                "tool_results": len(parsed.get("tool_results") or []),
                "response_chars": len(meta.get("response_text") or ""),
            },
        }

    def _legacy_summary(self, req_id: str, path: Path) -> dict[str, Any]:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        projections = [float(r.get("projection", 0)) for r in records]
        text = "".join(r.get("token", "") for r in records)
        mtime = path.stat().st_mtime
        return {
            "req_id": req_id,
            "session_id": "legacy",
            "created_at": mtime - (records[-1].get("ts", 0) if records else 0),
            "finished_at": mtime,
            "duration_s": records[-1].get("ts", 0) if records else 0,
            "model": "qwen3-32b",
            "clamp": any(r.get("capped") for r in records),
            "thinking": THINKING_START in text,
            "error": None,
            "stats": projection_stats(projections),
            "preview": {
                "system_chars": 0,
                "last_user": "(legacy trace — request not recorded)",
                "input_tool_calls": 0,
                "tool_results": 0,
                "response_chars": len(text),
            },
            "legacy_path": str(path),
        }

    def _agent_turn_summary(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        session_id = data.get("session_id") or path.parent.name
        turn_num = int(data.get("turn") or 0)
        req_id = f"{session_id}/turn-{turn_num:02d}"
        messages = data.get("messages") or []
        last_user = next(
            (
                _msg_content(m)
                for m in reversed(messages)
                if m.get("role") == "user" and not str(_msg_content(m)).startswith("<tool_result")
            ),
            "",
        )
        mtime = path.stat().st_mtime
        duration_s = float(data.get("duration_s") or 0)
        stats = data.get("stats") or projection_stats(
            [float(t.get("projection", 0)) for t in data.get("tokens") or []]
        )
        return {
            "req_id": req_id,
            "session_id": session_id,
            "created_at": mtime - duration_s,
            "finished_at": mtime,
            "duration_s": duration_s,
            "model": "qwen3-32b",
            "clamp": bool(data.get("clamp", False)),
            "thinking": bool(data.get("thinking", False)),
            "fake_tools": bool(data.get("fake_tools", False)),
            "error": None,
            "stats": stats,
            "preview": {
                "system_chars": len(
                    next((_msg_content(m) for m in messages if m.get("role") == "system"), "")
                ),
                "last_user": last_user[:120],
                "input_tool_calls": len(data.get("tool_calls") or []),
                "tool_results": len(data.get("tool_results") or []),
                "response_chars": len(data.get("assistant_text") or ""),
            },
            "agent_path": str(path.relative_to(self.root)),
        }

    def _load_agent_turn(self, req_id: str) -> dict[str, Any]:
        if "/" not in req_id:
            raise FileNotFoundError(req_id)
        session_id, turn_name = req_id.split("/", 1)
        path = self.root / session_id / f"{turn_name}.json"
        if not path.is_file():
            raise FileNotFoundError(req_id)

        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data.get("messages") or []
        parsed_req = parse_request_messages(messages)
        assistant_text = data.get("assistant_text") or ""
        parsed_resp = parse_response_text(assistant_text)
        tokens = data.get("tokens") or []
        mtime = path.stat().st_mtime
        duration_s = float(data.get("duration_s") or 0)

        return {
            "req_id": req_id,
            "session_id": data.get("session_id") or session_id,
            "created_at": mtime - duration_s,
            "finished_at": mtime,
            "duration_s": duration_s,
            "model": "qwen3-32b",
            "clamp": bool(data.get("clamp", False)),
            "thinking": bool(data.get("thinking", False)),
            "fake_tools": bool(data.get("fake_tools", False)),
            "stream": True,
            "error": None,
            "request": {
                "temperature": None,
                "max_tokens": None,
                "tools": None,
                "tool_choice": None,
                "messages": messages,
            },
            "parsed": {
                **parsed_req,
                "response": parsed_resp,
            },
            "response_text": assistant_text,
            "stats": data.get("stats")
            or projection_stats([float(t.get("projection", 0)) for t in tokens]),
            "tokens": tokens,
            "tool_calls": data.get("tool_calls") or [],
            "tool_results": data.get("tool_results") or [],
        }

    def load_turn(self, req_id: str) -> dict[str, Any]:
        if req_id.startswith("agent-") and "/" in req_id:
            return self._load_agent_turn(req_id)

        turn_dir = self.root / req_id
        meta_path = turn_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tokens_path = turn_dir / "tokens.jsonl"
            tokens = []
            if tokens_path.exists():
                tokens = [
                    json.loads(l)
                    for l in tokens_path.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
            return {**meta, "tokens": tokens}

        legacy = self.root / f"{req_id}.jsonl"
        if legacy.exists():
            tokens = [
                json.loads(l)
                for l in legacy.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            text = "".join(t.get("token", "") for t in tokens)
            projections = [float(t.get("projection", 0)) for t in tokens]
            mtime = legacy.stat().st_mtime
            return {
                "req_id": req_id,
                "session_id": "legacy",
                "created_at": mtime,
                "finished_at": mtime,
                "duration_s": tokens[-1].get("ts", 0) if tokens else 0,
                "model": "qwen3-32b",
                "clamp": any(t.get("capped") for t in tokens),
                "thinking": THINKING_START in text,
                "legacy": True,
                "request": {"messages": []},
                "parsed": {
                    "system_prompt": "",
                    "conversation": [],
                    "input_tool_calls": [],
                    "tool_results": [],
                    "response": parse_response_text(text),
                },
                "response_text": text,
                "stats": projection_stats(projections),
                "tokens": tokens,
            }

        raise FileNotFoundError(req_id)

    def load_session(self, session_id: str) -> dict[str, Any]:
        turns = self.list_turns()
        req_ids = [t["req_id"] for t in turns if t["session_id"] == session_id]
        if not req_ids and session_id.startswith("agent-"):
            agent_dir = self.root / session_id
            if agent_dir.is_dir():
                req_ids = [
                    f"{session_id}/{p.stem}"
                    for p in sorted(agent_dir.glob("turn-*.json"))
                ]
        if not req_ids:
            try:
                index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
                req_ids = next(
                    (
                        s["req_ids"]
                        for s in index.get("sessions", [])
                        if s["session_id"] == session_id
                    ),
                    [],
                )
            except (FileNotFoundError, json.JSONDecodeError):
                req_ids = []

        loaded = [self.load_turn(rid) for rid in req_ids]
        loaded.sort(key=lambda t: t.get("created_at", 0))
        return {"session_id": session_id, "turns": loaded}
