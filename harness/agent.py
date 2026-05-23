"""Agent loop: Leash chat → parse tools → execute → repeat."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from harness.client import TokenChunk, iter_chat_tokens
from harness.parse import ToolCall, parse_tool_calls, strip_model_artifacts, visible_reply
from harness.prompts import build_system_prompt
from harness.fake_tools import FakeToolContext
from harness.tools import ToolResult, execute


@dataclass
class TurnRecord:
    turn: int
    assistant_text: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tokens: list[dict[str, Any]]
    stats: dict[str, float | int]
    duration_s: float


@dataclass
class AgentConfig:
    leash_url: str
    cwd: str
    max_turns: int = 20
    max_new_tokens: int = 2048
    temperature: float = 0.7
    seed: int | None = None
    thinking: bool = False
    clamp: bool = False
    fake_tools: bool = False
    trace_dir: Path | None = None
    stream: bool = True


@dataclass
class RunResult:
    reply: str
    turns_used: int
    last_turn: TurnRecord | None = None


@dataclass
class AgentSession:
    config: AgentConfig
    messages: list[dict[str, str]] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"agent-{int(time.time())}-{uuid.uuid4().hex[:6]}")
    fake_ctx: FakeToolContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.config.fake_tools and self.fake_ctx is None:
            self.fake_ctx = FakeToolContext(cwd=self.config.cwd)
        if not self.messages:
            self.messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        cwd=self.config.cwd,
                        thinking=self.config.thinking,
                        fake_tools=self.config.fake_tools,
                    ),
                }
            ]

    def sync_tool_mode(self) -> None:
        """Refresh system prompt and in-memory tool state from config."""
        if not self.messages:
            return
        self.messages[0] = {
            "role": "system",
            "content": build_system_prompt(
                cwd=self.config.cwd,
                thinking=self.config.thinking,
                fake_tools=self.config.fake_tools,
            ),
        }
        if self.config.fake_tools:
            if self.fake_ctx is None:
                self.fake_ctx = FakeToolContext(cwd=self.config.cwd)
        else:
            self.fake_ctx = None

    def run_user_message(
        self,
        user_text: str,
        *,
        on_token: Callable[[TokenChunk], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunResult:
        final_reply = ""
        last_record: TurnRecord | None = None

        for event in self.iter_user_message(user_text):
            if on_event is not None:
                on_event(event)
            if event["type"] == "token" and on_token is not None:
                on_token(
                    TokenChunk(
                        token=event["token"],
                        projection=event["projection"],
                        capped=event.get("capped", False),
                        token_id=event.get("token_id"),
                    )
                )
            elif event["type"] == "turn_end":
                last_record = self.turns[-1] if self.turns else None
            elif event["type"] == "done":
                final_reply = event.get("reply", "")
            elif event["type"] == "error":
                return RunResult(
                    reply=event.get("message", "error"),
                    turns_used=len(self.turns),
                    last_turn=last_record,
                )

        return RunResult(
            reply=final_reply,
            turns_used=len(self.turns),
            last_turn=last_record,
        )

    def iter_user_message(self, user_text: str) -> Iterator[dict[str, Any]]:
        """Yield SSE-friendly events for the dashboard agent UI."""
        self.messages.append({"role": "user", "content": user_text})
        global_i = sum(len(t.tokens) for t in self.turns)
        final_reply = ""

        yield {"type": "session", "session_id": self.session_id}

        for turn_idx in range(self.config.max_turns):
            t0 = time.time()
            yield {"type": "turn_start", "turn": turn_idx + 1}

            parts: list[str] = []
            tokens: list[TokenChunk] = []
            try:
                for chunk in iter_chat_tokens(
                    self.config.leash_url,
                    self.messages,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    seed=self.config.seed,
                    enable_thinking=self.config.thinking,
                    clamp=self.config.clamp,
                ):
                    parts.append(chunk.token)
                    tokens.append(chunk)
                    yield {
                        "type": "token",
                        "turn": turn_idx + 1,
                        "i": global_i,
                        "token": chunk.token,
                        "projection": chunk.projection,
                        "capped": chunk.capped,
                        "token_id": chunk.token_id,
                    }
                    global_i += 1
            except RuntimeError as exc:
                yield {"type": "error", "message": str(exc)}
                return

            text = "".join(parts)
            calls = parse_tool_calls(text)
            reply = visible_reply(text)

            record = TurnRecord(
                turn=turn_idx + 1,
                assistant_text=text,
                tool_calls=[
                    {"name": c.name, "arguments": c.arguments} for c in calls
                ],
                tool_results=[],
                tokens=[
                    {
                        "i": i,
                        "token": t.token,
                        "projection": t.projection,
                        "capped": t.capped,
                        "token_id": t.token_id,
                        "ts": time.time() - t0,
                    }
                    for i, t in enumerate(tokens)
                ],
                stats=_token_stats(tokens),
                duration_s=time.time() - t0,
            )

            self.messages.append({"role": "assistant", "content": text})
            self.turns.append(record)
            self._save_turn(record)

            yield {
                "type": "turn_end",
                "turn": turn_idx + 1,
                "stats": record.stats,
                "duration_s": record.duration_s,
                "has_tools": bool(calls),
            }

            if not calls:
                final_reply = reply
                break

            for call in calls:
                yield {
                    "type": "tool_call",
                    "turn": turn_idx + 1,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                if self.config.fake_tools and self.fake_ctx is None:
                    self.fake_ctx = FakeToolContext(cwd=self.config.cwd)
                result = execute(
                    call,
                    self.config.cwd,
                    fake=self.config.fake_tools,
                    ctx=self.fake_ctx,
                )
                record.tool_results.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": result.ok,
                        "output": result.output,
                    }
                )
                yield {
                    "type": "tool_result",
                    "turn": turn_idx + 1,
                    "name": call.name,
                    "ok": result.ok,
                    "output": result.output,
                }
                tool_msg = (
                    f"<tool_result name={call.name} ok={result.ok}>\n"
                    f"{result.output}\n"
                    f"</tool_result>"
                )
                self.messages.append({"role": "user", "content": tool_msg})

            self._save_turn(record)

        yield {"type": "done", "reply": final_reply, "session_id": self.session_id}

    def _save_turn(self, record: TurnRecord) -> None:
        if not self.config.trace_dir:
            return
        root = self.config.trace_dir / self.session_id
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"turn-{record.turn:02d}.json"
        path.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "turn": record.turn,
                    "messages": self.messages,
                    "assistant_text": record.assistant_text,
                    "tool_calls": record.tool_calls,
                    "tool_results": record.tool_results,
                    "stats": record.stats,
                    "duration_s": record.duration_s,
                    "clamp": self.config.clamp,
                    "thinking": self.config.thinking,
                    "fake_tools": self.config.fake_tools,
                    "tokens": record.tokens,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _token_stats(tokens: list[TokenChunk]) -> dict[str, float | int]:
    if not tokens:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "n_tokens": 0}
    projs = [t.projection for t in tokens]
    return {
        "min": min(projs),
        "max": max(projs),
        "mean": sum(projs) / len(projs),
        "n_tokens": len(projs),
    }
