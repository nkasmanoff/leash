export type AgentEvent =
  | { type: "session"; session_id: string }
  | { type: "turn_start"; turn: number }
  | {
      type: "token";
      turn: number;
      i: number;
      token: string;
      projection: number;
      capped?: boolean;
    }
  | {
      type: "turn_end";
      turn: number;
      stats: { min: number; max: number; mean: number; n_tokens: number };
      duration_s: number;
      has_tools: boolean;
    }
  | { type: "tool_call"; turn: number; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; turn: number; name: string; ok: boolean; output: string }
  | { type: "done"; reply: string; session_id: string }
  | { type: "error"; message: string };

export type AgentChatOptions = {
  message: string;
  sessionId: string | null;
  clamp: boolean;
  thinking: boolean;
  fakeTools?: boolean;
  temperature?: number;
  maxNewTokens?: number;
};

export async function streamAgentChat(
  opts: AgentChatOptions,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: opts.message,
      session_id: opts.sessionId,
      clamp: opts.clamp,
      thinking: opts.thinking,
      fake_tools: opts.fakeTools ?? false,
      temperature: opts.temperature ?? 0.7,
      max_new_tokens: opts.maxNewTokens ?? 2048,
    }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Agent HTTP ${res.status}: ${text}`);
  }
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      onEvent(JSON.parse(data) as AgentEvent);
    }
  }
}

export async function resetAgentSession(
  sessionId: string | null,
  opts: { fakeTools?: boolean; thinking?: boolean; clamp?: boolean } = {},
): Promise<string> {
  const res = await fetch("/api/agent/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      fake_tools: opts.fakeTools ?? false,
      thinking: opts.thinking ?? false,
      clamp: opts.clamp ?? false,
    }),
  });
  if (!res.ok) throw new Error(`Reset failed: HTTP ${res.status}`);
  const body = (await res.json()) as { session_id: string };
  return body.session_id;
}
