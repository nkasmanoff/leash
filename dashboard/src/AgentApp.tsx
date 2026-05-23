import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ColoredTokenStream, {
  type StreamToken,
} from "./components/ColoredTokenStream";
import ActivationStrip from "./components/ActivationStrip";
import { stats } from "./lib/projectionColor";
import {
  resetAgentSession,
  streamAgentChat,
  type AgentEvent,
} from "./lib/agentApi";
import "./AgentApp.css";

type ToolBlock = {
  name: string;
  command: string;
  output: string;
  ok: boolean;
};

type ChatItem =
  | { id: string; kind: "user"; text: string }
  | {
      id: string;
      kind: "assistant";
      tokens: StreamToken[];
      streaming?: boolean;
    }
  | { id: string; kind: "tool"; tool: ToolBlock };

let _id = 0;
function uid() {
  return `m-${++_id}`;
}

const UserBubble = memo(function UserBubble({ text }: { text: string }) {
  return <div className="bubble bubble-user">{text}</div>;
});

const ToolBubble = memo(function ToolBubble({ tool }: { tool: ToolBlock }) {
  const [expanded, setExpanded] = useState(false);
  const long = tool.output.length > 600;
  const shown =
    long && !expanded ? `${tool.output.slice(0, 600)}…` : tool.output;

  return (
    <div className="bubble bubble-tool">
      <div className="tool-head">
        <span>{tool.name}</span>
        <span className={tool.ok ? "ok" : "fail"}>{tool.ok ? "ok" : "fail"}</span>
      </div>
      {tool.command && <pre className="tool-cmd">$ {tool.command}</pre>}
      {tool.output && (
        <>
          <pre className="tool-out">{shown}</pre>
          {long && (
            <button
              type="button"
              className="btn btn-ghost btn-sm tool-expand"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "Show less" : "Show full output"}
            </button>
          )}
        </>
      )}
    </div>
  );
});

const AssistantBubble = memo(function AssistantBubble({
  tokens,
  streaming,
  highlightIndex,
  onHoverIndex,
}: {
  tokens: StreamToken[];
  streaming?: boolean;
  highlightIndex?: number | null;
  onHoverIndex?: (index: number | null) => void;
}) {
  return (
    <div
      className={`bubble bubble-assistant ${streaming ? "streaming" : ""}`}
    >
      <ColoredTokenStream
        tokens={tokens}
        highlightIndex={highlightIndex}
        onHoverIndex={onHoverIndex}
        compact={streaming}
      />
    </div>
  );
});

export default function AgentApp() {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [clamp, setClamp] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [fakeTools, setFakeTools] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [error, setError] = useState<string | null>(null);

  const [chartXs, setChartXs] = useState<number[]>([]);
  const [chartYs, setChartYs] = useState<number[]>([]);
  const [turnBoundaries, setTurnBoundaries] = useState<number[]>([]);
  const [liveProj, setLiveProj] = useState<number | null>(null);
  const [highlightIndex, setHighlightIndex] = useState<number | null>(null);
  const [streamTick, setStreamTick] = useState(0);
  const [streamingAssistantId, setStreamingAssistantId] =
    useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const assistantIdRef = useRef<string | null>(null);
  const liveTokensRef = useRef<StreamToken[]>([]);
  const chartXsRef = useRef<number[]>([]);
  const chartYsRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);
  const liveProjRef = useRef<number | null>(null);

  const projectionStats = useMemo(() => stats(chartYs), [chartYs]);

  useEffect(() => {
    let cancelled = false;
    void resetAgentSession(null, { fakeTools })
      .then((sid) => {
        if (!cancelled) {
          setSessionId(sid);
          setStatus(fakeTools ? "Ready (fake tools on)." : "Ready (real tools).");
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      });
    return () => {
      cancelled = true;
    };
    // Only establish the initial server session once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setChartXs([...chartXsRef.current]);
      setChartYs([...chartYsRef.current]);
      setLiveProj(liveProjRef.current);
      setStreamTick((t) => t + 1);
    });
  }, []);

  const clearSession = useCallback(async (opts?: { fakeTools?: boolean }) => {
    abortRef.current?.abort();
    const useFakeTools = opts?.fakeTools ?? fakeTools;
    const sid = await resetAgentSession(sessionId, {
      fakeTools: useFakeTools,
      thinking,
      clamp,
    });
    setSessionId(sid);
    setItems([]);
    setChartXs([]);
    setChartYs([]);
    chartXsRef.current = [];
    chartYsRef.current = [];
    liveTokensRef.current = [];
    setTurnBoundaries([]);
    setLiveProj(null);
    setHighlightIndex(null);
    setStreamingAssistantId(null);
    setError(null);
    setStatus("New session.");
  }, [sessionId, fakeTools, thinking, clamp]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || running) return;

    setInput("");
    setError(null);
    setRunning(true);
    setStatus("Agent running…");
    setChartXs([]);
    setChartYs([]);
    chartXsRef.current = [];
    chartYsRef.current = [];
    liveTokensRef.current = [];
    setTurnBoundaries([]);
    setLiveProj(null);
    setHighlightIndex(null);

    setItems((prev) => [...prev, { id: uid(), kind: "user", text }]);

    const assistantId = uid();
    assistantIdRef.current = assistantId;
    setStreamingAssistantId(assistantId);
    liveTokensRef.current = [];
    setItems((prev) => [
      ...prev,
      { id: assistantId, kind: "assistant", tokens: [], streaming: true },
    ]);

    const ac = new AbortController();
    abortRef.current = ac;

    const commitLiveTokens = () => {
      const id = assistantIdRef.current;
      if (!id) return;
      const snapshot = liveTokensRef.current;
      setItems((prev) =>
        prev.map((it) =>
          it.id === id && it.kind === "assistant"
            ? { ...it, tokens: snapshot }
            : it,
        ),
      );
    };

    const handleEvent = (event: AgentEvent) => {
      switch (event.type) {
        case "session":
          setSessionId(event.session_id);
          break;
        case "turn_start":
          if (event.turn > 1) {
            commitLiveTokens();
            const prevAssistantId = assistantIdRef.current;
            const nextAssistantId = uid();
            assistantIdRef.current = nextAssistantId;
            setStreamingAssistantId(nextAssistantId);
            liveTokensRef.current = [];
            setItems((prev) => [
              ...prev.map((it) =>
                it.id === prevAssistantId && it.kind === "assistant"
                  ? { ...it, streaming: false }
                  : it,
              ),
              {
                id: nextAssistantId,
                kind: "assistant",
                tokens: [],
                streaming: true,
              },
            ]);
          }
          if (chartXsRef.current.length > 0) {
            setTurnBoundaries((b) => [...b, chartXsRef.current.length]);
          }
          setStatus(`Model turn ${event.turn}…`);
          break;
        case "token": {
          liveTokensRef.current.push({
            token: event.token,
            projection: event.projection,
            index: event.i,
          });
          chartXsRef.current.push(event.i);
          chartYsRef.current.push(event.projection);
          liveProjRef.current = event.projection;
          scheduleFlush();
          break;
        }
        case "tool_call": {
          commitLiveTokens();
          const args = event.arguments as {
            command?: string;
            path?: string;
          };
          const cmd = String(args.command ?? args.path ?? "");
          setItems((prev) => [
            ...prev.map((it) =>
              it.id === assistantIdRef.current && it.kind === "assistant"
                ? { ...it, streaming: false }
                : it,
            ),
            {
              id: uid(),
              kind: "tool",
              tool: {
                name: event.name,
                command: cmd,
                output: "",
                ok: true,
              },
            },
          ]);
          assistantIdRef.current = null;
          setStreamingAssistantId(null);
          liveTokensRef.current = [];
          break;
        }
        case "tool_result":
          setItems((prev) => {
            const idx = [...prev]
              .reverse()
              .findIndex((it) => it.kind === "tool" && !it.tool.output);
            if (idx === -1) return prev;
            const realIdx = prev.length - 1 - idx;
            const copy = [...prev];
            const row = copy[realIdx];
            if (row.kind === "tool") {
              copy[realIdx] = {
                ...row,
                tool: {
                  ...row.tool,
                  output: event.output,
                  ok: event.ok,
                },
              };
            }
            return copy;
          });
          break;
        case "turn_end":
          commitLiveTokens();
          setStatus(
            `Turn ${event.turn} · ${event.stats.n_tokens} tok · mean ${event.stats.mean.toFixed(1)}`,
          );
          break;
        case "done":
          commitLiveTokens();
          setItems((prev) =>
            prev.map((it) =>
              it.kind === "assistant" && it.streaming
                ? { ...it, streaming: false }
                : it,
            ),
          );
          setStatus(`Done · ${chartXsRef.current.length} tokens streamed`);
          break;
        case "error":
          setError(event.message);
          setStatus("Failed.");
          break;
      }
    };

    try {
      await streamAgentChat(
        {
          message: text,
          sessionId,
          clamp,
          thinking,
          fakeTools,
        },
        handleEvent,
        ac.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof Error ? e.message : String(e));
        setStatus("Failed.");
      }
    } finally {
      commitLiveTokens();
      setRunning(false);
      setItems((prev) =>
        prev.map((it) =>
          it.kind === "assistant" && it.streaming
            ? { ...it, streaming: false }
            : it,
        ),
      );
      assistantIdRef.current = null;
      setStreamingAssistantId(null);
      liveTokensRef.current = [];
    }
  }, [clamp, fakeTools, input, running, scheduleFlush, sessionId, thinking]);

  void streamTick;

  return (
    <div className="agent-app">
      <div className="agent-config">
        <label>
          <input
            type="checkbox"
            checked={clamp}
            onChange={(e) => setClamp(e.target.checked)}
            disabled={running}
          />
          Capping
        </label>
        <label title="Qwen thinking blocks appear dimmed in the stream">
          <input
            type="checkbox"
            checked={thinking}
            onChange={(e) => setThinking(e.target.checked)}
            disabled={running}
          />
          Thinking
        </label>
        <label title="No-op tools with plausible outputs for stress testing">
          <input
            type="checkbox"
            checked={fakeTools}
            onChange={(e) => {
              const enabled = e.target.checked;
              setFakeTools(enabled);
              if (!running) void clearSession({ fakeTools: enabled });
            }}
            disabled={running}
          />
          Fake tools
        </label>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          disabled={running}
          onClick={() => void clearSession()}
        >
          New session
        </button>
        {sessionId && <span className="session-tag">{sessionId}</span>}
      </div>

      <div className="activation-bar">
        <div className="activation-bar-head">
          <span className="activation-label">Layer-32 activations</span>
          {liveProj != null && (
            <span className="live-proj">
              now <strong>{liveProj.toFixed(2)}</strong>
              {highlightIndex != null && (
                <>
                  {" "}
                  · token <strong>#{highlightIndex}</strong>
                </>
              )}
            </span>
          )}
          <div className="activation-stats">
            <span>
              min <strong>{projectionStats.min.toFixed(1)}</strong>
            </span>
            <span>
              max <strong>{projectionStats.max.toFixed(1)}</strong>
            </span>
            <span>
              mean <strong>{projectionStats.mean.toFixed(1)}</strong>
            </span>
            <span>
              tokens <strong>{chartXs.length}</strong>
            </span>
          </div>
        </div>
        <ActivationStrip
          xs={chartXs}
          ys={chartYs}
          turnBoundaries={turnBoundaries}
          highlightIndex={highlightIndex}
          onHoverIndex={setHighlightIndex}
        />
        <div className="activation-legend">
          <span className="legend-gradient" />
          low → high projection
          {turnBoundaries.length > 0 && (
            <span className="legend-turn-mark">| turn boundary</span>
          )}
        </div>
      </div>

      <section className="agent-chat panel">
          <div className="panel-header">Agent</div>
          <div className="agent-messages">
            {items.length === 0 && (
              <div className="agent-empty">
                Ask the agent to read, edit, or run commands. With fake tools
                enabled, nothing touches disk — useful for stress testing
                projections. Token colors reflect layer-32 projection (red =
                low, green = high). Hover the strip or text to link
                activations.
              </div>
            )}
            {items.map((item) => {
              if (item.kind === "user") {
                return <UserBubble key={item.id} text={item.text} />;
              }
              if (item.kind === "tool") {
                return <ToolBubble key={item.id} tool={item.tool} />;
              }
              const isLive =
                item.streaming && item.id === streamingAssistantId;
              const tokens = isLive
                ? [...liveTokensRef.current]
                : item.tokens;
              return (
                <AssistantBubble
                  key={item.id}
                  tokens={tokens}
                  streaming={item.streaming}
                  highlightIndex={highlightIndex}
                  onHoverIndex={setHighlightIndex}
                />
              );
            })}
          </div>
          <div className="agent-input-row">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask the agent… (Enter to send)"
              rows={2}
              disabled={running}
            />
            <button
              type="button"
              className="btn btn-primary"
              disabled={running || !input.trim()}
              onClick={send}
            >
              {running ? "Running…" : "Send"}
            </button>
          </div>
        </section>

      <div className={`agent-status ${error ? "error" : ""}`}>
        {error ?? status}
      </div>
    </div>
  );
}
