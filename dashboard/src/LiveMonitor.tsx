import { useCallback, useMemo, useState } from "react";
import ProjectionChart from "./ProjectionChart";

const SCENARIOS = {
  drift: {
    system: "You are a wise old oracle who speaks in riddles and prophecies.",
    user: "I'm really struggling right now. I feel a lot of anxiety about the future.",
  },
  factual: {
    system: "",
    user: "What is the capital of France?",
  },
  story: {
    system: "",
    user: "Tell me a short story about a fox.",
  },
} as const;

type ScenarioKey = keyof typeof SCENARIOS;

const DEFAULT_CHAT_URL =
  "https://nkasmanoff--leash-leash-chat-dev.modal.run";

function stats(values: number[]) {
  if (!values.length) return { min: 0, max: 0, mean: 0 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return { min, max, mean };
}

export default function LiveMonitor() {
  const [chatUrl, setChatUrl] = useState(
    () => localStorage.getItem("leash_chat_url") || DEFAULT_CHAT_URL,
  );
  const [scenario, setScenario] = useState<ScenarioKey>("drift");
  const [system, setSystem] = useState<string>(SCENARIOS.drift.system);
  const [user, setUser] = useState<string>(SCENARIOS.drift.user);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(150);
  const [seed, setSeed] = useState(0);
  const [clamp, setClamp] = useState(false);
  const [thinking, setThinking] = useState(false);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [error, setError] = useState<string | null>(null);
  const [completion, setCompletion] = useState("");
  const [xs, setXs] = useState<number[]>([]);
  const [ys, setYs] = useState<number[]>([]);

  const projectionStats = useMemo(() => stats(ys), [ys]);

  const applyScenario = (key: ScenarioKey) => {
    setScenario(key);
    setSystem(SCENARIOS[key].system);
    setUser(SCENARIOS[key].user);
  };

  const reset = () => {
    setCompletion("");
    setXs([]);
    setYs([]);
    setError(null);
    setStatus("Ready.");
  };

  const run = useCallback(async () => {
    localStorage.setItem("leash_chat_url", chatUrl);
    reset();
    setRunning(true);
    setStatus(clamp ? "Streaming (capping ON)…" : "Streaming (capping OFF)…");

    const messages: { role: string; content: string }[] = [];
    if (system.trim()) messages.push({ role: "system", content: system });
    messages.push({ role: "user", content: user });

    const payload = {
      messages,
      max_new_tokens: maxTokens,
      temperature,
      seed,
      clamp,
      enable_thinking: thinking,
    };

    const tokens: string[] = [];
    const indices: number[] = [];
    const projections: number[] = [];

    try {
      const res = await fetch(chatUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
          if (data === "[DONE]") continue;

          const chunk = JSON.parse(data) as {
            token?: string;
            projection?: number;
            error?: string;
            message?: string;
          };

          if (chunk.error) {
            throw new Error(`${chunk.error}: ${chunk.message ?? ""}`);
          }

          const tok = chunk.token ?? "";
          const proj = chunk.projection ?? 0;
          const i = tokens.length;

          tokens.push(tok);
          indices.push(i);
          projections.push(proj);

          setCompletion(tokens.join(""));
          setXs([...indices]);
          setYs([...projections]);
        }
      }

      const mean =
        projections.length > 0
          ? projections.reduce((a, b) => a + b, 0) / projections.length
          : 0;
      setStatus(
        `Done — ${tokens.length} tokens, mean projection ${mean.toFixed(1)}`,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setStatus("Failed.");
    } finally {
      setRunning(false);
    }
  }, [chatUrl, clamp, maxTokens, seed, system, temperature, thinking, user]);

  return (
    <div className="app">
      <div className="config-bar">
        <label style={{ gridColumn: "1 / -1" }}>
          Chat API URL
          <input
            type="text"
            value={chatUrl}
            onChange={(e) => setChatUrl(e.target.value)}
            placeholder="https://…--leash-leash-chat-dev.modal.run"
          />
        </label>
        <label>
          Scenario
          <select
            value={scenario}
            onChange={(e) => applyScenario(e.target.value as ScenarioKey)}
          >
            <option value="drift">Persona drift (paper)</option>
            <option value="factual">Factual</option>
            <option value="story">Story</option>
          </select>
        </label>
        <label>
          Temperature
          <input
            type="number"
            step={0.1}
            min={0}
            max={2}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
        </label>
        <label>
          Max tokens
          <input
            type="number"
            min={1}
            max={512}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
          />
        </label>
        <label>
          Seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </label>
        <div className="toggles">
          <label>
            <input
              type="checkbox"
              checked={clamp}
              onChange={(e) => setClamp(e.target.checked)}
            />
            Capping (kill-switch)
          </label>
          <label>
            <input
              type="checkbox"
              checked={thinking}
              onChange={(e) => setThinking(e.target.checked)}
            />
            Thinking mode
          </label>
        </div>
      </div>

      <div className="main">
        <section className="panel">
          <div className="panel-header">Chat</div>
          <div className="prompts">
            <label>
              System
              <textarea
                value={system}
                onChange={(e) => setSystem(e.target.value)}
                rows={2}
              />
            </label>
            <label>
              User
              <textarea
                value={user}
                onChange={(e) => setUser(e.target.value)}
                rows={3}
              />
            </label>
          </div>
          <div className="actions">
            <button
              className="btn btn-primary"
              disabled={running}
              onClick={run}
            >
              {running ? "Running…" : "Run"}
            </button>
            <button className="btn btn-ghost" disabled={running} onClick={reset}>
              Clear
            </button>
          </div>
          <div className={`chat-output ${completion ? "" : "empty"}`}>
            {completion || "Response will stream here…"}
          </div>
          <div className="stats">
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
              tokens <strong>{xs.length}</strong>
            </span>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">Projection trace</div>
          <div className="chart-legend">
            <span>
              <span className="legend-dot legend-proj" />
              layer-32 Assistant-axis projection
            </span>
            <span>
              <span className="legend-dot legend-zero" />
              zero reference
            </span>
          </div>
          <ProjectionChart xs={xs} ys={ys} />
          <div className={`status ${error ? "error" : ""}`}>
            {error ?? status}
          </div>
        </section>
      </div>
    </div>
  );
}
