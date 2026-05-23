import { useCallback, useEffect, useMemo, useState } from "react";
import SessionTrajectoryChart, {
  type TrajectoryPoint,
  type TurnBoundary,
} from "./SessionTrajectoryChart";
import {
  fetchSession,
  fetchTraceList,
  fetchTurn,
  fmtDuration,
  fmtTime,
} from "./lib/harnessApi";
import {
  analyzeSessionActivations,
  analyzeTurnActivations,
  driftBadge,
} from "./lib/traceAnalysis";
import {
  SessionActivationSummary,
  TurnActivationDetail,
} from "./components/ActivationInsightsPanel";
import type { TurnDetail, TurnSummary } from "./types/harness";
import "./HarnessApp.css";
import "./components/ActivationInsightsPanel.css";

function segmentClass(kind: string): string {
  if (kind === "thinking") return "seg-thinking";
  if (kind === "content") return "seg-content";
  return "seg-other";
}

function buildSessionTrajectory(turns: TurnDetail[]) {
  const points: TrajectoryPoint[] = [];
  const boundaries: TurnBoundary[] = [];
  let offset = 0;

  turns.forEach((turn, turnIndex) => {
    if (turnIndex > 0) {
      boundaries.push({ x: offset, label: `Turn ${turnIndex + 1}` });
    }
    turn.tokens.forEach((tok, i) => {
      points.push({
        x: offset + i,
        y: tok.projection,
        turnIndex,
        reqId: turn.req_id,
      });
    });
    offset += turn.tokens.length;
  });

  return { points, boundaries };
}

function ResponseView({ turn }: { turn: TurnDetail }) {
  const segments = turn.parsed?.response?.segments ?? [];
  if (!segments.length) {
    return (
      <pre className="response-raw">{turn.response_text || "(empty response)"}</pre>
    );
  }
  return (
    <div className="response-segments">
      {segments.map((seg, i) => (
        <div key={i} className={`response-seg ${segmentClass(seg.kind)}`}>
          <div className="seg-label">{seg.kind}</div>
          <pre>{seg.text}</pre>
        </div>
      ))}
    </div>
  );
}

export default function HarnessApp() {
  const [turns, setTurns] = useState<TurnSummary[]>([]);
  const [sessions, setSessions] = useState<
    Array<{ session_id: string; req_ids: string[]; turn_count: number }>
  >([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [selectedReqId, setSelectedReqId] = useState<string | null>(null);
  const [sessionTurns, setSessionTurns] = useState<TurnDetail[]>([]);
  const [activeTurn, setActiveTurn] = useState<TurnDetail | null>(null);
  const [hover, setHover] = useState<TrajectoryPoint | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshList = useCallback(async () => {
    try {
      const data = await fetchTraceList();
      setTurns(data.turns);
      setSessions(data.sessions);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refreshList();
  }, [refreshList]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(refreshList, 4000);
    return () => clearInterval(id);
  }, [autoRefresh, refreshList]);

  const loadSession = useCallback(async (sessionId: string) => {
    setLoading(true);
    setError(null);
    try {
      const bundle = await fetchSession(sessionId);
      setSessionTurns(bundle.turns);
      setSelectedSessionId(sessionId);
      const first = bundle.turns[0]?.req_id ?? null;
      setSelectedReqId(first);
      setActiveTurn(bundle.turns.find((t) => t.req_id === first) ?? bundle.turns[0] ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadTurn = useCallback(async (reqId: string) => {
    setLoading(true);
    setError(null);
    try {
      const turn = await fetchTurn(reqId);
      setActiveTurn(turn);
      setSelectedReqId(reqId);
      setSelectedSessionId(turn.session_id);
      setSessionTurns([turn]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const { points, boundaries } = useMemo(
    () => buildSessionTrajectory(sessionTurns),
    [sessionTurns],
  );

  const sessionInsights = useMemo(
    () => analyzeSessionActivations(sessionTurns),
    [sessionTurns],
  );

  const activeTurnInsight = useMemo(() => {
    if (!activeTurn) return null;
    const idx = sessionTurns.findIndex((t) => t.req_id === activeTurn.req_id);
    const label = idx >= 0 ? `Turn ${idx + 1}` : activeTurn.req_id;
    return analyzeTurnActivations(activeTurn, label);
  }, [activeTurn, sessionTurns]);

  const deviationTokenIndices = useMemo(() => {
    const set = new Set<number>();
    for (const ev of activeTurnInsight?.deviations ?? []) {
      set.add(ev.tokenIndex);
    }
    return set;
  }, [activeTurnInsight]);

  const sessionStats = sessionInsights.combined.stats;

  return (
    <div className="harness-app">
      <aside className="harness-sidebar">
        <div className="sidebar-toolbar">
          <button type="button" className="btn btn-ghost btn-sm" onClick={refreshList}>
            Refresh
          </button>
          <label className="auto-refresh">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto
          </label>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">Sessions</div>
          {sessions.length === 0 && (
            <div className="sidebar-empty">No sessions yet. Run OpenCode with the shim.</div>
          )}
          {sessions.map((s) => (
            <button
              key={s.session_id}
              type="button"
              className={`sidebar-item ${selectedSessionId === s.session_id ? "active" : ""}`}
              onClick={() => loadSession(s.session_id)}
            >
              <div className="item-title">{s.session_id}</div>
              <div className="item-meta">{s.turn_count} turn(s)</div>
            </button>
          ))}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">Turns</div>
          {turns.map((t) => (
            <button
              key={t.req_id}
              type="button"
              className={`sidebar-item ${selectedReqId === t.req_id ? "active" : ""}`}
              onClick={() => loadTurn(t.req_id)}
            >
              <div className="item-title">{t.req_id}</div>
              <div className="item-meta">
                {fmtTime(t.created_at)} · {t.stats.n_tokens} tok · mean{" "}
                {t.stats.mean.toFixed(1)} · min {t.stats.min.toFixed(1)}
              </div>
              <div
                className={`sidebar-drift ${
                  driftBadge(t.stats.mean) === "major drift"
                    ? "major"
                    : driftBadge(t.stats.mean) === "mild drift"
                      ? "mild"
                      : "aligned"
                }`}
              >
                {driftBadge(t.stats.mean)} · peak {t.stats.max.toFixed(1)}
              </div>
              <div className="item-preview">{t.preview.last_user}</div>
            </button>
          ))}
        </div>
      </aside>

      <main className="harness-main">
        {error && <div className="harness-banner error">{error}</div>}
        {loading && <div className="harness-banner">Loading…</div>}

        {!activeTurn && !loading && (
          <div className="harness-empty">
            Select a session or turn from the sidebar. Traces come from{" "}
            <code>traces/harness/</code> via the shim API on port 8787.
          </div>
        )}

        {activeTurn && (
          <>
            <header className="harness-header">
              <div>
                <h2>{selectedSessionId ?? activeTurn.session_id}</h2>
                <div className="harness-sub">
                  {sessionTurns.length} turn(s) · {sessionStats.nTokens} tokens · mean{" "}
                  {sessionStats.mean.toFixed(1)} · min {sessionStats.min.toFixed(1)}
                </div>
              </div>
              <div className="harness-badges">
                {activeTurn.clamp && <span className="badge badge-clamp">capping ON</span>}
                {activeTurn.thinking && <span className="badge badge-think">thinking</span>}
                {activeTurn.fake_tools && (
                  <span className="badge badge-fake">fake tools</span>
                )}
                {activeTurn.legacy && <span className="badge badge-legacy">legacy trace</span>}
                {activeTurn.error && <span className="badge badge-err">error</span>}
              </div>
            </header>

            <section className="harness-chart-panel">
              <div className="panel-header">Session trajectory</div>
              <div className="chart-legend">
                <span>
                  <span className="legend-dot legend-proj" /> projection
                </span>
                <span>
                  <span className="legend-dot legend-turn" /> turn boundary
                </span>
                {hover && (
                  <span className="hover-meta">
                    token {hover.x} · proj {hover.y.toFixed(2)} · turn{" "}
                    {hover.turnIndex + 1}
                  </span>
                )}
              </div>
              <SessionTrajectoryChart
                points={points}
                boundaries={boundaries}
                onHover={setHover}
              />
              <div className="stats">
                <span>
                  min <strong>{sessionStats.min.toFixed(1)}</strong>
                </span>
                <span>
                  max <strong>{sessionStats.max.toFixed(1)}</strong>
                </span>
                <span>
                  mean <strong>{sessionStats.mean.toFixed(1)}</strong>
                </span>
                <span>
                  major drift{" "}
                  <strong>{sessionInsights.combined.deepDrift.pct.toFixed(0)}%</strong>
                </span>
              </div>
            </section>

            <SessionActivationSummary insights={sessionInsights} />
            <TurnActivationDetail insight={activeTurnInsight} />

            {sessionTurns.length > 1 && (
              <div className="turn-tabs">
                {sessionTurns.map((t, i) => {
                  const insight = sessionInsights.turns[i];
                  return (
                    <button
                      key={t.req_id}
                      type="button"
                      className={`turn-tab ${selectedReqId === t.req_id ? "active" : ""}`}
                      onClick={() => {
                        setSelectedReqId(t.req_id);
                        setActiveTurn(t);
                      }}
                    >
                      Turn {i + 1} · {t.stats.n_tokens} tok · μ{" "}
                      {t.stats.mean.toFixed(1)}
                      {insight && (
                        <span className={`turn-tab-drift ${insight.severity}`}>
                          {insight.severity}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}

            <div className="harness-panels">
              <section className="harness-panel">
                <div className="panel-header">Request context</div>
                <div className="meta-grid">
                  <div>
                    <span className="meta-k">req_id</span>
                    <code>{activeTurn.req_id}</code>
                  </div>
                  <div>
                    <span className="meta-k">model</span>
                    <code>{activeTurn.model}</code>
                  </div>
                  <div>
                    <span className="meta-k">duration</span>
                    <code>{fmtDuration(activeTurn.duration_s)}</code>
                  </div>
                  <div>
                    <span className="meta-k">temperature</span>
                    <code>{activeTurn.request.temperature ?? "—"}</code>
                  </div>
                </div>

                <div className="block-label">System prompt</div>
                <pre className="code-block system-block">
                  {activeTurn.parsed.system_prompt || "(none in request)"}
                </pre>

                {activeTurn.request.tools && activeTurn.request.tools.length > 0 && (
                  <>
                    <div className="block-label">
                      Tools schema ({activeTurn.request.tools.length})
                    </div>
                    <pre className="code-block">
                      {JSON.stringify(activeTurn.request.tools, null, 2)}
                    </pre>
                  </>
                )}

                <div className="block-label">Message history</div>
                <div className="message-list">
                  {activeTurn.parsed.conversation.map((msg) => (
                    <div key={msg.index} className={`message message-${msg.kind}`}>
                      <div className="message-role">
                        {msg.role}
                        {msg.name ? ` (${msg.name})` : ""}
                      </div>
                      <pre>{msg.content || "(empty)"}</pre>
                      {msg.tool_calls && msg.tool_calls.length > 0 && (
                        <div className="tool-call-list">
                          {msg.tool_calls.map((tc, i) => (
                            <div key={i} className="tool-call">
                              <strong>{tc.function?.name}</strong>
                              <pre>{tc.function?.arguments}</pre>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {(activeTurn.parsed.input_tool_calls.length > 0 ||
                  activeTurn.parsed.tool_results.length > 0) && (
                  <>
                    <div className="block-label">Tool activity (from context)</div>
                    <div className="tool-summary">
                      <span>{activeTurn.parsed.input_tool_calls.length} call(s)</span>
                      <span>{activeTurn.parsed.tool_results.length} result(s)</span>
                    </div>
                  </>
                )}
              </section>

              <section className="harness-panel">
                <div className="panel-header">Model output</div>
                <ResponseView turn={activeTurn} />

                {activeTurn.parsed.response.tool_markers.length > 0 && (
                  <>
                    <div className="block-label">Detected tool markers</div>
                    <ul className="tool-markers">
                      {activeTurn.parsed.response.tool_markers.map((m, i) => (
                        <li key={i}>
                          <code>{m.kind}</code> {m.command}
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                <div className="block-label">Token stream</div>
                <div className="token-table-wrap">
                  <table className="token-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>proj</th>
                        <th>token</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeTurn.tokens.map((tok) => {
                        const idx = tok.i ?? 0;
                        const isDrift = tok.projection <= -30;
                        const isPeak = tok.projection > 0;
                        const flagged = deviationTokenIndices.has(idx);
                        const rowClass = flagged
                          ? isDrift
                            ? "token-row-drift"
                            : isPeak
                              ? "token-row-peak"
                              : "token-row-warn"
                          : "";
                        return (
                          <tr key={idx} className={rowClass}>
                            <td>{idx}</td>
                            <td>{tok.projection.toFixed(2)}</td>
                            <td>
                              <code>{JSON.stringify(tok.token)}</code>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
