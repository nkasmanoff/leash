import type { SessionBundle, TraceListResponse, TurnDetail } from "../types/harness";

const API = "/api";

export async function fetchTraceList(): Promise<TraceListResponse> {
  const res = await fetch(`${API}/traces`);
  if (!res.ok) throw new Error(`Failed to list traces: HTTP ${res.status}`);
  return res.json();
}

export async function fetchTurn(reqId: string): Promise<TurnDetail> {
  const res = await fetch(`${API}/traces/${encodeURIComponent(reqId)}`);
  if (!res.ok) throw new Error(`Failed to load trace: HTTP ${res.status}`);
  return res.json();
}

export async function fetchSession(sessionId: string): Promise<SessionBundle> {
  const res = await fetch(`${API}/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(`Failed to load session: HTTP ${res.status}`);
  return res.json();
}

export function fmtTime(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

export function fmtDuration(s: number): string {
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  return `${s.toFixed(1)}s`;
}
