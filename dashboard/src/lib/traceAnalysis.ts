import type { ResponseSegment, TraceToken, TurnDetail } from "../types/harness";

/** Projection below this is considered a major drift off the assistant axis. */
export const DEEP_DRIFT_THRESHOLD = -30;
/** Single-token drop larger than this is flagged as a sharp deviation. */
export const SHARP_DROP_THRESHOLD = 12;

export type SegmentActivation = {
  kind: string;
  nTokens: number;
  mean: number;
  min: number;
  max: number;
};

export type DeviationEvent = {
  kind: "deep_drift" | "sharp_drop" | "peak";
  tokenIndex: number;
  token: string;
  projection: number;
  delta?: number;
  label: string;
};

export type TurnActivationInsights = {
  reqId: string;
  turnLabel: string;
  stats: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
    range: number;
    nTokens: number;
  };
  belowZero: { count: number; pct: number };
  deepDrift: { count: number; pct: number };
  cappedCount: number;
  segments: SegmentActivation[];
  deviations: DeviationEvent[];
  headline: string;
  severity: "low" | "medium" | "high";
};

export type SessionActivationInsights = {
  turns: TurnActivationInsights[];
  combined: TurnActivationInsights;
  turnMeanTrend: number[];
  headline: string;
  severity: "low" | "medium" | "high";
};

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function stdDev(values: number[], mean: number): number {
  if (values.length < 2) return 0;
  const variance =
    values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function severityFromMean(mean: number, deepPct: number): "low" | "medium" | "high" {
  if (mean < -25 || deepPct >= 25) return "high";
  if (mean < -15 || deepPct >= 10) return "medium";
  return "low";
}

function tokenCharOffsets(tokens: TraceToken[]): number[] {
  const offsets: number[] = [];
  let pos = 0;
  for (const tok of tokens) {
    offsets.push(pos);
    pos += tok.token.length;
  }
  return offsets;
}

function segmentKindAt(
  charOffset: number,
  segments: ResponseSegment[],
): string {
  for (const seg of segments) {
    if (charOffset >= seg.start && charOffset < seg.end) return seg.kind;
  }
  return "other";
}

function segmentActivations(
  tokens: TraceToken[],
  segments: ResponseSegment[],
): SegmentActivation[] {
  if (!tokens.length) return [];
  const offsets = tokenCharOffsets(tokens);
  const buckets = new Map<string, number[]>();

  tokens.forEach((tok, i) => {
    const kind = segments.length
      ? segmentKindAt(offsets[i], segments)
      : "stream";
    const list = buckets.get(kind) ?? [];
    list.push(tok.projection);
    buckets.set(kind, list);
  });

  return [...buckets.entries()]
    .map(([kind, values]) => {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      return {
        kind,
        nTokens: values.length,
        mean,
        min: Math.min(...values),
        max: Math.max(...values),
      };
    })
    .sort((a, b) => b.nTokens - a.nTokens);
}

function findDeviations(tokens: TraceToken[]): DeviationEvent[] {
  const events: DeviationEvent[] = [];

  tokens.forEach((tok, i) => {
    if (tok.projection <= DEEP_DRIFT_THRESHOLD) {
      events.push({
        kind: "deep_drift",
        tokenIndex: tok.i ?? i,
        token: tok.token,
        projection: tok.projection,
        label: `Deep drift (${tok.projection.toFixed(1)})`,
      });
    }
    if (i > 0) {
      const delta = tok.projection - tokens[i - 1].projection;
      if (delta <= -SHARP_DROP_THRESHOLD) {
        events.push({
          kind: "sharp_drop",
          tokenIndex: tok.i ?? i,
          token: tok.token,
          projection: tok.projection,
          delta,
          label: `Sharp drop ${delta.toFixed(1)} → ${tok.projection.toFixed(1)}`,
        });
      }
    }
  });

  const peak = tokens.reduce(
    (best, tok, i) =>
      !best || tok.projection > best.tok.projection
        ? { tok, i }
        : best,
    null as { tok: TraceToken; i: number } | null,
  );
  if (peak && peak.tok.projection > 0) {
    events.push({
      kind: "peak",
      tokenIndex: peak.tok.i ?? peak.i,
      token: peak.tok.token,
      projection: peak.tok.projection,
      label: `Peak assistant alignment (+${peak.tok.projection.toFixed(1)})`,
    });
  }

  // Keep the most interesting events, deduped by token index (prefer sharp_drop)
  const byIndex = new Map<number, DeviationEvent>();
  const rank = { sharp_drop: 0, deep_drift: 1, peak: 2 };
  for (const ev of events) {
    const prev = byIndex.get(ev.tokenIndex);
    if (!prev || rank[ev.kind] < rank[prev.kind]) byIndex.set(ev.tokenIndex, ev);
  }

  return [...byIndex.values()]
    .sort((a, b) => a.projection - b.projection)
    .slice(0, 8);
}

function buildHeadline(
  stats: TurnActivationInsights["stats"],
  deepPct: number,
  deviations: DeviationEvent[],
  segments: SegmentActivation[],
): string {
  const parts: string[] = [];

  if (stats.mean >= -5) {
    parts.push("Stays close to the assistant axis on average.");
  } else if (stats.mean >= -18) {
    parts.push("Mild drift — mean projection below neutral but not extreme.");
  } else {
    parts.push("Sustained drift off the assistant axis across this turn.");
  }

  if (deepPct >= 15) {
    parts.push(
      `${deepPct.toFixed(0)}% of tokens fell below ${DEEP_DRIFT_THRESHOLD} (major deviation).`,
    );
  }

  const thinking = segments.find((s) => s.kind === "thinking");
  const content = segments.find((s) => s.kind === "content");
  if (thinking && content && thinking.nTokens > 8 && content.nTokens > 8) {
    const gap = content.mean - thinking.mean;
    if (Math.abs(gap) >= 5) {
      parts.push(
        gap > 0
          ? `Visible content is ${gap.toFixed(1)} pts more assistant-aligned than thinking blocks.`
          : `Thinking blocks run ${Math.abs(gap).toFixed(1)} pts higher on the axis than visible content.`,
      );
    }
  }

  const worst = deviations.find((d) => d.kind === "deep_drift");
  if (worst) {
    parts.push(
      `Lowest point: ${JSON.stringify(worst.token)} at ${worst.projection.toFixed(1)}.`,
    );
  }

  return parts.join(" ");
}

export function analyzeTurnActivations(
  turn: TurnDetail,
  turnLabel?: string,
): TurnActivationInsights {
  const tokens = turn.tokens ?? [];
  const projections = tokens.map((t) => t.projection);
  const segments = turn.parsed?.response?.segments ?? [];

  if (!projections.length) {
    return {
      reqId: turn.req_id,
      turnLabel: turnLabel ?? turn.req_id,
      stats: {
        min: 0,
        max: 0,
        mean: 0,
        median: 0,
        std: 0,
        range: 0,
        nTokens: 0,
      },
      belowZero: { count: 0, pct: 0 },
      deepDrift: { count: 0, pct: 0 },
      cappedCount: 0,
      segments: [],
      deviations: [],
      headline: "No token projections recorded for this turn.",
      severity: "low",
    };
  }

  const mean = projections.reduce((a, b) => a + b, 0) / projections.length;
  const min = Math.min(...projections);
  const max = Math.max(...projections);
  const belowZero = projections.filter((p) => p < 0).length;
  const deepCount = projections.filter((p) => p <= DEEP_DRIFT_THRESHOLD).length;
  const deepPct = (deepCount / projections.length) * 100;
  const segStats = segmentActivations(tokens, segments);
  const deviations = findDeviations(tokens);

  const stats = {
    min,
    max,
    mean,
    median: median(projections),
    std: stdDev(projections, mean),
    range: max - min,
    nTokens: projections.length,
  };

  return {
    reqId: turn.req_id,
    turnLabel: turnLabel ?? turn.req_id,
    stats,
    belowZero: {
      count: belowZero,
      pct: (belowZero / projections.length) * 100,
    },
    deepDrift: { count: deepCount, pct: deepPct },
    cappedCount: tokens.filter((t) => t.capped).length,
    segments: segStats,
    deviations,
    headline: buildHeadline(stats, deepPct, deviations, segStats),
    severity: severityFromMean(mean, deepPct),
  };
}

export function analyzeSessionActivations(
  turns: TurnDetail[],
): SessionActivationInsights {
  const perTurn = turns.map((t, i) =>
    analyzeTurnActivations(t, `Turn ${i + 1}`),
  );

  if (!perTurn.length) {
    return {
      turns: [],
      combined: analyzeTurnActivations({
        req_id: "",
        session_id: "",
        created_at: 0,
        finished_at: 0,
        duration_s: 0,
        model: "",
        clamp: false,
        thinking: false,
        error: null,
        request: { messages: [] },
        parsed: {
          system_prompt: "",
          conversation: [],
          input_tool_calls: [],
          tool_results: [],
          response: { segments: [], tool_markers: [], text: "" },
        },
        response_text: "",
        stats: { min: 0, max: 0, mean: 0, n_tokens: 0 },
        tokens: [],
      }),
      turnMeanTrend: [],
      headline: "No turns to analyze.",
      severity: "low",
    };
  }

  const allTokens = turns.flatMap((t) => t.tokens ?? []);
  const combined = analyzeTurnActivations(
    {
      ...turns[0],
      req_id: turns.map((t) => t.req_id).join("+"),
      tokens: allTokens,
      stats: {
        min: Math.min(...allTokens.map((t) => t.projection)),
        max: Math.max(...allTokens.map((t) => t.projection)),
        mean:
          allTokens.reduce((s, t) => s + t.projection, 0) /
          Math.max(allTokens.length, 1),
        n_tokens: allTokens.length,
      },
    },
    "Session total",
  );

  const turnMeanTrend = perTurn.map((t) => t.stats.mean);
  let trendNote = "";
  if (turnMeanTrend.length >= 2) {
    const delta = turnMeanTrend[turnMeanTrend.length - 1] - turnMeanTrend[0];
    if (delta <= -8) {
      trendNote = ` Drift worsened by ${Math.abs(delta).toFixed(1)} pts from first to last turn.`;
    } else if (delta >= 8) {
      trendNote = ` Assistant alignment improved by ${delta.toFixed(1)} pts over the session.`;
    }
  }

  const worstTurn = perTurn.reduce((a, b) =>
    a.stats.mean < b.stats.mean ? a : b,
  );
  const headline = `${combined.stats.nTokens} tokens across ${perTurn.length} turn(s). Session mean ${combined.stats.mean.toFixed(1)} (range ${combined.stats.min.toFixed(1)} to ${combined.stats.max.toFixed(1)}). Lowest turn: ${worstTurn.turnLabel} (${worstTurn.stats.mean.toFixed(1)}).${trendNote}`;

  return {
    turns: perTurn,
    combined,
    turnMeanTrend,
    headline,
    severity: combined.severity,
  };
}

export function driftBadge(mean: number): string {
  if (mean >= -10) return "aligned";
  if (mean >= -22) return "mild drift";
  return "major drift";
}
