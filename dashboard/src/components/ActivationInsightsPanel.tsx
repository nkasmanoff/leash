import type {
  SessionActivationInsights,
  TurnActivationInsights,
} from "../lib/traceAnalysis";
import { DEEP_DRIFT_THRESHOLD } from "../lib/traceAnalysis";
import "./ActivationInsightsPanel.css";

function severityClass(severity: string): string {
  if (severity === "high") return "severity-high";
  if (severity === "medium") return "severity-medium";
  return "severity-low";
}

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="insight-stat">
      <span className="insight-stat-k">{label}</span>
      <strong className="insight-stat-v">{value}</strong>
      {hint && <span className="insight-stat-h">{hint}</span>}
    </div>
  );
}

function TurnInsightBody({ insight }: { insight: TurnActivationInsights }) {
  return (
    <>
      <p className="insight-headline">{insight.headline}</p>
      <div className="insight-stats-grid">
        <StatCard label="Mean" value={insight.stats.mean.toFixed(1)} hint="assistant axis" />
        <StatCard
          label="Range"
          value={`${insight.stats.min.toFixed(1)} → ${insight.stats.max.toFixed(1)}`}
        />
        <StatCard
          label="Below 0"
          value={`${insight.belowZero.pct.toFixed(0)}%`}
          hint={`${insight.belowZero.count} tok`}
        />
        <StatCard
          label={`Below ${DEEP_DRIFT_THRESHOLD}`}
          value={`${insight.deepDrift.pct.toFixed(0)}%`}
          hint={
            insight.deepDrift.count
              ? `${insight.deepDrift.count} major deviation(s)`
              : "none"
          }
        />
        <StatCard label="Std dev" value={insight.stats.std.toFixed(1)} />
        <StatCard
          label="Capped"
          value={String(insight.cappedCount)}
          hint={insight.cappedCount ? "kill-switch fired" : "none"}
        />
      </div>

      {insight.segments.length > 0 && (
        <>
          <div className="insight-subtitle">By segment</div>
          <div className="segment-stats">
            {insight.segments.map((seg) => (
              <div key={seg.kind} className={`segment-stat seg-${seg.kind}`}>
                <span className="segment-kind">{seg.kind}</span>
                <span>
                  mean <strong>{seg.mean.toFixed(1)}</strong>
                </span>
                <span>
                  {seg.min.toFixed(1)}–{seg.max.toFixed(1)}
                </span>
                <span>{seg.nTokens} tok</span>
              </div>
            ))}
          </div>
        </>
      )}

      {insight.deviations.length > 0 && (
        <>
          <div className="insight-subtitle">Notable deviations</div>
          <ul className="deviation-list">
            {insight.deviations.map((ev) => (
              <li key={`${ev.kind}-${ev.tokenIndex}`} className={`deviation-${ev.kind}`}>
                <span className="deviation-label">{ev.label}</span>
                <code>
                  #{ev.tokenIndex} {JSON.stringify(ev.token)}
                </code>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}

export function SessionActivationSummary({
  insights,
}: {
  insights: SessionActivationInsights;
}) {
  if (!insights.combined.stats.nTokens) return null;

  return (
    <section className={`activation-insights ${severityClass(insights.severity)}`}>
      <div className="panel-header">
        Activation insights
        <span className={`insight-severity ${severityClass(insights.severity)}`}>
          {insights.severity} drift
        </span>
      </div>
      <div className="insight-body">
        <p className="insight-headline">{insights.headline}</p>
        <div className="insight-stats-grid">
          <StatCard
            label="Session mean"
            value={insights.combined.stats.mean.toFixed(1)}
          />
          <StatCard
            label="Session min"
            value={insights.combined.stats.min.toFixed(1)}
            hint="worst drift"
          />
          <StatCard
            label="Session max"
            value={insights.combined.stats.max.toFixed(1)}
            hint="peak alignment"
          />
          <StatCard
            label="Major deviations"
            value={`${insights.combined.deepDrift.pct.toFixed(0)}%`}
            hint={`below ${DEEP_DRIFT_THRESHOLD}`}
          />
        </div>

        {insights.turns.length > 1 && (
          <>
            <div className="insight-subtitle">Mean projection by turn</div>
            <div className="turn-mean-bars">
              {insights.turns.map((t) => (
                <div key={t.reqId} className="turn-mean-row">
                  <span className="turn-mean-label">{t.turnLabel}</span>
                  <div className="turn-mean-track">
                    <div
                      className={`turn-mean-fill ${severityClass(t.severity)}`}
                      style={{
                        width: `${Math.min(100, Math.max(8, ((t.stats.mean + 50) / 60) * 100))}%`,
                      }}
                    />
                  </div>
                  <span className="turn-mean-val">{t.stats.mean.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

export function TurnActivationDetail({
  insight,
}: {
  insight: TurnActivationInsights | null;
}) {
  if (!insight || !insight.stats.nTokens) return null;

  return (
    <section className={`activation-insights turn-detail ${severityClass(insight.severity)}`}>
      <div className="panel-header">
        {insight.turnLabel} · axis analysis
        <span className={`insight-severity ${severityClass(insight.severity)}`}>
          {insight.severity}
        </span>
      </div>
      <div className="insight-body">
        <TurnInsightBody insight={insight} />
      </div>
    </section>
  );
}
