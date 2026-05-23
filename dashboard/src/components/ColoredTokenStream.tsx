import { memo, useMemo } from "react";
import { projectionColor, stats } from "../lib/projectionColor";

export type StreamToken = {
  token: string;
  projection: number;
  index: number;
};

const THINKING_OPEN = "<think>";
const THINKING_CLOSE = "</think>";
const IM_END = "<|im_end|>";

type ParsedSpan = {
  key: number;
  text: string;
  projection: number;
  kind: "thinking" | "content" | "placeholder";
  index: number;
};

function isMarkerToken(raw: string): boolean {
  return (
    raw.includes(THINKING_OPEN) ||
    raw.includes(THINKING_CLOSE) ||
    raw.includes(IM_END)
  );
}

function parseTokens(tokens: StreamToken[]): ParsedSpan[] {
  const spans: ParsedSpan[] = [];
  let inThinking = false;
  let thinkingHadContent = false;
  let thinkingPlaceholderIndex = -1;

  for (const t of tokens) {
    const raw = t.token;

    if (raw.includes(THINKING_OPEN)) {
      inThinking = true;
      thinkingHadContent = false;
      thinkingPlaceholderIndex = t.index;
      const rest = raw.replace(THINKING_OPEN, "");
      if (rest.trim()) {
        thinkingHadContent = true;
        spans.push({
          key: t.index,
          text: rest,
          projection: t.projection,
          kind: "thinking",
          index: t.index,
        });
      }
      continue;
    }

    if (raw.includes(THINKING_CLOSE)) {
      inThinking = false;
      const rest = raw.replace(THINKING_CLOSE, "");
      if (rest.trim()) {
        thinkingHadContent = true;
        spans.push({
          key: t.index,
          text: rest,
          projection: t.projection,
          kind: "thinking",
          index: t.index,
        });
      }
      if (!thinkingHadContent && thinkingPlaceholderIndex >= 0) {
        spans.push({
          key: thinkingPlaceholderIndex,
          text: "∅ reasoning skipped",
          projection: tokens[thinkingPlaceholderIndex]?.projection ?? 0,
          kind: "placeholder",
          index: thinkingPlaceholderIndex,
        });
      }
      thinkingPlaceholderIndex = -1;
      continue;
    }

    if (raw.includes(IM_END)) continue;

    if (inThinking) {
      if (raw.trim()) thinkingHadContent = true;
      spans.push({
        key: t.index,
        text: raw,
        projection: t.projection,
        kind: "thinking",
        index: t.index,
      });
      continue;
    }

    spans.push({
      key: t.index,
      text: raw,
      projection: t.projection,
      kind: "content",
      index: t.index,
    });
  }

  return spans;
}

type Props = {
  tokens: StreamToken[];
  highlightIndex?: number | null;
  onHoverIndex?: (index: number | null) => void;
  compact?: boolean;
};

function ColoredTokenStreamInner({
  tokens,
  highlightIndex = null,
  onHoverIndex,
  compact = false,
}: Props) {
  const spans = useMemo(() => parseTokens(tokens), [tokens]);
  const projs = useMemo(
    () =>
      tokens
        .filter((t) => !isMarkerToken(t.token))
        .map((t) => t.projection),
    [tokens],
  );
  const { min, max } = useMemo(() => stats(projs), [projs]);

  if (!spans.length) {
    return <span className="tok-empty">{compact ? "…" : ""}</span>;
  }

  return (
    <span className="tok-stream">
      {spans.map((span) => {
        const { color, bg } = projectionColor(span.projection, min, max);
        const highlighted = highlightIndex === span.index;
        const isPlaceholder = span.kind === "placeholder";
        const isThinking = span.kind === "thinking";

        if (isPlaceholder) {
          return (
            <span
              key={span.key}
              className={`tok tok-think-empty ${highlighted ? "tok-hl" : ""}`}
              title={`#${span.index} · proj ${span.projection.toFixed(2)}`}
              onMouseEnter={() => onHoverIndex?.(span.index)}
              onMouseLeave={() => onHoverIndex?.(null)}
            >
              {span.text}
            </span>
          );
        }

        return (
          <span
            key={span.key}
            className={`tok ${isThinking ? "tok-think" : "tok-content"} ${highlighted ? "tok-hl" : ""}`}
            style={{ color, backgroundColor: bg }}
            title={`#${span.index} · proj ${span.projection.toFixed(2)}${isThinking ? " · thinking" : ""}`}
            onMouseEnter={() => onHoverIndex?.(span.index)}
            onMouseLeave={() => onHoverIndex?.(null)}
          >
            {span.text}
          </span>
        );
      })}
    </span>
  );
}

export default memo(ColoredTokenStreamInner);
