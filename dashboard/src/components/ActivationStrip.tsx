import { memo, useCallback, useEffect, useRef } from "react";
import { projectionColor, stats } from "../lib/projectionColor";

type Props = {
  xs: number[];
  ys: number[];
  turnBoundaries?: number[];
  highlightIndex?: number | null;
  onHoverIndex?: (index: number | null) => void;
};

const STRIP_H = 32;

function bucketIndex(pixel: number, n: number, width: number): number {
  if (n <= width) return pixel;
  return Math.min(n - 1, Math.floor((pixel * n) / width));
}

function ActivationStripInner({
  xs,
  ys,
  turnBoundaries = [],
  highlightIndex = null,
  onHoverIndex,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const dataRef = useRef({ xs, ys, turnBoundaries, highlightIndex });
  dataRef.current = { xs, ys, turnBoundaries, highlightIndex };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(wrap.clientWidth, 1);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(STRIP_H * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${STRIP_H}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { xs: xData, ys: yData, turnBoundaries: turns, highlightIndex: hi } =
      dataRef.current;
    const n = yData.length;

    ctx.fillStyle = "#0b0d12";
    ctx.fillRect(0, 0, width, STRIP_H);

    if (n === 0) {
      ctx.fillStyle = "#2a3142";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("No tokens yet", 8, 20);
      return;
    }

    const yMin = stats(yData).min;
    const yMax = stats(yData).max;
    const zeroY =
      yMax === yMin
        ? STRIP_H / 2
        : STRIP_H - ((0 - yMin) / (yMax - yMin)) * (STRIP_H - 4) - 2;

    for (let px = 0; px < width; px++) {
      const i = bucketIndex(px, n, width);
      const { color } = projectionColor(yData[i], yMin, yMax);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.75;
      ctx.fillRect(px, 0, 1, STRIP_H);
    }
    ctx.globalAlpha = 1;

    ctx.strokeStyle = "#f5c51866";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(width, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    for (const pos of turns) {
      if (pos <= 0 || pos >= n) continue;
      const px = n <= width ? pos : Math.floor((pos * width) / n);
      ctx.strokeStyle = "#3dd68caa";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px + 0.5, 0);
      ctx.lineTo(px + 0.5, STRIP_H);
      ctx.stroke();
    }

    if (hi != null) {
      const pos = xData.indexOf(hi);
      if (pos >= 0) {
        const px = n <= width ? pos : Math.floor((pos * width) / n);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px + 0.5, 0);
        ctx.lineTo(px + 0.5, STRIP_H);
        ctx.stroke();
      }
    }
  }, [ys]);

  useEffect(() => {
    draw();
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [draw, xs, ys, turnBoundaries, highlightIndex]);

  const onMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onHoverIndex || !wrapRef.current) return;
      const n = dataRef.current.ys.length;
      if (!n) {
        onHoverIndex(null);
        return;
      }
      const rect = e.currentTarget.getBoundingClientRect();
      const px = Math.max(
        0,
        Math.min(
          Math.floor(e.clientX - rect.left),
          Math.max(1, rect.width) - 1,
        ),
      );
      const width = Math.max(rect.width, 1);
      const i = bucketIndex(px, n, width);
      onHoverIndex(dataRef.current.xs[i] ?? i);
    },
    [onHoverIndex],
  );

  return (
    <div ref={wrapRef} className="activation-strip-wrap">
      <canvas
        ref={canvasRef}
        className="activation-strip"
        onMouseMove={onMove}
        onMouseLeave={() => onHoverIndex?.(null)}
      />
    </div>
  );
}

export default memo(ActivationStripInner);
