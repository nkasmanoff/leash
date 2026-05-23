import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export type TrajectoryPoint = {
  x: number;
  y: number;
  turnIndex: number;
  reqId: string;
};

export type TurnBoundary = {
  x: number;
  label: string;
};

type Props = {
  points: TrajectoryPoint[];
  boundaries: TurnBoundary[];
  height?: number;
  onHover?: (point: TrajectoryPoint | null) => void;
};

export default function SessionTrajectoryChart({
  points,
  boundaries,
  height = 280,
  onHover,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const dataRef = useRef({ points, boundaries });

  dataRef.current = { points, boundaries };

  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth || 400;

    if (!plotRef.current) {
      plotRef.current = new uPlot(
        {
          width,
          height,
          series: [
            {},
            {
              label: "projection",
              stroke: "#7744cc",
              width: 2,
              points: { show: true, size: 4 },
            },
          ],
          axes: [
            {
              stroke: "#8b95a8",
              grid: { stroke: "#2a3142" },
              label: "token # (session)",
            },
            {
              stroke: "#8b95a8",
              grid: { stroke: "#2a3142" },
              label: "Assistant-axis projection",
            },
          ],
          scales: { x: { time: false } },
          hooks: {
            draw: [
              (u) => {
                const ctx = u.ctx;
                const y0 = u.valToPos(0, "y", true);
                ctx.save();
                ctx.strokeStyle = "#f5c51855";
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(u.bbox.left, y0);
                ctx.lineTo(u.bbox.left + u.bbox.width, y0);
                ctx.stroke();
                ctx.restore();

                for (const b of dataRef.current.boundaries) {
                  const x = u.valToPos(b.x, "x", true);
                  ctx.save();
                  ctx.strokeStyle = "#3dd68c44";
                  ctx.lineWidth = 1;
                  ctx.setLineDash([2, 3]);
                  ctx.beginPath();
                  ctx.moveTo(x, u.bbox.top);
                  ctx.lineTo(x, u.bbox.top + u.bbox.height);
                  ctx.stroke();
                  ctx.restore();
                }
              },
            ],
            setCursor: [
              (u) => {
                if (!onHover) return;
                const idx = u.cursor.idx;
                if (idx == null || idx < 0) {
                  onHover(null);
                  return;
                }
                const pts = dataRef.current.points;
                onHover(pts[idx] ?? null);
              },
            ],
          },
        },
        [[], []],
        containerRef.current,
      );
    }

    return () => {
      plotRef.current?.destroy();
      plotRef.current = null;
    };
  }, [onHover]);

  useEffect(() => {
    if (!plotRef.current || !containerRef.current) return;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const width = containerRef.current.clientWidth;
    plotRef.current.setSize({ width, height });
    plotRef.current.setData([xs, ys]);
  }, [points, height]);

  useEffect(() => {
    const onResize = () => {
      if (!plotRef.current || !containerRef.current) return;
      plotRef.current.setSize({
        width: containerRef.current.clientWidth,
        height,
      });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [height]);

  return <div ref={containerRef} className="chart-wrap harness-chart" />;
}
