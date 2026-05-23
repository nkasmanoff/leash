import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

type Props = {
  xs: number[];
  ys: number[];
};

export default function ProjectionChart({ xs, ys }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth || 400;

    if (!plotRef.current) {
      plotRef.current = new uPlot(
        {
          width,
          height: 320,
          series: [
            {},
            {
              label: "projection",
              stroke: "#7744cc",
              width: 2,
              points: { show: false },
            },
          ],
          axes: [
            {
              stroke: "#8b95a8",
              grid: { stroke: "#2a3142" },
              label: "token #",
            },
            {
              stroke: "#8b95a8",
              grid: { stroke: "#2a3142" },
              label: "Assistant-axis projection",
            },
          ],
          scales: {
            x: { time: false },
          },
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
  }, []);

  useEffect(() => {
    if (!plotRef.current || !containerRef.current) return;
    const width = containerRef.current.clientWidth;
    plotRef.current.setSize({ width, height: 320 });
    plotRef.current.setData([xs, ys]);
  }, [xs, ys]);

  useEffect(() => {
    const onResize = () => {
      if (!plotRef.current || !containerRef.current) return;
      plotRef.current.setSize({
        width: containerRef.current.clientWidth,
        height: 320,
      });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return <div ref={containerRef} className="chart-wrap" />;
}
