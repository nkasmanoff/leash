/** Map assistant-axis projection to a display color (low=red drift, high=green). */

export function projectionColor(
  value: number,
  min: number,
  max: number,
): { color: string; bg: string } {
  const span = Math.max(max - min, 1e-3);
  const t = Math.max(0, Math.min(1, (value - min) / span));
  // red (#ff5a5f) → yellow (#f5c518) → green (#3dd68c)
  let r: number;
  let g: number;
  let b: number;
  if (t < 0.5) {
    const u = t / 0.5;
    r = Math.round(255 * (1 - u) + 245 * u);
    g = Math.round(90 * (1 - u) + 197 * u);
    b = Math.round(95 * (1 - u) + 24 * u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(245 * (1 - u) + 61 * u);
    g = Math.round(197 * (1 - u) + 214 * u);
    b = Math.round(24 * (1 - u) + 140 * u);
  }
  const color = `rgb(${r}, ${g}, ${b})`;
  const bg = `rgba(${r}, ${g}, ${b}, 0.12)`;
  return { color, bg };
}

export function stats(values: number[]) {
  if (!values.length) return { min: -30, max: 0, mean: 0 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return { min, max, mean };
}
