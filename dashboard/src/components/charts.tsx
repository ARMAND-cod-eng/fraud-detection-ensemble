// Small, dependency-free SVG charts. Conventions: one y-axis per chart, 2px lines, bars <= 24px
// with 4px rounded data-ends, hairline solid gridlines, crosshair/per-mark tooltips, legends for
// >= 2 series, and text in ink tokens (never the series colour).
import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => setW(Math.max(200, entries[0].contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function niceTicks(min: number, max: number, count = 4): number[] {
  if (max <= min) return [min];
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (count / span) * step0;
  const step = step0 * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const defaultFmt = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : `${+v.toFixed(2)}`);

interface Tip { x: number; y: number; content: ReactNode }

function Tooltip({ tip, width }: { tip: Tip | null; width: number }) {
  if (!tip) return null;
  const left = tip.x > width - 180 ? tip.x - 170 : tip.x + 14;
  return <div className="tooltip" style={{ left, top: Math.max(0, tip.y - 10) }} role="status">{tip.content}</div>;
}

export interface Series { key: string; label: string; color: string; kind?: "line" | "area" | "bar" }

export function Legend({ series, box }: { series: { label: string; color: string }[]; box?: boolean }) {
  if (series.length < 2) return null;
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.label}><i className={`key${box ? " box" : ""}`} style={{ background: s.color }} />{s.label}</span>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------------------ time series
export function TimeSeriesChart<T extends { t: number }>({
  data, series, height = 180, yFormat = defaultFmt, xFormat, label,
}: {
  data: T[]; series: Series[]; height?: number; yFormat?: (v: number) => string;
  xFormat: (t: number) => string; label: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { l: 40, r: 10, t: 8, b: 22 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const values = data.flatMap((d) => series.map((s) => Number((d as Record<string, unknown>)[s.key] ?? 0)));
  const yMax = Math.max(1, ...values);
  const ticks = niceTicks(0, yMax * 1.1, 3);
  const top = ticks[ticks.length - 1] || 1;
  const x = (i: number) => m.l + (data.length <= 1 ? iw / 2 : (i / (data.length - 1)) * iw);
  const y = (v: number) => m.t + ih - (v / top) * ih;
  const barW = Math.min(24, Math.max(2, (data.length ? iw / data.length : iw) - 2));

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const px = e.clientX - rect.left - m.l;
    const i = Math.round((px / iw) * (data.length - 1));
    setHover(Math.max(0, Math.min(data.length - 1, i)));
  };
  const xTickIdx = data.length ? [0, Math.floor((data.length - 1) / 2), data.length - 1] : [];
  const tip: Tip | null = hover == null || !data[hover] ? null : {
    x: x(hover), y: m.t,
    content: (
      <>
        <div className="t">{xFormat(data[hover].t)}</div>
        {series.map((s) => (
          <div className="r" key={s.key}>
            <span className="k"><i className="key" style={{ background: s.color }} />{s.label}</span>
            <b>{yFormat(Number((data[hover] as Record<string, unknown>)[s.key] ?? 0))}</b>
          </div>
        ))}
      </>
    ),
  };

  return (
    <div className="chart" ref={ref}>
      <Legend series={series} box={series.some((s) => s.kind === "bar")} />
      <svg width={width} height={height} role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} />
            <text x={m.l - 6} y={y(t) + 3.5} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        <line className="axisline" x1={m.l} x2={width - m.r} y1={y(0)} y2={y(0)} />
        {series.map((s) => {
          const pts = data.map((d, i) => [x(i), y(Number((d as Record<string, unknown>)[s.key] ?? 0))]);
          if (s.kind === "bar") {
            return (
              <g key={s.key}>
                {pts.map(([px, py], i) => {
                  const h = y(0) - py;
                  if (h <= 0) return null;
                  const r = Math.min(4, h, barW / 2);
                  const x0 = px - barW / 2;
                  return (
                    <path key={i} fill={s.color} opacity={hover == null || hover === i ? 1 : 0.55}
                      d={`M${x0},${y(0)} V${py + r} Q${x0},${py} ${x0 + r},${py} H${x0 + barW - r} Q${x0 + barW},${py} ${x0 + barW},${py + r} V${y(0)} Z`} />
                  );
                })}
              </g>
            );
          }
          const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
          return (
            <g key={s.key}>
              {s.kind === "area" && pts.length > 1 && (
                <path d={`${line} L${pts[pts.length - 1][0]},${y(0)} L${pts[0][0]},${y(0)} Z`} fill={s.color} opacity={0.1} />
              )}
              <path d={line} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            </g>
          );
        })}
        {hover != null && data[hover] && (
          <g>
            <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={m.t} y2={m.t + ih} />
            {series.filter((s) => s.kind !== "bar").map((s) => (
              <circle key={s.key} cx={x(hover)} cy={y(Number((data[hover] as Record<string, unknown>)[s.key] ?? 0))} r={4}
                fill={s.color} stroke="var(--surface)" strokeWidth={2} />
            ))}
          </g>
        )}
        {xTickIdx.map((i) => data[i] && (
          <text key={i} x={x(i)} y={height - 6} textAnchor={i === 0 ? "start" : i === data.length - 1 ? "end" : "middle"}>
            {xFormat(data[i].t)}
          </text>
        ))}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

// ------------------------------------------------------------------------------ categorical bars
export function BarChart({
  data, color = "var(--s1)", height = 180, valueFormat = defaultFmt, label, xLabel,
}: {
  data: { label: string; value: number }[]; color?: string; height?: number;
  valueFormat?: (v: number) => string; label: string; xLabel?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { l: 40, r: 8, t: 8, b: xLabel ? 36 : 22 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const max = Math.max(1, ...data.map((d) => d.value));
  const ticks = niceTicks(0, max * 1.1, 3);
  const top = ticks[ticks.length - 1] || 1;
  const band = iw / Math.max(1, data.length);
  const bw = Math.min(24, band * 0.6);
  const y = (v: number) => m.t + ih - (v / top) * ih;
  const tip: Tip | null = hover == null ? null : {
    x: m.l + band * hover + band / 2, y: y(data[hover].value) - 40,
    content: <><div className="t">{data[hover].label}</div><div className="r"><b>{valueFormat(data[hover].value)}</b></div></>,
  };
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} />
            <text x={m.l - 6} y={y(t) + 3.5} textAnchor="end">{valueFormat(t)}</text>
          </g>
        ))}
        {data.map((d, i) => {
          const cx = m.l + band * i + band / 2;
          const h = y(0) - y(d.value);
          const r = Math.min(4, h, bw / 2);
          const x0 = cx - bw / 2;
          const yy = y(d.value);
          return (
            <g key={d.label} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={m.l + band * i} y={m.t} width={band} height={ih} fill="transparent" />
              {h > 0 && (
                <path fill={color} opacity={hover == null || hover === i ? 1 : 0.6}
                  d={`M${x0},${y(0)} V${yy + r} Q${x0},${yy} ${x0 + r},${yy} H${x0 + bw - r} Q${x0 + bw},${yy} ${x0 + bw},${yy + r} V${y(0)} Z`} />
              )}
              <text x={cx} y={height - (xLabel ? 20 : 6)} textAnchor="middle">{d.label}</text>
            </g>
          );
        })}
        <line className="axisline" x1={m.l} x2={width - m.r} y1={y(0)} y2={y(0)} />
        {xLabel && <text x={m.l + iw / 2} y={height - 3} textAnchor="middle">{xLabel}</text>}
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

// ------------------------------------------------------------------------------ curves (PR / ROC / cost)
export interface Curve { key: string; label: string; color: string; points: [number, number][]; width?: number }

export function CurveChart({
  curves, height = 260, xLabel, yLabel, xLog = false, xDomain, yDomain = [0, 1], label,
  xFormat = (v) => `${+v.toFixed(3)}`, yFormat = (v) => `${+v.toFixed(2)}`, markers = [],
}: {
  curves: Curve[]; height?: number; xLabel: string; yLabel: string; xLog?: boolean;
  xDomain?: [number, number]; yDomain?: [number, number]; label: string;
  xFormat?: (v: number) => string; yFormat?: (v: number) => string;
  markers?: { x: number; label: string }[];
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hx, setHx] = useState<number | null>(null);
  const m = { l: 46, r: 12, t: 8, b: 36 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const allX = curves.flatMap((c) => c.points.map((p) => p[0]));
  const [x0, x1] = xDomain ?? [xLog ? Math.max(1e-4, Math.min(...allX.filter((v) => v > 0))) : 0, Math.max(...allX, 1e-9)];
  const tx = (v: number) => (xLog ? (Math.log10(Math.max(v, x0)) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) : (v - x0) / (x1 - x0));
  const X = (v: number) => m.l + Math.max(0, Math.min(1, tx(v))) * iw;
  const Y = (v: number) => m.t + ih - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * ih;
  const xTicks = xLog
    ? Array.from({ length: Math.round(Math.log10(x1) - Math.log10(x0)) + 1 }, (_, i) => x0 * 10 ** i)
    : niceTicks(x0, x1, 4);
  const yTicks = niceTicks(yDomain[0], yDomain[1], 4);
  const paths = useMemo(() => curves.map((c) => ({
    ...c,
    d: c.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(Math.max(yDomain[0], Math.min(yDomain[1], p[1]))).toFixed(1)}`).join(" "),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  })), [curves, width, height, x0, x1]);

  const valueAt = (c: Curve, xv: number): number | null => {
    const pts = c.points;
    if (!pts.length) return null;
    let best = pts[0];
    for (const p of pts) if (Math.abs(tx(p[0]) - tx(xv)) < Math.abs(tx(best[0]) - tx(xv))) best = p;
    return best[1];
  };
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - rect.left - m.l) / iw));
    setHx(xLog ? 10 ** (Math.log10(x0) + f * (Math.log10(x1) - Math.log10(x0))) : x0 + f * (x1 - x0));
  };
  const tip: Tip | null = hx == null ? null : {
    x: X(hx), y: m.t,
    content: (
      <>
        <div className="t">{xLabel}: {xFormat(hx)}</div>
        {curves.map((c) => {
          const v = valueAt(c, hx);
          return (
            <div className="r" key={c.key}>
              <span className="k"><i className="key" style={{ background: c.color }} />{c.label}</span>
              <b>{v == null ? "—" : yFormat(v)}</b>
            </div>
          );
        })}
      </>
    ),
  };
  return (
    <div className="chart" ref={ref}>
      <Legend series={curves} />
      <svg width={width} height={height} role="img" aria-label={label}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="gridline" x1={m.l} x2={width - m.r} y1={Y(t)} y2={Y(t)} />
            <text x={m.l - 6} y={Y(t) + 3.5} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={X(t)} y={m.t + ih + 14} textAnchor="middle">{xFormat(t)}</text>
        ))}
        <line className="axisline" x1={m.l} x2={width - m.r} y1={m.t + ih} y2={m.t + ih} />
        {markers.map((mk) => (
          <g key={mk.label}>
            <line x1={X(mk.x)} x2={X(mk.x)} y1={m.t} y2={m.t + ih} stroke="var(--ink-2)" strokeWidth={1} />
            <text x={X(mk.x) + 4} y={m.t + 10} style={{ fill: "var(--ink-2)" }}>{mk.label}</text>
          </g>
        ))}
        {paths.map((c) => (
          <path key={c.key} d={c.d} fill="none" stroke={c.color} strokeWidth={c.width ?? 2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {hx != null && <line className="crosshair" x1={X(hx)} x2={X(hx)} y1={m.t} y2={m.t + ih} />}
        <text x={m.l + iw / 2} y={height - 4} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(11 ${m.t + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHx(null)} />
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

// ------------------------------------------------------------------------------ horizontal bars
export function HBarChart({
  rows, diverging = false, valueFormat = (v) => v.toFixed(3), label, color = "var(--s1)", rowHeight = 26, labelWidth = 120,
}: {
  rows: { label: string; value: number; detail?: string }[]; diverging?: boolean; label: string;
  valueFormat?: (v: number) => string; color?: string; rowHeight?: number; labelWidth?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { l: labelWidth, r: 64, t: 4, b: 4 };
  const iw = width - m.l - m.r;
  const height = m.t + m.b + rows.length * rowHeight;
  const maxAbs = Math.max(1e-9, ...rows.map((r) => Math.abs(r.value)));
  const zero = diverging ? m.l + iw / 2 : m.l;
  const scale = diverging ? iw / 2 / maxAbs : iw / maxAbs;
  const bh = Math.min(16, rowHeight - 8);
  const tip: Tip | null = hover == null ? null : {
    x: zero, y: m.t + hover * rowHeight - 46,
    content: <><div className="t">{rows[hover].label}</div><div className="r"><b>{valueFormat(rows[hover].value)}</b></div>
      {rows[hover].detail && <div className="t" style={{ marginTop: 4, marginBottom: 0 }}>{rows[hover].detail}</div>}</>,
  };
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={label}>
        {rows.map((r, i) => {
          const yc = m.t + i * rowHeight + rowHeight / 2;
          const len = Math.abs(r.value) * scale;
          const neg = r.value < 0;
          const fill = diverging ? (neg ? "var(--div-neg)" : "var(--div-pos)") : color;
          const x0 = neg ? zero - len : zero;
          const rad = Math.min(4, len / 2, bh / 2);
          // rounded data-end only (the end away from the baseline)
          const d = neg
            ? `M${zero},${yc - bh / 2} H${x0 + rad} Q${x0},${yc - bh / 2} ${x0},${yc - bh / 2 + rad} V${yc + bh / 2 - rad} Q${x0},${yc + bh / 2} ${x0 + rad},${yc + bh / 2} H${zero} Z`
            : `M${zero},${yc - bh / 2} H${zero + len - rad} Q${zero + len},${yc - bh / 2} ${zero + len},${yc - bh / 2 + rad} V${yc + bh / 2 - rad} Q${zero + len},${yc + bh / 2} ${zero + len - rad},${yc + bh / 2} H${zero} Z`;
          return (
            <g key={r.label + i} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={0} y={yc - rowHeight / 2} width={width} height={rowHeight} fill="transparent" />
              <text x={m.l - 8} y={yc + 4} textAnchor="end" style={{ fill: "var(--ink-2)" }}>{r.label}</text>
              {len > 0.5 && <path d={d} fill={fill} opacity={hover == null || hover === i ? 1 : 0.6} />}
              <text x={neg ? zero + 6 : zero + len + 6} y={yc + 4} textAnchor="start" style={{ fill: "var(--ink-2)" }}>
                {valueFormat(r.value)}
              </text>
            </g>
          );
        })}
        <line className="axisline" x1={zero} x2={zero} y1={m.t} y2={height - m.b} />
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

// ------------------------------------------------------------------------------ confusion matrix
export function ConfusionMatrix({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
  const total = tp + fp + fn + tn;
  const cell = (v: number, label: string) => {
    const share = total ? Math.log10(1 + v) / Math.log10(1 + total) : 0;
    return (
      <div style={{
        padding: "14px 12px", borderRadius: 8,
        background: `color-mix(in srgb, var(--s1) ${Math.round(8 + share * 40)}%, var(--surface))`,
      }}>
        <div style={{ fontSize: 22, fontWeight: 640, fontVariantNumeric: "tabular-nums" }}>{v.toLocaleString()}</div>
        <div className="small ink2">{label}</div>
      </div>
    );
  };
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "90px 1fr 1fr", gap: 6, alignItems: "center" }}>
        <span />
        <span className="small muted">Predicted fraud</span>
        <span className="small muted">Predicted legit</span>
        <span className="small muted">Actual fraud</span>
        {cell(tp, "True positives")}
        {cell(fn, "Missed fraud")}
        <span className="small muted">Actual legit</span>
        {cell(fp, "False alarms")}
        {cell(tn, "True negatives")}
      </div>
    </div>
  );
}
