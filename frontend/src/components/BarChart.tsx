import { fmtDecimal, fmtNumber, humanize, isNum } from "../utils/format";

export interface BarDatum {
  label: string;
  value: number;
  tone?: "relevant" | "uncertain" | "irrelevant" | "mixed" | "default";
}

interface BarChartProps {
  items: BarDatum[];
  formatValue?: "count" | "score" | "percent";
  maxValue?: number;
  emptyText?: string;
}

const TONE_COLOR: Record<string, string> = {
  relevant: "var(--c-relevant, #16a34a)",
  uncertain: "var(--c-uncertain, #d97706)",
  irrelevant: "var(--c-irrelevant, #94a3b8)",
  mixed: "var(--c-mixed, #0284c7)",
  default: "var(--c-accent, #2563eb)",
};

export function BarChart({ items, formatValue = "count", maxValue, emptyText = "No data" }: BarChartProps) {
  const valid = items.filter((d) => isNum(d.value));
  if (valid.length === 0) {
    return <div className="muted-sm">{emptyText}</div>;
  }
  const computedMax = maxValue ?? Math.max(...valid.map((d) => d.value), 1e-6);

  const fmt = (v: number) => {
    if (formatValue === "score") return fmtDecimal(v, 2);
    if (formatValue === "percent") return `${(v * 100).toFixed(1)}%`;
    return fmtNumber(v);
  };

  return (
    <div className="bar-chart-container">
      {valid.map((d, idx) => {
        const ratio = computedMax > 0 ? Math.max(0, Math.min(1, d.value / computedMax)) : 0;
        const fill = TONE_COLOR[d.tone ?? "default"] ?? TONE_COLOR.default;
        const label = humanize(d.label);
        const percentWidth = Math.max(ratio > 0 ? 3 : 0, ratio * 100);

        return (
          <div key={d.label + idx} className="bar-row">
            <span className="bar-label" title={label}>
              {label}
            </span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${percentWidth}%`,
                  backgroundColor: fill,
                }}
              />
            </div>
            <span className="bar-value mono">{fmt(d.value)}</span>
          </div>
        );
      })}
    </div>
  );
}
