import { fmtDecimal, isNum } from "../utils/format";

export interface PRPoint {
  threshold?: number;
  precision: number;
  recall: number;
  f1?: number;
}

interface PRCurveProps {
  points: PRPoint[];
  chosenThreshold?: number | null;
  prAuc?: number | null;
}

export function PRCurve({ points, chosenThreshold, prAuc }: PRCurveProps) {
  const valid = points.filter(
    (p) => p && isNum(p.precision) && isNum(p.recall) && p.precision >= 0 && p.recall >= 0,
  );
  if (valid.length === 0) {
    return <div className="muted-sm">No precision-recall curve points available.</div>;
  }

  const sorted = [...valid].sort((a, b) => a.recall - b.recall);
  const width = 420;
  const height = 260;
  const padL = 46;
  const padR = 20;
  const padT = 18;
  const padB = 38;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const toX = (r: number) => padL + Math.max(0, Math.min(1, r)) * plotW;
  const toY = (p: number) => padT + (1 - Math.max(0, Math.min(1, p))) * plotH;

  const polyPoints = sorted.map((pt) => `${toX(pt.recall).toFixed(1)},${toY(pt.precision).toFixed(1)}`).join(" ");
  const areaPoints = `${toX(sorted[0].recall).toFixed(1)},${toY(0).toFixed(1)} ${polyPoints} ${toX(
    sorted[sorted.length - 1].recall,
  ).toFixed(1)},${toY(0).toFixed(1)}`;

  let chosenPt: PRPoint | null = null;
  if (isNum(chosenThreshold)) {
    let bestDiff = Infinity;
    for (const p of sorted) {
      if (isNum(p.threshold)) {
        const d = Math.abs(p.threshold - chosenThreshold);
        if (d < bestDiff) {
          bestDiff = d;
          chosenPt = p;
        }
      }
    }
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="pr-curve-wrap">
      <svg className="pr-curve-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Precision-Recall curve">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} y1={toY(t)} x2={padL + plotW} y2={toY(t)} className="svg-grid" />
            <text x={padL - 6} y={toY(t) + 4} textAnchor="end" className="svg-axis-tick">
              {t.toFixed(2)}
            </text>
            <line x1={toX(t)} y1={padT} x2={toX(t)} y2={padT + plotH} className="svg-grid" />
            <text x={toX(t)} y={padT + plotH + 15} textAnchor="middle" className="svg-axis-tick">
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        <polygon points={areaPoints} fill="var(--c-accent)" fillOpacity={0.12} />
        <polyline points={polyPoints} fill="none" stroke="var(--c-accent)" strokeWidth={2.2} />

        {sorted.map((pt, i) => (
          <circle
            key={i}
            cx={toX(pt.recall)}
            cy={toY(pt.precision)}
            r={2.5}
            fill="var(--c-accent)"
          >
            <title>
              {`Recall: ${fmtDecimal(pt.recall, 3)}, Precision: ${fmtDecimal(pt.precision, 3)}${
                isNum(pt.threshold) ? `, Threshold: ${fmtDecimal(pt.threshold, 2)}` : ""
              }`}
            </title>
          </circle>
        ))}

        {chosenPt && (
          <g>
            <circle
              cx={toX(chosenPt.recall)}
              cy={toY(chosenPt.precision)}
              r={5.5}
              fill="var(--c-relevant)"
              stroke="#fff"
              strokeWidth={1.5}
            />
          </g>
        )}

        <text x={padL + plotW / 2} y={height - 6} textAnchor="middle" className="svg-axis-label">
          Recall
        </text>
        <text
          x={14}
          y={padT + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${padT + plotH / 2})`}
          className="svg-axis-label"
        >
          Precision
        </text>
      </svg>
      <div className="pr-curve-legend">
        {isNum(prAuc) && <span>PR-AUC: <strong>{fmtDecimal(prAuc, 3)}</strong></span>}
        {chosenPt && (
          <span>
            Operating point (t={fmtDecimal(chosenPt.threshold, 2)}): P={fmtDecimal(chosenPt.precision, 2)}, R=
            {fmtDecimal(chosenPt.recall, 2)}
          </span>
        )}
      </div>
    </div>
  );
}
