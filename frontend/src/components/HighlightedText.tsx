import type { SignalEvidence } from "../api/types";
import { humanize } from "../utils/format";
import { parsePipeTable, segmentText, type EvidenceKind } from "../utils/highlight";

interface HighlightedTextProps {
  text: string;
  chunkType?: string;
  signals?: SignalEvidence[] | null;
  negativeSignals?: SignalEvidence[] | null;
}

function buildTooltip(evidence: (SignalEvidence & { kind: EvidenceKind })[]): string {
  return evidence
    .map((ev) => {
      const sign = ev.kind === "negative" ? "Negative signal" : "Signal";
      const cat = ev.category ? ` (${humanize(ev.category)})` : "";
      const w = typeof ev.weight === "number" ? ` [weight ${ev.weight.toFixed(2)}]` : "";
      return `${sign}: ${humanize(ev.signal)}${cat}${w}`;
    })
    .join(" | ");
}

/**
 * Renders extracted chunk text safely as React elements (never uses dangerouslySetInnerHTML).
 * Positive & negative signal spans are wrapped in <mark> with explanatory tooltips.
 * If chunkType === "table" and lines are pipe-delimited, also renders a structured HTML table.
 */
export function HighlightedText({
  text,
  chunkType,
  signals = [],
  negativeSignals = [],
}: HighlightedTextProps) {
  const segments = segmentText(text, signals, negativeSignals);
  const tableRows = chunkType === "table" ? parsePipeTable(text) : null;

  return (
    <div className="highlighted-chunk">
      {tableRows && tableRows.length > 0 && (
        <div className="table-scroll" style={{ marginBottom: "0.75rem" }}>
          <table className="data-table data-table-compact" aria-label="Extracted table preview">
            <thead>
              <tr>
                {tableRows[0].map((cell, ci) => (
                  <th key={ci}>{cell}</th>
                ))}
              </tr>
            </thead>
            {tableRows.length > 1 && (
              <tbody>
                {tableRows.slice(1).map((row, ri) => (
                  <tr key={ri}>
                    {row.map((cell, ci) => (
                      <td key={ci}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            )}
          </table>
        </div>
      )}

      <pre className="chunk-text" data-testid="highlighted-text">
        {segments.length === 0 ? (
          <span className="muted">—</span>
        ) : (
          segments.map((seg, idx) => {
            if (seg.evidence.length === 0) {
              return <span key={idx}>{seg.text}</span>;
            }
            const hasPositive = seg.evidence.some((e) => e.kind === "positive");
            const hasNegative = seg.evidence.some((e) => e.kind === "negative");
            const cls =
              hasPositive && hasNegative
                ? "mark-signal mark-mixed"
                : hasNegative
                  ? "mark-signal mark-negative"
                  : "mark-signal mark-positive";
            const primary = seg.evidence[0];
            return (
              <mark key={idx} className={cls} title={buildTooltip(seg.evidence)} data-signal={primary.signal}>
                {seg.text}
                <span className="mark-tag">{primary.signal}</span>
              </mark>
            );
          })
        )}
      </pre>
    </div>
  );
}
