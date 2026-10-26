interface ProgressBarProps {
  /** 0..1; null/undefined renders an indeterminate bar */
  value: number | null | undefined;
  label?: string;
  tone?: "default" | "success" | "error";
  showPercent?: boolean;
}

export function ProgressBar({ value, label, tone = "default", showPercent = true }: ProgressBarProps) {
  const indeterminate = typeof value !== "number" || !Number.isFinite(value);
  const pct = indeterminate ? 0 : Math.max(0, Math.min(1, value as number)) * 100;
  return (
    <div className="progress-wrap">
      {(label || showPercent) && (
        <div className="progress-meta">
          <span>{label}</span>
          {showPercent && !indeterminate && <span>{pct.toFixed(0)}%</span>}
        </div>
      )}
      <div
        className={`progress progress-${tone}${indeterminate ? " progress-indeterminate" : ""}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(pct)}
        aria-label={label}
      >
        <div className="progress-fill" style={indeterminate ? undefined : { width: `${pct}%` }} />
      </div>
    </div>
  );
}
