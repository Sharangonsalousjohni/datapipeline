import type { ReactNode } from "react";
import { Link } from "react-router-dom";

interface StatCardProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "relevant" | "uncertain" | "irrelevant" | "failed" | "mixed" | "running";
  to?: string;
}

export function StatCard({ label, value, hint, tone = "default", to }: StatCardProps) {
  const body = (
    <>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint !== undefined && <div className="stat-hint">{hint}</div>}
    </>
  );
  return to ? (
    <Link to={to} className={`stat-card stat-${tone} stat-link`}>
      {body}
    </Link>
  ) : (
    <div className={`stat-card stat-${tone}`}>{body}</div>
  );
}
