import type { ReactNode } from "react";
import { humanize } from "../utils/format";

export type BadgeTone = "relevant" | "uncertain" | "irrelevant" | "mixed" | "failed" | "info" | "neutral" | "running";

const DECISION_TONE: Record<string, BadgeTone> = {
  RELEVANT: "relevant",
  UNCERTAIN: "uncertain",
  IRRELEVANT: "irrelevant",
  FINANCIAL_RELEVANT: "relevant",
  MIXED_CONTENT: "mixed",
  NEEDS_REVIEW: "uncertain",
  NOT_FINANCIAL: "irrelevant",
  PROCESSING_FAILED: "failed",
  positive: "relevant",
  negative: "irrelevant",
  succeeded: "relevant",
  completed: "relevant",
  failed: "failed",
  queued: "neutral",
  running: "running",
  processing: "running",
  ACCEPT: "relevant",
  ACCEPT_WITH_WARNINGS: "uncertain",
  HOLD_FOR_REVIEW: "mixed",
  REJECT: "failed",
  PASSED: "relevant",
  WARNING: "uncertain",
  FAILED: "failed",
};

export function toneFor(value: string | null | undefined): BadgeTone {
  if (!value) return "neutral";
  return DECISION_TONE[value] ?? DECISION_TONE[value.toLowerCase()] ?? DECISION_TONE[value.toUpperCase()] ?? "neutral";
}

const LABELS: Record<string, string> = {
  FINANCIAL_RELEVANT: "Financially relevant",
  MIXED_CONTENT: "Mixed content",
  NEEDS_REVIEW: "Needs review",
  NOT_FINANCIAL: "Not financial (filtered)",
  PROCESSING_FAILED: "Processing failed",
  RELEVANT: "Relevant",
  UNCERTAIN: "Uncertain",
  IRRELEVANT: "Irrelevant",
};

export function labelFor(value: string | null | undefined): string {
  if (!value) return "—";
  return LABELS[value] ?? humanize(value);
}

interface BadgeProps {
  value?: string | null;
  tone?: BadgeTone;
  children?: ReactNode;
  title?: string;
  size?: "sm" | "md" | "lg";
}

/** Status/decision pill. Pass `value` (decision/status code) or explicit `tone` + children. */
export function Badge({ value, tone, children, title, size = "md" }: BadgeProps) {
  const t = tone ?? toneFor(value);
  return (
    <span className={`badge badge-${t} badge-${size}`} title={title}>
      {children ?? labelFor(value)}
    </span>
  );
}
