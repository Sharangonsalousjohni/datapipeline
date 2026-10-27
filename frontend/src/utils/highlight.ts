import type { SignalEvidence } from "../api/types";

export type EvidenceKind = "positive" | "negative";

export interface TextSegment {
  text: string;
  start: number;
  end: number;
  /** Evidence covering this segment (empty for plain text). */
  evidence: (SignalEvidence & { kind: EvidenceKind })[];
}

/**
 * Split `text` into contiguous segments at evidence boundaries.
 * - Offsets are clamped into [0, text.length]; invalid / empty / non-numeric spans are dropped.
 * - Overlapping spans are handled by splitting at every boundary; each segment lists all covering evidence.
 * - Concatenating segment texts always reproduces the original text exactly.
 */
export function segmentText(
  text: string,
  positive: SignalEvidence[] | null | undefined,
  negative: SignalEvidence[] | null | undefined = [],
): TextSegment[] {
  const src = typeof text === "string" ? text : "";
  const len = src.length;
  const spans: { ev: SignalEvidence & { kind: EvidenceKind }; s: number; e: number }[] = [];

  const add = (list: SignalEvidence[] | null | undefined, kind: EvidenceKind) => {
    if (!Array.isArray(list)) return;
    for (const ev of list) {
      if (!ev || typeof ev.start !== "number" || typeof ev.end !== "number") continue;
      if (!Number.isFinite(ev.start) || !Number.isFinite(ev.end)) continue;
      const s = Math.max(0, Math.min(len, Math.floor(ev.start)));
      const e = Math.max(0, Math.min(len, Math.floor(ev.end)));
      if (e <= s) continue;
      spans.push({ ev: { ...ev, kind }, s, e });
    }
  };
  add(positive, "positive");
  add(negative, "negative");

  if (len === 0) return [];
  if (spans.length === 0) return [{ text: src, start: 0, end: len, evidence: [] }];

  const bounds = new Set<number>([0, len]);
  for (const sp of spans) {
    bounds.add(sp.s);
    bounds.add(sp.e);
  }
  const points = Array.from(bounds).sort((a, b) => a - b);
  const segments: TextSegment[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (b <= a) continue;
    const covering = spans.filter((sp) => sp.s <= a && sp.e >= b).map((sp) => sp.ev);
    const prev = segments[segments.length - 1];
    // merge adjacent plain segments
    if (prev && covering.length === 0 && prev.evidence.length === 0) {
      prev.text += src.slice(a, b);
      prev.end = b;
    } else {
      segments.push({ text: src.slice(a, b), start: a, end: b, evidence: covering });
    }
  }
  return segments;
}

/** Parse " | "-separated table text into rows; returns null if it doesn't look like a table. */
export function parsePipeTable(text: string): string[][] | null {
  if (typeof text !== "string") return null;
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) return null;
  const pipeLines = lines.filter((l) => l.includes(" | "));
  if (pipeLines.length < Math.ceil(lines.length * 0.6)) return null;
  const rows = lines
    .filter((l) => !/^\s*\|?\s*:?-{2,}/.test(l)) // drop markdown separator rows
    .map((l) =>
      l
        .replace(/^\s*\|\s?/, "")
        .replace(/\s?\|\s*$/, "")
        .split(" | ")
        .map((c) => c.trim()),
    );
  return rows.length >= 1 ? rows : null;
}
