/** Formatting helpers. Every function renders null/undefined/NaN as an em dash ("—"). */

export const DASH = "—";

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function fmtNumber(v: unknown, maxFractionDigits = 0): string {
  if (!isNum(v)) return DASH;
  return v.toLocaleString("en-US", { maximumFractionDigits: maxFractionDigits });
}

/** Fixed decimals, e.g. scores 0.8123 → "0.81". */
export function fmtDecimal(v: unknown, digits = 2): string {
  if (!isNum(v)) return DASH;
  return v.toFixed(digits);
}

/** Fraction (0..1) → percentage string. 0.953 → "95.3%". */
export function fmtPercent(v: unknown, digits = 1): string {
  if (!isNum(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtBytes(v: unknown): string {
  if (!isNum(v) || v < 0) return DASH;
  if (v < 1024) return `${v} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let n = v / 1024;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 100 ? 0 : 1)} ${units[i]}`;
}

export function fmtMs(v: unknown): string {
  if (!isNum(v)) return DASH;
  if (v < 1000) return `${Math.round(v)} ms`;
  if (v < 60_000) return `${(v / 1000).toFixed(2)} s`;
  const m = Math.floor(v / 60_000);
  const s = Math.round((v % 60_000) / 1000);
  return `${m}m ${s}s`;
}

export function fmtUsd(v: unknown, digits = 4): string {
  if (!isNum(v)) return DASH;
  return `$${v.toFixed(digits)}`;
}

export function fmtDate(v: unknown): string {
  if (typeof v !== "string" || !v) return DASH;
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toLocaleString();
}

export function fmtText(v: unknown): string {
  if (v === null || v === undefined || v === "") return DASH;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return isNum(v) ? fmtNumber(v, 4) : DASH;
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

export function truncateHash(hash: unknown, n = 12): string {
  if (typeof hash !== "string" || !hash) return DASH;
  return hash.length > n ? `${hash.slice(0, n)}…` : hash;
}

/** "FINANCIAL_RELEVANT" → "Financial relevant", "payment_date" → "Payment date". */
export function humanize(v: unknown): string {
  if (typeof v !== "string" || !v) return DASH;
  const s = v.replace(/[_-]+/g, " ").trim().toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Safe nested lookup for loosely-typed objects: get(obj, "a", "b", "c"). */
export function get(obj: unknown, ...path: (string | number)[]): unknown {
  let cur: unknown = obj;
  for (const key of path) {
    if (cur === null || cur === undefined || typeof cur !== "object") return undefined;
    cur = (cur as Record<string | number, unknown>)[key];
  }
  return cur;
}

export function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

export function asArray(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}
