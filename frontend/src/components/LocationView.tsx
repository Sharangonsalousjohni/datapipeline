import type { SourceLocation } from "../api/types";
import { DASH, isNum } from "../utils/format";

interface LocationViewProps {
  location: SourceLocation | null | undefined;
  extraLocations?: SourceLocation[];
  compact?: boolean;
}

export function formatLocationParts(loc: SourceLocation | null | undefined): string[] {
  if (!loc) return [];
  const parts: string[] = [];

  if (isNum(loc.page)) {
    parts.push(`Page ${loc.page}`);
  }
  if (loc.sheet) {
    parts.push(`Sheet "${loc.sheet}"`);
  }
  if (loc.cell_range) {
    parts.push(`Cells ${loc.cell_range}`);
  }
  if (isNum(loc.table_index)) {
    parts.push(`Table #${loc.table_index + 1}`);
  }
  if (isNum(loc.row_start)) {
    const end = isNum(loc.row_end) && loc.row_end !== loc.row_start ? `–${loc.row_end}` : "";
    parts.push(`Row${end ? "s" : ""} ${loc.row_start}${end}`);
  }
  if (isNum(loc.col_start)) {
    const end = isNum(loc.col_end) && loc.col_end !== loc.col_start ? `–${loc.col_end}` : "";
    parts.push(`Col${end ? "s" : ""} ${loc.col_start}${end}`);
  }
  if (isNum(loc.paragraph_start)) {
    const end =
      isNum(loc.paragraph_end) && loc.paragraph_end !== loc.paragraph_start ? `–${loc.paragraph_end}` : "";
    parts.push(`Paragraph${end ? "s" : ""} ${loc.paragraph_start}${end}`);
  }
  if (isNum(loc.line_start)) {
    const end = isNum(loc.line_end) && loc.line_end !== loc.line_start ? `–${loc.line_end}` : "";
    parts.push(`Line${end ? "s" : ""} ${loc.line_start}${end}`);
  }
  if (isNum(loc.char_start) && isNum(loc.char_end)) {
    parts.push(`Chars ${loc.char_start}–${loc.char_end}`);
  }
  if (Array.isArray(loc.bboxes) && loc.bboxes.length > 0) {
    const first = loc.bboxes[0];
    if (Array.isArray(first) && first.length === 4) {
      const coords = first.map((n) => (isNum(n) ? Math.round(n) : 0)).join(", ");
      const extra = loc.bboxes.length > 1 ? ` (+${loc.bboxes.length - 1} more)` : "";
      const pageDim =
        isNum(loc.page_width) && isNum(loc.page_height)
          ? ` on ${Math.round(loc.page_width)}×${Math.round(loc.page_height)}`
          : "";
      parts.push(`BBox [${coords}]${extra}${pageDim}`);
    }
  }
  return parts;
}

export function LocationView({ location, extraLocations = [], compact = false }: LocationViewProps) {
  const parts = formatLocationParts(location);
  const headings =
    location && Array.isArray(location.heading_path)
      ? location.heading_path.filter((h) => typeof h === "string" && h.trim() !== "")
      : [];

  if (parts.length === 0 && headings.length === 0) {
    return <span className="location-view muted">{DASH}</span>;
  }

  return (
    <div className={`location-view${compact ? " location-compact" : ""}`}>
      {headings.length > 0 && (
        <div className="location-headings" title="Section heading path">
          {headings.join(" › ")}
        </div>
      )}
      {parts.length > 0 && (
        <div className="location-pills">
          {parts.map((p, i) => (
            <span key={i} className="location-pill">
              {p}
            </span>
          ))}
          {extraLocations.length > 0 && (
            <span className="location-pill location-pill-muted">
              +{extraLocations.length} extra span{extraLocations.length > 1 ? "s" : ""}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
