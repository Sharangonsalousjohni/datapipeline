"""Page layout helpers for native PDFs: page furniture, visual lines and text-aligned tables.

- :func:`find_page_furniture` flags watermarks (rotated, transparent or oversized spans) and running
  headers/footers (text repeated in the same position on every page) so they never reach headings,
  chunks or table cells.
- :func:`page_lines` groups the remaining spans into visual lines split into column segments.
- :func:`table_candidate_ok` rejects ``find_tables()`` regions that swallow prose or numbered headings,
  hold paragraph-sized cells, or collapse several visual rows into one.
- :func:`detect_aligned_tables` rebuilds tables from column-aligned lines: header row = first line of
  non-numeric labels, one data row per visual line, columns mapped by overlap with the header.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
BOLD_FLAG = 16
MAX_CELL_WORDS = 25

_PAGE_NO_RE = re.compile(r"\bpage\s+\d+(\s*(of|/)\s*\d+)?\b", re.I)
_UNDERSCORE_RE = re.compile(r"^[_\s.:]*_{4,}[_\s]*$|:\s*_{4,}")


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def furniture_key(text: str) -> str:
    """Text identity that ignores page numbers ("Page 1" == "Page 2")."""
    return re.sub(r"\d+", "#", _norm(text))


def _is_horizontal(line: dict) -> bool:
    dx, dy = line.get("dir", (1.0, 0.0))
    return dx > 0.99 and abs(dy) < 0.02


@dataclass
class Furniture:
    span_ids: list[set[tuple]]  # per page: ids of spans to drop
    texts: set[str] = field(default_factory=set)  # furniture_key of every dropped span
    boxes: list[list[list[float]]] = field(default_factory=list)  # per page: bboxes of dropped spans

    def is_furniture_text(self, text: str) -> bool:
        # Exact match only: a real title can be a substring of a running footer.
        return bool(text.strip()) and furniture_key(text) in self.texts


def span_id(span: dict) -> tuple:
    b = span.get("bbox", (0, 0, 0, 0))
    return (round(b[0], 1), round(b[1], 1), span.get("text", ""))


def find_page_furniture(pages: list[dict]) -> Furniture:
    """``pages`` are ``page.get_text("dict")`` results with the page height under ``"_height"``."""
    entries: list[list[tuple[tuple, str, tuple, dict, dict]]] = []
    sizes: list[float] = []
    for d in pages:
        page_entries = []
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for s in line.get("spans", []):
                    if not (s.get("text") or "").strip():
                        continue
                    if _is_horizontal(line) and s.get("alpha", 255) >= 128:
                        sizes.append(float(s.get("size", 10.0)))
                    pos = (round(s["bbox"][1] / 6), round(s["bbox"][0] / 12))
                    page_entries.append((span_id(s), furniture_key(s["text"]), pos, s, line))
        entries.append(page_entries)

    body_size = median(sizes) if sizes else 10.0
    n_pages = len(pages)
    seen_on: dict[tuple[str, tuple], set[int]] = defaultdict(set)
    for p, page_entries in enumerate(entries):
        for _, key, pos, _, _ in page_entries:
            seen_on[(key, pos)].add(p)

    flagged: list[set[tuple]] = [set() for _ in pages]
    boxes: list[list[list[float]]] = [[] for _ in pages]
    texts: set[str] = set()
    for p, page_entries in enumerate(entries):
        height = float(pages[p].get("_height") or 792.0)
        for sid, key, pos, s, line in page_entries:
            y0, y1 = s["bbox"][1], s["bbox"][3]
            in_margin = y1 < height * 0.08 or y0 > height * 0.92
            repeats = len(seen_on[(key, pos)])
            watermark = (
                not _is_horizontal(line)
                or s.get("alpha", 255) < 128
                or float(s.get("size", body_size)) >= 3.0 * body_size
            )
            # Body text repeated in place counts only with 3+ pages (two pages can coincide by chance);
            # margin text (running headers/footers) counts from 2 pages.
            running = (n_pages >= 3 and repeats == n_pages) or (
                n_pages >= 2 and in_margin and repeats >= max(2, round(0.6 * n_pages))
            )
            page_number = in_margin and bool(_PAGE_NO_RE.search(s["text"]))
            if watermark or running or page_number:
                flagged[p].add(sid)
                boxes[p].append(list(s["bbox"]))
                texts.add(key)
    return Furniture(span_ids=flagged, texts=texts, boxes=boxes)


# --------------------------------------------------------------------------- visual lines
@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    bold: bool
    size: float

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def bbox(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]


@dataclass
class VisualLine:
    segments: list[Segment]

    @property
    def y0(self) -> float:
        return min(s.y0 for s in self.segments)

    @property
    def y1(self) -> float:
        return max(s.y1 for s in self.segments)

    @property
    def text(self) -> str:
        return "  ".join(s.text for s in self.segments)

    @property
    def is_numbered_heading(self) -> bool:
        return len(self.segments) == 1 and bool(NUMBERED_HEADING_RE.match(self.segments[0].text))

    @property
    def is_prose(self) -> bool:
        return len(self.segments) == 1 and self.segments[0].words > 10


def page_lines(page_dict: dict, drop: set[tuple]) -> list[VisualLine]:
    spans = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            if not _is_horizontal(line):
                continue
            for s in line.get("spans", []):
                if (s.get("text") or "").strip() and span_id(s) not in drop:
                    spans.append(s)
    spans.sort(key=lambda s: ((s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][0]))

    rows: list[list[dict]] = []
    for s in spans:
        cy = (s["bbox"][1] + s["bbox"][3]) / 2
        if rows:
            last = rows[-1]
            last_cy = sum((t["bbox"][1] + t["bbox"][3]) / 2 for t in last) / len(last)
            if abs(cy - last_cy) <= 0.45 * max(float(s.get("size", 10)), 1.0):
                last.append(s)
                continue
        rows.append([s])

    lines: list[VisualLine] = []
    for row in rows:
        row.sort(key=lambda s: s["bbox"][0])
        segs: list[Segment] = []
        for s in row:
            size = float(s.get("size", 10.0))
            text = s["text"]
            if segs and s["bbox"][0] - segs[-1].x1 <= 1.2 * size:
                prev = segs[-1]
                gap = s["bbox"][0] - prev.x1
                prev.text = (prev.text + (" " if gap > 0.15 * size and not prev.text.endswith(" ") else "") + text)
                prev.x1, prev.y0, prev.y1 = s["bbox"][2], min(prev.y0, s["bbox"][1]), max(prev.y1, s["bbox"][3])
                prev.bold = prev.bold and bool(s.get("flags", 0) & BOLD_FLAG)
                continue
            segs.append(Segment(s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3], text, bool(s.get("flags", 0) & BOLD_FLAG), size))
        for seg in segs:
            seg.text = " ".join(seg.text.split())
        segs = [seg for seg in segs if seg.text]
        if segs:
            lines.append(VisualLine(segs))
    return lines


def _inside(line: VisualLine, bbox: list[float]) -> bool:
    cy = (line.y0 + line.y1) / 2
    cx = (min(s.x0 for s in line.segments) + max(s.x1 for s in line.segments)) / 2
    return bbox[0] - 1 <= cx <= bbox[2] + 1 and bbox[1] - 1 <= cy <= bbox[3] + 1


# --------------------------------------------------------------------------- table checks
def table_candidate_ok(rows: list[list[str]], bbox: list[float], row_bboxes: list[list[float]], lines: list[VisualLine]) -> bool:
    """Reject regions that are not one clean table (see module docstring)."""
    if any(len(c.split()) > MAX_CELL_WORDS for r in rows for c in r):
        return False
    inside = [ln for ln in lines if _inside(ln, bbox)]
    if any(ln.is_numbered_heading or ln.is_prose for ln in inside):
        return False
    for rb in row_bboxes:
        multi = [ln for ln in inside if _inside(ln, rb) and len(ln.segments) >= 2]
        if len(multi) > 1:  # several aligned visual rows collapsed into one table row
            return False
    return True


def _is_label(text: str) -> bool:
    return not re.search(r"\d", text) and not re.search(r"[$€£¥₹%]", text)


def _assign_columns(segments: list[Segment], bounds: list[tuple[float, float]]) -> list[int] | None:
    cols: list[int] = []
    for s in segments:
        overlaps = [max(0.0, min(s.x1, hi) - max(s.x0, lo)) for lo, hi in bounds]
        best = max(range(len(bounds)), key=lambda i: overlaps[i])
        if overlaps[best] <= 0:
            return None
        cols.append(best)
    return cols if len(set(cols)) == len(cols) else None


@dataclass
class AlignedTable:
    headers: list[str]
    rows: list[list[str]]
    cell_bboxes: list[list[list[float] | None]]
    bbox: list[float]
    lines: list[VisualLine]


def detect_aligned_tables(lines: list[VisualLine], page_width: float) -> list[AlignedTable]:
    """Find tables made of column-aligned visual lines (tables drawn with rules only, or none)."""
    runs: list[list[VisualLine]] = []
    cur: list[VisualLine] = []
    for ln in lines:
        tabular = (
            len(ln.segments) >= 2
            and all(s.words <= MAX_CELL_WORDS for s in ln.segments)
            and not NUMBERED_HEADING_RE.match(ln.segments[0].text)
        )
        if tabular and cur and ln.y0 - cur[-1].y1 > 2.5 * (cur[-1].y1 - cur[-1].y0):
            runs.append(cur)
            cur = []
        if tabular:
            cur.append(ln)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    rule_lines = [ln for ln in lines if any(_UNDERSCORE_RE.search(s.text) for s in ln.segments)]
    tables: list[AlignedTable] = []
    for run in runs:
        height = max(ln.y1 - ln.y0 for ln in run)
        if any(_UNDERSCORE_RE.search(s.text) for ln in run for s in ln.segments) or any(
            0 <= r.y0 - run[-1].y1 <= 6 * height for r in rule_lines
        ):
            continue  # signature block (names above signature rules), not a table
        h = next((i for i, ln in enumerate(run) if all(_is_label(s.text) for s in ln.segments)), None)
        if h is None or h == len(run) - 1:
            continue
        header, body = run[h], run[h + 1:]
        anchors = [s.x0 for s in header.segments]
        lead = min(s.x0 for ln in body for s in ln.segments)
        headers = [s.text for s in header.segments]
        if lead < anchors[0] - 6:  # unlabelled row-label column left of the first header
            anchors.insert(0, lead)
            headers.insert(0, "")
        bounds = [(a, anchors[i + 1] if i + 1 < len(anchors) else page_width) for i, a in enumerate(anchors)]

        rows: list[list[str]] = []
        boxes: list[list[list[float] | None]] = []
        header_boxes: list[list[float] | None] = [None] * len(headers)
        hcols = _assign_columns(header.segments, bounds)
        if hcols is None:
            continue
        for c, s in zip(hcols, header.segments):
            header_boxes[c] = s.bbox
        ok = True
        for ln in body:
            cols = _assign_columns(ln.segments, bounds)
            if cols is None:
                ok = False
                break
            row = [""] * len(headers)
            row_b: list[list[float] | None] = [None] * len(headers)
            for c, s in zip(cols, ln.segments):
                row[c], row_b[c] = s.text, s.bbox
            rows.append(row)
            boxes.append(row_b)
        if not ok or not any(re.search(r"\d", c) for r in rows for c in r):
            continue
        used = run[h:]
        bbox = [
            round(min(s.x0 for ln in used for s in ln.segments), 1),
            round(min(ln.y0 for ln in used), 1),
            round(max(s.x1 for ln in used for s in ln.segments), 1),
            round(max(ln.y1 for ln in used), 1),
        ]
        tables.append(AlignedTable(headers, rows, [header_boxes] + boxes, bbox, used))
    return tables


def is_furniture_span(span: dict, line: dict, drop_boxes: list[list[float]], body_size: float) -> bool:
    """For spans from a clipped extraction, whose ids differ from the full-page ones: judged by
    appearance, or by lying inside a span already flagged on this page (never by text alone)."""
    if not _is_horizontal(line) or span.get("alpha", 255) < 128 or float(span.get("size", body_size)) >= 3.0 * body_size:
        return True
    b = span.get("bbox", (0, 0, 0, 0))
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(d[0] <= cx <= d[2] and d[1] <= cy <= d[3] for d in drop_boxes)


def clip_text(page, rect, drop_boxes: list[list[float]], body_size: float) -> str:
    """Text inside ``rect`` without page furniture (``page.get_text(clip=...)`` would keep watermarks)."""
    parts: list[str] = []
    for b in page.get_text("dict", clip=rect).get("blocks", []):
        for line in b.get("lines", []):
            text = "".join(
                s.get("text", "") for s in line.get("spans", []) if not is_furniture_span(s, line, drop_boxes, body_size)
            )
            if text.strip():
                parts.append(text)
    return " ".join(" ".join(parts).split())


def line_inside(line: VisualLine, bbox: list[float]) -> bool:
    return _inside(line, bbox)
