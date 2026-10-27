"""Shared helpers for grid-shaped sources (XLSX, XLS, CSV).

A sheet is split into table regions separated by fully empty rows. Within a
region the header row is detected heuristically; rows above it become the
table caption (e.g. report titles) and are kept with the table.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from app.extraction.base import TABLE, ContentBlock, TableData

UNSAFE_FORMULA_RE = re.compile(r"(?i)(\bcmd\s*\||\bDDE\b|WEBSERVICE\s*\(|HYPERLINK\s*\(|\bIMPORTXML\b|\bIMPORTDATA\b|\bCALL\s*\(|\bEXEC\b|\[\d+\]|https?://)")


def col_letter(idx: int) -> str:
    """1-based column index -> Excel column letters."""
    letters = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


@dataclass
class GridCell:
    value: str  # display value ("" when empty)
    raw: object = None
    formula: str | None = None
    number_format: str | None = None

    @property
    def is_text(self) -> bool:
        return isinstance(self.raw, str) and not _looks_numeric(self.raw)


@dataclass
class GridRow:
    number: int  # 1-based
    cells: list[GridCell] = field(default_factory=list)  # index 0 => column 1

    @property
    def non_empty(self) -> list[tuple[int, GridCell]]:
        return [(i + 1, c) for i, c in enumerate(self.cells) if c.value.strip()]


_NUM_RE = re.compile(r"^[\s\(\-+]*[$€£¥₹]?\s*[\d.,]+\s*%?\)?\s*$")


def _looks_numeric(s: str) -> bool:
    return bool(_NUM_RE.match(s)) and any(ch.isdigit() for ch in s)


def format_value(value: object, number_format: str | None = None) -> str:
    """Render a cell value for display, honouring common number formats."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (dt.datetime, dt.date)):
        if isinstance(value, dt.datetime) and value.time() == dt.time(0, 0):
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, (int, float)):
        fmt = (number_format or "General").lower()
        symbol = ""
        for sym in ("$", "€", "£", "¥", "₹"):
            if sym in fmt:
                symbol = sym
                break
        if "%" in fmt:
            decimals = len(fmt.split(".")[1].split("%")[0]) if "." in fmt else 0
            return f"{value * 100:.{decimals}f}%"
        if "0.00" in fmt or "#,##0.0" in fmt:
            text = f"{value:,.2f}" if "," in fmt else f"{value:.2f}"
        elif "#,##0" in fmt:
            text = f"{value:,.0f}"
        elif isinstance(value, float):
            text = f"{value:.10g}"
        else:
            text = str(value)
        if symbol:
            text = f"-{symbol}{text[1:]}" if text.startswith("-") else f"{symbol}{text}"
        return text
    return str(value).strip()


def _is_header_candidate(row: GridRow) -> bool:
    cells = row.non_empty
    if len(cells) < 2:
        return False
    texts = sum(1 for _, c in cells if c.is_text)
    return texts / len(cells) >= 0.6


def split_regions(rows: list[GridRow]) -> list[list[GridRow]]:
    regions: list[list[GridRow]] = []
    current: list[GridRow] = []
    last_number: int | None = None
    for row in rows:
        empty = not row.non_empty
        gap = last_number is not None and row.number - last_number > 1
        if empty or gap:
            if current:
                regions.append(current)
                current = []
            if empty:
                continue
        current.append(row)
        last_number = row.number
    if current:
        regions.append(current)
    return regions


def regions_to_blocks(
    rows: list[GridRow],
    *,
    sheet: str | None,
    method: str,
    start_table_index: int = 0,
) -> list[ContentBlock]:
    """Convert a sheet's rows into table blocks (one per region)."""
    blocks: list[ContentBlock] = []
    regions = split_regions(rows)
    # Merge single-line title regions into the following region as caption.
    pending_caption: list[str] = []
    table_index = start_table_index
    for region in regions:
        if len(region) == 1 and len(region[0].non_empty) <= 2 and not _is_header_candidate(region[0]):
            pending_caption.append(" ".join(c.value for _, c in region[0].non_empty))
            continue
        header_pos = next((i for i, r in enumerate(region[:5]) if _is_header_candidate(r)), None)
        caption_rows = region[:header_pos] if header_pos else []
        caption = " ".join(pending_caption + [" ".join(c.value for _, c in r.non_empty) for r in caption_rows]).strip()
        pending_caption = []
        min_col = min(i for r in region for i, _ in r.non_empty)
        max_col = max(i for r in region for i, _ in r.non_empty)
        if header_pos is not None:
            header_row = region[header_pos]
            headers = [
                (header_row.cells[c - 1].value if c - 1 < len(header_row.cells) else "") or col_letter(c)
                for c in range(min_col, max_col + 1)
            ]
            data_rows = region[header_pos + 1 :]
            header_number = header_row.number
        else:
            headers = [col_letter(c) for c in range(min_col, max_col + 1)]
            data_rows = region
            header_number = None
        table_rows: list[list[str]] = []
        numbers: list[int] = []
        formulas: dict[str, str] = {}
        formats: dict[str, str] = {}
        for r in data_rows:
            vals = []
            for c in range(min_col, max_col + 1):
                cell = r.cells[c - 1] if c - 1 < len(r.cells) else GridCell("")
                vals.append(cell.value)
                ref = f"{col_letter(c)}{r.number}"
                if cell.formula:
                    formulas[ref] = cell.formula
                if cell.number_format and cell.number_format != "General" and cell.value:
                    formats[ref] = cell.number_format
            table_rows.append(vals)
            numbers.append(r.number)
        # Treat trailing single-cell text rows as footnotes.
        footnotes: list[str] = []
        while table_rows and sum(1 for v in table_rows[-1] if v.strip()) == 1 and len(table_rows) > 1:
            only = next(v for v in table_rows[-1] if v.strip())
            if _looks_numeric(only):
                break
            footnotes.insert(0, only)
            table_rows.pop()
            numbers.pop()
        first_row = (caption_rows[0].number if caption_rows else (header_number or region[0].number))
        last_row = region[-1].number
        cell_range = f"{col_letter(min_col)}{first_row}:{col_letter(max_col)}{last_row}"
        table = TableData(
            headers=headers,
            rows=table_rows,
            row_numbers=numbers,
            header_row_number=header_number,
            col_offset=min_col,
            caption=caption or None,
            footnotes=footnotes,
            formulas=formulas,
            number_formats=formats,
        )
        location = {
            "sheet": sheet,
            "table_index": table_index,
            "row_start": first_row,
            "row_end": last_row,
            "col_start": min_col,
            "col_end": max_col,
            "cell_range": f"{sheet}!{cell_range}" if sheet else cell_range,
        }
        blocks.append(
            ContentBlock(block_type=TABLE, text=render_table(table), location=location, extraction_method=method, table=table)
        )
        table_index += 1
    if pending_caption and blocks:
        blocks[-1].table.footnotes.extend(pending_caption)  # type: ignore[union-attr]
        blocks[-1].text = render_table(blocks[-1].table)  # type: ignore[arg-type]
    return blocks


def render_table(table: TableData, rows: list[list[str]] | None = None) -> str:
    lines: list[str] = []
    if table.caption:
        lines.append(table.caption)
    if table.unit_line and table.unit_line not in (table.caption or ""):
        lines.append(table.unit_line)
    lines.append(" | ".join(h.strip() for h in table.headers))
    for r in rows if rows is not None else table.rows:
        lines.append(" | ".join(v.strip() for v in r))
    if table.footnotes:
        lines.extend(table.footnotes)
    return "\n".join(lines)


def find_unsafe_formulas(formulas: dict[str, str]) -> list[str]:
    return [ref for ref, f in formulas.items() if UNSAFE_FORMULA_RE.search(f)]
