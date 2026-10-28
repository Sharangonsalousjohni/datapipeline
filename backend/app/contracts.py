"""Contract / lease term extraction from clauses and tables.

Runs when a document is a lease (``statement_type == "lease"``). Every field records the exact text it
came from (``original_text`` plus ``char_start``/``char_end`` in the chunk) and a typed value:

- monetary / limit: decimal amount with currency; an explicit currency symbol and no scale
  declaration means whole currency units (``reported_scale="units"``).
- duration (months, days, years), percentage, area (sq ft), count, date (ISO), year, text.

Written numbers with a numeral in parentheses ("thirty-six (36) months", "five percent (5%)") are read
from the numeral. The rent schedule comes from the structured table attached to the chunk.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from app.schemas import ExtractedFinancialField, TableRowClassification
from app.taxonomy import FinancialCategory

C = FinancialCategory

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18, "twenty": 20, "twenty-four": 24, "thirty": 30,
    "thirty-six": 36, "forty-five": 45, "sixty": 60, "ninety": 90, "half": Decimal("0.5"),
}
DATE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}"
MONEY = r"(?:US)?\$\s?\d[\d,]*(?:\.\d{2})?"
NUM = r"\d[\d,]*(?:\.\d+)?"

_UNITS = "one|two|three|four|five|six|seven|eight|nine"
_ORDINAL_UNITS = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth"
_NUMBER_WORD = (
    rf"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-(?:{_UNITS}|{_ORDINAL_UNITS}))?"
    rf"|{_UNITS}|{_ORDINAL_UNITS}|zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen"
    r"|nineteen|tenth|eleventh|twelfth|twentieth|thirtieth|(?:one\s+)?hundred(?:\s+and)?(?:\s+\w+)?)"
)
# "thirty-six (36)" -> "36", "five percent (5%)" -> "5%", "first (1st)" -> "1st"; only real number words
_WORD_WITH_NUMERAL_RE = re.compile(
    rf"\b{_NUMBER_WORD}(?:\s+percent)?\s*\((\d+(?:\.\d+)?(?:%|st|nd|rd|th)?)\)", re.I
)


class _Text:
    """Whitespace-collapsed, numeral-normalized text with a map back to raw offsets."""

    def __init__(self, raw: str) -> None:
        chars: list[str] = []
        pos: list[int] = []
        prev_space = True
        for i, ch in enumerate(raw):
            if ch.isspace():
                if not prev_space:
                    chars.append(" ")
                    pos.append(i)
                prev_space = True
                continue
            chars.append(ch)
            pos.append(i)
            prev_space = False
        flat = "".join(chars)
        out: list[str] = []
        out_pos: list[tuple[int, int]] = []  # raw span each output char stands for
        last = 0
        for m in _WORD_WITH_NUMERAL_RE.finditer(flat):
            for j in range(last, m.start()):
                out.append(flat[j])
                out_pos.append((pos[j], pos[j] + 1))
            span = (pos[m.start()], pos[m.end() - 1] + 1)
            for ch in m.group(1):
                out.append(ch)
                out_pos.append(span)
            last = m.end()
        for j in range(last, len(flat)):
            out.append(flat[j])
            out_pos.append((pos[j], pos[j] + 1))
        self.raw = raw
        self.text = "".join(out)
        self._pos = out_pos

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        return self._pos[start][0], self._pos[end - 1][1]

    def raw_text(self, start: int, end: int) -> str:
        a, b = self.raw_span(start, end)
        return " ".join(self.raw[a:b].split())


def _dec(s: str) -> Decimal | None:
    try:
        return Decimal(re.sub(r"[^\d.\-]", "", s))
    except InvalidOperation:
        return None


def parse_date(s: str) -> str | None:
    m = re.match(r"([a-z]+)\.?\s+(\d{1,2}),\s+(\d{4})", s.strip(), re.I)
    if not m or m.group(1)[:3].lower() not in _MONTHS:
        return None
    try:
        return date(int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))).isoformat()
    except ValueError:
        return None


def _money(s: str) -> str:
    d = _dec(s)
    return f"{d:.2f}" if d is not None else s


# --------------------------------------------------------------------------- field construction
_CATEGORY = {
    "rent": C.RENT_LEASES_RECURRING_PAYMENTS, "escalation": C.RENT_LEASES_RECURRING_PAYMENTS,
    "deposit": C.DEPOSITS_AND_ESCROW, "late_fee": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS,
    "returned_payment": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS, "opex": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS,
    "insurance": C.INSURANCE_AND_CLAIMS, "renewal": C.MATURITY_RENEWALS_DEADLINES,
    "term": C.MATURITY_RENEWALS_DEADLINES, "commencement": C.MATURITY_RENEWALS_DEADLINES,
    "expiration": C.MATURITY_RENEWALS_DEADLINES, "cure": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS,
}


def _category(name: str) -> str | None:
    return next((c.value for key, c in _CATEGORY.items() if key in name), None)


def make_field(
    name: str,
    original: str,
    value: str,
    data_type: str,
    *,
    unit: str | None = None,
    currency: str | None = None,
    page: int | None = None,
    span: tuple[int, int] | None = None,
    source_kind: str = "clause",
    period: str | None = None,
    row_label: str | None = None,
    source_location: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> ExtractedFinancialField:
    money = data_type in ("monetary", "limit")
    details: dict[str, Any] = {"unit": unit or (currency if money else None), "data_type": data_type}
    if money:
        details["scale_source"] = "symbol: explicit currency symbol, no scale declaration"
    if source_kind == "table":
        details["metric_match"] = {"method": "structure", "score": 1.0}
    details.update(extra or {})
    return ExtractedFinancialField(
        field=name,
        original_text=original,
        normalized_value=value,
        reported_value=value if money else None,
        reported_scale="units" if money else None,
        scale_multiplier=1.0 if money else None,
        normalized_amount=value if money else None,
        currency=currency if money else None,
        data_type=data_type,
        reporting_period=period,
        row_label=row_label,
        category=_category(name),
        details=details,
        validation_notes=f"{name} = {value}{' ' + unit if unit and not money else ''} from '{original}'",
        source_location={"page": page, **(source_location or {})},
        statement_type="lease",
        source_kind=source_kind,
        char_start=span[0] if span else None,
        char_end=span[1] if span else None,
    )


# --------------------------------------------------------------------------- clause patterns
# (field, pattern, data_type, unit). The value is group 1; the whole match is the evidence span.
_CLAUSE_PATTERNS: list[tuple[str, str, str, str | None]] = [
    ("lease_number", r"\blease\s+(?:no\.?|number|#)\s*:?\s*([A-Z0-9][A-Z0-9-]{2,})", "text", None),
    ("effective_date", rf"\b(?:entered\s+into\s+(?:as\s+of\s+)?|effective\s+(?:as\s+of\s+)?|dated\s+(?:as\s+of\s+)?)({DATE})", "date", None),
    ("premises", r"\bknown\s+as\s+(.+?)\s*\(the\s+[\"“]?premises[\"”]?\)", "text", None),
    ("rentable_sqft", rf"({NUM})\s+(?:rentable\s+|usable\s+)?(?:square\s+feet|sq\.?\s*ft\.?)", "area", "sq_ft"),
    ("parking_spaces", r"(\d+)\s+(?:[a-z-]+\s+){0,2}parking\s+spaces?", "count", "spaces"),
    ("term_months", r"\bterm\s+of\s+(?:this\s+)?(?:lease|agreement)\s+(?:is|shall\s+be)\s+(\d+)\s+(months?|years?)", "duration", "months"),
    ("commencement_date", rf"\bcommenc\w*\s+(?:on\s+)?({DATE})", "date", None),
    ("expiration_date", rf"\b(?:ending|expir\w*|terminat\w*)\s+(?:on\s+)?({DATE})", "date", None),
    ("escalation_pct", r"\b(?:increases?|escalates?|escalation\s+of)\s+(?:by\s+)?(\d+(?:\.\d+)?)%", "percentage", "percent"),
    ("rent_due_day", r"\bon\s+the\s+(\d+)(?:st|nd|rd|th)?\s+day\s+of\s+each\s+(?:calendar\s+)?month", "count", "day_of_month"),
    ("late_fee_grace_days", r"\bwithin\s+(\d+)\s+days?\s+after\s+(?:its|the)\s+due\s+date", "duration", "days"),
    ("late_fee_pct", r"\blate\s+(?:charge|fee|penalty)\s+of\s+(\d+(?:\.\d+)?)%", "percentage", "percent"),
    ("returned_payment_fee", rf"\breturned\s+(?:payments?|checks?)\b[^.]{{0,60}}?({MONEY})", "monetary", None),
    ("security_deposit", rf"\b(?:security\s+)?deposit\s+(?:of\s+|in\s+the\s+amount\s+of\s+)?({MONEY})", "monetary", None),
    ("deposit_return_days", r"\breturned\s+within\s+(\d+)\s+days?", "duration", "days"),
    ("opex_share_pct", r"\b(?:pro[- ]rata|proportionate)\s+share\s*\(?\s*(\d+(?:\.\d+)?)%", "percentage", "percent"),
    ("opex_base_year", r"\b(?:over|above)\s+the\s+((?:19|20)\d\d)\s+base\s+year|\bbase\s+year\s+(?:of\s+)?((?:19|20)\d\d)", "year", None),
    ("opex_billing_frequency", r"\bbilled\s+(monthly|quarterly|semi-annually|annually|yearly)", "text", None),
    ("insurance_per_occurrence", rf"({MONEY})\s+per\s+occurrence", "limit", None),
    ("insurance_aggregate", rf"({MONEY})\s+(?:in\s+the\s+)?aggregate", "limit", None),
    ("renewal_term_months", r"\brenew\w*\b[^.]{0,80}?\bterm\s+of\s+(\d+)\s+(months?|years?)", "duration", "months"),
    ("renewal_notice_days", r"\bnotice\s+at\s+least\s+(\d+)\s+days|\b(\d+)\s+days[’']?\s+(?:prior\s+)?(?:written\s+)?notice", "duration", "days"),
    ("monetary_cure_days", r"\bfails?\s+to\s+pay\s+(?:rent|any\s+(?:amount|sum))\b[^.]{0,20}?\bwithin\s+(\d+)\s+days", "duration", "days"),
    ("nonmonetary_cure_days", r"\bfails?\s+to\s+cure\s+any\s+other\s+(?:breach|default)\s+within\s+(\d+)\s+days", "duration", "days"),
    ("governing_law", r"\bgoverned\s+by\s+the\s+laws?\s+of\s+(?:the\s+)?(.+?)(?:\.|,|;|$)", "text", None),
]
_COMPILED = [(n, re.compile(p, re.I), t, u) for n, p, t, u in _CLAUSE_PATTERNS]
_DEPOSIT_MULTIPLE_RE = re.compile(r"\bequal\s+to\s+([a-z]+(?:-[a-z]+)?|\d+(?:\.\d+)?)\s+months?(?:['’]s?)?\s+(?:of\s+)?", re.I)


def _value(name: str, data_type: str, m: re.Match[str], unit: str | None) -> tuple[str, str | None, str | None]:
    """Returns (normalized value, unit, currency)."""
    g = next(x for x in m.groups() if x is not None)
    if data_type in ("monetary", "limit"):
        return _money(g), None, "USD" if "$" in g else None
    if data_type == "date":
        return parse_date(g) or g, None, None
    if data_type == "duration":
        n = _dec(g) or Decimal(0)
        word = (m.group(2) or "") if (m.lastindex or 0) >= 2 and unit == "months" else ""
        if word.lower().startswith("year"):
            n *= 12
        return f"{n:f}", unit, None
    if data_type in ("percentage", "area", "count"):
        return f"{_dec(g):f}", unit, None
    if data_type == "text":
        v = " ".join(g.split()).strip(" ,")
        if name == "opex_billing_frequency":
            v = v.lower()
        return v, None, None
    return g, unit, None


def extract_clause_fields(raw: str, page: int | None) -> list[ExtractedFinancialField]:
    t = _Text(raw)
    fields: list[ExtractedFinancialField] = []
    for name, rx, dtype, unit in _COMPILED:
        m = rx.search(t.text)
        if not m:
            continue
        value, f_unit, currency = _value(name, dtype, m, unit)
        span = t.raw_span(m.start(), m.end())
        extra = None
        if name == "security_deposit":
            mm = _DEPOSIT_MULTIPLE_RE.search(t.text, m.end())
            if mm:
                word = mm.group(1).lower()
                mult = _WORD_NUMBERS.get(word) if not re.match(r"\d", word) else _dec(word)
                if mult is not None:
                    extra = {"stated_multiple_of_monthly_rent": str(mult), "multiple_text": t.raw_text(mm.start(), mm.end()).strip()}
        fields.append(
            make_field(name, t.raw_text(m.start(), m.end()), value, dtype, unit=f_unit, currency=currency,
                       page=page, span=span, extra=extra)
        )
    return fields


# --------------------------------------------------------------------------- tables
def parse_pipe_table(text: str) -> dict[str, Any] | None:
    lines = [ln for ln in text.splitlines() if "|" in ln]
    if len(lines) < 2:
        return None
    rows = [[c.strip() for c in ln.split("|")] for ln in lines]
    return {"headers": rows[0], "rows": rows[1:], "cell_bboxes": None, "table_bbox": None}


def _col(headers: list[str], *words: str) -> int | None:
    for i, h in enumerate(headers):
        if any(w in h.lower() for w in words):
            return i
    return None


def _cell_box(table: dict[str, Any], r: int, c: int) -> list[float] | None:
    boxes = table.get("cell_bboxes") or []
    try:
        return boxes[r + 1][c]
    except (IndexError, TypeError):
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def extract_table_terms(
    table: dict[str, Any], page: int | None
) -> tuple[list[TableRowClassification], list[ExtractedFinancialField]]:
    """Rent schedules (one row per lease year) and party tables (Landlord / Tenant columns)."""
    headers = [h or "" for h in table["headers"]]
    rows: list[TableRowClassification] = []
    fields: list[ExtractedFinancialField] = []
    loc_base = {"table_index": table.get("table_index"), "table_bbox": table.get("table_bbox")}

    monthly_c, annual_c = _col(headers, "monthly"), _col(headers, "annual")
    if monthly_c is not None or annual_c is not None:
        period_c = _col(headers, "period", "term", "dates")
        for r_i, row in enumerate(table["rows"]):
            label = row[0].strip() if row else ""
            year = re.match(r"(?:lease\s+)?year\s+(\d+)", label, re.I)
            is_total = bool(re.match(r"total\b", label, re.I))
            if not (year or is_total):
                continue
            suffix = f"year_{year.group(1)}" if year else "total"
            row_fields: list[ExtractedFinancialField] = []
            for c_i, kind in ((period_c, "period"), (monthly_c, "monthly"), (annual_c, "annual")):
                if c_i is None or c_i >= len(row) or not row[c_i].strip():
                    continue
                cell = row[c_i].strip()
                src = {**loc_base, "row_index": r_i + 1, "column_index": c_i, "row_label": label,
                       "column": headers[c_i], "bbox": _cell_box(table, r_i, c_i)}
                if is_total and kind == "annual":
                    name = "total_rent"
                else:
                    name = f"base_rent_{kind}_{suffix}"
                if kind == "period":
                    dur = re.fullmatch(r"(\d+)\s+(months?|years?)", cell, re.I)
                    if dur:  # "36 months" is a duration, never currency
                        n = int(dur.group(1)) * (12 if dur.group(2).lower().startswith("y") else 1)
                        f = make_field(name, cell, str(n), "duration", unit="months", page=page, source_kind="table",
                                       period=label, row_label=label, source_location=src)
                    else:
                        parts = [parse_date(p) for p in re.split(r"\s+[-–—]\s+|\s+to\s+", cell)]
                        value = " / ".join(p for p in parts if p) if all(parts) and len(parts) == 2 else cell
                        f = make_field(name, cell, value, "period", page=page, source_kind="table", period=label,
                                       row_label=label, source_location=src)
                elif re.search(r"[$€£]", cell) and _dec(cell) is not None:
                    f = make_field(name, cell, _money(cell), "monetary", currency="USD" if "$" in cell else None,
                                   page=page, source_kind="table", period=label, row_label=label,
                                   source_location=src, extra={"column": headers[c_i]})
                else:
                    continue
                row_fields.append(f)
            fields.extend(row_fields)
            rows.append(
                TableRowClassification(
                    row_index=r_i + 1, row_label=label, category=C.RENT_LEASES_RECURRING_PAYMENTS.value,
                    decision="RELEVANT", cells=row, periods=headers[1:], fields=row_fields,
                    statement_type="lease",
                )
            )
        return rows, fields

    landlord_c, tenant_c = _col(headers, "landlord", "lessor"), _col(headers, "tenant", "lessee")
    if landlord_c is not None and tenant_c is not None:
        for r_i, row in enumerate(table["rows"]):
            if not row or not re.match(r"(?:legal\s+)?name\b", row[0].strip(), re.I):
                continue
            for c_i, name in ((landlord_c, "landlord"), (tenant_c, "tenant")):
                if c_i < len(row) and row[c_i].strip():
                    src = {**loc_base, "row_index": r_i + 1, "column_index": c_i, "row_label": row[0],
                           "column": headers[c_i], "bbox": _cell_box(table, r_i, c_i)}
                    fields.append(make_field(name, row[c_i].strip(), row[c_i].strip(), "text", page=page,
                                             source_kind="table", row_label=row[0], source_location=src))
    return rows, fields


def extract_lease_fields(
    text: str, location: dict[str, Any] | None, chunk_type: str
) -> tuple[list[TableRowClassification], list[ExtractedFinancialField]]:
    loc = location or {}
    page = loc.get("page")
    tables = list(loc.get("tables") or [])
    if chunk_type == "table" and not tables:
        parsed = parse_pipe_table(text)
        if parsed:
            parsed.update({"cell_bboxes": loc.get("cell_bboxes"), "table_bbox": loc.get("table_bbox"),
                           "table_index": loc.get("table_index")})
            tables.append(parsed)
    rows: list[TableRowClassification] = []
    fields: list[ExtractedFinancialField] = []
    for tbl in tables:
        r, f = extract_table_terms(tbl, tbl.get("page") or page)
        rows.extend(r)
        fields.extend(f)
    if chunk_type != "table":
        # Blank out embedded table rows (handled above) without shifting character offsets.
        table_lines = {" | ".join(c.strip() for c in r) for tbl in tables for r in [tbl["headers"], *tbl["rows"]]}
        prose = "\n".join(" " * len(ln) if ln.strip() in table_lines else ln for ln in text.split("\n"))
        taken = {f.field for f in fields}
        fields.extend(f for f in extract_clause_fields(prose, page) if f.field not in taken)
    return rows, fields
