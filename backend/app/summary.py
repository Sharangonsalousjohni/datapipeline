"""Financial summary: a compact, human-readable view of the extraction handoff.

The handoff (``RelevantChunksResponse``) is an audit-grade payload: every chunk, score, reference
match and rule evaluation. This module reduces it to what a person or a downstream system needs:

- ``key_figures``: headline metrics, one entry per (metric, measure, period); when the same figure
  appears in a statement, a highlights table and the narrative, the statement value wins.
- ``statements``: each financial table as label x column values.
- ``terms``: contract terms without a reporting period (rent, deposit, late fee, ...).
- ``needs_review``: every value that failed or did not pass validation. Unverified values never
  appear in the sections above.

Values are exact decimal strings plus a ``display`` string; no floats are introduced.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from app.financial_extraction import detect_document_type
from app.taxonomy import match_metric, normalize_period
from app.schemas import (
    ContractTerm,
    ExtractedFinancialField,
    ExtractionHandoff,
    FinancialSummary,
    KeyFigure,
    RelevantChunksResponse,
    ReviewItem,
    StatementLine,
    StatementSummary,
    SummaryChecks,
    SummaryDocument,
    SummaryOutcome,
    SummaryReporting,
    SummaryValue,
)

# Headline metrics shown in key_figures, in display order.
HEADLINE_METRICS = [
    "revenue", "gross_profit", "gross_margin", "operating_income", "operating_margin", "ebitda",
    "pretax_income", "net_income", "net_margin", "eps_diluted", "eps_basic", "dividend_per_share",
    "cfo", "capex", "free_cash_flow", "cash", "net_cash", "total_assets", "total_liabilities",
    "equity", "long_term_debt", "share_repurchases",
]
_HEADLINE_RANK = {m: i for i, m in enumerate(HEADLINE_METRICS)}

STATEMENT_TITLES = {
    "income_statement": "Income statement",
    "balance_sheet": "Balance sheet",
    "cash_flow_statement": "Cash flow statement",
    "highlights": "Financial highlights",
    "notes": "Notes to the financial statements",
    "lease": "Rent schedule",
    "contract": "Contract schedule",
    "other": "Other tables",
}
_STATEMENT_ORDER = list(STATEMENT_TITLES)
# Lower is more authoritative when the same figure is reported more than once.
_SOURCE_PRIORITY = {"income_statement": 0, "balance_sheet": 0, "cash_flow_statement": 0, "highlights": 1, "notes": 2, "other": 3}

DOCUMENT_TYPE_LABELS = {
    "residential_rental_agreement": "Residential rental agreement",
    "commercial_lease_agreement": "Commercial lease agreement",
    "financial_statement": "Financial statement",
    "invoice": "Invoice",
    "bank_statement": "Bank statement",
    "loan_agreement": "Loan agreement",
    "general_financial_document": "Financial document",
}

METRIC_LABELS = {
    "cfo": "Cash from operations",
    "cfi": "Cash from investing",
    "cff": "Cash from financing",
    "capex": "Capital expenditures",
    "free_cash_flow": "Free cash flow",
    "ebitda": "EBITDA",
    "eps_diluted": "Diluted EPS",
    "eps_basic": "Basic EPS",
    "pretax_income": "Income before taxes",
    "cash": "Cash and equivalents",
    "net_cash": "Net cash",
    "equity": "Stockholders' equity",
    "ppe": "Property, plant and equipment",
    "sga": "SG&A",
    "research_and_development": "R&D expense",
    "interest_rate": "Interest rate",
    "debt_maturity_year": "Debt maturity year",
    "credit_facility_maturity_year": "Credit facility maturity year",
    "shares_repurchased": "Shares repurchased",
    "average_repurchase_price": "Average repurchase price",
}

_CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}
_SYMBOL_CURRENCIES = {v: k for k, v in _CURRENCY_SYMBOLS.items()}
_PERIOD_RE = re.compile(r"^(?:FY(\d{4})|Q([1-4])-(\d{4})|H([12])-(\d{4}))$")
_TRUSTED = ("PASSED", "WARNING")


# --------------------------------------------------------------------------- value formatting
def _dec(v: str | None) -> Decimal | None:
    if v in (None, ""):
        return None
    try:
        return Decimal(str(v).replace(",", ""))
    except InvalidOperation:
        return None


def _trim(d: Decimal, places: int = 2) -> str:
    s = f"{d:,.{places}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


_NON_NUMERIC_TYPES = ("text", "date", "period")
_DURATION_UNITS = ("months", "days", "years", "weeks")


def _unit(f: ExtractedFinancialField) -> str:
    dt = f.data_type
    if dt in _NON_NUMERIC_TYPES:
        return dt
    if dt in ("duration", "area", "count"):
        unit = (f.details or {}).get("unit")
        return unit if isinstance(unit, str) and unit else dt
    if dt == "limit":
        return f.currency or "number"
    if dt in ("monetary", "monetary_change"):
        if not f.currency and "%" in f.original_text:
            return "percent"  # e.g. a late-fee percentage stored with the monetary default
        return f.currency or "number"
    if dt in ("per_share", "price") or f.measure == "price":
        # Per-share amounts are money: "$2.71" should display as currency.
        return f.currency or next((c for s, c in _SYMBOL_CURRENCIES.items() if s in f.original_text), dt)
    return {"percentage": "percent", "basis_points": "basis_points"}.get(dt, dt or "number")


def _measure(f: ExtractedFinancialField, unit: str) -> str:
    measure = f.measure or "level"
    if unit == "basis_points" and measure == "level":
        return "change_bps"  # basis points express a movement, never a level
    if measure == "change_pct" and len(unit) == 3 and unit.isupper():
        return "change_amount"  # a "Change ($)" column holds money, not a percentage
    return measure


def _display(amount: Decimal, unit: str, *, signed: bool = False, exact_cents: bool = False) -> str:
    sign = "-" if amount < 0 else ("+" if signed and amount > 0 else "")
    a = abs(amount)
    if unit == "percent":
        return f"{sign}{_trim(a, 4)}%"
    if unit == "basis_points":
        return f"{sign}{_trim(a, 2)} bps"
    if unit == "ratio":
        return f"{sign}{_trim(a, 2)}x"
    if unit == "year":
        return f"{amount:.0f}"
    if unit in _DURATION_UNITS:
        return f"{sign}{_trim(a, 2)} {unit[:-1] if a == 1 else unit}"
    if unit in ("sq_ft", "sq_m"):
        return f"{sign}{_trim(a, 2)} {'sq ft' if unit == 'sq_ft' else 'sq m'}"
    if len(unit) == 3 and unit.isupper():
        sym = _CURRENCY_SYMBOLS.get(unit)
        if a >= 1_000_000_000:
            body = f"{_trim(a / 1_000_000_000)} billion"
        elif a >= 1_000_000:
            body = f"{_trim(a / 1_000_000)} million"
        else:
            body = f"{a:,.2f}" if exact_cents or a != a.to_integral_value() else f"{a:,.0f}"
        return f"{sign}{sym}{body}" if sym else f"{sign}{body} {unit}"
    return f"{sign}{_trim(a, 4)}"


def _value(f: ExtractedFinancialField) -> SummaryValue:
    raw = f.normalized_amount or f.normalized_value
    unit = _unit(f)
    as_reported = " ".join(f.original_text.split())
    if unit in _NON_NUMERIC_TYPES:
        return SummaryValue(value=str(raw), unit=unit, display=str(raw), as_reported=as_reported)
    amount = _dec(raw)
    if amount is None:
        return SummaryValue(value=str(raw), unit=unit, display=f.original_text.strip(), as_reported=f.original_text.strip())
    signed = f.data_type == "monetary_change" or _measure(f, unit).startswith("change")
    display = _display(amount, unit, signed=signed, exact_cents=bool(re.search(r"\.\d\d\b", f.original_text)))
    return SummaryValue(value=str(raw), unit=unit, display=display, as_reported=" ".join(f.original_text.split()))


def _humanize(slug: str) -> str:
    s = slug.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def _period_sort_key(p: str) -> tuple[int, int]:
    m = _PERIOD_RE.match(p)
    if not m:
        return (0, 0)
    if m.group(1):
        return (int(m.group(1)), 9)
    if m.group(2):
        return (int(m.group(3)), int(m.group(2)) * 2)
    return (int(m.group(5)), int(m.group(4)) * 4)


# --------------------------------------------------------------------------- field routing
def _page(item: ExtractionHandoff, f: ExtractedFinancialField) -> int | None:
    page = (f.source_location or {}).get("page")
    return page if isinstance(page, int) else item.source_location.page


def _is_trusted(f: ExtractedFinancialField) -> bool:
    return f.validation_status in _TRUSTED and not f.review_required


def _review_reason(f: ExtractedFinancialField) -> str:
    for issue in f.validation_issues:
        msg = issue.get("message") or issue.get("rule_id")
        if msg:
            return str(msg)
    if f.validation_status not in _TRUSTED:
        return f"Validation status {f.validation_status}"
    return f.validation_notes or "Flagged for manual review"


def _statement_of(item: ExtractionHandoff, f: ExtractedFinancialField) -> str:
    st = f.statement_type or item.statement_type or "other"
    return st if st in STATEMENT_TITLES else "other"


def _column_of(f: ExtractedFinancialField) -> str:
    column = (f.source_location or {}).get("column")
    if column:  # schedules (one row per lease year) name their real column header
        return str(column)
    if (f.measure in (None, "level")) and f.period_key:
        return f.period_key
    return (f.reporting_period or f.period_key or "Value").strip()


def build_financial_summary(
    handoff: RelevantChunksResponse,
    *,
    filename: str,
    source_type: str,
) -> FinancialSummary:
    pairs = [(item, f) for item in handoff.items for f in item.financial_fields]

    needs_review: list[ReviewItem] = []
    seen_review: set[tuple[str, str]] = set()

    def flag(item: ExtractionHandoff, label: str, as_reported: str, reason: str) -> None:
        key = (label, as_reported)
        if key in seen_review:
            return
        seen_review.add(key)
        needs_review.append(
            ReviewItem(
                label=label, as_reported=as_reported, reason=reason,
                page=item.source_location.page, section=item.source_location.section_heading,
            )
        )

    # Key-figure candidates as (rank, statement or None, figure); rank decides which duplicate wins.
    candidates: list[tuple[int, str | None, KeyFigure]] = []
    tables: dict[str, dict] = {}
    terms: dict[str, ContractTerm] = {}

    for item, f in pairs:
        label = (f.row_label or "").strip() or _humanize(f.field)
        if not _is_trusted(f):
            flag(item, label, " ".join(f.original_text.split()), _review_reason(f))
            continue

        val = _value(f)
        page = _page(item, f)
        # Table cells without a reporting period (party names, "Label: value" grids) read as contract terms.
        is_table = (
            f.source_kind == "table" or bool(f.row_label and f.source_location and "row_index" in f.source_location)
        ) and bool(f.period_key or f.reporting_period)

        if is_table:
            st = _statement_of(item, f)
            t = tables.setdefault(st, {"columns": [], "lines": {}, "currencies": Counter(), "scales": Counter(), "page": page})
            col = _column_of(f)
            if col not in t["columns"]:
                t["columns"].append(col)
            line = t["lines"].setdefault(label, StatementLine(label=label, metric=f.metric, values={}))
            line.values.setdefault(col, val)
            if f.currency and f.data_type in ("monetary", "monetary_change"):
                t["currencies"][f.currency] += 1
            if f.reported_scale and f.reported_scale != "unknown":
                t["scales"][f.reported_scale] += 1

        period = f.period_key or normalize_period(f.reporting_period) or f.reporting_period
        metric = f.metric
        if not metric and not is_table and period:
            # KPI tiles such as "Total Revenue Summary" carry no metric; map the label to one.
            metric = match_metric(re.sub(r"\bsummary\b", "", label, flags=re.I).strip(), allow_embedding=False).metric
        if (is_table and metric in _HEADLINE_RANK) or (not is_table and period):
            if not is_table and metric:
                label = METRIC_LABELS.get(metric) or _humanize(metric)
            candidates.append(
                (
                    _SOURCE_PRIORITY.get(_statement_of(item, f), 3) if is_table else 4,
                    _statement_of(item, f) if is_table else None,
                    KeyFigure(
                        **val.model_dump(), label=label, metric=metric, measure=_measure(f, val.unit),
                        period=period, page=page,
                    ),
                )
            )
        elif not is_table and not period and not f.metric:
            term = _contract_term(f, val, label, page)
            existing = terms.get(term.term)
            if existing is None:
                terms[term.term] = term
            elif _dec(existing.value) != _dec(term.value):
                flag(item, term.label, term.as_reported, f"Conflicts with {existing.as_reported} stated elsewhere in the document")

    # A table's "Change" column compares its newest period with the prior one.
    newest = {
        st: max((c for c in t["columns"] if _PERIOD_RE.match(c)), key=_period_sort_key, default=None)
        for st, t in tables.items()
    }
    figures: dict[tuple[str, str, str], tuple[int, KeyFigure]] = {}
    for rank, st, kf in candidates:
        if st and kf.measure != "level" and not _PERIOD_RE.match(kf.period or "") and newest.get(st):
            kf.period = newest[st]
        key = (kf.metric or re.sub(r"[^a-z0-9]+", "_", kf.label.lower()), kf.measure, kf.period or "")
        if key not in figures or rank < figures[key][0]:
            figures[key] = (rank, kf)

    key_figures = [kf for _, kf in figures.values()]
    key_figures.sort(
        key=lambda k: (
            _HEADLINE_RANK.get(k.metric or "", len(_HEADLINE_RANK)),
            k.label.lower(),
            k.measure != "level",
            tuple(-x for x in _period_sort_key(k.period or "")),
        )
    )

    statements = [
        StatementSummary(
            statement=st,
            title=STATEMENT_TITLES[st],
            currency=t["currencies"].most_common(1)[0][0] if t["currencies"] else None,
            reported_in=t["scales"].most_common(1)[0][0] if t["scales"] else None,
            columns=t["columns"],
            lines=list(t["lines"].values()),
            page=t["page"],
        )
        for st, t in sorted(tables.items(), key=lambda kv: _STATEMENT_ORDER.index(kv[0]))
    ]

    trusted = [f for _, f in pairs if _is_trusted(f)]
    currencies = Counter(f.currency for f in trusted if f.currency and f.data_type in ("monetary", "monetary_change"))
    scales = Counter(f.reported_scale for f in trusted if f.reported_scale and f.reported_scale != "unknown")
    periods = sorted(
        {f.period_key for f in trusted if f.period_key and _PERIOD_RE.match(f.period_key)},
        key=_period_sort_key, reverse=True,
    )

    checks = _checks(handoff)
    outcome = _outcome(handoff, checks, len(key_figures), statements, len(terms), len(needs_review))

    sample = " ".join(
        f"{i.source_location.section_heading or ''} {i.context or ''} {i.extracted_text[:2000]}" for i in handoff.items[:4]
    )
    doc_type = detect_document_type(sample, filename)

    return FinancialSummary(
        generated_at=datetime.now(timezone.utc),
        document=SummaryDocument(
            id=handoff.document_id,
            filename=filename,
            source_type=source_type,
            document_type=DOCUMENT_TYPE_LABELS.get(doc_type, _humanize(doc_type)),
            relevance_status=handoff.relevance_status,
        ),
        outcome=outcome,
        reporting=SummaryReporting(
            currency=currencies.most_common(1)[0][0] if currencies else None,
            reported_in=scales.most_common(1)[0][0] if scales else None,
            periods=periods,
        ),
        key_figures=key_figures,
        statements=statements,
        terms=list(terms.values()),
        needs_review=needs_review,
        checks=checks,
    )


def _contract_term(f: ExtractedFinancialField, val: SummaryValue, label: str, page: int | None) -> ContractTerm:
    display = val.display
    d = f.details or {}
    if d.get("type") == "conditional_greater_of" and d.get("fixed_amount") and d.get("percentage"):
        fixed = _dec(d["fixed_amount"])
        fixed_txt = _display(fixed, d.get("currency") or val.unit, exact_cents=True) if fixed is not None else d["fixed_amount"]
        base = f" of the {d['percentage_base']}" if d.get("percentage_base") else ""
        display = f"{fixed_txt} or {d['percentage']}%{base}, {d.get('condition') or 'whichever is greater'}"
    return ContractTerm(
        **{**val.model_dump(), "display": display},
        term=f.field,
        # The row label names the term only when it is the term ("Monthly Rent"), not a table row ("Name").
        label=label if f.row_label and re.sub(r"[^a-z0-9]+", "_", f.row_label.lower()).strip("_") == f.field
        else _humanize(f.field),
        frequency=f.frequency,
        due=f.due_date,
        condition=f.condition,
        page=page,
    )


def _checks(handoff: RelevantChunksResponse) -> SummaryChecks:
    vs = handoff.validation_summary
    if vs is None:
        return SummaryChecks()
    rows = list(vs.accounting_validations) + list(vs.cross_reference_checks)
    status = Counter(str(r.get("status", "")).upper() for r in rows)
    issues = [str(r.get("message") or r.get("rule_id")) for r in rows if str(r.get("status", "")).upper() != "PASSED"]
    return SummaryChecks(
        total=len(rows),
        passed=status["PASSED"],
        warnings=status["WARNING"],
        failed=len(rows) - status["PASSED"] - status["WARNING"],
        issues=issues,
    )


_DECISION_WORDS = {
    "ACCEPT": "Accepted",
    "ACCEPT_WITH_WARNINGS": "Accepted with warnings",
    "HOLD_FOR_REVIEW": "Held for review",
    "REJECT": "Rejected",
}


def _outcome(
    handoff: RelevantChunksResponse,
    checks: SummaryChecks,
    n_figures: int,
    statements: list[StatementSummary],
    n_terms: int,
    n_review: int,
) -> SummaryOutcome:
    vs = handoff.validation_summary
    decision = vs.quality_gate_decision if vs else "ACCEPT"

    found = [
        f"{n} {noun}{'' if n == 1 else 's'}"
        for n, noun in ((n_figures, "key figure"), (len(statements), "table"), (n_terms, "contract term"))
        if n
    ]
    parts = [f"{', '.join(found)} extracted" if found else "No validated financial values extracted"]
    if checks.total == 1:
        parts.append("1 check passed" if checks.passed else "1 check did not pass")
    elif checks.total:
        parts.append(
            f"all {checks.total} checks passed" if checks.passed == checks.total
            else f"{checks.passed} of {checks.total} checks passed"
        )
    if n_review:
        parts.append(f"{n_review} value{'' if n_review == 1 else 's'} need{'s' if n_review == 1 else ''} review")
    headline = f"{_DECISION_WORDS.get(decision, decision)}: {'; '.join(parts)}."

    return SummaryOutcome(
        decision=decision,
        requires_review=bool((vs and vs.requires_review) or decision in ("HOLD_FOR_REVIEW", "REJECT") or n_review),
        headline=headline,
        reasons=list(vs.gate_reasons) if vs else [],
    )
