"""Arithmetic and completeness checks for extracted lease terms (see app.contracts).

- RULE_LEASE_RENT_ANNUAL_001: monthly x 12 == annual, per lease year
- RULE_LEASE_RENT_TOTAL_001: sum of annual rents == total_rent
- RULE_LEASE_ESCALATION_001: each year's monthly rent == previous year's x (1 + escalation), to the cent
- RULE_LEASE_DEPOSIT_001: security_deposit == stated multiple x Year 1 monthly rent
- RULE_LEASE_TERM_DATES_001: expiration == commencement + term months - 1 day
- RULE_RENTAL_TERMS_001: essential terms (rent, deposit, late fee, term) are present among extracted fields

A failed check marks its result field FAILED and the operands review_required; every check updates the
participating fields' pass/fail counts so confidence reflects it.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.metric_validation import FieldRef

VERSION = "1.0.0"
_CENT = Decimal("0.01")


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _iso(v: str | None) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None


class _Checks:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def record(
        self,
        rule_id: str,
        *,
        period: str | None,
        formula: str,
        expected: Any,
        actual: Any,
        difference: Any,
        tolerance: Any,
        passed: bool,
        message: str,
        result: FieldRef | None,
        operands: list[FieldRef],
    ) -> None:
        status = "PASSED" if passed else "FAILED"
        participants = [r for r in [result, *operands] if r is not None]
        self.checks.append({
            "rule_id": rule_id,
            "rule_version": VERSION,
            "statement": "Lease",
            "period": period,
            "formula": formula,
            "expected": str(expected),
            "actual": str(actual),
            "difference": str(difference),
            "tolerance": str(tolerance),
            "status": status,
            "message": f"{message}: expected {expected}, got {actual}",
            "fields": [r.field.field for r in participants],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        for r in participants:
            if passed:
                r.checks_passed += 1
            else:
                r.checks_failed += 1
                r.field.review_required = True
        if not passed and result is not None:
            f = result.field
            f.validation_status = "FAILED"
            f.validation_issues.append({
                "rule_id": rule_id, "rule_version": VERSION, "severity": "ERROR",
                "message": self.checks[-1]["message"], "expected": str(expected), "actual": str(actual),
                "review_required": True,
            })


def validate_lease(refs: list[FieldRef]) -> list[dict[str, Any]]:
    by_name: dict[str, FieldRef] = {}
    for r in refs:
        by_name.setdefault(r.field.field, r)  # first occurrence wins; duplicates are reported by the summary
    c = _Checks()

    years = sorted(
        int(m.group(1)) for n in by_name if (m := re.fullmatch(r"base_rent_monthly_year_(\d+)", n))
    )
    for y in years:
        monthly, annual = by_name.get(f"base_rent_monthly_year_{y}"), by_name.get(f"base_rent_annual_year_{y}")
        if monthly and annual and monthly.value is not None and annual.value is not None:
            expected = monthly.value * 12
            diff = abs(expected - annual.value)
            c.record("RULE_LEASE_RENT_ANNUAL_001", period=f"Year {y}", formula="annual rent == monthly rent x 12",
                     expected=f"{expected:.2f}", actual=f"{annual.value:.2f}", difference=f"{diff:.2f}", tolerance=_CENT,
                     passed=diff <= _CENT, message=f"Year {y} annual rent", result=annual, operands=[monthly])

    annuals = [by_name.get(f"base_rent_annual_year_{y}") for y in years]
    total = by_name.get("total_rent")
    if total and total.value is not None and annuals and all(a and a.value is not None for a in annuals):
        expected = sum((a.value for a in annuals), Decimal(0))  # type: ignore[union-attr,misc]
        diff = abs(expected - total.value)
        c.record("RULE_LEASE_RENT_TOTAL_001", period="Term", formula="total rent == sum of annual rents",
                 expected=f"{expected:.2f}", actual=f"{total.value:.2f}", difference=f"{diff:.2f}", tolerance=_CENT,
                 passed=diff <= _CENT, message="Total rent", result=total, operands=[a for a in annuals if a])

    esc = by_name.get("escalation_pct")
    if esc and esc.value is not None:
        factor = 1 + esc.value / 100
        for prev_y, y in zip(years, years[1:]):
            prev, cur = by_name.get(f"base_rent_monthly_year_{prev_y}"), by_name.get(f"base_rent_monthly_year_{y}")
            if prev and cur and prev.value is not None and cur.value is not None:
                expected = (prev.value * factor).quantize(_CENT, rounding=ROUND_HALF_UP)
                diff = abs(expected - cur.value)
                c.record("RULE_LEASE_ESCALATION_001", period=f"Year {y}",
                         formula=f"monthly rent year {y} == year {prev_y} x (1 + {esc.value}%)",
                         expected=f"{expected:.2f}", actual=f"{cur.value:.2f}", difference=f"{diff:.2f}", tolerance=_CENT,
                         passed=diff <= _CENT, message=f"Year {y} escalated monthly rent", result=cur, operands=[prev, esc])

    dep, first = by_name.get("security_deposit"), by_name.get("base_rent_monthly_year_1")
    multiple = (dep.field.details or {}).get("stated_multiple_of_monthly_rent") if dep else None
    if dep and first and multiple and dep.value is not None and first.value is not None:
        expected = first.value * Decimal(multiple)
        diff = abs(expected - dep.value)
        c.record("RULE_LEASE_DEPOSIT_001", period="Term", formula=f"security deposit == {multiple} x Year 1 monthly rent",
                 expected=f"{expected:.2f}", actual=f"{dep.value:.2f}", difference=f"{diff:.2f}", tolerance=_CENT,
                 passed=diff <= _CENT, message="Security deposit", result=dep, operands=[first])

    start, end, term = by_name.get("commencement_date"), by_name.get("expiration_date"), by_name.get("term_months")
    s_d, e_d = _iso(start.field.normalized_value) if start else None, _iso(end.field.normalized_value) if end else None
    if s_d and e_d and term and term.value is not None:
        expected_d = _add_months(s_d, int(term.value)) - timedelta(days=1)
        diff_days = abs((e_d - expected_d).days)
        c.record("RULE_LEASE_TERM_DATES_001", period="Term",
                 formula="expiration date == commencement date + term months - 1 day",
                 expected=expected_d.isoformat(), actual=e_d.isoformat(), difference=f"{diff_days} days", tolerance="0 days",
                 passed=diff_days == 0, message="Lease expiration date", result=end, operands=[start, term])  # type: ignore[list-item]

    c.checks.append(_essential_terms(by_name))
    return c.checks


_ESSENTIAL = {
    "rent": ("base_rent_monthly_year_1", "monthly_rent", "total_rent"),
    "deposit": ("security_deposit",),
    "late_fee": ("late_fee_pct", "late_fee_amount", "late_fee_clause"),
    "term": ("term_months", "lease_start_date", "commencement_date"),
}


def _essential_terms(by_name: dict[str, FieldRef]) -> dict[str, Any]:
    """RULE_RENTAL_TERMS_001 judged on the extracted fields, so terms present in the text are found."""
    found = {term: next((n for n in names if n in by_name), None) for term, names in _ESSENTIAL.items()}
    missing = [t for t, n in found.items() if n is None]
    # Rent and deposit are required (as before); late fee and term are reported when absent.
    passed = found["rent"] is not None and found["deposit"] is not None
    return {
        "rule_id": "RULE_RENTAL_TERMS_001",
        "rule_version": VERSION,
        "statement": "Rental Agreement Terms",
        "period": "Term",
        "formula": "rent, security deposit, late fee and term are extracted",
        "expected": "rent, deposit, late_fee, term",
        "actual": ", ".join(f"{t}={n}" for t, n in found.items() if n) or "none",
        "status": "PASSED" if passed else "WARNING",
        "message": "Essential rental terms present" if not missing else f"Essential rental terms not found: {', '.join(missing)}",
        "fields": [n for n in found.values() if n],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
