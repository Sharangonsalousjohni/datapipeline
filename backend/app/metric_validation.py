"""Metric-level validation: checks that can actually fail.

Operates on fields that carry a canonical ``metric`` and ``period_key`` (tables and narrative):

- Cross-table consistency: the same metric/period in two tables must agree after scaling.
- Statement arithmetic (per table, per period), tolerance = half a unit of each operand's printed
  precision, summed over the operands:
    income statement: GP = revenue - COGS; total opex = sum of opex lines; OI = GP - opex;
                      pre-tax = OI + other income; NI = pre-tax - tax
    balance sheet:    each total = sum of its components; TA = TL + equity = TL&E
- Recomputed period-over-period change % vs the reported change column.
- Narrative facts vs table facts (levels, growth %, margins, % of revenue, bps, net cash).
- Evidence-based confidence for every field.

Failing checks mark the most likely culprit FAILED (the field involved in the most failed checks,
ties broken towards the check's result/total), and every other participant review_required.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from app.numbers import format_decimal
from app.schemas import ExtractedFinancialField
from app.taxonomy import shift_period

VERSION = "2.0.0"
_ZERO = Decimal(0)
_HUNDRED = Decimal(100)

_EXPENSE_METRICS = {
    "cost_of_revenue", "research_and_development", "sales_and_marketing", "general_and_administrative", "sga",
    "total_operating_expenses", "income_tax", "interest_expense",
}
_TOTAL_PREFIXES = ("total_",)
_STATEMENT_LABEL = {"income_statement": "Income Statement", "balance_sheet": "Balance Sheet"}


@dataclass
class FieldRef:
    field: ExtractedFinancialField
    group: str  # chunk id (one table / one narrative chunk)
    extraction_method: str | None = None
    ocr_confidence: float | None = None
    checks_passed: int = 0
    checks_failed: int = 0
    in_failed: list[int] = field(default_factory=list)

    @property
    def value(self) -> Decimal | None:
        f = self.field
        raw = f.normalized_amount if f.normalized_amount not in (None, "") else f.normalized_value
        try:
            return Decimal(str(raw)) if raw not in (None, "") else None
        except InvalidOperation:
            return None

    @property
    def tol(self) -> Decimal:
        t = (self.field.details or {}).get("tolerance")
        try:
            return Decimal(str(t)) if t not in (None, "") else _ZERO
        except InvalidOperation:
            return _ZERO

    @property
    def row(self) -> int:
        return int((self.field.source_location or {}).get("row_index") or 0)

    @property
    def is_table(self) -> bool:
        return self.field.source_kind == "table"


def _fmt(d: Decimal | None) -> str | None:
    return format_decimal(d.quantize(Decimal("0.0001")) if d is not None and d != d.to_integral_value() else d)


class MetricValidator:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self._participants: list[list[tuple[FieldRef, str]]] = []  # per check: (ref, role) role=result|operand

    # ------------------------------------------------------------------ bookkeeping
    def _record(
        self,
        rule_id: str,
        *,
        statement: str,
        period: str | None,
        formula: str,
        expected: Decimal,
        actual: Decimal,
        tolerance: Decimal,
        result: FieldRef,
        operands: list[FieldRef],
        message: str,
    ) -> dict[str, Any]:
        diff = abs(expected - actual)
        status = "PASSED" if diff <= tolerance else "FAILED"
        chk = {
            "rule_id": rule_id,
            "rule_version": VERSION,
            "statement": statement,
            "period": period,
            "formula": formula,
            "expected": _fmt(expected),
            "actual": _fmt(actual),
            "difference": _fmt(diff),
            "tolerance": _fmt(tolerance),
            "status": status,
            "message": f"{message}: expected {_fmt(expected)}, got {_fmt(actual)} (diff {_fmt(diff)}, tolerance {_fmt(tolerance)})",
            "fields": [r.field.field for r in [result, *operands]],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.checks.append(chk)
        self._participants.append([(result, "result")] + [(o, "operand") for o in operands])
        idx = len(self.checks) - 1
        for ref in [result, *operands]:
            if status == "PASSED":
                ref.checks_passed += 1
            else:
                ref.checks_failed += 1
                ref.in_failed.append(idx)
        return chk

    # ------------------------------------------------------------------ entry point
    def run(self, refs: list[FieldRef]) -> list[dict[str, Any]]:
        scoped = [r for r in refs if r.field.metric and r.value is not None and r.field.data_type != "null"]
        tables: dict[str, list[FieldRef]] = defaultdict(list)
        for r in scoped:
            if r.is_table:
                tables[r.group].append(r)
        for group, trefs in tables.items():
            st = next((r.field.statement_type for r in trefs if r.field.statement_type), None)
            self._statement_arithmetic(trefs, st)
            self._change_columns(trefs)
        self._cross_table(scoped)
        self._narrative_vs_tables(scoped)
        self._assign_failures()
        for r in refs:
            score_confidence(r)
        return self.checks

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _levels(refs: list[FieldRef]) -> dict[str, dict[str, FieldRef]]:
        """period_key -> metric -> first level field (table order)."""
        out: dict[str, dict[str, FieldRef]] = defaultdict(dict)
        for r in sorted(refs, key=lambda x: x.row):
            f = r.field
            if f.measure == "level" and f.period_key and f.data_type in ("monetary", "per_share", "shares"):
                out[f.period_key].setdefault(f.metric, r)  # type: ignore[arg-type]
        return out

    @staticmethod
    def _signed(r: FieldRef) -> Decimal:
        v = r.value or _ZERO
        return -abs(v) if r.field.metric in _EXPENSE_METRICS else v

    def _sum_check(
        self, rule: str, statement: str, period: str, result: FieldRef, parts: list[FieldRef], formula: str, msg: str,
        *, absolute: bool = False,
    ) -> None:
        if not parts:
            return
        expected = sum(((abs(p.value or _ZERO) if absolute else (p.value or _ZERO)) for p in parts), _ZERO)
        actual = abs(result.value or _ZERO) if absolute else (result.value or _ZERO)
        tol = result.tol + sum((p.tol for p in parts), _ZERO)
        self._record(rule, statement=statement, period=period, formula=formula, expected=expected, actual=actual,
                     tolerance=tol, result=result, operands=parts, message=msg)

    # ------------------------------------------------------------------ statement arithmetic
    def _statement_arithmetic(self, refs: list[FieldRef], statement_type: str | None) -> None:
        levels = self._levels(refs)
        for period, m in levels.items():
            rows = sorted((r for r in m.values() if r.field.data_type == "monetary"), key=lambda r: r.row)

            def between(a: FieldRef | None, b: FieldRef) -> list[FieldRef]:
                lo = a.row if a else 0
                return [r for r in rows if lo < r.row < b.row]

            is_label = _STATEMENT_LABEL.get(statement_type or "", "Income Statement")
            rev, cogs, gp = m.get("revenue"), m.get("cost_of_revenue"), m.get("gross_profit")
            opex, oi = m.get("total_operating_expenses"), m.get("operating_income")
            pretax, tax, ni = m.get("pretax_income"), m.get("income_tax"), m.get("net_income")
            if rev and cogs and gp:
                self._record("RULE_ARITH_IS_GP_001", statement=is_label, period=period,
                             formula="gross_profit = revenue - cost_of_revenue",
                             expected=(rev.value or _ZERO) - abs(cogs.value or _ZERO), actual=gp.value or _ZERO,
                             tolerance=rev.tol + cogs.tol + gp.tol, result=gp, operands=[rev, cogs],
                             message=f"Gross profit {period}")
            if opex:
                lines = [r for r in between(gp or cogs, opex) if r.field.metric not in ("gross_profit", "revenue")]
                if not lines:
                    lines = [m[k] for k in ("research_and_development", "sales_and_marketing", "general_and_administrative", "sga") if k in m]
                self._sum_check("RULE_ARITH_IS_OPEX_001", is_label, period, opex, lines,
                                "total_operating_expenses = sum(operating expense lines)", f"Total operating expenses {period}",
                                absolute=True)
            if gp and opex and oi:
                self._record("RULE_ARITH_IS_OPINC_001", statement=is_label, period=period,
                             formula="operating_income = gross_profit - total_operating_expenses",
                             expected=(gp.value or _ZERO) - abs(opex.value or _ZERO), actual=oi.value or _ZERO,
                             tolerance=gp.tol + opex.tol + oi.tol, result=oi, operands=[gp, opex],
                             message=f"Operating income {period}")
            if oi and pretax:
                others = between(oi, pretax)
                if others:
                    expected = (oi.value or _ZERO) + sum((self._signed(r) for r in others), _ZERO)
                    self._record("RULE_ARITH_IS_PRETAX_001", statement=is_label, period=period,
                                 formula="pretax_income = operating_income + other income (net)",
                                 expected=expected, actual=pretax.value or _ZERO,
                                 tolerance=oi.tol + pretax.tol + sum((r.tol for r in others), _ZERO),
                                 result=pretax, operands=[oi, *others], message=f"Income before taxes {period}")
            if pretax and tax and ni:
                self._record("RULE_ARITH_IS_NETINC_001", statement=is_label, period=period,
                             formula="net_income = pretax_income - income_tax",
                             expected=(pretax.value or _ZERO) - abs(tax.value or _ZERO), actual=ni.value or _ZERO,
                             tolerance=pretax.tol + tax.tol + ni.tol, result=ni, operands=[pretax, tax],
                             message=f"Net income {period}")

            # balance sheet: positional component sums
            bs = "Balance Sheet"
            tca, ta = m.get("total_current_assets"), m.get("total_assets")
            tcl, tl = m.get("total_current_liabilities"), m.get("total_liabilities")
            eq, tle = m.get("equity"), m.get("total_liabilities_and_equity")

            def components(a: FieldRef | None, b: FieldRef) -> list[FieldRef]:
                return [r for r in between(a, b) if not (r.field.metric or "").startswith(_TOTAL_PREFIXES)]

            if tca:
                self._sum_check("RULE_ARITH_BS_TCA_001", bs, period, tca, components(None, tca),
                                "total_current_assets = sum(current asset lines)", f"Total current assets {period}")
            if ta:
                parts = ([tca] if tca else []) + components(tca, ta)
                self._sum_check("RULE_ARITH_BS_TA_001", bs, period, ta, parts,
                                "total_assets = total_current_assets + non-current asset lines", f"Total assets {period}")
            if tcl:
                self._sum_check("RULE_ARITH_BS_TCL_001", bs, period, tcl, components(ta, tcl),
                                "total_current_liabilities = sum(current liability lines)", f"Total current liabilities {period}")
            if tl:
                parts = ([tcl] if tcl else []) + components(tcl or ta, tl)
                self._sum_check("RULE_ARITH_BS_TL_001", bs, period, tl, parts,
                                "total_liabilities = total_current_liabilities + non-current liability lines", f"Total liabilities {period}")
            if eq and tl:
                eq_parts = components(tl, eq)
                if eq_parts and (eq.field.row_label or "").lower().startswith("total"):
                    self._sum_check("RULE_ARITH_BS_EQ_001", bs, period, eq, eq_parts,
                                    "equity = sum(equity lines)", f"Stockholders' equity {period}")
            if tle and tl and eq:
                self._record("RULE_ARITH_BS_TLE_001", statement=bs, period=period,
                             formula="total_liabilities_and_equity = total_liabilities + equity",
                             expected=(tl.value or _ZERO) + (eq.value or _ZERO), actual=tle.value or _ZERO,
                             tolerance=tl.tol + eq.tol + tle.tol, result=tle, operands=[tl, eq],
                             message=f"Total liabilities and equity {period}")
            if ta and tl and eq:
                self._record("RULE_ARITH_BS_BALANCE_001", statement=bs, period=period,
                             formula="total_assets = total_liabilities + equity",
                             expected=(tl.value or _ZERO) + (eq.value or _ZERO), actual=ta.value or _ZERO,
                             tolerance=tl.tol + eq.tol + ta.tol, result=ta, operands=[tl, eq],
                             message=f"Balance sheet equation {period}")
            if ta and tle:
                self._record("RULE_ARITH_BS_TA_TLE_001", statement=bs, period=period,
                             formula="total_assets = total_liabilities_and_equity",
                             expected=tle.value or _ZERO, actual=ta.value or _ZERO, tolerance=ta.tol + tle.tol,
                             result=tle, operands=[ta], message=f"Assets vs liabilities and equity {period}")

    # ------------------------------------------------------------------ change columns
    @staticmethod
    def _growth(cur: FieldRef, prior: FieldRef) -> tuple[Decimal, Decimal] | None:
        c, p = cur.value, prior.value
        if c is None or p is None or p == 0:
            return None
        g = (c - p) / abs(p) * _HUNDRED
        # rounding of the inputs propagates into the recomputed percentage
        prop = _HUNDRED * (cur.tol / abs(p) + abs(c) * prior.tol / (p * p))
        return g, prop

    def _change_columns(self, refs: list[FieldRef]) -> None:
        by_row: dict[int, list[FieldRef]] = defaultdict(list)
        for r in refs:
            by_row[r.row].append(r)
        for row, rs in by_row.items():
            changes = [r for r in rs if r.field.measure == "change_pct" and r.field.data_type == "percentage"]
            levels = sorted((r for r in rs if r.field.measure == "level" and r.field.period_key), key=lambda r: r.field.period_key or "")
            if not changes or len(levels) < 2:
                continue
            prior, cur = levels[0], levels[-1]
            res = self._growth(cur, prior)
            if res is None:
                continue
            g, prop = res
            for ch in changes:
                self._record("RULE_CHANGE_PCT_001", statement=ch.field.statement_type or "table", period=cur.field.period_key,
                             formula=f"change % = ({cur.field.period_key} - {prior.field.period_key}) / |{prior.field.period_key}| x 100",
                             expected=g, actual=ch.value or _ZERO, tolerance=ch.tol + prop, result=ch, operands=[cur, prior],
                             message=f"Reported change for {ch.field.row_label}")

    # ------------------------------------------------------------------ cross-table
    def _cross_table(self, refs: list[FieldRef]) -> None:
        groups: dict[tuple[str, str, str], list[FieldRef]] = defaultdict(list)
        for r in refs:
            f = r.field
            if r.is_table and f.measure == "level" and f.period_key:
                groups[(f.metric, f.period_key, f.data_type)].append(r)  # type: ignore[index]
        for (metric, period, _), all_rs in groups.items():
            if len({r.group for r in all_rs}) < 2:
                continue
            rs = [r for r in all_rs if r.field.reported_scale != "unknown"]
            # A value without a known scale cannot be compared; report the scale that would reconcile it.
            for unk in (r for r in all_rs if r.field.reported_scale == "unknown"):
                known = next((r for r in rs if r.group != unk.group and r.value), None)
                if known is None or not unk.value:
                    continue
                ratio = (known.value or _ZERO) / (unk.value or Decimal(1))
                implied = {Decimal(1000): "thousands", Decimal(1_000_000): "millions", Decimal(1_000_000_000): "billions"}
                hit = next((name for mult, name in implied.items() if abs(ratio - mult) <= mult * Decimal("0.001")), None)
                if hit:
                    unk.field.validation_issues.append({
                        "rule_id": "RULE_SCALE_INFERENCE_001",
                        "rule_version": VERSION,
                        "severity": "INFO",
                        "message": f"{metric} {period} would match {known.field.statement_type or 'another table'} if this table were in {hit}; scale left as unknown for review",
                        "suggested_scale": hit,
                        "review_required": True,
                    })
            if len({r.group for r in rs}) < 2:
                continue
            anchor = min(rs, key=lambda r: r.tol)  # most precise table is the reference
            for other in rs:
                if other.group == anchor.group:
                    continue
                self._record("RULE_XTABLE_001", statement="Cross-table", period=period,
                             formula=f"{metric} must agree across tables after scaling",
                             expected=anchor.value or _ZERO, actual=other.value or _ZERO,
                             tolerance=max(anchor.tol, other.tol), result=other, operands=[anchor],
                             message=f"{metric} {period} in {other.field.statement_type or 'table'} vs {anchor.field.statement_type or 'table'}")

    # ------------------------------------------------------------------ narrative vs tables
    def _narrative_vs_tables(self, refs: list[FieldRef]) -> None:
        table_levels: dict[tuple[str, str], FieldRef] = {}
        narrative_levels: dict[tuple[str, str], FieldRef] = {}
        for r in refs:
            f = r.field
            if f.measure != "level" or not f.period_key or f.data_type not in ("monetary", "per_share", "shares"):
                continue
            if f.reported_scale == "unknown":
                continue
            key = (f.metric, f.period_key)
            pool = table_levels if r.is_table else narrative_levels
            if key not in pool or r.tol < pool[key].tol:
                pool[key] = r  # type: ignore[index]

        def level(metric: str, period: str | None, *, allow_narrative: bool = False) -> FieldRef | None:
            if not period:
                return None
            return table_levels.get((metric, period)) or (narrative_levels.get((metric, period)) if allow_narrative else None)

        for r in refs:
            f = r.field
            if r.is_table or f.source_kind != "narrative" or not f.metric:
                continue
            period, prior = f.period_key, shift_period(f.period_key, -1)
            if f.measure == "level" and f.data_type in ("monetary", "per_share", "shares"):
                t = level(f.metric, period)
                if t is not None:
                    self._record("RULE_NARRATIVE_TABLE_001", statement="Narrative vs table", period=period,
                                 formula=f"narrative {f.metric} = table {f.metric}",
                                 expected=t.value or _ZERO, actual=r.value or _ZERO, tolerance=r.tol + t.tol,
                                 result=r, operands=[t], message=f"'{f.original_text}' vs table {f.metric}")
                elif f.metric == "net_cash":
                    cash, debt = level("cash", period), level("long_term_debt", period)
                    if cash and debt:
                        self._record("RULE_NARRATIVE_NETCASH_001", statement="Narrative vs table", period=period,
                                     formula="net_cash = cash - long_term_debt",
                                     expected=(cash.value or _ZERO) - (debt.value or _ZERO), actual=r.value or _ZERO,
                                     tolerance=r.tol + cash.tol + debt.tol, result=r, operands=[cash, debt],
                                     message=f"Net cash '{f.original_text}'")
            elif f.measure == "change_pct" and f.data_type == "percentage":
                cur, pri = level(f.metric, period), level(f.metric, prior)
                if cur and pri:
                    res = self._growth(cur, pri)
                    if res:
                        g, prop = res
                        self._record("RULE_NARRATIVE_GROWTH_001", statement="Narrative vs table", period=period,
                                     formula=f"{f.metric} growth = ({period} - {prior}) / |{prior}| x 100",
                                     expected=g, actual=r.value or _ZERO, tolerance=r.tol + prop, result=r,
                                     operands=[cur, pri], message=f"Growth '{f.original_text}' for {f.metric}")
            elif f.data_type == "percentage" and f.measure in ("level", "pct_of_revenue"):
                ratio = self._ratio_for(f.metric, f.measure, period, level)
                if ratio:
                    num, den, label = ratio
                    expected = (num.value or _ZERO) / (den.value or Decimal(1)) * _HUNDRED
                    prop = _HUNDRED * (num.tol / abs(den.value or Decimal(1)) + abs(num.value or _ZERO) * den.tol / ((den.value or Decimal(1)) ** 2))
                    self._record("RULE_NARRATIVE_RATIO_001", statement="Narrative vs table", period=period,
                                 formula=label, expected=expected, actual=r.value or _ZERO, tolerance=r.tol + prop,
                                 result=r, operands=[num, den], message=f"Ratio '{f.original_text}' for {f.metric}")
            elif f.measure == "change_bps" and f.metric in ("gross_margin", "operating_margin", "net_margin"):
                cur_r = self._ratio_for(f.metric, "level", period, level)
                pri_r = self._ratio_for(f.metric, "level", prior, level)
                if cur_r and pri_r:
                    # margins are quoted at the precision of the narrative level fact (e.g. 14.9%), so compare
                    # the bps change computed from margins rounded to one decimal, +/- one margin step
                    def margin(x: tuple[FieldRef, FieldRef, str]) -> Decimal:
                        return ((x[0].value or _ZERO) / (x[1].value or Decimal(1)) * _HUNDRED).quantize(Decimal("0.1"))

                    expected = (margin(cur_r) - margin(pri_r)) * _HUNDRED
                    self._record("RULE_NARRATIVE_BPS_001", statement="Narrative vs table", period=period,
                                 formula=f"{f.metric} change (bps) = (margin {period} - margin {prior}) x 100",
                                 expected=expected, actual=r.value or _ZERO, tolerance=r.tol + Decimal(10),
                                 result=r, operands=[cur_r[0], cur_r[1], pri_r[0], pri_r[1]],
                                 message=f"Margin change '{f.original_text}'")

    @staticmethod
    def _ratio_for(metric: str, measure: str | None, period: str | None, level) -> tuple[FieldRef, FieldRef, str] | None:
        numerators = {"gross_margin": "gross_profit", "operating_margin": "operating_income", "net_margin": "net_income"}
        rev = level("revenue", period)
        if rev is None:
            return None
        if measure == "pct_of_revenue":
            num = level(metric, period, allow_narrative=True)
            return (num, rev, f"{metric} / revenue x 100") if num and num is not rev else None
        if metric in numerators:
            num = level(numerators[metric], period)
            return (num, rev, f"{numerators[metric]} / revenue x 100") if num else None
        return None

    # ------------------------------------------------------------------ failure attribution
    def _assign_failures(self) -> None:
        for idx, chk in enumerate(self.checks):
            if chk["status"] != "FAILED":
                continue
            parts = self._participants[idx]
            worst = max(len(ref.in_failed) for ref, _ in parts)
            culprits = [ref for ref, _ in parts if len(ref.in_failed) == worst]
            if len(culprits) > 1:
                results = [ref for ref, role in parts if role == "result" and ref in culprits]
                culprits = results or culprits
            if chk["rule_id"] == "RULE_XTABLE_001":
                culprits = [ref for ref, _ in parts]  # a unit conflict: both sides are suspect
            for ref, _ in parts:
                issue = {
                    "rule_id": chk["rule_id"],
                    "rule_version": VERSION,
                    "severity": "ERROR" if ref in culprits else "WARNING",
                    "message": chk["message"],
                    "expected": chk["expected"],
                    "actual": chk["actual"],
                    "formula": chk["formula"],
                    "review_required": True,
                }
                if issue not in ref.field.validation_issues:
                    ref.field.validation_issues.append(issue)
                ref.field.review_required = True
                if ref in culprits:
                    ref.field.validation_status = "FAILED"
                elif ref.field.validation_status == "PASSED":
                    ref.field.validation_status = "WARNING"
        for idx, chk in enumerate(self.checks):
            if chk["status"] == "PASSED":
                for ref, _ in self._participants[idx]:
                    ref.field.validation_evidence.append(
                        {"rule_id": chk["rule_id"], "formula": chk["formula"], "status": "PASSED", "difference": chk["difference"]}
                    )


# --------------------------------------------------------------------------- confidence
_SCALE_SOURCE_SCORE = {"header": 1.0, "caption": 1.0, "line_above": 1.0, "inline": 1.0, "caller": 0.9,
                       "section_heading": 0.9, "document": 0.85}
_LABEL_SCORE = {"synonym": 1.0, "contains": 0.92, "embedding": 0.8, "none": 0.65}


def score_confidence(ref: FieldRef) -> None:
    """Confidence from evidence: scale explicitness, label match, validation outcome, OCR."""
    f = ref.field
    d = f.details or {}
    factors: dict[str, float] = {}
    if f.data_type in ("monetary", "monetary_change", "shares"):
        if f.reported_scale == "unknown":
            factors["scale"] = 0.35
        else:
            src = (d.get("unit") or {}).get("source") if str(d.get("scale_source", "")).startswith("table") else None
            if str(d.get("scale_source", "")).startswith("inline"):
                src = "inline"
            factors["scale"] = _SCALE_SOURCE_SCORE.get(src or "", 0.9)
    else:
        factors["scale"] = 1.0
    if f.source_kind == "table":
        factors["label"] = _LABEL_SCORE.get(((d.get("metric_match") or {}).get("method")) or "none", 0.65)
    elif f.source_kind == "narrative":
        factors["label"] = 0.95 if f.metric and f.metric in _known_metrics() else (0.8 if f.metric else 0.6)
    else:
        factors["label"] = 0.9 if f.row_label or f.field else 0.7
    if ref.checks_failed:
        factors["validation"] = 0.25 if f.validation_status == "FAILED" else 0.6
    elif ref.checks_passed:
        factors["validation"] = 1.0
    else:
        factors["validation"] = 0.85
    if f.validation_status == "FAILED" and not ref.checks_failed:
        factors["validation"] = 0.3  # failed a field-level rule
    method = ref.extraction_method or ""
    factors["ocr"] = float(ref.ocr_confidence) if method.startswith("ocr") and ref.ocr_confidence is not None else 1.0
    conf = 1.0
    for v in factors.values():
        conf *= v
    f.confidence = round(min(0.99, conf), 3)
    f.details = {**d, "confidence_factors": factors, "checks_passed": ref.checks_passed, "checks_failed": ref.checks_failed}
    if f.confidence < 0.6 or f.validation_status in ("FAILED", "REVIEW_REQUIRED"):
        f.review_required = True


def _known_metrics() -> set[str]:
    from app.taxonomy import METRICS_BY_NAME

    return set(METRICS_BY_NAME)
