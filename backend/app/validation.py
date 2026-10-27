"""Independent Financial Validation and Quality-Gating Layer.

Evaluates extracted financial fields and statements completely independently from the extraction stage.
Provides deterministic accounting consistency checks, sign and change-column semantics validation,
single-application currency/scale checks, summary-to-table cross-referencing, and document-level
quality gating decisions (ACCEPT, ACCEPT_WITH_WARNINGS, HOLD_FOR_REVIEW, REJECT).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from app.metric_validation import FieldRef, MetricValidator
from app.schemas import (
    ExtractedFinancialField,
    StructuredExtractionItem,
    StructuredExtractionReport,
    TableRowClassification,
    ValidationSummary,
)

ValidationStatusT = Literal["PASSED", "WARNING", "FAILED", "NEEDS_REVIEW", "NOT_APPLICABLE"]
QualityGateDecisionT = Literal["ACCEPT", "ACCEPT_WITH_WARNINGS", "HOLD_FOR_REVIEW", "REJECT"]


class QualityPolicy:
    """Configurable quality policy for document evaluation."""

    def __init__(
        self,
        rounding_tolerance: Decimal = Decimal("0.01"),
        require_balance_sheet_balance: bool = True,
        max_acceptable_warnings: int = 2,
        allow_minor_subtotal_warning: bool = True,
    ):
        self.rounding_tolerance = rounding_tolerance
        self.require_balance_sheet_balance = require_balance_sheet_balance
        self.max_acceptable_warnings = max_acceptable_warnings
        self.allow_minor_subtotal_warning = allow_minor_subtotal_warning


class FinancialValidationEngine:
    """Independent deterministic financial validation and gating engine."""

    VERSION = "1.0.0"

    def __init__(self, policy: QualityPolicy | None = None):
        self.policy = policy or QualityPolicy()

    def validate_document(
        self,
        report: StructuredExtractionReport,
        policy: QualityPolicy | None = None,
    ) -> StructuredExtractionReport:
        """Execute independent validation across all items and fields in a structured extraction report.
        
        Evaluates:
        1. Field-level parsing and integrity.
        2. Accounting signs and change column semantics.
        3. Currency and scale single application.
        4. Cross-referencing between executive summary and statement tables.
        5. Mathematical statement consistency (Income Statement & Balance Sheet).
        6. Overall quality gate decision.
        """
        active_policy = policy or self.policy
        audit_trail: list[dict[str, Any]] = []
        cross_ref_checks: list[dict[str, Any]] = []
        accounting_checks: list[dict[str, Any]] = []

        all_fields: list[ExtractedFinancialField] = []
        for item in report.items:
            all_fields.extend(item.financial_fields)
        # Fields mapped to a canonical metric use the metric validator; the label-based legacy
        # checks below only see fields without one, so no check runs twice.
        legacy_fields = [f for f in all_fields if not f.metric]

        # 1. Field-level validation: Integrity, Signs & Change Semantics, Scale & Currency
        for field in all_fields:
            self._validate_field_integrity(field, audit_trail)
            self._validate_change_semantics(field, audit_trail)
            self._validate_scale_and_currency(field, audit_trail)

        # 2. Cross-reference summary KPIs against table items
        cross_ref_checks = self._cross_check_summary_and_tables(report.items, audit_trail)

        # 3. Mathematical statement consistency checks (only for financial statements)
        if report.document_type in ("financial_statement", "general_financial_document") or any(
            item.table_rows for item in report.items
        ):
            accounting_checks = self._validate_accounting_statements(
                legacy_fields, active_policy, audit_trail
            )

        # 3b. Metric-level checks: cross-table, statement arithmetic, change %, narrative vs table,
        #     then evidence-based confidence for every field.
        refs = [
            FieldRef(field=f, group=item.chunk_id, extraction_method=item.extraction_method,
                     ocr_confidence=item.extraction_confidence)
            for item in report.items
            for f in item.financial_fields
        ]
        metric_checks = MetricValidator().run(refs)
        audit_trail.extend(metric_checks)
        cross_ids = ("RULE_XTABLE_", "RULE_NARRATIVE_")
        accounting_checks.extend(c for c in metric_checks if not c["rule_id"].startswith(cross_ids))
        cross_ref_checks.extend(c for c in metric_checks if c["rule_id"].startswith(cross_ids))

        # 4. Rental agreement specific checks
        if report.document_type == "residential_rental_agreement":
            rental_checks = self._validate_rental_agreement(all_fields, audit_trail)
            accounting_checks.extend(rental_checks)

        # 5. Quality gate evaluation
        gate_decision, gate_reasons = self._evaluate_quality_gate(
            report, all_fields, accounting_checks, cross_ref_checks, active_policy
        )

        # Update report items requiring review
        for item in report.items:
            has_failed_fields = any(
                f.validation_status in ("FAILED", "NEEDS_REVIEW", "REVIEW_REQUIRED") or f.review_required
                for f in item.financial_fields
            )
            if has_failed_fields or (item.decision == "UNCERTAIN"):
                item.requires_review = True

        # Build validation summary
        failed_count = sum(1 for f in all_fields if f.validation_status == "FAILED")
        failed_count += sum(1 for chk in accounting_checks if chk.get("status") == "FAILED")
        warning_count = sum(1 for f in all_fields if f.validation_status == "WARNING")
        warning_count += sum(1 for chk in accounting_checks if chk.get("status") == "WARNING")
        needs_review_count = sum(1 for f in all_fields if f.validation_status in ("NEEDS_REVIEW", "REVIEW_REQUIRED"))
        failed_count += sum(1 for chk in cross_ref_checks if chk.get("status") == "FAILED" and chk.get("rule_id", "").startswith(cross_ids))

        requires_review = (
            gate_decision in ("HOLD_FOR_REVIEW", "REJECT")
            or failed_count > 0
            or needs_review_count > 0
        )

        details = [
            f"Quality Gate: {gate_decision}. Evaluated {len(all_fields)} fields across {len(report.items)} chunks.",
            f"Results: {failed_count} failed, {warning_count} warnings, {needs_review_count} needs review.",
            *gate_reasons,
        ]

        report.validation_summary = ValidationSummary(
            monetary_values_checked=len(all_fields) > 0,
            failed_checks=failed_count,
            requires_review=requires_review,
            details=details,
            accounting_validations=accounting_checks,
            quality_gate_decision=gate_decision,
            gate_reasons=gate_reasons,
            cross_reference_checks=cross_ref_checks,
            rule_audit_trail=audit_trail,
        )

        return report

    # -----------------------------------------------------------------------
    # 1. Field Integrity
    # -----------------------------------------------------------------------
    def _validate_field_integrity(
        self, field: ExtractedFinancialField, audit_trail: list[dict[str, Any]]
    ) -> None:
        """Validate numeric parse integrity and uncorrupted values."""
        rule_id = "RULE_PARSE_INTEGRITY_001"
        now_ts = datetime.now(timezone.utc).isoformat()

        # If original text is unparseable or corrupted
        if field.normalized_value in ("0.00", "0") and field.original_text and not any(c.isdigit() for c in field.original_text):
            issue = {
                "rule_id": rule_id,
                "rule_version": self.VERSION,
                "severity": "ERROR",
                "message": f"Field '{field.field}' contains corrupted text '{field.original_text}' that yielded no digits",
                "expected": "Valid numeric representation",
                "actual": field.original_text,
                "review_required": True,
            }
            field.validation_issues.append(issue)
            field.validation_status = "FAILED"
            field.normalization_status = "FAILED"
            field.review_required = True
            audit_trail.append({
                "rule_id": rule_id,
                "rule_version": self.VERSION,
                "field": field.field,
                "status": "FAILED",
                "message": issue["message"],
                "timestamp": now_ts,
            })
            return

        # Ambiguous comma/decimal check: e.g. text like "$14200 ," or orphan commas
        if re.search(r",\s*$", field.original_text) or re.search(r"\b\d{5,}\s*,", field.original_text):
            issue = {
                "rule_id": rule_id,
                "rule_version": self.VERSION,
                "severity": "WARNING",
                "message": f"Suspicious punctuation or unnormalized spacing in original text: '{field.original_text}'",
                "expected": "Standard numeric formatting",
                "actual": field.original_text,
                "review_required": True,
            }
            field.validation_issues.append(issue)
            if field.validation_status == "PASSED":
                field.validation_status = "WARNING"
            field.review_required = True

    # -----------------------------------------------------------------------
    # 2. Accounting Signs & Change Semantics
    # -----------------------------------------------------------------------
    def _validate_change_semantics(
        self, field: ExtractedFinancialField, audit_trail: list[dict[str, Any]]
    ) -> None:
        """Validate parenthesized negative numbers and change column semantics."""
        rule_id = "RULE_CHANGE_SIGN_001"
        now_ts = datetime.now(timezone.utc).isoformat()
        orig = field.original_text.strip()

        # Check for parenthesized negative: e.g. "(8,370)", "($510)", "(5%)"
        is_parenthesized = (orig.startswith("(") and orig.endswith(")")) or " (" in orig
        is_explicit_minus = orig.startswith("-") or " -" in orig
        is_explicit_plus = orig.startswith("+") or " +" in orig

        # Parse normalized amount or value
        val_str = field.normalized_amount or field.normalized_value or "0"
        try:
            val_dec = Decimal(val_str)
        except InvalidOperation:
            val_dec = Decimal("0")

        # Set signed_accounting_value
        if is_parenthesized or is_explicit_minus:
            signed_dec = -abs(val_dec)
        else:
            signed_dec = val_dec
        field.signed_accounting_value = f"{signed_dec:.2f}"

        # Determine change semantics if field indicates change / variance / growth
        is_change_field = (
            "change" in (field.field or "").lower()
            or "variance" in (field.field or "").lower()
            or "delta" in (field.field or "").lower()
            or "bps" in orig.lower()
            or is_explicit_plus
            or (is_explicit_minus and field.data_type in ("monetary_change", "percentage", "basis_points"))
            or field.data_type == "monetary_change"
            or is_parenthesized
        )

        if is_change_field:
            if field.data_type == "percentage" or "%" in orig:
                field.change_semantics = "PERCENTAGE_CHANGE"
            elif field.data_type == "basis_points" or "bps" in orig.lower():
                field.change_semantics = "BASIS_POINTS_DELTA"
            elif "cogs" in (field.row_label or "").lower() or "expense" in (field.row_label or "").lower() or "cost" in (field.row_label or "").lower():
                # For expenses, "+$475" is a magnitude delta (expense increased)
                field.change_semantics = "MAGNITUDE_DELTA"
            else:
                field.change_semantics = "SIGNED_VALUE_DELTA"

            # Direction (narrative facts already carry the direction stated in the text: "grew", "declined")
            if field.source_kind == "narrative" and field.change_direction:
                pass
            elif is_parenthesized or is_explicit_minus or (signed_dec < 0):
                field.change_direction = "DECREASE"
            elif is_explicit_plus or (signed_dec > 0):
                field.change_direction = "INCREASE"
            else:
                field.change_direction = "FLAT"

            field.change_value = field.signed_accounting_value
            field.change_basis = "period_over_period"

            evidence = {
                "rule_id": rule_id,
                "rule_version": self.VERSION,
                "change_semantics": field.change_semantics,
                "change_direction": field.change_direction,
                "signed_accounting_value": field.signed_accounting_value,
                "original_text": field.original_text,
                "timestamp": now_ts,
            }
            field.validation_evidence.append(evidence)
        else:
            if is_parenthesized or is_explicit_minus:
                field.change_direction = "DECREASE"
                field.change_semantics = "SIGNED_VALUE_DELTA"
            field.change_basis = "not_applicable"

    # -----------------------------------------------------------------------
    # 3. Currency and Scale Single-Application
    # -----------------------------------------------------------------------
    def _validate_scale_and_currency(
        self, field: ExtractedFinancialField, audit_trail: list[dict[str, Any]]
    ) -> None:
        """Validate currency and scale are applied exactly once and audit trail is preserved."""
        rule_id = "RULE_SCALE_UNIT_001"
        now_ts = datetime.now(timezone.utc).isoformat()

        # Non-monetary metrics (percentages, basis points, ratios) must never have currency scale applied
        if field.data_type in ("percentage", "basis_points", "ratio"):
            if field.reported_scale and field.reported_scale != "units":
                issue = {
                    "rule_id": rule_id,
                    "rule_version": self.VERSION,
                    "severity": "WARNING",
                    "message": f"Non-monetary metric '{field.field}' ({field.data_type}) had reported_scale '{field.reported_scale}'",
                    "expected": "units / no scaling",
                    "actual": field.reported_scale,
                    "review_required": True,
                }
                field.validation_issues.append(issue)
                field.validation_status = "WARNING"
                field.review_required = True
            return

        if not field.reported_scale or field.reported_scale == "units":
            return

        # If reported_scale is thousands or millions, verify normalized_amount == reported_value * multiplier
        if field.reported_value and field.normalized_amount and field.scale_multiplier:
            try:
                rep_dec = Decimal(field.reported_value.replace(",", ""))
                mult_dec = Decimal(str(field.scale_multiplier))
                expected_norm = rep_dec * mult_dec
                actual_norm = Decimal(field.normalized_amount)

                diff = abs(expected_norm - actual_norm)
                if diff > Decimal("0.01"):
                    issue = {
                        "rule_id": rule_id,
                        "rule_version": self.VERSION,
                        "severity": "ERROR",
                        "message": f"Double scaling or scale multiplier mismatch for {field.field}: expected {expected_norm}, got {actual_norm}",
                        "expected": f"{expected_norm:.2f}",
                        "actual": f"{actual_norm:.2f}",
                        "review_required": True,
                    }
                    field.validation_issues.append(issue)
                    field.validation_status = "FAILED"
                    field.normalization_status = "FAILED"
                    field.review_required = True
                else:
                    field.validation_evidence.append({
                        "rule_id": rule_id,
                        "rule_version": self.VERSION,
                        "formula": "normalized_amount == reported_value * scale_multiplier",
                        "reported_value": field.reported_value,
                        "scale_multiplier": field.scale_multiplier,
                        "expected": f"{expected_norm:.2f}",
                        "actual": f"{actual_norm:.2f}",
                        "status": "PASSED",
                        "timestamp": now_ts,
                    })
            except InvalidOperation:
                pass

    # -----------------------------------------------------------------------
    # 4. Cross-Reference Summary KPIs vs Table Line Items
    # -----------------------------------------------------------------------
    def _cross_check_summary_and_tables(
        self, items: list[StructuredExtractionItem], audit_trail: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Cross-check executive summary KPI blocks against statement table items."""
        rule_id = "RULE_CROSS_REF_001"
        now_ts = datetime.now(timezone.utc).isoformat()
        cross_checks: list[dict[str, Any]] = []

        summary_fields: list[tuple[ExtractedFinancialField, str, int | None]] = []
        table_fields: list[tuple[ExtractedFinancialField, str, int | None]] = []

        for item in items:
            for field in item.financial_fields:
                if field.metric:
                    continue  # covered by the metric validator (RULE_XTABLE / RULE_NARRATIVE)
                is_summary = (
                    "summary" in (field.field or "").lower()
                    or "kpi" in (field.field or "").lower()
                    or item.section_heading and "summary" in item.section_heading.lower()
                    or not field.row_label
                )
                if is_summary:
                    summary_fields.append((field, item.chunk_id, item.page_number))
                if field.row_label:
                    table_fields.append((field, item.chunk_id, item.page_number))

        # Check key metrics: revenue, operating income, net income
        metric_matches = [
            ("total_revenue", ["total_revenue", "total_revenues", "revenue"]),
            ("operating_income", ["operating_income", "ebit"]),
            ("net_income", ["net_income", "net_profit"]),
        ]

        for canonical_name, aliases in metric_matches:
            # Find in summary
            s_cand = None
            s_chunk = None
            s_page = None
            for sf, sc, sp in summary_fields:
                f_norm = sf.field.lower().replace(" ", "_")
                if any(alias in f_norm for alias in aliases):
                    s_cand = sf
                    s_chunk = sc
                    s_page = sp
                    break

            # Find in table (latest period e.g. Q3 2024 / current period)
            t_cand = None
            t_chunk = None
            t_page = None
            for tf, tc, tp in table_fields:
                f_norm = (tf.row_label or tf.field).lower().replace(" ", "_")
                if any(alias in f_norm for alias in aliases):
                    # Prefer current/latest period
                    if not tf.reporting_period or "2024" in tf.reporting_period or "current" in tf.reporting_period.lower():
                        t_cand = tf
                        t_chunk = tc
                        t_page = tp
                        break

            if s_cand and t_cand:
                # Compare reported or normalized values
                s_val = s_cand.normalized_amount or s_cand.normalized_value
                t_val = t_cand.normalized_amount or t_cand.normalized_value

                try:
                    s_dec = Decimal(s_val)
                    t_dec = Decimal(t_val)
                    diff = abs(s_dec - t_dec)
                    status = "PASSED" if diff <= Decimal("0.01") else "FAILED"
                except InvalidOperation:
                    diff = Decimal("-1")
                    status = "WARNING"

                chk_result = {
                    "rule_id": rule_id,
                    "rule_version": self.VERSION,
                    "metric": canonical_name,
                    "status": status,
                    "summary_location": {
                        "chunk_id": s_chunk,
                        "page": s_page,
                        "text": s_cand.original_text,
                        "value": s_val,
                    },
                    "table_location": {
                        "chunk_id": t_chunk,
                        "page": t_page,
                        "text": t_cand.original_text,
                        "value": t_val,
                    },
                    "difference": f"{diff:.2f}",
                    "timestamp": now_ts,
                }
                cross_checks.append(chk_result)
                audit_trail.append(chk_result)

                if status == "FAILED":
                    s_cand.validation_issues.append({
                        "rule_id": rule_id,
                        "severity": "ERROR",
                        "message": f"Summary {canonical_name} ({s_val}) conflicts with Table {canonical_name} ({t_val})",
                        "expected": t_val,
                        "actual": s_val,
                        "review_required": True,
                    })
                    s_cand.validation_status = "FAILED"
                    s_cand.review_required = True

                    t_cand.validation_issues.append({
                        "rule_id": rule_id,
                        "severity": "ERROR",
                        "message": f"Table {canonical_name} ({t_val}) conflicts with Summary {canonical_name} ({s_val})",
                        "expected": s_val,
                        "actual": t_val,
                        "review_required": True,
                    })
                    t_cand.validation_status = "FAILED"
                    t_cand.review_required = True
                else:
                    s_cand.validation_evidence.append(chk_result)
                    t_cand.validation_evidence.append(chk_result)

        return cross_checks

    # -----------------------------------------------------------------------
    # 5. Deterministic Accounting Consistency Checks
    # -----------------------------------------------------------------------
    def _validate_accounting_statements(
        self,
        fields: list[ExtractedFinancialField],
        policy: QualityPolicy,
        audit_trail: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate Income Statement and Balance Sheet equations with configurable rounding tolerance."""
        now_ts = datetime.now(timezone.utc).isoformat()
        validations: list[dict[str, Any]] = []

        # Index fields by canonical label and reporting period
        by_label_period: dict[tuple[str, str], Decimal] = {}
        for f in fields:
            if not f.reporting_period:
                continue
            pk = re.sub(r"[^a-z0-9]+", "_", f.reporting_period.lower()).strip("_")
            val_str = f.normalized_amount or f.normalized_value
            try:
                dec_val = Decimal(val_str)
                # If negative is in original text
                if "(" in f.original_text and ")" in f.original_text:
                    dec_val = -abs(dec_val)
                if f.row_label:
                    lbl_k = re.sub(r"[^a-z0-9]+", "_", f.row_label.lower()).strip("_")
                    by_label_period[(lbl_k, pk)] = dec_val
                if f.field:
                    fld_k = re.sub(r"[^a-z0-9]+", "_", f.field.lower()).strip("_")
                    by_label_period[(fld_k, pk)] = dec_val
            except InvalidOperation:
                continue

        # Get unique reporting periods
        periods = sorted({f.reporting_period for f in fields if f.reporting_period and not f.reporting_period.lower().startswith("chg")})

        for prd in periods:
            pk = re.sub(r"[^a-z0-9]+", "_", prd.lower()).strip("_")

            # --- Income Statement ---
            # 1. Total Revenue = Product Revenue + Service Revenue
            prod_rev = (
                by_label_period.get(("product_revenue", pk))
                or by_label_period.get(("software_subscription_revenue", pk))
                or by_label_period.get(("subscription_revenue", pk))
            )
            serv_rev = (
                by_label_period.get(("service_revenue", pk))
                or by_label_period.get(("professional_services_consulting", pk))
                or by_label_period.get(("professional_services", pk))
            )
            tot_rev = (
                by_label_period.get(("total_revenues", pk))
                or by_label_period.get(("total_revenue", pk))
                or by_label_period.get(("total_net_revenue", pk))
                or by_label_period.get(("total_net_revenues", pk))
            )

            if prod_rev is not None and serv_rev is not None and tot_rev is not None:
                calc_rev = prod_rev + serv_rev
                diff = abs(calc_rev - tot_rev)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_IS_REV_001",
                    "rule_version": self.VERSION,
                    "statement": "Income Statement",
                    "period": prd,
                    "formula": "Total Revenues == Product Revenue + Service Revenue",
                    "expected": f"{calc_rev:.2f}",
                    "actual": f"{tot_rev:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Revenue component check: expected {calc_rev}, got {tot_rev} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 2. Gross Profit = Total Revenue - Cost of Goods Sold
            cogs = (
                by_label_period.get(("cost_of_goods_sold_cogs", pk))
                or by_label_period.get(("cost_of_goods_sold", pk))
                or by_label_period.get(("cogs", pk))
            )
            gp = by_label_period.get(("gross_profit", pk))
            if tot_rev is not None and cogs is not None and gp is not None:
                calc_gp = tot_rev - abs(cogs)
                diff = abs(calc_gp - gp)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_IS_GP_001",
                    "rule_version": self.VERSION,
                    "statement": "Income Statement",
                    "period": prd,
                    "formula": "Gross Profit == Total Revenues - Cost of Goods Sold",
                    "expected": f"{calc_gp:.2f}",
                    "actual": f"{gp:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Gross profit check: expected {calc_gp}, got {gp} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 3. Operating Expenses Subtotal = R&D + SG&A (or R&E + S&M + G&A)
            rnd = (
                by_label_period.get(("research_development", pk))
                or by_label_period.get(("research_engineering", pk))
                or by_label_period.get(("r_d", pk))
            )
            sm = by_label_period.get(("sales_marketing", pk))
            ga = by_label_period.get(("general_administrative", pk))
            sga = (
                by_label_period.get(("selling_general_administrative", pk))
                or by_label_period.get(("sga", pk))
            )
            tot_opex = (
                by_label_period.get(("total_operating_expenses", pk))
                or by_label_period.get(("operating_expenses", pk))
                or by_label_period.get(("total_opex", pk))
            )
            if tot_opex is not None:
                calc_opex = None
                if rnd is not None and sm is not None and ga is not None:
                    calc_opex = abs(rnd) + abs(sm) + abs(ga)
                elif rnd is not None and sga is not None:
                    calc_opex = abs(rnd) + abs(sga)

                if calc_opex is not None:
                    diff = abs(calc_opex - abs(tot_opex))
                    is_ok = diff <= policy.rounding_tolerance
                    status = "PASSED" if is_ok else "FAILED"
                    chk = {
                        "rule_id": "RULE_ACCT_IS_OPEX_001",
                        "rule_version": self.VERSION,
                        "statement": "Income Statement",
                        "period": prd,
                        "formula": "Total Operating Expenses == Research & Development + SG&A",
                        "expected": f"{calc_opex:.2f}",
                        "actual": f"{abs(tot_opex):.2f}",
                        "difference": f"{diff:.2f}",
                        "tolerance": f"{policy.rounding_tolerance:.2f}",
                        "status": status,
                        "message": f"Operating expenses check: expected {calc_opex}, got {abs(tot_opex)} (diff {diff})",
                        "timestamp": now_ts,
                    }
                    validations.append(chk)
                    audit_trail.append(chk)

            # 4. Operating Income (EBIT) = Gross Profit - Total Operating Expenses
            ebit = (
                by_label_period.get(("operating_income_ebit", pk))
                or by_label_period.get(("operating_income", pk))
                or by_label_period.get(("ebit", pk))
            )
            if gp is not None and tot_opex is not None and ebit is not None:
                calc_ebit = gp - abs(tot_opex)
                diff = abs(calc_ebit - ebit)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_IS_OPINC_001",
                    "rule_version": self.VERSION,
                    "statement": "Income Statement",
                    "period": prd,
                    "formula": "Operating Income (EBIT) == Gross Profit - Total Operating Expenses",
                    "expected": f"{calc_ebit:.2f}",
                    "actual": f"{ebit:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Operating income check: expected {calc_ebit}, got {ebit} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 5. Net Income = Operating Income - Interest Expense - Income Tax
            interest = (
                by_label_period.get(("interest_expense", pk))
                or by_label_period.get(("net_interest", pk))
                or by_label_period.get(("net_interest_expense", pk))
            )
            taxes = (
                by_label_period.get(("income_tax_expense", pk))
                or by_label_period.get(("income_taxes", pk))
                or by_label_period.get(("income_tax", pk))
            )
            net_inc = (
                by_label_period.get(("net_income_after_tax", pk))
                or by_label_period.get(("net_income", pk))
                or by_label_period.get(("net_profit", pk))
            )
            if ebit is not None and interest is not None and taxes is not None and net_inc is not None:
                calc_net = ebit - abs(interest) - abs(taxes)
                diff = abs(calc_net - net_inc)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_IS_NETINC_001",
                    "rule_version": self.VERSION,
                    "statement": "Income Statement",
                    "period": prd,
                    "formula": "Net Income == Operating Income - Interest Expense - Income Tax Expense",
                    "expected": f"{calc_net:.2f}",
                    "actual": f"{net_inc:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Net income check: expected {calc_net}, got {net_inc} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # --- Balance Sheet ---
            # 6. Total Current Assets = Cash + AR + Prepaid
            cash = (
                by_label_period.get(("cash_and_cash_equivalents", pk))
                or by_label_period.get(("cash", pk))
            )
            ar = (
                by_label_period.get(("accounts_receivable_net", pk))
                or by_label_period.get(("accounts_receivable", pk))
            )
            prepaid = (
                by_label_period.get(("prepaid_expenses_other_current_assets", pk))
                or by_label_period.get(("prepaid_expenses", pk))
            )
            tot_ca = by_label_period.get(("total_current_assets", pk))

            if cash is not None and ar is not None and prepaid is not None and tot_ca is not None:
                calc_ca = cash + ar + prepaid
                diff = abs(calc_ca - tot_ca)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_BS_CA_001",
                    "rule_version": self.VERSION,
                    "statement": "Balance Sheet",
                    "period": prd,
                    "formula": "Total Current Assets == Cash + Accounts Receivable + Prepaid Expenses",
                    "expected": f"{calc_ca:.2f}",
                    "actual": f"{tot_ca:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Current assets check: expected {calc_ca}, got {tot_ca} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 7. Total Assets = Current Assets + PP&E Net + Software Net
            ppe = (
                by_label_period.get(("property_plant_equipment_net", pk))
                or by_label_period.get(("property_plant_equipment", pk))
            )
            sw = (
                by_label_period.get(("capitalized_software_development", pk))
                or by_label_period.get(("capitalized_software", pk))
            )
            tot_assets = by_label_period.get(("total_assets", pk))

            if tot_ca is not None and ppe is not None and sw is not None and tot_assets is not None:
                calc_ta = tot_ca + ppe + sw
                diff = abs(calc_ta - tot_assets)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_BS_TA_001",
                    "rule_version": self.VERSION,
                    "statement": "Balance Sheet",
                    "period": prd,
                    "formula": "TOTAL ASSETS == Total Current Assets + Property, Plant & Equipment + Capitalized Software",
                    "expected": f"{calc_ta:.2f}",
                    "actual": f"{tot_assets:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Total assets check: expected {calc_ta}, got {tot_assets} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 8. Total Liabilities = Current Liabilities + Long-Term Borrowings
            tot_cl = by_label_period.get(("total_current_liabilities", pk))
            borrow = (
                by_label_period.get(("long_term_borrowings", pk))
                or by_label_period.get(("long_term_debt", pk))
            )
            tot_liab = by_label_period.get(("total_liabilities", pk))

            if tot_cl is not None and borrow is not None and tot_liab is not None:
                calc_tl = tot_cl + borrow
                diff = abs(calc_tl - tot_liab)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_BS_TL_001",
                    "rule_version": self.VERSION,
                    "statement": "Balance Sheet",
                    "period": prd,
                    "formula": "Total Liabilities == Total Current Liabilities + Long-Term Borrowings",
                    "expected": f"{calc_tl:.2f}",
                    "actual": f"{tot_liab:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Total liabilities check: expected {calc_tl}, got {tot_liab} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 9. Total Liabilities & Equity = Total Liabilities + Stockholders' Equity
            equity = (
                by_label_period.get(("stockholders_equity_retained_earnings", pk))
                or by_label_period.get(("stockholders_equity", pk))
                or by_label_period.get(("total_equity", pk))
            )
            tot_le = (
                by_label_period.get(("total_liabilities_equity", pk))
                or by_label_period.get(("total_liabilities_and_equity", pk))
                or by_label_period.get(("total_liabilities_and_stockholders_equity", pk))
            )

            if tot_liab is not None and equity is not None and tot_le is not None:
                calc_tle = tot_liab + equity
                diff = abs(calc_tle - tot_le)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_BS_EQUITY_001",
                    "rule_version": self.VERSION,
                    "statement": "Balance Sheet",
                    "period": prd,
                    "formula": "TOTAL LIABILITIES & EQUITY == Total Liabilities + Stockholders' Equity & Retained Earnings",
                    "expected": f"{calc_tle:.2f}",
                    "actual": f"{tot_le:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Total liabilities & equity check: expected {calc_tle}, got {tot_le} (diff {diff})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

            # 10. Fundamental Accounting Equation: TOTAL ASSETS == TOTAL LIABILITIES & EQUITY
            if tot_assets is not None and tot_le is not None:
                diff = abs(tot_assets - tot_le)
                is_ok = diff <= policy.rounding_tolerance
                status = "PASSED" if is_ok else "FAILED"
                chk = {
                    "rule_id": "RULE_ACCT_BS_EQ_001",
                    "rule_version": self.VERSION,
                    "statement": "Balance Sheet",
                    "period": prd,
                    "formula": "TOTAL ASSETS == TOTAL LIABILITIES & EQUITY",
                    "expected": f"{tot_assets:.2f}",
                    "actual": f"{tot_le:.2f}",
                    "difference": f"{diff:.2f}",
                    "tolerance": f"{policy.rounding_tolerance:.2f}",
                    "status": status,
                    "message": f"Balance sheet fundamental equation for {prd}: Assets ({tot_assets}) == Liabilities & Equity ({tot_le})",
                    "timestamp": now_ts,
                }
                validations.append(chk)
                audit_trail.append(chk)

        return validations

    # -----------------------------------------------------------------------
    # 6. Rental Agreement Verification
    # -----------------------------------------------------------------------
    def _validate_rental_agreement(
        self, fields: list[ExtractedFinancialField], audit_trail: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate core contractual obligations for rental agreements."""
        rule_id = "RULE_RENTAL_TERMS_001"
        now_ts = datetime.now(timezone.utc).isoformat()
        results: list[dict[str, Any]] = []

        has_rent = False
        has_deposit = False
        has_late_fee = False

        for f in fields:
            fname = (f.field or "").lower()
            if "monthly_rent" in fname or "rent" in fname:
                has_rent = True
            elif "security_deposit" in fname or "deposit" in fname:
                has_deposit = True
            elif "late_fee" in fname:
                has_late_fee = True

        chk = {
            "rule_id": rule_id,
            "rule_version": self.VERSION,
            "statement": "Rental Agreement Terms",
            "period": "Term",
            "formula": "Contains monthly_rent and security_deposit",
            "expected": "Rent and deposit fields present",
            "actual": f"rent={has_rent}, deposit={has_deposit}, late_fee={has_late_fee}",
            "status": "PASSED" if (has_rent and has_deposit) else "WARNING",
            "message": "Rental agreement essential financial terms check",
            "timestamp": now_ts,
        }
        results.append(chk)
        audit_trail.append(chk)
        return results

    # -----------------------------------------------------------------------
    # 7. Quality Gate Evaluator
    # -----------------------------------------------------------------------
    def _evaluate_quality_gate(
        self,
        report: StructuredExtractionReport,
        fields: list[ExtractedFinancialField],
        accounting_checks: list[dict[str, Any]],
        cross_ref_checks: list[dict[str, Any]],
        policy: QualityPolicy,
    ) -> tuple[QualityGateDecisionT, list[str]]:
        """Evaluate overall document quality and assign a gating decision."""
        reasons: list[str] = []

        # Count failures and warnings
        failed_fields = [f for f in fields if f.validation_status == "FAILED"]
        failed_acct = [chk for chk in accounting_checks if chk.get("status") == "FAILED"]
        failed_cross = [chk for chk in cross_ref_checks if chk.get("status") == "FAILED"]

        warning_fields = [f for f in fields if f.validation_status == "WARNING"]
        warning_acct = [chk for chk in accounting_checks if chk.get("status") == "WARNING"]
        needs_review_fields = [f for f in fields if f.validation_status in ("NEEDS_REVIEW", "REVIEW_REQUIRED")]

        # 1. Fundamental balance sheet breakage -> REJECT or HOLD_FOR_REVIEW
        broken_bs_equation = any(
            chk.get("rule_id") in ("RULE_ACCT_BS_EQ_001", "RULE_ARITH_BS_BALANCE_001") and chk.get("status") == "FAILED"
            for chk in accounting_checks
        )
        if broken_bs_equation and policy.require_balance_sheet_balance:
            reasons.append("REJECT: Balance sheet fundamental equation (Assets == Liabilities + Equity) failed")
            return "REJECT", reasons

        # 2. Corrupted or unparseable primary monetary fields -> REJECT
        corrupted_fields = [
            f for f in failed_fields
            if any(issue.get("rule_id") == "RULE_PARSE_INTEGRITY_001" and issue.get("severity") == "ERROR" for issue in f.validation_issues)
        ]
        if len(corrupted_fields) > 2:
            reasons.append(f"REJECT: Multiple corrupted or unparseable monetary fields ({len(corrupted_fields)})")
            return "REJECT", reasons

        # 3. Discrepancies between executive summary KPI and statement table -> HOLD_FOR_REVIEW
        if failed_cross:
            reasons.append(f"HOLD_FOR_REVIEW: {len(failed_cross)} cross-reference discrepancy between summary KPIs and table rows")
            return "HOLD_FOR_REVIEW", reasons

        # 4. Other accounting equation failures (e.g. revenue subtotal, gross profit) -> HOLD_FOR_REVIEW
        if failed_acct:
            reasons.append(f"HOLD_FOR_REVIEW: {len(failed_acct)} accounting equation checks failed")
            return "HOLD_FOR_REVIEW", reasons

        # 5. Field-level review requirements or warnings
        total_warnings = len(warning_fields) + len(warning_acct) + len(needs_review_fields)
        if total_warnings > policy.max_acceptable_warnings:
            reasons.append(f"HOLD_FOR_REVIEW: Warnings ({total_warnings}) exceeded policy threshold ({policy.max_acceptable_warnings})")
            return "HOLD_FOR_REVIEW", reasons

        if total_warnings > 0:
            reasons.append(f"ACCEPT_WITH_WARNINGS: Document accepted with {total_warnings} non-fatal warning(s)")
            return "ACCEPT_WITH_WARNINGS", reasons

        reasons.append("ACCEPT: All accounting equations, field validations, and cross-checks passed successfully")
        return "ACCEPT", reasons
