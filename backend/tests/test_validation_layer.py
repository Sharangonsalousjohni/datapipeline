"""Comprehensive automated test suite for the Independent Financial Validation and Quality-Gating Layer.

Covers all 18 validation and gating scenarios:
1. Valid financial statement passes validation (ACCEPT).
2. Revenue subtotal discrepancy triggers FAILED check and HOLD_FOR_REVIEW.
3. Gross profit / operating income mismatch triggers failure.
4. Broken fundamental balance sheet equation triggers REJECT.
5. Parenthesized negative amount parsing (8,370) -> -8370.00 and signed_accounting_value.
6. Expense change magnitude delta semantics (+$475 COGS increase).
7. Single-application scale validation (no double-scaling).
8. Rounding tolerance behavior (diff <= tol passes; diff > tol fails).
9. Ambiguous / missing reporting periods handled safely.
10. Mixed currency/scale within a document handled cleanly (non-monetary metrics not scaled).
11. Summary KPI vs statement table value conflict triggers HOLD_FOR_REVIEW with both locations retained.
12. Missing optional fields handled without rejecting document.
13. Independence: High extraction confidence (1.0) with bad data fails validation.
14. Independence: Low extraction confidence (0.65) with sound data passes accounting validation.
15. Nonfinancial documents (rental agreement) do not receive irrelevant accounting checks.
16. JSON handoff backward-compatibility verification.
17. Malformed numeric strings and unparseable table cells marked as validation failures.
18. Deterministic reproducibility across repeated runs.
"""

from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
import pytest

from app.db import session_scope
from app.financial_extraction import build_structured_extraction
from app.models import Document, DocumentChunk, ClassificationResult
from app.schemas import (
    ExtractedFinancialField,
    StructuredExtractionItem,
    StructuredExtractionReport,
    ValidationSummary,
)
from app.validation import FinancialValidationEngine, QualityPolicy

_DIR = Path(__file__).parent.parent
FIN_DOC_ID = "4c78499ca5e64be3881cfeab033497ed"
RENTAL_DOC_ID = "24fd3ddc6989474687c13d70e932e89e"


# Helper to fetch chunks for fixture
def _get_document_chunk_dicts(doc_id: str) -> list[dict]:
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        if not doc:
            pytest.skip(f"Document {doc_id} not in DB")
        res = (
            session.query(DocumentChunk, ClassificationResult)
            .join(ClassificationResult, DocumentChunk.id == ClassificationResult.chunk_id)
            .filter(DocumentChunk.document_id == doc_id, ClassificationResult.is_current.is_(True))
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        return [
            {
                "chunk_id": c.id,
                "page_number": c.location.get("page") if isinstance(c.location, dict) else None,
                "section_heading": c.location.get("section_heading") if isinstance(c.location, dict) else None,
                "decision": cr.decision,
                "top_category": cr.top_category,
                "score": cr.score,
                "semantic_score": cr.semantic_score or 0.0,
                "signal_score": cr.signal_score or 0.0,
                "ocr_confidence": c.ocr_confidence or 1.0,
                "signals": cr.signals or [],
                "text": c.text,
                "location": c.location,
                "bboxes": c.location.get("bboxes") if isinstance(c.location, dict) else None,
            }
            for c, cr in res
        ]


# 1. Valid financial statement passes validation (ACCEPT)
def test_scenario_01_valid_financial_statement_passes():
    chunks = _get_document_chunk_dicts(FIN_DOC_ID)
    report = build_structured_extraction(FIN_DOC_ID, "Financial Statements & Performance Report.pdf", "FINANCIAL_RELEVANT", chunks)
    assert report.validation_summary.quality_gate_decision in ("ACCEPT", "ACCEPT_WITH_WARNINGS")
    assert report.validation_summary.failed_checks == 0
    # Verify primary accounting equations passed. Metric-mapped fields are checked by MetricValidator
    # (RULE_ARITH_*); the legacy RULE_ACCT_* checks only run for fields without a metric.
    bs_eq_rules = ("RULE_ACCT_BS_EQ_001", "RULE_ARITH_BS_BALANCE_001", "RULE_ARITH_BS_TA_TLE_001")
    bs_eq = [chk for chk in report.validation_summary.accounting_validations if chk.get("rule_id") in bs_eq_rules]
    assert len(bs_eq) >= 1
    assert all(chk["status"] == "PASSED" for chk in bs_eq)


# 2. Revenue subtotal discrepancy triggers FAILED check and HOLD_FOR_REVIEW
def test_scenario_02_revenue_subtotal_discrepancy():
    engine = FinancialValidationEngine()
    fields = [
        ExtractedFinancialField(
            field="product_revenue", original_text="$10,000", normalized_value="10000.00",
            row_label="Product Revenue", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="service_revenue", original_text="$4,000", normalized_value="4000.00",
            row_label="Service Revenue", reporting_period="Q3 2024"
        ),
        # Flawed total revenue: 15,000 instead of 14,000
        ExtractedFinancialField(
            field="total_revenues", original_text="$15,000", normalized_value="15000.00",
            row_label="Total Revenues", reporting_period="Q3 2024"
        ),
    ]
    item = StructuredExtractionItem(
        chunk_id="chk1", decision="RELEVANT", source_text="table", financial_fields=fields
    )
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    rev_chk = next(chk for chk in validated.validation_summary.accounting_validations if chk["rule_id"] == "RULE_ACCT_IS_REV_001")
    assert rev_chk["status"] == "FAILED"
    assert rev_chk["expected"] == "14000.00"
    assert rev_chk["actual"] == "15000.00"
    assert validated.validation_summary.quality_gate_decision == "HOLD_FOR_REVIEW"


# 3. Gross profit or operating income mismatch triggers failure
def test_scenario_03_gross_profit_mismatch():
    engine = FinancialValidationEngine()
    fields = [
        ExtractedFinancialField(
            field="total_revenues", original_text="$18,450", normalized_value="18450.00",
            row_label="Total Revenues", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="cogs", original_text="$5,875", normalized_value="5875.00",
            row_label="Cost of Goods Sold (COGS)", reporting_period="Q3 2024"
        ),
        # Flawed GP: 10,000 instead of 18450 - 5875 = 12575
        ExtractedFinancialField(
            field="gross_profit", original_text="$10,000", normalized_value="10000.00",
            row_label="Gross Profit", reporting_period="Q3 2024"
        ),
    ]
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="table", financial_fields=fields)
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    gp_chk = next(chk for chk in validated.validation_summary.accounting_validations if chk["rule_id"] == "RULE_ACCT_IS_GP_001")
    assert gp_chk["status"] == "FAILED"
    assert gp_chk["expected"] == "12575.00"
    assert gp_chk["actual"] == "10000.00"


# 4. Broken fundamental balance sheet equation triggers REJECT
def test_scenario_04_broken_balance_sheet_equation_rejects():
    engine = FinancialValidationEngine()
    fields = [
        ExtractedFinancialField(
            field="total_assets", original_text="$48,500", normalized_value="48500.00",
            row_label="TOTAL ASSETS", reporting_period="Q3 2024"
        ),
        # Broken Total Liabilities & Equity: 40,000 instead of 48,500
        ExtractedFinancialField(
            field="total_liabilities_equity", original_text="$40,000", normalized_value="40000.00",
            row_label="TOTAL LIABILITIES & EQUITY", reporting_period="Q3 2024"
        ),
    ]
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="table", financial_fields=fields)
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    assert validated.validation_summary.quality_gate_decision == "REJECT"
    assert any("fundamental equation" in r.lower() or "reject" in r.lower() for r in validated.validation_summary.gate_reasons)


# 5. Parenthesized negative amount parsing and signed accounting value
def test_scenario_05_parenthesized_negative_amount_and_signed_value():
    engine = FinancialValidationEngine()
    field = ExtractedFinancialField(
        field="net_interest",
        original_text="($8,370)",
        normalized_value="8370.00",
        normalized_amount="8370.00",
    )
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="text", financial_fields=[field])
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    vfield = validated.items[0].financial_fields[0]
    assert vfield.signed_accounting_value == "-8370.00"
    assert vfield.change_direction == "DECREASE"


# 6. Expense change magnitude delta semantics (+$475 COGS increase)
def test_scenario_06_expense_change_magnitude_delta():
    engine = FinancialValidationEngine()
    field = ExtractedFinancialField(
        field="cogs_change",
        original_text="+$475",
        normalized_value="475.00",
        row_label="Cost of Goods Sold (COGS)",
        data_type="monetary_change",
    )
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="text", financial_fields=[field])
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    vfield = validated.items[0].financial_fields[0]
    assert vfield.change_semantics == "MAGNITUDE_DELTA"
    assert vfield.change_direction == "INCREASE"
    assert vfield.change_basis == "period_over_period"


# 7. Single-application scale validation (no double scaling)
def test_scenario_07_single_application_scale_validation():
    engine = FinancialValidationEngine()
    # Double-scaled field: reported 14,200 with multiplier 1,000 but normalized_amount erroneously multiplied again to 14,200,000,000
    field = ExtractedFinancialField(
        field="total_revenue",
        original_text="$14,200",
        reported_value="14200.00",
        reported_scale="thousands",
        scale_multiplier=1000.0,
        normalized_amount="14200000000.00",  # Erroneous extra scaling
        normalized_value="14200.00",
    )
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="text", financial_fields=[field])
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    vfield = validated.items[0].financial_fields[0]
    assert vfield.validation_status == "FAILED"
    assert any("double scaling" in issue["message"].lower() for issue in vfield.validation_issues)


# 8. Rounding tolerance behavior (diff <= tol passes; diff > tol fails)
def test_scenario_08_rounding_tolerance_behavior():
    custom_policy = QualityPolicy(rounding_tolerance=Decimal("1.00"))
    engine = FinancialValidationEngine(policy=custom_policy)

    # Off by 0.50 (within 1.00 tolerance)
    fields = [
        ExtractedFinancialField(
            field="product_revenue", original_text="$10,000.00", normalized_value="10000.00",
            row_label="Product Revenue", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="service_revenue", original_text="$4,000.00", normalized_value="4000.00",
            row_label="Service Revenue", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="total_revenues", original_text="$14,000.50", normalized_value="14000.50",
            row_label="Total Revenues", reporting_period="Q3 2024"
        ),
    ]
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="table", financial_fields=fields)
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report, policy=custom_policy)
    rev_chk = next(chk for chk in validated.validation_summary.accounting_validations if chk["rule_id"] == "RULE_ACCT_IS_REV_001")
    assert rev_chk["status"] == "PASSED", "Difference within tolerance should pass"

    # Now test with strict tolerance 0.01
    strict_policy = QualityPolicy(rounding_tolerance=Decimal("0.01"))
    validated_strict = engine.validate_document(report, policy=strict_policy)
    rev_chk_strict = next(chk for chk in validated_strict.validation_summary.accounting_validations if chk["rule_id"] == "RULE_ACCT_IS_REV_001")
    assert rev_chk_strict["status"] == "FAILED", "Difference exceeding tolerance must fail"


# 9. Ambiguous or missing reporting periods handled safely
def test_scenario_09_ambiguous_period_handling():
    engine = FinancialValidationEngine()
    field = ExtractedFinancialField(
        field="general_expense",
        original_text="$1,500",
        normalized_value="1500.00",
        reporting_period=None,  # No reporting period
    )
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="text", financial_fields=[field])
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    assert validated.validation_summary.quality_gate_decision in ("ACCEPT", "ACCEPT_WITH_WARNINGS")


# 10. Mixed currency/scale within a document (percentages/bps not scaled)
def test_scenario_10_mixed_scale_protection_for_non_monetary():
    engine = FinancialValidationEngine()
    pct_field = ExtractedFinancialField(
        field="gross_margin_pct",
        original_text="68.2%",
        normalized_value="68.2",
        data_type="percentage",
        reported_scale="thousands",  # Erroneously attached table scale
    )
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="table", financial_fields=[pct_field])
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    vfield = validated.items[0].financial_fields[0]
    assert vfield.validation_status == "WARNING"
    assert any("non-monetary" in issue["message"].lower() for issue in vfield.validation_issues)


# 11. Conflicting summary KPI vs table value triggers HOLD_FOR_REVIEW with both locations
def test_scenario_11_cross_reference_conflict_triggers_review():
    engine = FinancialValidationEngine()
    summary_field = ExtractedFinancialField(
        field="summary_total_revenue",
        original_text="TOTAL REVENUE: $18,450",
        normalized_value="18450.00",
        normalized_amount="18450000.00",
        source_location={"page": 1, "section": "Executive Summary"},
    )
    table_field = ExtractedFinancialField(
        field="table_total_revenues",
        original_text="Total Revenues: $14,200",  # Conflicting value
        normalized_value="14200.00",
        normalized_amount="14200000.00",
        row_label="Total Revenues",
        reporting_period="Q3 2024",
        source_location={"page": 2, "section": "Condensed Statement of Operations"},
    )
    item1 = StructuredExtractionItem(chunk_id="c1", page_number=1, decision="RELEVANT", source_text="summary", financial_fields=[summary_field])
    item2 = StructuredExtractionItem(chunk_id="c2", page_number=2, decision="RELEVANT", source_text="table", financial_fields=[table_field])

    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item1, item2], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    assert validated.validation_summary.quality_gate_decision == "HOLD_FOR_REVIEW"
    assert len(validated.validation_summary.cross_reference_checks) > 0
    cross = validated.validation_summary.cross_reference_checks[0]
    assert cross["status"] == "FAILED"
    assert cross["summary_location"]["chunk_id"] == "c1"
    assert cross["table_location"]["chunk_id"] == "c2"


# 12. Missing optional fields handled without rejecting document
def test_scenario_12_missing_optional_fields_no_rejection():
    engine = FinancialValidationEngine()
    # Statement with only current assets, no software development row
    fields = [
        ExtractedFinancialField(
            field="cash", original_text="$5,000", normalized_value="5000.00",
            row_label="Cash and Cash Equivalents", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="ar", original_text="$3,000", normalized_value="3000.00",
            row_label="Accounts Receivable, net", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="prepaid", original_text="$1,000", normalized_value="1000.00",
            row_label="Prepaid expenses & other current assets", reporting_period="Q3 2024"
        ),
        ExtractedFinancialField(
            field="total_current_assets", original_text="$9,000", normalized_value="9000.00",
            row_label="Total Current Assets", reporting_period="Q3 2024"
        ),
    ]
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="table", financial_fields=fields)
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    assert validated.validation_summary.quality_gate_decision == "ACCEPT"


# 13. Independence proof: High extraction confidence (1.0) with flawed data fails validation
def test_scenario_13_high_confidence_bad_data_fails_validation():
    engine = FinancialValidationEngine()
    bad_field = ExtractedFinancialField(
        field="operating_income",
        original_text="$99,999",  # Hallucinated or erroneous
        normalized_value="99999.00",
        confidence=1.0,  # High extraction confidence
        row_label="Operating Income (EBIT)",
        reporting_period="Q3 2024",
    )
    gp_field = ExtractedFinancialField(
        field="gross_profit", original_text="$10,000", normalized_value="10000.00",
        row_label="Gross Profit", reporting_period="Q3 2024"
    )
    opex_field = ExtractedFinancialField(
        field="total_opex", original_text="$5,000", normalized_value="5000.00",
        row_label="Total Operating Expenses", reporting_period="Q3 2024"
    )
    item = StructuredExtractionItem(
        chunk_id="chk1", extraction_confidence=1.0, decision="RELEVANT", source_text="table",
        financial_fields=[gp_field, opex_field, bad_field]
    )
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    # Must fail validation despite extraction_confidence = 1.0
    assert validated.validation_summary.quality_gate_decision == "HOLD_FOR_REVIEW"
    ebit_chk = next(chk for chk in validated.validation_summary.accounting_validations if chk["rule_id"] == "RULE_ACCT_IS_OPINC_001")
    assert ebit_chk["status"] == "FAILED"


# 14. Independence proof: Low extraction confidence (0.65) with sound data passes accounting validation
def test_scenario_14_low_confidence_sound_data_passes_accounting():
    engine = FinancialValidationEngine()
    gp_field = ExtractedFinancialField(
        field="gross_profit", original_text="$10,000", normalized_value="10000.00",
        confidence=0.65, row_label="Gross Profit", reporting_period="Q3 2024"
    )
    opex_field = ExtractedFinancialField(
        field="total_opex", original_text="$6,000", normalized_value="6000.00",
        confidence=0.65, row_label="Total Operating Expenses", reporting_period="Q3 2024"
    )
    ebit_field = ExtractedFinancialField(
        field="operating_income", original_text="$4,000", normalized_value="4000.00",
        confidence=0.65, row_label="Operating Income (EBIT)", reporting_period="Q3 2024"
    )
    item = StructuredExtractionItem(
        chunk_id="chk1", extraction_confidence=0.65, decision="RELEVANT", source_text="table",
        financial_fields=[gp_field, opex_field, ebit_field]
    )
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    ebit_chk = next(chk for chk in validated.validation_summary.accounting_validations if chk["rule_id"] == "RULE_ACCT_IS_OPINC_001")
    assert ebit_chk["status"] == "PASSED"


# 15. Nonfinancial documents (rental agreement) do not receive irrelevant accounting checks
def test_scenario_15_rental_agreement_uses_contract_rules():
    chunks = _get_document_chunk_dicts(RENTAL_DOC_ID)
    report = build_structured_extraction(RENTAL_DOC_ID, "Residential Rental Agreement.pdf", "FINANCIAL_RELEVANT", chunks)
    assert report.document_type == "residential_rental_agreement"
    # No balance sheet equations
    bs_eq = [chk for chk in report.validation_summary.accounting_validations if chk.get("statement") == "Balance Sheet"]
    assert len(bs_eq) == 0
    # Rental terms check present
    rental_chk = [chk for chk in report.validation_summary.accounting_validations if chk.get("rule_id") == "RULE_RENTAL_TERMS_001"]
    assert len(rental_chk) == 1
    assert rental_chk[0]["status"] == "PASSED"
    assert report.validation_summary.quality_gate_decision == "ACCEPT"


# 16. JSON handoff backward-compatibility verification
def test_scenario_16_json_handoff_backward_compatibility():
    chunks = _get_document_chunk_dicts(FIN_DOC_ID)
    report = build_structured_extraction(FIN_DOC_ID, "Financial Statements & Performance Report.pdf", "FINANCIAL_RELEVANT", chunks)
    json_dump = report.model_dump_json()
    assert isinstance(json_dump, str)
    # Verify backward compatible fields exist in dict
    data = report.model_dump()
    assert "validation_summary" in data
    assert "quality_gate_decision" in data["validation_summary"]
    assert "accounting_validations" in data["validation_summary"]
    assert "monetary_values_checked" in data["validation_summary"]
    assert "failed_checks" in data["validation_summary"]


# 17. Malformed numeric strings and unparseable table cells marked as validation failures
def test_scenario_17_malformed_numeric_string_fails():
    engine = FinancialValidationEngine()
    corrupt_field = ExtractedFinancialField(
        field="corrupt_metric",
        original_text="N/A $$$ ,,,",
        normalized_value="0.00",
        row_label="Corrupted Row",
    )
    item = StructuredExtractionItem(chunk_id="chk1", decision="RELEVANT", source_text="corrupt", financial_fields=[corrupt_field])
    report = StructuredExtractionReport(
        document_id="test_doc", document_type="financial_statement", document_decision="FINANCIAL_RELEVANT",
        items=[item], validation_summary=ValidationSummary()
    )
    validated = engine.validate_document(report)
    vfield = validated.items[0].financial_fields[0]
    assert vfield.validation_status == "FAILED"
    assert vfield.review_required is True


# 18. Repeated deterministic execution produces identical validation results
def test_scenario_18_deterministic_reproducibility():
    chunks = _get_document_chunk_dicts(FIN_DOC_ID)
    report1 = build_structured_extraction(FIN_DOC_ID, "Financial Statements & Performance Report.pdf", "FINANCIAL_RELEVANT", copy.deepcopy(chunks))
    report2 = build_structured_extraction(FIN_DOC_ID, "Financial Statements & Performance Report.pdf", "FINANCIAL_RELEVANT", copy.deepcopy(chunks))

    dump1 = report1.model_dump(exclude={"timestamp", "generated_at"})
    dump2 = report2.model_dump(exclude={"timestamp", "generated_at"})

    assert dump1["validation_summary"]["quality_gate_decision"] == dump2["validation_summary"]["quality_gate_decision"]
    assert dump1["validation_summary"]["failed_checks"] == dump2["validation_summary"]["failed_checks"]
    assert len(dump1["validation_summary"]["accounting_validations"]) == len(dump2["validation_summary"]["accounting_validations"])
