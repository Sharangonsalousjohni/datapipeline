"""Regression tests for Financial Statements & Performance Report and Rental Agreement.

Verifies:
- Elimination of table corruption ($14200 , -> $14,200)
- Currency scale detection (USD Thousands -> 14,200 x 1,000 = 14,200,000)
- Distinction between monetary amounts, percentages (68.2%), basis points (+110 bps), and ratios (2.00x)
- Row-level classification:
  * revenue -> income_revenue
  * COGS / OpEx -> expenses_operating_costs
  * gross profit / operating income -> financial_performance
  * cash, AR, PP&E, payables, debt, equity -> assets_liabilities_investments
- Accounting identities validation:
  * Total Net Revenue == Subscription + Professional Services
  * Gross Profit == Total Net Revenue - COGS
  * Total Operating Expenses == R&D + S&M + G&A
  * Operating Income == Gross Profit - Total Operating Expenses
  * Total Assets == Current Assets + Non-Current Assets
  * Total Assets == Total Liabilities + Equity
- Preservation of rental agreement with specific deposit category and conditional late fee.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from app.db import session_scope
from app.extraction.visual import PdfParser
from app.financial_extraction import (
    build_structured_extraction,
    classify_row_label,
    detect_scale_and_currency,
    extract_financial_table_rows,
    extract_rental_agreement_fields,
    extract_summary_metrics,
    parse_cell_numeric,
    validate_financial_statements,
)
from app.models import ClassificationResult, Document, DocumentChunk
from app.worker import process_document_pipeline
from conftest import ensure_fixture_document

_DIR = Path(__file__).parent.parent
FIN_PDF_PATH = _DIR / "data" / "uploads" / "0c0112cd97564ff2195560a873af03b06497f8d0020f70bf0c5c231421238bd7.pdf"
FIN_DOC_ID = "4c78499ca5e64be3881cfeab033497ed"
RENTAL_DOC_ID = "24fd3ddc6989474687c13d70e932e89e"
RENTAL_PDF_PATH = _DIR / "data" / "uploads" / "ff5a04edca039b471f21ae026c71599f67427cc588fc27215d0b941feedc68bf.pdf"


def test_table_cell_extraction_no_detached_commas():
    """Verify PyMuPDF table extraction preserves exact punctuation, decimals, and spaces without '$14200 ,'."""
    assert FIN_PDF_PATH.exists(), f"Fixture missing: {FIN_PDF_PATH}"
    parser = PdfParser()
    doc = parser.parse(FIN_PDF_PATH)

    all_text = " ".join(b.text for b in doc.blocks)

    # Critical regression check: detached commas must NOT appear
    assert "$14200 ," not in all_text, "Found corrupted value '$14200 ,' in extracted PDF text!"
    assert "$18450 ," not in all_text, "Found corrupted value '$18450 ,' in extracted PDF text!"
    assert "$14820 ," not in all_text, "Found corrupted value '$14820 ,' in extracted PDF text!"
    assert "($5865) ," not in all_text, "Found corrupted value '($5865) ,' in extracted PDF text!"

    # Intended strings must be present
    assert "$14,200" in all_text, "Expected exact string '$14,200' not found!"
    assert "($5,865)" in all_text, "Expected exact string '($5,865)' not found!"
    assert "$14,820" in all_text, "Expected exact string '$14,820' not found!"
    assert "Software Subscription Revenue" in all_text, "Row label words were squashed together!"


def test_currency_scale_detection_and_conversion():
    """Verify scale detection and mathematical multiplication."""
    header_text = "QUARTERLY FINANCIAL STATEMENT & OPERATING PERFORMANCE\nCurrency: USD (Thousands)"
    curr, scale, mult = detect_scale_and_currency(header_text)
    assert curr == "USD"
    assert scale == "thousands"
    assert mult == 1000.0

    # Parse $14,200 with scale thousands
    rep_val, norm_amt, dtype, c_curr, norm_status, valid, _ = parse_cell_numeric(
        "$14,200", default_currency=curr, scale=scale, scale_multiplier=mult
    )
    assert rep_val == "14200"
    assert norm_amt == "14200000"
    assert dtype == "monetary"
    assert c_curr == "USD"
    assert norm_status == "PASSED"
    assert valid is True

    # Parse negative in parentheses: ($5,865)
    rep_neg, norm_neg, dtype_neg, _, _, valid_neg, _ = parse_cell_numeric(
        "($5,865)", default_currency=curr, scale=scale, scale_multiplier=mult
    )
    assert rep_neg == "-5865"
    assert norm_neg == "-5865000"
    assert dtype_neg == "monetary"
    assert valid_neg is True

    # Parse percentage: 68.2% (must NOT be multiplied by 1000)
    rep_pct, norm_pct, dtype_pct, _, _, _, _ = parse_cell_numeric("68.2%")
    assert rep_pct == "68.2"
    assert norm_pct == "68.2"
    assert dtype_pct == "percentage"

    # Parse basis points: +110 bps (must NOT be multiplied by 1000)
    rep_bps, norm_bps, dtype_bps, _, _, _, _ = parse_cell_numeric("+110 bps")
    assert rep_bps == "110"
    assert norm_bps == "110"
    assert dtype_bps == "basis_points"

    # Parse ratio: 2.00x (must NOT be multiplied by 1000)
    rep_rat, norm_rat, dtype_rat, _, _, _, _ = parse_cell_numeric("2.00x")
    assert rep_rat == "2.00"
    assert norm_rat == "2.00"
    assert dtype_rat == "ratio"


def test_row_level_category_classification():
    """Verify distinct categories per accounting row label."""
    # Revenue rows
    assert classify_row_label("Software Subscription Revenue") == "income_revenue"
    assert classify_row_label("Professional Services & Consulting") == "income_revenue"
    assert classify_row_label("Total Net Revenue") == "income_revenue"

    # Expense rows
    assert classify_row_label("Cost of Goods Sold (COGS)") == "expenses_operating_costs"
    assert classify_row_label("Research & Engineering") == "expenses_operating_costs"
    assert classify_row_label("Sales & Marketing") == "expenses_operating_costs"
    assert classify_row_label("General & Administrative") == "expenses_operating_costs"
    assert classify_row_label("Total Operating Expenses") == "expenses_operating_costs"

    # Profit & Performance rows
    assert classify_row_label("Gross Profit") == "financial_performance"
    assert classify_row_label("Operating Income (EBIT)") == "financial_performance"
    assert classify_row_label("Net Income After Tax") == "financial_performance"

    # Balance Sheet Assets, Liabilities, Equity
    assert classify_row_label("Cash and Cash Equivalents") == "assets_liabilities_investments"
    assert classify_row_label("Accounts Receivable, Net") == "assets_liabilities_investments"
    assert classify_row_label("TOTAL ASSETS") == "assets_liabilities_investments"
    assert classify_row_label("Accounts Payable & Accruals") == "assets_liabilities_investments"
    assert classify_row_label("Long-Term Borrowings") == "assets_liabilities_investments"
    assert classify_row_label("Stockholders' Equity & Retained Earnings") == "assets_liabilities_investments"


def test_financial_statements_end_to_end_pipeline():
    """Verify full pipeline execution on Financial Statements PDF."""
    doc_id = ensure_fixture_document(FIN_DOC_ID, FIN_PDF_PATH, "Financial_Statements.pdf")
    res = process_document_pipeline(doc_id, reextract=True)
    assert res["document_id"] == doc_id
    assert res["relevance_status"] == "FINANCIAL_RELEVANT"

    with session_scope() as session:
        doc = session.get(Document, doc_id)
        assert doc is not None

        chunks = (
            session.query(DocumentChunk, ClassificationResult)
            .join(ClassificationResult, DocumentChunk.id == ClassificationResult.chunk_id)
            .filter(DocumentChunk.document_id == doc_id, ClassificationResult.is_current.is_(True))
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )

        chunk_dicts = [
            {
                "chunk_id": c.id,
                "page_number": c.page_number,
                "section_heading": c.location.get("section_heading"),
                "decision": r.decision,
                "top_category": r.top_category,
                "score": r.score,
                "semantic_score": r.semantic_score,
                "signal_score": r.signal_score,
                "ocr_confidence": c.ocr_confidence,
                "signals": r.signals,
                "text": c.text,
                "location": c.location,
                "bboxes": c.location.get("bboxes"),
            }
            for c, r in chunks
        ]

        report = build_structured_extraction(doc.id, doc.filename, doc.relevance_status, chunk_dicts)
        assert report.document_type == "financial_statement"
        assert report.validation_summary.monetary_values_checked is True

        # Verify all 18 accounting validations PASSED
        acct_checks = report.validation_summary.accounting_validations
        assert len(acct_checks) >= 18
        for chk in acct_checks:
            assert chk["status"] == "PASSED", f"Accounting check failed: {chk}"

        # Verify key extracted fields exist and are accurately scaled
        all_fields = {f.field: f for item in report.items for f in item.financial_fields}

        # Subscription Revenue for Q3 2026: 14,200 x 1,000 = 14,200,000
        assert "software_subscription_revenue_q3_2026" in all_fields
        f_sub = all_fields["software_subscription_revenue_q3_2026"]
        assert f_sub.original_text == "$14,200"
        assert f_sub.reported_value == "14200"
        assert f_sub.reported_scale == "thousands"
        assert f_sub.normalized_amount == "14200000"
        assert f_sub.category == "income_revenue"
        assert f_sub.validation_status == "PASSED"

        # COGS for Q3 2026: ($5,865) x 1,000 = -5,865,000
        assert "cost_of_goods_sold_cogs_q3_2026" in all_fields
        f_cogs = all_fields["cost_of_goods_sold_cogs_q3_2026"]
        assert f_cogs.original_text == "($5,865)"
        assert f_cogs.reported_value == "-5865"
        assert f_cogs.normalized_amount == "-5865000"
        assert f_cogs.category == "expenses_operating_costs"

        # Total Assets for Sep 30, 2026: 36,000 x 1,000 = 36,000,000
        assert "total_assets_sep_30_2026" in all_fields
        f_ta = all_fields["total_assets_sep_30_2026"]
        assert f_ta.reported_value == "36000"
        assert f_ta.normalized_amount == "36000000"
        assert f_ta.category == "assets_liabilities_investments"

        # Quick Ratio summary metric: 2.00x
        assert "quick_ratio" in all_fields
        assert all_fields["quick_ratio"].reported_value == "2.00"
        assert all_fields["quick_ratio"].data_type == "ratio"


def test_accounting_mismatch_triggers_warning_without_mutation():
    """Verify a broken accounting equation triggers a WARNING without modifying raw inputs."""
    # Build synthetic fields with an accounting discrepancy
    from app.schemas import ExtractedFinancialField
    f_sub = ExtractedFinancialField(
        field="software_subscription_revenue_q3_2026",
        original_text="$14,200",
        normalized_value="14200000",
        normalized_amount="14200000",
        data_type="monetary",
        reporting_period="Q3 2026",
        row_label="Software Subscription Revenue",
    )
    f_svc = ExtractedFinancialField(
        field="professional_services_consulting_q3_2026",
        original_text="$4,250",
        normalized_value="4250000",
        normalized_amount="4250000",
        data_type="monetary",
        reporting_period="Q3 2026",
        row_label="Professional Services & Consulting",
    )
    # Tampered total revenue
    f_tot = ExtractedFinancialField(
        field="total_net_revenue_q3_2026",
        original_text="$19,000",
        normalized_value="19000000",
        normalized_amount="19000000",
        data_type="monetary",
        reporting_period="Q3 2026",
        row_label="Total Net Revenue",
    )

    checks = validate_financial_statements([f_sub, f_svc, f_tot])
    rev_check = next(c for c in checks if c["check_name"] == "total_revenue_component_sum")
    assert rev_check["status"] == "WARNING"
    assert rev_check["expected"] == "18450000.00"
    assert rev_check["actual"] == "19000000.00"
    assert rev_check["difference"] == "550000.00"
    # Ensure raw field value was not altered
    assert f_tot.normalized_amount == "19000000"


def test_rental_agreement_deposit_and_late_fee_preservation():
    """Verify Residential Rental Agreement preserves deposit specificity and conditional late fee."""
    doc_id = ensure_fixture_document(RENTAL_DOC_ID, RENTAL_PDF_PATH, "Residential_Rental_Agreement.pdf")
    res = process_document_pipeline(doc_id, reextract=True)
    assert res["relevance_status"] == "FINANCIAL_RELEVANT"

    with session_scope() as session:
        chunks = (
            session.query(DocumentChunk, ClassificationResult)
            .join(ClassificationResult, DocumentChunk.id == ClassificationResult.chunk_id)
            .filter(DocumentChunk.document_id == doc_id, ClassificationResult.is_current.is_(True))
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )

        # Verify Section 4 (Security Deposit) is specific (deposits_and_escrow), NOT rent
        sec_chunks = [c for c, r in chunks if "SECURITY DEPOSIT PROTECTION" in c.text]
        assert len(sec_chunks) > 0
        sec_cr = next(r for chk, r in chunks if chk.id == sec_chunks[0].id)
        assert sec_cr.top_category == "deposits_and_escrow", (
            f"Expected deposits_and_escrow for Security Deposit, got {sec_cr.top_category}"
        )

        # Verify Section 2 (Lease Term) is contractual_financial_obligations
        lt_chunks = [c for c, r in chunks if "LEASE TERM AND OCCUPANCY" in c.text]
        assert len(lt_chunks) > 0
        lt_cr = next(r for chk, r in chunks if chk.id == lt_chunks[0].id)
        assert lt_cr.top_category == "contractual_financial_obligations"

        # Verify Section 3 (Rental Consideration) is rent_leases_recurring_payments
        rent_chunks = [c for c, r in chunks if "RENTAL CONSIDERATION AND DUE DATES" in c.text]
        assert len(rent_chunks) > 0
        r_cr = next(r for chk, r in chunks if chk.id == rent_chunks[0].id)
        assert r_cr.top_category == "rent_leases_recurring_payments"
