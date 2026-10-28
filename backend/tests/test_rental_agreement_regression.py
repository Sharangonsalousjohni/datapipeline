"""Regression tests for Residential Rental Agreement processing pipeline.

Verifies fixes for:
- Monetary corruption ($2,400.00 USD not mangled into $240000USD)
- Correct rent and deposit amount/currency extraction
- Conditional late fee preservation ($75.00 or 5%, whichever is greater)
- Signature block classification as administrative_content
- Elimination of excessive UNCERTAIN classifications
- Document-level classification as FINANCIAL_RELEVANT
- Structured JSON output conforms to Section 7 schema
"""

from __future__ import annotations

from pathlib import Path
from decimal import Decimal
import pytest

from app.db import session_scope
from app.extraction.visual import PdfParser
from app.financial_extraction import (
    build_structured_extraction,
    extract_rental_agreement_fields,
    parse_decimal_safe,
)
from app.models import Document, DocumentChunk, ClassificationResult
from app.worker import process_document_pipeline
from conftest import ensure_fixture_document

_DIR = Path(__file__).parent.parent
RENTAL_PDF_PATH = _DIR / "data" / "uploads" / "ff5a04edca039b471f21ae026c71599f67427cc588fc27215d0b941feedc68bf.pdf"
RENTAL_DOC_ID = "24fd3ddc6989474687c13d70e932e89e"


def test_pdf_extraction_no_monetary_corruption():
    """Verify PyMuPDF table extraction preserves exact punctuation, decimals, and spaces."""
    assert RENTAL_PDF_PATH.exists(), f"Fixture missing: {RENTAL_PDF_PATH}"
    parser = PdfParser()
    doc = parser.parse(RENTAL_PDF_PATH)
    
    all_text = " ".join(b.text for b in doc.blocks)
    
    # Critical regression check: $2,400.00 USD must NOT become $240000USD
    assert "$240000USD" not in all_text, "Found corrupted value $240000USD in extracted PDF text!"
    assert "$2,400.00 USD" in all_text or "$2,400.00" in all_text, "Expected exact string '$2,400.00' not found!"
    
    # Verify table is ordered after heading
    headings = [i for i, b in enumerate(doc.blocks) if "RESIDENTIAL TENANCY AGREEMENT" in b.text]
    tables = [i for i, b in enumerate(doc.blocks) if b.block_type == "table"]
    assert headings and tables, "Missing heading or table blocks"
    assert headings[0] < tables[0], f"Table block ({tables[0]}) preceded document title ({headings[0]})"


def test_decimal_safe_parsing():
    """Verify decimal-safe numeric handling for monetary values."""
    dec, curr, valid = parse_decimal_safe("$2,400.00 USD")
    assert dec == Decimal("2400.00")
    assert curr == "USD"
    assert valid is True

    dec_fixed, curr_fixed, valid_fixed = parse_decimal_safe("$75.00")
    assert dec_fixed == Decimal("75.00")
    assert curr_fixed == "USD"
    assert valid_fixed is True

    dec_int, curr_int, valid_int = parse_decimal_safe("$75")
    assert dec_int == Decimal("75.00")
    assert curr_int == "USD"
    assert valid_int is True


def test_rental_agreement_structured_field_extraction():
    """Verify field extraction from rental agreement text blocks."""
    # Table summary snippet
    table_text = (
        "Landlord: [Landlord Full Legal Name] | Tenant: [Tenant Full Legal Name]\n"
        "Monthly Rent: [$2,400.00 USD] (Payable on the 1st day of | Security Deposit: [$2,400.00 USD] (Held in designated\n"
        "each month) | escrow)"
    )
    fields = extract_rental_agreement_fields(table_text)
    field_names = {f.field: f for f in fields}
    
    assert "monthly_rent" in field_names
    assert field_names["monthly_rent"].normalized_value == "2400.00"
    assert field_names["monthly_rent"].currency == "USD"
    assert field_names["monthly_rent"].frequency == "monthly"
    assert field_names["monthly_rent"].due_date == "1st day of each month"
    assert field_names["monthly_rent"].validation_status == "PASSED"

    assert "security_deposit" in field_names
    assert field_names["security_deposit"].normalized_value == "2400.00"
    assert field_names["security_deposit"].currency == "USD"
    assert field_names["security_deposit"].validation_status == "PASSED"

    # Rent & conditional late fee clause snippet
    clause_text = (
        "The agreed rental consideration for the Premises shall be [$2,400.00] per calendar month. "
        "Rent is due and payable on or before the first (1st) day of each month via electronic funds transfer (ACH). "
        "If full rent is not received by 11:59 PM on the fifth (5th) calendar day of the month, "
        "a late processing charge of [$75.00] or 5% of the unpaid balance, whichever is greater, shall apply."
    )
    c_fields = extract_rental_agreement_fields(clause_text)
    c_dict = {f.field: f for f in c_fields}

    # Verify conditional late fee preservation: both $75 and 5% and comparison condition preserved
    assert "late_fee_clause" in c_dict
    assert c_dict["late_fee_clause"].condition == "whichever is greater"
    assert c_dict["late_fee_clause"].details["fixed_amount"] == "75.00"
    assert c_dict["late_fee_clause"].details["percentage"] == "5"
    assert c_dict["late_fee_clause"].details["condition"] == "whichever is greater"

    assert "late_fee_amount" in c_dict
    assert c_dict["late_fee_amount"].normalized_value == "75.00"

    assert "late_fee_percentage" in c_dict
    assert c_dict["late_fee_percentage"].normalized_value == "5"


def test_rental_agreement_end_to_end_pipeline():
    """Verify full pipeline execution on the uploaded rental agreement document."""
    doc_id = ensure_fixture_document(RENTAL_DOC_ID, RENTAL_PDF_PATH, "Residential_Rental_Agreement.pdf")
    res = process_document_pipeline(doc_id, reextract=True)
    assert res["document_id"] == doc_id
    assert res["relevance_status"] == "FINANCIAL_RELEVANT"
    assert res["relevance_score"] >= 0.80

    with session_scope() as session:
        doc = session.get(Document, doc_id)
        assert doc is not None
        assert doc.relevance_status == "FINANCIAL_RELEVANT"

        chunks = (
            session.query(DocumentChunk, ClassificationResult)
            .join(ClassificationResult, DocumentChunk.id == ClassificationResult.chunk_id)
            .filter(DocumentChunk.document_id == doc_id, ClassificationResult.is_current.is_(True))
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        
        # Verify no chunks are UNCERTAIN
        uncertains = [c for c, r in chunks if r.decision == "UNCERTAIN"]
        assert len(uncertains) == 0, f"Found unexpected UNCERTAIN chunks: {[c.text[:50] for c in uncertains]}"

        # Verify Section 2 (Lease Term) classified specifically as contractual_financial_obligations
        lease_term_chunks = [c for c, r in chunks if "LEASE TERM AND OCCUPANCY" in c.text]
        assert len(lease_term_chunks) > 0
        lt_cr = next(r for chk, r in chunks if chk.id == lease_term_chunks[0].id)
        assert lt_cr.top_category == "contractual_financial_obligations", (
            f"Expected contractual_financial_obligations for Lease Term, got {lt_cr.top_category}"
        )
        assert lt_cr.decision == "RELEVANT"

        # Verify Section 3 (Rental Consideration) classified as rent_leases_recurring_payments
        rent_chunks = [c for c, r in chunks if "RENTAL CONSIDERATION AND DUE DATES" in c.text]
        assert len(rent_chunks) > 0
        rc_cr = next(r for chk, r in chunks if chk.id == rent_chunks[0].id)
        assert rc_cr.top_category == "rent_leases_recurring_payments"
        assert rc_cr.decision == "RELEVANT"

        # Verify Section 4 (Security Deposit) classified as deposits_and_escrow
        sec_chunks = [c for c, r in chunks if "SECURITY DEPOSIT PROTECTION" in c.text]
        assert len(sec_chunks) > 0
        sec_cr = next(r for chk, r in chunks if chk.id == sec_chunks[0].id)
        assert sec_cr.top_category in ("deposits_and_escrow", "rent_leases_recurring_payments")
        assert sec_cr.decision == "RELEVANT"

        # Verify signature blocks classified as administrative_content
        sig_chunks = [c for c, r in chunks if "EXECUTION AND ACKNOWLEDGMENT" in c.text or "AUTHORIZED SIGNATURE" in c.text]
        assert len(sig_chunks) > 0
        for c in sig_chunks:
            cr = next(r for chk, r in chunks if chk.id == c.id)
            assert cr.top_category == "administrative_content"
            assert cr.decision == "IRRELEVANT"

        # Verify maintenance chunk classified as non_financial_content
        maint_chunks = [c for c, r in chunks if "UTILITIES, UPKEEP, AND MAINTENANCE" in c.text]
        assert len(maint_chunks) > 0
        m_cr = next(r for chk, r in chunks if chk.id == maint_chunks[0].id)
        assert m_cr.top_category == "non_financial_content"
        assert m_cr.decision == "IRRELEVANT"

        # Verify structured extraction report
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
            }
            for c, r in chunks
        ]
        report = build_structured_extraction(doc.id, doc.filename, doc.relevance_status, chunk_dicts)
        
        assert report.document_decision == "FINANCIAL_RELEVANT"
        assert report.document_type == "residential_rental_agreement"
        assert report.validation_summary.monetary_values_checked is True
        assert report.validation_summary.failed_checks == 0
        assert report.validation_summary.requires_review is False

        # Verify four distinct scoring dimensions are preserved in structured extraction items
        for item in report.items:
            assert isinstance(item.relevance_score, float)
            assert isinstance(item.semantic_score, float)
            assert isinstance(item.signal_score, float)
            assert isinstance(item.extraction_confidence, float)
            if item.decision == "RELEVANT":
                assert item.relevance_score > 0.0

        # Verify all financial fields present with provenance and validation status
        all_fields = [f for item in report.items for f in item.financial_fields]
        field_by_name = {f.field: f for f in all_fields}
        assert "monthly_rent" in field_by_name
        assert field_by_name["monthly_rent"].normalized_value == "2400.00"
        assert field_by_name["monthly_rent"].validation_status == "PASSED"
        assert "Normalized" in field_by_name["monthly_rent"].validation_notes

        assert "security_deposit" in field_by_name
        assert field_by_name["security_deposit"].normalized_value == "2400.00"
        assert field_by_name["security_deposit"].validation_status == "PASSED"

        assert "late_fee_clause" in field_by_name
        assert field_by_name["late_fee_clause"].condition == "whichever is greater"
        assert field_by_name["late_fee_clause"].details["fixed_amount"] == "75.00"
        assert field_by_name["late_fee_clause"].details["percentage"] == "5"
        assert field_by_name["late_fee_clause"].details["percentage_base"] == "unpaid balance"
        assert field_by_name["late_fee_clause"].details["condition"] == "whichever is greater"
        assert field_by_name["late_fee_clause"].validation_status == "PASSED"
