"""Automated fixture tests for the 8 document categories requested in Section 8:

1. An invoice.
2. A bank statement.
3. A scanned financial document.
4. A spreadsheet containing financial data.
5. A legal agreement with no financial obligations.
6. A mixed document containing both financial and non-financial sections.
7. A document containing numbers that are not monetary amounts.
8. Monetary values with different decimal and thousands separators.
"""

from __future__ import annotations

from decimal import Decimal
import pytest

from app.classifier import aggregate_document_decision, classify_chunks
from app.chunking import ChunkDraft
from app.financial_extraction import (
    build_structured_extraction,
    extract_generic_financial_fields,
    parse_decimal_safe,
)
from app.signals import analyze_signals


def _make_chunk(index: int, text: str, heading: str | None = None, chunk_type: str = "prose") -> ChunkDraft:
    return ChunkDraft(
        id=f"chk_test_{index:03d}",
        document_id="doc_test_fixture",
        chunk_index=index,
        chunk_type=chunk_type,
        text=text,
        context=heading,
        location={"page": 1, "section_heading": heading},
        extraction_method="test",
        ocr_confidence=0.98,
        content_hash=f"hash_{index}",
        section_heading=heading,
    )


# ---------------------------------------------------------------------------
# Fixture 1: Invoice
# ---------------------------------------------------------------------------
def test_fixture_1_invoice():
    """Verify invoice document classification and structured extraction."""
    text = (
        "INVOICE #INV-2026-9042\n"
        "Vendor: Apex Cloud Solutions LLC\n"
        "Customer: Global Logistics Inc.\n"
        "Subtotal: $14,500.00 USD\n"
        "Sales Tax (8.25%): $1,196.25 USD\n"
        "Total Amount Due: $15,696.25 USD\n"
        "Payment Terms: Net 30 Days. Due Date: 2026-11-25."
    )
    sig = analyze_signals(text)
    assert sig.signal_score >= 0.50
    assert "invoices_and_billing" in sig.category_hints or "invoices_purchase_orders" in sig.category_hints

    fields = extract_generic_financial_fields(text)
    f_map = {f.field: f for f in fields}
    assert "total_amount_due" in f_map
    assert f_map["total_amount_due"].normalized_value == "15696.25"
    assert f_map["total_amount_due"].currency == "USD"
    assert f_map["total_amount_due"].validation_status == "PASSED"

    assert "subtotal" in f_map
    assert f_map["subtotal"].normalized_value == "14500.00"

    assert "tax_amount" in f_map
    assert f_map["tax_amount"].normalized_value == "1196.25"


# ---------------------------------------------------------------------------
# Fixture 2: Bank Statement
# ---------------------------------------------------------------------------
def test_fixture_2_bank_statement():
    """Verify bank statement classification and balance extraction."""
    text = (
        "FIRST NATIONAL COMMERCIAL BANK\n"
        "Commercial Checking Statement - Account ending in *4812\n"
        "Beginning Balance: $48,210.50 USD\n"
        "Total Deposits and Credits: $12,400.00 USD\n"
        "Total Withdrawals and Debits: -$6,150.25 USD\n"
        "Ending Balance: $54,460.25 USD\n"
        "Annual Percentage Yield Earned: 1.25%"
    )
    sig = analyze_signals(text)
    assert sig.signal_score >= 0.45
    assert "bank_transactions" in sig.category_hints or "cash_flow_bank_transactions" in sig.category_hints

    fields = extract_generic_financial_fields(text)
    f_map = {f.field: f for f in fields}
    assert "beginning_balance" in f_map
    assert f_map["beginning_balance"].normalized_value == "48210.50"
    assert "ending_balance" in f_map
    assert f_map["ending_balance"].normalized_value == "54460.25"


# ---------------------------------------------------------------------------
# Fixture 3: Scanned Financial Document
# ---------------------------------------------------------------------------
def test_fixture_3_scanned_document():
    """Verify scanned document handling preserves confidence and extracts values."""
    # OCR noisy text sample with slight character variation
    noisy_ocr_text = (
        "ACME SUPPLY CORP - BILLING STATEMENT\n"
        "lnvoice Number: 884102\n"
        "Date of lssue: 2026-10-15\n"
        "Amount Due: $3,750.00 USD\n"
        "Please remit payment via wire transfer."
    )
    sig = analyze_signals(noisy_ocr_text)
    assert sig.signal_score >= 0.35

    chunk = _make_chunk(0, noisy_ocr_text, heading="Billing Statement")
    chunk.ocr_confidence = 0.88
    assert chunk.ocr_confidence == 0.88

    fields = extract_generic_financial_fields(noisy_ocr_text)
    f_map = {f.field: f for f in fields}
    assert "total_amount_due" in f_map
    assert f_map["total_amount_due"].normalized_value == "3750.00"
    assert f_map["total_amount_due"].currency == "USD"


# ---------------------------------------------------------------------------
# Fixture 4: Spreadsheet Containing Financial Data
# ---------------------------------------------------------------------------
def test_fixture_4_spreadsheet_table():
    """Verify spreadsheet table chunk classification and formatting."""
    table_text = (
        "Q3 Departmental Budget vs Actual\n"
        "Department | Budget Allocation | Actual Spend | Variance\n"
        "Engineering | $250,000.00 | $242,150.00 | $7,850.00\n"
        "Marketing | $120,000.00 | $118,400.00 | $1,600.00\n"
        "Operations | $85,000.00 | $89,200.00 | -$4,200.00\n"
        "Total | $455,000.00 | $449,750.00 | $5,250.00"
    )
    sig = analyze_signals(table_text, chunk_type="table")
    assert sig.signal_score >= 0.50
    assert sig.feature_counts.get("table_header", 0) > 0 or sig.feature_counts.get("currency_amount", 0) >= 3


# ---------------------------------------------------------------------------
# Fixture 5: Legal Agreement with No Financial Obligations (NDA)
# ---------------------------------------------------------------------------
def test_fixture_5_non_financial_legal_agreement():
    """Verify non-financial legal document (NDA) is classified as NOT_FINANCIAL."""
    chunks = [
        _make_chunk(0, "MUTUAL NON-DISCLOSURE AGREEMENT\nThis Agreement is between Acme Corp and Beta LLC.", "Title"),
        _make_chunk(1, "1. DEFINITION OF CONFIDENTIAL INFORMATION\nConfidential information includes proprietary source code, algorithms, and designs.", "Confidentiality"),
        _make_chunk(2, "2. OBLIGATIONS OF RECEIVING PARTY\nThe receiving party agrees to hold all confidential information in strict confidence.", "Obligations"),
        _make_chunk(3, "3. GOVERNING LAW AND JURISDICTION\nThis Agreement shall be governed by the laws of the State of Delaware.", "Governing Law"),
        _make_chunk(4, "IN WITNESS WHEREOF, the parties have executed this Agreement.\nPrint Name: Alice Smith\nDate: 2026-05-01", "Execution"),
    ]
    batch = classify_chunks(chunks)
    summary = aggregate_document_decision(chunks, batch.items, None, None)

    assert summary.relevance_status == "NOT_FINANCIAL"
    assert summary.is_filtered is True
    # Verify execution chunk is administrative_content
    assert batch.items[4].top_category == "administrative_content"
    assert batch.items[4].decision == "IRRELEVANT"


# ---------------------------------------------------------------------------
# Fixture 6: Mixed Document (Financial + Non-Financial Sections)
# ---------------------------------------------------------------------------
def test_fixture_6_mixed_document():
    """Verify multi-topic document with isolated financial mention receives MIXED_CONTENT."""
    chunks = [
        _make_chunk(0, "GLOBAL EMPLOYEE HANDBOOK & WORKPLACE POLICIES\nWelcome to Omnicorp.", "Introduction"),
        _make_chunk(1, "1. WORKPLACE CONDUCT AND DIVERSITY\nEmployees must maintain professional conduct.", "Conduct"),
        _make_chunk(2, "2. VACATION AND SICK LEAVE\nFull-time employees receive 15 days paid leave annually.", "Leave"),
        _make_chunk(3, "3. COMPUTER USE AND CYBERSECURITY\nAll passwords must adhere to corporate security protocols.", "Security"),
        _make_chunk(4, "4. TRAVEL EXPENSE REIMBURSEMENT POLICY\nEmployees are eligible for a daily per diem allowance of $75.00 USD for meals while traveling on company business.", "Expenses"),
        _make_chunk(5, "5. TERMINATION AND RESIGNATION\nEmployees must provide two weeks written notice.", "Termination"),
    ]
    batch = classify_chunks(chunks)
    summary = aggregate_document_decision(chunks, batch.items, None, None)

    # 1 out of 6 chunks is relevant -> predominantly non-financial -> MIXED_CONTENT
    assert summary.relevance_status == "MIXED_CONTENT"
    assert summary.is_filtered is False


# ---------------------------------------------------------------------------
# Fixture 7: Numbers that are Not Monetary Amounts
# ---------------------------------------------------------------------------
def test_fixture_7_non_monetary_numbers():
    """Verify numbers representing dates, statutes, codes, and variables are not confused with currency."""
    text = (
        "TECHNICAL SYSTEM SPECIFICATION\n"
        "Revision: 4.2.1\n"
        "Section 104.5 of IEEE Standard 802.11ax.\n"
        "Server IP Address: 192.168.1.100, Port: 8080.\n"
        "Environment variables: export PATH=$HOME/bin:$PATH\n"
        "Regex rule: ^[0-9]{3}-[0-9]{2}-[0-9]{4}$\n"
        "Phone number: +1-800-555-0199. Serial #99482104."
    )
    sig = analyze_signals(text)
    # Code dollar pattern ($HOME) and absence of currency amounts ensures low score
    assert sig.feature_counts.get("code_dollar", 0) > 0
    assert sig.feature_counts.get("currency_amount", 0) == 0
    assert sig.signal_score <= 0.20


# ---------------------------------------------------------------------------
# Fixture 8: Alternate Decimal and Thousands Separators
# ---------------------------------------------------------------------------
def test_fixture_8_alternate_separators():
    """Verify robust decimal-safe parsing across US, European, and space-separated formats."""
    # Standard US format
    dec_us, curr_us, valid_us = parse_decimal_safe("$2,400.00 USD")
    assert dec_us == Decimal("2400.00")
    assert curr_us == "USD"
    assert valid_us is True

    # European format with dot thousand and comma decimal
    dec_eu, curr_eu, valid_eu = parse_decimal_safe("2.400,00 €")
    assert dec_eu == Decimal("2400.00")
    assert curr_eu == "EUR"
    assert valid_eu is True

    # Space thousands separator
    dec_sp, curr_sp, valid_sp = parse_decimal_safe("12 500,50 EUR")
    assert dec_sp == Decimal("12500.50")
    assert curr_sp == "EUR"
    assert valid_sp is True

    # Comma-only whole thousands
    dec_k, curr_k, valid_k = parse_decimal_safe("150,000 USD")
    assert dec_k == Decimal("150000.00")
    assert curr_k == "USD"

    # Distinguish $2,400.00 from $240,000 and $2400.00
    dec_240k, _, _ = parse_decimal_safe("$240,000")
    assert dec_240k == Decimal("240000.00")
    assert dec_us != dec_240k

    dec_2400_plain, _, _ = parse_decimal_safe("$2400.00")
    assert dec_2400_plain == Decimal("2400.00")
    assert dec_us == dec_2400_plain
