"""Page furniture, aligned-table detection, unit typing and quality-gate count consistency."""

from __future__ import annotations

import re

import pytest

from app.extraction.layout import VisualLine, Segment, detect_aligned_tables, find_page_furniture
from app.financial_extraction import extract_table_fields
from app.schemas import ExtractedFinancialField, StructuredExtractionItem, StructuredExtractionReport, ValidationSummary
from app.validation import FinancialValidationEngine


# --------------------------------------------------------------------------- page furniture
def _span(text, x, y, size=10.0, alpha=255, flags=0):
    return {"text": text, "bbox": (x, y, x + 6 * len(text), y + size), "size": size, "alpha": alpha, "flags": flags}


def _page(*lines):
    return {"_height": 792.0, "blocks": [{"type": 0, "lines": [{"dir": d, "spans": [s]} for d, s in lines]}]}


H = (1.0, 0.0)


def test_furniture_flags_watermarks_and_running_footers_only():
    pages = [
        _page(
            ((0.77, -0.64), _span("SAMPLE", 170, 266, size=60, alpha=26)),  # rotated, transparent, huge
            (H, _span("Contract X (SAMPLE)", 54, 747, size=8)),  # running footer
            (H, _span(f"Page {n}", 533, 747, size=8)),
            (H, _span("Total", 74, 400)),  # body text that happens to repeat on two pages
            (H, _span(f"Body text on page {n}", 67, 300)),
        )
        for n in (1, 2)
    ]
    fur = find_page_furniture(pages)
    dropped = {sid[2] for sid in fur.span_ids[0]}
    assert dropped == {"SAMPLE", "Contract X (SAMPLE)", "Page 1"}


# --------------------------------------------------------------------------- aligned tables
def _line(*cells, y):
    return VisualLine([Segment(x, y, x + w, y + 9, t, False, 9.2) for x, w, t in cells])


def test_aligned_table_header_and_one_row_per_line():
    lines = [
        _line((67, 400, "Tenant shall pay monthly Base Rent in advance on the first day of each month without demand"), y=420),
        _line((74, 50, "Lease Year"), (154, 30, "Period"), (364, 60, "Monthly Rent"), (483, 55, "Annual Rent"), y=474),
        _line((74, 30, "Year 1"), (154, 110, "Jan 1, 2026 - Dec 31, 2026"), (381, 40, "$4,725.00"), (492, 45, "$56,700.00"), y=493),
        _line((74, 30, "Year 2"), (154, 110, "Jan 1, 2027 - Dec 31, 2027"), (381, 40, "$4,866.75"), (492, 45, "$58,401.00"), y=512),
        _line((74, 25, "Total"), (154, 45, "36 months"), (486, 50, "$175,254.00"), y=531),
        _line((67, 120, "4. Late Charges"), y=575),
    ]
    (table,) = detect_aligned_tables(lines, 612)
    assert table.headers == ["Lease Year", "Period", "Monthly Rent", "Annual Rent"]
    assert table.rows[-1] == ["Total", "36 months", "", "$175,254.00"]
    assert len(table.rows) == 3


def test_signature_lines_are_not_a_table():
    lines = [
        _line((67, 60, "LANDLORD"), (312, 50, "TENANT"), y=95),
        _line((67, 140, "Maple Ridge 2 LLC"), (312, 150, "Brightwater 3 Inc."), y=113),
        _line((67, 150, "______________________________"), (312, 150, "______________________________"), y=153),
    ]
    assert detect_aligned_tables(lines, 612) == []


# --------------------------------------------------------------------------- units and types
def _fields(table: str, statement_type: str | None):
    _, fields = extract_table_fields(table, {"statement_type": statement_type, "page": 1}, allow_embedding=False)
    return {f.field: f for f in fields}


def test_explicit_currency_symbol_means_units_outside_statements():
    f = _fields("Item | Amount\nRepair invoice | $1,250.00", "other")["repair_invoice_amount"]
    assert (f.reported_scale, f.validation_status, f.review_required) == ("units", "PASSED", False)


def test_statement_tables_without_unit_stay_unknown():
    f = _fields("Line | FY2025\nRevenue | $482,600", "income_statement")["revenue_fy2025"]
    assert f.reported_scale == "unknown" and f.review_required


@pytest.mark.parametrize(
    "cell, data_type",
    [("36 months", "duration"), ("45 days", "duration"), ("3,150 sq ft", "area"), ("$1,000,000 per occurrence", "limit")],
)
def test_unit_words_set_the_data_type(cell, data_type):
    f = next(iter(_fields(f"Term | Value\nItem | {cell}", "other").values()))
    assert f.data_type == data_type
    if data_type in ("duration", "area"):
        assert f.currency is None and f.reported_scale is None


def test_key_value_grid_names_fields_from_cell_labels():
    table = "Landlord: [Name] | Monthly Rent: [$2,400.00 USD] (Payable on the 1st)\nTerm of Lease: [12 Months] | Security Deposit: [$2,400.00 USD]"
    names = set(_fields(table, None))
    assert names == {"monthly_rent", "term_of_lease", "security_deposit"}


# --------------------------------------------------------------------------- quality gate
def test_gate_reasons_and_details_report_the_same_counts():
    fields = [
        ExtractedFinancialField(field=f"x{i}", original_text=f"${i},000", normalized_value=f"{i}000",
                                currency="USD", validation_status="REVIEW_REQUIRED", review_required=True)
        for i in range(1, 4)
    ]
    report = StructuredExtractionReport(
        document_id="d", document_type="invoice", document_decision="FINANCIAL_RELEVANT",
        items=[StructuredExtractionItem(chunk_id="c", decision="RELEVANT", source_text="x", financial_fields=fields)],
        validation_summary=ValidationSummary(),
    )
    vs = FinancialValidationEngine().validate_document(report).validation_summary
    results = next(d for d in vs.details if d.startswith("Results:"))
    failed, warnings, review = map(int, re.findall(r"(\d+) (?:failed|warnings|needs review)", results))
    reason = vs.gate_reasons[0]
    assert vs.quality_gate_decision == "HOLD_FOR_REVIEW"
    assert f"{warnings} warnings, {review} needs review" in reason
    assert re.search(rf"\b{warnings + review} non-fatal", reason)
    assert vs.failed_checks == failed
