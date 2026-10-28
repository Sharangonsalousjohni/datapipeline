"""Golden test for tests/fixtures/commercial_lease_agreement.pdf (3-page commercial lease).

The PDF carries a rotated, transparent "SAMPLE / FICTITIOUS DATA" watermark and a running footer on
every page, a parties table and a rent schedule drawn with horizontal rules only, and 14 numbered
clauses. Runs the real pipeline in memory and checks layout, chunking, extracted terms and validation.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import pytest

from app.chunking import chunk_document
from app.extraction.registry import parse_document
from app.handoff import process_file_to_handoff

PDF = Path(__file__).parent / "fixtures" / "commercial_lease_agreement.pdf"
WATERMARK = ("SAMPLE", "FICTITIOUS")  # case-sensitive: the preamble legitimately says "sample purposes"


@pytest.fixture(scope="module")
def extracted():
    return parse_document(PDF, "pdf")


@pytest.fixture(scope="module")
def chunks(extracted):
    return chunk_document("lease-golden", extracted)


@pytest.fixture(scope="module")
def handoff():
    return process_file_to_handoff(PDF, document_id="lease-golden", filename=PDF.name)


@pytest.fixture(scope="module")
def fields(handoff):
    out = {}
    for it in handoff.items:
        for f in it.financial_fields:
            out.setdefault(f.field, f)
    return out


def _dec(f) -> Decimal:
    return Decimal(f.normalized_amount or f.normalized_value)


# --------------------------------------------------------------------------- layout and chunking
def test_watermark_and_footer_never_reach_text_headings_or_cells(extracted, chunks):
    texts = [b.text for b in extracted.blocks] + [c.text for c in chunks]
    texts += [c for b in extracted.blocks if b.table for r in [b.table.headers, *b.table.rows] for c in r]
    texts += [c.location.get("section_heading") or "" for c in chunks]
    for word in WATERMARK:
        assert not any(word in t for t in texts), word
    assert not any(re.search(r"\bPage \d+\b", t) for t in texts)


def test_seventeen_chunks_one_per_clause(chunks):
    assert len(chunks) == 17
    assert chunks[0].text.startswith("COMMERCIAL LEASE AGREEMENT")  # preamble
    assert chunks[1].chunk_type == "table" and "Landlord" in chunks[1].text  # parties table
    clauses = chunks[2:16]
    assert [c.text.split("\n")[0].split(".")[0] for c in clauses] == [str(n) for n in range(1, 15)]
    assert all(len(c.text.split("\n\n", 1)) == 2 and c.text.split("\n\n", 1)[1].strip() for c in clauses)
    assert chunks[16].location["chunk_role"] == "signature_block"
    assert chunks[16].text.startswith("IN WITNESS WHEREOF")


def test_rent_schedule_has_four_data_rows(extracted):
    schedule = next(b.table for b in extracted.blocks if b.table and "Monthly Rent" in b.table.headers)
    assert schedule.headers == ["Lease Year", "Period", "Monthly Rent", "Annual Rent"]
    assert [r[0] for r in schedule.rows] == ["Year 1", "Year 2", "Year 3", "Total"]
    assert all(len(c.split()) <= 25 for r in schedule.rows for c in r)


def test_rent_schedule_stays_in_clause_three(chunks):
    clause3 = next(c for c in chunks if c.text.startswith("3. Base Rent"))
    assert "Year 3 | Jan 1, 2028 - Dec 31, 2028 | $5,012.75 | $60,153.00" in clause3.text
    assert clause3.location["tables"][0]["rows"][3][0] == "Total"


# --------------------------------------------------------------------------- extracted terms
def test_rent_schedule_fields(fields):
    monthly = [_dec(fields[f"base_rent_monthly_year_{y}"]) for y in (1, 2, 3)]
    annual = [_dec(fields[f"base_rent_annual_year_{y}"]) for y in (1, 2, 3)]
    assert monthly == [Decimal("4725.00"), Decimal("4866.75"), Decimal("5012.75")]
    assert annual == [Decimal("56700.00"), Decimal("58401.00"), Decimal("60153.00")]
    assert _dec(fields["total_rent"]) == Decimal("175254.00")
    assert fields["base_rent_period_year_1"].normalized_value == "2026-01-01 / 2026-12-31"
    for name in ("base_rent_monthly_year_1", "total_rent"):
        f = fields[name]
        assert (f.data_type, f.currency, f.reported_scale) == ("monetary", "USD", "units")
        assert not f.review_required


def test_total_term_is_a_duration_not_currency(fields):
    f = fields["base_rent_period_total"]
    assert (f.original_text, f.data_type, f.normalized_value, f.currency) == ("36 months", "duration", "36", None)


def test_deposit_is_two_months_of_year_one_rent(fields):
    dep = fields["security_deposit"]
    assert _dec(dep) == Decimal("9450.00")
    assert dep.details["stated_multiple_of_monthly_rent"] == "2"
    assert _dec(dep) == 2 * _dec(fields["base_rent_monthly_year_1"])


@pytest.mark.parametrize(
    "name, value, data_type",
    [
        ("late_fee_pct", "5", "percentage"),
        ("late_fee_grace_days", "5", "duration"),
        ("returned_payment_fee", "50.00", "monetary"),
        ("opex_share_pct", "6.3", "percentage"),
        ("opex_base_year", "2025", "year"),
        ("opex_billing_frequency", "quarterly", "text"),
        ("insurance_per_occurrence", "1000000.00", "limit"),
        ("insurance_aggregate", "2000000.00", "limit"),
        ("term_months", "36", "duration"),
        ("commencement_date", "2026-01-01", "date"),
        ("expiration_date", "2028-12-31", "date"),
        ("renewal_term_months", "36", "duration"),
        ("renewal_notice_days", "180", "duration"),
        ("monetary_cure_days", "10", "duration"),
        ("nonmonetary_cure_days", "30", "duration"),
        ("escalation_pct", "3", "percentage"),
        ("rent_due_day", "1", "count"),
        ("deposit_return_days", "45", "duration"),
        ("rentable_sqft", "3150", "area"),
        ("parking_spaces", "8", "count"),
        ("lease_number", "MRP-2025-0310", "text"),
        ("effective_date", "2025-12-01", "date"),
        ("landlord", "Maple Ridge Properties LLC", "text"),
        ("tenant", "Brightwater Design Studio, Inc.", "text"),
        ("premises", "Suite 310, 4200 Harborview Parkway, Oakridge, VA 20199", "text"),
        ("governing_law", "Commonwealth of Virginia", "text"),
    ],
)
def test_contract_terms(fields, name, value, data_type):
    f = fields[name]
    assert (f.normalized_value, f.data_type) == (value, data_type)
    assert f.statement_type == "lease"


def test_clause_fields_point_at_their_source_text(handoff):
    for it in handoff.items:
        for f in it.financial_fields:
            if f.source_kind == "clause":
                quoted = " ".join(it.extracted_text[f.char_start:f.char_end].split())
                assert quoted == f.original_text, f.field


def test_durations_are_never_currency(fields):
    for f in fields.values():
        if f.data_type == "duration":
            assert f.currency is None and f.reported_scale is None, f.field


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "rule_id, count",
    [
        ("RULE_LEASE_RENT_ANNUAL_001", 3),
        ("RULE_LEASE_RENT_TOTAL_001", 1),
        ("RULE_LEASE_ESCALATION_001", 2),
        ("RULE_LEASE_DEPOSIT_001", 1),
        ("RULE_LEASE_TERM_DATES_001", 1),
        ("RULE_RENTAL_TERMS_001", 1),
    ],
)
def test_lease_checks_run_and_pass(handoff, rule_id, count):
    checks = [c for c in handoff.validation_summary.accounting_validations if c["rule_id"] == rule_id]
    assert len(checks) == count
    assert all(c["status"] == "PASSED" for c in checks), [c["message"] for c in checks]


def test_essential_terms_are_not_reported_missing(handoff):
    chk = next(c for c in handoff.validation_summary.accounting_validations if c["rule_id"] == "RULE_RENTAL_TERMS_001")
    assert "not found" not in chk["message"]


def test_gate_accepts_and_counts_agree(handoff):
    vs = handoff.validation_summary
    assert vs.quality_gate_decision == "ACCEPT"
    assert "Results: 0 failed, 0 warnings, 0 needs review." in vs.details
    assert not any(f.review_required for it in handoff.items for f in it.financial_fields)


def test_broken_escalation_fails_the_check(handoff):
    """A schedule whose Year 3 rent ignores the escalation must fail RULE_LEASE_ESCALATION_001."""
    from app.lease_validation import validate_lease
    from app.metric_validation import FieldRef

    refs = [FieldRef(field=f.model_copy(deep=True), group=it.chunk_id) for it in handoff.items for f in it.financial_fields]
    y3 = next(r for r in refs if r.field.field == "base_rent_monthly_year_3")
    y3.field.normalized_value = y3.field.normalized_amount = "5100.00"
    checks = validate_lease(refs)
    esc = [c for c in checks if c["rule_id"] == "RULE_LEASE_ESCALATION_001" and c["period"] == "Year 3"]
    assert esc and esc[0]["status"] == "FAILED"
    assert y3.field.validation_status == "FAILED"
