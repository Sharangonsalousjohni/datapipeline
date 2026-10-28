"""Tests for the human-readable financial summary (app.summary)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.handoff import process_file_to_handoff
from app.schemas import (
    ExtractedFinancialField,
    ExtractionHandoff,
    RelevantChunksResponse,
    SourceLocation,
    ValidationSummary,
)
from app.summary import build_financial_summary

PDF = Path(__file__).parent / "fixtures" / "sample_annual_report.pdf"


def _item(fields: list[ExtractedFinancialField], *, statement_type: str | None = None, text: str = "") -> ExtractionHandoff:
    return ExtractionHandoff(
        document_id="doc", chunk_id=f"chk_{len(fields)}_{statement_type}", source_filename="doc.pdf", source_type="pdf",
        source_location=SourceLocation(page=1, section_heading="Section"), extracted_text=text, context=None,
        financial_category=None, category_scores={}, relevance_score=0.9, decision="RELEVANT",
        matching_references=[], evidence=[], evidence_bboxes=None, extraction_method="test", ocr_confidence=None,
        financial_fields=fields, extraction_quality=None, classifier_version="t", embedding_model="t",
        dataset_version="t", statement_type=statement_type,
    )


def _handoff(*items: ExtractionHandoff, gate: str = "ACCEPT") -> RelevantChunksResponse:
    return RelevantChunksResponse(
        document_id="doc", relevance_status="FINANCIAL_RELEVANT", include_uncertain=True, items=list(items),
        validation_summary=ValidationSummary(quality_gate_decision=gate, gate_reasons=[f"{gate}: test"]),
    )


def _table_field(label: str, period: str, raw: str, amount: str, metric: str | None, **kw) -> ExtractedFinancialField:
    values = dict(
        field=f"{label}_{period}".lower(), original_text=raw, normalized_value=amount, normalized_amount=amount,
        currency="USD", data_type="monetary", reporting_period=period, period_key=period, row_label=label,
        metric=metric, measure="level", statement_type="income_statement", source_kind="table",
        reported_scale="millions",
    )
    return ExtractedFinancialField(**{**values, **kw})


@pytest.fixture(scope="module")
def golden_summary():
    handoff = process_file_to_handoff(PDF, document_id="summary-golden", filename=PDF.name)
    return build_financial_summary(handoff, filename=PDF.name, source_type="pdf")


def test_golden_report_summary(golden_summary):
    s = golden_summary
    assert s.document.document_type == "Financial statement"
    assert s.outcome.decision == "ACCEPT" and not s.outcome.requires_review
    assert s.reporting.currency == "USD" and s.reporting.periods[:2] == ["FY2025", "FY2024"]
    assert s.checks.total > 0 and s.checks.failed == 0
    assert {t.statement for t in s.statements} >= {"income_statement", "balance_sheet"}

    revenue = [k for k in s.key_figures if k.metric == "revenue" and k.measure == "level" and k.period == "FY2025"]
    assert len(revenue) == 1, "duplicate figures must be merged"
    assert Decimal(revenue[0].value) == Decimal("482600000")
    assert revenue[0].display == "$482.6 million"


def test_summary_is_much_smaller_than_handoff(golden_summary):
    handoff = process_file_to_handoff(PDF, document_id="summary-golden", filename=PDF.name)
    assert len(golden_summary.model_dump_json()) < len(handoff.model_dump_json()) / 10


def test_statement_value_wins_over_narrative_and_change_column_gets_period():
    narrative = ExtractedFinancialField(
        field="revenue_fy2025", original_text="$482.6 million", normalized_value="482600000",
        normalized_amount="482600000", currency="USD", data_type="monetary", period_key="FY2025",
        row_label="sales", metric="revenue", measure="level", source_kind="narrative",
    )
    level = _table_field("Revenue", "FY2025", "482.6", "482600000", "revenue")
    change = _table_field("Revenue", "Change", "+12.0", "12000000", "revenue", data_type="monetary_change")
    change.period_key, change.measure = None, "change_pct"
    s = build_financial_summary(
        _handoff(_item([narrative]), _item([level, change], statement_type="income_statement")),
        filename="doc.pdf", source_type="pdf",
    )
    rev = {(k.measure, k.period): k for k in s.key_figures if k.metric == "revenue"}
    assert rev[("level", "FY2025")].label == "Revenue"  # table label, not the narrative fragment
    assert rev[("change_amount", "FY2025")].display == "+$12 million"
    assert len(rev) == 2
    assert s.statements[0].columns == ["FY2025", "Change"]


def test_unverified_values_only_appear_in_needs_review():
    bad = _table_field("Account number", "FY2026", "XXXX 4821", "4821", None, statement_type="other")
    bad.validation_status, bad.review_required = "REVIEW_REQUIRED", True
    bad.validation_issues = [{"rule_id": "RULE_X", "message": "scale is unknown"}]
    s = build_financial_summary(_handoff(_item([bad]), gate="HOLD_FOR_REVIEW"), filename="d.pdf", source_type="pdf")
    assert not s.key_figures and not s.statements and not s.terms
    assert [(r.label, r.reason) for r in s.needs_review] == [("Account number", "scale is unknown")]
    assert s.outcome.requires_review
    assert s.outcome.headline.startswith("Held for review")


def test_contract_terms_are_deduplicated_and_conflicts_flagged():
    def rent(raw: str, amount: str) -> ExtractedFinancialField:
        return ExtractedFinancialField(
            field="monthly_rent", original_text=raw, normalized_value=amount, currency="USD",
            frequency="monthly", due_date="1st day of each month",
        )

    late_fee = ExtractedFinancialField(
        field="late_fee_clause", original_text="[$75.00] or 5%, whichever is greater", normalized_value="75.00",
        currency="USD", condition="whichever is greater",
        details={"type": "conditional_greater_of", "fixed_amount": "75.00", "currency": "USD", "percentage": "5",
                 "percentage_base": "unpaid balance", "condition": "whichever is greater"},
    )
    s = build_financial_summary(
        _handoff(_item([rent("[$2,400.00]", "2400.00"), rent("$2,400.00", "2400.00"), late_fee]),
                 _item([rent("$2,500.00", "2500.00")])),
        filename="lease.pdf", source_type="pdf",
    )
    terms = {t.term: t for t in s.terms}
    assert terms["monthly_rent"].display == "$2,400.00" and terms["monthly_rent"].frequency == "monthly"
    assert terms["late_fee_clause"].display == "$75.00 or 5% of the unpaid balance, whichever is greater"
    assert [r.as_reported for r in s.needs_review] == ["$2,500.00"]


def test_basis_points_are_a_change_not_a_level():
    bps = ExtractedFinancialField(
        field="gross_margin_q3", original_text="110 bps", normalized_value="110", data_type="basis_points",
        period_key="Q3-2026", row_label="GROSS MARGIN", metric="gross_margin", measure="level", source_kind="narrative",
    )
    s = build_financial_summary(_handoff(_item([bps])), filename="d.pdf", source_type="pdf")
    assert (s.key_figures[0].measure, s.key_figures[0].display) == ("change_bps", "+110 bps")
