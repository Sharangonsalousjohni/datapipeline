"""Golden-file and negative tests for the sample annual report (tests/fixtures/sample_annual_report.pdf).

Runs the real pipeline in memory (parse -> chunk -> classify -> extract -> validate -> handoff)
and asserts exact normalized values, narrative facts and that every statement check passes.
The negative tests corrupt copies of the PDF and require validation to flag the injected error.
"""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.handoff import process_file_to_handoff
from app.taxonomy import FinancialCategory

FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "sample_annual_report.pdf"
GOLDEN = FIXTURES / "sample_annual_report.golden.json"

_spec = importlib.util.spec_from_file_location("accuracy_report", Path(__file__).parent.parent / "scripts" / "accuracy_report.py")
accuracy_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(accuracy_report)  # type: ignore[union-attr]

CANONICAL = {c.value for c in FinancialCategory}


@pytest.fixture(scope="module")
def handoff():
    return process_file_to_handoff(PDF, document_id="golden-sample", filename=PDF.name)


def _item(handoff, statement_type):
    return next(it for it in handoff.items if it.statement_type == statement_type and it.chunk_role == "table")


def _cell(handoff, statement_type, label, period):
    it = _item(handoff, statement_type)
    return next(f for f in it.financial_fields if f.row_label == label and f.reporting_period == period)


def _fact(handoff, metric, measure="level", period="FY2025"):
    return [
        f for it in handoff.items for f in it.financial_fields
        if f.source_kind == "narrative" and f.metric == metric and f.measure == measure and f.period_key == period
    ]


# --------------------------------------------------------------------------- exact values
@pytest.mark.parametrize("statement_type", ["highlights", "income_statement"])
def test_revenue_in_both_tables(handoff, statement_type):
    assert Decimal(_cell(handoff, statement_type, "Revenue", "2025").normalized_amount) == Decimal("482600000")
    assert Decimal(_cell(handoff, statement_type, "Revenue", "2024").normalized_amount) == Decimal("422600000")


def test_scale_metadata(handoff):
    hl = _cell(handoff, "highlights", "Revenue", "2025")
    assert (hl.reported_value, hl.reported_scale, hl.scale_multiplier) == ("482.6", "millions", 1_000_000.0)
    inc = _cell(handoff, "income_statement", "Revenue", "2025")
    assert (inc.reported_value, inc.reported_scale, inc.scale_multiplier) == ("482600", "thousands", 1000.0)


def test_net_income(handoff):
    assert Decimal(_cell(handoff, "income_statement", "Net income", "2025").normalized_amount) == Decimal("54300000")
    assert Decimal(_cell(handoff, "highlights", "Net income", "2025").normalized_amount) == Decimal("54300000")


def test_diluted_eps_is_per_share_and_unscaled(handoff):
    eps = _cell(handoff, "highlights", "Diluted EPS", "2025")
    assert eps.data_type == "per_share"
    assert eps.scale_multiplier == 1.0
    assert Decimal(eps.normalized_amount) == Decimal("2.71")
    assert eps.metric == "eps_diluted"


def test_percent_change_is_never_scaled(handoff):
    ch = _cell(handoff, "highlights", "Revenue", "Change")
    assert ch.data_type == "percentage" and ch.normalized_value == "14.2" and ch.scale_multiplier == 1.0


def test_balance_sheet_values(handoff):
    assert Decimal(_cell(handoff, "balance_sheet", "Total assets", "Dec 31, 2025").normalized_amount) == Decimal("546200000")
    assert Decimal(_cell(handoff, "balance_sheet", "Long-term debt", "Dec 31, 2025").normalized_amount) == Decimal("125000000")


def test_all_76_cells_match_golden(handoff):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    result = accuracy_report.score(handoff.model_dump(mode="json"), golden)
    assert result["cell_capture_pct"] == 100.0
    assert result["normalized_value_accuracy_pct"] == 100.0, result["value_errors"]
    assert result["text_match_pct"] == 100.0
    assert result["narrative_recall_pct"] >= 95.0, result["missed_facts"]


# --------------------------------------------------------------------------- narrative facts
def test_narrative_facts(handoff):
    (sub,) = _fact(handoff, "subscription_revenue")
    assert Decimal(sub.normalized_amount) == Decimal("351200000")
    (net_cash,) = _fact(handoff, "net_cash")
    assert Decimal(net_cash.normalized_amount) == Decimal("87700000")
    (rate,) = _fact(handoff, "interest_rate", "rate")
    assert Decimal(rate.normalized_value) == Decimal("4.5") and rate.data_type == "percentage"
    (shares,) = _fact(handoff, "shares_repurchased", "count")
    assert shares.data_type == "shares" and Decimal(shares.normalized_amount) == Decimal("520000")
    (price,) = _fact(handoff, "average_repurchase_price", "price")
    assert price.data_type == "per_share" and Decimal(price.normalized_amount) == Decimal("48.08")


def test_narrative_spans_and_direction(handoff):
    for it in handoff.items:
        for f in it.financial_fields:
            if f.source_kind == "narrative":
                assert it.extracted_text[f.char_start:f.char_end] == f.original_text
    (growth,) = _fact(handoff, "subscription_revenue", "change_pct")
    assert growth.change_direction == "INCREASE" and growth.normalized_value == "19"
    (prior,) = _fact(handoff, "subscription_revenue", "pct_of_revenue", "FY2024")
    assert prior.normalized_value == "69"


def test_letter_operating_income_checked_against_table(handoff):
    (oi,) = _fact(handoff, "operating_income")
    assert any(e.get("rule_id") == "RULE_NARRATIVE_TABLE_001" for e in oi.validation_evidence)
    assert oi.validation_status == "PASSED"


# --------------------------------------------------------------------------- validation
_IS_RULES = ["RULE_ARITH_IS_GP_001", "RULE_ARITH_IS_OPEX_001", "RULE_ARITH_IS_OPINC_001", "RULE_ARITH_IS_PRETAX_001", "RULE_ARITH_IS_NETINC_001"]
_BS_RULES = ["RULE_ARITH_BS_TCA_001", "RULE_ARITH_BS_TA_001", "RULE_ARITH_BS_TCL_001", "RULE_ARITH_BS_TL_001",
             "RULE_ARITH_BS_TLE_001", "RULE_ARITH_BS_BALANCE_001", "RULE_ARITH_BS_TA_TLE_001"]


@pytest.mark.parametrize("rule_id", _IS_RULES + _BS_RULES)
def test_every_statement_check_runs_and_passes(handoff, rule_id):
    checks = [c for c in handoff.validation_summary.accounting_validations if c["rule_id"] == rule_id]
    assert {c["period"] for c in checks} == {"FY2025", "FY2024"}, rule_id
    assert all(c["status"] == "PASSED" for c in checks), checks


def test_cross_table_and_change_checks_pass(handoff):
    xt = [c for c in handoff.validation_summary.cross_reference_checks if c["rule_id"] == "RULE_XTABLE_001"]
    assert len(xt) == 8  # revenue, gross profit, operating income, net income x 2 periods
    assert all(c["status"] == "PASSED" for c in xt)
    ch = [c for c in handoff.validation_summary.accounting_validations if c["rule_id"] == "RULE_CHANGE_PCT_001"]
    assert len(ch) == 6 and all(c["status"] == "PASSED" for c in ch)
    assert handoff.validation_summary.quality_gate_decision == "ACCEPT"


def test_confidence_reflects_evidence(handoff):
    fields = [f for it in handoff.items for f in it.financial_fields]
    assert len({f.confidence for f in fields}) > 1
    assert all(f.confidence is not None and f.confidence <= 0.99 for f in fields)
    assert all("confidence_factors" in (f.details or {}) for f in fields)


# --------------------------------------------------------------------------- chunks, categories, bboxes
def test_highlights_chunk_is_relevant_with_row_category(handoff):
    hl = _item(handoff, "highlights")
    assert hl.decision == "RELEVANT"
    assert hl.financial_category == "financial_performance"
    assert all(r.decision == "RELEVANT" for r in hl.table_rows)
    assert hl.decision_basis == "table_rows"


def test_similar_rows_share_categories(handoff):
    for st in ("highlights", "income_statement"):
        assert _cell(handoff, st, "Revenue", "2025").category == "income_revenue"
        assert _cell(handoff, st, "Gross profit", "2025").category == "financial_performance"


def test_every_category_is_canonical(handoff):
    seen = set()
    for it in handoff.items:
        seen.add(it.financial_category)
        seen.update(it.category_scores)
        seen.update(m.category for m in it.matching_references)
        seen.update(e.category for e in it.evidence if e.category)
        seen.update(r.category for r in it.table_rows)
        seen.update(f.category for f in it.financial_fields if f.category)
    seen.discard(None)
    assert seen <= CANONICAL, seen - CANONICAL


def test_non_financial_chunks_are_kept(handoff):
    roles = {it.chunk_role: it for it in handoff.items}
    assert roles["cover_page"].decision == "IRRELEVANT"
    assert roles["cover_page"].financial_category == "non_financial_content"
    assert roles["signature_block"].financial_category == "administrative_content"


def test_tables_have_their_own_chunk_and_metadata(handoff):
    tables = [it for it in handoff.items if it.chunk_role == "table"]
    assert [t.statement_type for t in tables] == ["highlights", "income_statement", "balance_sheet"]
    for t in tables:
        md = t.chunk_metadata
        assert md["unit"]["scale"] in ("millions", "thousands")
        assert md["period_columns"] and md["page"] and md["section"]
        header = t.extracted_text.splitlines()[1]
        assert "($ in" in header and "|" in header  # unit line and header row lead the chunk


def test_each_cell_has_its_own_bbox(handoff):
    for t in (it for it in handoff.items if it.chunk_role == "table"):
        boxes = set()
        for f in t.financial_fields:
            loc = f.source_location
            cell, table = loc["bbox"], loc["table_bbox"]
            assert cell and table and cell != table
            assert table[0] - 1 <= cell[0] < cell[2] <= table[2] + 1 and table[1] - 1 <= cell[1] < cell[3] <= table[3] + 1
            boxes.add(tuple(cell))
        assert len(boxes) == len(t.financial_fields)


# --------------------------------------------------------------------------- negative tests
def _corrupted(tmp_path, page, old, new):
    dst = tmp_path / "corrupted.pdf"
    accuracy_report._rewrite_pdf(PDF, dst, page, old, new)
    return process_file_to_handoff(dst, document_id="golden-negative", filename=PDF.name)


def test_missing_unit_line_gives_unknown_scale(tmp_path):
    out = _corrupted(tmp_path, 3, "($ in thousands)", None)
    inc = next(it for it in out.items if it.chunk_role == "table" and "Statements of Income" in it.extracted_text)
    money = [f for f in inc.financial_fields if f.data_type == "monetary"]
    assert money and all(f.reported_scale == "unknown" and f.review_required for f in money)
    assert all(f.validation_status == "REVIEW_REQUIRED" for f in money)
    assert inc.chunk_metadata["unit"]["scale"] == "unknown"


def test_corrupted_total_is_failed(tmp_path):
    out = _corrupted(tmp_path, 4, "327,900", "337,900")
    bs = next(it for it in out.items if it.statement_type == "balance_sheet")
    tca = next(f for f in bs.financial_fields if f.row_label == "Total current assets" and f.reporting_period == "Dec 31, 2025")
    assert tca.normalized_amount == "337900000"
    assert tca.validation_status == "FAILED"
    assert any(i["rule_id"] == "RULE_ARITH_BS_TCA_001" for i in tca.validation_issues)
    assert tca.review_required and tca.confidence < 0.5
    # the correct total assets figure is not blamed
    ta = next(f for f in bs.financial_fields if f.row_label == "Total assets" and f.reporting_period == "Dec 31, 2025")
    assert ta.validation_status != "FAILED"
    assert out.validation_summary.quality_gate_decision in ("HOLD_FOR_REVIEW", "REJECT")


def test_disagreeing_table_units_fail_cross_table(tmp_path):
    out = _corrupted(tmp_path, 3, "($ in thousands)", "($ in millions)")
    xt = [c for c in out.validation_summary.cross_reference_checks if c["rule_id"] == "RULE_XTABLE_001"]
    assert xt and any(c["status"] == "FAILED" for c in xt)
    inc_rev = _cell(out, "income_statement", "Revenue", "2025")
    assert inc_rev.validation_status == "FAILED"
    assert any(i["rule_id"] == "RULE_XTABLE_001" for i in inc_rev.validation_issues)
