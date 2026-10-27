"""Unit tests for narrative numeric-fact extraction on synthetic sentences."""

from __future__ import annotations

from decimal import Decimal

from app.narrative import extract_narrative_facts


def _by_metric(facts, metric, measure="level"):
    return next(f for f in facts if f.metric == metric and f.measure == measure)


def test_decline_with_inline_billions():
    text = "Net income declined 5% to $1.2bn, while gross margin fell 40 basis points."
    facts = extract_narrative_facts(text, default_period="FY2025")
    ni = _by_metric(facts, "net_income")
    assert Decimal(ni.normalized_amount) == Decimal("1200000000") and ni.reported_scale == "billions"
    change = _by_metric(facts, "net_income", "change_pct")
    assert change.change_direction == "DECREASE" and change.normalized_value == "5"
    bps = _by_metric(facts, "gross_margin", "change_bps")
    assert bps.data_type == "basis_points" and bps.change_direction == "DECREASE"


def test_thousands_suffix_and_prior_year():
    text = "Capital expenditures were $750K compared with $610K in 2024."
    facts = extract_narrative_facts(text, default_period="FY2025")
    cur, prior = [f for f in facts if f.metric == "capex"]
    assert (cur.period_key, Decimal(cur.normalized_amount)) == ("FY2025", Decimal("750000"))
    assert (prior.period_key, Decimal(prior.normalized_amount)) == ("FY2024", Decimal("610000"))


def test_overlap_span_is_skipped():
    text = "Revenue was $10.0 million. Net income was $2.0 million."
    start = text.index("Revenue")
    facts = extract_narrative_facts(text, default_period="FY2025", skip_span=(start, start + 26))
    assert [f.metric for f in facts] == ["net_income"]


def test_years_are_not_facts_unless_maturities():
    facts = extract_narrative_facts("Looking ahead to 2026, the notes are due 2031.", default_period="FY2025")
    assert [(f.metric, f.normalized_value) for f in facts] == [("debt_maturity_year", "2031")]
