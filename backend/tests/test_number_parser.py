"""Unit tests for the Decimal-exact number parser and unit-declaration detection."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.numbers import detect_document_unit, detect_unit, format_decimal, parse_number


def test_parenthesised_negative():
    p = parse_number("(1,234)")
    assert p.value == Decimal("-1234")
    assert p.negative is True
    assert p.kind == "number"


def test_unicode_minus_percentage():
    p = parse_number("−5.2%")
    assert p.value == Decimal("-5.2")
    assert p.kind == "percentage"
    assert p.decimals == 1


def test_inline_billions():
    p = parse_number("$1.2bn")
    assert p.kind == "monetary"
    assert p.currency == "USD"
    assert p.scale == "billions"
    assert p.normalized == Decimal("1200000000")


def test_basis_points():
    p = parse_number("90 basis points")
    assert p.kind == "basis_points"
    assert p.value == Decimal("90")
    assert p.scale is None


def test_share_count_with_scale():
    p = parse_number("0.52 million shares")
    assert p.kind == "shares"
    assert p.normalized == Decimal("520000")
    assert format_decimal(p.normalized) == "520000"


@pytest.mark.parametrize("raw", ["—", "-", "–", "", "  ", "n/a", "NM"])
def test_dashes_and_blanks_are_null(raw):
    p = parse_number(raw)
    assert p.is_null
    assert p.value is None


@pytest.mark.parametrize(
    "raw,value,kind",
    [
        ("$482.6 million", "482600000", "monetary"),
        ("$750K", "750000", "monetary"),
        ("($5,865)", "-5865", "monetary"),
        ("– 12.5", "-12.5", "number"),
        ("-$12", "-12", "monetary"),
        ("+110 bps", "110", "basis_points"),
        ("2.00x", "2.00", "ratio"),
        ("$48.08 per share", "48.08", "per_share"),
        ("2.400,00 €", "2400.00", "monetary"),
        ("2025", "2025", "year"),
        ("482,600", "482600", "number"),
    ],
)
def test_parse_variants(raw, value, kind):
    p = parse_number(raw)
    assert p.kind == kind, raw
    assert p.normalized == Decimal(value), raw


def test_precision_half_unit():
    assert parse_number("482.6").half_unit == Decimal("0.05")
    assert parse_number("482,600").half_unit == Decimal("0.5")


@pytest.mark.parametrize(
    "text,scale,except_ps",
    [
        ("($ in millions, except per share)", "millions", True),
        ("($ in thousands)", "thousands", False),
        ("$  IN   MILLIONS", "millions", False),
        ("(000s)", "thousands", False),
        ("USD m", "millions", False),
        ("€bn", "billions", False),
        ("in millions, except per share", "millions", True),
        ("(in millions of U.S. dollars, except share and per-share data)", "millions", True),
        ("Amounts in thousands; except per share amounts", "thousands", True),
    ],
)
def test_detect_unit(text, scale, except_ps):
    u = detect_unit(text)
    assert u is not None, text
    assert u.scale == scale, text
    assert u.except_per_share is except_ps, text


def test_except_share_counts():
    assert detect_unit("(in millions, except share and per share data)").except_shares is True
    assert detect_unit("(in millions, except per share data)").except_shares is False


@pytest.mark.parametrize("text", ["Revenue 482,000", "Dec 31, 2025", "Revenue grew 14% to $5 million", "2025"])
def test_detect_unit_ignores_non_declarations(text):
    assert detect_unit(text) is None


def test_document_unit_only_from_explicit_declarations():
    assert detect_document_unit("Currency: USD (Thousands)").scale == "thousands"
    assert detect_document_unit("All amounts are presented in millions of euros").scale == "millions"
    assert detect_document_unit("Consolidated Statements of Income ($ in thousands)") is None
