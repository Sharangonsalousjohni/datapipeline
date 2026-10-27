"""Narrative numeric-fact extraction.

Turns sentences such as "Subscription revenue grew 19% to $351.2 million" or "the Company
repurchased 0.52 million shares at an average price of $48.08 per share" into
:class:`ExtractedFinancialField` objects with the same schema as table fields, plus the
measure (level / change_pct / pct_of_revenue / ...), direction of change, period and a
``char_start``/``char_end`` span into the chunk's ``extracted_text``.

Metric linking, per mention:
1. the closest canonical-metric phrase to the left of the mention (within the clause);
2. if the left context is only a connector ("to", "versus", ", or", "(2024:"), the previous
   mention's metric;
3. a canonical-metric phrase in a short window to the right ("$88.4 million in cash from operations");
4. otherwise the previous mention's metric, else unlinked (metric=None).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.numbers import format_decimal, parse_number
from app.schemas import ExtractedFinancialField
from app.taxonomy import METRICS, METRICS_BY_NAME, normalize_period, shift_period

# --------------------------------------------------------------------------- mention patterns
_NUM = r"\d[\d,]*(?:\.\d+)?"
_SCALE = r"(?:thousand|million|billion|trillion|mn|bn|tn|k|m|b)\b"
_MENTION_RE = re.compile(
    rf"(?P<money>(?:US\$|C\$|A\$|HK\$|\$|€|£|¥|₹)\s?{_NUM}(?:\s?{_SCALE})?(?:\s+per\s+(?:diluted\s+|basic\s+|common\s+)?share\b)?"
    rf"|\b(?:USD|EUR|GBP|CHF|JPY)\s?{_NUM}(?:\s?{_SCALE})?)"
    rf"|(?P<bps>\b{_NUM}\s?(?:basis\s+points?|bps|bp)\b)"
    rf"|(?P<pct>(?<![\w.])[-−–]?{_NUM}\s?(?:%|percent\b|per\s?cent\b))"
    rf"|(?P<shares>\b{_NUM}(?:\s?(?:thousand|million|billion)\b)?\s+(?:common\s+)?shares\b)"
    rf"|(?P<year>(?<=\bdue\s)(?:19|20)\d\d\b|(?<=\bmaturing\sin\s)(?:19|20)\d\d\b|(?<=\bmatures\sin\s)(?:19|20)\d\d\b"
    rf"|(?<=\bmaturity\sin\s)(?:19|20)\d\d\b|(?<=\bexpiring\sin\s)(?:19|20)\d\d\b|(?<=\bexpires\sin\s)(?:19|20)\d\d\b)",
    re.I,
)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'‘“])|\n\s*\n")
_INCREASE_RE = re.compile(r"\b(grew|grow|growth|increased?|increasing|rose|rise|up|expanded|improved|higher|gained|climbed)\b", re.I)
_DECREASE_RE = re.compile(r"\b(declined?|decreased?|decreasing|fell|fall|down|contracted|lower|dropped|shrank)\b", re.I)
_CHANGE_RE = re.compile(rf"{_INCREASE_RE.pattern}|{_DECREASE_RE.pattern}|\bby\s*$|\bchange\b", re.I)
_CONNECTOR_RE = re.compile(
    r"^[\s,]*(?:to|of|at|versus|vs\.?|compared\s+(?:with|to)|or|from|and|up|down|reaching|totaling|totalling|was|were|of\s+which)?[\s,]*$",
    re.I,
)
_PRIOR_YEAR_PAREN_RE = re.compile(r"\(\s*(?:FY\s*)?((?:19|20)\d\d)\s*:\s*$", re.I)
_PCT_OF_REVENUE_RE = re.compile(r"^\s*of\s+(?:total\s+|net\s+)?(?:revenues?|sales)\b", re.I)
_PRIOR_RE = re.compile(r"\b(?:in\s+the\s+prior[\s-]year|prior[\s-]year|last\s+year|a\s+year\s+(?:ago|earlier)|the\s+previous\s+year)\b", re.I)
_IN_YEAR_RE = re.compile(r"^\s*(?:in|for|during|at\s+(?:the\s+end\s+of\s+)?)\s+(?:fiscal\s+)?((?:19|20)\d\d)\b", re.I)
# A sentence-wide period: "During 2025, ..." / "In fiscal 2024, ..." at the start, or "fiscal 2025" anywhere.
_SENTENCE_YEAR_RE = re.compile(
    r"^\W*(?:during|in|for|throughout)\s+(?:fiscal\s+)?((?:19|20)\d\d)\b|\bfiscal\s+(?:year\s+)?((?:19|20)\d\d)\b", re.I
)
_REGION_RE = re.compile(
    r"\b(north\s+america|international|europe|emea|apac|asia[\s-]pacific|americas|latin\s+america|united\s+states|u\.s\.|domestic)\b",
    re.I,
)

# --------------------------------------------------------------------------- metric phrase matcher


def _phrase_pattern(syn: str) -> str:
    parts = []
    for tok in syn.split():
        if tok == "and":
            parts.append(r"(?:&|and)")
        else:
            parts.append(re.escape(tok) + r"['’]?s?" if not tok.endswith("s") else re.escape(tok) + r"['’]?")
    return r"\b" + r"[\s\-]*".join(parts) + r"\b"


_PHRASES: list[tuple[re.Pattern[str], str, int]] = []
for _md in METRICS:
    for _syn in _md.synonyms:
        _PHRASES.append((re.compile(_phrase_pattern(_syn), re.I), _md.name, len(_syn)))


def _find_metric(text: str, *, prefer: str = "right") -> tuple[str | None, int, int, str | None]:
    """Find a metric phrase in ``text``. ``prefer='right'`` returns the occurrence ending last
    (closest to a following mention); ``'left'`` the one starting first. Longest wins ties."""
    best: tuple[int, int, str, int, int] | None = None  # (pos_key, length, name, start, end)
    for pat, name, length in _PHRASES:
        for m in pat.finditer(text):
            key = m.end() if prefer == "right" else -m.start()
            cand = (key, length, name, m.start(), m.end())
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return None, -1, -1, None
    return best[2], best[3], best[4], text[best[3]:best[4]]


@dataclass
class _Mention:
    kind: str  # money | bps | pct | shares | year
    start: int
    end: int
    text: str
    metric: str | None = None
    phrase: str | None = None
    measure: str = "level"
    period_key: str | None = None
    direction: str | None = None
    consumed_end: int = 0
    region: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _clauses(text: str) -> list[tuple[int, int]]:
    """Sentence spans, further split at semicolons, as absolute (start, end) offsets."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END_RE.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        sub = s
        for sm in re.finditer(r";\s*", text[s:e]):
            out.append((sub, s + sm.start()))
            sub = s + sm.end()
        out.append((sub, e))
    return [(s, e) for s, e in out if text[s:e].strip()]


def _direction(text: str) -> str | None:
    inc = list(_INCREASE_RE.finditer(text))
    dec = list(_DECREASE_RE.finditer(text))
    if not inc and not dec:
        return None
    last_inc = inc[-1].end() if inc else -1
    last_dec = dec[-1].end() if dec else -1
    return "INCREASE" if last_inc > last_dec else "DECREASE"


def _sentence_period(text: str, default_key: str | None) -> str | None:
    for m in _SENTENCE_YEAR_RE.finditer(text):
        yr = m.group(1) or m.group(2)
        # "due 2030" / "maturing in 2029" are maturities, not reporting periods
        if not re.search(r"(?:due|maturing|matures|expiring|expires)\s+(?:in\s+)?$", text[: m.start()], re.I):
            return f"FY{yr}"
    return default_key


def extract_narrative_facts(
    text: str,
    *,
    default_period: str | None = None,
    default_currency: str = "USD",
    page: int | None = None,
    skip_span: tuple[int, int] | None = None,
    statement_type: str | None = None,
) -> list[ExtractedFinancialField]:
    """Extract numeric facts from narrative text. Offsets refer to ``text``."""
    facts: list[ExtractedFinancialField] = []
    for c_start, c_end in _clauses(text):
        clause = text[c_start:c_end]
        sentence_period = _sentence_period(clause, default_period)
        mentions: list[_Mention] = []
        for m in _MENTION_RE.finditer(clause):
            kind = next(k for k, v in m.groupdict().items() if v)
            mentions.append(_Mention(kind=kind, start=m.start(), end=m.end(), text=m.group(0)))
        prev: _Mention | None = None
        for mt in mentions:
            left_start = prev.consumed_end if prev else 0
            left = clause[left_start:mt.start]
            right_full = clause[mt.end:]
            right_window = re.split(r"[,.;:()]", right_full, maxsplit=1)[0]
            right_window = " ".join(right_window.split()[:6]) if right_window.strip() else ""
            mt.consumed_end = mt.end
            paren_year = _PRIOR_YEAR_PAREN_RE.search(left)

            # 1-4: metric linking
            metric, phrase = None, None
            inherited = False
            if paren_year and prev is not None:
                metric, phrase, inherited = prev.metric, prev.phrase, True
            else:
                metric, _, _, phrase = _find_metric(left)
                if metric is None and prev is not None and _CONNECTOR_RE.match(left):
                    metric, phrase, inherited = prev.metric, prev.phrase, True
                if metric is None and mt.kind != "year" and not _PCT_OF_REVENUE_RE.match(right_full):
                    r_metric, r_s, r_e, r_phrase = _find_metric(right_window, prefer="left")
                    if r_metric:
                        metric, phrase = r_metric, r_phrase
                        # advance past the phrase used so the next mention's left context excludes it
                        idx = right_full.find(r_phrase) if r_phrase else -1
                        if idx >= 0:
                            mt.consumed_end = mt.end + idx + len(r_phrase)
                if metric is None and prev is not None:
                    metric, phrase, inherited = prev.metric, prev.phrase, True

            region = _REGION_RE.search(left)
            if region and not inherited:
                mt.region = re.sub(r"[^a-z]+", "_", region.group(1).lower()).strip("_")

            # measure
            if mt.kind == "pct":
                base = _PCT_OF_REVENUE_RE.match(right_full)
                if base:
                    mt.measure = "pct_of_revenue"
                    mt.consumed_end = mt.end + base.end()  # "of revenue" is the base, not the next subject
                    if metric in (None, "revenue") and prev is not None and not mt.region:
                        metric, phrase = prev.metric, prev.phrase
                elif inherited and prev is not None and prev.kind == "pct":
                    mt.measure = prev.measure
                elif _CHANGE_RE.search(left) and not re.search(r"\bto\s*$", left, re.I):
                    mt.measure = "change_pct"
                elif metric and METRICS_BY_NAME.get(metric) and METRICS_BY_NAME[metric].data_type == "percentage":
                    mt.measure = "rate" if metric == "interest_rate" else "level"
                elif re.search(r"\binterest\b", left, re.I):
                    metric, mt.measure = "interest_rate", "rate"
                else:
                    mt.measure = "level"
            elif mt.kind == "bps":
                mt.measure = "change_bps" if _CHANGE_RE.search(left) else "level"
            elif mt.kind == "shares":
                mt.measure = "count"
            elif mt.kind == "year":
                mt.measure = "maturity_year"
                facility = any(p.metric == "credit_facility" for p in mentions if p.start < mt.start) or re.search(
                    r"facility", clause[: mt.start], re.I
                )
                metric = "credit_facility_maturity_year" if facility else "debt_maturity_year"
            elif mt.kind == "money" and re.search(r"per\s+(?:\w+\s+)?share", mt.text, re.I):
                mt.measure = "price" if metric == "average_repurchase_price" else "level"

            if mt.region:
                base = metric if metric and metric.endswith("revenue") else "revenue"
                metric = f"{mt.region}_{base}" if mt.measure == "pct_of_revenue" else f"{mt.region}_{base}"
            mt.metric, mt.phrase = metric, phrase

            # period
            right_period = _IN_YEAR_RE.match(right_full)
            if paren_year:
                mt.period_key = f"FY{paren_year.group(1)}"
            elif right_period:
                mt.period_key = f"FY{right_period.group(1)}"
            elif _PRIOR_RE.search(right_window or right_full[:40]):
                mt.period_key = shift_period(sentence_period, -1)
            elif inherited and prev is not None and prev.period_key and prev.period_key != sentence_period and mt.kind == prev.kind:
                mt.period_key = prev.period_key
            else:
                mt.period_key = sentence_period

            # direction of change for this metric in the clause
            seg_start = 0
            for p in mentions:
                if p is mt:
                    break
                if p.metric != metric:
                    seg_start = p.end
            mt.direction = _direction(clause[seg_start:mt.start]) if mt.measure in ("change_pct", "change_bps", "level") else None
            if mt.measure == "level" and mt.direction and not re.search(r"\b(?:to|by)\s*$", clause[seg_start:mt.start], re.I):
                if not any(p.metric == metric and p.measure.startswith("change") for p in mentions):
                    mt.direction = None
            prev = mt

        for mt in mentions:
            abs_start, abs_end = c_start + mt.start, c_start + mt.end
            if skip_span and skip_span[0] <= abs_start < skip_span[1]:
                continue
            facts.append(_to_field(mt, abs_start, abs_end, default_currency, page, statement_type))
    return facts


def _to_field(
    mt: _Mention, start: int, end: int, default_currency: str, page: int | None, statement_type: str | None
) -> ExtractedFinancialField:
    parsed = parse_number(mt.text, default_currency)
    value = parsed.value if parsed.value is not None else Decimal(0)
    if mt.kind == "pct" and mt.text.lstrip().startswith(("-", "−", "–")):
        value = -abs(value)
    if mt.kind == "year":
        dtype, scale, mult = "year", None, Decimal(1)
    elif mt.kind == "pct":
        dtype, scale, mult = "percentage", None, Decimal(1)
    elif mt.kind == "bps":
        dtype, scale, mult = "basis_points", None, Decimal(1)
    elif mt.kind == "shares":
        dtype, scale, mult = "shares", parsed.scale or "units", parsed.multiplier
    elif parsed.kind == "per_share":
        dtype, scale, mult = "per_share", "units", Decimal(1)
    else:
        dtype, scale, mult = "monetary", parsed.scale or "units", parsed.multiplier
    normalized = value * mult
    reported = f"{value:f}"
    norm_str = format_decimal(normalized) if dtype in ("monetary", "shares", "per_share") else reported
    currency = parsed.currency or default_currency if dtype in ("monetary", "per_share") else None
    mdef = METRICS_BY_NAME.get(mt.metric or "")
    category = mdef.category if mdef else ("income_revenue" if (mt.metric or "").endswith("revenue") else None)
    suffix = "" if mt.measure in ("level", "count", "rate", "price", "maturity_year") else f"_{mt.measure}"
    name = f"{mt.metric or 'unlinked_' + dtype}{suffix}_{(mt.period_key or 'na').lower()}"
    return ExtractedFinancialField(
        field=name,
        original_text=mt.text,
        normalized_value=norm_str or reported,
        reported_value=reported,
        reported_scale=scale,
        scale_multiplier=float(mult),
        normalized_amount=norm_str if dtype in ("monetary", "shares", "per_share") else None,
        currency=currency,
        data_type=dtype,
        reporting_period=mt.period_key,
        row_label=mt.phrase,
        category=category,
        details={
            "measure": mt.measure,
            "direction": mt.direction,
            "matched_phrase": mt.phrase,
            "scale_source": f"inline '{parsed.scale}'" if parsed.scale else "as written",
            "tolerance": format_decimal(parsed.half_unit * mult),
            "decimals": parsed.decimals,
        },
        validation_status="PASSED",
        normalization_status="PASSED",
        validation_notes=f"Narrative fact '{mt.text}' -> {mt.metric or 'unlinked'} ({mt.measure}, {mt.period_key})",
        confidence=None,
        source_location={"page": page, "char_start": start, "char_end": end},
        change_direction=mt.direction,
        metric=mt.metric,
        measure=mt.measure,
        period_key=mt.period_key,
        statement_type=statement_type,
        source_kind="narrative",
        char_start=start,
        char_end=end,
    )


def normalize_period_label(label: str | None) -> str | None:
    return normalize_period(label)
