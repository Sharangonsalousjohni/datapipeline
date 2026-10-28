"""Decimal-exact parsing of financial numbers and detection of reporting units.

``parse_number`` turns a cell or phrase ("(1,234)", "−5.2%", "$1.2bn", "90 basis points",
"0.52 million shares", "—") into a :class:`ParsedNumber` without ever going through float.

``detect_unit`` recognises unit declarations such as "($ in millions, except per share)",
"in thousands", "(000s)", "USD m" or "€bn" and returns a :class:`UnitSpec`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

SCALE_MULTIPLIERS: dict[str, Decimal] = {
    "units": Decimal(1),
    "thousands": Decimal(1_000),
    "millions": Decimal(1_000_000),
    "billions": Decimal(1_000_000_000),
    "trillions": Decimal(1_000_000_000_000),
}

_SCALE_WORDS: dict[str, str] = {
    "k": "thousands", "thousand": "thousands", "thousands": "thousands", "000s": "thousands", "000": "thousands",
    "m": "millions", "mn": "millions", "mm": "millions", "mio": "millions", "million": "millions", "millions": "millions",
    "b": "billions", "bn": "billions", "billion": "billions", "billions": "billions",
    "t": "trillions", "tn": "trillions", "trillion": "trillions", "trillions": "trillions",
}

_CURRENCY_SYMBOLS: list[tuple[str, str]] = [
    ("US$", "USD"), ("C$", "CAD"), ("A$", "AUD"), ("S$", "SGD"), ("HK$", "HKD"), ("NZ$", "NZD"),
    ("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("¥", "JPY"), ("₹", "INR"),
]
_CURRENCY_CODES = ("USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "SGD", "HKD", "NZD")
_CURRENCY_WORDS = {"dollars": "USD", "dollar": "USD", "euros": "EUR", "euro": "EUR", "pounds": "GBP", "rupees": "INR", "yen": "JPY"}

# Every dash-like character that may stand for a minus sign or an empty cell.
_DASHES = "-‐‑‒–—―−﹘﹣－"
_DASH_CLASS = re.escape(_DASHES)
_NULL_TOKENS = {"", "n/a", "na", "nm", "n.m.", "null", "none", "nil", "*", "--", "——"}


@dataclass
class ParsedNumber:
    raw: str
    value: Decimal | None  # value as printed, signed, before any scale
    kind: str  # number | monetary | percentage | basis_points | ratio | shares | per_share | year | empty | text
    currency: str | None = None
    scale: str | None = None  # inline scale word ("$482.6 million" -> millions)
    negative: bool = False
    decimals: int = 0  # digits after the decimal point in the printed value

    @property
    def multiplier(self) -> Decimal:
        return SCALE_MULTIPLIERS.get(self.scale or "units", Decimal(1))

    @property
    def normalized(self) -> Decimal | None:
        """Value with the inline scale applied (table scale is applied by the caller)."""
        return None if self.value is None else self.value * self.multiplier

    @property
    def half_unit(self) -> Decimal:
        """Half a unit of the printed precision, in printed units (e.g. 482.6 -> 0.05)."""
        return Decimal(5) * Decimal(10) ** (-(self.decimals + 1))

    @property
    def is_null(self) -> bool:
        return self.kind == "empty"


def _detect_currency(s: str) -> str | None:
    for sym, code in _CURRENCY_SYMBOLS:
        if sym in s:
            return code
    up = s.upper()
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", up):
            return code
    for word, code in _CURRENCY_WORDS.items():
        if re.search(rf"\b{word}\b", s, re.I):
            return code
    return None


_NUMBER_CORE_RE = re.compile(r"\d[\d,.  ]*\d|\d")
_SCALE_SUFFIX_RE = re.compile(
    r"^\s*(?P<w>thousands?|millions?|billions?|trillions?|mn|mm|mio|bn|tn|k|m|b|t)\b\.?", re.I
)


def _to_decimal(num: str) -> tuple[Decimal | None, int]:
    s = num.replace(" ", "").replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):  # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:  # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        # "75,50" is a decimal comma; "2,400" / "1,234,567" are thousands groups
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = f"{parts[0]}.{parts[1]}"
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1:  # 1.234.567
        s = s.replace(".", "")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None, 0
    exp = d.as_tuple().exponent
    return d, (-exp if isinstance(exp, int) and exp < 0 else 0)


def parse_number(text: str | None, default_currency: str | None = None) -> ParsedNumber:
    """Parse a financial number. Never scales by table units; inline scale words are recorded."""
    raw = text if text is not None else ""
    s = " ".join(raw.split())
    if s.lower() in _NULL_TOKENS or (s and all(ch in _DASHES for ch in s)):
        return ParsedNumber(raw=raw, value=None, kind="empty")

    negative = False
    # Parentheses negatives: (1,234)  ($1,234)  $(1,234)  (5.2%)
    paren = re.fullmatch(r"(?P<pre>[^\d()]*)\(\s*(?P<body>[^()]*\d[^()]*)\s*\)(?P<post>.*)", s)
    if paren:
        negative = True
        s = f"{paren.group('pre')}{paren.group('body')}{paren.group('post')}".strip()
    # Leading minus / en dash / em dash / unicode minus, possibly after a currency symbol: -5.2%, −$12, $-12
    lead = re.match(rf"^(?P<pre>[^\d{_DASH_CLASS}]{{0,4}}?)\s*[{_DASH_CLASS}]\s*(?=[\d$€£¥₹.])", s)
    if lead:
        negative = True
        s = (lead.group("pre") + s[lead.end():]).strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    # Trailing minus used by some ledgers: 1,234-
    if re.search(rf"\d\s*[{_DASH_CLASS}]$", s):
        negative = True
        s = s.rstrip(_DASHES).strip()

    m = _NUMBER_CORE_RE.search(s)
    if not m:
        return ParsedNumber(raw=raw, value=None, kind="text")
    value, decimals = _to_decimal(m.group(0).strip())
    if value is None:
        return ParsedNumber(raw=raw, value=None, kind="text")
    if negative:
        value = -value
    before, after = s[: m.start()], s[m.end():]
    after_l = after.lower().strip()
    currency = _detect_currency(before + " " + after) or None

    scale = None
    sm = _SCALE_SUFFIX_RE.match(after)
    if sm:
        word = sm.group("w").lower()
        # a bare single letter is a scale only when glued to the number or followed by nothing/shares
        if len(word) > 1 or re.match(r"^\s?[kmbt]\b", after, re.I):
            scale = _SCALE_WORDS.get(word) or _SCALE_WORDS.get(word.rstrip("s"))
            after_l = after[sm.end():].lower().strip()

    if after_l.startswith("%") or re.match(r"^(percent|per\s*cent|pct)\b", after_l):
        kind = "percentage"
    elif re.match(r"^(basis\s+points?|bps|bp)\b", after_l):
        kind = "basis_points"
    elif re.match(r"^x\b|^×", after_l):
        kind = "ratio"
    elif re.match(r"^(?:\w+\s+)?(?:common\s+)?shares?\b", after_l) and "per share" not in after_l:
        kind = "shares"
    elif re.search(r"\bper\s+(?:diluted\s+|basic\s+)?share\b", after_l):
        kind = "per_share"
        currency = currency or default_currency
    elif currency:
        kind = "monetary"
    elif (
        scale is None and decimals == 0 and "," not in m.group(0) and not negative
        and re.fullmatch(r"(19|20)\d\d", m.group(0)) and not before.strip() and not after_l
    ):
        kind = "year"
    else:
        kind = "number"
    return ParsedNumber(
        raw=raw, value=value, kind=kind, currency=currency, scale=scale, negative=negative, decimals=decimals
    )


# --------------------------------------------------------------------------- unit declarations
@dataclass
class UnitSpec:
    scale: str  # units | thousands | millions | billions | unknown
    currency: str | None = None
    except_per_share: bool = False
    except_shares: bool = False
    source: str | None = None  # header | caption | line_above | section_heading | document | inline
    text: str | None = None  # the matched declaration
    extras: dict = field(default_factory=dict)

    @property
    def multiplier(self) -> Decimal:
        return SCALE_MULTIPLIERS.get(self.scale, Decimal(1))

    @property
    def known(self) -> bool:
        return self.scale != "unknown"

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "multiplier": str(self.multiplier) if self.known else None,
            "currency": self.currency,
            "except_per_share": self.except_per_share,
            "except_shares": self.except_shares,
            "source": self.source,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> UnitSpec | None:
        if not d:
            return None
        return cls(
            scale=d.get("scale") or "unknown",
            currency=d.get("currency"),
            except_per_share=bool(d.get("except_per_share")),
            except_shares=bool(d.get("except_shares")),
            source=d.get("source"),
            text=d.get("text"),
        )


UNKNOWN_UNIT = UnitSpec(scale="unknown")

_CUR_TOKEN = r"(?:US\$|C\$|A\$|HK\$|\$|€|£|¥|₹|USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY|SGD|U\.?S\.?\s+dollars|dollars|euros?|pounds)"
_SCALE_WORD = r"(?P<scale>thousands?|millions?|billions?|trillions?)"

_UNIT_PATTERNS: list[re.Pattern[str]] = [
    # "$ in millions", "(in millions of U.S. dollars)", "amounts in thousands", "USD in thousands", "dollars in millions"
    re.compile(
        rf"(?:(?P<cur1>{_CUR_TOKEN})\s*[,;:]?\s*)?(?:(?:all\s+)?(?:amounts?|figures?|numbers?|values?)\s+(?:are\s+)?(?:expressed\s+|presented\s+|stated\s+|shown\s+)?)?"
        rf"\bin\s+{_SCALE_WORD}\b(?:\s+of\s+(?P<cur2>{_CUR_TOKEN}))?",
        re.I,
    ),
    # "(000s)", "$000s", "USD '000", "€'000", "(in 000's)"
    re.compile(rf"(?:(?P<cur1>{_CUR_TOKEN})\s*)?(?:in\s+)?['’]?\s*(?<![\d,.])(?P<scale>000)(?!\d)\s*['’]?s?\b", re.I),
    # "USD m", "€bn", "$m", "£k", "EUR million(s)" used as a header token (no digits around it)
    re.compile(
        rf"(?<![\w.])(?P<cur1>{_CUR_TOKEN})\s?(?P<scale>thousands?|millions?|billions?|mn|mm|bn|m|k|b)\b(?!\s*\d)",
        re.I,
    ),
    # "Currency: USD (Thousands)", "USD (Millions)", "(Thousands USD)"
    re.compile(rf"(?P<cur1>{_CUR_TOKEN})?\s*\(\s*{_SCALE_WORD}\s*(?P<cur2>{_CUR_TOKEN})?\s*\)", re.I),
]
_EXCEPT_RE = re.compile(r"except\b[^)\n]{0,60}?\bper[\s-]*share", re.I)
_EXCEPT_SHARES_RE = re.compile(r"except\b[^)\n]{0,40}?(?<!per )(?<!per-)\bshares?\b", re.I)


def _currency_code(tok: str | None) -> str | None:
    if not tok:
        return None
    return _detect_currency(tok) or ("USD" if "dollar" in tok.lower() else None)


def detect_unit(text: str | None, source: str | None = None) -> UnitSpec | None:
    """Detect a unit declaration in a short piece of text (header cell, caption, heading)."""
    if not text:
        return None
    s = " ".join(text.replace("’", "'").split())
    for pat in _UNIT_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        word = m.group("scale").lower()
        scale = _SCALE_WORDS.get(word) or _SCALE_WORDS.get(word.rstrip("s"))
        if not scale:
            continue
        gd = m.groupdict()
        currency = _currency_code(gd.get("cur1")) or _currency_code(gd.get("cur2")) or _detect_currency(s)
        except_ps = bool(_EXCEPT_RE.search(s))
        return UnitSpec(
            scale=scale,
            currency=currency,
            except_per_share=except_ps,
            except_shares=bool(_EXCEPT_SHARES_RE.search(s)),
            source=source,
            text=m.group(0).strip(),
        )
    return None


_DOC_UNIT_RE = re.compile(
    r"\bcurrency\s*:\s*[A-Z]{3}\s*\([^)]*\)|\ball\s+(?:amounts|figures)\b[^.\n]{0,60}\bin\s+(?:thousands|millions|billions)\b",
    re.I,
)


def detect_document_unit(text: str) -> UnitSpec | None:
    """Only explicit document-wide declarations ("Currency: USD (Thousands)", "All amounts in millions")."""
    m = _DOC_UNIT_RE.search(text or "")
    if not m:
        return None
    return detect_unit(m.group(0), source="document")


_PER_SHARE_LABEL_RE = re.compile(r"\bper\s+(?:common\s+|diluted\s+|basic\s+)?share\b|\beps\b|\bearnings\s+per\b|\bdividends?\s+per\b", re.I)
_SHARE_COUNT_LABEL_RE = re.compile(r"\b(?:weighted[\s-]+average\s+)?(?:shares|share\s+count)\b(?!.*\bper\b)|\bshares\s+outstanding\b", re.I)


def is_per_share_label(label: str) -> bool:
    return bool(_PER_SHARE_LABEL_RE.search(label or ""))


def is_share_count_label(label: str) -> bool:
    return bool(_SHARE_COUNT_LABEL_RE.search(label or "")) and not is_per_share_label(label)


_DURATION_RE = re.compile(r"\b(months?|days?|years?|weeks?)\b", re.I)
_AREA_RE = re.compile(r"\b(?:sq\.?\s*ft\.?|square\s+(?:feet|foot)|sf|sq\.?\s*m|square\s+met(?:er|re)s?)\b", re.I)
_LIMIT_RE = re.compile(r"\bper\s+occurrence\b|\baggregate\b", re.I)


def infer_unit_type(text: str) -> tuple[str, str] | None:
    """Data type from unit words next to a number: (data_type, unit).

    "36 months" -> duration, "3,150 sq ft" -> area, "$1,000,000 per occurrence" -> limit.
    Percentages are handled by :func:`parse_number`.
    """
    if _LIMIT_RE.search(text):
        return "limit", "per_occurrence" if "occurrence" in text.lower() else "aggregate"
    if _AREA_RE.search(text):
        return "area", "sq_m" if re.search(r"sq\.?\s*m\b|met(?:er|re)", text, re.I) else "sq_ft"
    m = _DURATION_RE.search(text)
    if m and not re.search(r"[$€£¥₹]", text):
        word = m.group(1).lower()
        return "duration", word if word.endswith("s") else word + "s"
    return None


def format_decimal(d: Decimal | None) -> str | None:
    """Exact plain-notation string for a Decimal (no exponent, trailing zeros of the scale removed)."""
    if d is None:
        return None
    if d == d.to_integral_value():
        return f"{d.quantize(Decimal(1)):f}"
    return f"{d.normalize():f}"
