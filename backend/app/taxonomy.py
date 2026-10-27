"""Canonical financial taxonomy: categories, metrics and statement types.

This is the single source of truth for every category string the pipeline emits
(chunk ``top_category``, ``category_scores`` keys, table-row categories, field
categories, signal hints and reference-dataset labels). Older aliases found in
stored data or the reference JSONL are migrated through :data:`CATEGORY_ALIASES`.

Row labels and narrative phrases are mapped to canonical metric names with a
synonym dictionary first; embedding similarity is only used as a fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class FinancialCategory(str, Enum):
    INCOME_REVENUE = "income_revenue"
    EXPENSES_OPERATING_COSTS = "expenses_operating_costs"
    FINANCIAL_PERFORMANCE = "financial_performance"
    FINANCIAL_STATEMENTS = "financial_statements"
    ASSETS_LIABILITIES_INVESTMENTS = "assets_liabilities_investments"
    CASH_FLOW_BANK_TRANSACTIONS = "cash_flow_bank_transactions"
    INVOICES_AND_BILLING = "invoices_and_billing"
    LOANS_AND_INTEREST = "loans_and_interest"
    TAX_AND_PAYROLL = "tax_and_payroll"
    INSURANCE_AND_CLAIMS = "insurance_and_claims"
    RENT_LEASES_RECURRING_PAYMENTS = "rent_leases_recurring_payments"
    DEPOSITS_AND_ESCROW = "deposits_and_escrow"
    CONTRACTUAL_FINANCIAL_OBLIGATIONS = "contractual_financial_obligations"
    MATURITY_RENEWALS_DEADLINES = "maturity_renewals_deadlines"
    ADMINISTRATIVE_CONTENT = "administrative_content"
    NON_FINANCIAL_CONTENT = "non_financial_content"
    UNCERTAIN = "uncertain"


C = FinancialCategory

#: Categories that mean "this content is not financial".
NON_FINANCIAL_CATEGORIES = frozenset({C.ADMINISTRATIVE_CONTENT.value, C.NON_FINANCIAL_CONTENT.value})

#: Old / alternative names -> canonical value. Applied to stored results and reference labels.
CATEGORY_ALIASES: dict[str, str] = {
    "invoices_purchase_orders": C.INVOICES_AND_BILLING.value,
    "accounts_payable_receivable": C.INVOICES_AND_BILLING.value,
    "billing_statements": C.INVOICES_AND_BILLING.value,
    "bank_transactions": C.CASH_FLOW_BANK_TRANSACTIONS.value,
    "cash_flow": C.CASH_FLOW_BANK_TRANSACTIONS.value,
    "taxes_duties": C.TAX_AND_PAYROLL.value,
    "payroll_benefits": C.TAX_AND_PAYROLL.value,
    "loans_interest_repayment": C.LOANS_AND_INTEREST.value,
    "insurance": C.INSURANCE_AND_CLAIMS.value,
    "investments_and_assets": C.ASSETS_LIABILITIES_INVESTMENTS.value,
    "contracts_payment_obligations": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS.value,
    "penalties_late_fees": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS.value,
    "refunds_credits_settlements": C.CONTRACTUAL_FINANCIAL_OBLIGATIONS.value,
    "non_financial": C.NON_FINANCIAL_CONTENT.value,
}

_CANONICAL_VALUES = frozenset(c.value for c in FinancialCategory)


def canonicalize_category(cat: str | None) -> str:
    """Return the canonical category for ``cat`` (unknown strings pass through unchanged)."""
    if not cat:
        return C.UNCERTAIN.value
    if isinstance(cat, FinancialCategory):
        return cat.value
    key = cat.strip().lower()
    if key in _CANONICAL_VALUES:
        return key
    return CATEGORY_ALIASES.get(key, cat)


def canonicalize_scores(scores: dict[str, float] | None) -> dict[str, float]:
    """Merge category score dicts keyed by aliases into canonical keys (max wins)."""
    out: dict[str, float] = {}
    for k, v in (scores or {}).items():
        ck = canonicalize_category(k)
        out[ck] = max(out.get(ck, float("-inf")), float(v))
    return out


# --------------------------------------------------------------------------- statement types
class StatementType(str, Enum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    HIGHLIGHTS = "highlights"
    NOTES = "notes"
    NARRATIVE = "narrative"
    OTHER = "other"


_STATEMENT_HEADING_PATTERNS: list[tuple[re.Pattern[str], StatementType]] = [
    (re.compile(r"\bcash\s+flows?\b", re.I), StatementType.CASH_FLOW_STATEMENT),
    (re.compile(r"\bbalance\s+sheets?\b|\bstatements?\s+of\s+financial\s+position\b|\bfinancial\s+condition\b", re.I), StatementType.BALANCE_SHEET),
    (re.compile(r"\bstatements?\s+of\s+(?:income|operations|earnings|comprehensive\s+income)\b|\bincome\s+statements?\b|\bprofit\s+(?:and|&)\s+loss\b|\bp\s*&\s*l\b", re.I), StatementType.INCOME_STATEMENT),
    (re.compile(r"\bhighlights\b|\bkey\s+(?:metrics|figures|indicators)\b|\bkpis?\b|\bat\s+a\s+glance\b|\bselected\s+financial\s+data\b|\bexecutive\s+summary\b", re.I), StatementType.HIGHLIGHTS),
    (re.compile(r"\bnotes?\b(?:\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements)?|\bnote\s+\d+", re.I), StatementType.NOTES),
]


def detect_statement_type(
    headings: list[str] | None,
    metrics: list[str] | None = None,
    *,
    is_table: bool = False,
    has_change_column: bool = False,
) -> str:
    """Recognise the statement type from headings first, then from row-metric patterns."""
    for h in reversed([h for h in (headings or []) if h]):
        for pat, st in _STATEMENT_HEADING_PATTERNS:
            if pat.search(h):
                # "Notes" inside a table heading is unusual; a table under a notes heading is still notes.
                return st.value
    m = set(metrics or [])
    if {"total_assets", "total_liabilities"} & m and len({"total_assets", "total_liabilities", "equity", "total_liabilities_and_equity"} & m) >= 2:
        return StatementType.BALANCE_SHEET.value
    if "cfo" in m and ({"cfi", "cff", "capex"} & m):
        return StatementType.CASH_FLOW_STATEMENT.value
    if "revenue" in m and "net_income" in m and ({"cost_of_revenue", "total_operating_expenses", "income_tax"} & m):
        return StatementType.INCOME_STATEMENT.value
    if is_table and (has_change_column or len(m) >= 3):
        return StatementType.HIGHLIGHTS.value
    if not is_table and headings:
        return StatementType.NARRATIVE.value
    return StatementType.OTHER.value


# --------------------------------------------------------------------------- canonical metrics
@dataclass(frozen=True)
class MetricDef:
    name: str
    category: str
    synonyms: tuple[str, ...]
    data_type: str = "monetary"  # monetary | percentage | per_share | shares | year


def _m(name: str, cat: C, *syn: str, data_type: str = "monetary") -> MetricDef:
    return MetricDef(name, cat.value, tuple(syn), data_type)


METRICS: list[MetricDef] = [
    # income statement
    _m("revenue", C.INCOME_REVENUE, "revenue", "revenues", "total revenue", "total revenues", "net revenue",
       "net revenues", "total net revenue", "total net revenues", "net sales", "sales", "total sales", "turnover"),
    _m("subscription_revenue", C.INCOME_REVENUE, "subscription revenue", "software subscription revenue",
       "cloud subscription revenue", "subscription and support revenue", "subscription"),
    _m("services_revenue", C.INCOME_REVENUE, "professional services revenue", "professional services",
       "services revenue", "service revenue", "professional services and consulting", "consulting revenue"),
    _m("product_revenue", C.INCOME_REVENUE, "product revenue", "products revenue", "license revenue"),
    _m("cost_of_revenue", C.EXPENSES_OPERATING_COSTS, "cost of revenue", "cost of revenues", "cost of sales",
       "cost of goods sold", "cogs", "cost of goods sold cogs", "cost of services"),
    _m("gross_profit", C.FINANCIAL_PERFORMANCE, "gross profit", "gross income"),
    _m("research_and_development", C.EXPENSES_OPERATING_COSTS, "research and development", "r and d",
       "research and engineering", "r and d investment", "research and development expense", "product development"),
    _m("sales_and_marketing", C.EXPENSES_OPERATING_COSTS, "sales and marketing", "selling and marketing",
       "s and m", "marketing and sales"),
    _m("general_and_administrative", C.EXPENSES_OPERATING_COSTS, "general and administrative", "g and a",
       "administrative expenses"),
    _m("sga", C.EXPENSES_OPERATING_COSTS, "selling general and administrative", "sg and a"),
    _m("total_operating_expenses", C.EXPENSES_OPERATING_COSTS, "total operating expenses", "operating expenses",
       "total opex", "opex", "total costs and expenses"),
    _m("operating_income", C.FINANCIAL_PERFORMANCE, "operating income", "income from operations",
       "operating profit", "ebit", "operating income ebit", "operating loss"),
    _m("other_income", C.INCOME_REVENUE, "interest and other income net", "interest and other income",
       "other income net", "other income", "other income expense net", "non operating income", "interest income"),
    _m("interest_expense", C.LOANS_AND_INTEREST, "interest expense", "net interest expense", "interest expense net"),
    _m("pretax_income", C.FINANCIAL_PERFORMANCE, "income before taxes", "income before income taxes",
       "pretax income", "pre tax income", "profit before tax", "earnings before taxes", "earnings before income taxes"),
    _m("income_tax", C.TAX_AND_PAYROLL, "provision for income taxes", "income tax expense", "income taxes",
       "income tax", "tax expense", "provision for taxes", "income tax provision"),
    _m("net_income", C.FINANCIAL_PERFORMANCE, "net income", "net profit", "net earnings", "profit after tax",
       "net income after tax", "net loss"),
    _m("ebitda", C.FINANCIAL_PERFORMANCE, "ebitda", "adjusted ebitda"),
    _m("eps_diluted", C.FINANCIAL_PERFORMANCE, "diluted eps", "diluted earnings per share", "eps diluted",
       "earnings per share diluted", "diluted net income per share", "net income per share diluted",
       data_type="per_share"),
    _m("eps_basic", C.FINANCIAL_PERFORMANCE, "basic eps", "basic earnings per share", "eps basic",
       "earnings per share basic", "earnings per share", "eps", data_type="per_share"),
    _m("dividend_per_share", C.FINANCIAL_PERFORMANCE, "dividends per share", "dividend per share",
       "dividends declared per share", "cash dividends per share", data_type="per_share"),
    # margins / ratios (percentages)
    _m("gross_margin", C.FINANCIAL_PERFORMANCE, "gross margin", data_type="percentage"),
    _m("operating_margin", C.FINANCIAL_PERFORMANCE, "operating margin", data_type="percentage"),
    _m("net_margin", C.FINANCIAL_PERFORMANCE, "net margin", "net profit margin", data_type="percentage"),
    # cash flow
    _m("cfo", C.CASH_FLOW_BANK_TRANSACTIONS, "cash from operations", "cash flow from operations",
       "operating cash flow", "net cash provided by operating activities", "cash provided by operating activities",
       "net cash from operating activities", "cash generated from operations"),
    _m("cfi", C.CASH_FLOW_BANK_TRANSACTIONS, "net cash used in investing activities", "cash used in investing activities",
       "net cash from investing activities"),
    _m("cff", C.CASH_FLOW_BANK_TRANSACTIONS, "net cash used in financing activities", "cash used in financing activities",
       "net cash from financing activities"),
    _m("capex", C.CASH_FLOW_BANK_TRANSACTIONS, "capital expenditures", "capital expenditure", "capex",
       "purchases of property and equipment"),
    _m("free_cash_flow", C.CASH_FLOW_BANK_TRANSACTIONS, "free cash flow"),
    # balance sheet
    _m("cash", C.ASSETS_LIABILITIES_INVESTMENTS, "cash and cash equivalents", "cash and equivalents", "cash"),
    _m("accounts_receivable", C.ASSETS_LIABILITIES_INVESTMENTS, "accounts receivable", "accounts receivable net",
       "trade receivables", "receivables net", "trade and other receivables"),
    _m("prepaid_and_other_current_assets", C.ASSETS_LIABILITIES_INVESTMENTS, "prepaid and other current assets",
       "prepaid expenses and other current assets", "prepaid expenses", "prepaid expenses other current assets",
       "other current assets"),
    _m("inventory", C.ASSETS_LIABILITIES_INVESTMENTS, "inventory", "inventories"),
    _m("total_current_assets", C.ASSETS_LIABILITIES_INVESTMENTS, "total current assets", "current assets"),
    _m("ppe", C.ASSETS_LIABILITIES_INVESTMENTS, "property and equipment net", "property and equipment",
       "property plant and equipment net", "property plant and equipment", "ppe"),
    _m("goodwill_and_intangibles", C.ASSETS_LIABILITIES_INVESTMENTS, "goodwill and intangibles",
       "goodwill and intangible assets", "goodwill", "intangible assets", "intangibles"),
    _m("capitalized_software", C.ASSETS_LIABILITIES_INVESTMENTS, "capitalized software development",
       "capitalized software"),
    _m("other_assets", C.ASSETS_LIABILITIES_INVESTMENTS, "other assets", "other non current assets",
       "other long term assets"),
    _m("total_assets", C.ASSETS_LIABILITIES_INVESTMENTS, "total assets"),
    _m("accounts_payable", C.ASSETS_LIABILITIES_INVESTMENTS, "accounts payable", "trade payables",
       "trade and other payables"),
    _m("accrued_expenses", C.ASSETS_LIABILITIES_INVESTMENTS, "accrued expenses", "accrued liabilities",
       "accruals", "accrued expenses and other current liabilities"),
    _m("deferred_revenue", C.ASSETS_LIABILITIES_INVESTMENTS, "deferred revenue", "unearned revenue",
       "contract liabilities"),
    _m("total_current_liabilities", C.ASSETS_LIABILITIES_INVESTMENTS, "total current liabilities",
       "current liabilities"),
    _m("long_term_debt", C.ASSETS_LIABILITIES_INVESTMENTS, "long term debt", "long term borrowings",
       "long term debt net", "senior notes", "notes payable", "borrowings"),
    _m("other_long_term_liabilities", C.ASSETS_LIABILITIES_INVESTMENTS, "other long term liabilities",
       "other non current liabilities", "other liabilities"),
    _m("total_liabilities", C.ASSETS_LIABILITIES_INVESTMENTS, "total liabilities"),
    _m("equity", C.ASSETS_LIABILITIES_INVESTMENTS, "total stockholders equity", "stockholders equity",
       "total shareholders equity", "shareholders equity", "total equity", "stockholders equity and retained earnings",
       "stockholders equity retained earnings"),
    _m("total_liabilities_and_equity", C.ASSETS_LIABILITIES_INVESTMENTS, "total liabilities and equity",
       "total liabilities and stockholders equity", "total liabilities and shareholders equity"),
    # liquidity / capital / debt facts usually found in narrative
    _m("net_cash", C.ASSETS_LIABILITIES_INVESTMENTS, "net cash position", "net cash"),
    _m("share_repurchases", C.ASSETS_LIABILITIES_INVESTMENTS, "share repurchases", "share repurchase",
       "stock repurchases", "share buybacks", "buybacks"),
    _m("shares_repurchased", C.ASSETS_LIABILITIES_INVESTMENTS, "repurchased", "shares repurchased",
       data_type="shares"),
    _m("average_repurchase_price", C.ASSETS_LIABILITIES_INVESTMENTS, "average price", "average repurchase price",
       data_type="per_share"),
    _m("credit_facility", C.LOANS_AND_INTEREST, "revolving credit facility", "credit facility",
       "revolving facility", "credit line", "line of credit"),
    _m("interest_rate", C.LOANS_AND_INTEREST, "bearing interest at", "interest rate", "interest at", "coupon",
       data_type="percentage"),
    _m("debt_maturity_year", C.MATURITY_RENEWALS_DEADLINES, "due", data_type="year"),
    _m("credit_facility_maturity_year", C.MATURITY_RENEWALS_DEADLINES, "maturing", data_type="year"),
]

METRICS_BY_NAME: dict[str, MetricDef] = {m.name: m for m in METRICS}

# Synonyms that only make sense with surrounding sentence context are never used for row labels.
_NARRATIVE_ONLY = {"repurchased", "average price", "bearing interest at", "interest at", "due", "maturing", "coupon"}
# Short generic words that identify a row only when they are the whole label ("Cash", not "Restricted cash").
_EXACT_ONLY_FOR_ROWS = {"cash", "sales", "borrowings", "subscription"}


def normalize_label(label: str) -> str:
    """Lower-case, unify '&'/'and', drop punctuation (incl. apostrophes) and collapse spaces."""
    s = label.lower()
    s = s.replace("’", "'").replace("&", " and ")
    s = re.sub(r"\(\s*(?:note\s*)?\d+\s*\)", " ", s)  # footnote markers like "(1)"
    s = re.sub(r"'s\b", "s", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_SYNONYM_INDEX: dict[str, str] = {}
for _md in METRICS:
    for _syn in _md.synonyms:
        _SYNONYM_INDEX.setdefault(normalize_label(_syn), _md.name)
# longest synonyms first so "cost of revenue" beats "revenue"
_SYNONYMS_BY_LENGTH: list[tuple[str, str]] = sorted(_SYNONYM_INDEX.items(), key=lambda kv: -len(kv[0]))


@dataclass
class MetricMatch:
    metric: str | None
    method: str  # synonym | contains | embedding | none
    score: float

    @property
    def category(self) -> str | None:
        return METRICS_BY_NAME[self.metric].category if self.metric else None


def match_metric(label: str, *, allow_embedding: bool = True, narrative: bool = False) -> MetricMatch:
    """Map a row label (or phrase) to a canonical metric.

    Order: exact synonym -> longest whole-word synonym contained in the label -> embedding fallback.
    """
    norm = normalize_label(label)
    if not norm:
        return MetricMatch(None, "none", 0.0)
    hit = _SYNONYM_INDEX.get(norm)
    if hit and (narrative or norm not in _NARRATIVE_ONLY):
        return MetricMatch(hit, "synonym", 1.0)
    padded = f" {norm} "
    for syn, name in _SYNONYMS_BY_LENGTH:
        if not narrative and (syn in _NARRATIVE_ONLY or syn in _EXACT_ONLY_FOR_ROWS):
            continue
        if f" {syn} " in padded:
            return MetricMatch(name, "contains", 0.85)
    if allow_embedding:
        return _embedding_fallback(norm)
    return MetricMatch(None, "none", 0.0)


_EMB_STATE: dict[str, object] = {}
_EMBEDDING_MIN_SIM = 0.72


def _embedding_fallback(norm_label: str) -> MetricMatch:
    try:
        import numpy as np

        from app.embeddings import get_embedding_service

        svc = get_embedding_service()
        model_id = svc.active_model_id()
        if _EMB_STATE.get("model") != model_id:
            syns = [s for s, _ in _SYNONYMS_BY_LENGTH if s not in _NARRATIVE_ONLY and s not in _EXACT_ONLY_FOR_ROWS]
            _EMB_STATE["model"] = model_id
            _EMB_STATE["names"] = [_SYNONYM_INDEX[s] for s in syns]
            _EMB_STATE["matrix"] = svc.embed_texts(syns).vectors
        vec = svc.embed_texts([norm_label]).vectors[0]
        sims = _EMB_STATE["matrix"] @ vec  # type: ignore[operator]
        best = int(np.argmax(sims))
        sim = float(sims[best])
        if sim >= _EMBEDDING_MIN_SIM:
            return MetricMatch(_EMB_STATE["names"][best], "embedding", round(sim, 4))  # type: ignore[index]
    except Exception as exc:  # embeddings are optional for metric mapping
        log.debug("metric_embedding_fallback_failed", extra={"error": str(exc)})
    return MetricMatch(None, "none", 0.0)


def metric_category(metric: str | None) -> str | None:
    md = METRICS_BY_NAME.get(metric or "")
    return md.category if md else None


# --------------------------------------------------------------------------- periods
_QUARTER_RE = re.compile(r"\b(?:Q([1-4])\s*(?:FY\s*)?['’]?((?:19|20)?\d\d)\b|((?:19|20)\d\d)\s*Q([1-4])\b)", re.I)
_YEAR_RE = re.compile(r"(?<![\d.,])((?:19|20)\d\d)(?![\d.,]\d)")
_FY_SHORT_RE = re.compile(r"\bFY\s*['’]?(\d\d)\b", re.I)
_NON_PERIOD_RE = re.compile(r"\b(change|chg|variance|var\.?|growth|yoy|y/y|%|delta|diff)\b", re.I)


def normalize_period(label: str | None) -> str | None:
    """Map a period column / phrase to a comparable key: 'FY2025', 'Q3-2026'. Change columns -> None."""
    if not label:
        return None
    s = label.strip()
    q = _QUARTER_RE.search(s)
    if q:
        num, yr = (q.group(1), q.group(2)) if q.group(1) else (q.group(4), q.group(3))
        yr = yr if len(yr) == 4 else f"20{yr}"
        return f"Q{num}-{yr}"
    years = _YEAR_RE.findall(s)
    if len(set(years)) == 1:
        return f"FY{years[0]}"
    fy = _FY_SHORT_RE.search(s)
    if fy:
        return f"FY20{fy.group(1)}"
    return None


def is_change_column(label: str | None) -> bool:
    return bool(label) and normalize_period(label) is None and bool(_NON_PERIOD_RE.search(label or ""))


_FISCAL_PATTERNS = [
    re.compile(r"fiscal\s+period\s*:\s*(Q[1-4]\s*(?:19|20)\d\d)", re.I),
    re.compile(r"(?:fiscal\s+)?(?:year|twelve\s+months)\s+ended\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+((?:19|20)\d\d))", re.I),
    re.compile(r"\b(?:fiscal|FY)\s*((?:19|20)\d\d)\b", re.I),
    re.compile(r"\b((?:19|20)\d\d)\s+annual\s+report\b", re.I),
]


def detect_fiscal_period(text: str) -> dict | None:
    """Detect the document's primary reporting period, e.g. {'key': 'FY2025', 'label': 'December 31, 2025'}."""
    for pat in _FISCAL_PATTERNS:
        m = pat.search(text or "")
        if m:
            label = " ".join(m.group(1).split())
            key = normalize_period(label)
            if key:
                return {"key": key, "label": label}
    return None


def shift_period(key: str | None, years: int) -> str | None:
    """'FY2025' -1 -> 'FY2024'; 'Q3-2026' -1 -> 'Q3-2025'."""
    if not key:
        return None
    m = re.fullmatch(r"(FY|Q[1-4]-)(\d{4})", key)
    if not m:
        return None
    return f"{m.group(1)}{int(m.group(2)) + years}"
