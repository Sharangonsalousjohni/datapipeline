"""Stage E.1: Inexpensive Rule/Pattern Signals for Financial Relevance.

Detects:
- Currency & amount patterns (including OCR-noisy numbers and currency codes).
- Financial terminology mapped to the canonical categories in ``app.taxonomy``.
- Payment and transaction phrases (even without currency symbols).
- Relevant table headers (Invoice, Debit, Credit, Amount, Balance, Principal, etc.).
- Dates associated with payment terms / deadlines / maturity.
- Financial relationships and obligation indicators (Borrower, Lender, Lessee, shall pay, etc.).
- Negative / non-financial context indicators (scientific senses of 'charge'/'interest'/'bond'/'deposit',
  code variables like $HOME / ${var}, ordinary non-financial contract boilerplate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import SignalEvidence
from app.taxonomy import canonicalize_category

__all__ = ["SignalAnalysis", "analyze_signals", "canonicalize_category"]


@dataclass
class SignalAnalysis:
    signal_score: float
    positive_signals: list[SignalEvidence] = field(default_factory=list)
    negative_signals: list[SignalEvidence] = field(default_factory=list)
    category_hints: dict[str, float] = field(default_factory=dict)
    feature_counts: dict[str, float] = field(default_factory=dict)


# 1. Currency and amount patterns (handles symbols, ISO codes, OCR '0'/'O'/'1'/'l'/'S' noise)
_CURRENCY_AMOUNT_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    r"(?:US\$|C\$|A\$|S\$|HK\$|NZ\$|[\$€£¥₹])\s*\d[\d,.\sOolS]{0,14}(?:\s*(?:million|billion|thousand|k|m|bn|cr|lakh))?"
    r"|\b(?:USD|EUR|GBP|INR|JPY|CAD|AUD|CHF|CNY|SGD|Rs\.?)\s*\d[\d,.\sOolS]{0,14}(?:\s*(?:million|billion|thousand|k|m|bn|cr|lakh))?"
    r"|\b\d[\d,.]{0,12}\s*(?:USD|EUR|GBP|INR|JPY|CAD|AUD|CHF|CNY|SGD|dollars|euros|pounds|rupees|yen|cents)\b"
    r")",
    re.I,
)

# 2. Code / shell / LaTeX / regex false currency patterns ($HOME, shell $1 without decimals, ${var}, $x$)
_CODE_DOLLAR_RE = re.compile(
    r"(\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z0-9_.:-]+\}|\$[0-9](?![0-9,.]|\s*(?:million|billion|thousand|k|m|bn))\b|\$\([^\)]+\)|\^[A-Za-z0-9.*+?]+\$|\$[a-zA-Z]\$)"
)

# 3. Payment & transaction phrases (including obligations expressed without currency symbols)
_PAYMENT_PHRASE_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (
        re.compile(
            r"\b(shall\s+(?:pay|repay|remit|reimburse|compensate)|agrees\s+to\s+pay|obligation\s+to\s+pay|payment\s+obligation|promise[sd]?\s+to\s+pay)\b",
            re.I,
        ),
        "contractual_financial_obligations",
        0.32,
    ),
    (
        re.compile(
            r"\b(equal\s+monthly\s+installments|monthly\s+installments|repayment\s+schedule|amortization\s+schedule|principal\s+and\s+interest|promissory\s+note|loan\s+facility|credit\s+facility)\b",
            re.I,
        ),
        "loans_and_interest",
        0.34,
    ),
    (
        re.compile(
            r"\b(net\s*(?:15|30|45|60|90)\s*(?:days)?|due\s+upon\s+receipt|amount\s+due|total\s+due|balance\s+due|remit\s+to|wire\s+transfer|ach\s+transfer|purchase\s+order|invoice\s*(?:no|number|#)|lnvoice|t0tal\s+due)\b",
            re.I,
        ),
        "invoices_and_billing",
        0.34,
    ),
    (
        re.compile(
            r"\b(accounts\s+payable|accounts\s+receivable|aged\s+receivables|aged\s+payables|aging\s+schedule|trade\s+payables|trade\s+receivables|vendor\s+ledger)\b",
            re.I,
        ),
        "invoices_and_billing",
        0.34,
    ),
    (
        re.compile(
            r"\b(monthly\s+rent|base\s+rent|rent\s+escalation|rental\s+consideration|annual\s+retainer|subscription\s+fee|recurring\s+(?:fee|payment|charge))\b",
            re.I,
        ),
        "rent_leases_recurring_payments",
        0.34,
    ),
    (
        re.compile(
            r"\b(security\s+deposit|damage\s+deposit|escrow\s+deposit|held\s+in\s+designated\s+escrow|deposit\s+with\s+the\s+landlord)\b",
            re.I,
        ),
        "deposits_and_escrow",
        0.34,
    ),
    (
        re.compile(
            r"\b(lease\s+term|term\s+of\s+this\s+lease|occupancy\s+period|commencing\s*:\s*\[|lease\s+commencement|expiration\s+date|subletting\s+prohibited)\b",
            re.I,
        ),
        "contractual_financial_obligations",
        0.34,
    ),
    (
        re.compile(
            r"\b(late\s+(?:fee|payment|charge|penalty)|overdue\s+balance|default\s+interest|liquidated\s+damages|past\s+due|service\s+charge\s+of\s+\d)\b",
            re.I,
        ),
        "contractual_financial_obligations",
        0.34,
    ),
    (
        re.compile(
            r"\b(insurance\s+premium|annual\s+premium|policy\s+deductible|coverage\s+limit|claim\s+payout|indemnity\s+limit|co-?pay(?:ment)?)\b",
            re.I,
        ),
        "insurance_and_claims",
        0.34,
    ),
    (
        re.compile(
            r"\b(withholding\s+tax|value\s+added\s+tax|sales\s+tax|corporate\s+income\s+tax|customs\s+duty|excise\s+duty|gst|vat\s+(?:rate|registration|amount)|tax\s+assessment|statutory\s+(?:levy|payment|contribution))\b",
            re.I,
        ),
        "tax_and_payroll",
        0.34,
    ),
    (
        re.compile(
            r"\b(credit\s+memo|credit\s+note|refund\s+(?:amount|policy|issued)|expense\s+reimbursement|rebate\s+amount|settlement\s+amount|chargeback)\b",
            re.I,
        ),
        "contractual_financial_obligations",
        0.32,
    ),
    (
        re.compile(
            r"\b(gross\s+revenue|net\s+revenue|recognized\s+revenue|deferred\s+revenue|royalty\s+(?:income|rate|payment)|dividend\s+income|licensing\s+fee)\b",
            re.I,
        ),
        "income_revenue",
        0.34,
    ),
    (
        re.compile(
            r"\b(operating\s+expenses|cost\s+of\s+goods\s+sold|cogs|selling,\s*general\s+and\s+administrative|sg&a|capital\s+expenditure|capex|opex|payroll\s+expense|depreciation\s+expense)\b",
            re.I,
        ),
        "expenses_operating_costs",
        0.34,
    ),
    (
        re.compile(
            r"\b(cash\s+and\s+cash\s+equivalents|operating\s+cash\s+flow|free\s+cash\s+flow|bank\s+reconciliation|"
            r"opening\s+balance|closing\s+balance|beginning\s+balance|ending\s+balance|statement\s+balance|"
            r"total\s+deposits|total\s+withdrawals|checking\s+statement|checking\s+account|"
            r"bank\s+statement|iban|swift\s*/?\s*bic)\b",
            re.I,
        ),
        "cash_flow_bank_transactions",
        0.34,
    ),
    (
        re.compile(
            r"\b(total\s+assets|total\s+liabilities|shareholders'?\s+equity|retained\s+earnings|goodwill|marketable\s+securities|fair\s+value|book\s+value|working\s+capital|balance\s+sheet)\b",
            re.I,
        ),
        "assets_liabilities_investments",
        0.34,
    ),
    (
        re.compile(
            r"\b(maturity\s+date|auto-?renewal|renewal\s+term|payment\s+deadline|grace\s+period|installment\s+due\s+date|amended\s+to\s+replace)\b",
            re.I,
        ),
        "maturity_renewals_deadlines",
        0.30,
    ),
]

# 4. Financial relationship / party / rate indicators
_OBLIGATION_PARTY_RE = re.compile(
    r"\b(borrower|lender|lessor|lessee|landlord|tenant|mortgagor|mortgagee|guarantor|creditor|debtor|payee|payer|licensor|licensee)\b",
    re.I,
)
_RATE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*%\s*(?:per\s+annum|p\.?a\.?|per\s+month|interest|apr|apy|discount|late\s+fee|royalty|commission|vat|gst|withholding)|(?:sofr|libor|euribor|prime\s+rate)\s*\+\s*\d+\s*bps)\b",
    re.I,
)
_PAYMENT_DATE_RE = re.compile(
    r"\b((?:due|payable|maturity|effective|invoice|billing|settlement)\s+(?:date|on|by)\s*[:\-]?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})|within\s+\d+\s+(?:business\s+|calendar\s+)?days\s+(?:of|from|after)\s+(?:receipt|invoice|delivery|billing))\b",
    re.I,
)

# 5. Table header patterns
_FINANCIAL_TABLE_HEADER_RE = re.compile(
    r"(^|\|\s*)(invoice\s*(?:no|#)?|unit\s+price|line\s+total|subtotal|amount(?:\s+due)?|debit|credit|balance|principal|interest|tax|vat|premium|deductible|fy\s*20\d\d|q[1-4]\s*20\d\d|revenue|expense|cost|budget|allocation|actual(?:\s+spend)?|variance|total|acct|account\s+code)(\s*\||$)",
    re.I | re.M,
)

# 6. Hard-negative disambiguation patterns (scientific, technical, non-financial legal boilerplate)
_SIGNATURE_ADMIN_RE = re.compile(
    r"\b(authorized\s+signature|tenant\s+signature|landlord\s+signature|signature\s+of\s+tenant|signature\s+of\s+landlord|"
    r"execution\s+and\s+acknowledgment|in\s+witness\s+whereof|duly\s+authorized\s+representative|"
    r"print\s+name\s*:|date\s*:\s*_{2,}|signed,\s*sealed\s*and\s*delivered|"
    r"residential\s+(?:lease|tenancy|rental)\s+agreement|standard\s+fixed-term\s+residential\s+lease)\b",
    re.I,
)

_NON_FINANCIAL_OPERATIONAL_RE = re.compile(
    r"\b(utilities,\s*upkeep,\s*and\s*maintenance|responsible\s+for\s+contracting\s+and\s+paying\s+all\s+utility\s+services|"
    r"keep\s+the\s+premises\s+clean|maintain\s+plumbing|quiet\s+enjoyment|alterations,\s*and\s*rules|"
    r"landlord\s+right\s+of\s+inspection\s+and\s+entry|governing\s+law\s+and\s+entire\s+agreement)\b",
    re.I,
)

_NEGATIVE_CONTEXT_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (
        re.compile(
            r"\b(electric(?:al)?\s+charge|surface\s+charge|point\s+charge|charge\s+density|positively\s+charged|negatively\s+charged|coulomb|electron\s+volt|ion\s+channel)\b",
            re.I,
        ),
        "scientific_charge",
        0.35,
    ),
    (
        re.compile(
            r"\b(chemical\s+bond|covalent\s+bond|hydrogen\s+bond|ionic\s+bond|disulfide\s+bond|peptide\s+bond|bond\s+angle|bond\s+dissociation)\b",
            re.I,
        ),
        "scientific_bond",
        0.35,
    ),
    (
        re.compile(
            r"\b(mineral\s+deposit|sedimentary\s+deposit|alluvial\s+deposit|glacial\s+deposit|calcium\s+deposit|fatty\s+deposit|carbon\s+deposit)\b",
            re.I,
        ),
        "scientific_deposit",
        0.35,
    ),
    (
        re.compile(
            r"\b(principal\s+investigator|principal\s+component\s+analysis|principal\s+quantum|school\s+principal|yield\s+strength|reaction\s+yield|crop\s+yield|quantum\s+yield|mass\s+balance|white\s+balance|nitrogen\s+balance|conflict\s+of\s+interest|research\s+interest|interest\s+of\s+safety)\b",
            re.I,
        ),
        "non_financial_polysemy",
        0.32,
    ),
    (
        re.compile(
            r"\b(confidential\s+information\s+shall\s+not\s+include|receiving\s+party\s+shall\s+hold|non-?disclosure\s+agreement|governing\s+law\s+and\s+jurisdiction|executed\s+in\s+counterparts|severability|force\s+majeure|entire\s+agreement\s+between\s+the\s+parties)\b",
            re.I,
        ),
        "non_financial_legal_boilerplate",
        0.22,
    ),
]


def analyze_signals(text: str, context: str | None = None, chunk_type: str = "prose") -> SignalAnalysis:
    full_text = text
    pos: list[SignalEvidence] = []
    neg: list[SignalEvidence] = []
    cat_hints: dict[str, float] = {}
    counts: dict[str, float] = {
        "currency_amount": 0.0,
        "payment_phrase": 0.0,
        "obligation_party": 0.0,
        "rate": 0.0,
        "payment_date": 0.0,
        "table_header": 0.0,
        "negative_context": 0.0,
        "code_dollar": 0.0,
    }

    # Code / regex dollar signs first so we don't confuse them with currency
    code_spans: list[tuple[int, int]] = []
    for m in _CODE_DOLLAR_RE.finditer(full_text):
        code_spans.append((m.start(), m.end()))
        counts["code_dollar"] += 1.0
        neg.append(
            SignalEvidence(
                signal="code_or_symbol_non_currency",
                text=m.group(0),
                start=m.start(),
                end=m.end(),
                weight=-0.20,
            )
        )

    def _overlaps_code(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in code_spans)

    for m in _CURRENCY_AMOUNT_RE.finditer(full_text):
        if _overlaps_code(m.start(), m.end()):
            continue
        counts["currency_amount"] += 1.0
        pos.append(
            SignalEvidence(
                signal="currency_amount",
                text=m.group(0).strip(),
                start=m.start(),
                end=m.end(),
                weight=0.28,
            )
        )

    for pat, cat, weight in _PAYMENT_PHRASE_PATTERNS:
        for m in pat.finditer(full_text):
            counts["payment_phrase"] += 1.0
            cat_hints[cat] = cat_hints.get(cat, 0.0) + weight
            pos.append(
                SignalEvidence(
                    signal="payment_phrase",
                    text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    category=cat,
                    weight=weight,
                )
            )

    is_admin_signature = bool(_SIGNATURE_ADMIN_RE.search(full_text)) and counts["currency_amount"] == 0
    is_non_fin_operational = bool(_NON_FINANCIAL_OPERATIONAL_RE.search(full_text)) and counts["currency_amount"] == 0

    if not is_admin_signature:
        for m in _OBLIGATION_PARTY_RE.finditer(full_text):
            counts["obligation_party"] += 1.0
            pos.append(
                SignalEvidence(
                    signal="financial_party",
                    text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    category="contractual_financial_obligations",
                    weight=0.16,
                )
            )
    else:
        cat_hints["administrative_content"] = 1.0
        neg.append(
            SignalEvidence(
                signal="administrative_signature_block",
                text="signature_block",
                start=0,
                end=len(full_text),
                weight=-0.35,
            )
        )

    if is_non_fin_operational:
        cat_hints["non_financial_content"] = 1.0
        neg.append(
            SignalEvidence(
                signal="non_financial_operational_clause",
                text="operational_clause",
                start=0,
                end=len(full_text),
                weight=-0.25,
            )
        )

    for m in _RATE_RE.finditer(full_text):
        counts["rate"] += 1.0
        pos.append(
            SignalEvidence(
                signal="financial_rate",
                text=m.group(0),
                start=m.start(),
                end=m.end(),
                category="loans_and_interest",
                weight=0.26,
            )
        )

    for m in _PAYMENT_DATE_RE.finditer(full_text):
        counts["payment_date"] += 1.0
        cat_hints["maturity_renewals_deadlines"] = cat_hints.get("maturity_renewals_deadlines", 0.0) + 0.22
        pos.append(
            SignalEvidence(
                signal="payment_date",
                text=m.group(0),
                start=m.start(),
                end=m.end(),
                category="maturity_renewals_deadlines",
                weight=0.22,
            )
        )

    if chunk_type == "table" or "|" in full_text:
        for m in _FINANCIAL_TABLE_HEADER_RE.finditer(full_text):
            counts["table_header"] += 1.0
            pos.append(
                SignalEvidence(
                    signal="table_header",
                    text=m.group(2),
                    start=m.start(2),
                    end=m.end(2),
                    weight=0.20,
                )
            )

    if context:
        for pat, cat, weight in _PAYMENT_PHRASE_PATTERNS:
            if pat.search(context):
                cat_hints[cat] = cat_hints.get(cat, 0.0) + weight * 0.5

    for pat, label, penalty in _NEGATIVE_CONTEXT_PATTERNS:
        for m in pat.finditer(full_text):
            counts["negative_context"] += 1.0
            neg.append(
                SignalEvidence(
                    signal=label,
                    text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    weight=-penalty,
                )
            )

    # Combine positive evidence with diminishing returns so one repeated term cannot saturate to 1.0
    raw_pos = (
        min(0.42, counts["currency_amount"] * 0.24)
        + min(0.55, counts["payment_phrase"] * 0.30)
        + min(0.24, counts["obligation_party"] * 0.14)
        + min(0.28, counts["rate"] * 0.24)
        + min(0.25, counts["payment_date"] * 0.22)
        + min(0.25, counts["table_header"] * 0.14)
    )
    # Require corroboration: a single isolated currency mention without any financial phrase/table/party/rate
    # is capped so news articles mentioning "$5 billion economy" do not automatically pass as RELEVANT.
    corroborated = (
        counts["payment_phrase"] > 0
        or counts["payment_date"] > 0
        or counts["table_header"] >= 2
        or (counts["obligation_party"] > 0 and (counts["currency_amount"] > 0 or counts["rate"] > 0))
        or (counts["currency_amount"] >= 2 and counts["rate"] > 0)
    )
    if not corroborated:
        raw_pos = min(raw_pos, 0.38)

    raw_neg = min(0.55, counts["negative_context"] * 0.30 + counts["code_dollar"] * 0.20)
    score = max(0.0, min(1.0, raw_pos - raw_neg))
    if is_admin_signature:
        score = min(score, 0.12)
    elif is_non_fin_operational:
        score = min(score, 0.22)

    pos.sort(key=lambda s: (s.start, -s.weight))
    return SignalAnalysis(
        signal_score=round(score, 4),
        positive_signals=pos[:25],
        negative_signals=neg[:15],
        category_hints=cat_hints,
        feature_counts=counts,
    )
