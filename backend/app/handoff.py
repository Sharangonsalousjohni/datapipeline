"""Per-chunk financial content analysis and the document-level extraction handoff.

One code path builds fields for every consumer (classifier reconciliation, the
``/relevant-chunks`` handoff, the structured-extraction report and the tests):

1. :func:`analyze_chunk_content` extracts table rows/fields or narrative facts for a chunk.
2. :func:`reconcile_decision` derives the chunk decision and category from that content
   (a chunk with a RELEVANT table row can never be UNCERTAIN; its category is the
   count-weighted main category of its rows; structural chunks are non-financial).
3. :func:`extract_document` runs the document-level validation engine over all fields.
4. :func:`build_handoff` turns the result into :class:`ExtractionHandoff` items, keeping cover
   pages / TOCs / signature blocks and listing every excluded chunk with a reason.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.chunking import NON_FINANCIAL_ROLES, ROLE_SIGNATURE
from app.contracts import extract_lease_fields
from app.financial_extraction import (
    detect_document_type,
    extract_generic_financial_fields,
    extract_rental_agreement_fields,
    extract_summary_metrics,
    extract_table_fields,
)
from app.narrative import extract_narrative_facts
from app.numbers import UnitSpec
from app.schemas import (
    ExtractedFinancialField,
    ExtractionHandoff,
    MatchedReference,
    RelevantChunksResponse,
    SignalEvidence,
    SourceLocation,
    StructuredExtractionItem,
    StructuredExtractionReport,
    TableRowClassification,
    ValidationSummary,
)
from app.taxonomy import (
    METRICS_BY_NAME,
    FinancialCategory,
    StatementType,
    canonicalize_category,
    canonicalize_scores,
    detect_fiscal_period,
)

LEASE_DOCUMENT_TYPES = frozenset({"commercial_lease_agreement"})


@dataclass
class ChunkContent:
    rows: list[TableRowClassification] = field(default_factory=list)
    fields: list[ExtractedFinancialField] = field(default_factory=list)
    statement_type: str | None = None
    role: str | None = None


@dataclass
class ChunkView:
    """Format-independent view of a stored or freshly classified chunk."""

    id: str
    text: str
    location: dict[str, Any]
    chunk_type: str = "prose"
    context: str | None = None
    extraction_method: str = "unknown"
    ocr_confidence: float | None = None
    decision: str = "IRRELEVANT"
    top_category: str | None = None
    category_scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    semantic_score: float = 0.0
    signal_score: float = 0.0
    matched_references: list[Any] = field(default_factory=list)
    signals: list[Any] = field(default_factory=list)
    classifier_version: str = "unknown"
    embedding_model: str = "unknown"
    dataset_version: str = "unknown"


@dataclass
class ChunkResult:
    chunk: ChunkView
    content: ChunkContent
    decision: str
    category: str | None
    category_scores: dict[str, float]
    decision_basis: str


# --------------------------------------------------------------------------- content analysis
def analyze_chunk_content(
    text: str,
    location: dict[str, Any] | None,
    chunk_type: str,
    *,
    document_type: str | None = None,
    default_currency: str = "USD",
    allow_embedding: bool = True,
) -> ChunkContent:
    loc = location or {}
    role = loc.get("chunk_role")
    if role in NON_FINANCIAL_ROLES:
        return ChunkContent(statement_type=loc.get("statement_type"), role=role)
    if document_type in LEASE_DOCUMENT_TYPES:
        # Leases: contract terms from clauses and from rent schedule / party tables.
        rows, fields = extract_lease_fields(text, loc, chunk_type)
        if chunk_type == "table" and not rows and not fields:
            rows, fields = extract_table_fields(text, loc, default_currency=default_currency, allow_embedding=allow_embedding)
        return ChunkContent(rows=rows, fields=fields, statement_type=StatementType.LEASE.value, role=role)
    page = loc.get("page")
    if chunk_type == "table" or ("|" in text and "\n" in text and role in (None, "table")):
        rows, fields = extract_table_fields(text, loc, default_currency=default_currency, allow_embedding=allow_embedding)
        return ChunkContent(rows=rows, fields=fields, statement_type=loc.get("statement_type"), role=role or "table")
    if document_type == "residential_rental_agreement":
        return ChunkContent(fields=extract_rental_agreement_fields(text, context=loc.get("section_heading")), role=role)
    fields: list[ExtractedFinancialField] = []
    doc_unit = UnitSpec.from_dict(loc.get("unit"))
    kpis = extract_summary_metrics(
        text,
        default_currency=default_currency,
        scale=doc_unit.scale if doc_unit and doc_unit.known else "units",
        scale_multiplier=float(doc_unit.multiplier) if doc_unit and doc_unit.known else 1.0,
        page_num=page,
    )
    fields.extend(kpis)
    if document_type == "invoice":
        fields.extend(extract_generic_financial_fields(text))
    fiscal = loc.get("fiscal_period") or {}
    overlap = loc.get("overlap")
    facts = extract_narrative_facts(
        text,
        default_period=fiscal.get("key"),
        default_currency=default_currency,
        page=page,
        skip_span=tuple(overlap) if overlap else None,
        statement_type=loc.get("statement_type"),
    )
    kpi_texts = {k.original_text for k in kpis}
    fields.extend(f for f in facts if f.original_text not in kpi_texts)
    return ChunkContent(fields=fields, statement_type=loc.get("statement_type"), role=role)


def reconcile_decision(
    decision: str,
    category: str | None,
    category_scores: dict[str, float] | None,
    content: ChunkContent,
) -> tuple[str, str | None, dict[str, float], str]:
    """Derive the chunk decision/category from its content. Returns (decision, category, scores, basis)."""
    scores = canonicalize_scores(category_scores)
    category = canonicalize_category(category) if category else None
    if content.role in NON_FINANCIAL_ROLES:
        cat = (FinancialCategory.ADMINISTRATIVE_CONTENT if content.role == ROLE_SIGNATURE else FinancialCategory.NON_FINANCIAL_CONTENT).value
        return "IRRELEVANT", cat, scores, "structural_role"
    if content.statement_type == StatementType.LEASE.value and content.fields:
        # A clause or table that yields contract terms is relevant, categorised by its terms.
        counts = Counter(f.category for f in content.fields if f.category)
        if counts:
            category = max(counts, key=lambda c: (counts[c], scores.get(c, 0.0)))
            total = sum(counts.values())
            scores = {**scores, **{c: max(scores.get(c, 0.0), round(n / total, 4)) for c, n in counts.items()}}
        return "RELEVANT", category, scores, "contract_terms"
    relevant_rows = [r for r in content.rows if r.decision == "RELEVANT"]
    if relevant_rows:
        mapped = [r.category for r in relevant_rows if r.metric]
        if mapped:
            counts = Counter(mapped)
            total = sum(counts.values())
            # count-weighted main category; ties go to the category the classifier scored higher
            category = max(counts, key=lambda c: (counts[c], scores.get(c, 0.0)))
            scores = {c: round(n / total, 4) for c, n in counts.most_common()}
        return "RELEVANT", category, scores, "table_rows"
    linked = [
        f for f in content.fields
        if f.metric in METRICS_BY_NAME and f.data_type in ("monetary", "percentage", "per_share", "shares", "basis_points")
    ]
    if linked:
        counts = Counter(f.category for f in linked if f.category)
        if counts:
            total = sum(counts.values())
            category = max(counts, key=lambda c: (counts[c], scores.get(c, 0.0)))
            scores = {**scores, **{c: max(scores.get(c, 0.0), round(n / total, 4)) for c, n in counts.items()}}
        return "RELEVANT", category, scores, "narrative_facts"
    return decision, category, scores, "classifier"


# --------------------------------------------------------------------------- document level
def extract_document(
    document_id: str,
    filename: str,
    document_decision: str,
    chunks: list[ChunkView],
    *,
    default_currency: str = "USD",
) -> tuple[list[ChunkResult], StructuredExtractionReport]:
    sample = " ".join(c.text for c in chunks[:6])
    doc_type = detect_document_type(sample, filename)
    # Chunks stored before chunk metadata existed carry no fiscal period; detect it once for the document.
    doc_fiscal = next((c.location.get("fiscal_period") for c in chunks if c.location.get("fiscal_period")), None)
    doc_fiscal = doc_fiscal or detect_fiscal_period("\n".join(c.text for c in chunks[:10]))
    results: list[ChunkResult] = []
    items: list[StructuredExtractionItem] = []
    for c in chunks:
        if doc_fiscal and not c.location.get("fiscal_period"):
            c.location = {**c.location, "fiscal_period": doc_fiscal}
        # Legacy behaviour for rental agreements: only classified-financial chunks are mined.
        if doc_type == "residential_rental_agreement" and c.decision == "IRRELEVANT":
            content = ChunkContent(role=c.location.get("chunk_role"))
        else:
            content = analyze_chunk_content(
                c.text, c.location, c.chunk_type, document_type=doc_type, default_currency=default_currency
            )
        decision, category, scores, basis = reconcile_decision(c.decision, c.top_category, c.category_scores, content)
        results.append(ChunkResult(c, content, decision, category, scores, basis))
        items.append(
            StructuredExtractionItem(
                chunk_id=c.id,
                statement_type=content.statement_type,
                chunk_role=content.role,
                page_number=c.location.get("page"),
                section_heading=c.location.get("section_heading"),
                decision=decision,
                category=category,
                source_text=c.text,
                relevance_score=float(c.score or 0.0),
                semantic_score=float(c.semantic_score or 0.0),
                signal_score=float(c.signal_score or 0.0),
                extraction_confidence=float(c.ocr_confidence if c.ocr_confidence is not None else 1.0),
                extraction_method=c.extraction_method,
                rule_evidence=[s if isinstance(s, SignalEvidence) else SignalEvidence(**s) for s in (c.signals or [])],
                financial_fields=content.fields,
                table_rows=content.rows,
                requires_review=decision == "UNCERTAIN",
            )
        )
    report = StructuredExtractionReport(
        document_id=document_id,
        document_type=doc_type,
        document_decision=document_decision,
        items=items,
        validation_summary=ValidationSummary(),
    )
    from app.validation import FinancialValidationEngine

    report = FinancialValidationEngine().validate_document(report)
    for res, item in zip(results, report.items):
        # validation may have replaced lists; keep the validated field objects
        res.content.fields = item.financial_fields
        res.content.rows = item.table_rows
        for row in res.content.rows:
            if any(f.validation_status == "FAILED" for f in row.fields):
                row.validation_status = "FAILED"
            elif any(f.review_required for f in row.fields):
                row.validation_status = "REVIEW_REQUIRED"
    return results, report


def _chunk_metadata(loc: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": loc.get("page"),
        "page_end": loc.get("page_end"),
        "section": loc.get("section_heading"),
        "heading_path": loc.get("heading_path"),
        "unit": loc.get("unit"),
        "period_columns": loc.get("period_columns"),
        "fiscal_period": loc.get("fiscal_period"),
        "statement_type": loc.get("statement_type"),
        "chunk_role": loc.get("chunk_role"),
        "table_bbox": loc.get("table_bbox"),
    }


def build_handoff(
    *,
    document_id: str,
    filename: str,
    source_type: str,
    relevance_status: str | None,
    extraction_quality: dict | None,
    chunks: list[ChunkView],
    include_uncertain: bool = True,
) -> RelevantChunksResponse:
    results, report = extract_document(document_id, filename, relevance_status or "NOT_FINANCIAL", chunks)
    allowed = {"RELEVANT", "UNCERTAIN"} if include_uncertain else {"RELEVANT"}
    items: list[ExtractionHandoff] = []
    excluded: list[dict[str, Any]] = []
    for res in results:
        c = res.chunk
        structural = res.content.role in NON_FINANCIAL_ROLES
        if res.decision not in allowed and not structural:
            excluded.append(
                {
                    "chunk_id": c.id,
                    "decision": res.decision,
                    "category": res.category,
                    "page": c.location.get("page"),
                    "section": c.location.get("section_heading"),
                    "reason": f"{res.decision} (score {c.score:.2f}, basis {res.decision_basis})",
                }
            )
            continue
        loc = dict(c.location)
        items.append(
            ExtractionHandoff(
                document_id=document_id,
                chunk_id=c.id,
                source_filename=filename,
                source_type=source_type,
                source_location=SourceLocation(**loc),
                extracted_text=c.text,
                context=c.context,
                financial_category=res.category,
                category_scores=res.category_scores,
                relevance_score=c.score,
                semantic_score=c.semantic_score or 0.0,
                signal_score=c.signal_score or 0.0,
                decision=res.decision,  # type: ignore[arg-type]
                matching_references=[
                    m if isinstance(m, MatchedReference) else MatchedReference(**{**m, "category": canonicalize_category(m.get("category"))})
                    for m in (c.matched_references or [])
                ],
                evidence=[s if isinstance(s, SignalEvidence) else SignalEvidence(**s) for s in (c.signals or [])],
                evidence_bboxes=loc.get("bboxes"),
                extraction_method=c.extraction_method,
                ocr_confidence=c.ocr_confidence,
                extraction_confidence=c.ocr_confidence if c.ocr_confidence is not None else 1.0,
                financial_fields=res.content.fields,
                table_rows=res.content.rows,
                extraction_quality=extraction_quality,
                classifier_version=c.classifier_version,
                embedding_model=c.embedding_model,
                dataset_version=c.dataset_version,
                statement_type=res.content.statement_type,
                chunk_role=res.content.role,
                decision_basis=res.decision_basis,
                chunk_metadata=_chunk_metadata(loc),
            )
        )
    return RelevantChunksResponse(
        document_id=document_id,
        relevance_status=relevance_status,  # type: ignore[arg-type]
        include_uncertain=include_uncertain,
        items=items,
        excluded_chunks=excluded,
        validation_summary=report.validation_summary,
    )


def process_file_to_handoff(
    path,
    *,
    document_id: str = "local-document",
    filename: str | None = None,
    detected_type: str | None = None,
    include_uncertain: bool = True,
    load_reference_index: bool = True,
) -> RelevantChunksResponse:
    """Run parse -> chunk -> classify -> handoff in memory (no Document/Chunk rows are written).

    Used by the golden tests and the accuracy script; the API path stores chunks first and then
    calls :func:`build_handoff` with the same inputs.
    """
    from pathlib import Path

    from app.chunking import chunk_document
    from app.classifier import aggregate_document_decision, classify_chunks
    from app.extraction.registry import parse_document

    p = Path(path)
    if load_reference_index:
        from app.db import session_scope
        from app.reference import ensure_reference_dataset_loaded

        with session_scope() as session:
            ensure_reference_dataset_loaded(session)
    extracted = parse_document(p, detected_type or p.suffix.lower().lstrip("."))
    drafts = chunk_document(document_id, extracted)
    batch = classify_chunks(drafts)
    summary = aggregate_document_decision(drafts, batch.items, extracted.quality, extracted.warnings)
    views = [
        ChunkView(
            id=d.id,
            text=d.text,
            location=d.location,
            chunk_type=d.chunk_type,
            context=d.context,
            extraction_method=d.extraction_method,
            ocr_confidence=d.ocr_confidence,
            decision=cls.decision,
            top_category=cls.top_category,
            category_scores=cls.category_scores,
            score=cls.score,
            semantic_score=cls.semantic_score,
            signal_score=cls.signal_score,
            matched_references=cls.matched_references,
            signals=cls.signals,
            classifier_version=cls.classifier_version,
            embedding_model=cls.embedding_model,
            dataset_version=cls.dataset_version,
        )
        for d, cls in zip(drafts, batch.items)
    ]
    return build_handoff(
        document_id=document_id,
        filename=filename or p.name,
        source_type=extracted.detected_type,
        relevance_status=summary.relevance_status,
        extraction_quality=extracted.quality,
        chunks=views,
        include_uncertain=include_uncertain,
    )
