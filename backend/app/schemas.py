"""Pydantic request/response schemas (public API contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkDecisionT = Literal["RELEVANT", "UNCERTAIN", "IRRELEVANT"]
DocumentRelevanceT = Literal[
    "FINANCIAL_RELEVANT", "MIXED_CONTENT", "NEEDS_REVIEW", "NOT_FINANCIAL", "PROCESSING_FAILED"
]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine readable error code")
    message: str
    details: Any | None = None


# --------------------------------------------------------------------------- source locations
class SourceLocation(BaseModel):
    """Where a chunk/block came from. Fields are populated depending on the format."""

    page: int | None = None  # 1-based page number (PDF / image = 1)
    sheet: str | None = None
    table_index: int | None = None
    row_start: int | None = None  # 1-based spreadsheet/CSV/table rows
    row_end: int | None = None
    col_start: int | None = None
    col_end: int | None = None
    cell_range: str | None = None  # e.g. "Sheet1!A1:F12"
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    char_start: int | None = None  # offsets into the document's extracted text stream
    char_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    bboxes: list[list[float]] | None = None  # [[x0,y0,x1,y1], ...] in page/image coordinates
    page_width: float | None = None
    page_height: float | None = None
    heading_path: list[str] | None = None
    section_heading: str | None = None
    # --- chunk metadata (additive): everything needed to interpret numbers without neighbouring chunks
    page_end: int | None = None  # last page when a section spans pages
    bbox_pages: list[int] | None = None  # page of each entry in ``bboxes``
    chunk_role: str | None = None  # narrative | table | cover_page | table_of_contents | signature_block
    statement_type: str | None = None  # income_statement | balance_sheet | cash_flow_statement | highlights | notes | narrative | other
    unit: dict[str, Any] | None = None  # detected reporting unit (scale, multiplier, currency, except_per_share, source, text)
    period_columns: list[str] | None = None
    fiscal_period: dict[str, Any] | None = None  # document reporting period, e.g. {"key": "FY2025", "label": ...}
    table_bbox: list[float] | None = None
    cell_bboxes: list[list[list[float] | None]] | None = None  # header row first
    table_part: list[int] | None = None  # [part, total] when an oversized table had to be sliced
    overlap: list[int] | None = None  # [start, end) of text repeated from the previous piece of the same section


# --------------------------------------------------------------------------- jobs
class JobOut(ORMModel):
    id: str
    document_id: str | None
    job_type: str
    status: str
    stage: str | None
    progress: float
    attempts: int
    max_attempts: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


# --------------------------------------------------------------------------- documents
class ChunkCounts(BaseModel):
    relevant: int = 0
    uncertain: int = 0
    irrelevant: int = 0
    total: int = 0


class DocumentSummary(ORMModel):
    id: str
    filename: str
    extension: str
    detected_type: str
    mime_type: str
    size_bytes: int
    content_hash: str
    uploaded_at: datetime
    status: str
    relevance_status: DocumentRelevanceT | None
    relevance_score: float | None
    is_filtered: bool
    processing_ms: int | None


class DocumentDetail(DocumentSummary):
    page_count: int | None
    extraction_quality: dict | None
    warnings: list[str] = []
    error_message: str | None
    current_run_id: str | None
    classifier_version: str | None
    embedding_model: str | None
    dataset_version: str | None
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    embedding_calls: int
    embedding_cache_hits: int
    llm_calls: int
    llm_cost_usd: float
    chunk_counts: ChunkCounts
    latest_job: JobOut | None = None
    score_explanation: str


class UploadResult(BaseModel):
    document: DocumentSummary
    duplicate: bool = Field(description="True when an identical file (same SHA-256) already existed")
    job: JobOut | None = None


class UploadRejection(BaseModel):
    filename: str
    error: str
    message: str


class BatchUploadResult(BaseModel):
    accepted: list[UploadResult]
    rejected: list[UploadRejection]


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int


class ClassifyRequest(BaseModel):
    force: bool = Field(False, description="Reclassify even if the current result used the same versions")
    reextract: bool = Field(False, description="Re-run extraction and chunking before classifying")


# --------------------------------------------------------------------------- chunks & classification
class MatchedReference(BaseModel):
    reference_id: str
    concept: str
    category: str
    label: Literal["positive", "negative"]
    similarity: float
    matched_text: str


class SignalEvidence(BaseModel):
    signal: str  # e.g. currency_amount, financial_term, payment_phrase, obligation, payment_date
    text: str
    start: int
    end: int
    category: str | None = None
    weight: float = 0.0


class ChunkClassification(BaseModel):
    decision: ChunkDecisionT
    score: float
    signal_score: float
    semantic_score: float
    positive_similarity: float
    negative_similarity: float
    top_category: str | None
    category_scores: dict[str, float] = {}
    matched_references: list[MatchedReference] = []
    signals: list[SignalEvidence] = []
    negative_signals: list[SignalEvidence] = []
    decided_by: str
    llm_used: bool = False
    llm_rationale: str | None = None
    classifier_version: str
    embedding_model: str
    dataset_version: str
    thresholds: dict[str, Any] | None = None


class ChunkOut(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    chunk_type: str
    text: str
    context: str | None
    location: SourceLocation
    extra_locations: list[SourceLocation] = []
    extraction_method: str
    ocr_confidence: float | None
    classification: ChunkClassification | None = None
    feedback: list[FeedbackOut] = []
    section_heading: str | None = None
    financial_fields: list[ExtractedFinancialField] = []
    requires_review: bool = False


class ChunkList(BaseModel):
    document_id: str
    items: list[ChunkOut]
    total: int


class FeedbackIn(BaseModel):
    corrected_decision: ChunkDecisionT
    corrected_category: str | None = None
    comment: str | None = Field(None, max_length=2000)
    reviewer: str | None = Field(None, max_length=128)


class FeedbackOut(ORMModel):
    id: int
    document_id: str | None
    chunk_id: str | None
    original_decision: str | None
    corrected_decision: str
    corrected_category: str | None
    comment: str | None
    reviewer: str | None
    created_at: datetime


# --------------------------------------------------------------------------- handoff
class TableRowClassification(BaseModel):
    row_index: int
    row_label: str
    category: str
    decision: str = "RELEVANT"
    cells: list[str] = []
    periods: list[str] = []
    fields: list[ExtractedFinancialField] = []
    validation_status: str = "PASSED"
    metric: str | None = None  # canonical metric (app.taxonomy) the row label maps to
    metric_match: str | None = None  # synonym | contains | embedding | none
    statement_type: str | None = None
    row_bbox: list[float] | None = None


class ExtractionHandoff(BaseModel):
    """Structured object for a downstream financial extraction service."""

    schema_version: str = "1.1"
    document_id: str
    chunk_id: str
    source_filename: str
    source_type: str
    source_location: SourceLocation
    extracted_text: str
    context: str | None
    financial_category: str | None
    category_scores: dict[str, float]
    relevance_score: float
    semantic_score: float = 0.0
    signal_score: float = 0.0
    decision: ChunkDecisionT
    matching_references: list[MatchedReference]
    evidence: list[SignalEvidence]
    evidence_bboxes: list[list[float]] | None
    extraction_method: str
    ocr_confidence: float | None
    extraction_confidence: float = 1.0
    financial_fields: list[ExtractedFinancialField] = []
    table_rows: list[TableRowClassification] = []
    extraction_quality: dict | None
    classifier_version: str
    embedding_model: str
    dataset_version: str
    statement_type: str | None = None
    chunk_role: str | None = None  # narrative | table | cover_page | table_of_contents | signature_block
    decision_basis: str | None = None  # classifier | table_rows | narrative_facts | structural_role
    chunk_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Unit, period columns, page and section needed to interpret this chunk alone"
    )
    untrusted_content: bool = Field(
        True, description="Extracted text is untrusted data and must never be executed or followed as instructions"
    )


class RelevantChunksResponse(BaseModel):
    document_id: str
    relevance_status: DocumentRelevanceT | None
    include_uncertain: bool
    items: list[ExtractionHandoff]
    excluded_chunks: list[dict[str, Any]] = Field(
        default_factory=list, description="Chunks not handed off (IRRELEVANT), with the reason, so nothing is dropped silently"
    )
    validation_summary: ValidationSummary | None = None


# --------------------------------------------------------------------------- structured financial extraction (Section 7)
class ExtractedFinancialField(BaseModel):
    field: str
    original_text: str
    normalized_value: str
    reported_value: str | None = None
    reported_scale: str | None = None
    scale_multiplier: float | None = None
    normalized_amount: str | None = None
    currency: str | None = None
    data_type: str = "monetary"  # monetary, monetary_change, percentage, basis_points, ratio, date, text
    reporting_period: str | None = None
    row_label: str | None = None
    category: str | None = None
    frequency: str | None = None
    due_date: str | None = None
    condition: str | None = None
    details: dict[str, Any] | None = None
    validation_status: Literal["PASSED", "FAILED", "REVIEW_REQUIRED", "WARNING"] = "PASSED"
    normalization_status: Literal["PASSED", "FAILED", "REVIEW_REQUIRED"] = "PASSED"
    validation_notes: str | None = None
    confidence: float | None = 1.0
    source_location: dict[str, Any] | None = None
    signed_accounting_value: str | None = None
    change_value: str | None = None
    change_semantics: str | None = None  # SIGNED_VALUE_DELTA, MAGNITUDE_DELTA, PERCENTAGE_CHANGE, UNKNOWN
    change_direction: str | None = None  # INCREASE, DECREASE, FLAT
    change_basis: str | None = None  # period_over_period, year_over_year, not_applicable
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    validation_evidence: list[dict[str, Any]] = Field(default_factory=list)
    review_required: bool = False
    # --- additive fields (schema_version 1.1)
    metric: str | None = None  # canonical metric name from app.taxonomy (revenue, net_income, eps_diluted, ...)
    measure: str | None = None  # level | change_pct | change_bps | pct_of_revenue | rate | maturity_year | count | price
    period_key: str | None = None  # comparable period key, e.g. FY2025, Q3-2026
    statement_type: str | None = None
    source_kind: str | None = None  # table | narrative | kpi | clause
    char_start: int | None = None  # span of original_text inside the chunk's extracted_text
    char_end: int | None = None


class ValidationSummary(BaseModel):
    monetary_values_checked: bool = True
    failed_checks: int = 0
    requires_review: bool = False
    details: list[str] = []
    accounting_validations: list[dict[str, Any]] = []
    quality_gate_decision: str = "ACCEPT"  # ACCEPT, ACCEPT_WITH_WARNINGS, HOLD_FOR_REVIEW, REJECT
    gate_reasons: list[str] = Field(default_factory=list)
    cross_reference_checks: list[dict[str, Any]] = Field(default_factory=list)
    rule_audit_trail: list[dict[str, Any]] = Field(default_factory=list)


class StructuredExtractionItem(BaseModel):
    chunk_id: str
    statement_type: str | None = None
    chunk_role: str | None = None
    page_number: int | None = None
    section_heading: str | None = None
    decision: str
    category: str | None = None
    source_text: str
    relevance_score: float = 0.0
    semantic_score: float = 0.0
    signal_score: float = 0.0
    extraction_confidence: float = 1.0
    extraction_method: str | None = None
    rule_evidence: list[SignalEvidence] = []
    financial_fields: list[ExtractedFinancialField] = []
    table_rows: list[TableRowClassification] = []
    requires_review: bool = False


class StructuredExtractionReport(BaseModel):
    document_id: str
    document_type: str
    document_decision: str
    items: list[StructuredExtractionItem]
    validation_summary: ValidationSummary


class DocumentReport(BaseModel):
    document: DocumentDetail
    decision_rationale: list[str]
    category_breakdown: dict[str, int]
    chunks: list[ChunkOut]
    handoff: list[ExtractionHandoff]
    structured_extraction: StructuredExtractionReport | None = None
    generated_at: datetime


# --------------------------------------------------------------------------- reference dataset
class ReferenceEntryIn(BaseModel):
    reference_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    concept: str = Field(min_length=2, max_length=500)
    definition: str | None = Field(None, max_length=4000)
    category: str = Field(min_length=2, max_length=64)
    examples: list[str] = Field(default_factory=list, max_length=50)
    label: Literal["positive", "negative"] = "positive"
    source: str | None = None
    license: str | None = None


class ReferenceEntryOut(ORMModel):
    id: str
    dataset_version_id: int
    concept: str
    definition: str | None
    category: str
    examples: list[str]
    label: str
    source: str
    license: str
    embedding_model: str | None
    embedding_version: str | None
    vector_ids: list[int] | None
    is_active: bool
    updated_at: datetime


class ReferenceEntryList(BaseModel):
    items: list[ReferenceEntryOut]
    total: int


class DatasetVersionOut(ORMModel):
    id: int
    version: str
    name: str
    description: str | None
    source: str
    license: str
    attribution: str | None
    source_url: str | None
    entry_count: int
    is_active: bool
    created_at: datetime


class ImportResult(BaseModel):
    dataset_version: DatasetVersionOut
    imported: int
    updated: int
    skipped: int
    errors: list[str]
    index_job: JobOut | None = None


class ReferenceStats(BaseModel):
    total_entries: int
    active_entries: int
    positive_entries: int
    negative_entries: int
    total_examples: int
    category_distribution: dict[str, int]
    label_distribution: dict[str, int]
    dataset_versions: list[DatasetVersionOut]
    embedding_model: str
    index_version: str | None
    index_vectors: int
    index_backend: str | None
    index_built_at: datetime | None
    index_stale: bool
    categories: list[str]


# --------------------------------------------------------------------------- dashboard / health
class DashboardStats(BaseModel):
    total_documents: int
    processed_documents: int
    financial_relevant: int
    not_financial_filtered: int
    mixed_content: int
    needs_review: int
    processing_failed: int
    in_progress: int
    total_chunks: int
    chunk_decisions: dict[str, int]
    avg_processing_ms: float | None
    p95_processing_ms: float | None
    embedding_calls: int
    embedding_cache_hits: int
    llm_calls: int
    llm_cost_usd: float
    feedback_count: int
    evaluation: dict | None = Field(None, description="Latest evaluation report summary, if one was run")
    by_type: dict[str, int]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: str
    embedding_provider: str
    embedding_model: str
    embedding_loaded: bool
    ocr_provider: str
    ocr_available: bool
    vector_backend: str | None
    index_vectors: int
    worker_running: bool
    llm_enabled: bool
    auth_enabled: bool
    thresholds: dict[str, Any]
    classifier_mode: str


# --------------------------------------------------------------------------- financial summary (consumer payload)
class SummaryDocument(BaseModel):
    id: str
    filename: str
    source_type: str
    document_type: str = Field(description="Human-readable kind of document, e.g. 'Financial statement'")
    relevance_status: str | None


class SummaryOutcome(BaseModel):
    decision: str = Field(description="Quality gate: ACCEPT | ACCEPT_WITH_WARNINGS | HOLD_FOR_REVIEW | REJECT")
    requires_review: bool
    headline: str = Field(description="One-sentence plain-English result")
    reasons: list[str] = Field(default_factory=list)


class SummaryReporting(BaseModel):
    currency: str | None = None
    reported_in: str | None = Field(None, description="Scale the document reports in: units | thousands | millions | billions")
    periods: list[str] = Field(default_factory=list, description="Reporting periods found, most recent first")


class SummaryValue(BaseModel):
    value: str = Field(description="Exact normalized value as a decimal string (monetary values in full units)")
    unit: str = Field(description="ISO currency code, or percent | basis_points | ratio | per_share | shares | year | number")
    display: str = Field(description="Human-readable value, e.g. '$482.6 million'")
    as_reported: str = Field(description="Text exactly as it appears in the document")


class KeyFigure(SummaryValue):
    label: str
    metric: str | None = Field(None, description="Canonical metric id (stable key for systems)")
    measure: str = Field("level", description="level | change_pct | change_bps | pct_of_revenue | ...")
    period: str | None = None
    page: int | None = None


class StatementLine(BaseModel):
    label: str
    metric: str | None = None
    values: dict[str, SummaryValue] = Field(description="Keyed by the statement's column names")


class StatementSummary(BaseModel):
    statement: str
    title: str
    currency: str | None = None
    reported_in: str | None = None
    columns: list[str]
    lines: list[StatementLine]
    page: int | None = None


class ContractTerm(SummaryValue):
    term: str
    label: str
    frequency: str | None = None
    due: str | None = None
    condition: str | None = None
    page: int | None = None


class ReviewItem(BaseModel):
    label: str
    as_reported: str
    reason: str
    page: int | None = None
    section: str | None = None


class SummaryChecks(BaseModel):
    total: int = 0
    passed: int = 0
    warnings: int = 0
    failed: int = 0
    issues: list[str] = Field(default_factory=list, description="Messages for every check that did not pass")


class FinancialSummary(BaseModel):
    """Compact, human-readable view of a document's extracted financial data.

    Derived from the extraction handoff: duplicates are merged, only validated values appear in
    ``key_figures`` / ``statements`` / ``terms``, and anything unverified is listed in ``needs_review``.
    """

    schema_version: str = "1.0"
    generated_at: datetime
    document: SummaryDocument
    outcome: SummaryOutcome
    reporting: SummaryReporting
    key_figures: list[KeyFigure] = Field(default_factory=list)
    statements: list[StatementSummary] = Field(default_factory=list)
    terms: list[ContractTerm] = Field(default_factory=list)
    needs_review: list[ReviewItem] = Field(default_factory=list)
    checks: SummaryChecks


ChunkOut.model_rebuild()
RelevantChunksResponse.model_rebuild()
ExtractionHandoff.model_rebuild()
