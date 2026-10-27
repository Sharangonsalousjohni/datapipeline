/**
 * TypeScript mirror of backend/app/schemas.py (public API contract).
 * Datetimes are serialised as ISO-8601 strings.
 */

export type ChunkDecision = "RELEVANT" | "UNCERTAIN" | "IRRELEVANT";
export const CHUNK_DECISIONS: ChunkDecision[] = ["RELEVANT", "UNCERTAIN", "IRRELEVANT"];

export type DocumentRelevance =
  | "FINANCIAL_RELEVANT"
  | "MIXED_CONTENT"
  | "NEEDS_REVIEW"
  | "NOT_FINANCIAL"
  | "PROCESSING_FAILED";
export const DOCUMENT_RELEVANCE_VALUES: DocumentRelevance[] = [
  "FINANCIAL_RELEVANT",
  "MIXED_CONTENT",
  "NEEDS_REVIEW",
  "NOT_FINANCIAL",
  "PROCESSING_FAILED",
];

export type ISODateTime = string;

export interface ErrorResponse {
  error: string;
  message: string;
  details?: unknown;
}

// ---------------------------------------------------------------- source locations
export interface SourceLocation {
  page?: number | null;
  sheet?: string | null;
  table_index?: number | null;
  row_start?: number | null;
  row_end?: number | null;
  col_start?: number | null;
  col_end?: number | null;
  cell_range?: string | null;
  paragraph_start?: number | null;
  paragraph_end?: number | null;
  char_start?: number | null;
  char_end?: number | null;
  line_start?: number | null;
  line_end?: number | null;
  bboxes?: number[][] | null;
  page_width?: number | null;
  page_height?: number | null;
  heading_path?: string[] | null;
}

// ---------------------------------------------------------------- jobs
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | string;

export interface JobOut {
  id: string;
  document_id: string | null;
  job_type: string;
  status: JobStatus;
  stage: string | null;
  progress: number;
  attempts: number;
  max_attempts: number;
  error: string | null;
  created_at: ISODateTime;
  started_at: ISODateTime | null;
  finished_at: ISODateTime | null;
}

// ---------------------------------------------------------------- documents
export interface ChunkCounts {
  relevant: number;
  uncertain: number;
  irrelevant: number;
  total: number;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  extension: string;
  detected_type: string;
  mime_type: string;
  size_bytes: number;
  content_hash: string;
  uploaded_at: ISODateTime;
  status: string;
  relevance_status: DocumentRelevance | null;
  relevance_score: number | null;
  is_filtered: boolean;
  processing_ms: number | null;
}

export interface DocumentDetail extends DocumentSummary {
  page_count: number | null;
  extraction_quality: Record<string, unknown> | null;
  warnings: string[];
  error_message: string | null;
  current_run_id: string | null;
  classifier_version: string | null;
  embedding_model: string | null;
  dataset_version: string | null;
  processing_started_at: ISODateTime | null;
  processing_completed_at: ISODateTime | null;
  embedding_calls: number;
  embedding_cache_hits: number;
  llm_calls: number;
  llm_cost_usd: number;
  chunk_counts: ChunkCounts;
  latest_job: JobOut | null;
  score_explanation: string;
}

export interface UploadResult {
  document: DocumentSummary;
  duplicate: boolean;
  job: JobOut | null;
}

export interface UploadRejection {
  filename: string;
  error: string;
  message: string;
}

export interface BatchUploadResult {
  accepted: UploadResult[];
  rejected: UploadRejection[];
}

export interface DocumentList {
  items: DocumentSummary[];
  total: number;
}

export interface ClassifyRequest {
  force: boolean;
  reextract: boolean;
}

// ---------------------------------------------------------------- chunks & classification
export interface MatchedReference {
  reference_id: string;
  concept: string;
  category: string;
  label: "positive" | "negative";
  similarity: number;
  matched_text: string;
}

export interface SignalEvidence {
  signal: string;
  text: string;
  start: number;
  end: number;
  category?: string | null;
  weight?: number;
}

export interface ChunkClassification {
  decision: ChunkDecision;
  score: number;
  signal_score: number;
  semantic_score: number;
  positive_similarity: number;
  negative_similarity: number;
  top_category: string | null;
  category_scores: Record<string, number>;
  matched_references: MatchedReference[];
  signals: SignalEvidence[];
  negative_signals: SignalEvidence[];
  decided_by: string;
  llm_used: boolean;
  llm_rationale: string | null;
  classifier_version: string;
  embedding_model: string;
  dataset_version: string;
  thresholds: Record<string, unknown> | null;
}

export interface FeedbackIn {
  corrected_decision: ChunkDecision;
  corrected_category?: string | null;
  comment?: string | null;
  reviewer?: string | null;
}

export interface FeedbackOut {
  id: number;
  document_id: string | null;
  chunk_id: string | null;
  original_decision: string | null;
  corrected_decision: string;
  corrected_category: string | null;
  comment: string | null;
  reviewer: string | null;
  created_at: ISODateTime;
}

export interface ChunkOut {
  id: string;
  document_id: string;
  chunk_index: number;
  chunk_type: string;
  text: string;
  context: string | null;
  location: SourceLocation;
  extra_locations: SourceLocation[];
  extraction_method: string;
  ocr_confidence: number | null;
  classification: ChunkClassification | null;
  feedback: FeedbackOut[];
}

export interface ChunkList {
  document_id: string;
  items: ChunkOut[];
  total: number;
}

// ---------------------------------------------------------------- handoff
export interface ExtractionHandoff {
  schema_version: string;
  document_id: string;
  chunk_id: string;
  source_filename: string;
  source_type: string;
  source_location: SourceLocation;
  extracted_text: string;
  context: string | null;
  financial_category: string | null;
  category_scores: Record<string, number>;
  relevance_score: number;
  decision: ChunkDecision;
  matching_references: MatchedReference[];
  evidence: SignalEvidence[];
  evidence_bboxes: number[][] | null;
  extraction_method: string;
  ocr_confidence: number | null;
  extraction_quality: Record<string, unknown> | null;
  classifier_version: string;
  embedding_model: string;
  dataset_version: string;
  untrusted_content: boolean;
}

export interface RelevantChunksResponse {
  document_id: string;
  relevance_status: DocumentRelevance | null;
  include_uncertain: boolean;
  items: ExtractionHandoff[];
}

export interface DocumentReport {
  document: DocumentDetail;
  decision_rationale: string[];
  category_breakdown: Record<string, number>;
  chunks: ChunkOut[];
  handoff: ExtractionHandoff[];
  structured_extraction?: StructuredExtractionReport | null;
  generated_at: ISODateTime;
}

// ---------------------------------------------------------------- validation & structured extraction
export type QualityGateDecision = "ACCEPT" | "ACCEPT_WITH_WARNINGS" | "HOLD_FOR_REVIEW" | "REJECT";
export type ValidationStatus = "PASSED" | "WARNING" | "FAILED" | "NEEDS_REVIEW" | "NOT_APPLICABLE";

export interface AccountingValidationCheck {
  rule_id: string;
  rule_version?: string;
  statement: string;
  period: string;
  formula: string;
  expected: string;
  actual: string;
  difference: string;
  tolerance?: string;
  status: ValidationStatus;
  message: string;
  timestamp?: string;
}

export interface ValidationSummary {
  monetary_values_checked: boolean;
  failed_checks: number;
  requires_review: boolean;
  details: string[];
  accounting_validations: AccountingValidationCheck[];
  quality_gate_decision: QualityGateDecision;
  gate_reasons: string[];
  cross_reference_checks: Record<string, any>[];
  rule_audit_trail: Record<string, any>[];
}

export interface ExtractedFinancialField {
  field: string;
  original_text: string;
  normalized_value: string;
  reported_value?: string | null;
  reported_scale?: string | null;
  scale_multiplier?: number | null;
  normalized_amount?: string | null;
  signed_accounting_value?: string | null;
  change_value?: string | null;
  change_semantics?: string | null;
  change_direction?: string | null;
  change_basis?: string | null;
  currency?: string | null;
  data_type: string;
  reporting_period?: string | null;
  row_label?: string | null;
  category?: string | null;
  validation_status: ValidationStatus;
  normalization_status: string;
  validation_notes?: string | null;
  confidence?: number | null;
  review_required?: boolean;
}

export interface TableRowClassification {
  row_index: number;
  row_label: string;
  category: string;
  decision: string;
  cells: string[];
  periods: string[];
  fields: ExtractedFinancialField[];
  validation_status: string;
}

export interface StructuredExtractionItem {
  chunk_id: string;
  page_number?: number | null;
  section_heading?: string | null;
  decision: string;
  category?: string | null;
  source_text: string;
  relevance_score: number;
  semantic_score: number;
  signal_score: number;
  extraction_confidence: number;
  financial_fields: ExtractedFinancialField[];
  table_rows: TableRowClassification[];
  requires_review: boolean;
}

export interface StructuredExtractionReport {
  document_id: string;
  document_type: string;
  document_decision: string;
  items: StructuredExtractionItem[];
  validation_summary: ValidationSummary;
}

// ---------------------------------------------------------------- reference dataset
export type ReferenceLabel = "positive" | "negative";

export interface ReferenceEntryIn {
  reference_id?: string | null;
  concept: string;
  definition?: string | null;
  category: string;
  examples: string[];
  label: ReferenceLabel;
  source?: string | null;
  license?: string | null;
}

export interface ReferenceEntryOut {
  id: string;
  dataset_version_id: number;
  concept: string;
  definition: string | null;
  category: string;
  examples: string[];
  label: string;
  source: string;
  license: string;
  embedding_model: string | null;
  embedding_version: string | null;
  vector_ids: number[] | null;
  is_active: boolean;
  updated_at: ISODateTime;
}

export interface ReferenceEntryList {
  items: ReferenceEntryOut[];
  total: number;
}

export interface DatasetVersionOut {
  id: number;
  version: string;
  name: string;
  description: string | null;
  source: string;
  license: string;
  attribution: string | null;
  source_url: string | null;
  entry_count: number;
  is_active: boolean;
  created_at: ISODateTime;
}

export interface ImportResult {
  dataset_version: DatasetVersionOut;
  imported: number;
  updated: number;
  skipped: number;
  errors: string[];
  index_job: JobOut | null;
}

export interface ReferenceStats {
  total_entries: number;
  active_entries: number;
  positive_entries: number;
  negative_entries: number;
  total_examples: number;
  category_distribution: Record<string, number>;
  label_distribution: Record<string, number>;
  dataset_versions: DatasetVersionOut[];
  embedding_model: string;
  index_version: string | null;
  index_vectors: number;
  index_backend: string | null;
  index_built_at: ISODateTime | null;
  index_stale: boolean;
  categories: string[];
}

// ---------------------------------------------------------------- dashboard / health
export interface DashboardStats {
  total_documents: number;
  processed_documents: number;
  financial_relevant: number;
  not_financial_filtered: number;
  mixed_content: number;
  needs_review: number;
  processing_failed: number;
  in_progress: number;
  total_chunks: number;
  chunk_decisions: Record<string, number>;
  avg_processing_ms: number | null;
  p95_processing_ms: number | null;
  embedding_calls: number;
  embedding_cache_hits: number;
  llm_calls: number;
  llm_cost_usd: number;
  feedback_count: number;
  /** Latest evaluation report summary (shape is loose — render defensively). */
  evaluation: Record<string, unknown> | null;
  by_type: Record<string, number>;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  database: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_loaded: boolean;
  ocr_provider: string;
  ocr_available: boolean;
  vector_backend: string | null;
  index_vectors: number;
  worker_running: boolean;
  llm_enabled: boolean;
  auth_enabled: boolean;
  thresholds: Record<string, unknown>;
  classifier_mode: string;
}

/** Full evaluation report (GET /api/evaluation/latest). Loose by design — every key may be missing. */
export type EvaluationReport = Record<string, unknown>;
