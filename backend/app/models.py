"""SQLAlchemy ORM models.

Core fields are typed columns; JSON is used only for flexible metadata such as
source locations, signal evidence and matched references. Embedding vectors
are *not* stored here - they live in the vector index / embedding cache.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return uuid.uuid4().hex


class ProcessingStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    CLASSIFYING = "CLASSIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class DocumentRelevance(str, enum.Enum):
    FINANCIAL_RELEVANT = "FINANCIAL_RELEVANT"
    MIXED_CONTENT = "MIXED_CONTENT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NOT_FINANCIAL = "NOT_FINANCIAL"
    PROCESSING_FAILED = "PROCESSING_FAILED"


class ChunkDecision(str, enum.Enum):
    RELEVANT = "RELEVANT"
    UNCERTAIN = "UNCERTAIN"
    IRRELEVANT = "IRRELEVANT"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobType(str, enum.Enum):
    PROCESS = "process"  # extract + chunk + classify
    CLASSIFY = "classify"  # reclassify existing chunks (optionally re-extract)
    REBUILD_INDEX = "rebuild_index"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    extension: Mapped[str] = mapped_column(String(16))
    detected_type: Mapped[str] = mapped_column(String(16))
    mime_type: Mapped[str] = mapped_column(String(128))
    declared_mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_path: Mapped[str] = mapped_column(String(1024))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    status: Mapped[str] = mapped_column(String(24), default=ProcessingStatus.UPLOADED.value, index=True)
    relevance_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_filtered: Mapped[bool] = mapped_column(Boolean, default=False)

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    current_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    classifier_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_calls: Mapped[int] = mapped_column(Integer, default=0)
    embedding_cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentChunk.chunk_index"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # stable content/location id
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[dict] = mapped_column(JSON)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sheet_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_method: Mapped[str] = mapped_column(String(32))
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    extra_locations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ClassificationResult(Base):
    __tablename__ = "classification_results"
    __table_args__ = (Index("ix_cls_doc_current", "document_id", "is_current"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("document_chunks.id", ondelete="CASCADE"), index=True)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float)
    signal_score: Mapped[float] = mapped_column(Float)
    semantic_score: Mapped[float] = mapped_column(Float)
    positive_similarity: Mapped[float] = mapped_column(Float)
    negative_similarity: Mapped[float] = mapped_column(Float)
    top_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    matched_references: Mapped[list | None] = mapped_column(JSON, nullable=True)
    signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decided_by: Mapped[str] = mapped_column(String(24))  # baseline | trained | llm | rule
    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    classifier_version: Mapped[str] = mapped_column(String(128))
    embedding_model: Mapped[str] = mapped_column(String(256))
    dataset_version: Mapped[str] = mapped_column(String(128))
    thresholds: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReferenceDatasetVersion(Base):
    __tablename__ = "reference_dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(512))
    license: Mapped[str] = mapped_column(String(256))
    attribution: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReferenceEntry(Base):
    __tablename__ = "reference_entries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # reference_id
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("reference_dataset_versions.id"), index=True)
    concept: Mapped[str] = mapped_column(Text)
    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    examples: Mapped[list] = mapped_column(JSON, default=list)
    label: Mapped[str] = mapped_column(String(16), index=True)  # positive | negative
    source: Mapped[str] = mapped_column(String(512))
    license: Mapped[str] = mapped_column(String(256))
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vector_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)  # ids inside the vector index
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ReviewFeedback(Base):
    __tablename__ = "review_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    original_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    corrected_decision: Mapped[str] = mapped_column(String(16))
    corrected_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    text_snapshot: Mapped[str] = mapped_column(Text)
    location_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (Index("ix_jobs_status_next", "status", "next_attempt_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True)
    job_type: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # embedding | classifier | reference_index | evaluation
    name: Mapped[str] = mapped_column(String(256))
    version: Mapped[str] = mapped_column(String(128))
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
