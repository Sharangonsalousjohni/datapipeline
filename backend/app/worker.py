"""Background Job Worker for document processing and index rebuilds."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.chunking import chunk_document
from app.classifier import aggregate_document_decision, classify_chunks
from app.config import get_settings
from app.db import session_scope
from app.extraction.registry import parse_document
from app.models import (
    ClassificationResult,
    Document,
    DocumentChunk,
    JobStatus,
    JobType,
    ProcessingJob,
    ProcessingStatus,
)
from app.reference import ensure_reference_dataset_loaded, get_reference_index

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def process_document_pipeline(document_id: str, reextract: bool = False) -> dict:
    start_time = time.time()
    with session_scope() as session:
        ensure_reference_dataset_loaded(session)
        doc = session.get(Document, document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found")

        doc.status = ProcessingStatus.EXTRACTING.value
        doc.processing_started_at = utcnow()
        session.commit()

        path = Path(doc.storage_path)
        extracted = parse_document(path, doc.detected_type)

        doc.page_count = extracted.page_count
        doc.extraction_quality = extracted.quality
        doc.warnings = extracted.warnings
        doc.status = ProcessingStatus.CHUNKING.value
        session.commit()

        # Delete existing chunks if re-extracting
        if reextract or doc.chunks:
            session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            session.query(ClassificationResult).filter(ClassificationResult.document_id == doc.id).delete()
            session.flush()

        drafts = chunk_document(doc.id, extracted)
        for draft in drafts:
            chunk_row = DocumentChunk(
                id=draft.id,
                document_id=draft.document_id,
                chunk_index=draft.chunk_index,
                chunk_type=draft.chunk_type,
                text=draft.text,
                context=draft.context,
                location=draft.location,
                page_number=draft.location.get("page"),
                sheet_name=draft.location.get("sheet"),
                char_start=draft.location.get("char_start"),
                char_end=draft.location.get("char_end"),
                extraction_method=draft.extraction_method,
                ocr_confidence=draft.ocr_confidence,
                content_hash=draft.content_hash,
                extra_locations=draft.extra_locations,
            )
            session.add(chunk_row)
        session.commit()

        doc.status = ProcessingStatus.CLASSIFYING.value
        session.commit()

        cls_output = classify_chunks(drafts)

        run_id = f"run_{utcnow().strftime('%Y%m%d%H%M%S')}_{doc.id[:6]}"
        for draft, res in zip(drafts, cls_output.items):
            cls_row = ClassificationResult(
                run_id=run_id,
                document_id=doc.id,
                chunk_id=draft.id,
                decision=res.decision,
                score=res.score,
                signal_score=res.signal_score,
                semantic_score=res.semantic_score,
                positive_similarity=res.positive_similarity,
                negative_similarity=res.negative_similarity,
                top_category=res.top_category,
                category_scores=res.category_scores,
                matched_references=[m.model_dump() for m in res.matched_references],
                signals=[s.model_dump() for s in res.signals],
                decided_by=res.decided_by,
                llm_used=res.llm_used,
                llm_rationale=res.llm_rationale,
                classifier_version=res.classifier_version,
                embedding_model=res.embedding_model,
                dataset_version=res.dataset_version,
                thresholds=res.thresholds,
                is_current=True,
            )
            session.add(cls_row)

        doc_summary = aggregate_document_decision(
            drafts, cls_output.items, doc.extraction_quality, doc.warnings
        )

        doc.relevance_status = doc_summary.relevance_status
        doc.relevance_score = doc_summary.relevance_score
        doc.is_filtered = doc_summary.is_filtered
        doc.current_run_id = run_id
        doc.classifier_version = cls_output.classifier_version
        doc.embedding_model = cls_output.embedding_model
        doc.dataset_version = cls_output.dataset_version
        doc.status = ProcessingStatus.COMPLETED.value
        doc.processing_completed_at = utcnow()
        doc.processing_ms = int((time.time() - start_time) * 1000)
        doc.embedding_calls = cls_output.embedding_calls
        doc.embedding_cache_hits = cls_output.embedding_cache_hits
        doc.llm_calls = cls_output.llm_calls
        doc.llm_cost_usd = cls_output.llm_cost_usd

        meta = doc.metadata_json or {}
        meta["rationale"] = doc_summary.rationale
        meta["score_explanation"] = doc_summary.score_explanation
        doc.metadata_json = meta

        session.commit()
        return {
            "document_id": doc.id,
            "relevance_status": doc.relevance_status,
            "relevance_score": doc.relevance_score,
            "chunks_count": len(drafts),
        }


def _run_worker_loop():
    settings = get_settings()
    while True:
        try:
            with session_scope() as session:
                job = session.scalars(
                    select(ProcessingJob)
                    .where(ProcessingJob.status == JobStatus.QUEUED.value)
                    .order_by(ProcessingJob.created_at.asc())
                ).first()

                if job:
                    job.status = JobStatus.RUNNING.value
                    job.started_at = utcnow()
                    job.attempts += 1
                    session.commit()

                    try:
                        if job.job_type == JobType.PROCESS.value:
                            reextract = bool(job.payload.get("reextract", False)) if job.payload else False
                            res = process_document_pipeline(job.document_id, reextract=reextract)
                            job.result = res
                        elif job.job_type == JobType.REBUILD_INDEX.value:
                            res = get_reference_index().build_from_db(session)
                            job.result = res

                        job.status = JobStatus.SUCCEEDED.value
                        job.progress = 1.0
                        job.finished_at = utcnow()
                    except Exception as exc:
                        log.exception("job_failed")
                        job.error = str(exc)
                        if job.attempts < job.max_attempts:
                            job.status = JobStatus.QUEUED.value
                        else:
                            job.status = JobStatus.FAILED.value
                            if job.document_id:
                                doc = session.get(Document, job.document_id)
                                if doc:
                                    doc.status = ProcessingStatus.FAILED.value
                                    doc.error_message = str(exc)
                        job.finished_at = utcnow()
                    session.commit()
        except Exception:
            log.exception("worker_loop_exception")

        time.sleep(settings.worker_poll_interval_s)


def start_background_worker():
    t = threading.Thread(target=_run_worker_loop, daemon=True, name="JobWorkerThread")
    t.start()
