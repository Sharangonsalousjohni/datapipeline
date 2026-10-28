"""FastAPI Application exposing all project endpoints."""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.classifier import TrainedClassifierBundle
from app.config import get_settings
from app.db import Base, get_db, get_engine, session_scope
from app.embeddings import get_embedding_service
from app.ingestion import (
    FileValidationError,
    store_upload_file,
    validate_upload_bytes,
)
from app.models import (
    ChunkDecision,
    ClassificationResult,
    Document,
    DocumentChunk,
    DocumentRelevance,
    JobStatus,
    JobType,
    ProcessingJob,
    ProcessingStatus,
    ReferenceDatasetVersion,
    ReferenceEntry,
    ReviewFeedback,
    new_uuid,
)
from app.ocr import get_ocr_provider, ocr_available
from app.taxonomy import canonicalize_category, canonicalize_scores
from app.reference import (
    CATEGORIES,
    ensure_reference_dataset_loaded,
    get_reference_index,
    import_reference_bytes,
)
from app.financial_extraction import build_structured_extraction
from app.handoff import ChunkView, build_handoff
from app.summary import build_financial_summary
from app.schemas import (
    BatchUploadResult,
    ChunkCounts,
    ChunkList,
    ChunkOut,
    ClassifyRequest,
    DashboardStats,
    DatasetVersionOut,
    DocumentDetail,
    DocumentList,
    DocumentReport,
    DocumentSummary,
    ErrorResponse,
    ExtractedFinancialField,
    ExtractionHandoff,
    FeedbackIn,
    FeedbackOut,
    FinancialSummary,
    HealthResponse,
    ImportResult,
    JobOut,
    StructuredExtractionReport,
    ValidationSummary,
    ReferenceEntryIn,
    ReferenceEntryList,
    ReferenceEntryOut,
    ReferenceStats,
    RelevantChunksResponse,
    SourceLocation,
    TableRowClassification,
    UploadRejection,
    UploadResult,
)
from app.worker import process_document_pipeline, start_background_worker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("frie")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    if settings.auto_migrate:
        Base.metadata.create_all(bind=get_engine())

    with session_scope() as session:
        ensure_reference_dataset_loaded(session)

    if settings.worker_enabled:
        start_background_worker()

    yield


app = FastAPI(
    title="Financial Relevance Intelligence Engine",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileValidationError)
async def file_validation_handler(request: Request, exc: FileValidationError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message, "details": None},
    )


def _get_chunk_counts(doc_id: str, db: Session) -> ChunkCounts:
    counts = dict(
        db.query(ClassificationResult.decision, func.count(ClassificationResult.id))
        .filter(ClassificationResult.document_id == doc_id, ClassificationResult.is_current.is_(True))
        .group_by(ClassificationResult.decision)
        .all()
    )
    rel = counts.get(ChunkDecision.RELEVANT.value, 0)
    unc = counts.get(ChunkDecision.UNCERTAIN.value, 0)
    irr = counts.get(ChunkDecision.IRRELEVANT.value, 0)
    return ChunkCounts(relevant=rel, uncertain=unc, irrelevant=irr, total=rel + unc + irr)


@app.get("/api/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    settings = get_settings()
    emb_svc = get_embedding_service()
    ref_idx = get_reference_index()
    t_rel, t_irr, t_src = settings.thresholds()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        database="sqlite" if "sqlite" in settings.resolved_database_url else "postgresql",
        embedding_provider=settings.embedding_provider,
        embedding_model=emb_svc.active_model_id(),
        embedding_loaded=emb_svc.is_loaded,
        ocr_provider=getattr(get_ocr_provider(), "name", "none"),
        ocr_available=ocr_available(),
        vector_backend=ref_idx.backend,
        index_vectors=ref_idx.vector_count,
        worker_running=settings.worker_enabled,
        llm_enabled=settings.llm_enabled,
        auth_enabled=settings.auth_enabled,
        thresholds={"relevant": t_rel, "irrelevant": t_irr, "source": t_src},
        classifier_mode=settings.classifier_mode,
    )


@app.get("/api/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_docs = db.query(Document).filter(Document.status != ProcessingStatus.DELETED.value).count()
    processed_docs = db.query(Document).filter(Document.status == ProcessingStatus.COMPLETED.value).count()
    fin_rel = db.query(Document).filter(Document.relevance_status == DocumentRelevance.FINANCIAL_RELEVANT.value).count()
    filtered = db.query(Document).filter(Document.relevance_status == DocumentRelevance.NOT_FINANCIAL.value).count()
    mixed = db.query(Document).filter(Document.relevance_status == DocumentRelevance.MIXED_CONTENT.value).count()
    review = db.query(Document).filter(Document.relevance_status == DocumentRelevance.NEEDS_REVIEW.value).count()
    failed = db.query(Document).filter(Document.status == ProcessingStatus.FAILED.value).count()
    in_prog = db.query(Document).filter(
        Document.status.in_([
            ProcessingStatus.QUEUED.value,
            ProcessingStatus.EXTRACTING.value,
            ProcessingStatus.CHUNKING.value,
            ProcessingStatus.CLASSIFYING.value,
        ])
    ).count()

    decisions = dict(
        db.query(ClassificationResult.decision, func.count(ClassificationResult.id))
        .filter(ClassificationResult.is_current.is_(True))
        .group_by(ClassificationResult.decision)
        .all()
    )

    types_count = dict(
        db.query(Document.detected_type, func.count(Document.id))
        .filter(Document.status != ProcessingStatus.DELETED.value)
        .group_by(Document.detected_type)
        .all()
    )

    perf = db.query(
        func.avg(Document.processing_ms),
        func.sum(Document.embedding_calls),
        func.sum(Document.embedding_cache_hits),
        func.sum(Document.llm_calls),
        func.sum(Document.llm_cost_usd),
    ).filter(Document.status == ProcessingStatus.COMPLETED.value).first()

    avg_ms = float(perf[0]) if perf and perf[0] else None
    emb_calls = int(perf[1] or 0)
    emb_hits = int(perf[2] or 0)
    llm_calls = int(perf[3] or 0)
    llm_cost = float(perf[4] or 0.0)

    fb_count = db.query(ReviewFeedback).count()

    eval_data = None
    res_path = settings.calibration_path.parent / "latest_report.json"
    if res_path.exists():
        try:
            eval_data = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return DashboardStats(
        total_documents=total_docs,
        processed_documents=processed_docs,
        financial_relevant=fin_rel,
        not_financial_filtered=filtered,
        mixed_content=mixed,
        needs_review=review,
        processing_failed=failed,
        in_progress=in_prog,
        total_chunks=sum(decisions.values()),
        chunk_decisions=decisions,
        avg_processing_ms=avg_ms,
        p95_processing_ms=None,
        embedding_calls=emb_calls,
        embedding_cache_hits=emb_hits,
        llm_calls=llm_calls,
        llm_cost_usd=llm_cost,
        feedback_count=fb_count,
        evaluation=eval_data,
        by_type=types_count,
    )


@app.post("/api/documents/upload", response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    response: Response = None,
):
    content = await file.read()
    val = validate_upload_bytes(file.filename, content, file.content_type)

    existing = db.scalars(select(Document).where(Document.content_hash == val.content_hash)).first()
    if existing:
        if response:
            response.status_code = status.HTTP_200_OK
        job = db.scalars(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == existing.id)
            .order_by(ProcessingJob.created_at.desc())
        ).first()
        return UploadResult(
            document=DocumentSummary.model_validate(existing),
            duplicate=True,
            job=JobOut.model_validate(job) if job else None,
        )

    saved_path = store_upload_file(val)
    doc_id = new_uuid()
    doc = Document(
        id=doc_id,
        filename=val.filename,
        extension=val.extension,
        detected_type=val.detected_type,
        mime_type=val.mime_type,
        declared_mime_type=val.declared_mime_type,
        size_bytes=val.size_bytes,
        content_hash=val.content_hash,
        storage_path=str(saved_path),
        status=ProcessingStatus.QUEUED.value,
    )
    db.add(doc)

    job_id = new_uuid()
    job = ProcessingJob(
        id=job_id,
        document_id=doc_id,
        job_type=JobType.PROCESS.value,
        status=JobStatus.QUEUED.value,
    )
    db.add(job)
    db.commit()
    db.refresh(doc)
    db.refresh(job)

    return UploadResult(
        document=DocumentSummary.model_validate(doc),
        duplicate=False,
        job=JobOut.model_validate(job),
    )


@app.post("/api/documents/batch-upload", response_model=BatchUploadResult)
async def batch_upload_documents(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    accepted: list[UploadResult] = []
    rejected: list[UploadRejection] = []

    for file in files:
        try:
            content = await file.read()
            val = validate_upload_bytes(file.filename, content, file.content_type)
            existing = db.scalars(select(Document).where(Document.content_hash == val.content_hash)).first()
            if existing:
                job = db.scalars(
                    select(ProcessingJob)
                    .where(ProcessingJob.document_id == existing.id)
                    .order_by(ProcessingJob.created_at.desc())
                ).first()
                accepted.append(
                    UploadResult(
                        document=DocumentSummary.model_validate(existing),
                        duplicate=True,
                        job=JobOut.model_validate(job) if job else None,
                    )
                )
                continue

            saved_path = store_upload_file(val)
            doc_id = new_uuid()
            doc = Document(
                id=doc_id,
                filename=val.filename,
                extension=val.extension,
                detected_type=val.detected_type,
                mime_type=val.mime_type,
                declared_mime_type=val.declared_mime_type,
                size_bytes=val.size_bytes,
                content_hash=val.content_hash,
                storage_path=str(saved_path),
                status=ProcessingStatus.QUEUED.value,
            )
            db.add(doc)
            job_id = new_uuid()
            job = ProcessingJob(
                id=job_id,
                document_id=doc_id,
                job_type=JobType.PROCESS.value,
                status=JobStatus.QUEUED.value,
            )
            db.add(job)
            db.commit()
            db.refresh(doc)
            db.refresh(job)
            accepted.append(
                UploadResult(
                    document=DocumentSummary.model_validate(doc),
                    duplicate=False,
                    job=JobOut.model_validate(job),
                )
            )
        except FileValidationError as fe:
            rejected.append(UploadRejection(filename=file.filename or "unknown", error=fe.code, message=fe.message))
        except Exception as e:
            rejected.append(UploadRejection(filename=file.filename or "unknown", error="internal_error", message=str(e)))

    return BatchUploadResult(accepted=accepted, rejected=rejected)


@app.get("/api/documents", response_model=DocumentList)
def list_documents(
    relevance: str | None = Query(None),
    status: str | None = Query(None),
    q: str | None = Query(None),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    stmt = select(Document).where(Document.status != ProcessingStatus.DELETED.value)
    if relevance:
        stmt = stmt.where(Document.relevance_status == relevance)
    if status:
        stmt = stmt.where(Document.status == status)
    if q:
        stmt = stmt.where(Document.filename.ilike(f"%{q}%"))

    total = db.query(func.count(Document.id)).filter(
        Document.status != ProcessingStatus.DELETED.value,
        Document.relevance_status == relevance if relevance else True,
        Document.status == status if status else True,
        Document.filename.ilike(f"%{q}%") if q else True,
    ).scalar() or 0

    docs = db.scalars(stmt.order_by(Document.uploaded_at.desc()).limit(limit).offset(offset)).all()
    return DocumentList(items=[DocumentSummary.model_validate(d) for d in docs], total=total)


@app.get("/api/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc or doc.status == ProcessingStatus.DELETED.value:
        raise HTTPException(status_code=404, detail="Document not found")

    job = db.scalars(
        select(ProcessingJob)
        .where(ProcessingJob.document_id == doc.id)
        .order_by(ProcessingJob.created_at.desc())
    ).first()

    meta = doc.metadata_json or {}
    counts = _get_chunk_counts(doc.id, db)
    return DocumentDetail(
        **DocumentSummary.model_validate(doc).model_dump(),
        page_count=doc.page_count,
        extraction_quality=doc.extraction_quality,
        warnings=doc.warnings or [],
        error_message=doc.error_message,
        current_run_id=doc.current_run_id,
        classifier_version=doc.classifier_version,
        embedding_model=doc.embedding_model,
        dataset_version=doc.dataset_version,
        processing_started_at=doc.processing_started_at,
        processing_completed_at=doc.processing_completed_at,
        embedding_calls=doc.embedding_calls,
        embedding_cache_hits=doc.embedding_cache_hits,
        llm_calls=doc.llm_calls,
        llm_cost_usd=doc.llm_cost_usd,
        chunk_counts=counts,
        latest_job=JobOut.model_validate(job) if job else None,
        score_explanation=meta.get("score_explanation", ""),
    )


@app.post("/api/documents/{document_id}/classify", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def trigger_classify(
    document_id: str,
    req: ClassifyRequest,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, document_id)
    if not doc or doc.status == ProcessingStatus.DELETED.value:
        raise HTTPException(status_code=404, detail="Document not found")

    job_id = new_uuid()
    job = ProcessingJob(
        id=job_id,
        document_id=doc.id,
        job_type=JobType.PROCESS.value,
        status=JobStatus.QUEUED.value,
        payload={"force": req.force, "reextract": req.reextract},
    )
    doc.status = ProcessingStatus.QUEUED.value
    db.add(job)
    db.commit()
    db.refresh(job)
    return JobOut.model_validate(job)


@app.get("/api/documents/{document_id}/chunks", response_model=ChunkList)
def get_document_chunks(
    document_id: str,
    decision: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    chunks = db.scalars(stmt.order_by(DocumentChunk.chunk_index.asc())).all()

    cls_map = {}
    if doc.current_run_id:
        for cr in db.scalars(
            select(ClassificationResult).where(
                ClassificationResult.document_id == document_id,
                ClassificationResult.is_current.is_(True),
            )
        ).all():
            cls_map[cr.chunk_id] = cr

    items: list[ChunkOut] = []
    for c in chunks:
        cls_obj = cls_map.get(c.id)
        if decision and cls_obj and cls_obj.decision != decision:
            continue
        fb_list = db.scalars(
            select(ReviewFeedback).where(ReviewFeedback.chunk_id == c.id).order_by(ReviewFeedback.created_at.desc())
        ).all()

        cls_pydantic = None
        if cls_obj:
            cls_pydantic = {
                "decision": cls_obj.decision,
                "score": cls_obj.score,
                "signal_score": cls_obj.signal_score,
                "semantic_score": cls_obj.semantic_score,
                "positive_similarity": cls_obj.positive_similarity,
                "negative_similarity": cls_obj.negative_similarity,
                "top_category": canonicalize_category(cls_obj.top_category) if cls_obj.top_category else None,
                "category_scores": canonicalize_scores(cls_obj.category_scores),
                "matched_references": cls_obj.matched_references or [],
                "signals": cls_obj.signals or [],
                "negative_signals": [],
                "decided_by": cls_obj.decided_by,
                "llm_used": cls_obj.llm_used,
                "llm_rationale": cls_obj.llm_rationale,
                "classifier_version": cls_obj.classifier_version,
                "embedding_model": cls_obj.embedding_model,
                "dataset_version": cls_obj.dataset_version,
                "thresholds": cls_obj.thresholds,
            }

        items.append(
            ChunkOut(
                id=c.id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                chunk_type=c.chunk_type,
                text=c.text,
                context=c.context,
                location=SourceLocation(**c.location),
                extra_locations=[SourceLocation(**el) for el in (c.extra_locations or [])],
                extraction_method=c.extraction_method,
                ocr_confidence=c.ocr_confidence,
                classification=cls_pydantic,
                feedback=[FeedbackOut.model_validate(fb) for fb in fb_list],
            )
        )

    total = len(items)
    paged = items[offset : offset + limit]
    return ChunkList(document_id=document_id, items=paged, total=total)


@app.get("/api/documents/{document_id}/relevant-chunks", response_model=RelevantChunksResponse)
def get_relevant_chunks(
    document_id: str,
    include_uncertain: bool = True,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    res = (
        db.query(DocumentChunk, ClassificationResult)
        .join(ClassificationResult, DocumentChunk.id == ClassificationResult.chunk_id)
        .filter(
            DocumentChunk.document_id == document_id,
            ClassificationResult.is_current.is_(True),
        )
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    # Every chunk is analysed (cross-table and narrative checks need the whole document); the handoff
    # keeps RELEVANT/UNCERTAIN plus structural non-financial chunks and lists the rest as excluded.
    views = [
        ChunkView(
            id=c.id,
            text=c.text,
            location=dict(c.location or {}),
            chunk_type=c.chunk_type,
            context=c.context,
            extraction_method=c.extraction_method,
            ocr_confidence=c.ocr_confidence,
            decision=cr.decision,
            top_category=cr.top_category,
            category_scores=cr.category_scores or {},
            score=cr.score,
            semantic_score=cr.semantic_score if cr.semantic_score is not None else 0.0,
            signal_score=cr.signal_score if cr.signal_score is not None else 0.0,
            matched_references=cr.matched_references or [],
            signals=cr.signals or [],
            classifier_version=cr.classifier_version,
            embedding_model=cr.embedding_model,
            dataset_version=cr.dataset_version,
        )
        for c, cr in res
    ]
    return build_handoff(
        document_id=doc.id,
        filename=doc.filename,
        source_type=doc.detected_type,
        relevance_status=doc.relevance_status,
        extraction_quality=doc.extraction_quality,
        chunks=views,
        include_uncertain=include_uncertain,
    )


@app.get("/api/documents/{document_id}/report", response_model=DocumentReport)
def get_document_report(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_detail = get_document(document_id, db)
    chunks_resp = get_document_chunks(document_id, limit=500, db=db)
    handoff_resp = get_relevant_chunks(document_id, include_uncertain=True, db=db)

    cat_breakdown: dict[str, int] = {}
    for h in handoff_resp.items:
        if h.financial_category:
            cat_breakdown[h.financial_category] = cat_breakdown.get(h.financial_category, 0) + 1

    chunk_dicts = [
        {
            "chunk_id": c.id,
            "page_number": c.location.page,
            "section_heading": c.location.section_heading,
            "decision": c.classification.decision if c.classification else "IRRELEVANT",
            "top_category": c.classification.top_category if c.classification else None,
            "score": c.classification.score if c.classification else 0.0,
            "semantic_score": c.classification.semantic_score if c.classification else 0.0,
            "signal_score": c.classification.signal_score if c.classification else 0.0,
            "ocr_confidence": c.ocr_confidence if c.ocr_confidence is not None else 1.0,
            "signals": c.classification.signals if c.classification else [],
            "text": c.text,
            "chunk_type": c.chunk_type,
            "extraction_method": c.extraction_method,
            "category_scores": c.classification.category_scores if c.classification else {},
            "location": c.location.model_dump(),
            "bboxes": c.location.bboxes,
        }
        for c in chunks_resp.items
    ]
    structured_report = build_structured_extraction(
        doc.id, doc.filename, doc.relevance_status or "NOT_FINANCIAL", chunk_dicts
    )

    meta = doc.metadata_json or {}
    return DocumentReport(
        document=doc_detail,
        decision_rationale=meta.get("rationale", []),
        category_breakdown=cat_breakdown,
        chunks=chunks_resp.items,
        handoff=handoff_resp.items,
        structured_extraction=structured_report,
        generated_at=datetime.now(timezone.utc),
    )


@app.get("/api/documents/{document_id}/structured-extraction", response_model=StructuredExtractionReport)
def get_document_structured_extraction(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_resp = get_document_chunks(document_id, limit=500, db=db)
    chunk_dicts = [
        {
            "chunk_id": c.id,
            "page_number": c.location.page,
            "section_heading": c.location.section_heading,
            "decision": c.classification.decision if c.classification else "IRRELEVANT",
            "top_category": c.classification.top_category if c.classification else None,
            "score": c.classification.score if c.classification else 0.0,
            "semantic_score": c.classification.semantic_score if c.classification else 0.0,
            "signal_score": c.classification.signal_score if c.classification else 0.0,
            "ocr_confidence": c.ocr_confidence if c.ocr_confidence is not None else 1.0,
            "signals": c.classification.signals if c.classification else [],
            "text": c.text,
            "chunk_type": c.chunk_type,
            "extraction_method": c.extraction_method,
            "category_scores": c.classification.category_scores if c.classification else {},
            "location": c.location.model_dump(),
            "bboxes": c.location.bboxes,
        }
        for c in chunks_resp.items
    ]
    return build_structured_extraction(
        doc.id, doc.filename, doc.relevance_status or "NOT_FINANCIAL", chunk_dicts
    )


@app.get("/api/documents/{document_id}/summary", response_model=FinancialSummary)
def get_document_summary(document_id: str, db: Session = Depends(get_db)):
    """Compact, human-readable financial summary derived from the extraction handoff."""
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != ProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=409,
            detail=f"Document is {doc.status.lower()}; a summary is available once processing completes",
        )
    handoff = get_relevant_chunks(document_id, include_uncertain=True, db=db)
    return build_financial_summary(handoff, filename=doc.filename, source_type=doc.detected_type)


@app.get("/api/documents/{document_id}/quality-gate", response_model=ValidationSummary)
def get_document_quality_gate(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    report = get_document_structured_extraction(document_id, db=db)
    return report.validation_summary


@app.get("/api/documents/{document_id}/file")
def download_original_file(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc or not Path(doc.storage_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=doc.storage_path,
        filename=doc.filename,
        media_type=doc.mime_type,
    )


@app.post("/api/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.status = JobStatus.QUEUED.value
    job.error = None
    job.attempts = 0
    db.commit()
    db.refresh(job)
    return JobOut.model_validate(job)


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut.model_validate(job)


@app.post("/api/chunks/{chunk_id}/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    chunk_id: str,
    fb: FeedbackIn,
    db: Session = Depends(get_db),
):
    chunk = db.get(DocumentChunk, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    latest_cls = db.scalars(
        select(ClassificationResult)
        .where(ClassificationResult.chunk_id == chunk_id, ClassificationResult.is_current.is_(True))
    ).first()

    orig_dec = latest_cls.decision if latest_cls else None
    feedback_row = ReviewFeedback(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        original_decision=orig_dec,
        corrected_decision=fb.corrected_decision,
        corrected_category=canonicalize_category(fb.corrected_category) if fb.corrected_category else None,
        comment=fb.comment,
        reviewer=fb.reviewer,
        text_snapshot=chunk.text,
        location_snapshot=chunk.location,
    )
    db.add(feedback_row)
    db.commit()
    db.refresh(feedback_row)
    return FeedbackOut.model_validate(feedback_row)


@app.get("/api/feedback", response_model=list[FeedbackOut])
def list_feedback(limit: int = 100, db: Session = Depends(get_db)):
    feedbacks = db.scalars(select(ReviewFeedback).order_by(ReviewFeedback.created_at.desc()).limit(limit)).all()
    return [FeedbackOut.model_validate(f) for f in feedbacks]


@app.get("/api/feedback/export")
def export_feedback_jsonl(db: Session = Depends(get_db)):
    feedbacks = db.scalars(select(ReviewFeedback).order_by(ReviewFeedback.created_at.asc())).all()
    lines = []
    for f in feedbacks:
        lines.append(
            json.dumps({
                "id": f.id,
                "document_id": f.document_id,
                "chunk_id": f.chunk_id,
                "original_decision": f.original_decision,
                "corrected_decision": f.corrected_decision,
                "corrected_category": f.corrected_category,
                "comment": f.comment,
                "reviewer": f.reviewer,
                "text": f.text_snapshot,
                "created_at": f.created_at.isoformat(),
            })
        )
    content = "\n".join(lines)
    return Response(content=content, media_type="application/x-jsonlines", headers={"Content-Disposition": "attachment; filename=feedback.jsonl"})


@app.get("/api/reference-dataset/stats", response_model=ReferenceStats)
def get_reference_stats(db: Session = Depends(get_db)):
    ref_idx = get_reference_index()
    emb_svc = get_embedding_service()

    total = db.query(ReferenceEntry).count()
    active = db.query(ReferenceEntry).filter(ReferenceEntry.is_active.is_(True)).count()
    pos = db.query(ReferenceEntry).filter(ReferenceEntry.is_active.is_(True), ReferenceEntry.label == "positive").count()
    neg = db.query(ReferenceEntry).filter(ReferenceEntry.is_active.is_(True), ReferenceEntry.label == "negative").count()

    cat_dist: dict[str, int] = {}
    for cat, n in (
        db.query(ReferenceEntry.category, func.count(ReferenceEntry.id))
        .filter(ReferenceEntry.is_active.is_(True))
        .group_by(ReferenceEntry.category)
        .all()
    ):
        ccat = canonicalize_category(cat)
        cat_dist[ccat] = cat_dist.get(ccat, 0) + n

    lbl_dist = dict(
        db.query(ReferenceEntry.label, func.count(ReferenceEntry.id))
        .filter(ReferenceEntry.is_active.is_(True))
        .group_by(ReferenceEntry.label)
        .all()
    )

    versions = db.scalars(select(ReferenceDatasetVersion).order_by(ReferenceDatasetVersion.created_at.desc())).all()

    # Sum examples
    entries = db.scalars(select(ReferenceEntry).where(ReferenceEntry.is_active.is_(True))).all()
    total_examples = sum(len(e.examples or []) for e in entries)

    return ReferenceStats(
        total_entries=total,
        active_entries=active,
        positive_entries=pos,
        negative_entries=neg,
        total_examples=total_examples,
        category_distribution=cat_dist,
        label_distribution=lbl_dist,
        dataset_versions=[DatasetVersionOut.model_validate(v) for v in versions],
        embedding_model=emb_svc.active_model_id(),
        index_version=ref_idx.version,
        index_vectors=ref_idx.vector_count,
        index_backend=ref_idx.backend,
        index_built_at=ref_idx.built_at,
        index_stale=ref_idx.is_stale,
        categories=CATEGORIES,
    )


@app.get("/api/reference-dataset/entries", response_model=ReferenceEntryList)
def list_reference_entries(
    category: str | None = Query(None),
    label: str | None = Query(None),
    q: str | None = Query(None),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    stmt = select(ReferenceEntry).where(ReferenceEntry.is_active.is_(True))
    if category:
        stmt = stmt.where(ReferenceEntry.category == category)
    if label:
        stmt = stmt.where(ReferenceEntry.label == label)
    if q:
        stmt = stmt.where(ReferenceEntry.concept.ilike(f"%{q}%"))

    total = db.query(func.count(ReferenceEntry.id)).filter(
        ReferenceEntry.is_active.is_(True),
        ReferenceEntry.category == category if category else True,
        ReferenceEntry.label == label if label else True,
        ReferenceEntry.concept.ilike(f"%{q}%") if q else True,
    ).scalar() or 0

    entries = db.scalars(stmt.order_by(ReferenceEntry.id.asc()).limit(limit).offset(offset)).all()
    return ReferenceEntryList(items=[ReferenceEntryOut.model_validate(e) for e in entries], total=total)


@app.post("/api/reference-dataset/entries", response_model=ReferenceEntryOut)
def create_reference_entry(
    entry_in: ReferenceEntryIn,
    db: Session = Depends(get_db),
):
    active_dv = db.scalars(select(ReferenceDatasetVersion).where(ReferenceDatasetVersion.is_active.is_(True)).order_by(ReferenceDatasetVersion.id.desc())).first()
    if not active_dv:
        active_dv = ReferenceDatasetVersion(
            version="1.0.0",
            name="Custom Entries",
            source="Manual addition",
            license="CC0-1.0",
        )
        db.add(active_dv)
        db.flush()

    ref_id = entry_in.reference_id or f"{entry_in.category[:4]}-{new_uuid()[:8]}"
    existing = db.get(ReferenceEntry, ref_id)
    if existing:
        raise HTTPException(status_code=400, detail="Reference ID already exists")

    entry = ReferenceEntry(
        id=ref_id,
        dataset_version_id=active_dv.id,
        concept=entry_in.concept,
        definition=entry_in.definition,
        category=entry_in.category,
        examples=entry_in.examples,
        label=entry_in.label,
        source=entry_in.source or active_dv.source,
        license=entry_in.license or active_dv.license,
        content_hash=new_uuid(),
        is_active=True,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    get_reference_index().mark_stale()
    return ReferenceEntryOut.model_validate(entry)


@app.delete("/api/reference-dataset/entries/{entry_id}")
def delete_reference_entry(entry_id: str, db: Session = Depends(get_db)):
    entry = db.get(ReferenceEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    entry.is_active = False
    db.commit()
    get_reference_index().mark_stale()
    return {"status": "deactivated"}


@app.post("/api/reference-dataset/import", response_model=ImportResult)
async def import_reference_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    version: str = Form(...),
    source: str = Form(...),
    license: str = Form(...),
    attribution: str | None = Form(None),
    source_url: str | None = Form(None),
    rebuild_index: str = Form("true"),
    db: Session = Depends(get_db),
):
    content = await file.read()
    rebuild_bool = rebuild_index.lower() in ("true", "1", "yes")

    dv, imported, updated, skipped, errors = import_reference_bytes(
        db,
        content,
        filename=file.filename or "dataset.jsonl",
        name=name,
        version=version,
        source=source,
        license_name=license,
        attribution=attribution,
        source_url=source_url,
        rebuild=rebuild_bool,
    )
    db.commit()

    return ImportResult(
        dataset_version=DatasetVersionOut.model_validate(dv),
        imported=imported,
        updated=updated,
        skipped=skipped,
        errors=errors,
        index_job=None,
    )


@app.post("/api/reference-dataset/rebuild-embeddings", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def rebuild_embeddings_job(db: Session = Depends(get_db)):
    job_id = new_uuid()
    job = ProcessingJob(
        id=job_id,
        job_type=JobType.REBUILD_INDEX.value,
        status=JobStatus.QUEUED.value,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return JobOut.model_validate(job)


@app.get("/api/reference-dataset/versions", response_model=list[DatasetVersionOut])
def list_dataset_versions(db: Session = Depends(get_db)):
    dvs = db.scalars(select(ReferenceDatasetVersion).order_by(ReferenceDatasetVersion.created_at.desc())).all()
    return [DatasetVersionOut.model_validate(dv) for dv in dvs]


@app.get("/api/evaluation/latest")
def get_latest_evaluation_report():
    report_file = settings.calibration_path.parent / "latest_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="No evaluation report has been generated yet.")
    return json.loads(report_file.read_text(encoding="utf-8"))
