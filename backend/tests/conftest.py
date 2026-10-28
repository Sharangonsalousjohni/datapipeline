from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app.db import Base, get_engine, session_scope
from app.ingestion import validate_upload_bytes
from app.models import Document


def ensure_fixture_document(doc_id: str, pdf_path: Path, filename: str) -> str:
    """Register a fixture PDF as a Document so pipeline tests do not depend on local DB state.

    Returns the id to use: ``doc_id`` if newly registered, or the id of an existing row
    that already holds the same file (content_hash is unique).
    """
    data = pdf_path.read_bytes()
    val = validate_upload_bytes(filename, data, "application/pdf")
    Base.metadata.create_all(get_engine())
    with session_scope() as session:
        if session.get(Document, doc_id):
            return doc_id
        existing = session.scalars(select(Document).where(Document.content_hash == val.content_hash)).first()
        if existing:
            return existing.id
        session.add(
            Document(
                id=doc_id,
                filename=val.filename,
                extension=val.extension,
                detected_type=val.detected_type,
                mime_type=val.mime_type,
                declared_mime_type=val.declared_mime_type,
                size_bytes=val.size_bytes,
                content_hash=val.content_hash,
                storage_path=str(pdf_path),
            )
        )
    return doc_id
