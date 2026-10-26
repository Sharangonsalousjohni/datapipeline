"""Stage A: Ingestion & File Security Validation.

Validates file extensions, MIME types, magic signatures, zip-bomb ratios,
and file sizes. Rejects macros/scripts (.docm, .xlsm) and binary spoofing.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".txt": "txt",
    ".csv": "csv",
}

CANONICAL_MIMES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "png": "image/png",
    "jpg": "image/jpeg",
    "txt": "text/plain",
    "csv": "text/csv",
}

ALLOWED_DECLARED_MIMES: dict[str, set[str]] = {
    "pdf": {"application/pdf", "application/x-pdf", "application/octet-stream", ""},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
        "",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
        "application/octet-stream",
        "",
    },
    "xls": {"application/vnd.ms-excel", "application/x-ole-storage", "application/octet-stream", ""},
    "png": {"image/png", "application/octet-stream", ""},
    "jpg": {"image/jpeg", "image/jpg", "application/octet-stream", ""},
    "txt": {"text/plain", "application/octet-stream", ""},
    "csv": {"text/csv", "text/plain", "application/csv", "application/vnd.ms-excel", "application/octet-stream", ""},
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\- ()]+")


class FileValidationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class ValidatedUpload:
    filename: str
    extension: str
    detected_type: str
    mime_type: str
    declared_mime_type: str | None
    size_bytes: int
    content_hash: str
    data: bytes


def sanitize_filename(raw_name: str | None) -> str:
    if not raw_name:
        raise FileValidationError("invalid_filename", "Filename is missing.")
    # Prevent path traversal by taking only the final path component on both POSIX and Windows
    cleaned = raw_name.replace("\\", "/").split("/")[-1].strip()
    cleaned = _SAFE_NAME_RE.sub("_", cleaned).lstrip(".")
    if not cleaned or cleaned in {".", ".."}:
        raise FileValidationError("invalid_filename", "Filename is invalid or unsafe.")
    return cleaned[:240]


def _inspect_zip(data: bytes, expected_type: str, max_uncompressed_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            total_uncompressed = 0
            for info in zf.infolist():
                if ".." in info.filename or info.filename.startswith(("/", "\\")):
                    raise FileValidationError("unsafe_archive", "Archive contains unsafe member paths.")
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise FileValidationError(
                        "zip_bomb",
                        f"Archive uncompressed size exceeds limit ({max_uncompressed_bytes // (1024 * 1024)} MB).",
                        status_code=413,
                    )
            if expected_type == "docx" and "word/document.xml" not in names:
                raise FileValidationError(
                    "signature_mismatch", "File does not contain a valid Word (.docx) structure.", status_code=415
                )
            if expected_type == "xlsx" and "xl/workbook.xml" not in names:
                raise FileValidationError(
                    "signature_mismatch", "File does not contain a valid Excel (.xlsx) structure.", status_code=415
                )
    except FileValidationError:
        raise
    except Exception as exc:
        raise FileValidationError(
            "malformed_container", f"Corrupted or invalid Office container: {type(exc).__name__}", status_code=415
        ) from exc


def _looks_like_text(data: bytes) -> bool:
    sample = data[:8192]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in sample:
        return False
    return True


def validate_upload_bytes(
    filename: str | None,
    data: bytes,
    declared_mime: str | None = None,
) -> ValidatedUpload:
    settings = get_settings()
    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()

    if ext in {".docm", ".xlsm", ".pptm", ".vbs", ".js", ".exe", ".bat", ".ps1", ".sh"}:
        raise FileValidationError(
            "unsafe_file_type",
            f"Macro-enabled or executable file extension '{ext}' is rejected for security.",
            status_code=415,
        )
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            "unsupported_extension",
            f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            status_code=415,
        )

    size_bytes = len(data)
    if size_bytes == 0:
        raise FileValidationError("empty_file", "Uploaded file is empty.", status_code=400)
    if size_bytes > settings.max_file_size_bytes:
        raise FileValidationError(
            "file_too_large",
            f"File size ({size_bytes} bytes) exceeds limit ({settings.max_file_size_mb} MB).",
            status_code=413,
        )

    expected_type = ALLOWED_EXTENSIONS[ext]
    norm_mime = (declared_mime or "").split(";")[0].strip().lower()
    if norm_mime and norm_mime not in ALLOWED_DECLARED_MIMES[expected_type]:
        raise FileValidationError(
            "mime_mismatch",
            f"Declared MIME type '{norm_mime}' does not match extension '{ext}'.",
            status_code=415,
        )

    # Magic signature verification
    if expected_type == "pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise FileValidationError("signature_mismatch", "File lacks %PDF- magic header.", status_code=415)
    elif expected_type in ("docx", "xlsx"):
        if not data.startswith(b"PK\x03\x04"):
            raise FileValidationError(
                "signature_mismatch", f"File lacks PK zip header required for .{expected_type}.", status_code=415
            )
        _inspect_zip(data, expected_type, settings.max_zip_uncompressed_mb * 1024 * 1024)
    elif expected_type == "xls":
        if not data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            raise FileValidationError(
                "signature_mismatch", "File lacks OLE2 Compound Document signature for .xls.", status_code=415
            )
    elif expected_type == "png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise FileValidationError("signature_mismatch", "File lacks PNG magic signature.", status_code=415)
    elif expected_type == "jpg":
        if not data.startswith(b"\xff\xd8\xff"):
            raise FileValidationError("signature_mismatch", "File lacks JPEG magic signature.", status_code=415)
    elif expected_type in ("txt", "csv"):
        if not _looks_like_text(data):
            raise FileValidationError(
                "binary_in_text", "File appears to contain binary data rather than text.", status_code=415
            )

    content_hash = hashlib.sha256(data).hexdigest()
    return ValidatedUpload(
        filename=safe_name,
        extension=ext,
        detected_type=expected_type,
        mime_type=CANONICAL_MIMES[expected_type],
        declared_mime_type=norm_mime or None,
        size_bytes=size_bytes,
        content_hash=content_hash,
        data=data,
    )


def store_upload_file(upload: ValidatedUpload) -> Path:
    settings = get_settings()
    settings.ensure_dirs()
    dest = settings.upload_dir / f"{upload.content_hash}{upload.extension}"
    if not dest.exists():
        dest.write_bytes(upload.data)
    return dest
