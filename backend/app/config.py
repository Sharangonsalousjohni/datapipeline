"""Central application configuration.

All tunables (model names, limits, thresholds, providers) live here and can be
overridden with environment variables prefixed ``FRIE_`` (or a ``.env`` file).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRIE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ general
    app_name: str = "Financial Relevance Intelligence Engine"
    environment: Literal["development", "test", "production"] = "development"
    data_dir: Path = BACKEND_DIR / "data"
    database_url: str = ""  # defaults to sqlite file in data_dir
    auto_migrate: bool = True
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"
    log_document_content: bool = False  # never log extracted text unless explicitly enabled

    # ------------------------------------------------------------------ auth
    # When disabled (default for local dev) every caller is treated as admin.
    auth_enabled: bool = False
    admin_api_key: str = ""
    reviewer_api_key: str = ""

    # ------------------------------------------------------------------ upload limits
    max_file_size_mb: int = 50
    max_pdf_pages: int = 500
    max_image_pixels: int = 60_000_000
    max_spreadsheet_cells: int = 2_000_000
    max_zip_uncompressed_mb: int = 300
    max_batch_files: int = 20

    # ------------------------------------------------------------------ extraction / OCR
    ocr_provider: Literal["auto", "rapidocr", "paddleocr", "tesseract", "none"] = "auto"
    ocr_languages: str = "eng"  # tesseract language string
    ocr_dpi: int = 200
    ocr_min_page_chars: int = 40  # pages with fewer native chars are OCR'd
    ocr_preprocess: bool = True
    ocr_low_confidence: float = 0.60  # below this mean confidence => quality warning
    pdf_detect_tables: bool = True

    # ------------------------------------------------------------------ chunking
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 40
    table_rows_per_chunk: int = 25  # legacy; tables are no longer sliced at this size
    chunk_max_tokens: int = 400  # narrative sections above this are split at paragraph boundaries
    table_max_rows_per_chunk: int = 200  # tables are kept whole unless they exceed this many rows

    # ------------------------------------------------------------------ embeddings
    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_max_chars: int = 2000
    hf_offline: bool = False

    # ------------------------------------------------------------------ reference dataset
    reference_dataset_path: Path = BACKEND_DIR / "reference_data" / "financial_reference_v1.jsonl"
    reference_manifest_path: Path = BACKEND_DIR / "reference_data" / "manifest.json"
    vector_backend: Literal["auto", "faiss", "numpy"] = "auto"

    # ------------------------------------------------------------------ classification
    semantic_top_k: int = 24
    # Default thresholds; overridden by calibration file produced by the evaluation pipeline.
    threshold_relevant: float = 0.60
    threshold_irrelevant: float = 0.35
    calibration_path: Path = BACKEND_DIR / "evaluation" / "results" / "calibration.json"
    classifier_mode: Literal["baseline", "trained", "auto", "lexical"] = "auto"
    trained_classifier_path: Path = BACKEND_DIR / "evaluation" / "results" / "trained_classifier.joblib"
    lexical_classifier_path: Path = BACKEND_DIR / "evaluation" / "results" / "lexical_classifier.joblib"

    # ------------------------------------------------------------------ document aggregation
    doc_relevant_fraction_for_financial: float = 0.5
    doc_min_text_chars: int = 30

    # ------------------------------------------------------------------ optional LLM (disabled by default)
    llm_enabled: bool = False
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = Field(default="", repr=False)
    llm_model: str = "gpt-4o-mini"
    llm_max_calls_per_document: int = 5
    llm_max_chars_per_call: int = 2500
    llm_timeout_s: float = 20.0
    llm_cost_per_call_usd: float = 0.0003

    # ------------------------------------------------------------------ jobs
    worker_enabled: bool = True
    worker_poll_interval_s: float = 0.5
    job_max_attempts: int = 3

    # ------------------------------------------------------------------ retention
    retention_days: int | None = None  # None => keep forever

    # ------------------------------------------------------------------ derived paths
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def embedding_cache_path(self) -> Path:
        return self.data_dir / "embedding_cache.sqlite"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'frie.sqlite').as_posix()}"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ calibration
    def thresholds(self) -> tuple[float, float, str]:
        """Return (relevant, irrelevant, source) thresholds.

        Calibrated thresholds from the evaluation pipeline take precedence over defaults
        only when they were produced for the currently configured embedding model.
        """
        path = self.calibration_path
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("embedding_model") == self.embedding_model_id():
                    return (
                        float(data["threshold_relevant"]),
                        float(data["threshold_irrelevant"]),
                        f"calibrated:{data.get('calibrated_at', 'unknown')}",
                    )
            except (ValueError, KeyError, OSError):
                pass
        return self.threshold_relevant, self.threshold_irrelevant, "default"

    def embedding_model_id(self) -> str:
        if self.embedding_provider == "hashing":
            return "hashing-v1"
        return self.embedding_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
