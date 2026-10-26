"""Common, format-independent document representation and parser interface.

Every parser converts a file into an :class:`ExtractedDocument` made of
:class:`ContentBlock` objects. The chunker and classification engine only
depend on this representation, so new formats can be added by implementing
:class:`DocumentParser` and registering it in :mod:`app.extraction.registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# block types
HEADING = "heading"
PARAGRAPH = "paragraph"
TABLE = "table"
LINE = "line"  # OCR line / plain-text line


@dataclass
class TableData:
    headers: list[str]
    rows: list[list[str]]
    # 1-based source row numbers for each row (spreadsheet / CSV row or table row index)
    row_numbers: list[int] = field(default_factory=list)
    header_row_number: int | None = None
    col_offset: int = 1  # 1-based index of first column
    caption: str | None = None  # title rows / caption above the table
    footnotes: list[str] = field(default_factory=list)
    formulas: dict[str, str] = field(default_factory=dict)  # cell ref -> formula text (never evaluated)
    number_formats: dict[str, str] = field(default_factory=dict)  # cell ref -> number format
    row_bboxes: list[list[float]] | None = None


@dataclass
class ContentBlock:
    block_type: str
    text: str
    location: dict[str, Any]
    extraction_method: str
    ocr_confidence: float | None = None
    heading_level: int | None = None
    table: TableData | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedDocument:
    detected_type: str
    blocks: list[ContentBlock] = field(default_factory=list)
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


class ExtractionError(Exception):
    """Raised when a file cannot be parsed (malformed, encrypted, limits exceeded)."""


class DocumentParser(ABC):
    """Interface implemented by every format parser."""

    #: detected types handled by this parser (e.g. ``{"pdf"}``)
    types: frozenset[str] = frozenset()

    @abstractmethod
    def parse(self, path: Path) -> ExtractedDocument:  # pragma: no cover - interface
        ...
