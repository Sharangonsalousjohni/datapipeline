"""Parser registry mapping detected document formats to their parser implementation."""

from __future__ import annotations

from pathlib import Path

from app.extraction.base import DocumentParser, ExtractedDocument, ExtractionError
from app.extraction.prose import DocxParser, TextParser
from app.extraction.spreadsheet import CsvParser, XlsParser, XlsxParser
from app.extraction.visual import ImageParser, PdfParser

_PARSERS: dict[str, DocumentParser] = {}


def register_parser(parser: DocumentParser) -> None:
    for t in parser.types:
        _PARSERS[t.lower()] = parser


def get_parser(detected_type: str) -> DocumentParser:
    key = detected_type.lower()
    if key == "jpeg":
        key = "jpg"
    parser = _PARSERS.get(key)
    if parser is None:
        raise ExtractionError(f"Unsupported document format: {detected_type}")
    return parser


def parse_document(path: Path, detected_type: str) -> ExtractedDocument:
    parser = get_parser(detected_type)
    return parser.parse(path)


for _cls in (PdfParser, ImageParser, DocxParser, TextParser, XlsxParser, XlsParser, CsvParser):
    register_parser(_cls())
