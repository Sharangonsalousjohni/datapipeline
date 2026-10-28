"""Visual format parsers: PDF (native + per-page OCR fallback) and images."""

from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.extraction.base import (
    HEADING,
    PARAGRAPH,
    TABLE,
    ContentBlock,
    DocumentParser,
    ExtractedDocument,
    ExtractionError,
    TableData,
)
from app.extraction.layout import (
    BOLD_FLAG,
    NUMBERED_HEADING_RE,
    clip_text,
    detect_aligned_tables,
    find_page_furniture,
    line_inside,
    page_lines,
    span_id,
    table_candidate_ok,
)
from app.extraction.tabular import render_table
from app.ocr import group_lines_into_paragraphs, run_ocr

_HEADING_RE = re.compile(
    r"^((article|section|schedule|exhibit|clause|part)\s+[\w.]+.{0,80}|\d+(\.\d+)*\.?\s+[A-Z][^.!?]{0,80}|[A-Z0-9][A-Z0-9 ,&/()'\-]{3,70})$"
)


def _rect_overlaps(a: list[float], b: list[float]) -> bool:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return (inter / area_a) > 0.4


def _fill_missing_cell_boxes(
    boxes: list[list[list[float] | None]], rows: list[list[str]], page
) -> list[list[list[float] | None]]:
    """Give every non-empty cell a bbox: use find_tables() cell rects, else the word boxes
    that fall inside the cell's row band and column band."""
    if all(b is not None for row_b, row_t in zip(boxes, rows) for b, t in zip(row_b, row_t) if t):
        return boxes
    n_cols = max((len(r) for r in boxes), default=0)
    col_x: list[tuple[float, float] | None] = []
    for c in range(n_cols):
        xs = [rb[c] for rb in boxes if c < len(rb) and rb[c] is not None]
        col_x.append((min(b[0] for b in xs), max(b[2] for b in xs)) if xs else None)
    words = page.get_text("words") or []
    out: list[list[list[float] | None]] = []
    for row_b, row_t in zip(boxes, rows):
        known = [b for b in row_b if b is not None]
        band = (min(b[1] for b in known), max(b[3] for b in known)) if known else None
        new_row: list[list[float] | None] = []
        for c, (b, txt) in enumerate(zip(row_b, row_t)):
            if b is not None or not txt or band is None or c >= len(col_x) or col_x[c] is None:
                new_row.append(b)
                continue
            x0, x1 = col_x[c]  # type: ignore[misc]
            hits = [
                w for w in words
                if x0 <= (w[0] + w[2]) / 2 <= x1 and band[0] <= (w[1] + w[3]) / 2 <= band[1]
            ]
            new_row.append(
                [round(min(w[0] for w in hits), 1), round(min(w[1] for w in hits), 1),
                 round(max(w[2] for w in hits), 1), round(max(w[3] for w in hits), 1)]
                if hits else None
            )
        out.append(new_row)
    return out


class PdfParser(DocumentParser):
    types = frozenset({"pdf"})

    def parse(self, path: Path) -> ExtractedDocument:
        import fitz  # PyMuPDF

        settings = get_settings()
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open PDF: {type(exc).__name__}") from exc

        try:
            if doc.is_encrypted:
                raise ExtractionError("PDF is password-protected / encrypted")
            if doc.page_count > settings.max_pdf_pages:
                raise ExtractionError(f"PDF exceeds maximum page count ({settings.max_pdf_pages})")

            out = ExtractedDocument(detected_type="pdf", page_count=doc.page_count)
            native_pages = 0
            ocr_pages = 0
            low_text_pages = 0
            ocr_confs: list[float] = []
            table_counter = 0

            # Page furniture (watermarks, running headers/footers) is found across all pages first,
            # so it never reaches headings, chunks or table cells.
            page_dicts: list[dict] = []
            for page_idx in range(doc.page_count):
                pg = doc.load_page(page_idx)
                d = pg.get_text("dict") or {}
                d["_height"] = float(pg.rect.height)
                page_dicts.append(d)
            furniture = find_page_furniture(page_dicts)
            body_sizes = [
                float(s.get("size", 10.0))
                for p_i, d in enumerate(page_dicts)
                for b in d.get("blocks", []) if b.get("type") == 0
                for ln in b.get("lines", [])
                for s in ln.get("spans", [])
                if (s.get("text") or "").strip() and span_id(s) not in furniture.span_ids[p_i]
            ]
            doc_body_size = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 10.0

            for page_idx in range(doc.page_count):
                page_num = page_idx + 1
                page = doc.load_page(page_idx)
                pw, ph = float(page.rect.width), float(page.rect.height)
                raw_text = (page.get_text("text") or "").strip()

                if len(raw_text) >= settings.ocr_min_page_chars:
                    native_pages += 1
                    table_bboxes: list[list[float]] = []
                    page_blocks: list[tuple[list[float], ContentBlock]] = []
                    drop = furniture.span_ids[page_idx]
                    page_dict = page_dicts[page_idx]
                    vlines = page_lines(page_dict, drop)
                    rejected_candidate = False

                    def add_table(tbl: TableData, t_bbox: list[float], boxes, method: str) -> None:
                        nonlocal table_counter
                        table_counter += 1
                        table_bboxes.append(t_bbox)
                        n_rows = len(tbl.rows) + 1
                        page_blocks.append(
                            (
                                t_bbox,
                                ContentBlock(
                                    block_type=TABLE,
                                    text=render_table(tbl),
                                    location={
                                        "page": page_num,
                                        "table_index": table_counter,
                                        "row_start": 1,
                                        "row_end": n_rows,
                                        "col_start": 1,
                                        "col_end": len(tbl.headers),
                                        "bboxes": [t_bbox],
                                        "table_bbox": t_bbox,
                                        "cell_bboxes": boxes,
                                        "page_width": pw,
                                        "page_height": ph,
                                    },
                                    extraction_method=method,
                                    table=tbl,
                                ),
                            )
                        )

                    if settings.pdf_detect_tables and hasattr(page, "find_tables"):
                        try:
                            tf = page.find_tables()
                            for t in getattr(tf, "tables", []) or []:
                                # PyMuPDF's t.extract() can strip punctuation, decimals, and spaces.
                                # Use cell-level clip extraction to preserve exact source formatting verbatim.
                                rows: list[list[str]] = []
                                boxes: list[list[list[float] | None]] = []
                                row_bboxes: list[list[float]] = []
                                t_rows = getattr(t, "rows", None)
                                if t_rows:
                                    for r in t_rows:
                                        row_text: list[str] = []
                                        row_boxes: list[list[float] | None] = []
                                        for cell in getattr(r, "cells", []):
                                            if cell is None:
                                                row_text.append("")
                                                row_boxes.append(None)
                                            else:
                                                row_text.append(
                                                    clip_text(page, fitz.Rect(cell), furniture.boxes[page_idx], doc_body_size)
                                                )
                                                row_boxes.append([round(float(x), 1) for x in cell])
                                        rows.append(row_text)
                                        boxes.append(row_boxes)
                                        row_bboxes.append([round(float(x), 1) for x in r.bbox])
                                else:
                                    extracted = t.extract() or []
                                    rows = [[(" ".join(str(c).split()) if c is not None else "") for c in r] for r in extracted]
                                    boxes = [[None] * len(r) for r in rows]

                                keep = [i for i, r in enumerate(rows) if any(c for c in r)]
                                rows = [rows[i] for i in keep]
                                row_bboxes = [row_bboxes[i] for i in keep if i < len(row_bboxes)]
                                boxes = _fill_missing_cell_boxes([boxes[i] for i in keep], rows, page)
                                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                                    t_bbox = [round(float(x), 1) for x in t.bbox]
                                    if not table_candidate_ok(rows, t_bbox, row_bboxes, vlines):
                                        rejected_candidate = True
                                        continue
                                    tbl = TableData(
                                        headers=rows[0],
                                        rows=rows[1:],
                                        row_numbers=list(range(2, len(rows) + 1)),
                                        header_row_number=1,
                                        cell_bboxes=boxes,
                                        bbox=t_bbox,
                                    )
                                    add_table(tbl, t_bbox, boxes, "pymupdf")
                        except Exception:
                            pass

                    if rejected_candidate:
                        # find_tables() merged prose/headings into a table or collapsed rows: rebuild the
                        # tables on this page from column-aligned text lines instead.
                        free = [ln for ln in vlines if not any(line_inside(ln, tb) for tb in table_bboxes)]
                        for at in detect_aligned_tables(free, pw):
                            tbl = TableData(
                                headers=at.headers,
                                rows=at.rows,
                                row_numbers=list(range(2, len(at.rows) + 2)),
                                header_row_number=1,
                                cell_bboxes=at.cell_bboxes,
                                bbox=at.bbox,
                            )
                            add_table(tbl, at.bbox, at.cell_bboxes, "pymupdf:aligned")

                    # Extract text blocks via get_text("dict") to inspect font sizes and coordinates
                    blocks = page_dict.get("blocks", [])
                    font_sizes: list[float] = []
                    for b in blocks:
                        if b.get("type") != 0:
                            continue
                        for line in b.get("lines", []):
                            for span in line.get("spans", []):
                                if (span.get("text") or "").strip() and span_id(span) not in drop:
                                    font_sizes.append(float(span.get("size", 11.0)))
                    median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 11.0

                    para_idx = 0
                    for b in blocks:
                        if b.get("type") != 0:
                            continue
                        bbox = [round(float(x), 1) for x in b.get("bbox", [0, 0, 0, 0])]
                        if any(_rect_overlaps(bbox, tb) for tb in table_bboxes):
                            continue
                        lines_text: list[str] = []
                        max_span_size = 0.0
                        all_bold = True
                        kept_boxes: list[list[float]] = []
                        for line in b.get("lines", []):
                            spans = [s for s in line.get("spans", []) if span_id(s) not in drop and (s.get("text") or "").strip()]
                            if not spans:
                                continue
                            lb = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
                                  max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
                            cx, cy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                            if any(tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3] for tb in table_bboxes):
                                continue  # table text belongs to the table block only
                            lt = "".join(s.get("text", "") for s in line.get("spans", []) if span_id(s) not in drop).strip()
                            if lt:
                                lines_text.append(lt)
                                kept_boxes.append(lb)
                                for s in spans:
                                    max_span_size = max(max_span_size, float(s.get("size", 0.0)))
                                    all_bold = all_bold and bool(s.get("flags", 0) & BOLD_FLAG)
                        text = "\n".join(lines_text).strip()
                        if not text:
                            continue
                        if kept_boxes and len(kept_boxes) < len(b.get("lines", [])):
                            bbox = [round(min(x[0] for x in kept_boxes), 1), round(min(x[1] for x in kept_boxes), 1),
                                    round(max(x[2] for x in kept_boxes), 1), round(max(x[3] for x in kept_boxes), 1)]

                        # Detect and filter repeated headers/footers (e.g. "Page 1 of 2", running page numbers)
                        y0, y1 = bbox[1], bbox[3]
                        if y0 > ph - 55 and (
                            re.search(r"\bpage\s+\d+\s*(?:of\s*\d+)?\b", text, re.I)
                            or re.search(r"page\s+\d+$", text, re.I)
                        ):
                            continue
                        if y1 < 45 and re.search(r"\bpage\s+\d+\s*(?:of\s*\d+)?\b", text, re.I):
                            continue

                        para_idx += 1
                        numbered = NUMBERED_HEADING_RE.match(text)
                        if numbered:
                            # numbered clause headings need a typographic cue; numbered list items do not qualify
                            is_heading = "\n" not in text and len(text) < 95 and (all_bold or max_span_size >= median_size * 1.1)
                        else:
                            is_heading = (
                                "\n" not in text
                                and len(text) < 95
                                and (max_span_size >= median_size * 1.18 or bool(_HEADING_RE.match(text)))
                            )
                        heading_level = (2 + text.split()[0].rstrip(".").count(".")) if numbered else 2
                        page_blocks.append(
                            (
                                bbox,
                                ContentBlock(
                                    block_type=HEADING if is_heading else PARAGRAPH,
                                    text=text,
                                    location={
                                        "page": page_num,
                                        "paragraph_start": para_idx,
                                        "paragraph_end": para_idx,
                                        "bboxes": [bbox],
                                        "page_width": pw,
                                        "page_height": ph,
                                    },
                                    extraction_method="pymupdf",
                                    heading_level=heading_level if is_heading else None,
                                ),
                            )
                        )

                    # Sort all content blocks (tables and text blocks) on the page in visual reading order (y0, then x0)
                    page_blocks.sort(key=lambda item: (item[0][1], item[0][0]))
                    for _, blk in page_blocks:
                        out.blocks.append(blk)
                else:
                    # Scanned page or sparse native text -> render and OCR
                    ocr_pages += 1
                    zoom = settings.ocr_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_res = run_ocr(img)
                    if ocr_res.error:
                        out.warnings.append(f"Page {page_num}: {ocr_res.error}")
                    scale_x = pw / max(1, ocr_res.width)
                    scale_y = ph / max(1, ocr_res.height)
                    paragraphs = group_lines_into_paragraphs(ocr_res.lines)
                    if not paragraphs and raw_text:
                        out.blocks.append(
                            ContentBlock(
                                block_type=PARAGRAPH,
                                text=raw_text,
                                location={"page": page_num, "paragraph_start": 1, "paragraph_end": 1, "page_width": pw, "page_height": ph},
                                extraction_method="pymupdf",
                            )
                        )
                    elif not paragraphs:
                        low_text_pages += 1
                        out.warnings.append(f"Page {page_num}: insufficient extractable text even after OCR.")
                    else:
                        for p_i, para_lines in enumerate(paragraphs, start=1):
                            p_text = "\n".join(l.text for l in para_lines).strip()
                            if not p_text:
                                continue
                            bboxes = [
                                [
                                    round(l.bbox[0] * scale_x, 1),
                                    round(l.bbox[1] * scale_y, 1),
                                    round(l.bbox[2] * scale_x, 1),
                                    round(l.bbox[3] * scale_y, 1),
                                ]
                                for l in para_lines
                            ]
                            conf = sum(l.confidence for l in para_lines) / len(para_lines)
                            ocr_confs.append(conf)
                            is_heading = len(para_lines) == 1 and len(p_text) < 90 and bool(_HEADING_RE.match(p_text))
                            out.blocks.append(
                                ContentBlock(
                                    block_type=HEADING if is_heading else PARAGRAPH,
                                    text=p_text,
                                    location={
                                        "page": page_num,
                                        "paragraph_start": p_i,
                                        "paragraph_end": p_i,
                                        "bboxes": bboxes,
                                        "page_width": pw,
                                        "page_height": ph,
                                    },
                                    extraction_method=f"ocr:{ocr_res.provider}",
                                    ocr_confidence=round(conf, 4),
                                    heading_level=2 if is_heading else None,
                                )
                            )
                        if sum(len(l.text) for l in ocr_res.lines) < settings.ocr_min_page_chars:
                            low_text_pages += 1

            mean_ocr_conf = round(sum(ocr_confs) / len(ocr_confs), 4) if ocr_confs else None
            if mean_ocr_conf is not None and mean_ocr_conf < settings.ocr_low_confidence:
                out.warnings.append(f"Low average OCR confidence ({mean_ocr_conf:.2f}); document may be poorly scanned.")
            out.quality = {
                "pages": doc.page_count,
                "native_pages": native_pages,
                "ocr_pages": ocr_pages,
                "low_text_pages": low_text_pages,
                "mean_ocr_confidence": mean_ocr_conf,
                "tables": table_counter,
            }
            return out
        finally:
            doc.close()


class ImageParser(DocumentParser):
    types = frozenset({"png", "jpg", "jpeg"})

    def parse(self, path: Path) -> ExtractedDocument:
        settings = get_settings()
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
        try:
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                if width * height > settings.max_image_pixels:
                    raise ExtractionError("Image exceeds maximum pixel limit")
                rgb = im.convert("RGB")
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Could not open image: {type(exc).__name__}") from exc

        ext = path.suffix.lower().lstrip(".")
        detected = "jpg" if ext == "jpeg" else (ext or "png")
        out = ExtractedDocument(detected_type=detected, page_count=1)
        ocr_res = run_ocr(rgb)
        if ocr_res.error:
            out.warnings.append(ocr_res.error)
        paragraphs = group_lines_into_paragraphs(ocr_res.lines)
        for p_i, para_lines in enumerate(paragraphs, start=1):
            p_text = "\n".join(l.text for l in para_lines).strip()
            if not p_text:
                continue
            bboxes = [l.bbox for l in para_lines]
            conf = sum(l.confidence for l in para_lines) / len(para_lines)
            is_heading = len(para_lines) == 1 and len(p_text) < 90 and bool(_HEADING_RE.match(p_text))
            out.blocks.append(
                ContentBlock(
                    block_type=HEADING if is_heading else PARAGRAPH,
                    text=p_text,
                    location={
                        "page": 1,
                        "paragraph_start": p_i,
                        "paragraph_end": p_i,
                        "bboxes": bboxes,
                        "page_width": float(width),
                        "page_height": float(height),
                    },
                    extraction_method=f"ocr:{ocr_res.provider}",
                    ocr_confidence=round(conf, 4),
                    heading_level=2 if is_heading else None,
                )
            )
        mean_conf = ocr_res.mean_confidence
        if mean_conf is not None and mean_conf < settings.ocr_low_confidence:
            out.warnings.append(f"Low OCR confidence ({mean_conf:.2f}); image may be blurry or low resolution.")
        if not out.blocks:
            out.warnings.append("No text could be extracted from image via OCR.")
        out.quality = {
            "pages": 1,
            "native_pages": 0,
            "ocr_pages": 1,
            "low_text_pages": 0 if out.total_chars >= settings.ocr_min_page_chars else 1,
            "mean_ocr_confidence": round(mean_conf, 4) if mean_conf is not None else None,
            "image_size": [width, height],
        }
        return out
