"""Stage C: Structure-Aware Chunking.

Splits extracted content into meaningful chunks while:
- Producing one chunk per section (split on detected headings) and keeping its heading path.
- Never splitting a table across chunks: each table gets its own chunk with its caption,
  unit line and header row at the top (only tables above ``table_max_rows_per_chunk`` rows are
  sliced, and every slice repeats caption, unit line and header).
- Splitting narrative sections larger than ``chunk_max_tokens`` at paragraph boundaries with a
  one-sentence overlap; every piece keeps its parent heading.
- Recording unit, period columns, page, section, statement type and role in each chunk's
  location so a number can be interpreted without neighbouring chunks.
- Labelling cover pages, tables of contents and signature blocks (kept, never dropped).
- Assigning a stable deterministic ID to each chunk and deduplicating identical chunks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.extraction.base import HEADING, TABLE, ContentBlock, ExtractedDocument, TableData
from app.extraction.tabular import col_letter, render_table
from app.numbers import UnitSpec, detect_document_unit, detect_unit
from app.taxonomy import detect_fiscal_period, detect_statement_type, is_change_column, match_metric

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_CURRENCY_HINT_RE = re.compile(r"[$€£¥₹]\s?\d|\b(?:USD|EUR|GBP)\s?\d|\d\s?(?:%|percent)\b", re.I)
_SIGNATURE_RE = re.compile(
    r"^(?:sincerely|respectfully(?:\s+submitted)?|yours\s+(?:truly|faithfully|sincerely)|(?:best|kind|warm)\s+regards|regards)\b[,.]?"
    r"|^/s/|\bin\s+witness\s+whereof\b|\b(?:authorized|tenant|landlord|witness)\s+signature\b|^signature\s*:|^signed\s*[:,]",
    re.I,
)
_TOC_HEADING_RE = re.compile(r"^\s*(?:table\s+of\s+)?contents\s*$", re.I)
_TOC_LINE_RE = re.compile(r"(?:\.{3,}|\s{2,}|\t)\s*\d{1,3}\s*$|^\S.{2,80}\s\d{1,3}$")

ROLE_NARRATIVE = "narrative"
ROLE_TABLE = "table"
ROLE_COVER = "cover_page"
ROLE_TOC = "table_of_contents"
ROLE_SIGNATURE = "signature_block"
NON_FINANCIAL_ROLES = frozenset({ROLE_COVER, ROLE_TOC, ROLE_SIGNATURE})


@dataclass
class ChunkDraft:
    id: str
    document_id: str
    chunk_index: int
    chunk_type: str  # prose | table
    text: str
    context: str | None
    location: dict[str, Any]
    extraction_method: str
    ocr_confidence: float | None
    content_hash: str
    extra_locations: list[dict[str, Any]] = field(default_factory=list)
    section_heading: str | None = None


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 words per token for English prose)."""
    return int(len(text.split()) * 4 / 3 + 0.5)


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_into_sentences(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Character-budget sentence splitter (kept for oversized single paragraphs)."""
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        if len(sent) > max_chars:
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_len = [], 0
            step = max(100, max_chars - overlap_chars)
            for i in range(0, len(sent), step):
                pieces.append(sent[i : i + max_chars])
            continue
        add_len = len(sent) + (1 if cur else 0)
        if cur and cur_len + add_len > max_chars:
            pieces.append(" ".join(cur))
            tail: list[str] = []
            t_len = 0
            for prev_s in reversed(cur):
                if t_len + len(prev_s) <= overlap_chars:
                    tail.insert(0, prev_s)
                    t_len += len(prev_s) + 1
                else:
                    break
            cur = tail + [sent]
            cur_len = sum(len(s) for s in cur) + max(0, len(cur) - 1)
        else:
            cur.append(sent)
            cur_len += add_len
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _merge_locations(locs: list[dict[str, Any]], heading_path: list[str]) -> dict[str, Any]:
    if not locs:
        return {"heading_path": heading_path or None, "section_heading": heading_path[-1] if heading_path else None}
    first = dict(locs[0])
    last = locs[-1]
    for k in ("paragraph_end", "char_end", "line_end"):
        if last.get(k) is not None:
            first[k] = last[k]
    bboxes: list[list[float]] = []
    bbox_pages: list[int] = []
    for l in locs:
        for b in l.get("bboxes") or []:
            bboxes.append(b)
            if l.get("page") is not None:
                bbox_pages.append(l["page"])
    if bboxes:
        first["bboxes"] = bboxes
        if len(bbox_pages) == len(bboxes) and len(set(bbox_pages)) > 1:
            first["bbox_pages"] = bbox_pages
    pages = [l["page"] for l in locs if l.get("page") is not None]
    if pages and max(pages) != pages[0]:
        first["page_end"] = max(pages)
    if heading_path:
        first["heading_path"] = list(heading_path)
        first["section_heading"] = heading_path[-1]
    return first


def _stable_chunk_id(document_id: str, index: int, content_hash: str, location: dict[str, Any]) -> str:
    loc_key = json.dumps(
        {
            "page": location.get("page"),
            "sheet": location.get("sheet"),
            "row_start": location.get("row_start"),
            "paragraph_start": location.get("paragraph_start"),
            "idx": index,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(f"{document_id}:{content_hash}:{loc_key}".encode("utf-8")).hexdigest()[:28]
    return f"chk_{digest}"


def _is_signature_block(block: ContentBlock) -> bool:
    return block.block_type != TABLE and bool(_SIGNATURE_RE.search(block.text.strip()))


def _is_toc(heading: str | None, blocks: list[ContentBlock]) -> bool:
    if heading and _TOC_HEADING_RE.match(heading):
        return True
    lines = [ln.strip() for b in blocks for ln in b.text.splitlines() if ln.strip()]
    return len(lines) >= 4 and sum(1 for ln in lines if _TOC_LINE_RE.search(ln)) / len(lines) >= 0.7


def _cover_page_blocks(extracted: ExtractedDocument) -> set[int]:
    """Indices of first-page blocks that form a cover page (title-only page without amounts)."""
    if (extracted.page_count or 0) < 2:
        return set()
    first_page = [i for i, b in enumerate(extracted.blocks) if b.location.get("page") == 1]
    if not first_page:
        return set()
    blocks = [extracted.blocks[i] for i in first_page]
    text = "\n".join(b.text for b in blocks)
    if any(b.block_type == TABLE for b in blocks) or _CURRENCY_HINT_RE.search(text) or len(text.split()) > 120:
        return set()
    return set(first_page)


def _table_unit(t: TableData, heading_path: list[str], doc_unit: UnitSpec | None) -> UnitSpec:
    """Unit search order: header row, caption, line just above the table, section heading, document."""
    candidates = [
        (" ".join(t.headers), "header"),
        (t.caption, "caption"),
        (t.unit_line, "line_above"),
        (" ".join(heading_path[-1:]), "section_heading"),
    ]
    for text, source in candidates:
        u = detect_unit(text, source=source)
        if u:
            return u
    if doc_unit is not None:
        return doc_unit
    return UnitSpec(scale="unknown", source=None)


def _table_metadata(t: TableData, heading_path: list[str], doc_unit: UnitSpec | None) -> dict[str, Any]:
    unit = _table_unit(t, heading_path, doc_unit)
    periods = [h for h in t.headers[1:] if h and h.strip()]
    metrics = [m for m in (match_metric(r[0], allow_embedding=False).metric for r in t.rows if r) if m]
    st = detect_statement_type(
        heading_path + ([t.caption] if t.caption else []),
        metrics,
        is_table=True,
        has_change_column=any(is_change_column(p) for p in periods),
    )
    return {"unit": unit.to_dict(), "period_columns": periods, "statement_type": st, "chunk_role": ROLE_TABLE}


def chunk_document(
    document_id: str,
    extracted: ExtractedDocument,
    *,
    chunk_size_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
    table_rows_per_chunk: int | None = None,
    chunk_max_tokens: int | None = None,
) -> list[ChunkDraft]:
    settings = get_settings()
    max_chars = chunk_size_chars or settings.chunk_size_chars
    overlap = chunk_overlap_chars if chunk_overlap_chars is not None else settings.chunk_overlap_chars
    max_table_rows = table_rows_per_chunk or settings.table_max_rows_per_chunk
    max_tokens = chunk_max_tokens or settings.chunk_max_tokens

    sample = "\n".join(b.text for b in extracted.blocks[:40])
    fiscal_period = detect_fiscal_period(sample)
    doc_unit = detect_document_unit(sample)

    heading_stack: list[tuple[int, str]] = []
    section_heading_block: ContentBlock | None = None
    section_blocks: list[ContentBlock] = []

    raw_chunks: list[tuple[str, str, str | None, dict[str, Any], str, float | None]] = []

    def current_headings() -> list[str]:
        return [t for _, t in heading_stack]

    def base_meta(extra: dict[str, Any]) -> dict[str, Any]:
        meta = {"fiscal_period": fiscal_period}
        meta.update(extra)
        return meta

    def emit_prose(blocks: list[ContentBlock], role: str, heading_block: ContentBlock | None) -> None:
        if not blocks and heading_block is None:
            return
        h_path = current_headings()
        context = " > ".join(h_path) if h_path else None
        all_blocks = ([heading_block] if heading_block is not None else []) + blocks
        loc = _merge_locations([b.location for b in all_blocks], h_path)
        method = all_blocks[0].extraction_method
        confs = [b.ocr_confidence for b in all_blocks if b.ocr_confidence is not None]
        mean_conf = round(sum(confs) / len(confs), 4) if confs else None
        st = detect_statement_type(h_path) if role == ROLE_NARRATIVE else None
        meta = base_meta({"chunk_role": role, "statement_type": st, "unit": doc_unit.to_dict() if doc_unit else None})
        if fiscal_period:
            meta["period_columns"] = [fiscal_period["key"]]
        head_text = heading_block.text if heading_block is not None else None
        paragraphs = [b.text for b in blocks]

        if role != ROLE_NARRATIVE or estimate_tokens("\n\n".join(([head_text] if head_text else []) + paragraphs)) <= max_tokens:
            text = "\n\n".join(([head_text] if head_text else []) + paragraphs)
            raw_chunks.append(("prose", text.strip(), context, {**loc, **meta}, method, mean_conf))
            return

        # Oversized narrative: pack paragraphs (sentences for oversized paragraphs) under the token budget,
        # carrying the parent heading on every piece and repeating the previous piece's last sentence.
        units: list[str] = []
        for p in paragraphs:
            if estimate_tokens(p) > max_tokens:
                budget_chars = max(200, int(max_tokens * 3))
                units.extend(_split_into_sentences(p, budget_chars, 0))
            else:
                units.append(p)
        pieces: list[list[str]] = []
        cur: list[str] = []
        for u in units:
            if cur and estimate_tokens("\n\n".join(cur + [u])) > max_tokens:
                pieces.append(cur)
                cur = []
            cur.append(u)
        if cur:
            pieces.append(cur)
        prev_last_sentence: str | None = None
        for i, piece in enumerate(pieces):
            prefix = f"{head_text}\n\n" if head_text else ""
            ov: list[int] | None = None
            if prev_last_sentence:
                ov = [len(prefix), len(prefix) + len(prev_last_sentence)]
                prefix += prev_last_sentence + "\n\n"
            text = prefix + "\n\n".join(piece)
            piece_meta = dict(meta)
            piece_meta["table_part"] = None
            if ov:
                piece_meta["overlap"] = ov
            raw_chunks.append(("prose", text.strip(), context, {**loc, **piece_meta}, method, mean_conf))
            sents = _sentences(piece[-1])
            prev_last_sentence = sents[-1].strip() if sents else None

    def flush_section() -> None:
        nonlocal section_blocks, section_heading_block
        blocks = section_blocks
        heading_block = section_heading_block
        section_blocks = []
        section_heading_block = None
        if not blocks and heading_block is None:
            return
        # Trailing sign-off / signature paragraphs become their own non-financial chunk.
        sig_start = len(blocks)
        while sig_start > 0 and _is_signature_block(blocks[sig_start - 1]):
            sig_start -= 1
        body, sig = blocks[:sig_start], blocks[sig_start:]
        role = ROLE_TOC if _is_toc(heading_block.text if heading_block else None, body) else ROLE_NARRATIVE
        if body or heading_block is not None:
            emit_prose(body, role, heading_block)
        if sig:
            emit_prose(sig, ROLE_SIGNATURE, None)

    def emit_table(block: ContentBlock) -> None:
        t: TableData = block.table  # type: ignore[assignment]
        h_path = current_headings()
        ctx_parts = list(h_path)
        if block.location.get("sheet"):
            ctx_parts.append(f"Sheet: {block.location['sheet']}")
        if t.caption and t.caption not in ctx_parts:
            ctx_parts.append(t.caption)
        context = " > ".join(ctx_parts) if ctx_parts else None
        meta = base_meta(_table_metadata(t, h_path, doc_unit))
        cell_boxes = t.cell_bboxes or block.location.get("cell_bboxes")

        total_rows = len(t.rows)
        n_parts = max(1, -(-total_rows // max_table_rows)) if total_rows else 1
        for part, start_i in enumerate(range(0, max(total_rows, 1), max_table_rows), start=1):
            slice_rows = t.rows[start_i : start_i + max_table_rows]
            slice_nums = t.row_numbers[start_i : start_i + max_table_rows] if t.row_numbers else []
            sub_table = TableData(
                headers=t.headers,
                rows=slice_rows,
                row_numbers=slice_nums,
                header_row_number=t.header_row_number,
                col_offset=t.col_offset,
                caption=t.caption,
                footnotes=t.footnotes if (start_i + max_table_rows >= total_rows) else [],
                unit_line=t.unit_line,
            )
            text = render_table(sub_table)
            loc = dict(block.location)
            loc.update(meta)
            if cell_boxes:
                loc["cell_bboxes"] = [cell_boxes[0]] + list(cell_boxes[1 + start_i : 1 + start_i + max_table_rows])
            if t.bbox and not loc.get("table_bbox"):
                loc["table_bbox"] = t.bbox
            if n_parts > 1:
                loc["table_part"] = [part, n_parts]
            if slice_nums:
                r_start = t.header_row_number if (start_i == 0 and t.header_row_number) else slice_nums[0]
                r_end = slice_nums[-1]
                loc["row_start"] = r_start
                loc["row_end"] = r_end
                c_start = loc.get("col_start", 1)
                c_end = loc.get("col_end", max(1, len(t.headers)))
                cell_r = f"{col_letter(c_start)}{r_start}:{col_letter(c_end)}{r_end}"
                loc["cell_range"] = f"{loc['sheet']}!{cell_r}" if loc.get("sheet") else cell_r
            if h_path:
                loc["heading_path"] = h_path
                loc["section_heading"] = h_path[-1]
            raw_chunks.append(("table", text, context, loc, block.extraction_method, block.ocr_confidence))
            if not total_rows:
                break

    cover = _cover_page_blocks(extracted)
    if cover:
        cover_blocks = [extracted.blocks[i] for i in sorted(cover)]
        emit_prose(cover_blocks, ROLE_COVER, None)

    for idx, block in enumerate(extracted.blocks):
        if idx in cover:
            continue
        # Signature labels should remain part of execution blocks, not treated as independent headings
        is_sig_header = (
            block.block_type == HEADING
            and bool(re.search(r"\b(authorized\s+signature|tenant\s+signature|landlord\s+signature|witness\s+signature)\b", block.text, re.I))
        )

        if block.block_type == HEADING and not is_sig_header:
            flush_section()
            lvl = block.heading_level or 2
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, block.text))
            section_heading_block = block
            continue

        if block.block_type == TABLE and block.table is not None:
            t = block.table
            if section_heading_block is not None and not section_blocks and not t.caption:
                # A heading directly above the table is its caption, not a separate prose chunk.
                t.caption = section_heading_block.text
                section_heading_block = None
            # A unit line ("($ in millions)") just above the table belongs to the table.
            if section_blocks and not t.unit_line:
                prev = section_blocks[-1]
                if (
                    prev.location.get("page") == block.location.get("page")
                    and len(prev.text) < 120
                    and detect_unit(prev.text)
                ):
                    t.unit_line = prev.text.strip()
                    section_blocks.pop()
            flush_section()
            emit_table(block)
            continue

        section_blocks.append(block)

    flush_section()

    # Deduplicate identical chunks while retaining extra source locations
    deduped: list[ChunkDraft] = []
    by_hash: dict[str, ChunkDraft] = {}
    stream_offset = 0
    for c_type, text, context, loc, method, conf in raw_chunks:
        if not text:
            continue
        loc = {k: v for k, v in loc.items() if v is not None}
        norm = " ".join(text.split()).lower()
        c_hash = hashlib.sha256(f"{c_type}:{context or ''}:{norm}".encode("utf-8")).hexdigest()
        if loc.get("char_start") is None:
            loc["char_start"] = stream_offset
            loc["char_end"] = stream_offset + len(text)
        stream_offset += len(text) + 2

        existing = by_hash.get(c_hash)
        if existing is not None:
            existing.extra_locations.append(loc)
            continue

        idx = len(deduped)
        cid = _stable_chunk_id(document_id, idx, c_hash, loc)
        sec_head = loc.get("section_heading") or (context.split(" > ")[-1] if context else None)
        draft = ChunkDraft(
            id=cid,
            document_id=document_id,
            chunk_index=idx,
            chunk_type=c_type,
            text=text,
            context=context,
            location=loc,
            extraction_method=method,
            ocr_confidence=conf,
            content_hash=c_hash,
            section_heading=sec_head,
        )
        by_hash[c_hash] = draft
        deduped.append(draft)

    return deduped
