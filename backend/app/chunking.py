"""Stage C: Structure-Aware Chunking.

Splits extracted content into meaningful chunks while:
- Preserving section headings and nearby context (heading breadcrumb path).
- Keeping financial clauses with surrounding sentences (sentence-boundary splitting).
- Keeping spreadsheet headers, captions, and footnotes with their row slices.
- Preserving source-location metadata for every chunk.
- Assigning a stable deterministic ID to each chunk and deduplicating identical chunks.
- Never silently discarding content because a document is long.
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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


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


def _split_into_sentences(text: str, max_chars: int, overlap_chars: int) -> list[str]:
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
            # keep trailing sentences up to overlap_chars
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
    if last.get("paragraph_end") is not None:
        first["paragraph_end"] = last["paragraph_end"]
    if last.get("char_end") is not None:
        first["char_end"] = last["char_end"]
    if last.get("line_end") is not None:
        first["line_end"] = last["line_end"]
    bboxes: list[list[float]] = []
    for l in locs:
        for b in l.get("bboxes") or []:
            bboxes.append(b)
    if bboxes:
        first["bboxes"] = bboxes
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


def chunk_document(
    document_id: str,
    extracted: ExtractedDocument,
    *,
    chunk_size_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
    table_rows_per_chunk: int | None = None,
) -> list[ChunkDraft]:
    settings = get_settings()
    max_chars = chunk_size_chars or settings.chunk_size_chars
    overlap = chunk_overlap_chars if chunk_overlap_chars is not None else settings.chunk_overlap_chars
    rows_per_chunk = table_rows_per_chunk or settings.table_rows_per_chunk

    heading_stack: list[tuple[int, str]] = []
    pending_heading: ContentBlock | None = None

    # Accumulator for adjacent prose blocks on the same page
    buf_blocks: list[ContentBlock] = []
    buf_len = 0

    raw_chunks: list[tuple[str, str, str | None, dict[str, Any], str, float | None]] = []

    def current_headings() -> list[str]:
        return [t for _, t in heading_stack]

    def flush_prose() -> None:
        nonlocal buf_blocks, buf_len, pending_heading
        if not buf_blocks and pending_heading is not None:
            buf_blocks.append(pending_heading)
            pending_heading = None
        if not buf_blocks:
            return
        h_path = current_headings()
        context = " > ".join(h_path) if h_path else None
        combined = "\n\n".join(b.text for b in buf_blocks)
        loc = _merge_locations([b.location for b in buf_blocks], h_path)
        method = buf_blocks[0].extraction_method
        confs = [b.ocr_confidence for b in buf_blocks if b.ocr_confidence is not None]
        mean_conf = round(sum(confs) / len(confs), 4) if confs else None

        for piece in _split_into_sentences(combined, max_chars, overlap):
            raw_chunks.append(("prose", piece.strip(), context, dict(loc), method, mean_conf))
        buf_blocks = []
        buf_len = 0

    for block in extracted.blocks:
        # Signature labels should remain part of execution blocks, not treated as independent headings
        is_sig_header = (
            block.block_type == HEADING
            and bool(re.search(r"\b(authorized\s+signature|tenant\s+signature|landlord\s+signature|witness\s+signature)\b", block.text, re.I))
        )

        if block.block_type == HEADING and not is_sig_header:
            flush_prose()
            lvl = block.heading_level or 2
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, block.text))
            pending_heading = block
            continue

        if block.block_type == TABLE and block.table is not None:
            t: TableData = block.table
            if pending_heading is not None:
                if not t.caption:
                    t.caption = pending_heading.text
                pending_heading = None
            flush_prose()
            h_path = current_headings()
            ctx_parts = list(h_path)
            if block.location.get("sheet"):
                ctx_parts.append(f"Sheet: {block.location['sheet']}")
            if t.caption and t.caption not in ctx_parts:
                ctx_parts.append(t.caption)
            context = " > ".join(ctx_parts) if ctx_parts else None

            total_rows = len(t.rows)
            if total_rows == 0:
                text = render_table(t, [])
                loc = dict(block.location)
                if h_path:
                    loc["heading_path"] = h_path
                    loc["section_heading"] = h_path[-1]
                raw_chunks.append(("table", text, context, loc, block.extraction_method, block.ocr_confidence))
            else:
                for start_i in range(0, total_rows, rows_per_chunk):
                    slice_rows = t.rows[start_i : start_i + rows_per_chunk]
                    slice_nums = t.row_numbers[start_i : start_i + rows_per_chunk] if t.row_numbers else []
                    sub_table = TableData(
                        headers=t.headers,
                        rows=slice_rows,
                        row_numbers=slice_nums,
                        header_row_number=t.header_row_number,
                        col_offset=t.col_offset,
                        caption=t.caption,
                        footnotes=t.footnotes if (start_i + rows_per_chunk >= total_rows) else [],
                    )
                    text = render_table(sub_table)
                    loc = dict(block.location)
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
            continue

        # Prose block
        if pending_heading is not None:
            if pending_heading.location.get("page") == block.location.get("page"):
                buf_blocks.append(pending_heading)
                buf_len += len(pending_heading.text) + 2
                pending_heading = None
            else:
                flush_prose()

        same_page = not buf_blocks or (buf_blocks[-1].location.get("page") == block.location.get("page"))
        if buf_blocks and (not same_page or buf_len + len(block.text) + 2 > max_chars):
            flush_prose()
        buf_blocks.append(block)
        buf_len += len(block.text) + 2

    flush_prose()

    # Deduplicate identical chunks while retaining extra source locations
    deduped: list[ChunkDraft] = []
    by_hash: dict[str, ChunkDraft] = {}
    stream_offset = 0
    for c_type, text, context, loc, method, conf in raw_chunks:
        if not text:
            continue
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
