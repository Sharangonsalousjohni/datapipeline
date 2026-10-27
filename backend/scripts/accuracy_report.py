"""Accuracy report: compare a handoff JSON with a golden expectations file.

Reports text-match %, cell-capture %, normalized-value accuracy %, narrative-fact recall % and
how many injected errors validation caught.

Usage (from backend/):
    python scripts/accuracy_report.py --handoff path/to/handoff.json
    python scripts/accuracy_report.py --run                 # run the pipeline on the golden PDF
    python scripts/accuracy_report.py --run --inject        # ...and on three corrupted copies

The script is self-contained on purpose: in --run mode it only calls functions that exist in every
version of the pipeline (parse/chunk/classify/build_structured_extraction), preferring the in-memory
handoff path when available, so it can also score older checkouts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parent.parent
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
DEFAULT_GOLDEN = BACKEND / "tests" / "fixtures" / "sample_annual_report.golden.json"
DEFAULT_PDF = BACKEND / "tests" / "fixtures" / "sample_annual_report.pdf"


# --------------------------------------------------------------------------- helpers
def _dec(v: Any) -> Decimal | None:
    if v in (None, ""):
        return None
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return None


def _items(handoff: dict) -> list[dict]:
    """Accept a RelevantChunksResponse, a DocumentReport, or a StructuredExtractionReport."""
    if "handoff" in handoff and isinstance(handoff["handoff"], list):
        return handoff["handoff"]
    return handoff.get("items") or []


def _text(item: dict) -> str:
    return item.get("extracted_text") or item.get("source_text") or ""


def _value(f: dict) -> Decimal | None:
    return _dec(f.get("normalized_amount")) if f.get("normalized_amount") not in (None, "") else _dec(f.get("normalized_value"))


def _ws(s: str) -> str:
    return " ".join((s or "").split())


# --------------------------------------------------------------------------- scoring
def score(handoff: dict, golden: dict) -> dict[str, Any]:
    items = _items(handoff)

    # 1. text match: multiset recall of golden words in the handed-off text
    gold_words = Counter(golden["words"])
    got_words = Counter(w for it in items for w in _text(it).replace(" | ", " ").split())
    text_hits = sum(min(n, got_words[w]) for w, n in gold_words.items())
    text_pct = 100.0 * text_hits / max(1, sum(gold_words.values()))

    # 2/3. cells: find each golden cell in the item holding its table, compare normalized value
    cells_total = cells_found = values_ok = types_ok = 0
    value_errors: list[str] = []
    for t in golden["tables"]:
        t_items = [it for it in items if t["caption"] in _text(it) and "|" in _text(it)]
        t_fields = [f for it in t_items for f in it.get("financial_fields", [])]
        for row in t["rows"]:
            for cell in row["cells"]:
                cells_total += 1
                match = next(
                    (f for f in t_fields
                     if _ws(f.get("row_label") or "") == _ws(row["label"])
                     and _ws(f.get("reporting_period") or "") == _ws(cell["period"])
                     and _ws(f.get("original_text") or "") == _ws(cell["raw"])),
                    None,
                )
                if match is None:
                    value_errors.append(f"MISSING {t['caption']} / {row['label']} / {cell['period']}")
                    continue
                cells_found += 1
                if _value(match) == Decimal(cell["value"]):
                    values_ok += 1
                else:
                    value_errors.append(
                        f"{t['caption']} / {row['label']} / {cell['period']}: got {_value(match)}, expected {cell['value']}"
                    )
                if match.get("data_type") == cell["data_type"]:
                    types_ok += 1

    # 4. narrative recall: each golden fact needs its own matching narrative field
    narrative_fields = [
        f for it in items if "|" not in _text(it) for f in it.get("financial_fields", [])
    ]
    used: set[int] = set()
    facts_found = 0
    missed: list[str] = []
    for fact in golden["narrative_facts"]:
        hit = None
        for i, f in enumerate(narrative_fields):
            if i in used:
                continue
            if (
                f.get("metric") == fact["metric"]
                and (f.get("measure") or "level") == fact["measure"]
                and (f.get("period_key") or f.get("reporting_period")) == fact["period"]
                and _value(f) == Decimal(fact["value"])
            ):
                hit = i
                break
        if hit is None:
            missed.append(f"{fact['metric']} {fact['measure']} {fact['period']} = {fact['value']} ('{fact['text']}')")
        else:
            used.add(hit)
            facts_found += 1

    all_fields = [f for it in items for f in it.get("financial_fields", [])]
    statuses = Counter(f.get("validation_status") for f in all_fields)
    confidences = Counter(f.get("confidence") for f in all_fields)
    return {
        "text_match_pct": round(text_pct, 1),
        "cells_total": cells_total,
        "cell_capture_pct": round(100.0 * cells_found / max(1, cells_total), 1),
        "normalized_value_accuracy_pct": round(100.0 * values_ok / max(1, cells_total), 1),
        "normalized_values_correct": values_ok,
        "data_type_accuracy_pct": round(100.0 * types_ok / max(1, cells_total), 1),
        "narrative_facts_total": len(golden["narrative_facts"]),
        "narrative_recall_pct": round(100.0 * facts_found / max(1, len(golden["narrative_facts"])), 1),
        "narrative_facts_found": facts_found,
        "narrative_extra_fields": len(narrative_fields) - len(used),
        "fields_total": len(all_fields),
        "validation_status_counts": dict(statuses),
        "review_required_fields": sum(1 for f in all_fields if f.get("review_required")),
        "distinct_confidence_values": len(confidences),
        "value_errors": value_errors[:12],
        "missed_facts": missed,
    }


# --------------------------------------------------------------------------- running the pipeline
def run_pipeline(pdf: Path, doc_id: str = "accuracy-run") -> dict:
    try:
        from app.handoff import process_file_to_handoff  # current pipeline
    except ImportError:
        process_file_to_handoff = None
    if process_file_to_handoff is not None:
        return process_file_to_handoff(pdf, document_id=doc_id, filename=DEFAULT_PDF.name).model_dump(mode="json")

    # Older checkouts: parse -> chunk -> classify -> structured extraction report
    from app.chunking import chunk_document
    from app.classifier import classify_chunks
    from app.db import session_scope
    from app.extraction.registry import parse_document
    from app.financial_extraction import build_structured_extraction
    from app.reference import ensure_reference_dataset_loaded

    with session_scope() as session:
        ensure_reference_dataset_loaded(session)
    extracted = parse_document(pdf, "pdf")
    drafts = chunk_document(doc_id, extracted)
    batch = classify_chunks(drafts)
    chunk_dicts = [
        {"chunk_id": d.id, "page_number": d.location.get("page"), "section_heading": d.location.get("section_heading"),
         "decision": c.decision, "top_category": c.top_category, "score": c.score, "text": d.text,
         "location": d.location, "bboxes": d.location.get("bboxes")}
        for d, c in zip(drafts, batch.items)
    ]
    report = build_structured_extraction(doc_id, DEFAULT_PDF.name, "FINANCIAL_RELEVANT", chunk_dicts)
    return report.model_dump(mode="json")


# --------------------------------------------------------------------------- error injection
def _rewrite_pdf(src: Path, dst: Path, page_no: int, old: str, new: str | None, occurrence: int = 0) -> None:
    import pymupdf

    doc = pymupdf.open(str(src))
    page = doc[page_no - 1]
    hits = page.search_for(old)
    if len(hits) <= occurrence:
        raise ValueError(f"'{old}' not found on page {page_no}")
    rect = hits[occurrence]
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
             if pymupdf.Rect(s["bbox"]).intersects(rect)]
    size = spans[0]["size"] if spans else 10
    page.add_redact_annot(rect)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    if new:
        # right-align replacement numbers in the same cell, left-align labels
        width = pymupdf.get_text_length(new, fontname="helv", fontsize=size)
        x = rect.x1 - width if re.fullmatch(r"[\d,.$()%-]+", new) else rect.x0
        page.insert_text((x, rect.y1 - size * 0.22), new, fontname="helv", fontsize=size)
    doc.save(str(dst))
    doc.close()


INJECTIONS = [
    {
        "name": "unit line removed from the income statement",
        "edit": (3, "($ in thousands)", None, 0),
        "expect": "income-statement monetary fields have reported_scale='unknown' and review_required=true",
    },
    {
        "name": "balance-sheet total corrupted (Total current assets 2025: 327,900 -> 337,900)",
        "edit": (4, "327,900", "337,900", 0),
        "expect": "that field is FAILED with a validation issue",
    },
    {
        "name": "income statement unit changed to millions (disagrees with Financial Highlights)",
        "edit": (3, "($ in thousands)", "($ in millions)", 0),
        "expect": "cross-table validation fails",
    },
]


def _flagged(name: str, out: dict) -> tuple[bool, str]:
    items = _items(out)
    is_fields = [f for it in items if "Statements of Income" in _text(it) for f in it.get("financial_fields", [])]
    if name.startswith("unit line removed"):
        money = [f for f in is_fields if f.get("data_type") in ("monetary", "monetary_change")]
        ok = bool(money) and all(f.get("reported_scale") == "unknown" and f.get("review_required") for f in money)
        return ok, f"{sum(f.get('reported_scale') == 'unknown' for f in money)}/{len(money)} IS monetary fields unknown-scale"
    if name.startswith("balance-sheet total"):
        f = next(
            (f for it in items for f in it.get("financial_fields", [])
             if _ws(f.get("row_label") or "") == "Total current assets" and "2025" in (f.get("reporting_period") or "")),
            None,
        )
        if f is None:
            return False, "field not found"
        ok = f.get("validation_status") == "FAILED" and bool(f.get("validation_issues"))
        return ok, f"status={f.get('validation_status')}, issues={[i.get('rule_id') for i in f.get('validation_issues') or []]}"
    summary = out.get("validation_summary") or {}
    checks = (summary.get("cross_reference_checks") or []) + (summary.get("accounting_validations") or [])
    failed = [c for c in checks if c.get("status") == "FAILED" and "XTABLE" in str(c.get("rule_id"))]
    rev_failed = [f for f in is_fields if (f.get("row_label") == "Revenue") and f.get("validation_status") == "FAILED"]
    return bool(failed or rev_failed), f"{len(failed)} failed cross-table checks, {len(rev_failed)} IS revenue fields FAILED"


def run_injections(pdf: Path) -> list[dict[str, Any]]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, inj in enumerate(INJECTIONS):
            dst = Path(tmp) / f"injected_{i}.pdf"
            page_no, old, new, occ = inj["edit"]
            _rewrite_pdf(pdf, dst, page_no, old, new, occ)
            out = run_pipeline(dst, doc_id=f"accuracy-inject-{i}")
            ok, detail = _flagged(inj["name"], out)
            results.append({"injection": inj["name"], "expected": inj["expect"], "caught": ok, "detail": detail})
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", type=Path, help="handoff / report JSON to score")
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--run", action="store_true", help="run the pipeline on --pdf instead of reading --handoff")
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--inject", action="store_true", help="also run the three error injections")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    if args.run:
        handoff = run_pipeline(args.pdf)
    elif args.handoff:
        handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    else:
        ap.error("pass --handoff FILE or --run")
    result = score(handoff, golden)
    if args.inject:
        result["injections"] = run_injections(args.pdf)
        result["injected_errors_caught"] = f"{sum(r['caught'] for r in result['injections'])}/{len(result['injections'])}"

    print(f"text match            {result['text_match_pct']:6.1f} %")
    print(f"cell capture          {result['cell_capture_pct']:6.1f} %  ({result['cells_total']} cells)")
    print(f"normalized accuracy   {result['normalized_value_accuracy_pct']:6.1f} %  ({result['normalized_values_correct']}/{result['cells_total']})")
    print(f"data-type accuracy    {result['data_type_accuracy_pct']:6.1f} %")
    print(f"narrative recall      {result['narrative_recall_pct']:6.1f} %  ({result['narrative_facts_found']}/{result['narrative_facts_total']}, {result['narrative_extra_fields']} extra)")
    print(f"fields                {result['fields_total']}  status={result['validation_status_counts']}  review_required={result['review_required_fields']}  distinct confidences={result['distinct_confidence_values']}")
    if "injections" in result:
        print(f"injected errors caught {result['injected_errors_caught']}")
        for r in result["injections"]:
            print(f"  [{'x' if r['caught'] else ' '}] {r['injection']}: {r['detail']}")
    if result["value_errors"]:
        print("first value errors:", *result["value_errors"][:5], sep="\n  ")
    if result["missed_facts"] and result["narrative_facts_found"]:
        print("missed facts:", *result["missed_facts"], sep="\n  ")
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
