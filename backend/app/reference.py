"""Stage D: Financial Reference Dataset & Vector Index Management.

Manages the curated reference entries, imports licensed JSONL/JSON/CSV datasets,
maintains a local FAISS (or NumPy fallback) inner-product index over L2-normalized
embeddings, and stores vector references on each ReferenceEntry row without
duplicating large vectors inside SQLite rows.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings import get_embedding_service
from app.models import ModelVersion, ReferenceDatasetVersion, ReferenceEntry

log = logging.getLogger(__name__)

CATEGORIES: list[str] = [
    "income_revenue",
    "expenses_operating_costs",
    "financial_performance",
    "accounts_payable_receivable",
    "invoices_purchase_orders",
    "cash_flow_bank_transactions",
    "loans_interest_repayment",
    "rent_leases_recurring_payments",
    "deposits_and_escrow",
    "contracts_payment_obligations",
    "insurance",
    "taxes_duties",
    "penalties_late_fees",
    "assets_liabilities_investments",
    "maturity_renewals_deadlines",
    "refunds_credits_settlements",
    "non_financial",
]


@dataclass
class IndexedPassage:
    vector_id: int
    reference_id: str
    concept: str
    category: str
    label: str  # positive | negative
    text: str


@dataclass
class SimilarityMatch:
    reference_id: str
    concept: str
    category: str
    label: str
    similarity: float
    matched_text: str


class ReferenceVectorIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._faiss_index = None
        self._matrix: np.ndarray | None = None
        self._passages: list[IndexedPassage] = []
        self._backend: str | None = None
        self._version: str | None = None
        self._embedding_model: str | None = None
        self._dataset_version: str | None = None
        self._built_at: datetime | None = None
        self._stale: bool = True

    @property
    def vector_count(self) -> int:
        return len(self._passages)

    @property
    def backend(self) -> str | None:
        return self._backend

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @property
    def dataset_version(self) -> str | None:
        return self._dataset_version

    @property
    def built_at(self) -> datetime | None:
        return self._built_at

    @property
    def is_stale(self) -> bool:
        return self._stale or self._embedding_model != get_embedding_service().active_model_id()

    def mark_stale(self) -> None:
        self._stale = True

    def build_from_db(self, session: Session) -> dict[str, Any]:
        with self._lock:
            settings = get_settings()
            emb_svc = get_embedding_service()
            entries = list(
                session.scalars(select(ReferenceEntry).where(ReferenceEntry.is_active.is_(True)).order_by(ReferenceEntry.id))
            )
            active_dv = session.scalars(
                select(ReferenceDatasetVersion)
                .where(ReferenceDatasetVersion.is_active.is_(True))
                .order_by(ReferenceDatasetVersion.id.desc())
            ).first()
            dv_str = active_dv.version if active_dv else "1.0.0"

            passages: list[IndexedPassage] = []
            texts: list[str] = []
            entry_vec_ids: dict[str, list[int]] = {}

            for entry in entries:
                vids: list[int] = []
                # Index the concept + definition passage and each example passage
                head = f"{entry.concept}: {entry.definition}" if entry.definition else entry.concept
                vid = len(passages)
                passages.append(
                    IndexedPassage(
                        vector_id=vid,
                        reference_id=entry.id,
                        concept=entry.concept,
                        category=entry.category,
                        label=entry.label,
                        text=head,
                    )
                )
                texts.append(head)
                vids.append(vid)

                for ex in entry.examples or []:
                    ex_str = str(ex).strip()
                    if not ex_str:
                        continue
                    vid = len(passages)
                    passages.append(
                        IndexedPassage(
                            vector_id=vid,
                            reference_id=entry.id,
                            concept=entry.concept,
                            category=entry.category,
                            label=entry.label,
                            text=ex_str,
                        )
                    )
                    texts.append(ex_str)
                    vids.append(vid)
                entry_vec_ids[entry.id] = vids

            batch = emb_svc.embed_texts(texts)
            vecs = batch.vectors
            model_id = batch.model_id
            idx_hash = hashlib.sha256(
                f"{model_id}:{dv_str}:{len(passages)}:{','.join(e.content_hash[:8] for e in entries)}".encode("utf-8")
            ).hexdigest()[:12]
            idx_version = f"idx-{idx_hash}"

            backend_choice = settings.vector_backend
            faiss_idx = None
            if backend_choice in ("auto", "faiss") and vecs.shape[0] > 0:
                try:
                    import faiss  # type: ignore

                    faiss_idx = faiss.IndexFlatIP(int(vecs.shape[1]))
                    faiss_idx.add(vecs)
                    self._backend = "faiss"
                except Exception as exc:
                    log.warning("faiss_fallback_to_numpy", extra={"error": str(exc)})
                    self._backend = "numpy"
            else:
                self._backend = "numpy"

            self._faiss_index = faiss_idx
            self._matrix = vecs
            self._passages = passages
            self._version = idx_version
            self._embedding_model = model_id
            self._dataset_version = dv_str
            self._built_at = datetime.now(timezone.utc)
            self._stale = False

            for entry in entries:
                entry.embedding_model = model_id
                entry.embedding_version = idx_version
                entry.vector_ids = entry_vec_ids.get(entry.id, [])

            session.add(
                ModelVersion(
                    kind="reference_index",
                    name=model_id,
                    version=idx_version,
                    config={"dataset_version": dv_str, "backend": self._backend, "vectors": len(passages)},
                    is_active=True,
                )
            )
            session.flush()
            return {
                "index_version": idx_version,
                "embedding_model": model_id,
                "dataset_version": dv_str,
                "vectors": len(passages),
                "entries": len(entries),
                "computed": batch.computed,
                "cache_hits": batch.cache_hits,
                "backend": self._backend,
            }

    def search_batch(self, query_vectors: np.ndarray, top_k: int = 24) -> list[list[SimilarityMatch]]:
        with self._lock:
            if query_vectors.shape[0] == 0 or not self._passages or self._matrix is None:
                return [[] for _ in range(query_vectors.shape[0])]
            k = min(top_k, len(self._passages))
            q = query_vectors.astype(np.float32)
            if self._faiss_index is not None:
                scores_mat, idx_mat = self._faiss_index.search(q, k)
            else:
                sims = q @ self._matrix.T
                idx_mat = np.argsort(-sims, axis=1)[:, :k]
                scores_mat = np.take_along_axis(sims, idx_mat, axis=1)

            results: list[list[SimilarityMatch]] = []
            for row_scores, row_idxs in zip(scores_mat, idx_mat):
                matches: list[SimilarityMatch] = []
                for sc, idx in zip(row_scores, row_idxs):
                    if idx < 0 or idx >= len(self._passages):
                        continue
                    p = self._passages[int(idx)]
                    matches.append(
                        SimilarityMatch(
                            reference_id=p.reference_id,
                            concept=p.concept,
                            category=p.category,
                            label=p.label,
                            similarity=round(float(sc), 4),
                            matched_text=p.text,
                        )
                    )
                results.append(matches)
            return results


_index = ReferenceVectorIndex()


def get_reference_index() -> ReferenceVectorIndex:
    return _index


def _entry_hash(concept: str, definition: str | None, category: str, examples: list[str], label: str) -> str:
    payload = json.dumps(
        {"c": concept, "d": definition or "", "cat": category, "ex": examples, "l": label},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_reference_dataset_loaded(session: Session) -> None:
    """Seed the reference dataset from disk on startup if DB is empty, and build index."""
    settings = get_settings()
    count = session.query(ReferenceEntry).count()
    if count == 0 and settings.reference_dataset_path.exists():
        manifest: dict[str, Any] = {}
        if settings.reference_manifest_path.exists():
            try:
                manifest = json.loads(settings.reference_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        raw_bytes = settings.reference_dataset_path.read_bytes()
        import_reference_bytes(
            session,
            raw_bytes,
            filename=settings.reference_dataset_path.name,
            name=manifest.get("name", "FRIE Curated Reference Dataset"),
            version=manifest.get("version", "1.0.0"),
            source=manifest.get("source", "FRIE curated reference set v1"),
            license_name=manifest.get("license", "CC0-1.0"),
            attribution=manifest.get("attribution", "Original curated entries authored for FRIE."),
            source_url=manifest.get("source_url"),
            description=manifest.get("description"),
            rebuild=False,
        )
        session.commit()

    idx = get_reference_index()
    if idx.vector_count == 0 or idx.is_stale:
        idx.build_from_db(session)
        session.commit()


def import_reference_bytes(
    session: Session,
    data: bytes,
    *,
    filename: str,
    name: str,
    version: str,
    source: str,
    license_name: str,
    attribution: str | None = None,
    source_url: str | None = None,
    description: str | None = None,
    rebuild: bool = True,
) -> tuple[ReferenceDatasetVersion, int, int, int, list[str]]:
    file_hash = hashlib.sha256(data).hexdigest()
    dv = session.scalars(select(ReferenceDatasetVersion).where(ReferenceDatasetVersion.version == version)).first()
    if dv is None:
        dv = ReferenceDatasetVersion(
            version=version,
            name=name,
            description=description,
            source=source,
            license=license_name,
            attribution=attribution,
            source_url=source_url,
            file_hash=file_hash,
            entry_count=0,
            is_active=True,
        )
        session.add(dv)
        session.flush()

    text = data.decode("utf-8-sig", errors="replace")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "jsonl"
    raw_items: list[dict[str, Any]] = []
    errors: list[str] = []

    if ext == "csv":
        reader = csv.DictReader(io.StringIO(text))
        for i, row in enumerate(reader, start=2):
            ex_raw = row.get("examples") or ""
            examples = [e.strip() for e in ex_raw.split("||") if e.strip()] if "||" in ex_raw else [e.strip() for e in ex_raw.split(";") if e.strip()]
            raw_items.append(
                {
                    "reference_id": (row.get("reference_id") or "").strip() or None,
                    "concept": (row.get("concept") or "").strip(),
                    "definition": (row.get("definition") or "").strip() or None,
                    "category": (row.get("category") or "").strip(),
                    "examples": examples,
                    "label": (row.get("label") or "positive").strip().lower(),
                    "source": (row.get("source") or source).strip(),
                    "license": (row.get("license") or license_name).strip(),
                    "_line": i,
                }
            )
    elif ext == "json":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "entries" in parsed:
                parsed = parsed["entries"]
            if isinstance(parsed, list):
                raw_items = [dict(x, _line=i + 1) for i, x in enumerate(parsed) if isinstance(x, dict)]
            else:
                errors.append("JSON root must be an array or an object with 'entries'.")
        except Exception as exc:
            errors.append(f"Invalid JSON: {exc}")
    else:
        for i, line in enumerate(text.splitlines(), start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                obj["_line"] = i
                raw_items.append(obj)
            except Exception as exc:
                errors.append(f"Line {i}: invalid JSON ({exc})")

    imported = 0
    updated = 0
    skipped = 0

    for item in raw_items:
        line_no = item.get("_line", "?")
        concept = str(item.get("concept") or item.get("text") or "").strip()
        category = str(item.get("category") or "").strip()
        label = str(item.get("label") or "positive").strip().lower()
        if not concept or not category:
            errors.append(f"Item {line_no}: missing concept or category.")
            skipped += 1
            continue
        if label not in ("positive", "negative"):
            errors.append(f"Item {line_no}: label must be 'positive' or 'negative'.")
            skipped += 1
            continue
        examples = [str(x).strip() for x in (item.get("examples") or []) if str(x).strip()]
        definition = str(item.get("definition") or "").strip() or None
        ref_id = str(item.get("reference_id") or item.get("id") or "").strip()
        if not ref_id:
            ref_id = f"{category[:6]}-{hashlib.sha1(concept.encode('utf-8')).hexdigest()[:8]}"
        c_hash = _entry_hash(concept, definition, category, examples, label)
        item_source = str(item.get("source") or source).strip()
        item_license = str(item.get("license") or license_name).strip()

        existing = session.get(ReferenceEntry, ref_id)
        if existing is None:
            session.add(
                ReferenceEntry(
                    id=ref_id,
                    dataset_version_id=dv.id,
                    concept=concept,
                    definition=definition,
                    category=category,
                    examples=examples,
                    label=label,
                    source=item_source,
                    license=item_license,
                    content_hash=c_hash,
                    is_active=True,
                )
            )
            imported += 1
        elif existing.content_hash != c_hash or not existing.is_active:
            existing.dataset_version_id = dv.id
            existing.concept = concept
            existing.definition = definition
            existing.category = category
            existing.examples = examples
            existing.label = label
            existing.source = item_source
            existing.license = item_license
            existing.content_hash = c_hash
            existing.is_active = True
            updated += 1
        else:
            skipped += 1

    session.flush()
    dv.entry_count = (
        session.query(ReferenceEntry).filter(ReferenceEntry.dataset_version_id == dv.id, ReferenceEntry.is_active.is_(True)).count()
    )
    get_reference_index().mark_stale()
    if rebuild:
        get_reference_index().build_from_db(session)
    return dv, imported, updated, skipped, errors
