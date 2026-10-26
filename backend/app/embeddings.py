"""Local embedding service with persistent SQLite caching and batching.

Supports:
- ``sentence-transformers`` (default: ``sentence-transformers/all-MiniLM-L6-v2``)
- ``hashing`` (fast deterministic word/char n-gram projection, offline fallback)

Embeddings are L2-normalized float32 vectors and cached by ``(content_hash, model_id)``
so unchanged reference entries and chunks are never re-embedded unnecessarily.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from dataclasses import dataclass

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class EmbeddingBatchResult:
    vectors: np.ndarray  # shape (N, D), float32, L2-normalized
    model_id: str
    computed: int
    cache_hits: int


class _EmbeddingCache:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self._path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL,
                    PRIMARY KEY (text_hash, model_id)
                )
                """
            )

    def get_many(self, hashes: list[str], model_id: str) -> dict[str, np.ndarray]:
        if not hashes:
            return {}
        unique = list(set(hashes))
        found: dict[str, np.ndarray] = {}
        with self._lock, sqlite3.connect(self._path, timeout=30) as conn:
            for start in range(0, len(unique), 400):
                batch = unique[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT text_hash, dim, vec FROM embedding_cache WHERE model_id = ? AND text_hash IN ({placeholders})",
                    [model_id, *batch],
                ).fetchall()
                for th, dim, blob in rows:
                    arr = np.frombuffer(blob, dtype=np.float32).copy()
                    if arr.size == dim:
                        found[th] = arr
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]], model_id: str) -> None:
        if not items:
            return
        with self._lock, sqlite3.connect(self._path, timeout=30) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, model_id, dim, vec) VALUES (?, ?, ?, ?)",
                [(th, model_id, int(vec.shape[0]), vec.astype(np.float32).tobytes()) for th, vec in items],
            )


class _HashingEncoder:
    """Deterministic local feature hashing encoder (384-d) for offline/test environments."""

    dim = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer

        vec_word = HashingVectorizer(
            n_features=256, analyzer="word", ngram_range=(1, 2), norm=None, alternate_sign=True
        )
        vec_char = HashingVectorizer(
            n_features=128, analyzer="char_wb", ngram_range=(3, 5), norm=None, alternate_sign=True
        )
        m_w = vec_word.transform(texts).toarray().astype(np.float32)
        m_c = vec_char.transform(texts).toarray().astype(np.float32) * 0.5
        mat = np.hstack([m_w, m_c])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


class EmbeddingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._st_model = None
        self._st_model_name: str | None = None
        self._fallback_to_hashing = False
        self._cache: _EmbeddingCache | None = None

    def _get_cache(self) -> _EmbeddingCache:
        settings = get_settings()
        settings.ensure_dirs()
        p = str(settings.embedding_cache_path)
        if self._cache is None or self._cache._path != p:
            self._cache = _EmbeddingCache(p)
        return self._cache

    def _load_model(self):
        settings = get_settings()
        if settings.embedding_provider == "hashing" or self._fallback_to_hashing:
            return _HashingEncoder(), "hashing-v1"
        with self._lock:
            if self._st_model is not None and self._st_model_name == settings.embedding_model:
                return self._st_model, settings.embedding_model
            try:
                from sentence_transformers import SentenceTransformer

                log.info("loading_sentence_transformer", extra={"model": settings.embedding_model})
                self._st_model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device,
                    local_files_only=settings.hf_offline,
                )
                self._st_model_name = settings.embedding_model
                return self._st_model, settings.embedding_model
            except Exception as exc:
                log.warning(
                    "sentence_transformer_fallback_to_hashing",
                    extra={"model": settings.embedding_model, "error": str(exc)},
                )
                self._fallback_to_hashing = True
                return _HashingEncoder(), "hashing-v1"

    @property
    def is_loaded(self) -> bool:
        return self._st_model is not None or get_settings().embedding_provider == "hashing" or self._fallback_to_hashing

    def active_model_id(self) -> str:
        settings = get_settings()
        if settings.embedding_provider == "hashing" or self._fallback_to_hashing:
            return "hashing-v1"
        return settings.embedding_model

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        settings = get_settings()
        encoder, model_id = self._load_model()
        if not texts:
            dim = getattr(encoder, "dim", 384)
            return EmbeddingBatchResult(np.zeros((0, dim), dtype=np.float32), model_id, 0, 0)

        trimmed = [t[: settings.embedding_max_chars].strip() or " " for t in texts]
        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in trimmed]
        cache = self._get_cache()
        cached = cache.get_many(hashes, model_id)

        missing_indices: list[int] = []
        missing_texts: list[str] = []
        missing_hashes: list[str] = []
        seen_missing: dict[str, int] = {}

        for i, (t, h) in enumerate(zip(trimmed, hashes)):
            if h not in cached and h not in seen_missing:
                seen_missing[h] = len(missing_texts)
                missing_indices.append(i)
                missing_texts.append(t)
                missing_hashes.append(h)

        computed_map: dict[str, np.ndarray] = {}
        if missing_texts:
            if isinstance(encoder, _HashingEncoder):
                raw = encoder.encode(missing_texts)
            else:
                raw = encoder.encode(
                    missing_texts,
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            raw = (raw / norms).astype(np.float32)
            to_store: list[tuple[str, np.ndarray]] = []
            for h, vec in zip(missing_hashes, raw):
                computed_map[h] = vec
                to_store.append((h, vec))
            cache.put_many(to_store, model_id)

        all_vecs = [cached[h] if h in cached else computed_map[h] for h in hashes]
        mat = np.vstack(all_vecs).astype(np.float32)
        cache_hits = sum(1 for h in hashes if h in cached)
        computed = len(hashes) - cache_hits
        return EmbeddingBatchResult(
            vectors=mat,
            model_id=model_id,
            computed=computed,
            cache_hits=cache_hits,
        )


_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    return _service
