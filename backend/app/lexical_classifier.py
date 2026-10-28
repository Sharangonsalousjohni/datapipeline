"""Lexical relevance classifier: TF-IDF word + character n-grams with logistic regression.

A supervised alternative to the embedding/reference-similarity baseline. Financial
relevance is carried mostly by vocabulary ("rent", "invoice", "EBITDA") and number
shapes ("$0,000.00", "(0,000)", "0.0%"), which sparse n-gram features capture directly.
Digits are mapped to ``0`` so the model learns number *formats*, not specific values.

Trained and evaluated by ``scripts/compare_classifiers.py``; loaded at runtime when
``classifier_mode == "lexical"``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from app.config import get_settings

log = logging.getLogger(__name__)

LEXICAL_VERSION = "lexical-tfidf-lr-v1"

_DIGIT_RE = re.compile(r"\d")


def normalize_text(text: str) -> str:
    return _DIGIT_RE.sub("0", text.lower())


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    token_pattern=r"(?u)\b\w[\w&'.-]*\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )
    clf = CalibratedClassifierCV(
        LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced"),
        method="sigmoid",
        cv=5,
    )
    return Pipeline([("features", features), ("clf", clf)])


def choose_thresholds(
    y_true: np.ndarray, probs: np.ndarray, max_false_rejection: float = 0.02, min_precision: float = 0.95
) -> tuple[float, float]:
    """Pick (relevant, irrelevant) thresholds from out-of-fold probabilities.

    ``irrelevant`` is the highest cut-off that rejects at most ``max_false_rejection`` of
    the relevant chunks (recall first: a wrongly rejected chunk is never reviewed).
    ``relevant`` is the lowest cut-off at which RELEVANT decisions reach ``min_precision``.
    Scores in between are UNCERTAIN and go to review.
    """
    pos = np.sort(probs[y_true == 1])
    allowed = int(np.floor(max_false_rejection * len(pos)))
    t_irrel = float(pos[allowed]) - 1e-6 if len(pos) else 0.35
    t_rel = 1.0
    for t in np.unique(probs)[::-1]:
        picked = probs >= t
        if picked.sum() and y_true[picked].mean() >= min_precision:
            t_rel = float(t)
        else:
            break
    t_irrel = round(max(0.0, min(t_irrel, t_rel)), 4)
    return round(max(t_rel, t_irrel), 4), t_irrel


class LexicalClassifierBundle:
    """Lazily loads the joblib bundle written by ``scripts/compare_classifiers.py --save``."""

    def __init__(self) -> None:
        self._mtime: float | None = None
        self._payload: dict[str, Any] | None = None

    def load(self) -> dict[str, Any] | None:
        path = get_settings().lexical_classifier_path
        if not path.exists():
            return None
        try:
            mtime = path.stat().st_mtime
            if self._payload is None or self._mtime != mtime:
                self._payload = joblib.load(path)
                self._mtime = mtime
            return self._payload
        except Exception as exc:
            log.warning("lexical_classifier_load_failed", extra={"error": str(exc)})
            return None


lexical_bundle = LexicalClassifierBundle()
