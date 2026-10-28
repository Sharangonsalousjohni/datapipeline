"""Compare the embedding/reference baseline with the lexical TF-IDF classifier.

Trains the lexical model on ``evaluation/datasets/chunks_validation.jsonl`` and scores both
classifiers on the held-out ``chunks_test.jsonl``. Also reports 5-fold cross-validation of the
lexical model over both files, since 120 test chunks give wide error bars.

Usage (from backend/):
    python scripts/compare_classifiers.py                # compare, print report
    python scripts/compare_classifiers.py --save         # ...and save the lexical model
    python scripts/compare_classifiers.py --no-baseline  # lexical model only (no embeddings)

UNCERTAIN chunks are not rejected (they go to review), so binary precision/recall count them
as relevant; ``uncertain_rate`` shows how much review load each classifier creates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
DATASETS = BACKEND / "evaluation" / "datasets"
RESULTS = BACKEND / "evaluation" / "results"


def load(name: str) -> list[dict[str, Any]]:
    path = DATASETS / f"chunks_{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([1 if r["label"] == "relevant" else 0 for r in rows])


def metrics(y: np.ndarray, decisions: list[str]) -> dict[str, Any]:
    """``decisions`` are RELEVANT / UNCERTAIN / IRRELEVANT."""
    kept = np.array([d != "IRRELEVANT" for d in decisions])
    confident = np.array([d == "RELEVANT" for d in decisions])
    tp = int((kept & (y == 1)).sum())
    fp = int((kept & (y == 0)).sum())
    fn = int((~kept & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": round(float((kept == (y == 1)).mean()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "false_rejections": fn,
        "false_accepts": fp,
        "uncertain_rate": round(float(np.mean([d == "UNCERTAIN" for d in decisions])), 4),
        "confident_relevant_precision": round(float(y[confident].mean()), 4) if confident.any() else None,
        "support": len(y),
    }


def by_difficulty(rows: list[dict[str, Any]], y: np.ndarray, decisions: list[str]) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get("difficulty", "standard")].append(i)
    return {
        g: round(float(np.mean([(decisions[i] != "IRRELEVANT") == (y[i] == 1) for i in idx])), 4)
        for g, idx in sorted(groups.items())
    }


def decide(probs: np.ndarray, t_rel: float, t_irrel: float) -> list[str]:
    return ["RELEVANT" if p >= t_rel else "IRRELEVANT" if p <= t_irrel else "UNCERTAIN" for p in probs]


# --------------------------------------------------------------------------- lexical
def run_lexical(val: list[dict], test: list[dict]) -> tuple[dict[str, Any], Any, tuple[float, float]]:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from app.lexical_classifier import build_pipeline, choose_thresholds

    x_val, y_val = [r["text"] for r in val], labels(val)
    x_test, y_test = [r["text"] for r in test], labels(test)

    # Thresholds come from out-of-fold probabilities on the training set only.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    oof = cross_val_predict(build_pipeline(), x_val, y_val, cv=skf, method="predict_proba")[:, 1]
    t_rel, t_irrel = choose_thresholds(y_val, oof)

    model = build_pipeline().fit(x_val, y_val)
    probs = model.predict_proba(x_test)[:, 1]
    dec = decide(probs, t_rel, t_irrel)
    errors = [
        {"id": r["id"], "label": r["label"], "difficulty": r.get("difficulty"), "prob": round(float(p), 3),
         "decision": d, "text": r["text"][:140]}
        for r, p, d in zip(test, probs, dec)
        if (d != "IRRELEVANT") != (r["label"] == "relevant")
    ]

    # 5-fold CV over all 240 chunks for a less noisy estimate (fixed 0.5 cut-off).
    all_rows = val + test
    x_all, y_all = [r["text"] for r in all_rows], labels(all_rows)
    cv_acc = []
    for seed in (13, 29, 47):
        cv_probs = cross_val_predict(
            build_pipeline(), x_all, y_all,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), method="predict_proba",
        )[:, 1]
        cv_acc.append(float(((cv_probs >= 0.5) == (y_all == 1)).mean()))

    result = {
        "thresholds": {"relevant": t_rel, "irrelevant": t_irrel, "source": "out-of-fold on validation"},
        "test": metrics(y_test, dec),
        "test_by_difficulty": by_difficulty(test, y_test, dec),
        "test_at_0.5": metrics(y_test, decide(probs, 0.5, 0.5)),
        "cv5_accuracy_all_240": {"mean": round(float(np.mean(cv_acc)), 4), "runs": [round(a, 4) for a in cv_acc]},
        "errors": errors,
    }
    return result, model, (t_rel, t_irrel)


# --------------------------------------------------------------------------- baseline
def run_baseline(test: list[dict]) -> dict[str, Any]:
    # Isolated data dir: never touch the real database or embedding cache.
    os.environ["FRIE_DATA_DIR"] = tempfile.mkdtemp(prefix="frie-compare-")
    os.environ["FRIE_CLASSIFIER_MODE"] = "baseline"
    os.environ["FRIE_LLM_ENABLED"] = "false"
    from app.config import get_settings

    get_settings.cache_clear()
    get_settings().ensure_dirs()

    from app.chunking import ChunkDraft
    from app.classifier import classify_chunks
    from app.db import get_engine, session_scope
    from app.models import Base
    from app.reference import ensure_reference_dataset_loaded

    Base.metadata.create_all(bind=get_engine())
    with session_scope() as session:
        ensure_reference_dataset_loaded(session)

    drafts = [
        ChunkDraft(
            id=r["id"], document_id="eval", chunk_index=i,
            chunk_type="table" if r.get("format_hint") in ("xlsx", "csv") else "prose",
            text=r["text"], context=None, location={}, extraction_method="eval",
            ocr_confidence=None, content_hash=r["id"],
        )
        for i, r in enumerate(test)
    ]
    out = classify_chunks(drafts)
    y = labels(test)
    dec = [c.decision for c in out.items]
    errors = [
        {"id": r["id"], "label": r["label"], "difficulty": r.get("difficulty"), "score": c.score,
         "decision": c.decision, "decided_by": c.decided_by, "text": r["text"][:140]}
        for r, c in zip(test, out.items)
        if (c.decision != "IRRELEVANT") != (r["label"] == "relevant")
    ]
    return {
        "embedding_model": out.embedding_model,
        "thresholds": out.items[0].thresholds if out.items else {},
        "test": metrics(y, dec),
        "test_by_difficulty": by_difficulty(test, y, dec),
        "errors": errors,
    }


# --------------------------------------------------------------------------- report
def print_report(report: dict[str, Any]) -> None:
    cols = ["accuracy", "precision", "recall", "f1", "false_rejections", "false_accepts", "uncertain_rate"]
    rows = [("lexical (tuned thresholds)", report["lexical"]["test"]), ("lexical (0.5 cut-off)", report["lexical"]["test_at_0.5"])]
    if "baseline" in report:
        rows.insert(0, ("embedding baseline", report["baseline"]["test"]))
    print(f"\nHeld-out test set: {report['lexical']['test']['support']} chunks\n")
    print(f"{'classifier':28}" + "".join(f"{c:>17}" for c in cols))
    for name, m in rows:
        print(f"{name:28}" + "".join(f"{m[c]:>17}" for c in cols))
    print("\naccuracy by difficulty:")
    for name in ("baseline", "lexical"):
        if name in report:
            print(f"  {name:10} {report[name]['test_by_difficulty']}")
    cv = report["lexical"]["cv5_accuracy_all_240"]
    print(f"\nlexical 5-fold CV accuracy over all 240 chunks: {cv['mean']} (runs {cv['runs']})")
    print(f"lexical thresholds: {report['lexical']['thresholds']}")
    for name in ("baseline", "lexical"):
        if name in report and report[name]["errors"]:
            print(f"\n{name} errors:")
            for e in report[name]["errors"]:
                print(f"  {e['id']} [{e['label']}/{e['difficulty']}] -> {e['decision']}: {e['text']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help="save the lexical model to evaluation/results/")
    ap.add_argument("--no-baseline", action="store_true", help="skip the embedding baseline")
    ap.add_argument("--json-out", type=Path, default=RESULTS / "classifier_comparison.json")
    args = ap.parse_args()

    val, test = load("validation"), load("test")
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not args.no_baseline:
        report["baseline"] = run_baseline(test)
    lexical, model, (t_rel, t_irrel) = run_lexical(val, test)
    report["lexical"] = lexical

    print_report(report)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {args.json_out}")

    if args.save:
        import joblib

        from app.lexical_classifier import LEXICAL_VERSION

        path = RESULTS / "lexical_classifier.joblib"
        joblib.dump(
            {
                "model": model,
                "version": LEXICAL_VERSION,
                "threshold_relevant": t_rel,
                "threshold_irrelevant": t_irrel,
                "trained_on": "chunks_validation.jsonl",
                "test_metrics": lexical["test"],
                "trained_at": report["generated_at"],
            },
            path,
        )
        print(f"lexical model saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
