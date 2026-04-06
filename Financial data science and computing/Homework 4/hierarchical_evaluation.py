"""
Metrics and k-fold CV for binary edge classifiers (hierarchical multi-label).

**Test-time metrics (after you have a fitted model):**

- Call :func:`evaluate_binary` with ``X_test``, ``y_test`` — returns one dict with
  ``accuracy``, ``precision``, ``recall``, ``f1``, and usually ``roc_auc``, ``avg_precision``.
- Or use :func:`metrics_binary` if you already have predictions.

**Train + test in one step:** :func:`fit_eval_train_test_binary` returns
  ``{"train": {...}, "test": {...}}`` with the same keys on each split.

**One classifier:** :func:`evaluate_binary_edge` takes ``parent`` and ``child`` (one edge) and
  returns that binary’s metrics. :func:`evaluate_binary_edges_from_parent` returns metrics for
  every outgoing edge from a branching ``parent``.

**Every binary in the tree:** :func:`evaluate_all_binary_edges` /
  :func:`evaluate_all_binary_edges_from_pool` — ``{(parent, child): metrics_dict, ...}``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# #region agent log
_AGENT_DEBUG_LOG = (
    "/Users/nikhileshbelulkar/Documents/HW-Spring-2026/Financial data science and computing/"
    ".cursor/debug-76675e.log"
)


def _agent_dbg(
    location: str,
    message: str,
    data: Dict[str, Any],
    *,
    hypothesis_id: str,
) -> None:
    try:
        payload = {
            "sessionId": "76675e",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# #endregion

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
except ImportError as e:
    raise ImportError("hierarchical_evaluation requires scikit-learn") from e

from hierarchical_classifier import BinaryEdgeModel, MultilabelHierarchyRouter
from hierarchical_training_data import MultilabelBinaryPoolData
from topic_hierarchy import BinaryEdgeSpec, binary_edge_specs

# Keys you can rely on for binary edge evaluation (test or train split):
# accuracy, precision, recall, f1 — always present from metrics_binary.
# roc_auc, avg_precision — present when scores are available and both classes exist.
BINARY_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "avg_precision",
)


def metrics_binary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Precision, recall, F1, accuracy; ROC-AUC / average precision if ``y_score`` given."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }
    if y_score is not None and len(np.unique(yt)) > 1:
        ys = np.asarray(y_score, dtype=float)
        try:
            out["roc_auc"] = float(roc_auc_score(yt, ys))
        except ValueError:
            out["roc_auc"] = float("nan")
        try:
            out["avg_precision"] = float(average_precision_score(yt, ys))
        except ValueError:
            out["avg_precision"] = float("nan")
    return out


def evaluate_binary(
    model: BinaryEdgeModel,
    X: Sequence[Any],
    y_true: Sequence[int],
) -> Dict[str, float]:
    """
    **Testing-time helper:** run a **fitted** ``model`` on inputs ``X`` and return all
    standard metrics for gold labels ``y_true`` (same length as ``X``).

    Returns the same structure as :func:`metrics_binary`: ``accuracy``, ``precision``,
    ``recall``, ``f1``, and when applicable ``roc_auc`` and ``avg_precision`` (if the model
    implements ``decision_function`` or ``predict_proba`` via the wrapped pipeline).
    """
    Xl = list(X)
    n = len(Xl)
    t0 = time.perf_counter()
    # #region agent log
    _agent_dbg(
        "hierarchical_evaluation.py:evaluate_binary",
        "start",
        {"n": n, "has_pipeline": hasattr(model, "pipeline"), "runId": "post-fix"},
        hypothesis_id="A",
    )
    # #endregion
    # Batch path: per-sample loops run TF-IDF once per document (very slow on train).
    pipe = getattr(model, "pipeline", None)
    if pipe is not None:
        y_pred_arr = np.asarray(pipe.predict(Xl), dtype=int)
        y_pred = y_pred_arr.tolist()
        if hasattr(pipe, "decision_function"):
            y_score = np.asarray(pipe.decision_function(Xl), dtype=float).ravel().tolist()
        else:
            proba = np.asarray(pipe.predict_proba(Xl), dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_score = (proba[:, 1] - 0.5).tolist()
            else:
                y_score = (proba.ravel() - 0.5).tolist()
        t1 = time.perf_counter()
        # #region agent log
        _agent_dbg(
            "hierarchical_evaluation.py:evaluate_binary",
            "timings_sec",
            {
                "n": n,
                "batch_path": True,
                "predict_plus_scores": round(t1 - t0, 4),
                "total": round(t1 - t0, 4),
            },
            hypothesis_id="A",
        )
        # #endregion
        return metrics_binary(y_true, y_pred, y_score)

    y_pred = [model.predict_binary(x) for x in Xl]
    t1 = time.perf_counter()
    y_score = _scores_for_model(model, Xl)
    t2 = time.perf_counter()
    # #region agent log
    _agent_dbg(
        "hierarchical_evaluation.py:evaluate_binary",
        "timings_sec",
        {
            "n": n,
            "batch_path": False,
            "predict_loop": round(t1 - t0, 4),
            "scores_loop": round(t2 - t1, 4),
            "total": round(t2 - t0, 4),
        },
        hypothesis_id="A",
    )
    # #endregion
    return metrics_binary(y_true, y_pred, y_score)


def evaluate_binary_edge(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    parent: str,
    child: str,
    split: str = "test",
    *,
    input_column: str = "article",
) -> Optional[Dict[str, float]]:
    """
    Metrics for **one** binary classifier: the edge ``parent -> child``.

    In :class:`~hierarchical_classifier.MultilabelHierarchyRouter`, a **branching** parent has
    **one binary per child** (binary relevance), not a single softmax over children. So the
    natural handle for “this classifier” is the pair ``(parent, child)``, not the parent label
    alone. If you want all classifiers at one routing step, use
    :func:`evaluate_binary_edges_from_parent`.

    Returns ``None`` if there is no fitted model for that edge or no rows in ``pool`` for the
    split (empty ``X``).
    """
    m = router.edge_model(parent, child)
    if m is None:
        return None
    X, y_true = pool.binary_edge_dataset(
        parent, child, split, input_column=input_column
    )
    if not X:
        return None
    # #region agent log
    _agent_dbg(
        "hierarchical_evaluation.py:evaluate_binary_edge",
        "edge_eval",
        {
            "parent": parent,
            "child": child,
            "split": split,
            "n": len(X),
        },
        hypothesis_id="B",
    )
    # #endregion
    return evaluate_binary(m, X, y_true)


def evaluate_binary_edges_from_parent(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    parent: str,
    split: str = "test",
    *,
    input_column: str = "article",
) -> Dict[str, Dict[str, float]]:
    """
    Metrics for **each** binary classifier on outgoing edges from ``parent`` (same ``split``).

    Returns ``child_topic_code -> metrics_dict``. Only includes children where a model is
    fitted and data is non-empty. Parents with a single child have no binary models in this
    setup, so the result is typically empty.
    """
    out: Dict[str, Dict[str, float]] = {}
    for c in router.tree.children.get(parent, []):
        met = evaluate_binary_edge(
            router, pool, parent, c, split, input_column=input_column
        )
        if met is not None:
            out[c] = met
    return out


def evaluate_all_binary_edges(
    router: MultilabelHierarchyRouter,
    get_xy: Callable[[BinaryEdgeSpec], Tuple[Sequence[Any], Sequence[int]]],
    *,
    edges: Optional[Sequence[BinaryEdgeSpec]] = None,
    skip_missing_model: bool = True,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Run :func:`evaluate_binary` for **each** binary edge classifier that exists on ``router``.

    **Which edges:** ``edges`` if given, else all specs from :func:`binary_edge_specs` applied to
    ``router.tree`` (every parent with ≥2 children, one classifier per ``(parent, child)``).

    **Data:** ``get_xy(spec)`` must return ``(X, y)`` for that edge (same split you care about,
    e.g. test). Typically implemented via
    :meth:`MultilabelBinaryPoolData.binary_edge_dataset` — see
    :func:`evaluate_all_binary_edges_from_pool`.

    Skips an edge when there is no fitted model (if ``skip_missing_model``) or when ``X`` is empty.
    Returns a map ``(parent, child) ->`` the same metric dict as :func:`evaluate_binary`.
    """
    specs = list(edges) if edges is not None else binary_edge_specs(router.tree)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for spec in specs:
        k = spec.key
        m = router.edge_model(spec.parent, spec.child)
        if m is None:
            if skip_missing_model:
                continue
            raise KeyError(f"No fitted model for edge {k!r}")
        X, y_true = get_xy(spec)
        if not X:
            continue
        out[k] = evaluate_binary(m, X, y_true)
    return out


def evaluate_all_binary_edges_from_pool(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    split: str = "test",
    *,
    edges: Optional[Sequence[BinaryEdgeSpec]] = None,
    input_column: str = "article",
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Convenience: same as :func:`evaluate_all_binary_edges`, with ``(X, y)`` taken from
    ``pool.binary_edge_dataset(parent, child, split, ...)`` for each edge.
    """
    def _get_xy(spec: BinaryEdgeSpec) -> Tuple[Sequence[Any], Sequence[int]]:
        return pool.binary_edge_dataset(
            spec.parent, spec.child, split, input_column=input_column
        )

    return evaluate_all_binary_edges(router, _get_xy, edges=edges)


def _scores_for_model(model: BinaryEdgeModel, X: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for x in X:
        if hasattr(model, "decision_function"):
            out.append(float(model.decision_function(x)))  # type: ignore[attr-defined]
        else:
            out.append(float(model.predict_binary(x)))
    return out


def kfold_binary_edge(
    X: Sequence[Any],
    y: Sequence[int],
    model_factory: Callable[[], BinaryEdgeModel],
    *,
    k: int = 5,
    random_state: int = 42,
    report_train: bool = True,
) -> Dict[str, Any]:
    """
    Stratified k-fold on (X, y). Each fold fits a fresh model from ``model_factory``.

    Returns mean/std for validation metrics; optional mean train metrics.
    """
    Xl = list(X)
    yl = list(y)
    n = len(Xl)
    if n == 0:
        return {"n": 0, "val": {}, "train": {}}
    y_arr = np.asarray(yl, dtype=int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    val_metrics: List[Dict[str, float]] = []
    train_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in skf.split(np.zeros(n), y_arr):
        m = model_factory()
        tr_x = [Xl[i] for i in train_idx]
        tr_y = [yl[i] for i in train_idx]
        va_x = [Xl[i] for i in val_idx]
        va_y = [yl[i] for i in val_idx]
        m.fit(tr_x, tr_y)
        tr_pred = [m.predict_binary(x) for x in tr_x]
        tr_sc = _scores_for_model(m, tr_x)
        va_pred = [m.predict_binary(x) for x in va_x]
        va_sc = _scores_for_model(m, va_x)
        if report_train:
            train_metrics.append(metrics_binary(tr_y, tr_pred, tr_sc))
        val_metrics.append(metrics_binary(va_y, va_pred, va_sc))

    def _mean_std(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        if not rows:
            return {}
        keys = rows[0].keys()
        out: Dict[str, Tuple[float, float]] = {}
        for key in keys:
            vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
            if not vals:
                continue
            out[key] = (float(np.mean(vals)), float(np.std(vals)))
        return out

    return {
        "n": n,
        "k": k,
        "val_mean_std": _mean_std(val_metrics),
        "train_mean_std": _mean_std(train_metrics) if report_train else {},
    }


def fit_eval_train_test_binary(
    X_train: Sequence[Any],
    y_train: Sequence[int],
    X_test: Sequence[Any],
    y_test: Sequence[int],
    model: BinaryEdgeModel,
) -> Dict[str, Dict[str, float]]:
    """
    Fit on train; return metrics on train and test.

    Each of ``result["train"]`` and ``result["test"]`` is a dict with at least
    ``accuracy``, ``precision``, ``recall``, ``f1``; ROC-style keys are added when scores
    exist (same as :func:`evaluate_binary`).
    """
    model.fit(list(X_train), list(y_train))
    return {
        "train": evaluate_binary(model, X_train, y_train),
        "test": evaluate_binary(model, X_test, y_test),
    }
