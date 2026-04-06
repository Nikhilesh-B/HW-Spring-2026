"""
Aggregate metrics for hierarchical binary relevance: H1 summaries, pooled edge F1, leaf multilabel F1.

**Linear model presets** (sklearn stand-ins for Autext-style tools):

- **LinearSVC** — linear SVM on TF-IDF.
- **GLMNet** — elastic-net *logistic* regression (``penalty='elasticnet'``, ``solver='saga'``), common GLMNet analogue.
- **MaxEnt** — L2 *logistic* regression; binary max-entropy / multinomial max-ent are log-linear (logistic) models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError as e:
    raise ImportError("hierarchical_summary_metrics requires scikit-learn") from e

from hierarchical_classifier import BinaryEdgeFactory, MultilabelHierarchyRouter, binary_edge_factory
from hierarchical_evaluation import evaluate_binary_edges_from_parent
from hierarchical_training_data import MultilabelBinaryPoolData
from topic_hierarchy import TopicTree


def linear_model_factories(
    *,
    tfidf_kw: Optional[Dict] = None,
    max_features: int = 8000,
) -> Dict[str, BinaryEdgeFactory]:
    """
    Return named ``binary_edge_factory`` callables for side-by-side comparison.

    Shared TF-IDF: ``min_df=2``, ``max_features`` as given.
    """
    from sklearn.svm import LinearSVC

    tkw = dict(min_df=2, max_features=max_features)
    if tfidf_kw:
        tkw.update(tfidf_kw)

    return {
        "LinearSVC": binary_edge_factory(
            tfidf_kw=tkw,
            estimator=LinearSVC,
            clf_kw=dict(C=1.0, dual=False, max_iter=8000),
        ),
        # GLMNet (R) ~ elastic-net penalized logistic regression
        "GLMNet_elasticnet": binary_edge_factory(
            tfidf_kw=tkw,
            estimator=None,
            clf_kw=dict(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=4000,
                random_state=42,
            ),
        ),
        # Max-ent (linear) ~ logistic regression / log-linear model with L2
        "MaxEnt_L2": binary_edge_factory(
            tfidf_kw=tkw,
            estimator=None,
            clf_kw=dict(
                penalty="l2",
                solver="saga",
                C=1.0,
                max_iter=4000,
                random_state=42,
            ),
        ),
    }


def fit_parent_edges(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    parent: str,
    depth: int,
) -> None:
    """Fit every binary edge ``parent → child`` where training labels have two classes."""
    for child in router.tree.children.get(parent, []):
        Xtr, ytr = pool.binary_edge_dataset(parent, child, "train")
        if len(set(ytr)) < 2:
            continue
        router.fit_edge(parent, child, Xtr, ytr, depth=depth)


def fit_h1(router: MultilabelHierarchyRouter, pool: MultilabelBinaryPoolData) -> None:
    """Fit all ``Root → child`` edges (depth 0)."""
    fit_parent_edges(router, pool, router.tree.traversal_root, depth=0)


def _predict_labels(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    parent: str,
    child: str,
    split: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    m = router.edge_model(parent, child)
    if m is None:
        return None
    X, y_true = pool.binary_edge_dataset(parent, child, split)
    if not X:
        return None
    pipe = getattr(m, "pipeline", None)
    if pipe is not None:
        y_pred = np.asarray(pipe.predict(X), dtype=int)
    else:
        y_pred = np.asarray([m.predict_binary(x) for x in X], dtype=int)
    return np.asarray(y_true, dtype=int), y_pred


def pooled_edge_f1_stats(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    edges: Sequence[Tuple[str, str]],
    split: str,
) -> Dict[str, float]:
    """
    Pool predictions over listed edges (same docs repeated per edge).

    - **pooled_micro_f1**: one F1 on concatenated (y_true, y_pred) over all edge×doc pairs.
    - **macro_f1**: mean of per-edge F1 (edges with <2 classes in split skipped).
    - **pos_weighted_f1**: Σ_w f1_e with weights w = positive count on that edge in the split.
    """
    yt_all: List[int] = []
    yp_all: List[int] = []
    f1_list: List[float] = []
    pos_counts: List[int] = []

    for parent, child in edges:
        pr = _predict_labels(router, pool, parent, child, split)
        if pr is None:
            continue
        yt, yp = pr
        if len(np.unique(yt)) < 2:
            continue
        f1e = float(f1_score(yt, yp, zero_division=0))
        f1_list.append(f1e)
        pos_counts.append(int(np.sum(yt)))
        yt_all.extend(yt.tolist())
        yp_all.extend(yp.tolist())

    pooled = (
        float(f1_score(yt_all, yp_all, zero_division=0)) if yt_all else float("nan")
    )
    macro = float(np.mean(f1_list)) if f1_list else float("nan")
    sp = sum(pos_counts)
    pos_w = (
        float(np.dot(pos_counts, f1_list) / sp) if sp > 0 and f1_list else float("nan")
    )
    return {
        "pooled_micro_f1": pooled,
        "macro_f1": macro,
        "pos_weighted_f1": pos_w,
        "n_edges_used": float(len(f1_list)),
    }


def h1_edge_tuples(tree: TopicTree) -> List[Tuple[str, str]]:
    r = tree.traversal_root
    return [(r, c) for c in tree.children.get(r, [])]


def gold_path_branching_edges_for_leaves(
    tree: TopicTree,
    gold_leaf_codes: Set[str],
) -> Set[Tuple[str, str]]:
    """
    Union of all **branching** edges ``(parent, child)`` that lie on ``Root → L`` for each
    leaf ``L`` in ``gold_leaf_codes`` (edges where ``parent`` has ≥2 children).

    These are exactly the binary decisions that must be **positive** for inference to be
    able to reach that leaf (unary edges are automatic). If the model predicts **0** on such
    an edge, the walk **cuts** and never reaches deeper nodes — that hurts leaf recall and
    is counted in :func:`gold_path_branch_recall`.
    """
    out: Set[Tuple[str, str]] = set()
    for L in gold_leaf_codes:
        path = tree.path_from_root_to(L)
        for i in range(len(path) - 1):
            p, c = path[i], path[i + 1]
            if len(tree.children.get(p, [])) >= 2:
                out.add((p, c))
    return out


def gold_path_branch_recall(
    pool: MultilabelBinaryPoolData,
    router: MultilabelHierarchyRouter,
    tree: TopicTree,
    split: str,
    *,
    input_column: str = "article",
) -> Dict[str, float]:
    """
    **Path-to-gold recall (branching edges only).**

    For each test article, take gold **leaf** labels. For every branching edge on the union
    of root-to-leaf paths, check whether that edge’s binary predicts **1**. A **0** (or a
    missing fitted model on that edge) is a failure to “open” that branch — i.e. the early
    **cut** you described — and is counted here.

    This is complementary to :func:`leaf_multilabel_f1`: leaf F1 compares predicted vs gold
    **leaf sets**; path recall directly measures “did we fire every required decision on the
    way to each gold leaf?”.

    Returns ``path_gold_branch_recall`` in ``[0, 1]`` and counts for inspection.
    """
    leaves = tree.leaf_nodes()
    ids = pool.train_ids() if split == "train" else pool.test_ids()
    art = pool.articles_df().set_index("id")

    n_req = 0
    n_ok = 0

    for aid in ids:
        if aid not in art.index:
            continue
        row = art.loc[aid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        text = row[input_column]
        gold_leaves = pool.article_labels.get(aid, set()) & leaves
        if not gold_leaves:
            continue
        req = gold_path_branching_edges_for_leaves(tree, gold_leaves)
        for parent, child in req:
            n_req += 1
            m = router.edge_model(parent, child)
            if m is None:
                continue
            if int(m.predict_binary(text)) == 1:
                n_ok += 1

    recall = float(n_ok / n_req) if n_req > 0 else float("nan")
    return {
        "path_gold_branch_recall": recall,
        "n_path_gold_branches": float(n_req),
        "n_path_gold_branches_correct": float(n_ok),
    }


def leaf_level_evaluation(
    pool: MultilabelBinaryPoolData,
    router: MultilabelHierarchyRouter,
    tree: TopicTree,
    split: str,
    *,
    input_column: str = "article",
) -> Dict[str, float]:
    """Leaf multilabel F1 and path-to-gold branching recall."""
    out: Dict[str, float] = dict(
        leaf_multilabel_f1(pool, router, tree, split, input_column=input_column)
    )
    out.update(
        gold_path_branch_recall(
            pool, router, tree, split, input_column=input_column
        )
    )
    return out


def leaf_multilabel_f1(
    pool: MultilabelBinaryPoolData,
    router: MultilabelHierarchyRouter,
    tree: TopicTree,
    split: str,
    *,
    input_column: str = "article",
) -> Dict[str, float]:
    """
    **Leaf multilabel F1:** gold leaf set (``news_topics`` ∩ leaves) vs ``predict_reached_leaves``.

    Stopping high in the tree (no positive on a branching edge) usually removes deeper
    reached nodes, so predicted **leaf** sets shrink → **recall** drops on gold leaves.
    Extra predicted leaves hurt **precision**. Uses indicator matrix over all taxonomy leaves;
    metrics are sklearn multilabel F1 variants (micro / macro / weighted / samples).
    """
    leaves = sorted(tree.leaf_nodes())
    if not leaves:
        return {}

    mlb = MultiLabelBinarizer(classes=leaves)
    mlb.fit([[]])

    ids = pool.train_ids() if split == "train" else pool.test_ids()
    art = pool.articles_df().set_index("id")
    leaf_set = set(leaves)

    gold_rows: List[Tuple[str, ...]] = []
    pred_rows: List[Tuple[str, ...]] = []

    for aid in ids:
        if aid not in art.index:
            continue
        row = art.loc[aid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        text = row[input_column]
        gold = (pool.article_labels.get(aid, set()) & leaf_set)
        pred = router.predict_reached_leaves(text)
        gold_rows.append(tuple(sorted(gold)))
        pred_rows.append(tuple(sorted(pred)))

    if not gold_rows:
        return {}

    Y_true = mlb.transform(gold_rows)
    Y_pred = mlb.transform(pred_rows)

    return {
        "leaf_micro_f1": float(
            f1_score(Y_true, Y_pred, average="micro", zero_division=0)
        ),
        "leaf_macro_f1": float(
            f1_score(Y_true, Y_pred, average="macro", zero_division=0)
        ),
        "leaf_weighted_f1": float(
            f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
        ),
        "leaf_samples_f1": float(
            f1_score(Y_true, Y_pred, average="samples", zero_division=0)
        ),
    }


def summary_row(
    name: str,
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    tree: TopicTree,
    split: str = "test",
    *,
    include_leaf: bool = False,
) -> Dict[str, Any]:
    """
    One row of summary metrics for **H1 (Root)**: macro / pooled-micro / pos-weighted F1.

    Set ``include_leaf=True`` to also add :func:`leaf_level_evaluation` (multilabel leaf F1
    + path-to-gold branching recall; slower than H1-only).
    """
    root = tree.traversal_root
    h1_edges = h1_edge_tuples(tree)
    per = evaluate_binary_edges_from_parent(router, pool, root, split=split)
    h1_f1s = [per[ch]["f1"] for ch in per if "f1" in per[ch]]
    h1_macro = float(np.mean(h1_f1s)) if h1_f1s else float("nan")

    pool_h1 = pooled_edge_f1_stats(router, pool, h1_edges, split)

    row: Dict[str, Any] = {
        "model": name,
        "h1_macro_f1": h1_macro,
        "h1_pooled_micro_f1": pool_h1["pooled_micro_f1"],
        "h1_pos_weighted_f1": pool_h1["pos_weighted_f1"],
    }
    if include_leaf:
        row.update(leaf_level_evaluation(pool, router, tree, split))
    return row


def comparison_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a summary table from rows produced by :func:`summary_row`."""
    return pd.DataFrame(rows)


def train_and_summarize(
    model_name: str,
    tree: TopicTree,
    pool: MultilabelBinaryPoolData,
    factory: BinaryEdgeFactory,
    split: str = "test",
    *,
    include_leaf: bool = False,
) -> Tuple[MultilabelHierarchyRouter, Dict[str, Any]]:
    """Fresh router: fit H1 only, return (router, summary dict for split)."""
    router = MultilabelHierarchyRouter(tree, factory)
    fit_h1(router, pool)
    summ = summary_row(
        model_name, router, pool, tree, split=split, include_leaf=include_leaf
    )
    return router, summ
