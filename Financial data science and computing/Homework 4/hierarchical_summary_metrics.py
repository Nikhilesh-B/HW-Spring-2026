"""
Aggregate metrics for hierarchical binary relevance: H1 summaries, full-tree training,
pooled edge F1, leaf multilabel F1.

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
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError as e:
    raise ImportError("hierarchical_summary_metrics requires scikit-learn") from e

from hierarchical_classifier import BinaryEdgeFactory, MultilabelHierarchyRouter, binary_edge_factory
from hierarchical_evaluation import evaluate_binary_edges_from_parent
from hierarchical_training_data import MultilabelBinaryPoolData
from topic_hierarchy import BinaryEdgeSpec, TopicTree, binary_edge_specs


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
    *,
    restrict_to_parent_subtree: bool = True,
) -> None:
    """Fit every binary edge ``parent → child`` where training labels have two classes."""
    for child in router.tree.children.get(parent, []):
        Xtr, ytr = pool.binary_edge_dataset(
            parent, child, "train", restrict_to_parent_subtree=restrict_to_parent_subtree
        )
        if len(set(ytr)) < 2:
            continue
        router.fit_edge(parent, child, Xtr, ytr, depth=depth)


def fit_h1(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    *,
    restrict_to_parent_subtree: bool = True,
) -> None:
    """Fit all ``Root → child`` edges (depth 0)."""
    fit_parent_edges(
        router,
        pool,
        router.tree.traversal_root,
        depth=0,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )


def fit_all_binary_edges(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    *,
    verbose: bool = True,
    restrict_to_parent_subtree: bool = True,
) -> Dict[str, int]:
    """
    Fit **every** binary edge in the taxonomy (all ``binary_edge_specs``): one TF-IDF + linear
    model per ``(parent, child)`` at branching nodes.

    Skips edges where the **training** pool has a single class. When ``verbose`` is True,
    prints progress ``[i/N] parent → child (depth d)`` and fit/skip lines so long runs are
    observable.

    ``restrict_to_parent_subtree``: if True (default), each edge trains only on articles whose
    gold intersects ``subtree(parent)``; if False, uses the legacy global pool (all split rows).
    """
    specs = binary_edge_specs(router.tree)
    n_fit = 0
    n_skip = 0
    for i, spec in enumerate(specs, start=1):
        if verbose:
            print(
                f"[{i}/{len(specs)}] {spec.parent} → {spec.child}  (depth {spec.depth})",
                flush=True,
            )
        Xtr, ytr = pool.binary_edge_dataset(
            spec.parent,
            spec.child,
            "train",
            restrict_to_parent_subtree=restrict_to_parent_subtree,
        )
        if len(set(ytr)) < 2:
            n_skip += 1
            if verbose:
                print(
                    f"    skip: need 2 classes in train (n={len(ytr)})",
                    flush=True,
                )
            continue
        router.fit_edge(spec.parent, spec.child, Xtr, ytr, depth=spec.depth)
        n_fit += 1
        if verbose:
            print(
                f"    fit: n={len(ytr)}  positives={int(sum(ytr))}",
                flush=True,
            )
    if verbose:
        print(
            f"Done fitting: {n_fit} edges trained, {n_skip} skipped (single class), "
            f"{len(specs)} total specs.",
            flush=True,
        )
    return {
        "n_specs": len(specs),
        "n_fitted": n_fit,
        "n_skipped_single_class_train": n_skip,
    }


def fit_binary_edges_subset(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    specs: Sequence[BinaryEdgeSpec],
    *,
    verbose: bool = True,
    restrict_to_parent_subtree: bool = True,
) -> Dict[str, int]:
    """
    Fit **only** the listed ``BinaryEdgeSpec`` edges: same TF-IDF + classifier behavior as
    :func:`fit_all_binary_edges`, but restricted to ``specs``.

    Skips an edge when the **training** pool has a single class for that edge.
    """
    n_fit = 0
    n_skip = 0
    for i, spec in enumerate(specs, start=1):
        if verbose:
            print(
                f"[{i}/{len(specs)}] {spec.parent} → {spec.child}  (depth {spec.depth})",
                flush=True,
            )
        Xtr, ytr = pool.binary_edge_dataset(
            spec.parent,
            spec.child,
            "train",
            restrict_to_parent_subtree=restrict_to_parent_subtree,
        )
        if len(set(ytr)) < 2:
            n_skip += 1
            if verbose:
                print(
                    f"    skip: need 2 classes in train (n={len(ytr)})",
                    flush=True,
                )
            continue
        router.fit_edge(spec.parent, spec.child, Xtr, ytr, depth=spec.depth)
        n_fit += 1
        if verbose:
            print(
                f"    fit: n={len(ytr)}  positives={int(sum(ytr))}",
                flush=True,
            )
    if verbose:
        print(
            f"Done fitting subset: {n_fit} edges trained, {n_skip} skipped (single class), "
            f"{len(specs)} specs in list.",
            flush=True,
        )
    return {
        "n_specs": len(specs),
        "n_fitted": n_fit,
        "n_skipped_single_class_train": n_skip,
    }


def full_tree_test_metrics(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    tree: TopicTree,
    split: str = "test",
    *,
    restrict_to_parent_subtree: bool = True,
) -> Dict[str, Any]:
    """
    **Whole taxonomy (all branching edges)** on ``split``: same aggregates as
    :func:`pooled_edge_f1_stats` — pooled / macro / positive-weighted **F1**, **precision**,
    **recall**, **accuracy** over all edge×document pairs; **pooled** 2×2 **confusion matrix**
    and **per-edge** confusion matrices (see keys below).

    Also runs :func:`leaf_level_evaluation` (leaf multilabel F1 + path-to-gold recall).

    **Scalar keys** (``ft_*``): edge aggregates. **Non-scalar:** ``ft_confusion_matrix_pooled``
    (``ndarray`` shape (2, 2), rows=true class, cols=predicted, ``labels=[0,1]``), and
    ``ft_per_edge_confusion`` (``dict`` mapping ``(parent, child) -> ndarray(2, 2)``).

    ``restrict_to_parent_subtree`` must match how edges were trained (see
    :func:`fit_all_binary_edges`).
    """
    edges = [(s.parent, s.child) for s in binary_edge_specs(tree)]
    edge_stats = pooled_edge_f1_stats(
        router,
        pool,
        edges,
        split,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )
    leaf_stats = leaf_level_evaluation(pool, router, tree, split)
    out: Dict[str, Any] = {
        "ft_pooled_micro_f1": edge_stats["pooled_micro_f1"],
        "ft_macro_f1": edge_stats["macro_f1"],
        "ft_pos_weighted_f1": edge_stats["pos_weighted_f1"],
        "ft_n_edges_scored": edge_stats["n_edges_used"],
        "ft_pooled_micro_precision": edge_stats["pooled_micro_precision"],
        "ft_pooled_micro_recall": edge_stats["pooled_micro_recall"],
        "ft_pooled_micro_accuracy": edge_stats["pooled_micro_accuracy"],
        "ft_macro_precision": edge_stats["macro_precision"],
        "ft_macro_recall": edge_stats["macro_recall"],
        "ft_pos_weighted_precision": edge_stats["pos_weighted_precision"],
        "ft_pos_weighted_recall": edge_stats["pos_weighted_recall"],
        "ft_confusion_matrix_pooled": edge_stats["confusion_matrix_pooled"],
        "ft_per_edge_confusion": edge_stats["per_edge_confusion"],
    }
    out.update(leaf_stats)
    return out


def train_full_tree_and_summarize(
    model_name: str,
    tree: TopicTree,
    pool: MultilabelBinaryPoolData,
    factory: BinaryEdgeFactory,
    *,
    split: str = "test",
    verbose: bool = True,
    restrict_to_parent_subtree: bool = True,
) -> Tuple[MultilabelHierarchyRouter, Dict[str, Any]]:
    """
    Fresh router: :func:`fit_all_binary_edges`, then :func:`full_tree_test_metrics` on
    ``split`` (default held-out test).
    """
    router = MultilabelHierarchyRouter(tree, factory)
    fit_counts = fit_all_binary_edges(
        router,
        pool,
        verbose=verbose,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )
    stats = full_tree_test_metrics(
        router,
        pool,
        tree,
        split=split,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )
    row: Dict[str, Any] = {
        "model": model_name,
        **stats,
        "fit_n_specs": float(fit_counts["n_specs"]),
        "fit_n_edges_trained": float(fit_counts["n_fitted"]),
        "fit_n_skipped_single_class": float(fit_counts["n_skipped_single_class_train"]),
    }
    return router, row


def _predict_labels(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    parent: str,
    child: str,
    split: str,
    *,
    restrict_to_parent_subtree: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    m = router.edge_model(parent, child)
    if m is None:
        return None
    X, y_true = pool.binary_edge_dataset(
        parent, child, split, restrict_to_parent_subtree=restrict_to_parent_subtree
    )
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
    *,
    restrict_to_parent_subtree: bool = True,
) -> Dict[str, float]:
    """
    Pool predictions over listed edges (same docs repeated per edge).

    - **pooled_micro_f1**: one F1 on concatenated (y_true, y_pred) over all edge×doc pairs.
    - **macro_f1**: mean of per-edge F1 (edges with <2 classes in split skipped).
    - **pos_weighted_f1**: Σ_w f1_e with weights w = positive count on that edge in the split.

    Same **macro** / **pooled_micro** / **pos_weighted** pattern for **precision** and **recall**.

    **confusion_matrix_pooled**: sklearn ``confusion_matrix(..., labels=[0, 1])`` on pooled
    predictions (rows = true class, cols = predicted).

    **per_edge_confusion**: ``{(parent, child): ndarray(2, 2)}`` for each scored edge.

    ``restrict_to_parent_subtree`` must match training (see :meth:`MultilabelBinaryPoolData.binary_edge_dataset`).
    """
    yt_all: List[int] = []
    yp_all: List[int] = []
    f1_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []
    pos_counts: List[int] = []
    per_edge_cm: Dict[Tuple[str, str], np.ndarray] = {}

    for parent, child in edges:
        pr = _predict_labels(
            router,
            pool,
            parent,
            child,
            split,
            restrict_to_parent_subtree=restrict_to_parent_subtree,
        )
        if pr is None:
            continue
        yt, yp = pr
        if len(np.unique(yt)) < 2:
            continue
        f1e = float(f1_score(yt, yp, zero_division=0))
        prece = float(precision_score(yt, yp, zero_division=0))
        rece = float(recall_score(yt, yp, zero_division=0))
        f1_list.append(f1e)
        prec_list.append(prece)
        rec_list.append(rece)
        pos_counts.append(int(np.sum(yt)))
        yt_all.extend(yt.tolist())
        yp_all.extend(yp.tolist())
        per_edge_cm[(parent, child)] = confusion_matrix(yt, yp, labels=[0, 1])

    pooled = (
        float(f1_score(yt_all, yp_all, zero_division=0)) if yt_all else float("nan")
    )
    macro = float(np.mean(f1_list)) if f1_list else float("nan")
    macro_p = float(np.mean(prec_list)) if prec_list else float("nan")
    macro_r = float(np.mean(rec_list)) if rec_list else float("nan")
    sp = sum(pos_counts)
    pos_w = (
        float(np.dot(pos_counts, f1_list) / sp) if sp > 0 and f1_list else float("nan")
    )
    pos_w_p = (
        float(np.dot(pos_counts, prec_list) / sp) if sp > 0 and prec_list else float("nan")
    )
    pos_w_r = (
        float(np.dot(pos_counts, rec_list) / sp) if sp > 0 and rec_list else float("nan")
    )
    pooled_p = (
        float(precision_score(yt_all, yp_all, zero_division=0))
        if yt_all
        else float("nan")
    )
    pooled_r = (
        float(recall_score(yt_all, yp_all, zero_division=0))
        if yt_all
        else float("nan")
    )
    pooled_acc = (
        float(accuracy_score(yt_all, yp_all)) if yt_all else float("nan")
    )
    cm_pooled = (
        confusion_matrix(yt_all, yp_all, labels=[0, 1])
        if yt_all
        else np.zeros((2, 2), dtype=int)
    )
    return {
        "pooled_micro_f1": pooled,
        "macro_f1": macro,
        "pos_weighted_f1": pos_w,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "pos_weighted_precision": pos_w_p,
        "pos_weighted_recall": pos_w_r,
        "pooled_micro_precision": pooled_p,
        "pooled_micro_recall": pooled_r,
        "pooled_micro_accuracy": pooled_acc,
        "confusion_matrix_pooled": cm_pooled,
        "per_edge_confusion": per_edge_cm,
        "n_edges_used": float(len(f1_list)),
    }


def pooled_edge_f1_stats_by_parent_depth(
    router: MultilabelHierarchyRouter,
    pool: MultilabelBinaryPoolData,
    tree: TopicTree,
    split: str,
    *,
    restrict_to_parent_subtree: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Same metrics as :func:`pooled_edge_f1_stats`, computed **separately for each parent depth**
    (``BinaryEdgeSpec.depth``: distance of the **parent** node from ``traversal_root``).

    Keys are integer depths; values are the full dict from ``pooled_edge_f1_stats`` for edges
    at that depth only (empty depths omitted).
    """
    by_depth: Dict[int, List[Tuple[str, str]]] = {}
    for spec in binary_edge_specs(tree):
        by_depth.setdefault(spec.depth, []).append((spec.parent, spec.child))
    out: Dict[int, Dict[str, Any]] = {}
    for d in sorted(by_depth):
        edges = by_depth[d]
        if not edges:
            continue
        st = pooled_edge_f1_stats(
            router,
            pool,
            edges,
            split,
            restrict_to_parent_subtree=restrict_to_parent_subtree,
        )
        out[d] = st
    return out


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
    restrict_to_parent_subtree: bool = True,
) -> Dict[str, Any]:
    """
    One row of summary metrics for **H1 (Root)**: macro / pooled-micro / pos-weighted F1,
    the same three aggregates for **precision** and **recall**, plus **pooled** and
    **per-edge** binary confusion matrices.

    Set ``include_leaf=True`` to also add :func:`leaf_level_evaluation` (multilabel leaf F1
    + path-to-gold branching recall; slower than H1-only).

    ``restrict_to_parent_subtree`` must match how H1 was fitted (see :func:`fit_h1`).
    """
    root = tree.traversal_root
    h1_edges = h1_edge_tuples(tree)
    per = evaluate_binary_edges_from_parent(
        router,
        pool,
        root,
        split=split,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )
    h1_f1s = [per[ch]["f1"] for ch in per if "f1" in per[ch]]
    h1_macro = float(np.mean(h1_f1s)) if h1_f1s else float("nan")

    pool_h1 = pooled_edge_f1_stats(
        router,
        pool,
        h1_edges,
        split,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )

    row: Dict[str, Any] = {
        "model": name,
        "h1_macro_f1": h1_macro,
        "h1_macro_precision": pool_h1["macro_precision"],
        "h1_macro_recall": pool_h1["macro_recall"],
        "h1_pooled_micro_f1": pool_h1["pooled_micro_f1"],
        "h1_pooled_micro_precision": pool_h1["pooled_micro_precision"],
        "h1_pooled_micro_recall": pool_h1["pooled_micro_recall"],
        "h1_pooled_micro_accuracy": pool_h1["pooled_micro_accuracy"],
        "h1_pos_weighted_f1": pool_h1["pos_weighted_f1"],
        "h1_pos_weighted_precision": pool_h1["pos_weighted_precision"],
        "h1_pos_weighted_recall": pool_h1["pos_weighted_recall"],
        "h1_pooled_confusion_matrix": pool_h1["confusion_matrix_pooled"],
        "h1_per_edge_confusion": pool_h1["per_edge_confusion"],
    }
    if include_leaf:
        row.update(leaf_level_evaluation(pool, router, tree, split))
    return row


def pilot_kernel_svc_on_edge(
    pool: MultilabelBinaryPoolData,
    parent: str,
    child: str,
    *,
    max_train: int = 4000,
    max_features: int = 8000,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    **Phase 4 pilot:** fit ``TfidfVectorizer`` + ``sklearn.svm.SVC`` on a **subsample** of the
    **train** split for one edge; compare linear / RBF / polynomial kernels with timing.

    Not full-tree — estimates whether nonlinear kernels are worth the cost at all.
    """
    import time

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    X, y = pool.binary_edge_dataset(parent, child, "train")
    if len(X) < 20 or len(set(y)) < 2:
        return {}
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    if len(idx) > max_train:
        idx = rng.choice(idx, size=max_train, replace=False)
    Xs = [X[i] for i in idx]
    ys = [int(y[i]) for i in idx]
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xs, ys, test_size=0.2, random_state=random_state, stratify=ys
    )
    tfidf_kw = dict(min_df=2, max_features=max_features)
    configs = {
        "SVC_linear": dict(kernel="linear", C=1.0),
        "SVC_rbf": dict(kernel="rbf", C=1.0, gamma="scale"),
        "SVC_poly": dict(kernel="poly", C=1.0, degree=2, gamma="scale"),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, skw in configs.items():
        t0 = time.perf_counter()
        pipe = Pipeline(
            [("tfidf", TfidfVectorizer(**tfidf_kw)), ("clf", SVC(**skw, max_iter=5000))]
        )
        pipe.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        y_pred = pipe.predict(X_va)
        f1 = float(f1_score(y_va, y_pred, zero_division=0))
        out[name] = {"f1_val": f1, "fit_time_sec": fit_s, "kernel_params": skw}
    return out


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
    restrict_to_parent_subtree: bool = True,
) -> Tuple[MultilabelHierarchyRouter, Dict[str, Any]]:
    """Fresh router: fit H1 only, return (router, summary dict for split)."""
    router = MultilabelHierarchyRouter(tree, factory)
    fit_h1(router, pool, restrict_to_parent_subtree=restrict_to_parent_subtree)
    summ = summary_row(
        model_name,
        router,
        pool,
        tree,
        split=split,
        include_leaf=include_leaf,
        restrict_to_parent_subtree=restrict_to_parent_subtree,
    )
    return router, summ
