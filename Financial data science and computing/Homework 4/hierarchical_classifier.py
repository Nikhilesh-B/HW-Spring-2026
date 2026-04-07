"""
Pluggable local decision models + top-down hierarchical router.

The router does not assume a specific learning algorithm; inject a factory that builds
one NodeDecisionModel per LocalClassifierSpec.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable

from topic_hierarchy import BinaryEdgeSpec, LocalClassifierSpec, TopicTree, load_topic_tree


@runtime_checkable
class NodeDecisionModel(Protocol):
    """Contract for a trainable router at one branching node."""

    def fit(self, X: Sequence[Any], y: Sequence[str]) -> None:
        """Fit on inputs and gold **child** topic codes (same strings as in TopicTree.children)."""
        ...

    def predict_next_child(self, x: Any) -> str:
        """Return predicted child topic code for a single example."""
        ...


class NodeDecisionModelBase(ABC):
    """Optional ABC if you prefer subclassing over Protocol."""

    @abstractmethod
    def fit(self, X: Sequence[Any], y: Sequence[str]) -> None:
        pass

    @abstractmethod
    def predict_next_child(self, x: Any) -> str:
        pass


ModelFactory = Callable[[LocalClassifierSpec], NodeDecisionModel]


@runtime_checkable
class BinaryEdgeModel(Protocol):
    """
    **Contract** for any per-edge binary classifier you plug into the hierarchy.

    **Required (must implement):**

    - ``fit(X, y)`` — train on batch inputs ``X`` (typically raw article strings) and
      binary labels ``y`` in ``{0, 1}``.
    - ``predict_binary(x)`` — return ``0`` or ``1`` for a **single** input ``x`` (same type
      as elements of ``X``, usually one document string).

    **Optional (recommended for ROC / AUC in** ``hierarchical_evaluation`` **):**

    - ``decision_function(x)`` — real-valued score for the positive class (higher = more
      likely positive). If missing, evaluation falls back to ``predict`` only.

    The router (``MultilabelHierarchyRouter``) and training loop only call ``fit`` and
    ``predict_binary``; evaluation may call ``decision_function`` if you define it.

    Your backend (sklearn, PyTorch, custom code, etc.) stays behind this surface. Provide a
    ``BinaryEdgeFactory``: ``Callable[[BinaryEdgeSpec], BinaryEdgeModel]`` — e.g.
    ``lambda spec: MyClassifier()`` — and pass it to ``MultilabelHierarchyRouter``.
    """

    def fit(self, X: Sequence[Any], y: Sequence[int]) -> None:
        ...

    def predict_binary(self, x: Any) -> int:
        ...


BinaryEdgeFactory = Callable[[BinaryEdgeSpec], BinaryEdgeModel]


class BinaryEdgeModelBase(ABC):
    """
    Subclass this and implement ``fit`` + ``predict_binary`` for a custom edge model.

    Override ``decision_function`` when your backend exposes scores (needed for full ROC in
    :mod:`hierarchical_evaluation`).
    """

    @abstractmethod
    def fit(self, X: Sequence[Any], y: Sequence[int]) -> None:
        pass

    @abstractmethod
    def predict_binary(self, x: Any) -> int:
        pass

    def decision_function(self, x: Any) -> float:
        raise NotImplementedError


class HierarchicalClassifier:
    """
    Top-down inference: at each branching node, delegate to the local model; unary nodes follow the only child.
    """

    def __init__(
        self,
        tree: TopicTree,
        model_factory: ModelFactory,
        models: Optional[Dict[str, NodeDecisionModel]] = None,
    ) -> None:
        self.tree = tree
        self.model_factory = model_factory
        self._models: Dict[str, NodeDecisionModel] = dict(models or {})
        self._spec_by_node: Dict[str, LocalClassifierSpec] = {
            s.node_id: s for s in tree.local_classifier_specs()
        }

    def spec(self, node_id: str) -> Optional[LocalClassifierSpec]:
        return self._spec_by_node.get(node_id)

    def has_model(self, node_id: str) -> bool:
        """True if this branching node has been fitted (present in the router)."""
        return node_id in self._models

    def ensure_model(self, node_id: str) -> NodeDecisionModel:
        if node_id not in self._models:
            sp = self._spec_by_node.get(node_id)
            if sp is None:
                raise KeyError(f"No branching spec for node {node_id!r}")
            self._models[node_id] = self.model_factory(sp)
        return self._models[node_id]

    def fit_node(
        self,
        node_id: str,
        X: Sequence[Any],
        y: Sequence[str],
    ) -> None:
        """Fit (or refit) the local model for one branching node."""
        m = self.ensure_model(node_id)
        m.fit(X, y)

    def train_all(
        self,
        build_xy: Callable[[str], Tuple[Sequence[Any], Sequence[str]]],
        *,
        skip_empty: bool = True,
    ) -> Dict[str, int]:
        """
        Train every branching node.

        ``build_xy(node_id) -> (X, y_child_codes)`` for the training split.
        Returns map node_id -> n_samples trained.
        """
        counts: Dict[str, int] = {}
        for spec in self.tree.local_classifier_specs():
            X, y = build_xy(spec.node_id)
            n = len(X)
            if n == 0 and skip_empty:
                counts[spec.node_id] = 0
                continue
            if len(set(y)) < 2:
                counts[spec.node_id] = 0
                continue
            self.fit_node(spec.node_id, X, y)
            counts[spec.node_id] = n
        return counts

    def predict_path(self, x: Any, stop_at_leaf: bool = True) -> List[str]:
        """
        Walk from traversal_root. Return visited nodes [root, ..., leaf_or_stop].
        """
        path: List[str] = []
        u: Optional[str] = self.tree.traversal_root
        while u is not None:
            path.append(u)
            chs = self.tree.children.get(u, [])
            if not chs:
                break
            if len(chs) == 1:
                u = chs[0]
                continue
            model = self._models.get(u)
            if model is None:
                raise RuntimeError(f"No model fitted for branching node {u!r}")
            nxt = model.predict_next_child(x)
            if nxt not in chs:
                raise ValueError(
                    f"Predicted child {nxt!r} not among children of {u!r}: {chs}")
            u = nxt
            if stop_at_leaf and not self.tree.children.get(u):
                path.append(u)
                break
        return path

    def predict_leaf(self, x: Any) -> str:
        """Last node on the predicted path."""
        pth = self.predict_path(x)
        return pth[-1]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        meta = {
            "traversal_root": self.tree.traversal_root,
            "children": self.tree.children,
            "parent": self.tree.parent,
        }
        with open(os.path.join(directory, "tree_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        with open(os.path.join(directory, "models.pkl"), "wb") as f:
            pickle.dump(self._models, f)

    @classmethod
    def load(
        cls,
        directory: str,
        model_factory: ModelFactory,
    ) -> "HierarchicalClassifier":
        with open(os.path.join(directory, "tree_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        tree = TopicTree(
            children=meta["children"],
            parent=meta["parent"],
            traversal_root=meta["traversal_root"],
        )
        with open(os.path.join(directory, "models.pkl"), "rb") as f:
            models = pickle.load(f)
        return cls(tree, model_factory, models=models)


def load_tree_from_topics(topics_path: str, traversal_root: str = "Root") -> TopicTree:
    return load_topic_tree(topics_path, traversal_root=traversal_root)


class MultilabelHierarchyRouter:
    """
    Multi-label hierarchical inference: at each branching parent, run one binary per child;
    propagate all positive children; unary nodes pass through without a model.
    """

    def __init__(
        self,
        tree: TopicTree,
        model_factory: BinaryEdgeFactory,
        models: Optional[Dict[Tuple[str, str], BinaryEdgeModel]] = None,
    ) -> None:
        self.tree = tree
        self.model_factory = model_factory
        self._models: Dict[Tuple[str, str],
                           BinaryEdgeModel] = dict(models or {})

    def ensure_edge_model(self, spec: BinaryEdgeSpec) -> BinaryEdgeModel:
        k = spec.key
        if k not in self._models:
            self._models[k] = self.model_factory(spec)
        return self._models[k]

    def fit_edge(
        self,
        parent: str,
        child: str,
        X: Sequence[Any],
        y: Sequence[int],
        depth: int = 0,
    ) -> None:
        sp = BinaryEdgeSpec(parent=parent, child=child, depth=depth)
        m = self.ensure_edge_model(sp)
        m.fit(X, y)

    def has_edge_model(self, parent: str, child: str) -> bool:
        return (parent, child) in self._models

    def edge_model(self, parent: str, child: str) -> Optional[BinaryEdgeModel]:
        """Return the fitted binary model for ``parent -> child``, or ``None`` if absent."""
        return self._models.get((parent, child))

    def fitted_edge_keys(self) -> List[Tuple[str, str]]:
        """All ``(parent, child)`` keys that currently have a model object in this router."""
        return list(self._models.keys())

    def predict_reached_nodes(self, x: Any) -> Set[str]:
        """All topic nodes reached by positive binary decisions + unary pass-through."""
        tree = self.tree
        reached: Set[str] = set()
        queue: List[str] = [tree.traversal_root]
        seen_expand: Set[str] = set()
        while queue:
            v = queue.pop(0)
            if v in seen_expand:
                continue
            seen_expand.add(v)
            chs = tree.children.get(v, [])
            if not chs:
                continue
            if len(chs) == 1:
                c = chs[0]
                reached.add(c)
                if tree.children.get(c):
                    queue.append(c)
                continue
            for c in chs:
                m = self._models.get((v, c))
                if m is None:
                    continue
                if m.predict_binary(x) == 1:
                    reached.add(c)
                    if tree.children.get(c):
                        queue.append(c)
        return reached

    def predict_reached_leaves(self, x: Any) -> Set[str]:
        """Reached nodes that are leaves in the taxonomy."""
        r = self.predict_reached_nodes(x)
        leaves = self.tree.leaf_nodes()
        return r & leaves


class SklearnBinaryEdgeNode(BinaryEdgeModelBase):
    """TF-IDF + linear model with 0/1 labels; supports ``decision_function`` or ``predict_proba``."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def fit(self, X: Sequence[Any], y: Sequence[int]) -> None:
        self.pipeline.fit(list(X), list(y))

    def predict_binary(self, x: Any) -> int:
        return int(self.pipeline.predict([x])[0])

    def decision_function(self, x: Any) -> float:
        if hasattr(self.pipeline, "decision_function"):
            return float(self.pipeline.decision_function([x])[0])
        proba = self.pipeline.predict_proba([x])[0]
        return float(proba[1] - 0.5)


def binary_edge_factory(
    tfidf_kw: Optional[Dict[str, Any]] = None,
    clf_kw: Optional[Dict[str, Any]] = None,
    *,
    estimator: Optional[
        Union[Type[Any], Callable[[BinaryEdgeSpec], Any], Any]
    ] = None,
    text_vectorizer: str = "tfidf",
) -> BinaryEdgeFactory:
    """
    Build ``Pipeline(vectorizer, classifier)`` wrappers for each edge.

    **text_vectorizer** (first step):

    - ``"tfidf"`` — :class:`~sklearn.feature_extraction.text.TfidfVectorizer` with IDF (default).
    - ``"tf_l2"`` — ``TfidfVectorizer(use_idf=False)`` (TF with L2 norm; no IDF).
    - ``"count"`` — :class:`~sklearn.feature_extraction.text.CountVectorizer` (shared keys only).

    **estimator** (how to pick the classifier):

    - ``None`` (default): ``LogisticRegression(**clf_kw)``.
    - A **class** (e.g. ``LinearSVC``, ``RandomForestClassifier``): ``estimator(**clf_kw)`` — use
      ``clf_kw`` for constructor kwargs (``C``, ``max_iter``, ``n_estimators``, …).
    - A **fitted-style instance** (e.g. ``LinearSVC(C=0.5)``): ``sklearn.base.clone(estimator)``
      so each edge gets a fresh copy.
    - A **callable** ``f(spec: BinaryEdgeSpec) -> sklearn estimator``: return a new estimator per
      edge (full control; ignore ``clf_kw`` unless your callable uses it).

    The last step must implement ``fit``, ``predict`` (labels 0/1), and ideally ``decision_function``
    or ``predict_proba`` for ROC metrics in :mod:`hierarchical_evaluation`.
    """

    def _make_classifier_step(spec: BinaryEdgeSpec) -> Any:
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression

        ckw = dict(max_iter=3000)
        if clf_kw:
            ckw.update(clf_kw)

        if estimator is None:
            return LogisticRegression(**ckw)
        if isinstance(estimator, type):
            return estimator(**ckw)
        if callable(estimator):
            return estimator(spec)
        return clone(estimator)

    def _vectorizer_step():
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        tkw = dict(min_df=2, max_features=30_000)
        if tfidf_kw:
            tkw.update(tfidf_kw)
        mode = (text_vectorizer or "tfidf").lower().strip()
        if mode == "tfidf":
            return TfidfVectorizer(**tkw)
        if mode in ("tf_l2", "tf", "tf_only"):
            t2 = dict(tkw)
            t2["use_idf"] = False
            return TfidfVectorizer(**t2)
        if mode == "count":
            cv_allowed = {
                "analyzer",
                "binary",
                "decode_error",
                "dtype",
                "encoding",
                "input",
                "lowercase",
                "max_df",
                "max_features",
                "min_df",
                "ngram_range",
                "preprocessor",
                "stop_words",
                "strip_accents",
                "token_pattern",
                "tokenizer",
                "vocabulary",
            }
            cv_kw = {k: v for k, v in tkw.items() if k in cv_allowed}
            return CountVectorizer(**cv_kw)
        raise ValueError(
            "text_vectorizer must be 'tfidf', 'tf_l2', or 'count', "
            f"got {text_vectorizer!r}"
        )

    def _factory(spec: BinaryEdgeSpec) -> BinaryEdgeModel:
        from sklearn.pipeline import Pipeline

        vec_step = _vectorizer_step()
        clf_step = _make_classifier_step(spec)
        pipe = Pipeline([("vect", vec_step), ("clf", clf_step)])
        return SklearnBinaryEdgeNode(pipe)

    return _factory


def svm_linear_binary_edge_factory(
    *,
    max_features: int = 8000,
    text_vectorizer: str = "tfidf",
    svm_kw: Optional[Dict[str, Any]] = None,
    vectorizer_kw: Optional[Dict[str, Any]] = None,
) -> BinaryEdgeFactory:
    """
    :class:`~sklearn.svm.LinearSVC` with TF-IDF, TF (no IDF), or count features.

    ``text_vectorizer``: ``tfidf`` | ``tf_l2`` | ``count`` (see :func:`binary_edge_factory`).
    """
    from sklearn.svm import LinearSVC

    tkw = dict(min_df=2, max_features=max_features)
    if vectorizer_kw:
        tkw.update(vectorizer_kw)
    skw: Dict[str, Any] = dict(C=1.0, dual=False, max_iter=8000)
    if svm_kw:
        skw.update(svm_kw)
    return binary_edge_factory(
        tfidf_kw=tkw,
        estimator=LinearSVC,
        clf_kw=skw,
        text_vectorizer=text_vectorizer,
    )


def svm_tfidf_truncated_svd_linear_edge_factory(
    *,
    max_features: int = 8000,
    n_components: int = 500,
    random_state: int = 42,
    svm_kw: Optional[Dict[str, Any]] = None,
    vectorizer_kw: Optional[Dict[str, Any]] = None,
) -> BinaryEdgeFactory:
    """
    ``TfidfVectorizer`` → ``TruncatedSVD`` → ``LinearSVC`` (LSI-style reduction per edge).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    tkw = dict(min_df=2, max_features=max_features)
    if vectorizer_kw:
        tkw.update(vectorizer_kw)
    skw: Dict[str, Any] = dict(C=1.0, dual=False, max_iter=8000)
    if svm_kw:
        skw.update(svm_kw)

    def _factory(spec: BinaryEdgeSpec) -> BinaryEdgeModel:
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tkw)),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=n_components, random_state=random_state
                    ),
                ),
                ("clf", LinearSVC(**skw)),
            ]
        )
        return SklearnBinaryEdgeNode(pipe)

    return _factory


def linear_svc_estimator_by_depth(
    c_by_depth: Dict[int, float],
    *,
    default_c: float = 1.0,
    dual: bool = False,
    max_iter: int = 8000,
    class_weight: Optional[str] = None,
) -> Callable[[BinaryEdgeSpec], Any]:
    """
    Build a callable ``estimator(spec)`` for :func:`binary_edge_factory` that sets
    ``LinearSVC(C=...)`` from ``spec.depth`` (parent depth from ``Root``).
    """

    def _estimator(spec: BinaryEdgeSpec) -> Any:
        from sklearn.svm import LinearSVC

        c = float(c_by_depth.get(spec.depth, default_c))
        kw: Dict[str, Any] = dict(
            C=c, dual=dual, max_iter=max_iter, random_state=42
        )
        if class_weight is not None:
            kw["class_weight"] = class_weight
        return LinearSVC(**kw)

    return _estimator


class SklearnMulticlassNode(NodeDecisionModelBase):
    """
    Example backend: sklearn ``Pipeline`` with a classifier whose ``predict`` returns class indices.
    Child topic codes are encoded with ``LabelEncoder``.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self._le: Any = None

    def fit(self, X: Sequence[Any], y: Sequence[str]) -> None:
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder()
        yy = self._le.fit_transform(list(y))
        self.pipeline.fit(list(X), yy)

    def predict_next_child(self, x: Any) -> str:
        if self._le is None:
            raise RuntimeError("fit must be called before predict_next_child")
        pred = self.pipeline.predict([x])[0]
        return str(self._le.inverse_transform([pred])[0])


def sklearn_linear_factory(
    tfidf_kw: Optional[Dict[str, Any]] = None,
    svm_kw: Optional[Dict[str, Any]] = None,
) -> ModelFactory:
    """Factory producing ``SklearnMulticlassNode`` with TF-IDF + ``LinearSVC`` per spec."""

    def _factory(spec: LocalClassifierSpec) -> NodeDecisionModel:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        tkw = dict(min_df=2, max_features=50_000)
        if tfidf_kw:
            tkw.update(tfidf_kw)
        skw = dict(C=1.0, dual=False)
        if svm_kw:
            skw.update(svm_kw)
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tkw)),
                ("clf", LinearSVC(**skw)),
            ]
        )
        return SklearnMulticlassNode(pipe)

    return _factory
