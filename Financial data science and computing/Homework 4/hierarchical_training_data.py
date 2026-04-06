"""
Build per-branching-node (X, y_child) from the leaf pool CSVs + primary leaf path rule.

Primary path rule (deterministic, multi-label RCV1):
  For each article, take the **deepest** topic row(s) in ``labels_leaf_*_long.csv`` (by h1..h5 rank);
  if ties, choose **lexicographically smallest** ``cat``. Build Root->...->leaf using ``TopicTree.path_from_root_to``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from topic_hierarchy import TopicTree, load_topic_tree

LEVEL_RANK = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5}

# train | test
Split = str


def _base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def primary_leaf_per_article(labels_long: pd.DataFrame) -> Dict[int, str]:
    """Map article id -> one leaf topic code (deepest level; tie-break min cat)."""
    if labels_long.empty:
        return {}
    df = labels_long.copy()
    df["rank"] = df["level"].str.lower().map(LEVEL_RANK)
    mx = df.groupby("id")["rank"].transform("max")
    df = df[df["rank"] == mx]
    return (
        df.sort_values(["id", "cat"])
        .groupby("id")["cat"]
        .first()
        .map(lambda s: str(s).strip())
        .to_dict()
    )


def paths_for_primary_leaves(
    tree: TopicTree, primary_leaf: Dict[int, str]
) -> Dict[int, List[str]]:
    """id -> [Root, ..., leaf] for each article with a resolvable leaf in the tree."""
    all_n = tree.all_nodes()
    out: Dict[int, List[str]] = {}
    for aid, leaf in primary_leaf.items():
        leaf = str(leaf).strip()
        if leaf not in all_n:
            continue
        p = tree.path_from_root_to(leaf)
        if p and p[0] == tree.traversal_root:
            out[int(aid)] = p
    return out


def load_pool_ids(path: str) -> List[int]:
    df = pd.read_csv(path)
    col = "id" if "id" in df.columns else df.columns[0]
    return [int(x) for x in df[col].tolist()]


def load_article_label_sets(news_topics_path: str, pool_ids: Set[int]) -> Dict[int, Set[str]]:
    """
    Full multi-label assignment: article id -> set of topic codes from ``news_topics.csv``.

    Reads in chunks so large files stay tractable; only rows with ``id`` in ``pool_ids`` are kept.
    """
    if not pool_ids:
        return {}
    pool_ids = set(int(x) for x in pool_ids)
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        news_topics_path,
        chunksize=400_000,
        usecols=["id", "cat"],
    ):
        sub = chunk[chunk["id"].isin(pool_ids)]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        return {}
    df = pd.concat(chunks, ignore_index=True)
    df["cat"] = df["cat"].astype(str).str.strip()
    out: Dict[int, Set[str]] = {}
    for aid, grp in df.groupby("id"):
        out[int(aid)] = set(grp["cat"].unique())
    return out


class LeafPoolData:
    """
    Article pool + primary paths; produces training rows for any branching node.

    Parameters use default Homework 4 filenames when paths are None.
    """

    def __init__(
        self,
        tree: TopicTree,
        articles_path: Optional[str] = None,
        labels_long_path: Optional[str] = None,
        train_ids_path: Optional[str] = None,
        test_ids_path: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> None:
        self.tree = tree
        root = base_path or _base_dir()
        self.articles_path = articles_path or os.path.join(
            root, "articles_leaf_min60_cap200.csv"
        )
        self.labels_long_path = labels_long_path or os.path.join(
            root, "labels_leaf_min60_cap200_long.csv"
        )
        self.train_ids_path = train_ids_path or os.path.join(
            root, "articles_leaf_min60_cap200_train_ids.csv"
        )
        self.test_ids_path = test_ids_path or os.path.join(
            root, "articles_leaf_min60_cap200_test_ids.csv"
        )

        self._articles: Optional[pd.DataFrame] = None
        self._paths: Optional[Dict[int, List[str]]] = None
        self._train_ids: Optional[List[int]] = None
        self._test_ids: Optional[List[int]] = None

    @property
    def articles(self) -> pd.DataFrame:
        if self._articles is None:
            self._articles = pd.read_csv(self.articles_path)
            self._articles["id"] = self._articles["id"].astype(int)
        return self._articles

    def train_ids(self) -> List[int]:
        if self._train_ids is None:
            self._train_ids = load_pool_ids(self.train_ids_path)
        return self._train_ids

    def test_ids(self) -> List[int]:
        if self._test_ids is None:
            self._test_ids = load_pool_ids(self.test_ids_path)
        return self._test_ids

    def paths(self) -> Dict[int, List[str]]:
        if self._paths is None:
            labels = pd.read_csv(self.labels_long_path)
            labels["cat"] = labels["cat"].astype(str).str.strip()
            pool = set(self.train_ids()) | set(self.test_ids())
            labels = labels[labels["id"].isin(pool)]
            primary = primary_leaf_per_article(labels)
            self._paths = paths_for_primary_leaves(self.tree, primary)
        return self._paths

    def node_dataset(
        self,
        node_id: str,
        split: Split,
        input_column: str = "article",
    ) -> Tuple[List[Any], List[str]]:
        """
        Rows where the primary path passes through ``node_id``; target is the **next** hop child.

        Returns (inputs, y_child_topic_codes). Empty if no examples.
        """
        ids = self.train_ids() if split == "train" else self.test_ids()
        paths = self.paths()
        art = self.articles.set_index("id")
        chs = self.tree.children.get(node_id, [])
        if len(chs) < 2:
            return [], []

        X: List[Any] = []
        y: List[str] = []
        for aid in ids:
            pth = paths.get(aid)
            if not pth or node_id not in pth:
                continue
            i = pth.index(node_id)
            if i + 1 >= len(pth):
                continue
            nxt = pth[i + 1]
            if nxt not in chs:
                continue
            if aid not in art.index:
                continue
            row = art.loc[aid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            val = row[input_column]
            X.append(val)
            y.append(nxt)
        return X, y

    def node_class_counts(self, node_id: str, split: Split) -> Dict[str, int]:
        _, y = self.node_dataset(node_id, split)
        out: Dict[str, int] = {}
        for c in y:
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))


class MultilabelBinaryPoolData:
    """
    Multi-label gold from ``news_topics``: for each branching edge (parent -> child),
    binary positive iff ``labels(article) ∩ subtree(child)`` is non-empty.

    Use with local binary relevance training and :class:`MultilabelHierarchyRouter`.
    """

    def __init__(
        self,
        tree: TopicTree,
        article_labels: Dict[int, Set[str]],
        articles: pd.DataFrame,
        train_ids: List[int],
        test_ids: List[int],
    ) -> None:
        self.tree = tree
        self.article_labels = article_labels
        self._articles = articles
        self._train_ids = train_ids
        self._test_ids = test_ids
        self._subtree_cache: Dict[str, Set[str]] = {}

    def subtree_codes(self, node: str) -> Set[str]:
        if node not in self._subtree_cache:
            self._subtree_cache[node] = self.tree.subtree_topic_codes(node)
        return self._subtree_cache[node]

    def gold_positive_edge(self, article_id: int, child: str) -> bool:
        """True if any assigned label lies in the subtree rooted at ``child``."""
        labs = self.article_labels.get(article_id, set())
        if not labs:
            return False
        st = self.subtree_codes(child)
        return not labs.isdisjoint(st)

    def binary_edge_dataset(
        self,
        parent: str,
        child: str,
        split: Split,
        input_column: str = "article",
    ) -> Tuple[List[Any], List[int]]:
        """
        All pool articles in the split with texts; ``y`` is 0/1 for subtree(child) membership.
        """
        ids = self._train_ids if split == "train" else self._test_ids
        art = self._articles.set_index("id")
        X: List[Any] = []
        y: List[int] = []
        for aid in ids:
            if aid not in art.index:
                continue
            row = art.loc[aid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            val = row[input_column]
            yy = 1 if self.gold_positive_edge(aid, child) else 0
            X.append(val)
            y.append(yy)
        return X, y

    def articles_df(self) -> pd.DataFrame:
        return self._articles

    def train_ids(self) -> List[int]:
        return self._train_ids

    def test_ids(self) -> List[int]:
        return self._test_ids


def make_multilabel_binary_pool_data(
    topics_path: Optional[str] = None,
    news_topics_path: Optional[str] = None,
    traversal_root: str = "Root",
    base_path: Optional[str] = None,
    **kwargs: Any,
) -> MultilabelBinaryPoolData:
    """
    Load ``TopicTree``, leaf-pool articles + train/test ids, and full label sets from ``news_topics``.
    """
    root = base_path or _base_dir()
    tpath = topics_path or os.path.join(root, "topics.csv")
    tree = load_topic_tree(tpath, traversal_root=traversal_root)
    nt_path = news_topics_path or os.path.join(root, "news_topics.csv")

    lp = LeafPoolData(tree, base_path=root, **kwargs)
    pool = set(lp.train_ids()) | set(lp.test_ids())
    labels = load_article_label_sets(nt_path, pool)
    articles = lp.articles
    return MultilabelBinaryPoolData(
        tree,
        labels,
        articles,
        lp.train_ids(),
        lp.test_ids(),
    )


def make_leaf_pool_data(
    topics_path: Optional[str] = None,
    traversal_root: str = "Root",
    base_path: Optional[str] = None,
    **kwargs: Any,
) -> LeafPoolData:
    """Convenience: load default ``TopicTree`` + ``LeafPoolData``."""
    root = base_path or _base_dir()
    tpath = topics_path or os.path.join(root, "topics.csv")
    tree = load_topic_tree(tpath, traversal_root=traversal_root)
    return LeafPoolData(tree, base_path=root, **kwargs)
