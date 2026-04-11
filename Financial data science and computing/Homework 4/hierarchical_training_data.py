"""
Build per-branching-node (X, y_child) from the leaf pool CSVs + primary leaf path rule.

Primary path rule (deterministic, multi-label RCV1):
  For each article, take the **deepest** topic row(s) in ``labels_leaf_*_long.csv`` (by h1..h5 rank);
  if ties, choose **lexicographically smallest** ``cat``. Build Root->...->leaf using ``TopicTree.path_from_root_to``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from typing import Literal
except ImportError:  # Python < 3.8
    from typing_extensions import Literal

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
        *,
        restrict_to_parent_subtree: bool = True,
    ) -> Tuple[List[Any], List[int]]:
        """
        Build ``(X, y)`` for the binary edge ``parent → child``.

        **Labels:** ``y = 1`` iff the article's gold topics intersect ``subtree(child)``;
        else ``0`` (same as :meth:`gold_positive_edge` on ``child``).

        **Rows (``restrict_to_parent_subtree``):**

        - ``True`` (**local** / parent-filtered): include only articles whose gold intersects
          ``subtree(parent)``. Deeper edges then train on a smaller ``n`` than the full split.
          Articles with **no** gold labels are excluded (they never match any subtree).
        - ``False`` (**global** / legacy): include every article in the split; negatives for
          ``child`` can be documents whose gold never touches ``parent`` at all.
        """
        ids = self._train_ids if split == "train" else self._test_ids
        art = self._articles.set_index("id")
        X: List[Any] = []
        y: List[int] = []
        for aid in ids:
            if aid not in art.index:
                continue
            if restrict_to_parent_subtree and not self.gold_positive_edge(aid, parent):
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


def make_multilabel_binary_pool_data_from_files(
    *,
    base_path: Optional[str] = None,
    articles_csv: str,
    train_ids_csv: str,
    test_ids_csv: Optional[str] = None,
    news_topics_csv: Optional[str] = None,
    topics_csv: Optional[str] = None,
    traversal_root: str = "Root",
    article_column: str = "article",
) -> MultilabelBinaryPoolData:
    """
    Build :class:`MultilabelBinaryPoolData` for **large** pools (e.g. ~400k train rows).

    Same labeling logic as :func:`make_multilabel_binary_pool_data`, but you pass explicit paths:

    - ``articles_csv`` — must contain at least ``id`` and ``article`` (or ``article_column``).
    - ``train_ids_csv`` — one column ``id`` (or first column) listing training article ids.
    - ``test_ids_csv`` — optional; omit or pass ``None`` for **train-only** (empty test split).
    - ``news_topics_csv`` — defaults to ``<base_path>/news_topics.csv`` (chunked read).
    - ``topics_csv`` — defaults to ``<base_path>/topics.csv``.

    Only rows whose ``id`` appears in train ∪ test are kept in the in-memory articles table
    (saves RAM when the articles file is a superset).

    Example::

        pool = make_multilabel_binary_pool_data_from_files(
            base_path=\"/data/rcv1\",
            articles_csv=\"/data/rcv1/articles_full.csv\",
            train_ids_csv=\"/data/rcv1/train_ids.csv\",
            test_ids_csv=\"/data/rcv1/test_ids.csv\",  # or None
        )
    """
    root = base_path or _base_dir()
    tpath = topics_csv or os.path.join(root, "topics.csv")
    tree = load_topic_tree(tpath, traversal_root=traversal_root)
    nt_path = news_topics_csv or os.path.join(root, "news_topics.csv")

    train_ids = load_pool_ids(train_ids_csv)
    test_ids = load_pool_ids(test_ids_csv) if test_ids_csv else []
    pool_ids = set(train_ids) | set(test_ids)
    labels = load_article_label_sets(nt_path, pool_ids)

    usecols = [article_column, "id"]
    articles = pd.read_csv(articles_csv, usecols=lambda c: c in set(usecols))
    articles["id"] = articles["id"].astype(int)
    articles = articles.drop_duplicates(subset=["id"], keep="first")
    articles = articles[articles["id"].isin(pool_ids)].copy()
    missing = pool_ids - set(articles["id"].tolist())
    if missing:
        raise ValueError(
            f"articles_csv is missing {len(missing)} ids present in train/test "
            f"(e.g. {next(iter(missing))})"
        )

    return MultilabelBinaryPoolData(
        tree,
        labels,
        articles,
        train_ids,
        test_ids,
    )


def prepare_news_split_corpus_bundle(
    *,
    base_path: str,
    output_dir: str,
    news_csv: Optional[str] = None,
    news_test_csv: Optional[str] = None,
    article_column: str = "article",
    merged_filename: str = "articles_merged.csv",
    train_ids_filename: str = "full_corpus_train_ids.csv",
    test_ids_filename: str = "full_corpus_test_ids.csv",
    chunksize: int = 200_000,
    exclude_test_ids_already_in_train: bool = True,
    pool_train_ids: Literal["union", "news_only"] = "union",
) -> Dict[str, Any]:
    """
    Build training-ready artifacts from the standard **split** layout (``news.csv`` + optional
    ``news_test.csv``), each with columns ``id`` and ``article`` (or ``article_column``).

    Writes under ``output_dir``:

    - **Merged articles** CSV: all training rows first, then test rows (streaming; low peak RAM).
    - **Id CSVs** for :func:`make_multilabel_binary_pool_data_from_files`:

      - **Default** (``pool_train_ids="union"``): ``full_corpus_train_ids.csv`` lists **every**
        unique id present in the merged file (``news`` ∪ ``news_test`` after overlap handling), and
        no separate test-id file is written — so edge training uses **both** splits.
      - **Legacy** (``pool_train_ids="news_only"``): train-id file = ids from ``news.csv`` only;
        test-id file = ids from ``news_test.csv`` (fit uses train split only; test split reserved
        for evaluation).

    If ``news_test_csv`` is omitted, looks for ``<base_path>/news_test.csv``; if that file is
    absent, only the training file is used (empty test split).

    When ``exclude_test_ids_already_in_train`` is True, rows whose ``id`` already appeared in
    ``news.csv`` are dropped from the merged output and from the test-id list (RCV1-style splits
    are usually disjoint).

    Returns a dict with absolute paths: ``articles_merged``, ``train_ids``, ``test_ids`` (path or
    ``None``), counts, etc.
    """
    root = os.path.abspath(base_path)
    out_abs = os.path.abspath(output_dir)
    os.makedirs(out_abs, exist_ok=True)

    n_path = news_csv or os.path.join(root, "news.csv")
    if not os.path.isfile(n_path):
        raise FileNotFoundError(f"Training articles CSV not found: {n_path}")

    if news_test_csv is not None:
        te_path = os.path.abspath(news_test_csv)
        if not os.path.isfile(te_path):
            raise FileNotFoundError(f"news_test_csv not found: {te_path}")
        use_test = True
    else:
        te_path = os.path.join(root, "news_test.csv")
        use_test = os.path.isfile(te_path)

    merged_path = os.path.join(out_abs, merged_filename)
    train_ids_path = os.path.join(out_abs, train_ids_filename)
    test_ids_path_default = os.path.join(out_abs, test_ids_filename)

    usecols = ["id", article_column]
    train_ids_accum: Set[int] = set()
    first_write = True
    n_train_rows = 0

    for chunk in pd.read_csv(
        n_path,
        usecols=lambda c: c in set(usecols),
        chunksize=chunksize,
    ):
        if article_column not in chunk.columns or "id" not in chunk.columns:
            raise ValueError(
                f"{n_path} must contain columns 'id' and {article_column!r}; got {list(chunk.columns)}"
            )
        chunk = chunk.copy()
        chunk["id"] = chunk["id"].astype(int)
        train_ids_accum.update(chunk["id"].tolist())
        chunk.to_csv(
            merged_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False
        n_train_rows += len(chunk)

    n_test_rows = 0
    test_ids_accum: Set[int] = set()
    if use_test:
        train_id_frozen = set(train_ids_accum)
        for chunk in pd.read_csv(
            te_path,
            usecols=lambda c: c in set(usecols),
            chunksize=chunksize,
        ):
            chunk = chunk.copy()
            chunk["id"] = chunk["id"].astype(int)
            if exclude_test_ids_already_in_train:
                chunk = chunk.loc[~chunk["id"].isin(train_id_frozen)]
            if chunk.empty:
                continue
            test_ids_accum.update(chunk["id"].tolist())
            chunk.to_csv(merged_path, mode="a", header=False, index=False)
            n_test_rows += len(chunk)

    out_test_ids_path: Optional[str] = None
    if not use_test:
        fit_ids = sorted(train_ids_accum)
        pd.DataFrame({"id": fit_ids}).to_csv(train_ids_path, index=False)
        n_fit_ids = len(fit_ids)
    elif pool_train_ids == "union":
        fit_ids = sorted(train_ids_accum | test_ids_accum)
        pd.DataFrame({"id": fit_ids}).to_csv(train_ids_path, index=False)
        n_fit_ids = len(fit_ids)
    else:
        pd.DataFrame({"id": sorted(train_ids_accum)}).to_csv(train_ids_path, index=False)
        pd.DataFrame({"id": sorted(test_ids_accum)}).to_csv(test_ids_path_default, index=False)
        out_test_ids_path = test_ids_path_default
        n_fit_ids = len(train_ids_accum)

    return {
        "base_path": root,
        "articles_merged": merged_path,
        "train_ids": train_ids_path,
        "test_ids": out_test_ids_path,
        "pool_train_ids": pool_train_ids,
        "n_train_ids": n_fit_ids,
        "n_news_ids": len(train_ids_accum),
        "n_test_ids": len(test_ids_accum),
        "n_train_article_rows": n_train_rows,
        "n_test_article_rows": n_test_rows,
        "news_csv": os.path.abspath(n_path),
        "news_test_csv": os.path.abspath(te_path) if use_test else None,
    }


def validate_router_training_inputs(
    *,
    base_path: Optional[str] = None,
    articles_csv: str,
    train_ids_csv: str,
    test_ids_csv: Optional[str] = None,
    news_topics_csv: Optional[str] = None,
    topics_csv: Optional[str] = None,
    article_column: str = "article",
    traversal_root: str = "Root",
    articles_id_chunksize: int = 400_000,
) -> Tuple[bool, List[str]]:
    """
    Check files and coverage before ``make_multilabel_binary_pool_data_from_files`` / CLI ``train``.

    Returns ``(ok, lines)`` where ``lines`` are human-readable messages (errors and warnings).
    Scans ``articles_csv`` **id** column in chunks (full pass) to verify every pool id exists.
    Loads labels via chunked ``news_topics`` read (same as training).
    """
    lines: List[str] = []
    root = base_path or _base_dir()
    tpath = topics_csv or os.path.join(root, "topics.csv")
    nt_path = news_topics_csv or os.path.join(root, "news_topics.csv")

    if not os.path.isfile(tpath):
        lines.append(f"ERROR: topics.csv not found: {tpath}")
        return False, lines
    if not os.path.isfile(nt_path):
        lines.append(f"ERROR: news_topics.csv not found: {nt_path}")
        return False, lines
    if not os.path.isfile(articles_csv):
        lines.append(f"ERROR: articles_csv not found: {articles_csv}")
        return False, lines
    if not os.path.isfile(train_ids_csv):
        lines.append(f"ERROR: train_ids_csv not found: {train_ids_csv}")
        return False, lines
    if test_ids_csv and not os.path.isfile(test_ids_csv):
        lines.append(f"ERROR: test_ids_csv not found: {test_ids_csv}")
        return False, lines

    try:
        tree = load_topic_tree(tpath, traversal_root=traversal_root)
        lines.append(f"OK: loaded TopicTree (root={tree.traversal_root!r})")
    except Exception as e:
        lines.append(f"ERROR: failed to load topics: {e}")
        return False, lines

    try:
        header = pd.read_csv(articles_csv, nrows=0)
    except Exception as e:
        lines.append(f"ERROR: cannot read articles_csv header: {e}")
        return False, lines
    if "id" not in header.columns:
        lines.append("ERROR: articles_csv must have column 'id'")
        return False, lines
    if article_column not in header.columns:
        lines.append(
            f"ERROR: articles_csv missing text column {article_column!r}; have {list(header.columns)}"
        )
        return False, lines

    train_ids = load_pool_ids(train_ids_csv)
    test_ids = load_pool_ids(test_ids_csv) if test_ids_csv else []
    if not train_ids:
        lines.append("ERROR: train_ids_csv produced no ids")
        return False, lines
    pool_ids: Set[int] = set(int(x) for x in train_ids) | set(int(x) for x in test_ids)
    lines.append(
        f"OK: train_ids={len(train_ids)}  test_ids={len(test_ids)}  pool_union={len(pool_ids)}"
    )

    article_ids: Set[int] = set()
    try:
        for chunk in pd.read_csv(
            articles_csv,
            usecols=["id"],
            chunksize=articles_id_chunksize,
        ):
            article_ids.update(chunk["id"].astype(int).tolist())
    except Exception as e:
        lines.append(f"ERROR: scanning articles_csv ids: {e}")
        return False, lines

    missing_art = pool_ids - article_ids
    if missing_art:
        sample = sorted(missing_art)[:5]
        lines.append(
            f"ERROR: {len(missing_art)} pool ids missing from articles_csv (e.g. {sample})"
        )
        return False, lines
    lines.append(f"OK: all {len(pool_ids)} pool ids appear in articles_csv")

    labels = load_article_label_sets(nt_path, pool_ids)
    empty_lab = [i for i in pool_ids if not labels.get(i)]
    if empty_lab:
        lines.append(
            f"WARNING: {len(empty_lab)} pool ids have no rows in news_topics (empty gold); "
            f"e.g. {empty_lab[:5]}"
        )
    else:
        lines.append(f"OK: news_topics has at least one label for all {len(pool_ids)} pool ids")

    dup_note = ""
    try:
        id_series = pd.read_csv(articles_csv, usecols=["id"])["id"].astype(int)
        nd = int(id_series.duplicated().sum())
        if nd:
            dup_note = f" (note: {nd} duplicate id rows; training uses first per id)"
    except Exception:
        pass
    lines.append(f"Validation finished.{dup_note}")
    ok = not missing_art
    return ok, lines


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
