"""
RCV1 topic taxonomy as a tree: adjacency from topics.csv.

Used to count branching nodes (local classifiers), enumerate specs, and navigate paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def read_topics_csv(path: str) -> pd.DataFrame:
    """Same manual parse as ProjectD.py — commas may appear inside description."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 4:
            parts = parts + [""] * (4 - len(parts))
        rows.append(
            {
                "id_cat": parts[0].strip(),
                "parent": parts[1].strip(),
                "child": parts[2].strip(),
                "description": ",".join(parts[3:]).strip(),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BinaryEdgeSpec:
    """One binary classifier for edge parent -> child (child subtree membership)."""

    parent: str
    child: str
    depth: int  # depth of parent from traversal_root

    @property
    def key(self) -> Tuple[str, str]:
        return (self.parent, self.child)


@dataclass(frozen=True)
class LocalClassifierSpec:
    """One trainable routing step: choose among child_labels of node_id."""

    node_id: str
    child_labels: Tuple[str, ...]
    n_classes: int
    depth: int  # depth from traversal_root (Root = 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_classes", len(self.child_labels))


@dataclass
class TopicTree:
    children: Dict[str, List[str]]
    parent: Dict[str, str]
    traversal_root: str = "Root"

    def all_nodes(self) -> Set[str]:
        out: Set[str] = set(self.children.keys())
        for chs in self.children.values():
            out.update(chs)
        return out

    def leaf_nodes(self) -> Set[str]:
        all_n = self.all_nodes()
        parents_with_edges = {p for p, chs in self.children.items() if chs}
        return all_n - parents_with_edges

    def branching_nodes(self) -> List[str]:
        """BFS from traversal_root; only nodes with >= 2 children."""
        order: List[str] = []
        seen: Set[str] = set()
        q: deque[str] = deque([self.traversal_root])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            chs = self.children.get(u, [])
            if len(chs) >= 2:
                order.append(u)
            for c in chs:
                if c not in seen:
                    q.append(c)
        return order

    def local_classifier_specs(self) -> List[LocalClassifierSpec]:
        """One spec per branching node; BFS order; depth from traversal_root."""
        specs: List[LocalClassifierSpec] = []
        depth_map: Dict[str, int] = {self.traversal_root: 0}
        q: deque[str] = deque([self.traversal_root])
        while q:
            u = q.popleft()
            du = depth_map[u]
            chs = self.children.get(u, [])
            for c in chs:
                if c not in depth_map:
                    depth_map[c] = du + 1
                    q.append(c)
            if len(chs) >= 2:
                specs.append(
                    LocalClassifierSpec(
                        node_id=u,
                        child_labels=tuple(chs),
                        n_classes=len(chs),
                        depth=du,
                    )
                )
        return specs

    def binary_edge_specs(self) -> List[BinaryEdgeSpec]:
        """One spec per directed edge (parent -> child) where parent has >= 2 children."""
        return binary_edge_specs(self)

    def path_from_root_to(self, code: str) -> List[str]:
        """Single path traversal_root -> ... -> code using parent pointers (code must be in tree)."""
        if code == self.traversal_root:
            return [self.traversal_root]
        up: List[str] = []
        cur: Optional[str] = code
        while cur is not None:
            up.append(cur)
            if cur == self.traversal_root:
                break
            cur = self.parent.get(cur)
        up.reverse()
        return up

    def subtree_nodes(self, subroot: str) -> Set[str]:
        """All nodes reachable from subroot (including subroot)."""
        out: Set[str] = set()
        stack = [subroot]
        while stack:
            u = stack.pop()
            if u in out:
                continue
            out.add(u)
            for c in self.children.get(u, []):
                stack.append(c)
        return out

    def subtree_topic_codes(self, subroot: str) -> Set[str]:
        """All topic codes in the subtree rooted at ``subroot`` (including ``subroot``). Same as ``subtree_nodes``."""
        return self.subtree_nodes(subroot)


def binary_edge_specs(tree: TopicTree) -> List[BinaryEdgeSpec]:
    """
    One spec per directed edge (parent -> child) where parent has >= 2 children.
    Same as ``tree.binary_edge_specs()``; use this import if the kernel cached an old ``TopicTree``.
    """
    out: List[BinaryEdgeSpec] = []
    depth_map: Dict[str, int] = {tree.traversal_root: 0}
    q: deque[str] = deque([tree.traversal_root])
    while q:
        u = q.popleft()
        du = depth_map[u]
        chs = tree.children.get(u, [])
        for c in chs:
            if c not in depth_map:
                depth_map[c] = du + 1
                q.append(c)
        if len(chs) >= 2:
            for c in chs:
                out.append(BinaryEdgeSpec(parent=u, child=c, depth=du))
    return out


def _normalize_skip_parent(p: str) -> bool:
    pl = p.lower()
    return p == "" or pl == "none" or pl == "nan"


def load_topic_tree(topics_path: str, traversal_root: str = "Root") -> TopicTree:
    topics = read_topics_csv(topics_path)
    children: Dict[str, List[str]] = {}
    parent: Dict[str, str] = {}

    for _, row in topics.iterrows():
        p, c = str(row["parent"]).strip(), str(row["child"]).strip()
        if _normalize_skip_parent(p) or c == "" or c.lower() == "nan":
            continue
        children.setdefault(p, []).append(c)
        if c not in parent:
            parent[c] = p

    for p in children:
        children[p] = sorted(set(children[p]))

    return TopicTree(children=children, parent=parent, traversal_root=traversal_root)


def summary(tree: TopicTree) -> Dict[str, object]:
    all_n = tree.all_nodes()
    leaves = tree.leaf_nodes()
    branch = [u for u in tree.all_nodes() if len(tree.children.get(u, [])) >= 2]
    specs = tree.local_classifier_specs()
    max_branch = max((len(tree.children[u]) for u in branch), default=0)

    def max_depth_from_root() -> int:
        best = 0
        for node in all_n:
            path = tree.path_from_root_to(node)
            best = max(best, len(path) - 1)
        return best

    n_binary_edges = sum(len(tree.children[u]) for u in branch)

    return {
        "traversal_root": tree.traversal_root,
        "n_nodes": len(all_n),
        "n_leaves": len(leaves),
        "n_branching_nodes": len(branch),
        "n_local_classifiers": len(specs),
        "n_binary_edge_classifiers": n_binary_edges,
        "max_branching_factor": max_branch,
        "max_depth_from_root": max_depth_from_root(),
    }


def filter_children_for_subtree(
    children: Dict[str, List[str]], allowed: Set[str]
) -> Dict[str, List[str]]:
    return {
        p: [c for c in chs if c in allowed]
        for p, chs in children.items()
        if p in allowed
    }


def subtree_under(tree: TopicTree, subroot: str) -> TopicTree:
    """Restrict tree to nodes under subroot (e.g. CCAT only). traversal_root becomes subroot."""
    nodes = tree.subtree_nodes(subroot)
    if subroot not in nodes:
        raise ValueError(f"subroot {subroot!r} not in tree")
    ch = filter_children_for_subtree(tree.children, nodes)
    parent: Dict[str, str] = {}
    for p, cs in ch.items():
        for c in cs:
            parent[c] = p
    return TopicTree(children=ch, parent=parent, traversal_root=subroot)
