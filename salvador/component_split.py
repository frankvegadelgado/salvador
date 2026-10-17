"""Strategy c1: amplify the number of connected components before the maximal matching (linear time).

Every vertex ``u`` of ``G`` has at most two copies in the split graph ``H``:
``(u, 0)`` and ``(u, 1)``. At the start every edge ``(u, v)`` is
``((u, 0), (v, 0))``. *Splitting* an edge whose two endpoints both have index
0 keeps that edge as an isolated ``K2`` of ``H`` and moves all the other edges
of ``u`` to ``(u, 1)`` and those of ``v`` to ``(v, 1)``. Only edges with index
0 at both endpoints are split, and an edge is split at most once, so no copy
index larger than 1 is ever needed. No bridge has to be found: any edge with
both endpoints still at index 0 can be split.

The edges split are those of ``M``, the NetworkX minimum maximal matching of
``G`` (the matching of c2), in NetworkX's order. They are vertex-disjoint, so
when an edge of ``M`` comes up both of its endpoints are still at index 0.
Splitting the first ``k`` edges ``F_k`` of ``M`` leaves ``H`` with

    count(k) = k + c(G - F_k)

components with an edge, where ``c`` counts the components of a graph that
have at least one edge (the split edges are ``k`` isolated ``K2``; all other
edges of ``G`` stay together through the copies ``(u, 1)``). ``count`` never
decreases with ``k`` and grows by at most 2 per split; ``count(0) = r``, the
number of components of ``G`` with an edge, and ``r <= |M|`` since every such
component contains an edge of ``M``. Splitting stops at the first ``k`` where
``count(k)`` reaches the largest value that does not exceed the target
``|M|``, so ``r' = count(k) <= |M| <= tau(G)`` and ``r' >= |M| - 1``. All the
values ``count(0..|M|)`` are computed in one pass, backwards, with a
union-find structure: ``G - F_{|M|}`` first, then the edges of ``M`` are added
back one by one. This costs ``O((n + m) alpha(n))``.

The NetworkX minimum maximal matching ``M_H`` of ``H`` is then decoded to
vertices of ``G``, the matched edges ``((a, i), (b, j))`` in increasing order
of ``(min(i, j), max(i, j))``:

* if ``a`` or ``b`` is already selected (through either copy), nothing is added;
* if ``i != j``, the endpoint is chosen by the rule: ``low`` (index 0),
  ``high`` (index 1), ``deg_low`` / ``deg_high`` (larger degree in ``G``, ties
  by index 0 / index 1);
* if ``i == j``, the endpoint of larger degree in ``G`` is selected (ties by
  position). With only the indices 0 and 1 no residual graph is needed.

The decoded set is completed to a vertex cover of ``G`` by a bucket-queue
greedy on the edges it leaves uncovered (largest uncovered degree first, in
linear time), and :mod:`salvador.algorithm` prunes it. Every step is linear,
so c1 runs in ``O((n + m) alpha(n))`` time.

Only NetworkX's ``nx.approximation.min_maximal_matching`` is used as the
maximal-matching subroutine, for ``M`` and for ``M_H``.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

#: Selection rules for a matched edge ((a, i), (b, j)) with i != j and a, b both unselected.
RULES: tuple[str, ...] = ("low", "high", "deg_low", "deg_high")

#: The rules c1 runs; the smallest pruned cover is kept.
DEFAULT_VARIANTS: tuple[str, ...] = ("deg_high", "high", "deg_low")


def _matching(H: nx.Graph) -> list[tuple[Any, Any]]:
    """NetworkX minimum maximal matching (the subroutine of strategy c2), as a list."""
    return list(nx.approximation.min_maximal_matching(H))


class _UnionFind:
    def __init__(self, items) -> None:
        self.parent = {x: x for x in items}
        self.size = dict.fromkeys(self.parent, 1)
        self.has_edge = dict.fromkeys(self.parent, False)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y) -> int:
        """Join the trees of ``x`` and ``y`` with the edge ``(x, y)``; return the change in the number
        of trees that contain an edge."""
        a, b = self.find(x), self.find(y)
        if a == b:
            return 0  # a tree with two or more vertices already has an edge
        before = self.has_edge[a] + self.has_edge[b]
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.has_edge[a] = True
        return 1 - before


def split_counts(G: nx.Graph, M: list[tuple[Any, Any]]) -> list[int]:
    """``count(k) = k + c(G - F_k)`` for ``k = 0..|M|`` (``F_k`` = first ``k`` edges of ``M``), by one
    backward union-find pass."""
    in_m = {frozenset(e) for e in M}
    uf = _UnionFind(G.nodes())
    components = 0
    for u, v in G.edges():
        if frozenset((u, v)) not in in_m:
            components += uf.union(u, v)
    counts = [0] * (len(M) + 1)
    counts[len(M)] = len(M) + components
    for k in range(len(M) - 1, -1, -1):
        u, v = M[k]
        components += uf.union(u, v)
        counts[k] = k + components
    return counts


def build_split_graph(G: nx.Graph, info: dict | None = None) -> nx.Graph:
    """Split graph ``H`` of ``G`` (nodes ``(u, 0)`` and ``(u, 1)``) with ``r' <= |M|`` components.

    ``G`` must have no self-loops. ``info`` receives ``target`` (``|M|``), ``r``
    (components of ``G`` with an edge), ``r_prime`` (components of ``H`` with an
    edge), ``splits`` (edges of ``M`` split), ``exact`` (``r' == |M|``),
    ``h_nodes`` and ``h_edges``.
    """
    pos = {u: k for k, u in enumerate(G)}
    # each edge with its earlier endpoint first, the edges by that endpoint (a counting sort: the
    # edges of M are vertex-disjoint, so the positions are distinct)
    slots: list = [None] * len(pos)
    for a, b in _matching(G):
        if pos[b] < pos[a]:
            a, b = b, a
        slots[pos[a]] = (a, b)
    M = [e for e in slots if e is not None]
    target = len(M)
    counts = split_counts(G, M)
    best = max(c for c in counts if c <= target)  # counts[0] = r <= |M|, so this exists
    k = counts.index(best)  # first k reaching it (counts never decrease)
    split_edges = {frozenset(e) for e in M[:k]}
    split_vertices = {x for e in M[:k] for x in e}
    H = nx.Graph()
    for u, v in G.edges():
        if frozenset((u, v)) in split_edges:
            H.add_edge((u, 0), (v, 0))
        else:
            H.add_edge((u, 1 if u in split_vertices else 0), (v, 1 if v in split_vertices else 0))
    if info is not None:
        info.update({"target": target, "r": counts[0], "r_prime": best, "splits": k,
                     "exact": best == target, "h_nodes": H.number_of_nodes(), "h_edges": H.number_of_edges()})
    return H


def decode(G: nx.Graph, H: nx.Graph, rule: str = "deg_high", info: dict | None = None,
           matching: list | None = None) -> set[Any]:
    """Decode the NetworkX minimum maximal matching of ``H`` to a set of vertices of ``G`` (not yet a
    cover of ``G`` in general; see :func:`complete_cover`). ``info`` receives ``matching_h`` (edges of
    the matching of ``H``), ``ties`` (matched edges with equal indices) and ``raw_size``."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    pos = {u: k for k, u in enumerate(G)}
    deg = G.degree
    # matched edges ordered by (i, j) with i <= j, then by the position of the first endpoint: a
    # counting sort, since each copy is matched once and the first endpoints of a group are distinct
    groups: dict[tuple[int, int], list] = {(0, 0): [None] * len(pos), (0, 1): [None] * len(pos),
                                           (1, 1): [None] * len(pos)}
    for x, y in (matching if matching is not None else _matching(H)):
        (a, i), (b, j) = x, y
        if (j, pos[b]) < (i, pos[a]):
            (a, i), (b, j) = (b, j), (a, i)
        groups[(i, j)][pos[a]] = ((a, i), (b, j))
    matched = [e for g in ((0, 0), (0, 1), (1, 1)) for e in groups[g] if e is not None]
    selected: set[Any] = set()
    ties = 0
    for (a, i), (b, j) in matched:  # here i <= j
        if a in selected or b in selected:
            continue
        if i == j:
            ties += 1
            pick = a if (deg[a], -pos[a]) >= (deg[b], -pos[b]) else b
        elif rule == "low":
            pick = a
        elif rule == "high":
            pick = b
        elif deg[a] != deg[b]:
            pick = a if deg[a] > deg[b] else b
        else:
            pick = a if rule == "deg_low" else b
        selected.add(pick)
    if info is not None:
        info.update({"matching_h": len(matched), "ties": ties, "raw_size": len(selected)})
    return selected


def complete_cover(G: nx.Graph, raw: set[Any]) -> set[Any]:
    """``raw`` plus a bucket-queue greedy cover (largest uncovered degree first) of the edges of ``G``
    that ``raw`` leaves uncovered, in O(n + m)."""
    adj: dict[Any, set[Any]] = {}
    for u, v in G.edges():
        if u not in raw and v not in raw:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    cover = set(raw)
    if not adj:
        return cover
    deg = {v: len(nb) for v, nb in adj.items()}
    top = max(deg.values())
    buckets: list[list[Any]] = [[] for _ in range(top + 1)]
    for v in reversed([u for u in G if u in adj]):  # pop() takes the earliest position first
        buckets[deg[v]].append(v)
    d = top
    while d > 0:
        if not buckets[d]:
            d -= 1
            continue
        v = buckets[d].pop()
        if v in cover or deg[v] != d:
            continue  # stale entry
        cover.add(v)
        for w in adj[v]:
            if w not in cover:
                deg[w] -= 1
                buckets[deg[w]].append(w)
        deg[v] = 0
    return cover


def component_split_covers(G: nx.Graph, info: dict | None = None,
                           variants: tuple[str, ...] = DEFAULT_VARIANTS) -> dict[str, set[Any]]:
    """Every decoding rule of c1 on the split graph of ``G``, completed to a vertex cover of ``G``
    (not pruned). ``info`` receives the split statistics of :func:`build_split_graph` and, per rule,
    ``info[rule]`` with the decoding statistics."""
    split_info: dict = {}
    H = build_split_graph(G, split_info)
    MH = _matching(H)
    out: dict[str, set[Any]] = {}
    for rule in variants:
        dec_info: dict = {}
        out[rule] = complete_cover(G, decode(G, H, rule, dec_info, MH))
        if info is not None:
            info[rule] = dec_info
    if info is not None:
        info.update(split_info)
    return out
