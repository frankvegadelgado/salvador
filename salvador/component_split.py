"""Strategy c0: amplify the number of connected components before the maximal matching.

Every vertex ``u`` of ``G`` starts as the copy ``(u, 0)`` of the split graph
``H`` and every edge ``(u, v)`` as ``((u, 0), (v, 0))``. *Splitting* an edge
``((u, i), (v, j))`` of ``H`` keeps that edge and moves all the other edges of
``(u, i)`` to a fresh copy ``(u, i')`` (``i'`` = next unused index of ``u``),
and those of ``(v, j)`` to a fresh copy ``(v, j')``; the kept edge becomes an
isolated ``K2`` of ``H``. When the split edge is a bridge of ``H`` the
component falls into the ``K2`` and the two sides, so the number of connected
components grows by up to 2, and the chain continues on the new copies, e.g.
``((u, 1), (w, 0))`` may be another bridge, which is then kept and the other
neighbours go to ``(u, 2)`` and ``(w, 1)``.

Splitting goes on until ``H`` has ``r' >= target`` connected components, where
``target = ceil(|C1| / 2) = |M|``, ``M`` the NetworkX minimum maximal matching
of ``G`` (strategy c1) and ``C1`` its endpoints. Each round splits the bridges
of ``H`` first (a bridge stays a bridge after other bridges are split, so one
bridge pass per round finds all of them); when no bridge is left and
``r' < target``, half of the deficit is closed by splitting non-``K2`` edges of
the NetworkX minimum maximal matching of ``H`` (each such split adds at least
one component and may create new bridges for the next round). The target is
always reached: while ``r' < |M| <= m`` some component is not a ``K2``.

Then the NetworkX minimum maximal matching ``M_H`` of ``H`` is taken (every
isolated ``K2`` is forced into it) and decoded to vertices of ``G``. The
matched edges ``((a, i), (b, j))`` are processed in increasing order of
``(min(i, j), max(i, j))``:

* if ``a`` (or ``b``) was already selected through some copy, the edge is
  covered and nothing is added;
* otherwise, if ``i != j``, the endpoint with the smaller index is selected
  (rule ``"low"``) or the endpoint with the larger index (rule ``"high"``);
* if ``i == j`` the edge is saved in a residual graph ``R`` (in ``H`` these
  edges form a perfect matching of their endpoints).

The residual edges still uncovered by the selected vertices are covered
optimally when ``R`` is bipartite (Koenig: a maximum matching gives a minimum
vertex cover), and otherwise greedily by number of occurrences (the vertex in
most uncovered residual edges first). The decoded set is not a cover of ``G``
in general: :mod:`salvador.algorithm` repairs and prunes it with
``prune_redundant_vertices``.

Only NetworkX's ``nx.approximation.min_maximal_matching`` is used as the
maximal-matching subroutine, the same one as strategy c1.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

#: Selection rules for a matched edge ((a, i), (b, j)) with i != j and a, b both unselected.
RULES: tuple[str, ...] = ("low", "high")


def _matching(H: nx.Graph) -> set[tuple[Any, Any]]:
    """NetworkX minimum maximal matching (the subroutine of strategy c1)."""
    return nx.approximation.min_maximal_matching(H)


def build_split_graph(G: nx.Graph, info: dict | None = None) -> nx.Graph:
    """Split graph ``H`` of ``G`` (nodes ``(u, i)``) with at least ``ceil(|C1| / 2)`` components.

    ``G`` must have no self-loops. ``info`` receives ``target``, ``r`` (components
    of ``G`` with an edge), ``r_prime`` (components of ``H``), ``bridge_splits``,
    ``matching_splits``, ``rounds`` and ``max_index`` (largest copy index).
    """
    pos = {u: k for k, u in enumerate(G)}
    m1 = _matching(G)
    target = len(m1)  # = |C1| / 2, C1 the endpoints of the c1 matching
    H = nx.Graph()
    H.add_nodes_from((u, 0) for u in G)
    H.add_edges_from(((u, 0), (v, 0)) for u, v in G.edges())
    H.remove_nodes_from([x for x in list(H) if H.degree(x) == 0])
    next_index = {u: 1 for u in G}
    r = nx.number_connected_components(H)
    r0 = r
    stats = {"bridge_splits": 0, "matching_splits": 0, "rounds": 0}

    def key(x: tuple[Any, int]) -> tuple[int, int]:
        return (x[1], pos[x[0]])

    moved: dict[tuple[Any, int], tuple[Any, int]] = {}  # copy -> the copy its other edges went to

    def chain(x: tuple[Any, int]):
        while True:
            yield x
            if x not in moved:
                return
            x = moved[x]

    def resolve(x: tuple[Any, int], y: tuple[Any, int]):
        """Current copies holding the edge listed as ``(x, y)`` before some splits of this pass."""
        if H.has_edge(x, y):
            return x, y
        for cx in chain(x):
            for cy in chain(y):
                if H.has_edge(cx, cy):
                    return cx, cy
        raise AssertionError("edge lost while splitting")  # pragma: no cover

    def move_rest(x: tuple[Any, int], y: tuple[Any, int]) -> bool:
        """Move every edge of ``x`` except ``xy`` to a fresh copy of ``x``; True if one was moved."""
        others = [z for z in H[x] if z != y]
        if not others:
            return False
        u = x[0]
        x2 = (u, next_index[u])
        next_index[u] += 1
        H.remove_edges_from((x, z) for z in others)
        H.add_edges_from((x2, z) for z in others)
        moved[x] = x2
        return True

    while r < target:
        stats["rounds"] += 1
        # Bridges first: an exact +1 per endpoint that still has other edges.
        for x, y in list(nx.bridges(H)):
            if r >= target:
                break
            x, y = resolve(x, y)
            if H.degree(x) == 1 and H.degree(y) == 1:
                continue  # already an isolated K2
            r += int(move_rest(x, y)) + int(move_rest(y, x))
            stats["bridge_splits"] += 1
        if r >= target:
            break
        # No bridge left: split non-K2 matching edges, half of the deficit per round.
        quota = max(1, (target - r + 1) // 2)
        edges = sorted((tuple(sorted(e, key=key)) for e in _matching(H)), key=lambda e: (key(e[0]), key(e[1])))
        done = 0
        for x, y in edges:
            if done >= quota:
                break
            x, y = resolve(x, y)
            if H.degree(x) == 1 and H.degree(y) == 1:
                continue
            move_rest(x, y)
            move_rest(y, x)
            done += 1
        stats["matching_splits"] += done
        r = nx.number_connected_components(H)

    if info is not None:
        info.update({"target": target, "r": r0, "r_prime": r,
                     "max_index": max((i for _, i in H), default=0),
                     "h_nodes": H.number_of_nodes(), "h_edges": H.number_of_edges(), **stats})
    return H


def _residual_cover(edges: list[tuple[Any, Any]], pos: dict[Any, int], info: dict) -> set[Any]:
    """Vertex cover of the residual edges: Koenig (optimal) when bipartite, else by occurrences."""
    if not edges:
        info.update({"residual_exact": True})
        return set()
    R = nx.Graph()
    R.add_edges_from(edges)
    if nx.is_bipartite(R):
        cover: set[Any] = set()
        for comp in nx.connected_components(R):
            S = R.subgraph(comp)
            top = {v for v, c in nx.bipartite.color(S).items() if c == 0}
            M = nx.bipartite.hopcroft_karp_matching(S, top_nodes=top)
            cover |= nx.bipartite.to_vertex_cover(S, M, top_nodes=top)
        info.update({"residual_exact": True})
        return cover
    # Greedy by number of occurrences, with a bucket queue (O(|R|)).
    adj = {v: set(R[v]) for v in R}
    deg = {v: len(a) for v, a in adj.items()}
    maxd = max(deg.values())
    buckets: list[deque] = [deque() for _ in range(maxd + 1)]
    for v in sorted(adj, key=lambda v: pos[v]):
        buckets[deg[v]].append(v)
    cover = set()
    for d in range(maxd, 0, -1):
        while buckets[d]:
            v = buckets[d].popleft()
            if v in cover or deg[v] != d:
                continue
            cover.add(v)
            for w in adj[v]:
                if w not in cover:
                    adj[w].discard(v)
                    deg[w] -= 1
                    buckets[deg[w]].append(w)
            # deg[w] can only decrease, so lower buckets are visited later
    info.update({"residual_exact": False})
    return cover


def decode(G: nx.Graph, H: nx.Graph, rule: str = "low", info: dict | None = None) -> set[Any]:
    """Decode the minimum maximal matching of ``H`` to a vertex set of ``G`` with ``rule`` ("low"/"high")."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    pos = {u: k for k, u in enumerate(G)}
    matched = []
    for x, y in _matching(H):
        (a, i), (b, j) = x, y
        if (j, pos[b]) < (i, pos[a]):
            (a, i), (b, j) = (b, j), (a, i)
        matched.append(((a, i), (b, j)))
    matched.sort(key=lambda e: (e[0][1], e[1][1], pos[e[0][0]], pos[e[1][0]]))
    selected: set[Any] = set()
    residual: list[tuple[Any, Any]] = []
    reused = 0
    for (a, i), (b, j) in matched:  # i <= j after the swap above
        if i == j:
            residual.append((a, b))
            continue
        if a in selected or b in selected:
            reused += 1
            continue
        selected.add(a if rule == "low" else b)
    open_residual = [(a, b) for a, b in residual if a not in selected and b not in selected]
    local: dict = {}
    selected |= _residual_cover(open_residual, pos, local)
    if info is not None:
        info.update({"matching_h": len(matched), "ties": len(residual), "residual_open": len(open_residual),
                     "reused": reused, "raw_size": len(selected), **local})
    return selected


def component_split_covers(G: nx.Graph, info: dict | None = None) -> dict[str, set[Any]]:
    """Decoded (unrepaired) vertex sets of c0 for every rule in :data:`RULES`.

    ``G`` must have no self-loops or isolated vertices. ``info`` receives the
    split statistics of :func:`build_split_graph` and, per rule, the decoding
    statistics of :func:`decode` under the key ``rule``.
    """
    split_info: dict = {}
    H = build_split_graph(G, split_info)
    out: dict[str, set[Any]] = {}
    for rule in RULES:
        rule_info: dict = {}
        out[rule] = decode(G, H, rule, rule_info)
        if info is not None:
            info[rule] = rule_info
    if info is not None:
        info.update(split_info)
    return out
