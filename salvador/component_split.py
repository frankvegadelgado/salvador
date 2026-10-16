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

Repairs after car_c0_fix.py (default since 0.1.3). The first c0 violated
``|c0| <= 2 tau - r'`` on 47 of 286 exact instances and had ratio 50.5 on the
double star, for three reasons, each fixed here:

* overshoot -- a bridge split adds two components, so ``r'`` could exceed
  ``tau`` (path on five vertices: ``r' = 3 > tau = 2``) and the bound became
  infeasible even for an optimal cover; splitting now stops exactly at
  ``r' = |M| <= tau`` (``exact_stop``);
* leaf-for-hub choices -- the index rules can select a leaf instead of its
  hub; the default variants select the endpoint of larger degree in ``G``
  (``deg_high``, ties by larger index) and resolve the tied residual edges the
  same way (``gdeg``);
* repair -- the decoded set is now completed greedily by largest uncovered
  degree (:func:`complete_cover`) instead of adding one neighbour per uncovered
  edge.

:mod:`salvador.algorithm` then prunes the cover, improves it with the budgeted
(1,2)-swap local search and keeps c1 when c1 is smaller, so ``|c0| <= |c1| <=
2 tau - r`` (``r`` = components of ``G``) holds on every graph. The bound
``2 tau - r'`` itself is not guaranteed: it cannot be checked while running
(``tau`` is unknown) and it would imply a ratio of at most 1.5, so it is
measured, not assumed, by car/car_c0_fix.py.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

#: Selection rules for a matched edge ((a, i), (b, j)) with i != j and a, b both unselected:
#: smaller index, larger index, or larger degree in G with ties by smaller/larger index.
RULES: tuple[str, ...] = ("low", "high", "deg_low", "deg_high")

#: Residual rules for the matched edges with i == j: Koenig (optimal when the residual graph is
#: bipartite, else by occurrences), greedy by occurrences, or the endpoint of larger degree in G.
RESIDUALS: tuple[str, ...] = ("konig", "occ", "gdeg")

#: Variants of c0 since 0.1.3, (select, residual) by name, best first in car_c0_fix.py (21,479 exact
#: instances: fewest violations of 2 tau - r', most optimal covers, lowest mean ratio). Each decoded
#: set is completed greedily (:func:`complete_cover`) before pruning.
DEFAULT_VARIANTS: dict[str, tuple[str, str]] = {
    "deg_high/gdeg": ("deg_high", "gdeg"),
    "high/gdeg": ("high", "gdeg"),
    "deg_low/gdeg": ("deg_low", "gdeg"),
}

#: The two variants of the first c0 (index rule, Koenig residual, no greedy completion).
LEGACY_VARIANTS: dict[str, tuple[str, str]] = {"low": ("low", "konig"), "high": ("high", "konig")}


def _matching(H: nx.Graph) -> set[tuple[Any, Any]]:
    """NetworkX minimum maximal matching (the subroutine of strategy c1)."""
    return nx.approximation.min_maximal_matching(H)


def build_split_graph(G: nx.Graph, info: dict | None = None, exact_stop: bool = True) -> nx.Graph:
    """Split graph ``H`` of ``G`` (nodes ``(u, i)``) with at least ``ceil(|C1| / 2)`` components.

    With ``exact_stop`` (the default) splitting never overshoots: a split adds
    at most two components (the isolated ``K2`` and the two sides of a bridge),
    so a bridge whose endpoints both keep other edges is split only while the
    deficit is at least 2, matching edges are split at most ``deficit // 2`` per
    round, and a deficit of 1 is closed by a split that adds exactly one
    component (a pendant edge or an edge on a cycle). Since every component
    with an edge contains an edge of ``M``, ``r <= |M|``, so splitting ends with
    ``r' = |M| <= tau(G)``: ``2 tau - r' >= tau`` and the bound ``2 tau - r'`` is
    never infeasible (with overshoot, e.g. on the path on five vertices, ``r' = 3 >
    tau = 2``). ``exact_stop=False`` keeps the original behaviour.

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

    def interior(x: tuple[Any, int], y: tuple[Any, int]) -> bool:
        return H.degree(x) > 1 and H.degree(y) > 1

    def split_plus_one() -> bool:
        """Split one edge that adds exactly one component (pendant bridge or cycle edge)."""
        bridge_set = {frozenset(e) for e in nx.bridges(H)}
        pendant = None
        for x, y in sorted(H.edges(), key=lambda e: (key(e[0]), key(e[1]))):
            if H.degree(x) == 1 and H.degree(y) == 1:
                continue  # isolated K2
            if frozenset((x, y)) not in bridge_set or not interior(x, y):
                pendant = (x, y)
                break
        if pendant is None:  # pragma: no cover - every non-K2 component has such an edge
            return False
        move_rest(*pendant)
        move_rest(pendant[1], pendant[0])
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
            if exact_stop and target - r == 1 and interior(x, y):
                continue  # would add 2 components and overshoot
            r += int(move_rest(x, y)) + int(move_rest(y, x))
            stats["bridge_splits"] += 1
        if r >= target:
            break
        if exact_stop and target - r == 1:
            if split_plus_one():
                r += 1
                stats["matching_splits"] += 1
            continue
        # No bridge left: split non-K2 matching edges; each split adds 1 or 2 components.
        quota = max(1, (target - r) // 2) if exact_stop else max(1, (target - r + 1) // 2)
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


def _gdeg_cover(edges: list[tuple[Any, Any]], G: nx.Graph, pos: dict[Any, int], selected: set[Any]) -> set[Any]:
    """Each still uncovered residual edge takes its endpoint of larger degree in ``G`` (ties by position)."""
    cover: set[Any] = set()
    for a, b in edges:
        if a in selected or b in selected or a in cover or b in cover:
            continue
        cover.add(a if (G.degree(a), -pos[a]) >= (G.degree(b), -pos[b]) else b)
    return cover


def decode(G: nx.Graph, H: nx.Graph, rule: str = "deg_high", info: dict | None = None,
           residual: str = "gdeg") -> set[Any]:
    """Decode the minimum maximal matching of ``H`` to a vertex set of ``G``.

    ``rule`` (see :data:`RULES`) resolves a matched edge ``((a, i), (b, j))``
    with ``i != j`` whose endpoints are both unselected; ``residual`` (see
    :data:`RESIDUALS`) covers the matched edges with ``i == j``.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if residual not in RESIDUALS:
        raise ValueError(f"residual must be one of {RESIDUALS}")
    pos = {u: k for k, u in enumerate(G)}
    matched = []
    for x, y in _matching(H):
        (a, i), (b, j) = x, y
        if (j, pos[b]) < (i, pos[a]):
            (a, i), (b, j) = (b, j), (a, i)
        matched.append(((a, i), (b, j)))
    matched.sort(key=lambda e: (e[0][1], e[1][1], pos[e[0][0]], pos[e[1][0]]))
    selected: set[Any] = set()
    ties: list[tuple[Any, Any]] = []
    reused = 0
    for (a, i), (b, j) in matched:  # i <= j after the swap above
        if i == j:
            ties.append((a, b))
            continue
        if a in selected or b in selected:
            reused += 1
            continue
        if rule == "low":
            pick = a
        elif rule == "high":
            pick = b
        else:
            da, db = G.degree(a), G.degree(b)
            pick = (a if da > db else b) if da != db else (a if rule == "deg_low" else b)
        selected.add(pick)
    open_ties = [(a, b) for a, b in ties if a not in selected and b not in selected]
    local: dict = {}
    if residual == "konig":
        selected |= _residual_cover(open_ties, pos, local)
    elif residual == "occ":
        selected |= _occurrence_cover(open_ties, pos)
        local["residual_exact"] = not open_ties
    else:
        selected |= _gdeg_cover(open_ties, G, pos, selected)
        local["residual_exact"] = not open_ties
    if info is not None:
        info.update({"matching_h": len(matched), "ties": len(ties), "residual_open": len(open_ties),
                     "reused": reused, "raw_size": len(selected), **local})
    return selected


def _occurrence_cover(edges: list[tuple[Any, Any]], pos: dict[Any, int]) -> set[Any]:
    """Greedy cover by largest current number of occurrences (lazy heap), ties by position."""
    import heapq

    adj: dict[Any, set[Any]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    heap = [(-len(nb), pos[v], v) for v, nb in adj.items()]
    heapq.heapify(heap)
    cover: set[Any] = set()
    while heap:
        d, _, v = heapq.heappop(heap)
        cur = len(adj[v])
        if v in cover or cur == 0:
            continue
        if -d != cur:
            heapq.heappush(heap, (-cur, pos[v], v))
            continue
        cover.add(v)
        for w in adj[v]:
            adj[w].discard(v)
            heapq.heappush(heap, (-len(adj[w]), pos[w], w))
        adj[v] = set()
    return cover


def complete_cover(G: nx.Graph, raw: set[Any]) -> set[Any]:
    """``raw`` plus a greedy cover (largest uncovered degree first) of the edges it leaves uncovered."""
    pos = {u: k for k, u in enumerate(G)}
    uncovered = [(u, v) for u, v in G.edges() if u not in raw and v not in raw]
    return set(raw) | _occurrence_cover(uncovered, pos)


def component_split_covers(G: nx.Graph, info: dict | None = None,
                           variants: dict[str, tuple[str, str]] | None = None,
                           exact_stop: bool = True, complete: bool = True) -> dict[str, set[Any]]:
    """Vertex sets of c0 for every variant (default :data:`DEFAULT_VARIANTS`).

    ``G`` must have no self-loops or isolated vertices. With ``complete`` each
    decoded set is completed greedily to a vertex cover of ``G``
    (:func:`complete_cover`); the caller prunes it. ``info`` receives the
    split statistics of :func:`build_split_graph` and, per variant, the
    decoding statistics of :func:`decode` under the variant's name.
    ``variants=LEGACY_VARIANTS, exact_stop=False, complete=False`` reproduces
    the first c0.
    """
    variants = DEFAULT_VARIANTS if variants is None else variants
    split_info: dict = {}
    H = build_split_graph(G, split_info, exact_stop=exact_stop)
    out: dict[str, set[Any]] = {}
    for name, (rule, residual) in variants.items():
        rule_info: dict = {}
        raw = decode(G, H, rule, rule_info, residual=residual)
        out[name] = complete_cover(G, raw) if complete else raw
        if info is not None:
            info[name] = rule_info
    if info is not None:
        info.update(split_info)
    return out
