"""Linear-time bipartite planar reduction for vertex cover.

This module implements the oriented-incidence reduction of the Salvador
manuscript (Algorithm ``ComponentCover``): vertices are processed one at a
time; when ``u`` is processed, every still-unprocessed edge ``{u, v_j}``
(``j = 1..d``) creates two incidence nodes ``x_{u v_j}`` (weight
``1/deg(u)``) and ``x_{v_j u}`` (weight ``1/deg(v_j)``) joined by a *forcing
edge*, and consecutive incidences are joined by *local edges*
``{x_{u v_j}, x_{v_{j-1} u}}`` closed by ``{x_{u v_1}, x_{v_d u}}``.

Structure of the auxiliary graph
--------------------------------
Every original edge is handled exactly once (from whichever endpoint is
processed first), so each incidence node belongs to the gadget of exactly one
processed vertex ``u``. The gadget of ``u`` is the closed walk

    x_{u v_1} - x_{v_1 u} - x_{u v_2} - x_{v_2 u} - ... - x_{u v_d} - x_{v_d u} - x_{u v_1}

that is, an even cycle ``C_{2d}`` when ``d >= 2`` and a single edge ``K_2``
when ``d = 1``. The auxiliary graph ``B`` is therefore a *disjoint union of
even cycles and single edges*: it is bipartite (``u``-side incidences versus
neighbour-side incidences) and planar by construction, so the bipartiteness
and planarity checks of the manuscript always succeed.

Linear-time exact solve
-----------------------
The manuscript solves minimum-weight vertex cover on ``B`` by a min-cut /
Hopcroft-Karp formulation in ``O(|B|^{3/2})``. Because ``B`` is a disjoint
union of cycles and edges, the same optimum is obtained exactly by a dynamic
program along each cycle in time linear in its length: total ``O(|V(B)|) =
O(m)``. Together with an ``O(n + m)`` bucket sort for the processing order,
the whole reduction runs in worst-case ``O(n + m)`` time and space, with
*no* loss of optimality on the auxiliary instance.

Quality refinements (all still ``O(n + m)``)
--------------------------------------------
The auxiliary objective is a surrogate: it is exact on ``B``, but the decoded
cover can still be redundant. :func:`bipartite_planar_covers` therefore
returns several variants, each a valid cover:

* ``order``: the manuscript's input order, or ascending degree (low-degree
  vertices first -- they then behave like independent-set members whose
  neighbours enter the cover), or descending degree.
* ``sticky``: auxiliary weights are made selection-aware -- an incidence node
  whose original vertex is already in the cover gets weight ``0``, so the
  exact cycle solve reuses selected vertices instead of adding new ones.
* ``lean`` decoding: if the cycle solution of ``u`` contains any ``x_{u v}``,
  then ``u`` alone covers the whole gadget, so the neighbour-side picks of
  that same cycle are not decoded (they would only be redundant).

``algorithm.find_vertex_cover`` prunes every variant and keeps the best.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

_EPS = 1e-12


def _processing_order(adj: Mapping[Any, Iterable[Any]], order: str) -> list[Any]:
    """Return the vertex processing order in O(n + m) (bucket sort by degree)."""
    nodes = list(adj)
    if order == "input":
        return nodes
    deg = {v: len(adj[v]) for v in nodes}
    max_deg = max(deg.values(), default=0)
    buckets: list[list[Any]] = [[] for _ in range(max_deg + 1)]
    for v in nodes:
        buckets[deg[v]].append(v)
    if order == "ascending":
        return [v for b in buckets for v in b]
    if order == "descending":
        return [v for b in reversed(buckets) for v in b]
    raise ValueError(f"unknown order {order!r}")


def _better(a: tuple[float, int], b: tuple[float, int]) -> bool:
    """Lexicographic (weight, u-side count) comparison with a float tolerance."""
    if a[0] < b[0] - _EPS:
        return True
    if a[0] > b[0] + _EPS:
        return False
    return a[1] < b[1]


def min_weight_vc_cycle(weights: list[float], u_side: list[int]) -> list[bool]:
    """Exact minimum-weight vertex cover of a cycle (or a single edge) in O(L).

    ``weights[i]`` is the weight of cycle node ``c_i``; the edges are
    ``c_i - c_{i+1}`` for ``i < L - 1`` plus the closing edge ``c_{L-1} - c_0``
    when ``L > 2``. ``u_side[i]`` is ``1`` for the processed vertex's own
    incidence nodes: among optimal covers, the one with the fewest such nodes
    is returned (ties prefer covering through the neighbours, which decodes to
    an independent-set-like choice and avoids redundant selections).

    Returns a boolean inclusion list.
    """
    L = len(weights)
    if L == 2:
        c0 = (weights[0], u_side[0])
        c1 = (weights[1], u_side[1])
        return [True, False] if _better(c0, c1) else [False, True]

    best_sol: list[bool] | None = None
    best_cost: tuple[float, int] | None = None
    # Case 0: c_0 is in the cover; Case 1: c_0 is out, so c_1 and c_{L-1} are in.
    for first_in in (True, False):
        # DP over the path c_0..c_{L-1}. state = whether c_i is included.
        INF = (float("inf"), 0)
        inc = [INF] * L  # best cost with c_i included
        exc = [INF] * L  # best cost with c_i excluded
        from_inc_prev = [False] * L  # for inc[i]: was c_{i-1} included?
        if first_in:
            inc[0] = (weights[0], u_side[0])
        else:
            exc[0] = (0.0, 0)
        for i in range(1, L):
            w = (weights[i], u_side[i])
            # include c_i: previous may be either.
            if _better(inc[i - 1], exc[i - 1]) or exc[i - 1][0] == float("inf"):
                prev, from_inc_prev[i] = inc[i - 1], True
            else:
                prev, from_inc_prev[i] = exc[i - 1], False
            if prev[0] != float("inf"):
                inc[i] = (prev[0] + w[0], prev[1] + w[1])
            # exclude c_i: previous must be included.
            exc[i] = inc[i - 1]
        # c_{L-1} must be included when c_0 is excluded (closing edge).
        last_options = [(inc[L - 1], True)]
        if first_in:
            last_options.append((exc[L - 1], False))
        cost, last_in = last_options[0]
        for c, flag in last_options[1:]:
            if _better(c, cost):
                cost, last_in = c, flag
        if cost[0] == float("inf"):
            continue
        # Backtrack.
        sol = [False] * L
        cur_in = last_in
        for i in range(L - 1, -1, -1):
            sol[i] = cur_in
            if i == 0:
                break
            cur_in = from_inc_prev[i] if cur_in else True
        if best_cost is None or _better(cost, best_cost):
            best_cost, best_sol = cost, sol
    assert best_sol is not None
    return best_sol


def bipartite_planar_vertex_cover(
    adj: Mapping[Any, Iterable[Any]],
    order: str = "input",
    sticky: bool = False,
    lean: bool = False,
) -> set[Any]:
    """Vertex cover via the bipartite planar oriented-incidence reduction.

    ``order="input"``, ``sticky=False``, ``lean=False`` reproduces the
    manuscript's Algorithm ``ComponentCover`` exactly (an optimal auxiliary
    cover, decoded by first coordinate), but in worst-case O(n + m) time
    instead of O(|B|^{3/2}). See the module docstring for the variants.
    """
    deg = {v: len(adj[v]) for v in adj}
    processed: set[Any] = set()
    cover: set[Any] = set()

    for u in _processing_order(adj, order):
        processed.add(u)
        # Remaining (not yet processed) neighbours: the edges handled at u.
        nbrs = [v for v in adj[u] if v not in processed and v != u]
        d = len(nbrs)
        if d == 0:
            continue
        a = 0.0 if (sticky and u in cover) else 1.0 / deg[u]
        weights: list[float] = []
        u_side: list[int] = []
        for v in nbrs:
            weights.append(a)  # x_{u v}
            u_side.append(1)
            weights.append(0.0 if (sticky and v in cover) else 1.0 / deg[v])  # x_{v u}
            u_side.append(0)
        sol = min_weight_vc_cycle(weights, u_side)

        takes_u = any(sol[2 * j] for j in range(d))
        if takes_u:
            cover.add(u)
        if not (lean and takes_u):
            for j, v in enumerate(nbrs):
                if sol[2 * j + 1]:
                    cover.add(v)
    return cover


def residual_bipartite_planar_vertex_cover(adj: Mapping[Any, Iterable[Any]]) -> set[Any]:
    """Residual (dynamic-order) variant of the bipartite planar reduction.

    Instead of fixing the processing order up front, the next vertex ``u`` to
    receive a gadget is always one of minimum *residual* degree, i.e. degree
    in the subgraph of still-uncovered edges, and the gadget weights are
    residual degrees too (``1/d_res(u)`` and ``1/d_res(v_j)``). After the
    exact cycle solve, the decoded vertices -- and ``u`` itself -- are
    deleted from the residual graph, so every later gadget only encodes
    edges that are still uncovered (this subsumes the ``sticky`` and
    ``lean`` refinements).

    Because ``u`` has minimum residual degree, ``d_res(v_j) >= d_res(u)``
    for each of its ``d`` gadget neighbours, so the neighbour side of the
    cycle never weighs more than the ``u`` side (``sum_j 1/d_res(v_j) <=
    d * 1/d_res(u) = 1``); with the neighbour-preferring tie-break the exact
    cycle solve therefore takes the neighbour side. The decoded cover is the
    complement of a greedy minimum-degree maximal independent set.

    Runs in worst-case O(n + m): a degree-indexed bucket queue is used,
    every residual edge deletion is O(1) and lowers the bucket pointer by at
    most one, and each gadget is solved in time linear in its size.
    """
    res: dict[Any, set[Any]] = {v: set(adj[v]) - {v} for v in adj}
    deg = {v: len(res[v]) for v in res}
    max_deg = max(deg.values(), default=0)
    buckets: list[deque[Any]] = [deque() for _ in range(max_deg + 1)]
    for v, d in deg.items():
        if d:
            buckets[d].append(v)
    cover: set[Any] = set()
    low = 1  # invariant: every non-empty bucket index is >= low

    def delete(x: Any) -> None:
        nonlocal low
        for y in res[x]:
            res[y].discard(x)
            deg[y] -= 1
            if deg[y] > 0:
                buckets[deg[y]].append(y)
                if deg[y] < low:
                    low = deg[y]
        res[x] = set()
        deg[x] = 0

    while True:
        while low <= max_deg and not buckets[low]:
            low += 1
        if low > max_deg:
            break
        u = buckets[low].popleft()
        if deg[u] != low:
            continue  # stale bucket entry
        nbrs = list(res[u])
        d = len(nbrs)
        weights: list[float] = []
        u_side: list[int] = []
        for v in nbrs:
            weights.append(1.0 / d)        # x_{u v}
            u_side.append(1)
            weights.append(1.0 / deg[v])   # x_{v u}
            u_side.append(0)
        sol = min_weight_vc_cycle(weights, u_side)
        if any(sol[2 * j] for j in range(d)):
            cover.add(u)                   # lean decoding: u covers its gadget
            delete(u)
        else:
            for v in nbrs:                 # forcing edges: every x_{v u} chosen
                cover.add(v)
                delete(v)
            delete(u)
    return cover


# Variants evaluated by algorithm.find_vertex_cover. The static
# "ascending"/"descending"/sticky/lean orders of bipartite_planar_vertex_cover
# remain available, but on the DIMACS clique complements and on random
# G(n,p), regular, Barabasi-Albert and geometric graphs none of them ever
# improved on min(input, residual) after pruning, so only these two run.
VARIANTS: tuple[tuple[str, bool, bool], ...] = (
    ("input", False, False),      # the manuscript's reduction, solved in linear time
    ("residual", True, True),     # residual_bipartite_planar_vertex_cover
)


def bipartite_planar_covers(adj: Mapping[Any, Iterable[Any]]) -> list[set[Any]]:
    """Return one valid cover per entry of :data:`VARIANTS` (each O(n + m))."""
    out = []
    for o, s, l in VARIANTS:
        if o == "residual":
            out.append(residual_bipartite_planar_vertex_cover(adj))
        else:
            out.append(bipartite_planar_vertex_cover(adj, o, s, l))
    return out


# ----------------------------------------------------------------------
# Explicit construction, kept for verification and for the manuscript's
# figure; the solver above never materialises B.
# ----------------------------------------------------------------------

def build_auxiliary_graph(graph: nx.Graph, order: str = "input") -> nx.Graph:
    """Materialise the auxiliary graph ``B`` of the manuscript as an nx.Graph.

    Node ``("x", u, v)`` is the oriented incidence ``x_{uv}``; its ``weight``
    attribute is ``1/deg(u)``.
    """
    adj = {v: set(graph[v]) - {v} for v in graph}
    deg = {v: len(adj[v]) for v in adj}
    B = nx.Graph()
    processed: set[Hashable] = set()
    for u in _processing_order(adj, order):
        processed.add(u)
        nbrs = [v for v in adj[u] if v not in processed]
        first = prev = None
        for v in nbrs:
            xu, xv = ("x", u, v), ("x", v, u)
            B.add_node(xu, weight=1.0 / deg[u])
            B.add_node(xv, weight=1.0 / deg[v])
            B.add_edge(xu, xv, kind="forcing")
            if prev is None:
                first = xu
            else:
                B.add_edge(xu, prev, kind="local")
            prev = xv
        if len(nbrs) > 1:
            B.add_edge(first, prev, kind="local")
    return B
