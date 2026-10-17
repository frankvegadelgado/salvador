"""Weighted subdivision reduction for vertex cover (candidate c11).

Reduction
---------
The original vertices are kept: vertex ``u`` of ``G`` is the vertex-node
``(u, u)`` of an auxiliary graph ``S``. Every edge ``{u, w}`` of ``G`` becomes an
edge-node ``(u, w)`` and the two half-edges::

    ((u, u), (u, w))   and   ((w, w), (u, w)).

``S`` is the subdivision of ``G``: ``n + m`` nodes, ``2m`` edges, bipartite
(vertex-nodes on one side, edge-nodes on the other). A set ``C`` of original
vertices is a vertex cover of ``G`` exactly when every edge-node of ``S`` has
a selected vertex-node neighbour, i.e. when the vertex-nodes of ``C``
*dominate* the edge-node side of ``S``.

Weights
-------
Vertex-node ``(u, u)`` has weight 1 (selecting ``u``). An edge-node left
undominated (neither endpoint selected: the original edge is uncovered) costs
``W = n + 1``. The objective is

    Phi(C) = |C| + W * #(undominated edge-nodes).

``W`` exceeds the weight ``n`` of selecting every vertex, so a minimiser of
Phi dominates every edge-node: it is a minimum vertex cover of ``G``.

Why the edge-nodes are not simply put in the cover of ``S``
------------------------------------------------------------
In a vertex cover of ``S`` the edge-node ``(u, w)`` must be selected unless
*both* ``(u, u)`` and ``(w, w)`` are, and it then covers both half-edges by
itself. With any weight ``c`` on it, an edge with one endpoint selected costs
``c`` and an edge with no endpoint selected also costs ``c``: the vertex
cover of ``S`` cannot tell a covered edge of ``G`` from an uncovered one (for a
forest the minimum cover of ``S`` is the ``m`` edge-nodes and decodes to the
empty set). This is what makes vertex cover on the bipartite ``S``
polynomial. The domination requirement above is the part that carries the
hardness of vertex cover, so the solver keeps it as a hard constraint.

Linear-time solver (reducing-peeling on ``S``)
-----------------------------------------------
Each vertex-node ``u`` has a residual degree: the number of its edge-nodes
that are still undominated with the other endpoint undecided. The solver
applies exact reductions while one is available and peels otherwise:

* **degree 0**: ``u`` dominates nothing new -> not selected (exact);
* **degree 1**: ``u``'s only open edge-node ``(u, w)`` must be dominated, and
  ``w`` dominates at least as much -> select ``w``, not ``u`` (exact);
* **degree 2, triangle**: ``u``'s two open edge-nodes go to ``a`` and ``b``
  with ``{a, b}`` an edge -> select ``a`` and ``b``, not ``u`` (exact: every
  cover takes two of the triangle ``u, a, b``);
* **degree 2, fold** (``fold=True``): ``u``'s two open edge-nodes go to
  non-adjacent ``a`` and ``b`` -> replace ``u, a, b`` by one new vertex-node
  ``z`` adjacent to ``N(a) | N(b)``; ``tau(G) = tau(G') + 1``, and ``z``
  selected unfolds to ``{a, b}``, ``z`` not selected to ``{u}`` (exact);
* **peel**, when no exact rule applies (heuristic), one of:

  - ``peel="max"``: select the vertex-node with the largest residual degree,
    which dominates the most undominated edge-nodes, i.e. lowers the penalty
    part of Phi the most per unit weight;
  - ``peel="min"``: leave the vertex-node with the smallest residual degree
    unselected, which forces its (fewest) open edge-nodes' other endpoints
    into the cover -- the minimum-degree independent-set rule of c5/c8, with
    the exact reductions re-applied after every step;
  - ``peel="ref"`` (guided): select the vertex-node of largest residual
    degree among those in a *reference cover* (in the ensemble: the best of
    c1..c10); a fold node counts as in the reference when both ``a`` and
    ``b`` are. When no undecided reference vertex is left it falls back to
    ``"max"``. The exact reductions still run between guided steps, so they
    can correct the reference locally.

Selecting ``u`` dominates its edge-nodes, which leave the residual graph and
lower the residual degree of their other endpoint; a vertex-node that is not
selected forces its open edge-nodes' other endpoints to be selected, so no
edge-node is ever left undominated. When no peel step was needed every
decision was an exact reduction, and the cover is a minimum vertex cover
(``info["exact"]``). Degree queues and degree bucket queues give
``O(n + m)`` time for the reductions and peels. A fold costs
``deg(a) + deg(b)``; repeated folds can grow neighbourhoods, so folding is
charged to a work counter and stops (the vertex then waits for a peel) once
``fold_budget * (n + m)`` would be exceeded: worst-case ``O(n + m)`` for a
fixed budget.

:data:`VARIANTS` lists the ``(peel, fold)`` settings c11 runs; the smallest
pruned cover is kept. ``S`` is not
materialised; :func:`build_subdivision_graph` builds it for verification.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping

import networkx as nx

#: (peel, fold) settings tried by c11 in the ensemble (the smallest cover wins).
#: "min" is still available but never improved on these two in car_c11_variants.
VARIANTS: tuple[tuple[str, bool], ...] = (("max", True), ("ref", True))

#: Fold work budget, in units of (n + m).
DEFAULT_FOLD_BUDGET: float = 4.0


class _FoldNode:
    """Vertex created by a degree-2 fold. Its hash is its creation index, so set
    iteration order -- and therefore the cover -- does not depend on Python's
    per-process string-hash randomisation (PYTHONHASHSEED)."""

    __slots__ = ("k",)

    def __init__(self, k: int) -> None:
        self.k = k

    def __hash__(self) -> int:
        return self.k

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"_FoldNode({self.k})"


def subdivision_vertex_cover(
    adj: Mapping[Any, Iterable[Any]],
    info: dict | None = None,
    peel: str = "max",
    fold: bool = False,
    fold_budget: float = DEFAULT_FOLD_BUDGET,
    reference: Iterable[Any] | None = None,
) -> set[Any]:
    """One c11 variant: vertex cover of ``G`` from the weighted subdivision ``S``.

    ``peel`` is ``"max"``, ``"min"`` or ``"ref"`` (needs ``reference``, a
    vertex cover of ``G`` to guide the peel) and ``fold`` enables the degree-2 fold
    (see the module docstring). ``info`` (optional) receives ``nodes``
    (``n + m``) and ``edges`` (``2m``) of ``S``, the number of each reduction
    (``deg0``, ``deg1``, ``triangle``, ``folds``), ``peels``, ``exact`` (no
    peel was needed: the cover is minimum), ``ref_peels`` (guided peels),
    ``fold_budget_hit`` and
    ``undominated`` (always 0).
    """
    if peel not in ("max", "min", "ref"):
        raise ValueError(f"unknown peel {peel!r}")
    if peel == "ref" and reference is None:
        raise ValueError('peel="ref" needs a reference cover')
    ref: set[Any] = set(reference) if peel == "ref" else set()
    res: dict[Any, set[Any]] = {v: set(adj[v]) - {v} for v in adj}
    n0 = len(res)
    m0 = sum(len(s) for s in res.values()) // 2
    deg = {v: len(s) for v, s in res.items()}
    max_deg = max(deg.values(), default=0)

    buckets: list[list[Any]] = [[] for _ in range(max_deg + 1)]
    low: deque[Any] = deque()  # vertices whose residual degree dropped to <= 2
    for v, d in deg.items():
        buckets[d].append(v)
        if d <= 2:
            low.append(v)
    rbuckets: list[list[Any]] = [[] for _ in range(max_deg + 1)]  # reference vertices only
    for v in ref:
        if v in deg:
            rbuckets[deg[v]].append(v)
    top = max_deg   # max-peel pointer: every live degree is <= top
    rtop = max_deg  # ref-peel pointer: every live reference degree is <= rtop
    bottom = 1      # min-peel pointer: every live degree >= 1 is >= bottom

    decided: set[Any] = set()
    alive = n0
    cover: set[Any] = set()
    folds: list[tuple[Any, Any, Any, Any]] = []  # (u, a, b, z) in creation order
    counts = {"deg0": 0, "deg1": 0, "triangle": 0, "folds": 0, "peels": 0, "ref_peels": 0}
    fold_limit = fold_budget * (n0 + m0)
    fold_work = 0
    budget_hit = False

    def push(y: Any, d: int) -> None:
        nonlocal top, bottom, rtop
        if d >= len(buckets):
            buckets.extend([] for _ in range(d - len(buckets) + 1))
            rbuckets.extend([] for _ in range(d - len(rbuckets) + 1))
        buckets[d].append(y)
        if y in ref:
            rbuckets[d].append(y)
            if d > rtop:
                rtop = d
        if d > top:
            top = d
        if 0 < d < bottom:
            bottom = d
        if d <= 2:
            low.append(y)

    def detach(x: Any) -> None:
        """Remove decided vertex x from the residual graph (its edge-nodes are dominated)."""
        for y in res[x]:
            res[y].discard(x)
            deg[y] -= 1
            push(y, deg[y])
        res[x] = set()
        deg[x] = 0

    def select(x: Any) -> None:
        nonlocal alive
        decided.add(x)
        alive -= 1
        cover.add(x)
        detach(x)

    def reject(x: Any) -> None:
        """x not selected: every open edge-node of x forces its other endpoint."""
        nonlocal alive
        decided.add(x)
        alive -= 1
        for y in list(res[x]):
            if y not in decided:
                select(y)
        detach(x)

    def do_fold(u: Any, a: Any, b: Any) -> None:
        nonlocal alive
        z = _FoldNode(len(folds))
        nz = (res[a] | res[b]) - {u, a, b}
        for x in (u, a, b):
            for y in res[x]:
                res[y].discard(x)
            res[x] = set()
            deg[x] = 0
            decided.add(x)  # resolved when unfolding
        alive -= 2
        res[z] = nz
        deg[z] = len(nz)
        if a in ref and b in ref:
            ref.add(z)
        for y in nz:
            res[y].add(z)
        # Degrees of N(a) and N(b) changed: re-queue them and z.
        for y in nz:
            deg[y] = len(res[y])
            push(y, deg[y])
        push(z, deg[z])
        folds.append((u, a, b, z))

    while alive > 0:
        progressed = False
        while low:
            v = low.popleft()
            if v in decided:
                continue
            d = deg[v]
            if d == 0:
                counts["deg0"] += 1
                reject(v)
            elif d == 1:
                counts["deg1"] += 1
                reject(v)
            elif d == 2:
                a, b = res[v]
                if b in res[a]:  # triangle in the current graph
                    counts["triangle"] += 1
                    reject(v)
                elif fold and not budget_hit:
                    cost = len(res[a]) + len(res[b])
                    if fold_work + cost > fold_limit:
                        budget_hit = True
                        continue
                    fold_work += cost
                    counts["folds"] += 1
                    do_fold(v, a, b)
                else:
                    continue  # no exact rule; v waits for a peel
            else:
                continue  # stale entry
            progressed = True
            break
        if progressed:
            continue
        # Peel.
        v = None
        if peel == "ref":
            while rtop > 0:
                bucket = rbuckets[rtop]
                while bucket and (bucket[-1] in decided or deg[bucket[-1]] != rtop):
                    bucket.pop()
                if bucket:
                    v = bucket.pop()
                    counts["ref_peels"] += 1
                    break
                rtop -= 1
        if v is not None:
            pass
        elif peel in ("max", "ref"):
            while top > 0:
                bucket = buckets[top]
                while bucket and (bucket[-1] in decided or deg[bucket[-1]] != top):
                    bucket.pop()
                if bucket:
                    v = bucket.pop()
                    break
                top -= 1
        else:
            while bottom < len(buckets):
                bucket = buckets[bottom]
                while bucket and (bucket[-1] in decided or deg[bucket[-1]] != bottom):
                    bucket.pop()
                if bucket:
                    v = bucket.pop()
                    break
                bottom += 1
        if v is None:
            # Only isolated undecided vertices remain.
            for x in list(res):
                if x not in decided:
                    counts["deg0"] += 1
                    decided.add(x)
            break
        counts["peels"] += 1
        if peel in ("max", "ref"):
            select(v)
        else:
            reject(v)

    # Unfold in reverse creation order (a fold node is resolved before the
    # nodes it was built from, which may themselves be fold nodes).
    for u, a, b, z in reversed(folds):
        if z in cover:
            cover.discard(z)
            cover.add(a)
            cover.add(b)
        else:
            cover.add(u)

    if info is not None:
        undominated = sum(1 for v in adj for w in adj[v] if v != w and v not in cover and w not in cover)
        info.update({"nodes": n0 + m0, "edges": 2 * m0, **counts,
                     "exact": counts["peels"] == 0, "fold_budget_hit": budget_hit,
                     "undominated": undominated // 2})
    return cover


def subdivision_covers(
    adj: Mapping[Any, Iterable[Any]],
    variants: Iterable[tuple[str, bool]] = VARIANTS,
    fold_budget: float = DEFAULT_FOLD_BUDGET,
    info: dict | None = None,
    reference: Iterable[Any] | None = None,
) -> dict[tuple[str, bool], set[Any]]:
    """Run every ``(peel, fold)`` variant; ``info[(peel, fold)]`` gets its counts.
    ``"ref"`` variants are skipped when no ``reference`` cover is given."""
    out = {}
    for peel, fold in variants:
        if peel == "ref" and reference is None:
            continue
        vi: dict = {}
        out[(peel, fold)] = subdivision_vertex_cover(adj, vi, peel, fold, fold_budget, reference)
        if info is not None:
            info[(peel, fold)] = vi
    return out


# ----------------------------------------------------------------------------
# Explicit construction and scoring, for verification (the solver never builds S).
# ----------------------------------------------------------------------------
def build_subdivision_graph(graph: nx.Graph) -> nx.Graph:
    """Materialise ``S``: vertex-node ``(u, u)`` (weight 1, ``bipartite`` 0) per
    vertex, edge-node ``(u, w)`` (weight ``W = n + 1`` when undominated,
    ``bipartite`` 1) per edge, and the half-edges ``((u, u), (u, w))``,
    ``((w, w), (u, w))``."""
    S = nx.Graph()
    W = graph.number_of_nodes() + 1
    for u in graph:
        S.add_node((u, u), weight=1, bipartite=0)
    for u, w in graph.edges():
        if u == w:
            continue
        S.add_node((u, w), weight=W, bipartite=1)
        S.add_edge((u, u), (u, w))
        S.add_edge((w, w), (u, w))
    return S


def phi(graph: nx.Graph, cover: set[Any]) -> int:
    """Phi(C) = |C| + (n + 1) * #(edge-nodes with no selected neighbour)."""
    W = graph.number_of_nodes() + 1
    return len(cover) + W * sum(1 for u, w in graph.edges() if u != w and u not in cover and w not in cover)
