"""Strategy c9: bipartite layers, solved one after another (linear time).

Every connected component of ``G`` is treated on its own.

**Bipartite components** are solved with a near-optimal linear-time bipartite
strategy (:func:`bipartite_cover`): the NetworkX minimum maximal matching is
enlarged by at most ``phases`` Hopcroft--Karp phases (each phase finds a maximal
set of vertex-disjoint shortest augmenting paths in ``O(n + m)``), and Koenig's
construction turns the matching into a vertex cover: with ``Z`` the vertices
reachable from the unmatched vertices of one side by alternating paths, the
cover is ``(top - Z) | (bottom & Z)``. This set is a vertex cover for *any*
matching, of size ``|matching|`` plus the unmatched vertices of the other side
in ``Z``; when no augmenting path is left it is a minimum vertex cover (Koenig).
After ``k`` phases the matching has at least ``k/(k+1)`` of the maximum size
(Hopcroft--Karp).

**Non-bipartite components** are split into bipartite layers. A 2-colouring of
the component is computed in breadth-first order, each vertex taking the colour
that disagrees with most of its already coloured neighbours (ties: the colour
opposite to its parent). At least half of the edges join different colours, and
those edges form a bipartite graph: layer 0, whose vertices get the copy index 0.
The remaining edges, without the vertices they leave isolated, are coloured the
same way: layer 1, copy index 1; and so on until no edge is left. Every layer
takes at least half of the remaining edges, so a component with ``m_c`` edges has
at most ``floor(log2 m_c) + 1`` layers, and building them all costs ``O(n + m)``.
Every edge of the component lies in exactly one layer.

The layers are then solved in an order: in layer ``t`` the copies of the
vertices already selected in an earlier layer of the same order are discarded
(their edges are covered), and the rest of the layer is solved with
:func:`bipartite_cover`. The union of the selected vertices is a vertex cover of
the component. Different orders give different covers; every order is tried when
there are at most ``max_orders`` of them (``L!`` orders for ``L`` layers), and
otherwise ``max_orders`` orders (the natural order, the reverse order, then the
rotations of both), and the smallest cover is kept.

Running time: ``O(max_orders * phases * (n + m))``, linear for the fixed defaults.
c9 is exact on every graph whose components are bipartite, when the phases find
a maximum matching (reported as ``exact``).
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Any, Iterable

import networkx as nx

#: Hopcroft--Karp phases after the NetworkX minimum maximal matching, per bipartite solve.
DEFAULT_PHASES: int = 8
#: Largest number of layer orders tried per component.
DEFAULT_MAX_ORDERS: int = 6


# ----------------------------------------------------------------------------
# Linear-time near-optimal bipartite vertex cover.
# ----------------------------------------------------------------------------
def _augment_phase(adj: dict[Any, Iterable[Any]], top: list[Any], mate: dict[Any, Any]) -> bool:
    """One Hopcroft--Karp phase: BFS layers from the free top vertices, then vertex-disjoint shortest
    augmenting paths by iterative DFS. Returns False when no augmenting path exists (the matching is
    maximum). O(n + m)."""
    dist: dict[Any, int] = {}
    q = deque()
    for x in top:
        if x not in mate:
            dist[x] = 0
            q.append(x)
    found = False
    limit = None
    while q:
        x = q.popleft()
        if limit is not None and dist[x] >= limit:
            continue
        for y in adj[x]:
            z = mate.get(y)
            if z is None:
                if limit is None:
                    limit = dist[x] + 1
                found = True
            elif z not in dist:
                dist[z] = dist[x] + 1
                q.append(z)
    if not found:
        return False
    it = {x: iter(adj[x]) for x in dist}
    used: set[Any] = set()  # bottom vertices already on an augmenting path of this phase
    for root in top:
        if root in mate or dist.get(root) != 0:
            continue
        stack = [root]
        path_bottom: list[Any] = []
        while stack:
            x = stack[-1]
            advanced = False
            for y in it[x]:
                if y in used:
                    continue
                z = mate.get(y)
                if z is None:
                    if dist[x] + 1 != limit:
                        continue
                    path_bottom.append(y)
                    # augment along stack / path_bottom
                    for tx, by in zip(stack, path_bottom):
                        mate[tx] = by
                        mate[by] = tx
                    used.update(path_bottom)
                    stack = []
                    advanced = True
                    break
                if dist.get(z) == dist[x] + 1:
                    used.add(y)
                    path_bottom.append(y)
                    stack.append(z)
                    advanced = True
                    break
            if stack and not advanced:
                stack.pop()
                dist[x] = -1  # dead end for the rest of the phase
                if path_bottom:
                    path_bottom.pop()
    return True


def bipartite_cover(adj: dict[Any, set[Any]], side: dict[Any, int], phases: int = DEFAULT_PHASES,
                    info: dict | None = None) -> set[Any]:
    """Vertex cover of the bipartite graph ``adj`` (``side[v]`` in {0, 1}, every edge between sides):
    NetworkX minimum maximal matching + at most ``phases`` Hopcroft--Karp phases + Koenig's
    construction, computed with each side as the top side, the smaller cover kept. Valid for any
    matching; minimum when no augmenting path is left. ``info`` receives ``exact`` and ``matching``."""
    nodes = [v for v in adj if adj[v]]
    if not nodes:
        if info is not None:
            info.update({"exact": True, "matching": 0})
        return set()
    G = nx.Graph()
    G.add_edges_from((u, v) for u in nodes for v in adj[u] if side[u] == 0)
    mate: dict[Any, Any] = {}
    for u, v in nx.approximation.min_maximal_matching(G):
        mate[u], mate[v] = v, u
    top = [v for v in nodes if side[v] == 0]
    exact = False
    for _ in range(phases):
        if not _augment_phase(adj, top, mate):
            exact = True
            break
    if not exact:
        exact = _has_no_augmenting_path(adj, top, mate)
    best: set[Any] | None = None
    for s in (0, 1):
        tops = [v for v in nodes if side[v] == s]
        Z = [v for v in tops if v not in mate]
        seen = set(Z)
        q = deque(Z)
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y in seen or mate.get(x) == y:
                    continue
                seen.add(y)
                z = mate.get(y)
                if z is not None and z not in seen:
                    seen.add(z)
                    q.append(z)
        cover = {v for v in tops if v not in seen} | {v for v in nodes if side[v] != s and v in seen}
        if best is None or len(cover) < len(best):
            best = cover
    if info is not None:
        info.update({"exact": exact, "matching": sum(1 for v in top if v in mate)})
    return best


def _has_no_augmenting_path(adj: dict[Any, Iterable[Any]], top: list[Any], mate: dict[Any, Any]) -> bool:
    """True when no alternating path joins a free top vertex to a free bottom vertex (BFS, O(n + m))."""
    seen = {x for x in top if x not in mate}
    q = deque(seen)
    while q:
        x = q.popleft()
        for y in adj[x]:
            z = mate.get(y)
            if z is None:
                return False
            if z not in seen:
                seen.add(z)
                q.append(z)
    return True


# ----------------------------------------------------------------------------
# Bipartite layers of a component.
# ----------------------------------------------------------------------------
def _majority_colouring(radj: dict[Any, set[Any]], order: list[Any]) -> dict[Any, int]:
    """Breadth-first 2-colouring in which every vertex disagrees with most of its coloured
    neighbours (ties: opposite to its parent). At least half of the edges join different colours."""
    colour: dict[Any, int] = {}
    for s in order:
        if s in colour or not radj.get(s):
            continue
        colour[s] = 0
        q = deque([s])
        parent = {s: None}
        while q:
            x = q.popleft()
            if x != s:
                c0 = c1 = 0
                for y in radj[x]:
                    c = colour.get(y)
                    if c == 0:
                        c0 += 1
                    elif c == 1:
                        c1 += 1
                colour[x] = 1 if c0 > c1 else 0 if c1 > c0 else 1 - colour[parent[x]]
            for y in radj[x]:
                if y not in parent:
                    parent[y] = x
                    q.append(y)
    return colour


def bipartite_layers(adj: dict[Any, set[Any]], comp: Iterable[Any], order: list[Any]) -> list[tuple[dict, dict]]:
    """Layers ``(layer_adj, side)`` of the component ``comp``: every edge in exactly one layer, every
    layer bipartite with the sides ``side``, each layer at least half of the edges left."""
    comp = set(comp)
    radj = {v: {w for w in adj[v] if w in comp} for v in comp}
    order = [v for v in order if v in comp]
    layers = []
    while True:
        radj = {v: nb for v, nb in radj.items() if nb}
        if not radj:
            return layers
        order = [v for v in order if v in radj]
        colour = _majority_colouring(radj, order)
        layer = {v: {w for w in nb if colour[w] != colour[v]} for v, nb in radj.items()}
        layer = {v: nb for v, nb in layer.items() if nb}
        layers.append((layer, {v: colour[v] for v in layer}))
        for v, nb in layer.items():
            radj[v] -= nb


def _orders(L: int, max_orders: int) -> list[tuple[int, ...]]:
    if L <= 1:
        return [tuple(range(L))]
    total = 1
    for k in range(2, L + 1):
        total *= k
        if total > max_orders:
            break
    if total <= max_orders:
        return list(itertools.permutations(range(L)))
    base = list(range(L))
    out: list[tuple[int, ...]] = []
    for seq in (base, base[::-1]):
        for shift in range(L):
            t = tuple(seq[shift:] + seq[:shift])
            if t not in out:
                out.append(t)
            if len(out) >= max_orders:
                return out
    return out


def _solve_order(layers: list[tuple[dict, dict]], perm: tuple[int, ...], phases: int, stats: dict) -> set[Any]:
    selected: set[Any] = set()
    for t in perm:
        layer, side = layers[t]
        sub = {v: {w for w in nb if w not in selected} for v, nb in layer.items() if v not in selected}
        info: dict = {}
        selected |= bipartite_cover(sub, side, phases, info)
        stats["exact"] = stats["exact"] and info["exact"]
    return selected


def bipartite_layers_cover(adj: dict[Any, set[Any]], phases: int = DEFAULT_PHASES,
                           max_orders: int = DEFAULT_MAX_ORDERS, info: dict | None = None) -> set[Any]:
    """Strategy c9 on the adjacency ``adj`` (no self-loops): a vertex cover (not pruned). ``info``
    receives ``bipartite_components``, ``nonbipartite_components``, ``max_layers``, ``layers``
    (total over the components), ``orders`` (orders tried) and ``exact`` (every bipartite solve of
    the kept orders ended with a maximum matching; on a graph whose components are all bipartite this
    means the cover is minimum)."""
    order = list(adj)
    G = nx.Graph()
    G.add_nodes_from(adj)
    G.add_edges_from((u, v) for u, nb in adj.items() for v in nb)
    stats = {"bipartite_components": 0, "nonbipartite_components": 0, "max_layers": 0, "layers": 0,
             "orders": 0, "exact": True}
    cover: set[Any] = set()
    for comp in nx.connected_components(G):
        if len(comp) < 2:
            continue
        layers = bipartite_layers(adj, comp, order)
        L = len(layers)
        stats["bipartite_components" if L == 1 else "nonbipartite_components"] += 1
        stats["max_layers"] = max(stats["max_layers"], L)
        stats["layers"] += L
        best, best_exact = None, True
        for perm in _orders(L, max_orders):
            st = {"exact": True}
            c = _solve_order(layers, perm, phases, st)
            stats["orders"] += 1
            if best is None or len(c) < len(best):
                best, best_exact = c, st["exact"]
        cover |= best
        stats["exact"] = stats["exact"] and best_exact
    if info is not None:
        info.update(stats)
    return cover
