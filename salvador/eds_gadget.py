"""Edge-dominating-set gadget reduction for vertex cover (candidate c10).

Gadget
------
For every original edge ``{u, v}`` there is an *edge-node* ``(u, v)``. For
every vertex ``u`` with neighbours ``v_1, ..., v_d`` (in adjacency order)
there are *incidence nodes* ``(u, 1), ..., (u, ceil(d/2))``: the incidence
node ``(u, k)`` is joined to the edge-nodes of ``u``'s ``(2k-1)``-th and
``2k``-th edges, i.e. ``((u, k), (u, v_{2k-1}))`` and
``((u, k), (u, v_{2k}))`` (only the first when ``d`` is odd and ``k`` is the
last index). There are no pendant nodes.

Every edge-node is adjacent to exactly two incidence nodes (one at each
endpoint) and every incidence node to at most two edge-nodes, so the gadget
is bipartite (edge-nodes versus incidence nodes) with maximum degree 2: a
disjoint union of paths and even cycles, with ``2m`` gadget edges.

Solving the gadget
------------------
Minimum edge dominating set is NP-hard in general, even on bipartite graphs
of maximum degree 3 (Yannakakis and Gavril 1980), but on a path or cycle
with ``L`` edges a minimum edge dominating set has exactly ``ceil(L/3)``
edges and is obtained by taking every third edge. Each component is solved
exactly in time linear in its length; on cycles the three rotations of the
optimal pattern are tried and the one that adds the fewest new original
vertices is kept. Total time ``O(n + m)``.

Translation to a vertex cover
-----------------------------
``u`` is put in the cover when some chosen gadget edge uses an incidence
node of ``u``. The result always covers every original edge ``{u, v}``: if
its edge-node lies on a chosen gadget edge, the other end of that gadget edge
is an incidence node of ``u`` or ``v``; otherwise both gadget edges at the
edge-node must be dominated from their incidence ends, so both ``u`` and
``v`` are selected.
"""

from __future__ import annotations

from typing import Any, Mapping, Iterable


def _build_gadget(adj: Mapping[Any, Iterable[Any]]):
    """Return (owner, gadget_adj): integer gadget nodes and their adjacency.

    Edge-nodes are ``0 .. m-1``; incidence nodes follow. ``owner[x]`` is the
    original vertex of incidence node ``x`` (``None`` for edge-nodes).
    """
    edge_id: dict[tuple[Any, Any], int] = {}
    owner: list[Any] = []
    for u in adj:
        for v in adj[u]:
            if u == v:
                continue
            key = (u, v) if (u, v) not in edge_id and (v, u) not in edge_id else None
            if key is not None:
                edge_id[(u, v)] = len(owner)
                owner.append(None)
    m = len(owner)
    gadget: list[list[int]] = [[] for _ in range(m)]

    def eid(a: Any, b: Any) -> int:
        e = edge_id.get((a, b))
        return e if e is not None else edge_id[(b, a)]

    for u in adj:
        nbrs = [v for v in adj[u] if v != u]
        for k in range(0, len(nbrs), 2):
            x = len(owner)
            owner.append(u)
            gadget.append([])
            for v in nbrs[k:k + 2]:
                e = eid(u, v)
                gadget[x].append(e)
                gadget[e].append(x)
    return owner, gadget


def _components(gadget: list[list[int]]) -> list[tuple[list[int], bool]]:
    """Split a max-degree-2 graph into paths/cycles, as ordered node lists.

    Returns ``(nodes, is_cycle)``; consecutive nodes are adjacent, and for a
    cycle the last node is also adjacent to the first.
    """
    seen = [False] * len(gadget)
    out: list[tuple[list[int], bool]] = []
    # Paths first: start from degree-1 endpoints.
    for start in range(len(gadget)):
        if seen[start] or len(gadget[start]) != 1:
            continue
        nodes = [start]
        seen[start] = True
        prev, cur = -1, start
        while True:
            nxt = [y for y in gadget[cur] if y != prev and not seen[y]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen[cur] = True
            nodes.append(cur)
        out.append((nodes, False))
    # Remaining non-isolated nodes lie on cycles.
    for start in range(len(gadget)):
        if seen[start] or not gadget[start]:
            continue
        nodes = [start]
        seen[start] = True
        prev, cur = -1, start
        while True:
            nxt = [y for y in gadget[cur] if y != prev and not seen[y]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen[cur] = True
            nodes.append(cur)
        out.append((nodes, True))
    return out


def _min_eds_positions(L: int, is_cycle: bool, offset: int = 0) -> list[int]:
    """Indices of the gadget edges (0..L-1 along the component) to choose.

    Edge ``i`` joins component nodes ``i`` and ``i+1`` (and, on a cycle, edge
    ``L-1`` joins the last node to the first). The returned set dominates every
    edge and has exactly ``ceil(L/3)`` elements, which is optimal.
    """
    if L == 0:
        return []
    c = -(-L // 3)
    if is_cycle:
        return [(offset + 3 * j) % L for j in range(c)]
    return sorted({min(1 + 3 * j, L - 1) for j in range(c)})


def eds_gadget_vertex_cover(adj: Mapping[Any, Iterable[Any]]) -> set[Any]:
    """Vertex cover from an exact minimum edge dominating set of the gadget."""
    owner, gadget = _build_gadget(adj)
    cover: set[Any] = set()
    for nodes, is_cycle in _components(gadget):
        L = len(nodes) if is_cycle else len(nodes) - 1
        if L <= 0:
            continue

        def decoded(positions: list[int]) -> set[Any]:
            picked: set[Any] = set()
            for i in positions:
                a, b = nodes[i], nodes[(i + 1) % len(nodes)]
                x = a if owner[a] is not None else b
                picked.add(owner[x])
            return picked

        if is_cycle:
            best = None
            for off in range(min(3, L)):
                vs = decoded(_min_eds_positions(L, True, off))
                if best is None or len(vs - cover) < len(best - cover):
                    best = vs
            cover |= best
        else:
            cover |= decoded(_min_eds_positions(L, False))
    return cover
