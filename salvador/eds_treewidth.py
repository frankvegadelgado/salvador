"""Edge-dominating-set gadget solved by a treewidth DP (candidate c10).

Gadget
------
Every edge ``{u, v}`` of ``G`` becomes an *edge-node* ``(u, v)``. If
``n_0, n_1, ..., n_{d-1}`` are the neighbours of ``u`` (adjacency order), ``u``
gets *incidence nodes* ``(u, 0), ..., (u, d-2)`` and the gadget edges

    ((u, k), (u, n_k))   and   ((u, k), (u, n_{k+1}))      for k = 0, ..., d-2,

so the edge-nodes of ``u`` form a path ``(u,n_0) - (u,0) - (u,n_1) - (u,1) -
... - (u,n_{d-1})``. For example, with neighbours ``v, w, z`` the gadget
edges of ``u`` are ``((u,0),(u,v))``, ``((u,0),(u,w))``, ``((u,1),(u,w))`` and
``((u,1),(u,z))``. A vertex of degree 1 gets one incidence node ``(u, 0)``
joined to its only edge-node. The edge-node ``(u, v)`` is shared by the paths
of ``u`` and ``v`` (e.g. it is also joined to ``(v, 0)``).

Every gadget edge joins an incidence node to an edge-node, so the gadget is
bipartite. Edge-nodes have degree at most 4 and incidence nodes degree at
most 2.

Decoding
--------
A chosen gadget edge ``((u, k), (u, x))`` puts ``u`` in the cover; if the two
edges ``((u, k), (u, v))`` and ``((v, j), (u, v))`` are both chosen, both
``u`` and ``v`` enter.

* **Validity.** Take an original edge ``uv`` and a gadget edge
  ``((u, k), (u, v))``. In an EDS ``F`` it is dominated by a chosen edge
  touching ``(u, k)`` (which decodes to ``u``) or touching ``(u, v)`` (which
  decodes to ``u`` or ``v``). So the decoded set ``D`` is a vertex cover of
  ``G``, and ``|D| <= |F|``.
* **Exactness of the reduction.** For a vertex cover ``C``, all the path
  edges of the vertices of ``C`` form an EDS that decodes to exactly ``C``.
  So the smallest cover decoded from any EDS is ``tau(G)``. But a *minimum*
  EDS (fewest edges) does not always decode to a minimum cover: a vertex of
  degree ``d`` needs about ``d/2`` path edges, so the EDS objective favours
  low-degree vertices. c10 is therefore a heuristic for vertex cover.

Dynamic program
---------------
c10 computes a minimum EDS of the gadget of every **bipartite** connected
component of ``G`` by bucket elimination along a greedy minimum-degree
elimination order of the gadget (same order and limits as c11, see
:mod:`salvador.treewidth_dp`). Every gadget node ``x`` has three states:

* ``0`` -- ``x`` is not incident to the chosen edge set ``F``;
* ``1`` -- ``x`` is incident to ``F``, and this factor does not have to
  provide a chosen edge at ``x``;
* ``2`` -- ``x`` is incident to ``F`` and this factor contains a chosen edge
  at ``x`` (its witness).

A gadget edge ``ab`` is the factor: not chosen and dominated (``(0,1)``,
``(1,0)``, ``(1,1)``: cost 0) or chosen (``(2,2)``, ``(2,1)``, ``(1,2)``: cost
1). Factors are joined with "same membership, witness if either factor is",
and a node is eliminated only if it is untouched or witnessed. Back-pointers
recover the chosen edges, which are decoded as above.

Limits and core
---------------
A node is eliminated only while its fill degree is at most ``max_width``
and the tables (``3^|bag|`` entries) fit in ``budget * (n + m)``. Nodes left
over form the core. The core is fixed from a reference cover ``C``: the EDS
"all path edges of ``C``" touches every edge-node and the incidence nodes of
the vertices of ``C``, and the rest of the EDS is solved exactly given that
choice. Every vertex of ``C`` with a core node in its path stays in the
cover. Non-bipartite components take the reference cover. The result is
pruned and, if it is larger than the reference, the reference is used.

Running time: ``O(n + m)`` for the bipartiteness test and the gadget
(``m`` edge-nodes, at most ``2m`` incidence nodes, at most ``4m`` gadget
edges), and at most ``budget * (n + m)`` DP work (every join is charged its
actual work; a component that would exceed the budget falls back to the
reference). Worst-case ``O(n + m)`` for fixed limits.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping

from .treewidth_dp import _elimination_order

DEFAULT_MAX_WIDTH = 6
DEFAULT_DP_BUDGET = 256

INF = float("inf")
_EDGE_TABLE = {(0, 1): 0, (1, 0): 0, (1, 1): 0, (2, 2): 1, (2, 1): 1, (1, 2): 1}


class _Budget(Exception):
    pass


def _components(adj: Mapping[Any, Iterable[Any]]):
    """Connected components and whether each one is bipartite."""
    colour: dict[Any, int] = {}
    for s in adj:
        if s in colour:
            continue
        colour[s] = 0
        comp = [s]
        ok = True
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w == u:
                    continue
                if w not in colour:
                    colour[w] = 1 - colour[u]
                    comp.append(w)
                    q.append(w)
                elif colour[w] == colour[u]:
                    ok = False
        yield comp, ok


def build_gadget(comp, adj):
    """The EDS gadget of one component.

    Returns ``(n_nodes, gadj, gedges, owner, node_vertex, n_edge_nodes)``:
    gadget nodes are integers (edge-nodes first), ``gedges`` lists
    ``(incidence_node, edge_node)``, ``owner[e]`` is the original vertex a
    chosen gadget edge ``e`` decodes to, and ``node_vertex[x]`` is the
    original vertex of an incidence node (``None`` for edge-nodes).
    """
    eid: dict[frozenset, int] = {}
    nbrs: dict[Any, list[Any]] = {}
    for u in comp:
        lst = [w for w in adj[u] if w != u]
        nbrs[u] = lst
        for w in lst:
            key = frozenset((u, w))
            if key not in eid:
                eid[key] = len(eid)
    n_edge_nodes = len(eid)
    nxt = n_edge_nodes
    gedges: list[tuple[int, int]] = []
    owner: list[Any] = []
    node_vertex: list[Any] = [None] * n_edge_nodes
    for u in comp:
        lst = nbrs[u]
        d = len(lst)
        if d == 0:
            continue
        if d == 1:
            x = nxt
            nxt += 1
            node_vertex.append(u)
            gedges.append((x, eid[frozenset((u, lst[0]))]))
            owner.append(u)
            continue
        for k in range(d - 1):
            x = nxt
            nxt += 1
            node_vertex.append(u)
            for w in (lst[k], lst[k + 1]):
                gedges.append((x, eid[frozenset((u, w))]))
                owner.append(u)
    gadj: dict[int, set[int]] = {x: set() for x in range(nxt)}
    for a, b in gedges:
        gadj[a].add(b)
        gadj[b].add(a)
    return nxt, gadj, gedges, owner, node_vertex, n_edge_nodes


def _solve_component(comp, adj, reference, max_width, budget, stats):
    """Minimum EDS of the gadget of one component, decoded to a cover.

    Returns ``(cover_of_component, exact, eds_size)``.
    """
    n_nodes, gadj, gedges, owner, node_vertex, n_edge_nodes = build_gadget(comp, adj)
    limit = budget * max(n_nodes + len(gedges), 1)
    order, scopes, core = _elimination_order(gadj, max_width, limit, states=3)
    stats["width"] = max(stats.get("width", 0), max((len(s) for s in scopes.values()), default=0))
    if core:
        stats["core"] = stats.get("core", 0) + len(core)
    pos = {x: p for p, x in enumerate(order)}

    def core_states(y):
        # States of a core node under the reference EDS "all path edges of C".
        if node_vertex[y] is None:
            return (1, 2)  # edge-node: always touched (C is a cover)
        return (1, 2) if node_vertex[y] in reference else (0,)

    buckets: dict[int, list] = {x: [] for x in order}
    roots: list = []
    work = 0

    def place(f):
        scope = f[0]
        if not scope:
            roots.append(f)
        else:
            buckets[min(scope, key=pos.__getitem__)].append(f)

    for e, (a, b) in enumerate(gedges):
        a_core, b_core = a in core, b in core
        if a_core and b_core:
            continue
        if a_core or b_core:
            x, y = (b, a) if a_core else (a, b)
            ys = core_states(y)
            tab: dict[tuple, float] = {}
            for (sa, sb), c in _EDGE_TABLE.items():
                sx, sy = (sb, sa) if a_core else (sa, sb)
                if sy in ys and c < tab.get((sx,), INF):
                    tab[(sx,)] = c
            place(((x,), tab, ("edge", e)))
        else:
            place(((a, b), dict(_EDGE_TABLE), ("edge", e)))

    def join(f, g):
        nonlocal work
        sf, tf, _ = f
        sg, tg, _ = g
        shared = [v for v in sf if v in sg]
        fi = [sf.index(v) for v in shared]
        gi = [sg.index(v) for v in shared]
        g_only = [j for j, v in enumerate(sg) if v not in sf]
        scope = tuple(sf) + tuple(sg[j] for j in g_only)
        index: dict[tuple, list] = {}
        for kg, cg in tg.items():
            index.setdefault(tuple(kg[j] != 0 for j in gi), []).append((kg, cg))
        out: dict[tuple, float] = {}
        back: dict[tuple, tuple] = {}
        for kf, cf in tf.items():
            lst = index.get(tuple(kf[j] != 0 for j in fi))
            if not lst:
                continue
            work += len(lst)
            for kg, cg in lst:
                key = list(kf)
                for a, b in zip(fi, gi):
                    if kg[b] == 2:
                        key[a] = 2
                key = tuple(key) + tuple(kg[j] for j in g_only)
                c = cf + cg
                if c < out.get(key, INF):
                    out[key] = c
                    back[key] = (kf, kg)
        if work > limit:
            raise _Budget
        return (scope, out, ("join", f, g, back))

    for x in order:
        fs = buckets.pop(x)
        if not fs:
            continue
        h = fs[0]
        for g in fs[1:]:
            h = join(h, g)
        scope, tab, _ = h
        ix = scope.index(x)
        new_scope = scope[:ix] + scope[ix + 1:]
        out: dict[tuple, float] = {}
        back: dict[tuple, tuple] = {}
        for key, c in tab.items():
            if key[ix] == 1:
                continue  # touched but not witnessed: not an EDS
            nk = key[:ix] + key[ix + 1:]
            if c < out.get(nk, INF):
                out[nk] = c
                back[nk] = key
        work += len(tab)
        if work > limit:
            raise _Budget
        place((new_scope, out, ("elim", h, back)))

    # Decode the chosen gadget edges.
    chosen: list[int] = []
    size = 0.0
    for f in roots:
        tab = f[1]
        key = min(tab, key=tab.__getitem__)
        size += tab[key]
        stack = [(f, key)]
        while stack:
            (scope, _, node), key = stack.pop()
            kind = node[0]
            if kind == "elim":
                _, inner, back = node
                stack.append((inner, back[key]))
            elif kind == "join":
                _, left, right, back = node
                kf, kg = back[key]
                stack.append((left, kf))
                stack.append((right, kg))
            elif 2 in key:  # base edge factor in a "chosen" state
                chosen.append(node[1])
    cover = {owner[e] for e in chosen}
    if core:
        # Vertices of C with a core node in their path keep their reference choice.
        core_vertices = set()
        for a, b in gedges:
            if a in core or b in core:
                core_vertices.add(node_vertex[a])
        cover |= {u for u in core_vertices if u in reference}
    return cover, not core, size


def eds_treewidth_cover(
    adj: Mapping[Any, Iterable[Any]],
    reference_cover: set[Any],
    max_width: int = DEFAULT_MAX_WIDTH,
    budget: float = DEFAULT_DP_BUDGET,
    info: dict | None = None,
) -> set[Any]:
    """Candidate c10: EDS gadget + treewidth DP on the bipartite components.

    ``reference_cover`` (a valid cover) fixes the DP core and covers the
    non-bipartite components. ``info`` (optional dict) receives
    ``bipartite_fraction`` (share of non-isolated vertices in bipartite
    components), ``bipartite_components``, ``exact_components`` (minimum EDS
    of the gadget computed with an empty core), ``core_size`` (gadget nodes),
    ``width``, ``eds_size`` and ``exact`` (every component bipartite and
    solved with an empty core; the EDS is then minimum, which does **not**
    make the decoded cover minimum).
    """
    cover: set[Any] = set()
    n_bip = n_tot = 0
    bip_comps = exact_comps = 0
    eds_size = 0.0
    stats: dict = {}
    for comp, bipartite in _components(adj):
        if len(comp) == 1:
            continue
        n_tot += len(comp)
        if not bipartite:
            cover |= {v for v in comp if v in reference_cover}
            continue
        n_bip += len(comp)
        bip_comps += 1
        try:
            part, exact, size = _solve_component(comp, adj, reference_cover, max_width, budget, stats)
            eds_size += size
        except _Budget:
            part, exact = {v for v in comp if v in reference_cover}, False
            stats["core"] = stats.get("core", 0) + len(comp)
        exact_comps += exact
        cover |= part
    if info is not None:
        info.update({
            "bipartite_fraction": n_bip / n_tot if n_tot else 1.0,
            "bipartite_components": bip_comps,
            "exact_components": exact_comps,
            "core_size": stats.get("core", 0),
            "width": stats.get("width", 0),
            "eds_size": eds_size,
            "exact": bip_comps == exact_comps and n_bip == n_tot,
        })
    return cover
