"""Edge-dominating-set gadget solved by a treewidth DP (candidate c10).

Gadget
------
Every edge ``{u, v}`` of ``G`` becomes an *edge-node* ``(u, v)``. If
``n_0, n_1, ..., n_{d-1}`` are the neighbours of ``u`` (adjacency order), ``u``
gets *incidence nodes* ``(u, 0), ..., (u, d-2)`` and the gadget edges

    ((u, k), (u, n_k))   and   ((u, k), (u, n_{k+1}))      for k = 0, ..., d-2,

so the edge-nodes of ``u`` form a path ``(u,n_0) - (u,0) - (u,n_1) - (u,1) -
... - (u,n_{d-1})``: the *path edges* of ``u``, its occurrences. A vertex of
degree 1 gets one incidence node ``(u, 0)`` joined to its only edge-node. The
edge-node ``(u, v)`` is shared by the paths of ``u`` and ``v``. Every gadget
edge joins an incidence node to an edge-node, so the gadget is bipartite.

Weights and labels
------------------
Every vertex ``u`` carries a *cover label* ``s_u`` (in / out) along its path:
every incidence node of ``u`` holds ``s_u``, every edge-node ``(u, v)`` holds
``s_u`` and ``s_v``, and a gadget edge forces its two endpoints to agree on the
label of its vertex. With ``p_u`` path edges, ``m`` edges in the component and
``K = max(max p_u, m) + 1``:

* **all or none** -- a path edge of ``u`` costs its weight when ``s_u = in``
  and it is chosen; choosing it with ``s_u = out``, or not choosing it with
  ``s_u = in``, costs the penalty ``P`` (larger than every other cost
  together). So an optimal solution chooses either all the occurrences of a
  vertex or none of them;
* **one unit per vertex** -- the path edges of ``u`` weigh ``K`` in total (the
  first one ``K - p_u + 1``, the others 1), whatever the degree of ``u``; this
  removes the bias of the unweighted gadget towards low-degree vertices;
* **overlap** -- an edge-node whose two vertices are both in costs 1: its edge
  is covered twice where one vertex suffices. The overlap is at most
  ``m < K``, so it only breaks ties among minimum covers and never makes a
  cover larger: the dominating solution stays optimal.

Exactness. A solution without penalty chooses exactly the path edges of the
vertices labelled in, ``C``. It is an EDS iff ``C`` is a vertex cover: the path
edges of ``C`` touch every edge-node of an edge with an endpoint in ``C`` (and
so dominate every gadget edge there), while for an uncovered edge ``uv`` the
gadget edge ``((u, k), (u, v))`` is untouched. Its weight is
``K |C| + overlap(C)``. Hence the minimum-weight solution labels a **minimum
vertex cover**, with the fewest doubly covered edges among minimum covers.

Dynamic program
---------------
Bucket elimination along a greedy minimum-degree elimination order of the
gadget (``salvador.treewidth_dp._elimination_order``). A node state is the
EDS state ``t`` -- ``0`` untouched, ``1`` touched with the chosen edge
provided elsewhere, ``2`` touched by a chosen edge of this factor (witness)
-- together with the labels (one on an incidence node, two on an edge-node;
6 and 12 states). Factors are joined with "same membership, same labels,
witness if either", a node is eliminated only if it is untouched or
witnessed, and back-pointers recover the chosen edges and the labels; the
cover is read from the labels.

Bags and core
-------------
**No bag has more than ``MAX_BAG = 6`` nodes**: a node is eliminated only while
its fill degree is at most ``max_width <= MAX_BAG - 1``, every joined table is
checked against ``MAX_BAG``, and the tables (product of the state counts of
a bag) must fit in ``max(budget * (n + m), MIN_DP_WORK)``. Nodes left over
form the core, fixed from a reference cover ``C``: its EDS "all path edges of
``C``" with labels ``s_u = [u in C]``; the rest is solved exactly given that
choice (a core node of ``u`` fixes the label of ``u``). With an empty core the
result is a minimum vertex cover of the component (``exact``). Non-bipartite
components take the reference cover, as before. The result is pruned and, if
it is larger than the reference, the reference is used.

Running time: ``O(n + m)`` for the gadget and at most
``max(budget * (n + m), MIN_DP_WORK)`` DP work; worst-case ``O(n + m)`` for
fixed limits.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping

from .treewidth_dp import _elimination_order

MAX_BAG = 6
DEFAULT_MAX_WIDTH = MAX_BAG - 1
DEFAULT_DP_BUDGET = 256
MIN_DP_WORK = 2_000_000  # floor of the table budget, so that small components are solved exactly

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

    Returns a dict with ``n_nodes``, ``gadj`` (adjacency), ``gedges`` (pairs
    ``(incidence_node, edge_node)``), ``owner[e]`` (the vertex whose path edge
    ``e`` is), ``node_vertex[x]`` (the vertex of an incidence node, ``None`` for
    edge-nodes), ``ends[x]`` (the two vertices ``(p, q)`` of an edge-node),
    ``side[e]`` (0 if ``owner[e]`` is ``ends[edge_node][0]``, else 1),
    ``weights[e]``, ``K`` and ``n_edge_nodes``.
    """
    eid: dict[frozenset, int] = {}
    ends: list[tuple[Any, Any]] = []
    nbrs: dict[Any, list[Any]] = {}
    for u in comp:
        lst = [w for w in adj[u] if w != u]
        nbrs[u] = lst
        for w in lst:
            key = frozenset((u, w))
            if key not in eid:
                eid[key] = len(eid)
                ends.append((u, w))
    n_edge_nodes = len(eid)
    nxt = n_edge_nodes
    gedges: list[tuple[int, int]] = []
    owner: list[Any] = []
    node_vertex: list[Any] = [None] * n_edge_nodes
    first_edge: dict[Any, int] = {}
    path_count: dict[Any, int] = {}
    for u in comp:
        lst = nbrs[u]
        d = len(lst)
        if d == 0:
            continue
        pairs = [(lst[0],)] if d == 1 else [(lst[k], lst[k + 1]) for k in range(d - 1)]
        for ws in pairs:
            x = nxt
            nxt += 1
            node_vertex.append(u)
            for w in ws:
                first_edge.setdefault(u, len(gedges))
                gedges.append((x, eid[frozenset((u, w))]))
                owner.append(u)
                path_count[u] = path_count.get(u, 0) + 1
    K = max(max(path_count.values(), default=0), n_edge_nodes) + 1
    weights = [1] * len(gedges)
    for u, e in first_edge.items():
        weights[e] = K - path_count[u] + 1
    side = [0 if ends[b][0] == owner[e] else 1 for e, (_, b) in enumerate(gedges)]
    gadj: dict[int, set[int]] = {x: set() for x in range(nxt)}
    for a, b in gedges:
        gadj[a].add(b)
        gadj[b].add(a)
    return {"n_nodes": nxt, "gadj": gadj, "gedges": gedges, "owner": owner, "node_vertex": node_vertex,
            "ends": ends, "side": side, "weights": weights, "K": K, "n_edge_nodes": n_edge_nodes}


def _solve_component(comp, adj, reference, max_width, budget, stats):
    """Minimum-weight labelled EDS of the gadget of one component, decoded to a cover.

    Returns ``(cover_of_component, exact, weight)``.
    """
    g = build_gadget(comp, adj)
    gadj, gedges, owner, node_vertex, ends, side, weights = (g["gadj"], g["gedges"], g["owner"], g["node_vertex"],
                                                             g["ends"], g["side"], g["weights"])
    n_edge_nodes = g["n_edge_nodes"]
    penalty = sum(weights) + n_edge_nodes + 1  # more than every weight and overlap together
    states = {x: (12 if x < n_edge_nodes else 6) for x in gadj}
    limit = max(budget * max(g["n_nodes"] + len(gedges), 1), MIN_DP_WORK)
    order, scopes, core = _elimination_order(gadj, max_width, limit, states=states)
    stats["width"] = max(stats.get("width", 0), max((len(s) for s in scopes.values()), default=0))
    if core:
        stats["core"] = stats.get("core", 0) + len(core)
    pos = {x: p for p, x in enumerate(order)}
    in_ref = {u: int(u in reference) for u in comp}

    def labels_of(x):
        if x < n_edge_nodes:
            return [(a, b) for a in (0, 1) for b in (0, 1)]
        return [(a,) for a in (0, 1)]

    def domain(x):
        return [(t,) + lab for t in (0, 1, 2) for lab in labels_of(x)]

    def fixed_state(y):
        """Class of a core node in the reference: (touched, labels)."""
        if y < n_edge_nodes:
            p, q = ends[y]
            return bool(in_ref[p] or in_ref[q]), (in_ref[p], in_ref[q])
        return bool(in_ref[node_vertex[y]]), (in_ref[node_vertex[y]],)

    def cost(e, sa, sb):
        """Cost of gadget edge ``e`` with its incidence node in state ``sa`` and its edge-node in ``sb``."""
        s_u = sa[1]
        if sb[1 + side[e]] != s_u:
            return None  # the vertex keeps one label along its path
        c = _EDGE_TABLE.get((sa[0], sb[0]))
        if c is None:
            return None  # undominated
        if c:  # chosen
            return weights[e] if s_u else weights[e] + penalty
        return penalty if s_u else 0  # an occurrence of a vertex in the cover left out

    # Reference EDS: all path edges of C; its edges with a core endpoint are forced.
    reference_edges = {e for e in range(len(gedges)) if in_ref[owner[e]]}
    forced = {e for e in reference_edges if gedges[e][0] in core or gedges[e][1] in core}

    buckets: dict[int, list] = {x: [] for x in order}
    roots: list = []
    work = 0
    weight = 0.0
    # Edges between two core nodes keep their reference choice.
    for e, (a, b) in enumerate(gedges):
        if a in core and b in core and e in reference_edges:
            weight += weights[e]

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
            x = b if a_core else a
            y = a if a_core else b
            y_touched, y_labels = fixed_state(y)
            tab: dict[tuple, float] = {}
            for sx in domain(x):
                for ty in ((1, 2) if y_touched else (0, 1)):
                    sy = (ty,) + y_labels
                    sa, sb = (sy, sx) if a_core else (sx, sy)
                    c = cost(e, sa, sb)
                    if c is None:
                        continue
                    is_chosen = 2 in (sa[0], sb[0])
                    if e in forced and not is_chosen:
                        continue
                    if is_chosen and sx[0] == 1:
                        continue  # x is witnessed by this chosen edge
                    if not is_chosen and ty == (2 if y_touched else 1):
                        continue  # y's state comes from the reference unless this edge is chosen
                    if c < tab.get((sx,), INF):
                        tab[(sx,)] = c
            place(((x,), tab, ("edge", e)))
        else:
            tab = {}
            for sa in domain(a):
                for sb in domain(b):
                    c = cost(e, sa, sb)
                    if c is not None:
                        tab[(sa, sb)] = c
            place(((a, b), tab, ("edge", e)))
    # Overlap: an edge-node whose two vertices are both in the cover.
    for x in range(n_edge_nodes):
        if x in core:
            p, q = ends[x]
            weight += in_ref[p] and in_ref[q]
            continue
        place(((x,), {(sx,): (1 if sx[1] and sx[2] else 0) for sx in domain(x)}, ("overlap", x)))

    def join(f, h):
        nonlocal work
        sf, tf, _ = f
        sg, tg, _ = h
        shared = [v for v in sf if v in sg]
        fi = [sf.index(v) for v in shared]
        gi = [sg.index(v) for v in shared]
        g_only = [j for j, v in enumerate(sg) if v not in sf]
        scope = tuple(sf) + tuple(sg[j] for j in g_only)
        index: dict[tuple, list] = {}
        for kg, cg in tg.items():
            index.setdefault(tuple((kg[j][0] != 0,) + kg[j][1:] for j in gi), []).append((kg, cg))
        out: dict[tuple, float] = {}
        back: dict[tuple, tuple] = {}
        for kf, cf in tf.items():
            lst = index.get(tuple((kf[j][0] != 0,) + kf[j][1:] for j in fi))
            if not lst:
                continue
            work += len(lst)
            for kg, cg in lst:
                key = list(kf)
                for a, b in zip(fi, gi):
                    if kg[b][0] == 2:
                        key[a] = kg[b]
                key = tuple(key) + tuple(kg[j] for j in g_only)
                c = cf + cg
                if c < out.get(key, INF):
                    out[key] = c
                    back[key] = (kf, kg)
        if work > limit:
            raise _Budget
        return (scope, out, ("join", f, h, back))

    for x in order:
        fs = buckets.pop(x)
        if not fs:
            continue
        h = fs[0]
        for f in fs[1:]:
            h = join(h, f)
        scope, tab, _ = h
        if len(scope) > MAX_BAG:  # cannot happen: the bucket of x lies in x and at most max_width later nodes
            raise AssertionError(f"bag of {len(scope)} nodes exceeds MAX_BAG = {MAX_BAG}")
        stats["bag"] = max(stats.get("bag", 0), len(scope))
        ix = scope.index(x)
        new_scope = scope[:ix] + scope[ix + 1:]
        out: dict[tuple, float] = {}
        back: dict[tuple, tuple] = {}
        for key, c in tab.items():
            if key[ix][0] == 1:
                continue  # touched but not witnessed: not an EDS
            nk = key[:ix] + key[ix + 1:]
            if c < out.get(nk, INF):
                out[nk] = c
                back[nk] = key
        work += len(tab)
        if work > limit:
            raise _Budget
        place((new_scope, out, ("elim", h, back)))

    # Decode the labels (the cover) and the weight.
    label: dict[Any, int] = {}
    for f in roots:
        tab = f[1]
        if not tab:
            raise _Budget  # infeasible under the fixed core; cannot happen for a valid reference
        key = min(tab, key=tab.__getitem__)
        weight += tab[key]
        stack = [(f, key)]
        while stack:
            (scope, _, node), key = stack.pop()
            kind = node[0]
            if kind == "elim":
                stack.append((node[1], node[2][key]))
            elif kind == "join":
                kf, kg = node[3][key]
                stack.append((node[1], kf))
                stack.append((node[2], kg))
            else:
                for v, st in zip(scope, key):
                    if v < n_edge_nodes:
                        p, q = ends[v]
                        label[p], label[q] = st[1], st[2]
                    else:
                        label[node_vertex[v]] = st[1]
    for y in core:  # a core node fixes the label of its vertex (or vertices) to the reference
        if y < n_edge_nodes:
            for u in ends[y]:
                label[u] = in_ref[u]
        else:
            label[node_vertex[y]] = in_ref[node_vertex[y]]
    cover = {u for u in comp if label.get(u, in_ref[u])}
    stats["penalized"] = stats.get("penalized", 0) + int(weight >= penalty)
    return cover, not core, weight


def eds_treewidth_cover(
    adj: Mapping[Any, Iterable[Any]],
    reference_cover: set[Any],
    max_width: int = DEFAULT_MAX_WIDTH,
    budget: float = DEFAULT_DP_BUDGET,
    info: dict | None = None,
) -> set[Any]:
    """Candidate c10: labelled, weighted EDS gadget + treewidth DP on the bipartite components.

    ``reference_cover`` (a valid cover) fixes the DP core and covers the
    non-bipartite components. ``max_width`` is clamped to ``MAX_BAG - 1``, so
    no bag has more than ``MAX_BAG`` nodes. ``info`` (optional dict) receives
    ``bipartite_fraction`` (share of non-isolated vertices in bipartite
    components), ``bipartite_components``, ``exact_components`` (solved with
    an empty core), ``core_size`` (gadget nodes), ``width`` (largest fill
    degree), ``max_bag`` (largest bag built, at most ``MAX_BAG``), ``eds_size``
    (total weight of the labelled EDS, ``K |C| + overlap`` per component) and
    ``exact`` (every component bipartite and solved with an empty core: the
    cover is then a **minimum** vertex cover).
    """
    max_width = min(max_width, MAX_BAG - 1)
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
        if any(u not in part and w not in part for u in comp for w in adj[u] if w != u):
            part, exact = {v for v in comp if v in reference_cover}, False  # safety net: never an invalid cover
        exact_comps += exact
        cover |= part
    if info is not None:
        info.update({
            "bipartite_fraction": n_bip / n_tot if n_tot else 1.0,
            "bipartite_components": bip_comps,
            "exact_components": exact_comps,
            "core_size": stats.get("core", 0),
            "width": stats.get("width", 0),
            "max_bag": stats.get("bag", 0),
            "eds_size": eds_size,
            "exact": bip_comps == exact_comps and n_bip == n_tot,
        })
    return cover
