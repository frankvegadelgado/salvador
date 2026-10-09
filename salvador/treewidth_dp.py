"""Bounded-treewidth dynamic program for vertex cover (candidate c11).

Vertex cover is solvable in time ``O(2^w * w * n)`` on graphs of treewidth
``w``, hence in linear time when ``w`` is bounded. This module realises that
idea with bucket (variable) elimination along a greedy minimum-degree
elimination order, under two limits that keep the whole routine worst-case
``O(n + m)``:

* **width cap** ``max_width``: a vertex is eliminated only while its current
  degree in the fill graph is at most ``max_width``, so every bag has at most
  ``max_width + 1`` vertices and the fill work per vertex is
  ``O(max_width^2)``;
* **work budget** ``budget``: eliminations stop once the total size of the
  dynamic-programming tables, ``sum 2^|bag|``, would exceed
  ``budget * (n + m)``.

The vertices that cannot be eliminated within these limits form the *core*
``R``. Their values are fixed from a given reference cover ``C_ref``
(``r`` is in the cover iff ``r in C_ref``), and the eliminated part is solved
**exactly** conditioned on that choice. Hence:

* if ``R`` is empty (the elimination width of the graph is at most
  ``max_width`` and the budget suffices -- trees, cycles, series-parallel and
  outerplanar graphs, thin grids, ...) the returned cover is a **minimum**
  vertex cover;
* in every case the returned cover is valid and no larger than ``C_ref``,
  because ``C_ref`` restricted to the eliminated part is one feasible
  completion.

The dynamic program works on the complementary independent set: it maximises
the number of eliminated vertices outside the cover.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

DEFAULT_MAX_WIDTH = 10
DEFAULT_DP_BUDGET = 256


def _elimination_order(adj: Mapping[Any, Iterable[Any]], max_width: int, limit: float, states: int = 2):
    """Greedy min-degree elimination with a width cap and a table budget.

    ``states`` is the number of states per variable of the dynamic program
    that will run on the order (2 for vertex cover); a bag of ``k`` vertices
    is charged ``states ** k`` table entries.

    Returns ``(order, scopes, core)``: the eliminated vertices in order, the
    later neighbours of each eliminated vertex at its elimination time (its
    bag minus itself), and the set of vertices left un-eliminated.
    """
    fill = {v: set(adj[v]) - {v} for v in adj}
    deg = {v: len(fill[v]) for v in fill}
    buckets: list[list[Any]] = [[] for _ in range(max_width + 1)]
    for v, d in deg.items():
        if d <= max_width:
            buckets[d].append(v)
    eliminated: set[Any] = set()
    order: list[Any] = []
    scopes: dict[Any, list[Any]] = {}
    table_work = 0
    low = 0
    while True:
        while low <= max_width and not buckets[low]:
            low += 1
        if low > max_width:
            break
        v = buckets[low].pop()
        if v in eliminated or deg[v] != low:
            continue  # stale entry
        nbrs = list(fill[v])
        cost = states ** (len(nbrs) + 1)
        if table_work + cost > limit:
            break
        table_work += cost
        eliminated.add(v)
        order.append(v)
        scopes[v] = nbrs
        # Make the later neighbourhood a clique (fill edges), then drop v.
        for i, a in enumerate(nbrs):
            fa = fill[a]
            fa.discard(v)
            for b in nbrs[i + 1:]:
                if b not in fa:
                    fa.add(b)
                    fill[b].add(a)
        for a in nbrs:
            d = len(fill[a])
            if d != deg[a]:
                deg[a] = d
            if d <= max_width:
                buckets[d].append(a)
                if d < low:
                    low = d
        fill[v] = set()
    core = {v for v in adj if v not in eliminated}
    return order, scopes, core


def treewidth_vertex_cover(
    adj: Mapping[Any, Iterable[Any]],
    reference_cover: set[Any],
    max_width: int = DEFAULT_MAX_WIDTH,
    budget: float = DEFAULT_DP_BUDGET,
    info: dict | None = None,
) -> set[Any]:
    """Exact DP on the low-width part, ``reference_cover`` on the core.

    ``info`` (optional dict) receives ``core_size``, ``width`` (largest bag
    minus one) and ``exact`` (``True`` when the core is empty, i.e. the result
    is a minimum vertex cover).
    """
    n_plus_m = len(adj) + sum(len(set(a)) for a in adj.values()) // 2
    order, scopes, core = _elimination_order(adj, max_width, budget * max(n_plus_m, 1))
    pos = {v: i for i, v in enumerate(order)}
    # Core vertices are fixed: x_r = 1 (independent) iff r is not in the reference cover.
    fixed_free = {r for r in core if r not in reference_cover}

    # Per eliminated vertex: variable scope (later, non-core), message tables, choices.
    var_scope: dict[Any, list[Any]] = {}
    choice: dict[Any, list[int]] = {}
    inbox: dict[Any, list[tuple[list[Any], list[float]]]] = {v: [] for v in order}
    NEG = float("-inf")

    for v in order:
        S = [w for w in scopes[v] if w not in core]
        S.sort(key=pos.__getitem__)
        var_scope[v] = S
        k = len(S)
        idx_of = {w: i for i, w in enumerate(S)}
        # Original-edge constraints handled here: neighbours of v not yet eliminated.
        later_orig = [w for w in adj[v] if w != v and (w in core or pos[w] > pos[v])]
        blocked_by_core = any(w in fixed_free for w in later_orig)
        orig_mask = 0
        for w in later_orig:
            if w in idx_of:
                orig_mask |= 1 << idx_of[w]
        # Pre-compute index maps of incoming messages.
        msgs = []
        for scope, table in inbox[v]:
            bits = [(-1 if w == v else idx_of[w]) for w in scope]
            msgs.append((bits, table))
        size = 1 << k
        out = [0.0] * size
        ch = [0] * size
        for a in range(size):
            best_val, best_x = NEG, 0
            for xv in (1, 0):
                if xv == 1 and (blocked_by_core or (a & orig_mask)):
                    continue
                val = float(xv)
                for bits, table in msgs:
                    j = 0
                    for t, b in enumerate(bits):
                        if (xv if b < 0 else (a >> b) & 1):
                            j |= 1 << t
                    val += table[j]
                    if val == NEG:
                        break
                if val > best_val:
                    best_val, best_x = val, xv
            out[a] = best_val
            ch[a] = best_x
        choice[v] = ch
        if S:
            inbox[S[0]].append((S, out))

    x: dict[Any, int] = {}
    for v in reversed(order):
        a = 0
        for i, w in enumerate(var_scope[v]):
            if x[w]:
                a |= 1 << i
        x[v] = choice[v][a]

    cover = {v for v in order if not x[v]} | (core & reference_cover)
    if info is not None:
        info["core_size"] = len(core)
        info["width"] = max((len(scopes[v]) for v in order), default=0)
        info["exact"] = not core
    return cover
