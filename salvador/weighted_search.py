"""
Weighted swap search for the minimum vertex cover (the move set of
:func:`salvador.local_search.improve_vertex_cover`).

The search keeps a set ``C`` of a fixed size ``k`` that may leave edges
uncovered, and minimises the weight of the uncovered edges:

    cost(C) = sum of w[e] over the edges e with no end in C.

``C`` is a vertex cover exactly when ``cost(C) = 0``; it is then recorded and one
vertex is removed to try ``k - 1``. This is the scheme of NuMVC (Cai, Su, Luo and
Sattar, "NuMVC: An efficient local search algorithm for minimum vertex cover",
J. Artif. Intell. Res. 46, 2013): edge weighting, two-stage exchange and
configuration checking.

Quantities (maintained incrementally):

* ``w[e] >= 1`` -- edge weights; after every step the weight of every uncovered
  edge grows by one, so edges that stay uncovered become more and more important
  (escape from local minima);
* ``dscore[v]`` -- the decrease of ``cost`` when ``v`` changes side: for ``v``
  outside ``C``, the weight of the uncovered edges at ``v``; for ``v`` in ``C``,
  minus the weight of the edges covered by ``v`` alone;
* ``conf[v]`` -- configuration checking: a vertex removed from ``C`` may come
  back only after one of its neighbours changed side.

A step: (1) remove the vertex of ``C`` of largest ``dscore`` among ``t`` sampled
vertices of ``C`` (best from multiple selection), never the vertex added in the
previous step; (2) pick a random uncovered edge and add its endpoint of larger
``dscore`` allowed by configuration checking, ties by age; (3) add one to the
weight of every uncovered edge.

Every update costs the degree of the changed vertex; all of it is charged to a
work counter and the search stops at a fixed work limit, so it runs in
``O(n + m)`` time for a limit of the form ``max(budget * (n + m), floor)``. Weights
are halved (and the scores recomputed, ``O(n + m)``) when their mean exceeds a
threshold (the weight cap). The best cap depends on the graph, so a search is split
into :data:`SEGMENTS` segments that cycle through :data:`WEIGHT_LIMITS`, each
restarted with fresh weights from the best cover found so far. The search never
returns a cover larger than its input, which must be a vertex cover.
"""

from __future__ import annotations

import random

#: Vertices of C sampled for the removal (best from multiple selection).
SAMPLE: int = 50
#: Mean weights above which the weights are halved; the segments of a search cycle through them.
WEIGHT_LIMITS: tuple[float, ...] = (10.0, 100.0, 30.0, 300.0)
#: Segments of one search (each with its own weight cap, restarted from the best cover).
SEGMENTS: int = 4


class _IndexedSet:
    """Set of integers with O(1) insert, delete and random choice."""

    __slots__ = ("items", "pos")

    def __init__(self, n: int):
        self.items: list[int] = []
        self.pos = [-1] * n

    def add(self, v: int) -> None:
        if self.pos[v] == -1:
            self.pos[v] = len(self.items)
            self.items.append(v)

    def discard(self, v: int) -> None:
        p = self.pos[v]
        if p != -1:
            last = self.items.pop()
            if last != v:
                self.items[p] = last
                self.pos[last] = p
            self.pos[v] = -1

    def __len__(self) -> int:
        return len(self.items)


def _search(adj: list[list[int]], in_c: list[bool], limit: int, seed: int,
            weight_limit: float) -> list[bool]:
    """Improve the vertex cover ``in_c`` by the weighted swap search within ``limit`` units
    of work. Returns a vertex cover no larger than ``in_c``."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    best = list(in_c)
    if m == 0:
        return best
    rng = random.Random(seed)
    work = 0

    # Edge identifiers, parallel to the adjacency lists (deterministic O(n + m): every edge
    # gets its id at its smaller end, the larger end finds its position with a marker array).
    eid: list[list[int]] = [[0] * len(a) for a in adj]
    ends: list[tuple[int, int]] = []
    pending: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for v in range(n):
        for j, y in enumerate(adj[v]):
            if v < y:
                e = len(ends)
                ends.append((v, y))
                eid[v][j] = e
                pending[y].append((v, e))
    where = [0] * n
    for y in range(n):
        if pending[y]:
            for j, x in enumerate(adj[y]):
                where[x] = j
            for v, e in pending[y]:
                eid[y][where[v]] = e
    del pending, where
    work += 4 * (n + m)

    num_edges = len(ends)
    C = list(in_c)
    we = [1] * num_edges
    total_w = num_edges
    dscore = [0] * n
    conf = [True] * n
    age = [0] * n
    cset = _IndexedSet(n)
    uncov = _IndexedSet(num_edges)
    for v in range(n):
        if C[v]:
            cset.add(v)
    for e, (a, b) in enumerate(ends):
        if not C[a] and not C[b]:
            uncov.add(e)

    def recompute() -> None:
        for v in range(n):
            dscore[v] = 0
        for e, (a, b) in enumerate(ends):
            ca, cb = C[a], C[b]
            if ca and cb:
                continue
            w = we[e]
            if ca:
                dscore[a] -= w
            elif cb:
                dscore[b] -= w
            else:
                dscore[a] += w
                dscore[b] += w

    recompute()
    work += 3 * (n + m)

    def flip(z: int) -> None:
        """Move ``z`` to the other side, updating the uncovered edges and the scores."""
        nonlocal work
        entering = not C[z]
        C[z] = entering
        if entering:
            cset.add(z)
        else:
            cset.discard(z)
        az = adj[z]
        ez = eid[z]
        for j, y in enumerate(az):
            conf[y] = True
            w = we[ez[j]]
            if C[y]:
                # the edge was covered by y alone (z leaving) / is now covered by both (z entering)
                dscore[y] += w if entering else -w
            else:
                if entering:
                    dscore[y] -= w
                    uncov.discard(ez[j])
                else:
                    dscore[y] += w
                    uncov.add(ez[j])
        dscore[z] = -dscore[z]
        work += 2 * len(az) + 2

    best_size = sum(best)
    step = 0
    last_added = -1
    while work <= limit and len(cset) > 0:
        if len(uncov) == 0:
            if len(cset) < best_size:
                best_size = len(cset)
                best = list(C)
                work += n
            # Try one vertex fewer: remove the vertex whose removal costs least.
            x = max(cset.items, key=lambda v: dscore[v])
            work += len(cset)
            flip(x)
            conf[x] = False
            continue
        step += 1
        items = cset.items
        cand = items if len(items) <= SAMPLE else [items[rng.randrange(len(items))] for _ in range(SAMPLE)]
        x = -1
        for c in cand:
            if c == last_added:
                continue
            if x == -1 or dscore[c] > dscore[x] or (dscore[c] == dscore[x] and age[c] < age[x]):
                x = c
        work += len(cand)
        if x != -1:
            flip(x)
            conf[x] = False
            age[x] = step
        a, b = ends[uncov.items[rng.randrange(len(uncov))]]
        if conf[a] != conf[b]:
            v = a if conf[a] else b
        else:
            v = a if (dscore[a] > dscore[b] or (dscore[a] == dscore[b] and age[a] < age[b])) else b
        flip(v)
        age[v] = step
        last_added = v
        # Weights of the uncovered edges.
        for e in uncov.items:
            we[e] += 1
            p, q = ends[e]
            dscore[p] += 1
            dscore[q] += 1
        total_w += len(uncov)
        work += len(uncov) + 1
        if total_w > weight_limit * num_edges:
            for e in range(num_edges):
                we[e] = max(1, we[e] // 2)
            total_w = sum(we)
            recompute()
            work += 4 * (n + m)
    return best


def weighted_cover_search(adj: list[list[int]], in_c: list[bool], limit: int, seed: int) -> list[bool]:
    """Improve the vertex cover ``in_c`` by the weighted swap search, in segments: the work
    ``limit`` is split into :data:`SEGMENTS` equal parts, segment ``i`` uses the weight cap
    ``WEIGHT_LIMITS[(seed + i) % len(WEIGHT_LIMITS)]`` and starts, with fresh weights, from the
    best cover found so far. Returns a vertex cover no larger than ``in_c``."""
    part = max(limit, 0) // SEGMENTS
    best = list(in_c)
    for i in range(SEGMENTS):
        cap = WEIGHT_LIMITS[(seed + i) % len(WEIGHT_LIMITS)]
        res = _search(adj, best, part, seed + 7919 * i, cap)
        if sum(res) < sum(best):
            best = res
    return best
