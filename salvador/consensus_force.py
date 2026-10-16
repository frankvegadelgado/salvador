"""Consensus forcing for vertex cover (candidate c14).

Idea
----
The strategies c1..c13 are different heuristics, and a vertex that all of
them (or most of them) put into their covers is likely to belong to a minimum
cover. Candidate c14 *forces* such vertices into the cover -- equivalently, it
puts a self-loop on each of them, which no cover can leave uncovered -- deletes
them from the graph, solves the residual graph again with linear-time solvers
(pruned NetworkX minimum maximal matching, Min-to-Min and the subdivision reduction c11 with
``max+fold``), and returns the forced vertices together with the residual
cover, pruned on the whole graph.

Rounds and stopping rule
------------------------
Round ``k`` (at most ``rounds``, default 3) forces the vertices that lie in at
least a fraction ``q_k`` of the covers in the pool, with ``q = 1, 0.9, 0.75``:
first the vertices on which *all* covers agree, then the ones on which most
agree. Every new cover joins the pool. Before every round the best cover is
checked against a certified lower bound ``L <= tau(G)``, computed in linear
time as the larger of the NetworkX minimum maximal matching approximation
(the matching of c1) and a Karp-Sipser matching
(any matching is a lower bound on tau). The rounds stop as soon as
``|best| <= target * L``, which proves the ratio ``|best| / tau <= target`` on
this graph; they also stop when a round would force the same set again.

The default target is 1 (stop only when the cover is certified optimal). A
target of R0 = 1.999999999 would stop before the first round on every graph
whose matching has fewer than 10^9 edges: the pruned maximal-matching cover
c1 already has |c1| <= 2|M| - 1 (Theorem "2 tau - r"), so
|best| / L <= 2 - 1/|M| <= R0 there, and the rounds would never run.

What is and is not guaranteed
-----------------------------
* The result is always a valid vertex cover, and the ensemble keeps the
  smallest candidate, so c14 can never make Salvador's answer worse.
* When the stopping rule fires, the ratio <= R0 is certified on that graph by
  the matching lower bound.
* Nothing guarantees that the forced vertices lie in a minimum cover: all the
  heuristics may agree on a wrong vertex. If a forced vertex lies in no
  minimum cover, forcing it costs one vertex. So c14 is a heuristic; it does
  not prove ratio <= R0 on graphs where the certificate fails, and on the hard
  instances of the hardness reductions the matching bound is far below tau,
  so the certificate cannot prove a ratio much better than the theorem.

Cost: one pass over the covers to count votes, then per round one residual
graph and three linear-time solvers: O(rounds * (n + m)).
"""

from __future__ import annotations

from collections import deque
from math import ceil
from typing import Any, Callable, Iterable, Mapping

import networkx as nx

R0 = 1.999999999
DEFAULT_ROUNDS = 3
DEFAULT_QUOTAS = (1.0, 0.9, 0.75)

Adj = dict[Any, set[Any]]


def greedy_matching_size(adj: Mapping[Any, Iterable[Any]]) -> int:
    """Size of the NetworkX minimum maximal matching approximation of ``adj`` (the matching of c1)."""
    G = nx.Graph()
    G.add_edges_from((u, v) for u, nb in adj.items() for v in nb if v != u)
    return len(nx.approximation.min_maximal_matching(G))


def karp_sipser_matching_size(adj: Mapping[Any, Iterable[Any]]) -> int:
    """Karp-Sipser: match degree-1 vertices first, otherwise any edge; O(n + m)."""
    A = {v: set(nb) - {v} for v, nb in adj.items()}
    deg = {v: len(nb) for v, nb in A.items()}
    alive = {v for v in A if deg[v] > 0}
    leaves = deque(v for v in alive if deg[v] == 1)
    order = deque(alive)
    size = 0

    def remove(x: Any) -> None:
        alive.discard(x)
        for y in A[x]:
            if y in alive:
                deg[y] -= 1
                if deg[y] == 1:
                    leaves.append(y)
                elif deg[y] == 0:
                    alive.discard(y)

    while alive:
        u = None
        while leaves:
            c = leaves.popleft()
            if c in alive and deg[c] == 1:
                u = c
                break
        if u is None:
            while order and order[0] not in alive:
                order.popleft()
            if not order:
                break
            u = order.popleft()
        v = next((y for y in A[u] if y in alive), None)
        if v is None:
            alive.discard(u)
            continue
        size += 1
        remove(u)
        remove(v)
    return size


def lower_bound(adj: Mapping[Any, Iterable[Any]]) -> int:
    """Certified lower bound on tau: the larger of two matchings."""
    return max(greedy_matching_size(adj), karp_sipser_matching_size(adj))


def consensus_force_cover(
    adj: Mapping[Any, Iterable[Any]],
    covers: list[set[Any]],
    inner: Callable[[Adj], set[Any]],
    prune: Callable[[Mapping[Any, Iterable[Any]], set[Any]], set[Any]],
    rounds: int = DEFAULT_ROUNDS,
    quotas: tuple[float, ...] = DEFAULT_QUOTAS,
    target: float = 1.0,
    info: dict | None = None,
) -> set[Any]:
    """Smallest cover found by forcing consensus vertices of ``covers`` and re-solving the rest.

    ``inner(adj)`` returns a vertex cover of the graph ``adj``; ``prune(adj, C)``
    removes redundant vertices. ``info`` (optional) receives ``lower_bound``,
    ``rounds_run``, ``forced`` (vertices forced in the last round run),
    ``certified`` (whether |result| <= R0 * lower_bound) and ``gain`` (best of
    ``covers`` minus the result, possibly 0); ``certified`` refers to R0 =
    1.999999999, and ``certified_ratio`` is |result| / lower_bound.
    """
    A: Adj = {v: set(nb) - {v} for v, nb in adj.items()}
    pool = [set(c) for c in covers if c is not None]
    best = min(pool, key=len) if pool else set(v for v in A if A[v])
    start = len(best)
    L = lower_bound(A)
    rounds_run = 0
    forced_last = 0
    previous: set[Any] | None = None
    for k in range(rounds):
        if len(best) <= target * L:
            break
        q = quotas[min(k, len(quotas) - 1)]
        votes: dict[Any, int] = {}
        for c in pool:
            for v in c:
                votes[v] = votes.get(v, 0) + 1
        need = max(1, ceil(q * len(pool)))
        S = {v for v, c in votes.items() if c >= need}
        if not S or S == previous:
            continue
        previous = S
        rounds_run += 1
        forced_last = len(S)
        residual: Adj = {}
        for v, nb in A.items():
            if v in S:
                continue
            r = {w for w in nb if w not in S}
            if r:
                residual[v] = r
        cover = prune(A, S | inner(residual))
        pool.append(cover)
        if len(cover) < len(best):
            best = cover
    if info is not None:
        info.update({"lower_bound": L, "rounds_run": rounds_run, "forced": forced_last,
                     "certified": len(best) <= R0 * L, "gain": start - len(best),
                     "certified_ratio": (len(best) / L) if L else 1.0})
    return set(best)
