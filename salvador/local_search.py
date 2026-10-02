"""Budgeted (1,2)-swap iterated local search for vertex cover (candidate c9).

A vertex cover ``C`` is the complement of an independent set ``S = V \\ C``,
so every independent vertex gained is a cover vertex saved. This module
improves a given cover by improving its complementary independent set with
the (1,2)-swap neighbourhood of Andrade, Resende and Werneck ("Fast local
search for the maximum independent set problem", J. Heuristics 18, 2012):

* ``tight[v]`` (for ``v`` not in ``S``) counts the neighbours of ``v`` in ``S``;
  ``v`` is *free* when ``tight[v] == 0`` and *1-tight* when it is 1.
* A **(1,2)-swap** removes one solution vertex ``x`` and inserts two
  non-adjacent 1-tight neighbours ``v, w`` of ``x``: ``|S|`` grows by one, so
  ``|C|`` shrinks by one. Free vertices exposed by the swap are inserted too.
* An **iterated local search** step forces a non-solution vertex ``u`` into
  ``S`` (evicting its solution neighbours), re-inserts free vertices and
  runs (1,2)-swaps around the change. The step is kept when ``|S|`` does not
  decrease and undone otherwise, through an undo log whose cost is
  proportional to the work of the step.

Linear-time guarantee
---------------------
Every primitive (insert, remove, swap test, perturbation) is charged its
exact adjacency-scan cost to a *work counter*, and the search stops as soon
as that counter exceeds ``budget * (n + m)`` for a constant ``budget``. With
the O(n + m) initialisation, the whole routine is worst-case O(n + m) for
any fixed ``budget``. The swap test for ``x`` is O(deg(x) + sum of deg(v))
over its 1-tight neighbours ``v``: when looking for a partner ``w`` of ``v``
the scan of the 1-tight list stops at the first non-neighbour of ``v``, so it
never costs more than ``deg(v) + 1``.

The routine never returns a worse cover than its input: the best independent
set seen is tracked and the output is its complement, which is always a
valid (and minimal) vertex cover. The pseudo-random choices use a fixed seed,
so results are reproducible.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Iterable, Mapping

DEFAULT_BUDGET = 100
DEFAULT_SEED = 20260627


class _IndependentSet:
    __slots__ = ("adj", "in_s", "tight", "size", "work", "log", "logging")

    def __init__(self, adj: list[list[int]], members: Iterable[int]) -> None:
        n = len(adj)
        self.adj = adj
        self.in_s = [False] * n
        self.tight = [0] * n
        self.size = 0
        self.work = 0
        self.log: list[tuple[bool, int]] = []
        self.logging = False
        for v in members:
            self.insert(v)

    def insert(self, v: int) -> None:
        self.in_s[v] = True
        self.size += 1
        nb = self.adj[v]
        self.work += len(nb) + 1
        for y in nb:
            self.tight[y] += 1
        if self.logging:
            self.log.append((True, v))

    def remove(self, v: int) -> None:
        self.in_s[v] = False
        self.size -= 1
        nb = self.adj[v]
        self.work += len(nb) + 1
        for y in nb:
            self.tight[y] -= 1
        if self.logging:
            self.log.append((False, v))

    def insert_free_around(self, v: int, out: list[int]) -> None:
        """Insert every free non-solution neighbour of ``v``."""
        nb = self.adj[v]
        self.work += len(nb)
        in_s, tight = self.in_s, self.tight
        for y in nb:
            if not in_s[y] and tight[y] == 0:
                self.insert(y)
                out.append(y)

    def undo(self) -> None:
        self.logging = False
        while self.log:
            added, v = self.log.pop()
            if added:
                self.remove(v)
            else:
                self.insert(v)

    def commit(self) -> None:
        self.log.clear()


def _try_two_improvement(s: _IndependentSet, x: int, mark: list[int], stamp: list[int]) -> list[int]:
    """Try a (1,2)-swap at solution vertex ``x``; return the inserted vertices."""
    if not s.in_s[x]:
        return []
    adj, in_s, tight = s.adj, s.in_s, s.tight
    nb = adj[x]
    s.work += len(nb)
    one_tight = [v for v in nb if not in_s[v] and tight[v] == 1]
    if len(one_tight) < 2:
        return []
    for v in one_tight:
        stamp[0] += 1
        st = stamp[0]
        nv = adj[v]
        s.work += len(nv)
        for y in nv:
            mark[y] = st
        for w in one_tight:
            s.work += 1
            if w != v and mark[w] != st:
                s.remove(x)
                s.insert(v)
                s.insert(w)
                added = [v, w]
                s.insert_free_around(x, added)
                return added
    return []


def _local_search(s: _IndependentSet, queue: deque[int], mark: list[int], stamp: list[int], limit: int) -> None:
    while queue and s.work <= limit:
        x = queue.popleft()
        queue.extend(_try_two_improvement(s, x, mark, stamp))


def improve_vertex_cover(
    adj_map: Mapping[Any, Iterable[Any]],
    cover: Iterable[Any],
    budget: float = DEFAULT_BUDGET,
    seed: int = DEFAULT_SEED,
) -> set[Any]:
    """Return a vertex cover no larger than ``cover``, in O(budget * (n + m)).

    ``adj_map`` maps every vertex to its neighbours (self-loops ignored);
    ``cover`` must be a valid vertex cover of that graph.
    """
    nodes = list(adj_map)
    n = len(nodes)
    if n == 0:
        return set()
    index = {v: i for i, v in enumerate(nodes)}
    adj = [[index[u] for u in adj_map[v] if u != v] for v in nodes]
    m2 = sum(len(a) for a in adj)
    cover_set = set(cover)
    s = _IndependentSet(adj, (i for i, v in enumerate(nodes) if v not in cover_set))
    # Make S maximal (the input cover may be non-minimal).
    for i in range(n):
        if not s.in_s[i] and s.tight[i] == 0:
            s.insert(i)
    limit = s.work + int(budget * (n + m2 // 2)) + 1

    mark = [0] * n
    stamp = [0]
    # Phase 1: plain (1,2)-swap descent over every solution vertex.
    _local_search(s, deque(i for i in range(n) if s.in_s[i]), mark, stamp, limit)

    best_size = s.size
    rng = random.Random(seed)

    # Phase 2: iterated local search with undo-on-worsening.
    while s.work <= limit:
        u = rng.randrange(n)
        tries = 0
        while (s.in_s[u] or s.tight[u] > 2) and tries < 4:
            u = rng.randrange(n)
            tries += 1
        s.work += tries + 1
        if s.in_s[u]:
            continue
        s.logging = True
        evicted = [y for y in adj[u] if s.in_s[y]]
        s.work += len(adj[u])
        for y in evicted:
            s.remove(y)
        s.insert(u)
        touched = [u]
        for y in evicted:
            s.insert_free_around(y, touched)
        _local_search(s, deque(touched), mark, stamp, limit)
        if s.size >= best_size:
            if s.size > best_size:
                best_size = s.size
            s.logging = False
            s.commit()
        else:
            s.undo()
            s.commit()
    # Invariant: the live set is always the best seen (worse steps are undone).
    return {nodes[i] for i in range(n) if not s.in_s[i]}
