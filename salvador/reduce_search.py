"""Reduce-and-search improver for vertex cover (candidate c15).

c15 starts from the best of c1..c14 and alternates, for ``rounds`` rounds:

1. **Swap search** -- the budgeted (1,2)-swap iterated local search of
   :mod:`salvador.local_search` (Andrade, Resende and Werneck), which accepts
   equal-size moves and so walks plateaus of the independent set
   ``V \\ C``; its budget is ``budget * (n + m)`` per round.
2. **Reduction pass** -- c11's reducing-peeling solver
   (:mod:`salvador.subdivision_reduction`, exact degree-0, degree-1,
   triangle and degree-2 fold reductions) with the guided peel along the
   cover produced by step 1. The exact reductions re-decide every vertex
   they reach, so the pass can leave a plateau the swap search is stuck on;
   its pruned result replaces the current cover only when it is not larger.

Each round uses a different pseudo-random seed (fixed, so runs are
reproducible). The current cover never grows, so the result is a valid cover
no larger than the starting one. Work per round is ``O(budget * (n + m))``
for the search plus ``O(n + m)`` for the pass (folds within their own
budget), hence worst-case ``O(n + m)`` overall for fixed ``rounds`` and
``budget``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import local_search, subdivision_reduction

DEFAULT_ROUNDS = 3
DEFAULT_BUDGET = local_search.DEFAULT_BUDGET


def _prune(adj: Mapping[Any, set[Any]], cover: set[Any]) -> set[Any]:
    """Drop cover vertices whose neighbours are all in the cover (one pass)."""
    out = set(cover)
    for v in list(out):
        if all(w in out for w in adj[v]):
            out.discard(v)
    return out


def reduce_search_cover(
    adj: Mapping[Any, Iterable[Any]],
    start: Iterable[Any],
    rounds: int = DEFAULT_ROUNDS,
    budget: float = DEFAULT_BUDGET,
    seed: int = local_search.DEFAULT_SEED,
    info: dict | None = None,
) -> set[Any]:
    """Improve the vertex cover ``start`` (must be valid); never returns a larger one.

    ``info`` (optional) receives ``start`` (size of the input), ``search_gain``
    and ``reduce_gain`` (vertices removed by each phase over all rounds),
    ``rounds_improved`` and ``rounds``.
    """
    A = {v: set(adj[v]) - {v} for v in adj}
    current = set(start)
    start_size = len(current)
    search_gain = reduce_gain = 0
    rounds_improved = 0
    for r in range(rounds):
        before = len(current)
        searched = local_search.improve_vertex_cover(A, current, budget=budget, seed=seed + 7919 * r)
        if len(searched) < len(current):
            search_gain += len(current) - len(searched)
            current = searched
        elif len(searched) == len(current):
            current = searched  # an equal cover from the plateau walk: a new starting point
        reduced = _prune(A, subdivision_reduction.subdivision_vertex_cover(
            A, None, "ref", True, reference=current))
        if len(reduced) <= len(current) and all(u in reduced or w in reduced for u in A for w in A[u]):
            reduce_gain += len(current) - len(reduced)
            current = reduced
        if len(current) < before:
            rounds_improved += 1
    if info is not None:
        info.update({"start": start_size, "search_gain": search_gain, "reduce_gain": reduce_gain,
                     "rounds_improved": rounds_improved, "rounds": rounds})
    return current
