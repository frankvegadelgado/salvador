"""Public vertex-cover solvers exposed by Salvador."""

from __future__ import annotations

import itertools
from typing import Any

import networkx as nx

from . import utils, vc_reduction


def prune_redundant_vertices(adj: dict[Any, set[Any]], cover: set[Any]) -> set[Any]:
    """Remove redundant vertices from a candidate cover in one pass.

    A vertex ``v`` can be removed when every neighbour of ``v`` remains in the
    current cover. In that case each edge incident to ``v`` is still covered by
    the neighbour endpoint after removal.
    """
    current_cover = set(cover)

    for v in list(current_cover):
        if all(u in current_cover for u in adj.get(v, ())):
            current_cover.remove(v)

    return current_cover


def find_vertex_cover(graph: nx.Graph, epsilon: float = 0.1) -> set[Any]:
    """Return Salvador's approximate vertex cover for ``graph``.

    The algorithmic logic is the same as the previous implementation: remove
    self-loops and isolated vertices, solve the spanning-forest-core reduction,
    repair uncovered original edges, and finish with a single redundancy-pruning
    pass.
    """
    working_graph = graph.copy()
    working_graph.remove_edges_from(nx.selfloop_edges(working_graph))
    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))

    if working_graph.number_of_edges() == 0:
        return set()

    cover, _ = vc_reduction.solve_vc(working_graph, epsilon)
    adj = {v: set(working_graph[v]) for v in working_graph}
    return prune_redundant_vertices(adj, set(cover))


def find_vertex_cover_brute_force(graph: nx.Graph) -> set[Any] | None:
    """Compute an exact minimum vertex cover by exhaustive search."""
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))
    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))

    if working_graph.number_of_nodes() == 0:
        return set()

    nodes = list(working_graph.nodes())
    for k in range(1, len(nodes) + 1):
        for candidate in itertools.combinations(nodes, k):
            cover_candidate = set(candidate)
            if utils.is_vertex_cover(working_graph, cover_candidate):
                return cover_candidate

    return None


def find_vertex_cover_approximation(graph: nx.Graph) -> set[Any] | None:
    """Return NetworkX's standard 2-approximation vertex cover baseline."""
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None
    return set(nx.approximation.vertex_cover.min_weighted_vertex_cover(graph))
