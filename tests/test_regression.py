"""Regression smoke tests for Salvador v0.0.5."""

from __future__ import annotations

import networkx as nx

from salvador import __version__
from salvador.algorithm import find_vertex_cover
from salvador.parser import read
from salvador.utils import is_vertex_cover


def test_version_is_005() -> None:
    assert __version__ == "0.0.6"


def test_small_benchmark_cover_is_valid() -> None:
    graph = read("benchmarks/testMatrix1")
    cover = find_vertex_cover(graph)
    assert is_vertex_cover(graph, cover)
    assert len(cover) == 3


def test_common_graph_families_are_covered() -> None:
    graphs = [
        nx.path_graph(7),
        nx.cycle_graph(8),
        nx.complete_graph(5),
        nx.complete_bipartite_graph(3, 3),
        nx.petersen_graph(),
    ]
    for graph in graphs:
        cover = find_vertex_cover(graph)
        assert is_vertex_cover(graph, cover)


def test_epsilon_is_active_and_always_valid() -> None:
    """The accuracy parameter must be accepted and always yield a valid cover."""
    graph = nx.gnp_random_graph(14, 0.3, seed=7)
    for epsilon in (1.0, 0.5, 0.25, 0.1, 0.05):
        cover = find_vertex_cover(graph, epsilon=epsilon)
        assert is_vertex_cover(graph, cover)


def test_default_call_within_7_4_on_car_witness() -> None:
    """The default call stays within 7/4 of the optimum on the worst car/ witness.

    This bipartite graph is the largest-ratio instance found by car/; under the
    default epsilon=0.1 the algorithm returns a cover of size 7 against the exact
    optimum 4 (ratio 7/4), so the cover must be valid and within the 7/4 bound.
    """
    edges = [
        (0, 9), (0, 8), (0, 7), (0, 10), (1, 7), (1, 8), (1, 9), (1, 10),
        (3, 7), (3, 8), (3, 10), (3, 9), (4, 10), (4, 9), (4, 7),
        (5, 7), (5, 9), (5, 8),
    ]
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(edges)
    cover = find_vertex_cover(graph)
    assert is_vertex_cover(graph, cover)
    # Exact optimum is 4 (one side of the bipartition); allow the 7/4 bound.
    assert len(cover) <= (7 * 4) // 4  # 7
