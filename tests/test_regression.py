"""Regression smoke tests for Salvador v0.0.4."""

from __future__ import annotations

import networkx as nx

from salvador import __version__
from salvador.algorithm import find_vertex_cover
from salvador.parser import read
from salvador.utils import is_vertex_cover


def test_version_is_004() -> None:
    assert __version__ == "0.0.4"


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
