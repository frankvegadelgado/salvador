"""Regression smoke tests for Salvador v0.0.8."""

from __future__ import annotations

import networkx as nx

from salvador import __version__
from salvador.algorithm import find_vertex_cover
from salvador.parser import read
from salvador.utils import is_vertex_cover


def test_version_is_008() -> None:
    assert __version__ == "0.0.8"


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
    """The default call stays within 7/4 of the optimum on the historical car/ witness.

    This bipartite graph was the largest-ratio instance found by car/ under the
    former default epsilon=0.1 (a single-pipeline call returned a cover of size 7
    against the exact optimum 4, ratio 7/4). Under the current default epsilon=1,
    find_vertex_cover instead runs the full linear-time ensemble (see
    salvador.algorithm), which is never worse than any single candidate; this
    test keeps the same conservative 7/4 upper bound as a regression backstop
    while the cover must still be valid.
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


def test_bipartite_planar_reduction_is_exact_and_valid() -> None:
    """The auxiliary graph is bipartite planar, the linear-time cycle DP
    matches brute force on every gadget, and every variant is a valid cover."""
    import itertools

    from salvador import bipartite_reduction as br

    graphs = [
        nx.petersen_graph(),
        nx.cycle_graph(9),
        nx.complete_graph(6),
        nx.gnp_random_graph(18, 0.3, seed=3),
        nx.relabel_nodes(nx.cycle_graph(6), {0: "a", 1: (1, 2)}),
    ]
    for graph in graphs:
        aux = br.build_auxiliary_graph(graph)
        assert nx.is_bipartite(aux) and nx.is_planar(aux)
        assert all(aux.degree(x) in (1, 2) for x in aux)
        adj = {v: set(graph[v]) for v in graph}
        for cover in br.bipartite_planar_covers(adj):
            assert is_vertex_cover(graph, cover)
    for length in (2, 4, 6, 8, 10):
        weights = [((7 * i) % 5 + 1) / 3 for i in range(length)]
        sol = br.min_weight_vc_cycle(weights, [i % 2 for i in range(length)])
        edges = [(i, i + 1) for i in range(length - 1)]
        if length > 2:
            edges.append((length - 1, 0))
        best = min(
            sum(w for w, b in zip(weights, bits) if b)
            for bits in itertools.product((0, 1), repeat=length)
            if all(bits[a] or bits[b] for a, b in edges)
        )
        assert abs(sum(w for w, b in zip(weights, sol) if b) - best) < 1e-9


def test_local_search_candidate_never_worse_and_valid() -> None:
    """c9 is a valid cover no larger than the best of c1..c8."""
    from salvador.algorithm import ensemble_candidates

    for seed in range(20):
        graph = nx.gnp_random_graph(40, 0.12, seed=seed)
        cands = ensemble_candidates(graph)
        for cover in cands.values():
            assert is_vertex_cover(graph, cover)
        best_other = min(len(c) for k, c in cands.items() if k != "c9")
        assert len(cands["c9"]) <= best_other

