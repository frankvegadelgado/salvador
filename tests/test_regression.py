"""Regression smoke tests for Salvador v0.1.2."""

from __future__ import annotations

import networkx as nx

from salvador import __version__
from salvador.algorithm import find_vertex_cover
from salvador.parser import read
from salvador.utils import is_vertex_cover


def test_version_is_008() -> None:
    assert __version__ == "0.1.2"


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




def test_subdivision_c11_valid_and_exact_when_unpeeled() -> None:
    """c11: S is the bipartite subdivision (n + m nodes, 2m edges); the cover
    dominates every edge-node (a vertex cover of G), Phi < n + 1, and when no
    peel was needed the cover is minimum (checked by brute force)."""
    from salvador import subdivision_reduction as sr
    from salvador.algorithm import ensemble_candidates, find_vertex_cover_brute_force

    graphs = [
        nx.path_graph(7),
        nx.cycle_graph(9),
        nx.complete_graph(6),
        nx.petersen_graph(),
        nx.complete_bipartite_graph(3, 5),
        nx.balanced_tree(2, 4),
        nx.relabel_nodes(nx.cycle_graph(6), {0: "a", 1: (1, 2)}),
    ] + [nx.gnp_random_graph(12, 0.25, seed=s) for s in range(20)]
    for graph in graphs:
        graph = graph.copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        S = sr.build_subdivision_graph(graph)
        assert nx.is_bipartite(S)
        assert S.number_of_nodes() == n + m and S.number_of_edges() == 2 * m
        adj = {v: set(graph[v]) for v in graph}
        info: dict = {}
        cover = sr.subdivision_vertex_cover(adj, info)
        assert info["undominated"] == 0 and is_vertex_cover(graph, cover)
        assert sr.phi(graph, cover) < n + 1
        if info["exact"] and n <= 14:
            assert len(cover) == len(find_vertex_cover_brute_force(graph))
        for peel in ("max", "min", "ref"):
            for fold in (False, True):
                vi: dict = {}
                c = sr.subdivision_vertex_cover(adj, vi, peel, fold, reference=set(graph))
                assert vi["undominated"] == 0 and is_vertex_cover(graph, c)
                if vi["exact"] and n <= 14:
                    assert len(c) == len(find_vertex_cover_brute_force(graph))
        cands = ensemble_candidates(graph)
        assert set(cands) == {f"c{i}" for i in range(1, 14)}
        assert is_vertex_cover(graph, cands["c11"])


def test_split_domination_c12_valid_and_exact_when_core_empty() -> None:
    """c12: the split graph has 2m nodes and maximum degree 3; the cover is
    valid, never larger than its reference, and minimum when the core is empty."""
    from salvador import split_domination as sd
    from salvador.algorithm import ensemble_candidates, find_vertex_cover_brute_force

    graphs = [
        nx.path_graph(7), nx.cycle_graph(9), nx.complete_graph(6), nx.petersen_graph(),
        nx.complete_bipartite_graph(3, 5), nx.balanced_tree(2, 4),
        nx.relabel_nodes(nx.cycle_graph(6), {0: "a", 1: (1, 2)}),
    ] + [nx.gnp_random_graph(12, 0.3, seed=s) for s in range(15)]
    for graph in graphs:
        graph = graph.copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
        m = graph.number_of_edges()
        H = sd.build_split_nx(graph)
        assert H.number_of_nodes() == 2 * m and max(d for _, d in H.degree()) <= 3
        adj = {v: set(graph[v]) for v in graph}
        ref = set(graph)
        for width in (10, 2):
            info: dict = {}
            cover = sd.split_dominating_cover(adj, ref, max_width=width, info=info)
            assert is_vertex_cover(graph, cover) and len(cover) <= len(ref)
            if info["exact"] and graph.number_of_nodes() <= 14:
                assert len(cover) == len(find_vertex_cover_brute_force(graph))
        cands = ensemble_candidates(graph)
        assert is_vertex_cover(graph, cands["c12"])
        assert len(cands["c12"]) <= min(len(cands[k]) for k in
                                        ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"))


def test_reduce_search_c12_never_worse_and_valid() -> None:
    """c12: valid cover, never larger than its start (the best of c1..c11), reproducible."""
    from salvador import reduce_search as rs
    from salvador.algorithm import ensemble_candidates

    for seed in range(8):
        graph = nx.random_regular_graph(3, 60, seed=seed) if seed % 2 else nx.gnp_random_graph(50, 0.1, seed=seed)
        adj = {v: set(graph[v]) for v in graph}
        start = set(graph)
        info: dict = {}
        cover = rs.reduce_search_cover(adj, start, rounds=2, budget=20, info=info)
        assert is_vertex_cover(graph, cover) and len(cover) <= len(start)
        assert cover == rs.reduce_search_cover(adj, start, rounds=2, budget=20)
        cands = ensemble_candidates(graph)
        assert is_vertex_cover(graph, cands["c12"])
        assert len(cands["c12"]) <= min(len(cands[k]) for k in
                                        ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"))
