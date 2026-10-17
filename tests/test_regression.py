"""Regression smoke tests for Salvador v0.1.4."""

from __future__ import annotations

import networkx as nx

from salvador import __version__
from salvador.algorithm import find_vertex_cover
from salvador.parser import read
from salvador.utils import is_vertex_cover


def test_version_is_014() -> None:
    assert __version__ == "0.1.4"


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
    """c14 is a valid cover no larger than the best of c1..c13."""
    from salvador.algorithm import ensemble_candidates

    for seed in range(20):
        graph = nx.gnp_random_graph(40, 0.12, seed=seed)
        cands = ensemble_candidates(graph)
        for cover in cands.values():
            assert is_vertex_cover(graph, cover)
        assert list(cands) == [f"c{i}" for i in range(1, 17)]
        assert len(cands["c14"]) <= min(len(cands[f"c{i}"]) for i in range(1, 14))




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
        assert set(cands) == {f"c{i}" for i in range(1, 17)}
        assert is_vertex_cover(graph, cands["c11"])


def test_reduce_search_c15_never_worse_and_valid() -> None:
    """c15: valid cover, never larger than its start (the best of c1..c14), reproducible."""
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
        assert is_vertex_cover(graph, cands["c15"])
        assert len(cands["c15"]) <= min(len(cands[f"c{i}"]) for i in range(1, 15))


def test_a_priori_bound_two_tau_minus_components() -> None:
    """Theorem (2 - r/tau): a pruned maximal-matching cover, hence Salvador's final cover, has
    at most 2 tau - r vertices, r = number of connected components with an edge."""
    import random

    from salvador.algorithm import find_vertex_cover_brute_force, final_candidates, prune_redundant_vertices

    graphs = [nx.path_graph(3), nx.disjoint_union(nx.path_graph(3), nx.path_graph(3)), nx.complete_graph(3),
              nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)), nx.cycle_graph(5),
              nx.complete_bipartite_graph(4, 4), nx.petersen_graph(), nx.star_graph(6)]
    crown = nx.complete_bipartite_graph(5, 5)
    crown.remove_edges_from([(i, 5 + i) for i in range(5)])
    graphs.append(crown)
    graphs += [nx.gnp_random_graph(11, p, seed=s) for s in range(10) for p in (0.2, 0.4)]
    rng = random.Random(7)
    for graph in graphs:
        graph = graph.copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if graph.number_of_edges() == 0:
            continue
        tau = len(find_vertex_cover_brute_force(graph))
        r = nx.number_connected_components(graph)
        adj = {v: set(graph[v]) for v in graph}
        for _ in range(10):  # every maximal matching, in random edge orders
            edges = list(graph.edges())
            rng.shuffle(edges)
            matched: set = set()
            for u, v in edges:
                if u not in matched and v not in matched:
                    matched |= {u, v}
            c1 = prune_redundant_vertices(adj, matched)
            assert is_vertex_cover(graph, c1) and len(c1) <= 2 * tau - r
        assert len(final_candidates(graph)["c17"]) <= 2 * tau - r


def test_consensus_c16_valid_and_never_breaks_the_ensemble() -> None:
    """c16: valid cover; the forced vertices are deleted and added to the cover directly (no self-loops,
    which _clean would drop); the lower bound used to stop is a true lower bound; reproducible."""
    from salvador import consensus_force as cf
    from salvador.algorithm import ensemble_candidates, final_candidates, find_vertex_cover_brute_force

    graphs = [nx.petersen_graph(), nx.cycle_graph(9), nx.barbell_graph(4, 2), nx.star_graph(5)]
    graphs += [nx.gnp_random_graph(12, p, seed=s) for s in range(10) for p in (0.2, 0.4)]
    for graph in graphs:
        graph = graph.copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if graph.number_of_edges() == 0:
            continue
        adj = {v: set(graph[v]) for v in graph}
        tau = len(find_vertex_cover_brute_force(graph))
        assert cf.lower_bound(adj) <= tau
        info: dict = {}
        cands = ensemble_candidates(graph, info=info)
        assert is_vertex_cover(graph, cands["c16"])
        assert cands["c16"] == ensemble_candidates(graph)["c16"]
        assert len(final_candidates(graph)["c17"]) <= min(len(cands[f"c{i}"]) for i in range(1, 17))


def test_component_split_c1_valid_and_linear_split() -> None:
    """c1: copies (u, 0) and (u, 1) only; only edges of M (both endpoints at index 0) are split; the split
    graph has r' components with r <= r' <= |M| and r' >= |M| - 1, equal to the union-find count; its edges
    are those of G relabelled by copy index; every decoding completes to a valid cover; reproducible."""
    from collections import Counter

    from salvador import component_split as cs
    from salvador.algorithm import ensemble_candidates

    graphs = [nx.path_graph(9), nx.cycle_graph(10), nx.star_graph(7), nx.complete_graph(6), nx.petersen_graph(),
              nx.barbell_graph(5, 3), nx.balanced_tree(3, 3), nx.complete_bipartite_graph(4, 6),
              nx.relabel_nodes(nx.cycle_graph(6), {0: "a", 1: (1, 2)})]
    graphs += [nx.gnp_random_graph(30, p, seed=s) for s in range(6) for p in (0.08, 0.2)]
    for graph in graphs:
        graph = graph.copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if graph.number_of_edges() == 0:
            continue
        info: dict = {}
        H = cs.build_split_graph(graph, info)
        assert {i for _, i in H} <= {0, 1}
        with_edge = sum(1 for c in nx.connected_components(H) if len(c) > 1)
        assert info["r_prime"] == with_edge
        assert info["target"] == len(nx.approximation.min_maximal_matching(graph))
        assert info["r"] <= info["r_prime"] <= info["target"] and info["r_prime"] >= info["target"] - 1
        assert Counter(frozenset((a, b)) for (a, _), (b, _) in H.edges()) == \
            Counter(frozenset(e) for e in graph.edges())
        for rule in cs.RULES:
            raw = cs.decode(graph, H, rule)
            assert raw <= set(graph)
            assert is_vertex_cover(graph, cs.complete_cover(graph, raw))
        cands = ensemble_candidates(graph)
        assert is_vertex_cover(graph, cands["c1"])
        assert cands["c1"] == ensemble_candidates(graph)["c1"]


def test_bipartite_layers_c9_valid_exact_on_bipartite_and_refined_by_c17() -> None:
    """c9 is a valid cover; on bipartite graphs it is one layer and minimum when its Hopcroft-Karp phases
    reach a maximum matching (always, with enough phases); the number of layers is at most
    floor(log2 m) + 1; c17 is never larger than c9."""
    import math
    import random

    from salvador import bipartite_peel as bp
    from salvador.algorithm import c9_cover, final_candidates

    rng = random.Random(9)
    graphs = [nx.complete_bipartite_graph(4, 7), nx.grid_2d_graph(5, 6), nx.cycle_graph(10), nx.path_graph(9),
              nx.cycle_graph(7), nx.petersen_graph(), nx.complete_graph(6), nx.wheel_graph(8),
              nx.disjoint_union(nx.cycle_graph(5), nx.complete_bipartite_graph(3, 5))]
    graphs += [nx.gnp_random_graph(rng.randint(5, 12), rng.choice([0.25, 0.5, 0.75]), seed=t) for t in range(40)]
    graphs += [nx.bipartite.random_graph(rng.randint(2, 8), rng.randint(2, 8), 0.4, seed=t) for t in range(40)]
    for graph in graphs:
        G = nx.convert_node_labels_to_integers(graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_edges() == 0:
            continue
        adj = {v: set(G[v]) for v in G}
        info: dict = {}
        c9 = c9_cover(adj, info=info)
        assert is_vertex_cover(G, c9)
        assert info["max_layers"] <= math.floor(math.log2(G.number_of_edges())) + 1
        if nx.is_bipartite(G):
            top = {v for c in nx.connected_components(G) for v, s in nx.bipartite.color(G.subgraph(c)).items() if s == 0}
            tau = len(nx.bipartite.hopcroft_karp_matching(G, top_nodes=top)) // 2
            assert info["max_layers"] == 1
            full = bp.bipartite_layers_cover(adj, phases=G.number_of_nodes())
            assert len(full) == tau
            if info["exact"]:
                assert len(c9) == tau
        cands = final_candidates(G)
        assert len(cands["c17"]) <= len(cands["c9"])
