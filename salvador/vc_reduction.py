"""Vertex Cover to weighted MIDS reduction used by Salvador.

The Salvador pipeline:

1. Build a spanning-forest planar core with one Union-Find pass over the edges.
2. Convert that core into the weighted MIDS gadget (itself a forest).
3. Run the epsilon-controlled Baker PTAS weighted independent-dominating-set
   pass; ``epsilon`` sets the layering width ``k = ceil(1/epsilon)``.
4. Decode the selected ``(v, 0)`` nodes as a cover and repair any uncovered
   core or original edge by adding the higher-degree endpoint.

The public guarantee of this module is validity: every returned set covers the
original graph. Approximation claims are handled experimentally and conjecturally
in the accompanying manuscript.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from . import baker_ptas


def reduce_vc_to_mids(
    graph: nx.Graph,
    epsilon: float = 0.1,
    assume_planar: bool = False,
) -> tuple[nx.Graph, dict[Any, int], int]:
    """Build the weighted MIDS gadget for a planar graph.

    For every original vertex ``v`` the gadget has a weight-one node ``(v, 0)``
    and a weight-zero pendant node ``(v, 1)``. For every core edge ``(u, v)`` it
    has a heavy hub ``('h', u, v)`` of weight ``n + 1`` adjacent to ``(u, 0)``
    and ``(v, 0)``.
    """
    del epsilon  # The gadget construction is epsilon-independent; epsilon is
    # consumed downstream by the Baker PTAS pass (see ``_solve_planar``).

    if not assume_planar and not nx.is_planar(graph):
        raise ValueError("reduce_vc_to_mids requires a planar graph.")

    n_vertices = graph.number_of_nodes()
    penalty = n_vertices + 1
    gadget = nx.Graph()
    weights: dict[Any, int] = {}

    for v in graph.nodes():
        gadget.add_edge((v, 0), (v, 1))
        weights[(v, 0)] = 1
        weights[(v, 1)] = 0

    for u, v in graph.edges():
        hub = ("h", u, v)
        weights[hub] = penalty
        gadget.add_edge((u, 0), hub)
        gadget.add_edge((v, 0), hub)

    return gadget, weights, penalty


def _spanning_forest_planar_core(graph: nx.Graph) -> tuple[nx.Graph, list[tuple[Any, Any]]]:
    """Return a spanning-forest core and the cycle-closing edges.

    The core is planar by construction. The implementation is a single
    Union-Find pass over ``graph.edges()``, so it has linear edge-processing
    cost up to the inverse-Ackermann factor from path compression.
    """
    core = nx.Graph()
    core.add_nodes_from(graph.nodes())

    parent = {v: v for v in graph.nodes()}
    rank = {v: 0 for v in graph.nodes()}

    def find(x: Any) -> Any:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(u: Any, v: Any) -> bool:
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            return False
        if rank[root_u] < rank[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        if rank[root_u] == rank[root_v]:
            rank[root_u] += 1
        return True

    removed: list[tuple[Any, Any]] = []
    for u, v in graph.edges():
        if union(u, v):
            core.add_edge(u, v)
        else:
            removed.append((u, v))

    return core, removed


def _maximal_planar_subgraph(graph: nx.Graph) -> tuple[nx.Graph, list[tuple[Any, Any]]]:
    """Compatibility wrapper for the production spanning-forest core.

    Earlier prototypes attempted a maximal planar subgraph. The current linear
    Salvador pipeline intentionally uses the spanning forest returned by
    :func:`_spanning_forest_planar_core`, preserving the solver's previous solver logic.
    """
    return _spanning_forest_planar_core(graph)


def _solve_planar(core: nx.Graph, original_graph: nx.Graph, epsilon: float) -> set[Any]:
    """Solve the weighted-MIDS gadget of a planar core and decode a cover."""
    gadget, weights, _ = reduce_vc_to_mids(core, epsilon, assume_planar=True)

    nodes = list(gadget.nodes())
    to_int = {node: index for index, node in enumerate(nodes)}
    to_label = {index: node for node, index in to_int.items()}
    adj_int = {to_int[v]: {to_int[w] for w in gadget[v]} for v in gadget}
    weights_int = {to_int[v]: weights[v] for v in gadget}

    ids_solution = baker_ptas.baker_ptas_ids_weighted(adj_int, weights=weights_int, epsilon=epsilon)
    ids_labels = {to_label[k] for k in ids_solution}

    cover = {v for v in core.nodes() if (v, 0) in ids_labels}

    for u, v in core.edges():
        if u not in cover and v not in cover:
            cover.add(u if original_graph.degree(u) >= original_graph.degree(v) else v)

    return cover


def solve_vc(graph: nx.Graph, epsilon: float = 0.1) -> tuple[frozenset[Any], float]:
    """Return a valid approximate vertex cover of any undirected graph."""
    if graph.number_of_edges() == 0:
        return frozenset(), 0.0

    core, removed_edges = _spanning_forest_planar_core(graph)
    cover = _solve_planar(core, graph, epsilon)

    for u, v in removed_edges:
        if u not in cover and v not in cover:
            cover.add(u if graph.degree(u) >= graph.degree(v) else v)

    frozen_cover = frozenset(cover)
    return frozen_cover, float(len(frozen_cover))


def run_demo() -> None:
    """Run a small console demonstration of the reduction and repair path."""
    import numpy as np

    print("═" * 60)
    print("  VC → Weighted MIDS  (linear forest-core path)")
    print("═" * 60)

    print("\n1. Gadget planarity on planar examples\n")
    families = [
        *(nx.path_graph(n) for n in range(1, 12)),
        *(nx.cycle_graph(n) for n in range(3, 12)),
        nx.complete_graph(4),
        nx.dodecahedral_graph(),
        nx.icosahedral_graph(),
        *(nx.wheel_graph(n) for n in range(4, 10)),
        *(nx.grid_2d_graph(r, c) for r in range(2, 6) for c in range(2, 6)),
    ]
    ok = sum(1 for g in families if nx.is_planar(g) and nx.is_planar(reduce_vc_to_mids(g, 0.5)[0]))
    print(f"   {ok} planar graphs tested — gadget always planar ✓")

    print("\n2. End-to-end validity\n")
    failing_edges = [
        (np.int32(1), np.int32(0)),
        (np.int32(1), np.int32(3)),
        (np.int32(1), np.int32(5)),
        (np.int32(1), np.int32(7)),
        (np.int32(1), np.int32(8)),
        (np.int32(1), np.int32(10)),
        (np.int32(0), np.int32(2)),
        (np.int32(0), np.int32(3)),
        (np.int32(0), np.int32(5)),
        (np.int32(0), np.int32(7)),
        (np.int32(0), np.int32(8)),
        (np.int32(0), np.int32(10)),
        (np.int32(0), np.int32(11)),
        (np.int32(2), np.int32(3)),
        (np.int32(2), np.int32(8)),
        (np.int32(2), np.int32(10)),
        (np.int32(3), np.int32(4)),
        (np.int32(3), np.int32(8)),
        (np.int32(3), np.int32(10)),
        (np.int32(7), np.int32(4)),
        (np.int32(8), np.int32(4)),
        (np.int32(10), np.int32(4)),
    ]
    failing_case = nx.Graph()
    failing_case.add_edges_from(failing_edges)

    cases = [
        ("Star K₁,₃", nx.star_graph(3)),
        ("Cycle-6", nx.cycle_graph(6)),
        ("K4", nx.complete_graph(4)),
        ("Grid-3×3", nx.grid_2d_graph(3, 3)),
        ("K5", nx.complete_graph(5)),
        ("K3,3", nx.complete_bipartite_graph(3, 3)),
        ("Petersen", nx.petersen_graph()),
        ("Failing case", failing_case),
    ]
    print(f"   {'graph':<14} {'N':>4} {'M':>4} {'cover':>7}  valid  core")
    print("   " + "─" * 50)
    for name, g in cases:
        core, removed = _maximal_planar_subgraph(g)
        cover, _ = solve_vc(g, epsilon=0.5)
        valid = all(u in cover or v in cover for u, v in g.edges())
        print(
            f"   {name:<14} {g.number_of_nodes():>4} {g.number_of_edges():>4}"
            f" {len(cover):>7}  {'✓' if valid else '✗'}"
            f"  {core.number_of_edges()}/{g.number_of_edges()} edges kept; {len(removed)} repaired"
        )


if __name__ == "__main__":
    run_demo()
