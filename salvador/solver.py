"""
Minimum Weight Vertex Cover for Bipartite (+ optionally Planar) Graphs
=======================================================================
Theoretical basis
-----------------
The vertex-cover LP on a bipartite graph has a totally unimodular (TU)
constraint matrix, so its LP relaxation is always integral.  By LP duality
this equals the maximum weight matching, and by the max-flow min-cut theorem
the optimum can be read off a single min-cut computation on a node-split
flow network.

Node-split construction
-----------------------
For a bipartite graph G = (L ∪ R, E) with node weights w:
  • Add a super-source  s  and super-sink  t.
  • Split every node v into v_in and v_out with arc capacity w[v].
  • s  → l_in  (cap w[l])  for every l ∈ L
  • r_out → t   (cap w[r])  for every r ∈ R
  • l_out → r_in (cap ∞)   for every edge (l,r) ∈ E  (direction L→R)

A minimum s-t cut of this network has value = min weight vertex cover.
Vertices whose *split arc* is cut belong to the cover:
  • l ∈ L is in the cover  iff  l_in is on the s-side but l_out is on the t-side
  • r ∈ R is in the cover  iff  r_in is on the s-side but r_out is on the t-side

Complexity
----------
O(√n · m)  using Hopcroft-Karp-equivalent push-relabel (NetworkX uses
           Boykov-Kolmogorov / Dinic internally for the flow).
For planar bipartite graphs m = O(n), giving O(n^{3/2}) overall.
"""

import networkx as nx
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node_names(v: Any):
    """Return the two names used for the split-node of v."""
    return (("in", v), ("out", v))


_INF = float("inf")


# ---------------------------------------------------------------------------
# main algorithm
# ---------------------------------------------------------------------------

def min_weight_vertex_cover_bipartite(
    G: nx.Graph,
    weight: str = "weight",
    default_weight: float = 1.0,
) -> tuple[set, float]:
    """
    Compute an exact minimum weight vertex cover of a bipartite graph G.

    Parameters
    ----------
    G : nx.Graph
        An undirected bipartite graph.  Every node should carry the attribute
        named by *weight*; nodes without it get *default_weight*.
    weight : str
        Node-attribute key for the vertex weight (default ``"weight"``).
    default_weight : float
        Weight assigned to nodes that lack the attribute (default ``1.0``).

    Returns
    -------
    cover : set
        The nodes forming the minimum weight vertex cover.
    total_weight : float
        Sum of weights of nodes in the cover.

    Raises
    ------
    nx.NetworkXError
        If G is not bipartite.
    ValueError
        If any node weight is negative or zero.

    Notes
    -----
    The planarity of G is not required for correctness (TU holds for all
    bipartite graphs) but can be verified cheaply with
    ``nx.check_planarity(G)`` before calling this function.
    """
    # ------------------------------------------------------------------ #
    # 1. Verify bipartiteness and obtain the 2-colouring (L, R partition) #
    # ------------------------------------------------------------------ #
    # nx.bipartite.sets raises AmbiguousSolution on disconnected graphs.
    # nx.bipartite.color colours each component independently and works for
    # any (dis)connected bipartite graph; it raises NetworkXError if the
    # graph contains an odd cycle (i.e. is not bipartite).
    try:
        coloring = nx.bipartite.color(G)
    except nx.NetworkXError:
        raise nx.NetworkXError("Graph is not bipartite.")
    L = {v for v, c in coloring.items() if c == 0}
    R = {v for v, c in coloring.items() if c == 1}

    # ------------------------------------------------------------------ #
    # 2. Validate weights                                                  #
    # ------------------------------------------------------------------ #
    def w(v):
        val = G.nodes[v].get(weight, default_weight)
        if val <= 0:
            raise ValueError(
                f"Node {v!r} has non-positive weight {val}; "
                "weights must be strictly positive."
            )
        return val

    # ------------------------------------------------------------------ #
    # 3. Build the node-split flow network                                 #
    # ------------------------------------------------------------------ #
    F = nx.DiGraph()

    S, T = "__source__", "__sink__"
    F.add_nodes_from([S, T])

    for v in G.nodes:
        v_in, v_out = _node_names(v)
        F.add_edge(v_in, v_out, capacity=w(v))

    for l in L:
        l_in, l_out = _node_names(l)
        F.add_edge(S, l_in, capacity=_INF)   # source feeds every L node
        for r in G.neighbors(l):
            r_in, _ = _node_names(r)
            F.add_edge(l_out, r_in, capacity=_INF)

    for r in R:
        _, r_out = _node_names(r)
        F.add_edge(r_out, T, capacity=_INF)   # every R node drains to sink

    # ------------------------------------------------------------------ #
    # 4. Compute minimum cut (= max flow by max-flow min-cut theorem)      #
    # ------------------------------------------------------------------ #
    cut_value, partition = nx.minimum_cut(F, S, T)
    reachable, _ = partition          # reachable = s-side of the cut

    # ------------------------------------------------------------------ #
    # 5. Recover the vertex cover from the cut partition                   #
    # ------------------------------------------------------------------ #
    # A node v is in the cover iff its split arc (v_in → v_out) is cut,
    # i.e. v_in is on the s-side AND v_out is on the t-side.
    cover = set()
    for v in G.nodes:
        v_in, v_out = _node_names(v)
        if v_in in reachable and v_out not in reachable:
            cover.add(v)

    total = sum(w(v) for v in cover)
    return cover, total


# ---------------------------------------------------------------------------
# convenience wrapper that also checks planarity
# ---------------------------------------------------------------------------

def min_weight_vertex_cover_bipartite_planar(
    G: nx.Graph,
    weight: str = "weight",
    default_weight: float = 1.0,
    *,
    verify_planarity: bool = True,
) -> tuple[set, float]:
    """
    Like :func:`min_weight_vertex_cover_bipartite` but also verifies planarity.

    Parameters
    ----------
    verify_planarity : bool
        If ``True`` (default) raise ``ValueError`` when G is not planar.
        Set to ``False`` to skip the O(n) planarity check when you have
        already verified planarity externally.
    """
    if verify_planarity:
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            raise ValueError("Graph is not planar.")
    return min_weight_vertex_cover_bipartite(G, weight=weight,
                                             default_weight=default_weight)


# ---------------------------------------------------------------------------
# correctness verification helper
# ---------------------------------------------------------------------------

def verify_vertex_cover(G: nx.Graph, cover: set) -> bool:
    """Return True iff *cover* is a valid vertex cover of G."""
    return all(u in cover or v in cover for u, v in G.edges)


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------

def _run_tests():
    import itertools

    def brute_force_mwvc(G, weight="weight", default=1.0):
        """Exhaustive O(2^n) reference implementation for small graphs."""
        nodes = list(G.nodes)
        best_w, best_c = _INF, None
        for r in range(len(nodes) + 1):
            for subset in itertools.combinations(nodes, r):
                s = set(subset)
                if verify_vertex_cover(G, s):
                    tw = sum(G.nodes[v].get(weight, default) for v in s)
                    if tw < best_w:
                        best_w, best_c = tw, s
        return best_c, best_w

    print("Running tests …\n")

    # ------------------------------------------------------------------
    # Test 1: simple path P4  (bipartite + planar)
    # ------------------------------------------------------------------
    G1 = nx.path_graph(4)
    for v, wt in zip(G1.nodes, [3, 1, 1, 3]):
        G1.nodes[v]["weight"] = wt
    cover1, w1 = min_weight_vertex_cover_bipartite_planar(G1)
    _, bw1 = brute_force_mwvc(G1)
    assert verify_vertex_cover(G1, cover1), "Cover invalid"
    assert abs(w1 - bw1) < 1e-9, f"Suboptimal: got {w1}, expected {bw1}"
    print(f"Test 1 (P4, asymmetric weights): cover={cover1}, weight={w1} ✓")

    # ------------------------------------------------------------------
    # Test 2: complete bipartite K_{3,3}  (bipartite, NOT planar)
    # ------------------------------------------------------------------
    G2 = nx.complete_bipartite_graph(3, 3)
    weights = [10, 1, 1, 1, 1, 10]
    for v, wt in zip(G2.nodes, weights):
        G2.nodes[v]["weight"] = wt
    cover2, w2 = min_weight_vertex_cover_bipartite(G2)   # skip planarity check
    _, bw2 = brute_force_mwvc(G2)
    assert verify_vertex_cover(G2, cover2)
    assert abs(w2 - bw2) < 1e-9, f"Suboptimal: got {w2}, expected {bw2}"
    print(f"Test 2 (K_{{3,3}}, mixed weights): cover={cover2}, weight={w2} ✓")

    # ------------------------------------------------------------------
    # Test 3: 6-cycle C6  (bipartite + planar)
    # ------------------------------------------------------------------
    G3 = nx.cycle_graph(6)
    for v in G3.nodes:
        G3.nodes[v]["weight"] = v + 1   # weights 1..6
    cover3, w3 = min_weight_vertex_cover_bipartite_planar(G3)
    _, bw3 = brute_force_mwvc(G3)
    assert verify_vertex_cover(G3, cover3)
    assert abs(w3 - bw3) < 1e-9, f"Suboptimal: got {w3}, expected {bw3}"
    print(f"Test 3 (C6, weights 1–6): cover={cover3}, weight={w3} ✓")

    # ------------------------------------------------------------------
    # Test 4: uniform-weight bipartite graph (should match König size)
    # ------------------------------------------------------------------
    G4 = nx.complete_bipartite_graph(4, 4)
    # all weights = 1, so MWVC weight = size of min vertex cover = max matching
    cover4, w4 = min_weight_vertex_cover_bipartite(G4, default_weight=1.0)
    mm = len(nx.max_weight_matching(G4, maxcardinality=True))
    assert verify_vertex_cover(G4, cover4)
    assert abs(w4 - mm) < 1e-9, f"König mismatch: |cover|={w4}, |matching|={mm}"
    print(f"Test 4 (K_{{4,4}}, unit weights): cover size={int(w4)}, "
          f"max matching={mm} ✓  (König check)")

    # ------------------------------------------------------------------
    # Test 5: disconnected bipartite graph (original crash case)
    # ------------------------------------------------------------------
    # Two disjoint P4 paths — nx.bipartite.sets raises AmbiguousSolution here
    G5a = nx.path_graph(4)
    G5b = nx.relabel_nodes(nx.path_graph(4), {i: i + 10 for i in range(4)})
    G5 = nx.union(G5a, G5b)
    for v, wt in zip(sorted(G5.nodes), [3, 1, 1, 3, 3, 1, 1, 3]):
        G5.nodes[v]["weight"] = wt
    cover5, w5 = min_weight_vertex_cover_bipartite(G5)
    _, bw5 = brute_force_mwvc(G5)
    assert verify_vertex_cover(G5, cover5), "Cover invalid"
    assert abs(w5 - bw5) < 1e-9, f"Suboptimal: got {w5}, expected {bw5}"
    print(f"Test 5 (disconnected bipartite, 2×P4): cover={cover5}, weight={w5} ✓")

    # ------------------------------------------------------------------
    # Test 6: planarity guard
    # ------------------------------------------------------------------
    G6 = nx.complete_bipartite_graph(3, 3)   # K_{3,3} is not planar
    try:
        min_weight_vertex_cover_bipartite_planar(G6)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"Test 6 (planarity guard): correctly raised ValueError: {e} ✓")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_tests()

# Patch: replace _run_tests to include disconnected test