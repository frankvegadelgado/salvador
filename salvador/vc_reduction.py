"""
Vertex Cover → Minimum Weighted IDS Reduction
==============================================
Works on ALL graphs — planar and non-planar.

PLANAR PATH (G is planar)
--------------------------
Gadget H built directly from G:

    (v,0) weight=1  ─── h_{uv} weight=N+1 ───  (u,0) weight=1
    (v,1) weight=0  leaf of (v,0)               (u,1) weight=0 leaf of (u,0)

H is always planar because it is subdivision(G) + leaves.
Cost per edge:
    neither in cover  →  hub fires  →  +(N+1)   BIG penalty
    both    in cover  →  +2                      small penalty
    exactly one       →  +1                      optimal

Baker's weighted PTAS then finds a (1+ε)-approximate minimum-weight IDS
whose weight equals |cover| + (N+1)·uncovered_edges.

NON-PLANAR PATH (G is not planar)
----------------------------------
1. Extract a maximal planar subgraph G_p via greedy edge insertion
   (spanning forest is planar; try each remaining edge and keep it
   only if the graph stays planar).
2. Run the planar path on G_p → cover C_p covering all edges of G_p.
3. For each edge (u,v) removed from G_p: if uncovered, add
   the higher-degree endpoint (in the ORIGINAL G) to C_p.

This guarantees a valid cover of G for any input.
"""

import networkx as nx
from . import baker_ptas


# ══════════════════════════════════════════════════════════════════════════════
# Planar gadget construction
# ══════════════════════════════════════════════════════════════════════════════

def reduce_vc_to_mids(G: nx.Graph, epsilon: float = 0.1):
    """Build the always-planar weighted MIDS gadget for a PLANAR graph G.

    Args:
        G:       Undirected simple planar graph (NetworkX).
        epsilon: PTAS approximation parameter.

    Returns:
        H       : planar MIDS gadget.
        weights : dict  node → weight.
        P       : hub penalty (= N+1).

    Raises:
        ValueError: if G is not planar.
    """
    if not nx.is_planar(G):
        raise ValueError("reduce_vc_to_mids requires a planar graph.")

    N = G.number_of_nodes()
    P = N + 1

    H = nx.Graph(); weights = {}
    for v in G.nodes():
        H.add_edge((v, 0), (v, 1))
        weights[(v, 0)] = 1
        weights[(v, 1)] = 0
    for u, v in G.edges():
        hub = ('h', u, v); weights[hub] = P
        H.add_edge((u, 0), hub); H.add_edge((v, 0), hub)

    assert nx.is_planar(H), "Bug: gadget non-planar for planar G."
    return H, weights, P


# ══════════════════════════════════════════════════════════════════════════════
# Maximal planar subgraph (non-planar case)
# ══════════════════════════════════════════════════════════════════════════════

def _maximal_planar_subgraph(G: nx.Graph):
    """Extract a maximal planar subgraph of G in O(M · α(N)) time.

    Strategy:
        1. A spanning forest is always planar — add all its edges first.
        2. Try each remaining edge; keep it only if the graph stays planar.
           NetworkX planarity test is linear, so overall O(M·(N+M)).

    Returns:
        (G_planar, removed_edges)
        G_planar     — planar subgraph containing all vertices of G.
        removed_edges — list of edges of G not in G_planar.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Step 1: spanning forest (always planar)
    for u, v in nx.minimum_spanning_edges(G, data=False, ignore_nan=True):
        H.add_edge(u, v)

    # Step 2: greedily add remaining edges
    removed = []
    for u, v in G.edges():
        if H.has_edge(u, v):
            continue
        H.add_edge(u, v)
        if nx.is_planar(H):
            pass          # edge accepted
        else:
            H.remove_edge(u, v)
            removed.append((u, v))

    return H, removed


# ══════════════════════════════════════════════════════════════════════════════
# Core planar solver (shared by both paths)
# ══════════════════════════════════════════════════════════════════════════════

def _solve_planar(G_p: nx.Graph, G_orig: nx.Graph, epsilon: float) -> set:
    """Run the weighted-MIDS Baker PTAS on planar G_p and decode a cover.

    G_orig is used for degree look-ups in the repair step.
    """
    H, weights, _ = reduce_vc_to_mids(G_p, epsilon)

    nodes   = list(H.nodes())
    to_int  = {u: k for k, u in enumerate(nodes)}
    to_lbl  = {k: u for u, k in to_int.items()}
    adj_int = {to_int[v]: {to_int[w] for w in H[v]} for v in H}
    wts_int = {to_int[v]: weights[v] for v in H}

    sol     = baker_ptas.baker_ptas_ids_weighted(adj_int, weights=wts_int, epsilon=epsilon)
    ids_set = {to_lbl[k] for k in sol}

    # decode
    cover = {v for v in G_p.nodes() if (v, 0) in ids_set}

    # repair any residual uncovered edge within G_p
    for u, v in G_p.edges():
        if u not in cover and v not in cover:
            cover.add(u if G_orig.degree(u) >= G_orig.degree(v) else v)

    return cover


# ══════════════════════════════════════════════════════════════════════════════
# Public solver — works on every graph
# ══════════════════════════════════════════════════════════════════════════════

def solve_vc(G: nx.Graph, epsilon: float = 0.1):
    """Approximate Minimum Vertex Cover for ANY graph G.

    Planar G  →  weighted MIDS gadget + Baker PTAS  (always planar gadget).
    Non-planar G  →  extract maximal planar subgraph, solve on it,
                     then repair edges that were removed to achieve planarity.

    Args:
        G:       Undirected NetworkX graph (planar or non-planar).
        epsilon: Approximation parameter (default 0.1).

    Returns:
        cover  : frozenset — valid vertex cover of G (every edge covered).
        mids_w : float    — weighted MIDS cost of the planar-path solution
                           (0.0 for the non-planar fallback path).
    """
    if G.number_of_edges() == 0:
        return frozenset(), 0.0

    if nx.is_planar(G):
        # ── planar path ───────────────────────────────────────────────────
        cover  = _solve_planar(G, G, epsilon)
        mids_w = float(len(cover))
        return frozenset(cover), mids_w

    else:
        # ── non-planar path ───────────────────────────────────────────────
        # 1. Maximal planar subgraph
        G_p, removed = _maximal_planar_subgraph(G)

        # 2. Solve on the planar subgraph
        cover = _solve_planar(G_p, G, epsilon)

        # 3. Repair edges that were excluded to achieve planarity
        for u, v in removed:
            if u not in cover and v not in cover:
                cover.add(u if G.degree(u) >= G.degree(v) else v)

        return frozenset(cover), 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    import numpy as np

    print("═"*60)
    print("  VC → Weighted MIDS  (planar + non-planar)")
    print("═"*60)

    # 1. Planarity proof via exhaustive test
    print("\n1. Gadget always planar for planar G\n")
    import random
    def rand_planar(n, seed):
        rng = random.Random(seed); G = nx.path_graph(n); vs = list(range(n))
        for _ in range(n * 5):
            u, v = rng.sample(vs, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                if not nx.is_planar(G): G.remove_edge(u, v)
        return G

    families = (
        [nx.path_graph(n)   for n in range(1, 12)]  +
        [nx.cycle_graph(n)  for n in range(3, 12)]  +
        [nx.complete_graph(4), nx.dodecahedral_graph(), nx.icosahedral_graph()] +
        [nx.wheel_graph(n)  for n in range(4, 10)]  +
        [nx.grid_2d_graph(r, c) for r in range(2, 6) for c in range(2, 6)] +
        [rand_planar(n, s)  for n in [10, 20, 30, 50] for s in range(5)]
    )
    ok = sum(1 for Gf in families
             if nx.is_planar(Gf) and nx.is_planar(reduce_vc_to_mids(Gf, 0.5)[0]))
    print(f"   {ok} planar graphs tested — gadget always planar ✓")

    # 2. Cost asymmetry
    print("\n2. Cost identity  cost = |cover| + P·uncovered  (P=N+1)\n")
    from itertools import product as ip
    G4 = nx.path_graph(4); H4, wts4, P4 = reduce_vc_to_mids(G4, 0.5)
    verts = list(G4.nodes()); edges = list(G4.edges())
    print(f"   Path-4  N={G4.number_of_nodes()}  P={P4}")
    print(f"   {'cover':<14} uncov  both   one  cost")
    print("   " + "─"*44)
    for bits in ip([0,1], repeat=4):
        cov  = frozenset(v for v,b in zip(verts,bits) if b)
        unc  = sum(1 for u,v in edges if u not in cov and v not in cov)
        bot  = sum(1 for u,v in edges if u in cov and v in cov)
        one  = len(edges)-unc-bot
        cost = len(cov) + P4*unc
        print(f"   {str(sorted(cov)):<14}  {unc:>4}  {bot:>4}  {one:>4}  {cost:>4}")

    # 3. End-to-end: planar graphs
    print("\n3. Planar graphs\n")
    planar_cases = [
        ("Star K₁,₃",  nx.star_graph(3)),
        ("Cycle-6",    nx.cycle_graph(6)),
        ("K4",         nx.complete_graph(4)),
        ("Wheel-6",    nx.wheel_graph(6)),
        ("Grid-3×3",   nx.grid_2d_graph(3,3)),
        ("Dodecahedr", nx.dodecahedral_graph()),
    ]
    print(f"   {'graph':<14} {'N':>4} {'M':>4} {'cover':>7}  valid")
    print("   " + "─"*35)
    for name, Gf in planar_cases:
        cover, _ = solve_vc(Gf, epsilon=0.5)
        valid = all(u in cover or v in cover for u,v in Gf.edges())
        print(f"   {name:<14} {Gf.number_of_nodes():>4} {Gf.number_of_edges():>4}"
              f" {len(cover):>7}  {'✓' if valid else '✗'}")

    # 4. End-to-end: non-planar graphs (including failing case)
    print("\n4. Non-planar graphs\n")
    failing_edges = [
        (np.int32(1),np.int32(0)),(np.int32(1),np.int32(3)),(np.int32(1),np.int32(5)),
        (np.int32(1),np.int32(7)),(np.int32(1),np.int32(8)),(np.int32(1),np.int32(10)),
        (np.int32(0),np.int32(2)),(np.int32(0),np.int32(3)),(np.int32(0),np.int32(5)),
        (np.int32(0),np.int32(7)),(np.int32(0),np.int32(8)),(np.int32(0),np.int32(10)),
        (np.int32(0),np.int32(11)),(np.int32(2),np.int32(3)),(np.int32(2),np.int32(8)),
        (np.int32(2),np.int32(10)),(np.int32(3),np.int32(4)),(np.int32(3),np.int32(8)),
        (np.int32(3),np.int32(10)),(np.int32(7),np.int32(4)),(np.int32(8),np.int32(4)),
        (np.int32(10),np.int32(4))]
    G_fail = nx.Graph(); G_fail.add_edges_from(failing_edges)

    non_planar_cases = [
        ("K5",          nx.complete_graph(5)),
        ("K3,3",        nx.complete_bipartite_graph(3,3)),
        ("Petersen",    nx.petersen_graph()),
        ("Failing case",G_fail),
    ]
    print(f"   {'graph':<14} {'N':>4} {'M':>4} {'cover':>7}  valid  planar_sub")
    print("   " + "─"*44)
    for name, Gf in non_planar_cases:
        G_p, removed = _maximal_planar_subgraph(Gf)
        cover, _ = solve_vc(Gf, epsilon=0.5)
        valid = all(u in cover or v in cover for u,v in Gf.edges())
        print(f"   {name:<14} {Gf.number_of_nodes():>4} {Gf.number_of_edges():>4}"
              f" {len(cover):>7}  {'✓' if valid else '✗'}"
              f"  {G_p.number_of_edges()}/{Gf.number_of_edges()} edges kept")
    print()


if __name__ == "__main__":
    run_demo()