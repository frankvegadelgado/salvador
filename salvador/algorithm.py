"""Public vertex-cover solvers exposed by Salvador.

:func:`find_vertex_cover` runs an ensemble of independently linear-time
O(n + m) heuristics and returns the smallest valid cover found. Four of the
six candidate heuristics, together with the redundant-vertex pruning pass,
adapt the ensemble published as the Hvala algorithm (Frank Vega, "The Hvala
Algorithm", Gauge Freedom Journal, v1 i1-004, DOI: 10.65323/gfj.2026.004,
2026; PyPI package ``hvala``), which proves an unconditional O(n + m) time
and space bound and a worst-case approximation ratio at most 2 for every
graph:

* :func:`maximal_matching_vertex_cover` -- the maximal-matching strategy.
* :func:`bucket_degree_greedy` -- the bucket-queue maximum-degree greedy
  method.
* :func:`covering_via_reduction_max_degree_1` /
  :func:`covering_via_reduction_max_degree_1` -- the degree-1
  weighted-reduction ("Hallelujah") heuristic.
* :func:`prune_redundant_vertices` -- redundant-vertex pruning.

Salvador extends this ensemble with two further linear-time candidates of
its own, the Min-to-Min bucket heuristic (:func:`min_to_min_vertex_cover_linear`)
and a primal-dual/local-ratio 2-approximation
(:func:`linear_min_weighted_vertex_cover`), plus a sixth candidate produced
by :func:`salvador.vc_reduction.solve_vc`, the planar forest-core reduction
to a weighted Minimum Independent Dominating Set gadget solved by an
accuracy-controlled Baker-style PTAS (:mod:`salvador.baker_ptas`). At the
default accuracy ``epsilon = 1`` that PTAS pass degenerates to its
linear-time greedy baseline (Baker layering width ``k = 1``), so every
candidate in the ensemble -- and hence the ensemble as a whole -- runs in
worst-case O(n + m) time; see the runtime analysis in the accompanying
paper for the full argument, including the linear-time fix applied to the
Min-to-Min heuristic.
"""

from __future__ import annotations

import itertools
from typing import Any

import networkx as nx

from . import utils, vc_reduction
from collections import deque

def min_to_min_vertex_cover_linear(adj):
    """
    Computes an approximate vertex cover using the Min-to-Min (MtM) heuristic
    in worst-case O(n + m) time using degree-indexed bucket queues.

    Each currently-minimum-degree vertex is popped and processed exactly
    once per degree level it passes through, following the same amortized
    bucket-queue argument used by :func:`bucket_degree_greedy`. The vertex it
    covers is picked in O(1) time as an arbitrary member of its *current*
    active neighbor set, never by rescanning a whole degree bucket.

    NOTE (linear-time fix): an earlier revision pooled the neighborhoods of
    *every* vertex sharing the current minimum degree on each outer-loop
    step and rescanned that pooled set with ``min(..., key=...)`` to break
    ties by degree. On a regular or near-regular graph the entire active
    vertex set can share one bucket, and because only one vertex of that
    bucket is consumed per rescan, the pooled neighborhood was recomputed
    from scratch up to |bucket| times, degrading to O(n^2) (confirmed by
    profiling on random 3-regular graphs, where wall-clock time grew
    super-linearly with n). This version pops and processes one vertex at a
    time and never rescans a bucket as a whole, which restores the O(n + m)
    bound unconditionally (see the runtime analysis in the accompanying
    paper).

    Parameters:
    - adj: dict mapping each vertex to its list/set of neighbors.

    Returns:
    - cover: set of vertices forming the vertex cover.
    """
    # Active adjacency lists as sets for O(1) edge removal
    adj_set = {v: set(neighbors) for v, neighbors in adj.items()}
    deg = {v: len(adj_set[v]) for v in adj_set}

    maxd = max(deg.values(), default=0)
    if maxd == 0:
        return set()

    # Bucket queues storing active vertices grouped by current degree
    buckets = [deque() for _ in range(maxd + 1)]
    for v, d in deg.items():
        if d > 0:
            buckets[d].append(v)

    cover = set()
    min_d = 1

    while min_d <= maxd:
        # Advance min_d pointer to the smallest non-empty degree bucket
        while min_d <= maxd and not buckets[min_d]:
            min_d += 1

        if min_d > maxd:
            break

        # Pop exactly one minimum-degree vertex; never pool the bucket.
        v = buckets[min_d].popleft()
        if deg[v] != min_d or deg[v] == 0:
            continue  # stale entry left behind by an earlier degree update

        # MtM Selection, O(1): an arbitrary current neighbor of v becomes
        # the cover vertex (deterministic tie-breaking is not needed for
        # correctness or for the linear-time guarantee).
        target = next(iter(adj_set[v]))

        # Add target to the vertex cover
        cover.add(target)

        # Remove target from the active graph and update degree buckets in O(deg(target))
        for nbr in list(adj_set[target]):
            adj_set[nbr].discard(target)
            deg[nbr] -= 1
            if deg[nbr] > 0:
                buckets[deg[nbr]].append(nbr)

        deg[target] = 0
        adj_set[target].clear()

        # v lost its edge to target; requeue it at its new (lower) degree
        # instead of leaving a stale higher-degree bucket entry behind.
        if deg[v] > 0:
            buckets[deg[v]].append(v)

        # Rewind min_d pointer if neighbor degree updates created a lower minimum
        if min_d > 1 and buckets[min_d - 1]:
            min_d -= 1

    return cover

def linear_min_weighted_vertex_cover(adj, weights=None):
    """
    Computes a 2-approximate minimum weighted vertex cover in O(n + m) time 
    using the primal-dual / local-ratio method.
    
    Parameters:
    - adj: dict mapping each node to a list/set of its neighbors.
    - weights: dict mapping each node to its weight (default: 1.0 for all nodes).
    
    Returns:
    - cover: set of nodes forming the vertex cover.
    """
    if weights is None:
        weights = {v: 1.0 for v in adj}
    else:
        weights = {v: float(weights.get(v, 1.0)) for v in adj}
        
    spent = {v: 0.0 for v in adj}
    cover = set()
    processed_edges = set()
    
    # Single pass over all edges: O(m) total iterations
    for u in adj:
        for v in adj[u]:
            edge = tuple(sorted((u, v)))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
            
            if u in cover or v in cover:
                continue
                
            # Find the maximum weight constraint delta for the edge endpoints
            rem_u = weights[u] - spent[u]
            rem_v = weights[v] - spent[v]
            delta = min(rem_u, rem_v)
            
            if delta > 0:
                spent[u] += delta
                spent[v] += delta
                
            # Include vertices whose budgets are fully saturated
            if abs(weights[u] - spent[u]) < 1e-9:
                cover.add(u)
            if abs(weights[v] - spent[v]) < 1e-9:
                cover.add(v)
                
    return cover

# ============================================================
# 1. Maximal matching (2-approx) -- Hvala strategy 1
# ============================================================

def maximal_matching_vertex_cover(G):
    cover = set()
    min_maximal_matching = nx.approximation.min_maximal_matching(G)
    for u, v in min_maximal_matching:
        cover.add(u)
        cover.add(v)
    return cover

# ============================================================
# 2. Bucket-queue max-degree greedy -- Hvala strategy 2
#    (linear-time O(n + m); worst-case approximation ratio
#    Theta(log Delta) by Johnson's classical bound)
# ============================================================

def bucket_degree_greedy(adj):
    """
    Linear-time max-degree greedy vertex cover.

    Runs in O(n + m) time using a bucket queue indexed by vertex degree.
    NOTE: the worst-case approximation ratio of the max-degree greedy
    heuristic is Theta(log Delta) (Johnson's classical bound), where
    Delta is the maximum degree of the input graph.
    """
    deg = {v: len(adj[v]) for v in adj}
    maxd = max(deg.values(), default=0)
    buckets = [deque() for _ in range(maxd + 1)]
    for v, d in deg.items():
        buckets[d].append(v)

    removed = set()
    cover = set()

    # Process from highest to lowest degree (guarantees validity)
    for d in reversed(range(maxd + 1)):
        q = buckets[d]
        while q:
            v = q.popleft()
            if v in removed or deg[v] != d:
                continue
            if deg[v] == 0:
                continue
            cover.add(v)
            removed.add(v)
            for u in adj[v]:
                if u not in removed:
                    deg[u] -= 1
                    buckets[deg[u]].append(u)

    return cover


# ============================================================
# 3. Weighted reduction to (near) degree-1 instance
#    -- Hvala strategy 3, the "Hallelujah heuristic"
# ============================================================

def min_weighted_vertex_cover_max_degree_1(G, weight='weight'):
    """
    Solver used by the reduction (works on the star-like auxiliary graph).
    """
    vertex_cover = set()
    visited = set()
    for node in list(G.nodes()):
        if node in visited:
            continue
        degree = G.degree(node)
        if degree == 0:
            visited.add(node)
        elif degree == 1:
            neighbor = list(G.neighbors(node))[0]
            if neighbor not in visited:
                node_weight = G.nodes[node].get(weight, 1)
                neighbor_weight = G.nodes[neighbor].get(weight, 1)
                # Tie-break with str(...) rather than a raw '<': node labels
                # are not guaranteed mutually comparable (e.g. a graph that
                # mixes string and tuple labels), and a raw '<' raises
                # TypeError on such inputs instead of returning a cover.
                if (node_weight < neighbor_weight or
                    (node_weight == neighbor_weight and str(node) < str(neighbor))):
                    vertex_cover.add(node)
                else:
                    vertex_cover.add(neighbor)
                visited.add(node)
                visited.add(neighbor)
    return vertex_cover


def covering_via_reduction_max_degree_1(graph):
    """
    Linear-time reduction heuristic.
    Creates one auxiliary vertex per original edge and solves the resulting
    (star-shaped) instance twice (unweighted + weighted 1/d). Maps back to
    original vertices.

    Implementation note: this used to build and mutate a full ``nx.Graph``
    for the auxiliary star-shaped instance (one ``add_edge``/``remove_node``
    call per original vertex and edge). NetworkX's per-call object overhead
    dominated the wall-clock cost on large graphs even though the algorithm
    is O(n + m). This version performs the identical construction directly
    on plain dict/set adjacency, which is asymptotically the same but with
    a much smaller constant factor; it has been checked to return exactly
    the same cover as the ``nx.Graph``-based version on randomized tests,
    including graphs whose original node labels are themselves tuples.
    """
    live_adj: dict[Any, set[Any]] = {v: set(graph[v]) for v in graph.nodes()}
    aux_weight: dict[Any, float] = {}

    for u in list(graph.nodes()):
        neighbors = list(live_adj.get(u, ()))
        k = len(neighbors)
        if k > 0:
            for i, w in enumerate(neighbors):
                aux_vertex = (u, i)
                aux_weight[aux_vertex] = 1.0 / k
                # aux_vertex takes over u's edge to w in the live graph, the
                # same rewiring nx.Graph.add_edge/remove_node performed.
                if w in live_adj:
                    live_adj[w].discard(u)
                    live_adj[w].add(aux_vertex)
                live_adj[aux_vertex] = {w}
        if u in live_adj:
            del live_adj[u]

    # Every remaining node is an auxiliary vertex with exactly one neighbor
    # (itself possibly another auxiliary vertex): a perfect matching over
    # the original edges. Extract each pair once.
    pairs: list[tuple[Any, Any]] = []
    seen: set[Any] = set()
    for a, nbrs in live_adj.items():
        if a in seen or not nbrs:
            continue
        b = next(iter(nbrs))
        pairs.append((a, b))
        seen.add(a)
        seen.add(b)

    def solve(weighted: bool) -> set[Any]:
        cover: set[Any] = set()
        for a, b in pairs:
            wa = aux_weight[a] if weighted else 1.0
            wb = aux_weight[b] if (weighted and b in aux_weight) else 1.0
            if wa < wb or (wa == wb and str(a) < str(b)):
                cover.add(a)
            else:
                cover.add(b)
        return cover

    unweighted_cover = solve(False)
    weighted_cover = solve(True)

    # Map back to original vertices
    def map_back(cover):
        return {x[0] if x in aux_weight else x for x in cover}

    unweighted_sol = map_back(unweighted_cover)
    weighted_sol = map_back(weighted_cover)

    return weighted_sol if len(weighted_sol) <= len(unweighted_sol) else unweighted_sol


# ============================================================
# Linear-time redundant-vertex pruning -- Hvala strategy 4
# (replaces bitsets + local search)
# ============================================================

def prune_redundant_vertices(adj, C):
    """
    Prunes redundant vertices and repairs any uncovered edges in O(n + m) time.
    
    Parameters:
    - adj: dict mapping each vertex to its list/set of neighbors.
    - C: iterable/set of candidate cover vertices.
    
    Returns:
    - C: set forming a valid, minimal vertex cover.
    """
    C = set(C)
    
    # ------------------------------------------------------------------
    # PHASE 1: PRUNE (O(n + m))
    # Remove vertices whose entire neighborhood is already inside C.
    # ------------------------------------------------------------------
    for v in list(C):
        if all(u in C for u in adj.get(v, [])):
            C.remove(v)
            
    # ------------------------------------------------------------------
    # PHASE 2: REPAIR (O(n + m))
    # Scan all edges in O(m) time. If an edge (u, v) is uncovered 
    # (neither endpoint in C), add an endpoint to repair the cover.
    # ------------------------------------------------------------------
    for u in adj:
        if u not in C:
            for v in adj[u]:
                if v not in C:
                    # Edge (u, v) is uncovered -> Add v to C to repair
                    C.add(v)
                    
    # ------------------------------------------------------------------
    # PHASE 3: FINAL PRUNE CLEANUP (O(n + m))
    # Remove any new redundancies created during the repair step.
    # ------------------------------------------------------------------
    for v in list(C):
        if all(u in C for u in adj.get(v, [])):
            C.remove(v)
            
    return C


# ============================================================
# Main ensemble (now strictly linear-time O(n + m))
# ============================================================

def find_vertex_cover(graph: nx.Graph, epsilon: float = 1) -> set[Any]:
    """Return the smallest cover found by the linear-time ensemble c1-c7.

    Parameters:
    - graph: an undirected NetworkX graph.
    - epsilon: accuracy parameter forwarded to the Baker-PTAS candidate
      ``c6`` (:func:`salvador.vc_reduction.solve_vc`); default ``1`` gives
      Baker layering width ``k = 1``, which reduces ``c6`` to its
      linear-time greedy baseline and keeps the whole ensemble strictly
      ``O(n + m)``. Passing ``epsilon < 1`` makes only ``c6`` more thorough
      (and ``O(n/\\varepsilon)`` for that one candidate); it is intended for
      offline quality experiments, not the default production call.

    Every one of ``c1``...``c6`` is independently a valid vertex cover
    computed in worst-case ``O(n + m)`` time at the default ``epsilon``; the
    union-and-reprune candidate ``c7`` costs one further ``O(n + m)`` pass.
    Taking the minimum over 7 linear-time candidates is still ``O(n + m)``.
    """
    G = graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_edges() == 0:
        return set()
    
    adj = {v: set(G[v]) for v in G}

    # 1–5: all linear-time heuristics producing valid covers
    c1 = maximal_matching_vertex_cover(G) # min-maximal-matching
    c2 = bucket_degree_greedy(adj)  # max-degree
    c3 = covering_via_reduction_max_degree_1(G) # reduction-based solver
    c4 = min_to_min_vertex_cover_linear(adj) # Min-to-Min (MtM) heuristic
    c5 = linear_min_weighted_vertex_cover(G) # NetworkX built-in 2-approx in O(m + n)
    c6 = vc_reduction.solve_vc(G, epsilon) # Using MIDS Baker PTAS reduction

    # Final pruning on every candidate (still linear)
    c1 = prune_redundant_vertices(adj, c1)
    c2 = prune_redundant_vertices(adj, c2)
    c3 = prune_redundant_vertices(adj, c3)
    c4 = prune_redundant_vertices(adj, c4)
    c5 = prune_redundant_vertices(adj, c5)
    c6 = prune_redundant_vertices(adj, c6)
    # 7: prune on the union (strongest starting cover)
    c7 = prune_redundant_vertices(adj, c1 | c2 | c3 | c4 | c5 | c6)

    return min([c1, c2, c3, c4, c5, c6, c7], key=len)


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
