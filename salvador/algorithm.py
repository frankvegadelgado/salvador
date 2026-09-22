"""Public vertex-cover solvers exposed by Salvador."""

from __future__ import annotations

import itertools
from typing import Any

import networkx as nx

from . import utils, vc_reduction
from collections import deque

def min_to_min_vertex_cover_linear(adj):
    """
    Computes an approximate vertex cover using the Min-to-Min (MtM) heuristic
    in O(n + m) linear time using degree-indexed bucket sets.

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

    # Buckets storing active vertices grouped by current degree
    buckets = [set() for _ in range(maxd + 1)]
    for v, d in deg.items():
        if d > 0:
            buckets[d].add(v)

    cover = set()
    min_d = 1

    while min_d <= maxd:
        # Advance min_d pointer to the smallest non-empty degree bucket
        while min_d <= maxd and not buckets[min_d]:
            min_d += 1
        
        if min_d > maxd:
            break

        # Pool all vertices that currently share the minimum non-zero degree
        min_vertices = buckets[min_d]
        
        # Collect active neighbors of all minimum-degree vertices
        neighbors = set()
        for v in min_vertices:
            neighbors.update(adj_set[v])

        if not neighbors:
            buckets[min_d].clear()
            continue

        # MtM Selection: Pick target neighbor with min degree (deterministic tie-breaking)
        target = min(neighbors, key=lambda x: (deg[x], x))

        # Add target to the vertex cover
        cover.add(target)

        # Remove target from the active graph and update degree buckets in O(deg(target))
        buckets[deg[target]].discard(target)
        deg[target] = 0

        for nbr in list(adj_set[target]):
            adj_set[nbr].remove(target)
            old_d = deg[nbr]
            buckets[old_d].discard(nbr)
            
            deg[nbr] -= 1
            new_d = deg[nbr]
            if new_d > 0:
                buckets[new_d].add(nbr)

        adj_set[target].clear()
        
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
# 1. Maximal matching (2-approx)
# ============================================================

def maximal_matching_vertex_cover(G):
    cover = set()
    min_maximal_matching = nx.approximation.min_maximal_matching(G)
    for u, v in min_maximal_matching:
        cover.add(u)
        cover.add(v)
    return cover

# ============================================================
# 2. Bucket-queue max-degree greedy
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
# 4. Weighted reduction to (near) degree-1 instance
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
                if (node_weight < neighbor_weight or
                    (node_weight == neighbor_weight and node < neighbor)):
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
    """
    G = graph.copy()
    weights = {}

    for u in list(graph.nodes()):
        neighbors = list(G.neighbors(u))
        G.remove_node(u)
        k = len(neighbors)
        if k == 0:
            continue
        for i, v in enumerate(neighbors):
            aux_vertex = (u, i)
            G.add_edge(aux_vertex, v)
            weights[aux_vertex] = 1.0 / k

    # Unweighted solve
    unweighted_cover = min_weighted_vertex_cover_max_degree_1(G)

    # Weighted solve
    nx.set_node_attributes(G, weights, 'weight')
    weighted_cover = min_weighted_vertex_cover_max_degree_1(G)

    # Map back to original vertices
    def map_back(cover):
        res = set()
        for x in cover:
            if isinstance(x, tuple):
                res.add(x[0])
            else:
                res.add(x)
        return res

    unweighted_sol = map_back(unweighted_cover)
    weighted_sol = map_back(weighted_cover)

    return weighted_sol if len(weighted_sol) <= len(unweighted_sol) else unweighted_sol


# ============================================================
# Linear-time redundant-vertex pruning (replaces bitsets + local search)
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
