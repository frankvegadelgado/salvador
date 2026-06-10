# Created on June 10th, 2026
# Author: Frank Vega

import itertools
from . import utils

import networkx as nx
from . import solverlp

def vertex_cover_via_bipartite_graph(component_graph):
    auxiliary = nx.Graph()
    weights = {}
    G = component_graph.copy()
    for u in component_graph.nodes():
        neighbors = list(G.neighbors(u))
        G.remove_node(u)
        if len(neighbors) > 0:
            first_auxiliary, previous_neighbor = None, None
            for v in neighbors:
                aux_vertex = (u, v)
                weights[aux_vertex] = 1.0 / component_graph.degree(u)
                weights[(v, u)] = 1.0 / component_graph.degree(v)
                auxiliary.add_edge(aux_vertex, (v, u))
                if previous_neighbor is None:
                    first_auxiliary = aux_vertex
                else:
                    auxiliary.add_edge(aux_vertex, previous_neighbor)       
                previous_neighbor = (v, u)

            if len(neighbors) > 1:
                auxiliary.add_edge(first_auxiliary, previous_neighbor) 

    if not nx.bipartite.is_bipartite(auxiliary):
        raise RuntimeError(f"Reduction failed: no bipartite graph was created")
    
    # Set node weights for the weighted vertex cover algorithm
    nx.set_node_attributes(auxiliary, weights, 'weight')

    auxiliary_solution  = solverlp.find_vertex_cover(auxiliary)
    solution = {
        vertex[0]
        for vertex in auxiliary_solution
    } 

    return solution    

def prune_redundant_vertices(adj, C):
    """
    Linear-time single-pass removal of redundant vertices.
    For every v in C we check (in O(deg(v)) time) whether all its neighbors
    are still in the current cover. If yes, we safely remove it immediately.
    Total time across all calls remains O(n + m).
    """
    C = set(C)
    for v in list(C):          # list() protects against modification during iteration
        # Check if every neighbor is still in C
        all_neighbors_covered = True
        for u in adj.get(v, []):
            if u not in C:
                all_neighbors_covered = False
                break
        if all_neighbors_covered:
            C.remove(v)
    return C


# ============================================================
# Main ensemble
# ============================================================

def find_vertex_cover(graph):
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0:
        return set()

    G = graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_edges() == 0:
        return set()
    
    cover = set()
    
    for component in nx.connected_components(G):
        component_graph = G.subgraph(component)
        cover.update(vertex_cover_via_bipartite_graph(component_graph))

    # Final pruning on final candidate (still linear)
    adj = {v: set(G[v]) for v in G}
    cover_prune = prune_redundant_vertices(adj, cover)
    
    return cover_prune

def find_vertex_cover_brute_force(graph):
    """
    Computes an exact minimum vertex cover in exponential time.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the exact vertex cover, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))
    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))
    
    if working_graph.number_of_nodes() == 0:
        return set()

    n_vertices = len(working_graph.nodes())

    for k in range(1, n_vertices + 1): # Iterate through all possible sizes of the cover
        for candidate in itertools.combinations(working_graph.nodes(), k):
            cover_candidate = set(candidate)
            if utils.is_vertex_cover(working_graph, cover_candidate):
                return cover_candidate
                
    return None



def find_vertex_cover_approximation(graph):
    """
    Computes an approximate vertex cover in polynomial time with an approximation ratio of at most 2 for undirected graphs.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the approximate vertex cover, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    #networkx doesn't have a guaranteed minimum vertex cover function, so we use approximation
    vertex_cover = nx.approximation.vertex_cover.min_weighted_vertex_cover(graph)
    return vertex_cover
