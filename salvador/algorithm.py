"""Public vertex-cover solvers exposed by Salvador (version 0.1.4).

:func:`find_vertex_cover` runs an ensemble of seventeen strategies, numbered in
the order in which they are computed. The base strategies come first, the
improvers (combination, local search, swaps, consensus, refinement) last:

Base strategies

* c1  -- component amplification (:mod:`salvador.component_split`): copies
  ``(u, 0)`` and ``(u, 1)`` only; edges of the NetworkX minimum maximal matching
  ``M`` with both endpoints at index 0 are split off as isolated edges until the
  split graph has as many components as possible without exceeding ``|M|``; the
  NetworkX minimum maximal matching of the split graph is decoded by copy index
  and degree and completed greedily.
* c2  -- maximal matching (:func:`maximal_matching_vertex_cover`, Hvala).
* c3  -- maximum-degree greedy with a bucket queue (:func:`bucket_degree_greedy`, Hvala).
* c4  -- degree-1 weighted reduction, "Hallelujah"
  (:func:`covering_via_reduction_max_degree_1`, Hvala).
* c5  -- Min-to-Min (:func:`min_to_min_vertex_cover_linear`).
* c6  -- primal-dual / local-ratio 2-approximation (:func:`linear_min_weighted_vertex_cover`).
* c7  -- planar forest-core reduction to a weighted MIDS gadget solved by the
  Baker-style PTAS at ``epsilon = 1`` (:mod:`salvador.vc_reduction`).
* c8  -- bipartite planar oriented-incidence reduction with a cycle DP
  (:mod:`salvador.bipartite_reduction`).
* c9  -- bipartite layers (:mod:`salvador.bipartite_peel`): bipartite components
  solved by matching + bounded Hopcroft--Karp phases + Koenig; non-bipartite
  components split into bipartite edge layers (copy indices 0, 1, ...), solved
  in several orders, a vertex selected in one layer discarded from the later ones.
* c10 -- labelled, weighted edge-dominating-set gadget + treewidth DP on the
  bipartite components (:mod:`salvador.eds_treewidth`); reference: best of c1..c9.
* c11 -- weighted subdivision reduction + reducing-peeling
  (:mod:`salvador.subdivision_reduction`); guided variant along the best of c1..c10.
* c12 -- bounded-treewidth exact DP (:mod:`salvador.treewidth_dp`); core fixed
  from the best of c1..c11.

Improvers

* c13 -- union of c2..c8, re-pruned (Hvala's pruning, :func:`prune_redundant_vertices`).
* c14 -- budgeted (1,2)-swap iterated local search (:mod:`salvador.local_search`)
  from the best of c1..c13.
* c15 -- reduce-and-search (:mod:`salvador.reduce_search`) from the best of c1..c14.
* c16 -- consensus forcing (:mod:`salvador.consensus_force`), votes of c1..c15.
* c17 -- budgeted swap-maximize refinement (:func:`refine_cover`) of every
  candidate c1..c16; the smallest refined cover is returned.

Every strategy runs in linear time ``O(n + m)`` for the default parameters (c1 in
``O((n + m) alpha(n))`` with a union-find structure; c9, c10, c12, c14, c15 and
c17 have fixed work budgets or a fixed number of phases), so the whole ensemble
is linear. Four of the candidates (c2, c3, c4, and the pruning) adapt the
ensemble published as the Hvala algorithm (Frank Vega, "The Hvala Algorithm",
Gauge Freedom Journal, v1 i1-004, DOI: 10.65323/gfj.2026.004, 2026; PyPI package
``hvala``), which has worst-case ratio 2; since c2 is a candidate, so does Salvador.
"""

from __future__ import annotations

import itertools
from typing import Any
import networkx as nx

from . import (bipartite_peel, bipartite_reduction, component_split, consensus_force, eds_treewidth, local_search, reduce_search, subdivision_reduction,
               treewidth_dp, utils, vc_reduction)
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

    # Each undirected edge {u, v} appears twice in the adjacency (once from
    # each endpoint). It is processed only from the endpoint that comes first
    # in iteration order, using integer positions. This used to deduplicate
    # with ``tuple(sorted((u, v)))``, which compares node labels with '<' and
    # raises TypeError on graphs whose labels are not mutually comparable
    # (e.g. a mix of strings and tuples). Integer positions avoid label
    # comparison entirely, drop the O(m) ``processed_edges`` set, and visit
    # the edges in exactly the same order as before, so the cover returned on
    # comparable-label graphs is unchanged.
    position = {v: i for i, v in enumerate(adj)}

    # Single pass over all edges: O(m) total iterations
    for u in adj:
        pu = position[u]
        for v in adj[u]:
            if v == u:
                # A self-loop can only be covered by u itself.
                cover.add(u)
                continue
            if position[v] < pu:
                continue  # already processed from v's side
            
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
# c2. Maximal matching (2-approx) -- Hvala strategy 1
# ============================================================

def maximal_matching_vertex_cover(G):
    cover = set()
    min_maximal_matching = nx.approximation.min_maximal_matching(G)
    for u, v in min_maximal_matching:
        cover.add(u)
        cover.add(v)
    return cover

# ============================================================
# c3. Bucket-queue max-degree greedy -- Hvala strategy 2
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
# c4. Weighted reduction to (near) degree-1 instance
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

    # ------------------------------------------------------------------
    # Selection-aware ("sticky") resolution of the auxiliary pairs.
    #
    # The original resolution decided every pair (a, b) independently, by
    # weight and then by an arbitrary str(...) tie-break. On graphs whose
    # vertices all share the same degree (regular / equal-bound-degree
    # graphs) every auxiliary weight 1/d is identical, so each edge fell
    # through to the label tie-break and the decoded cover could contain
    # almost every original vertex (e.g. n - 1 vertices on an n-cycle).
    #
    # The pairs are now resolved with three rules, in priority order:
    #   1. Reuse: if either endpoint's *original* vertex has already been
    #      selected, the edge is already covered and costs nothing -- no
    #      new vertex is added.
    #   2. Weight: otherwise take the endpoint of strictly smaller auxiliary
    #      weight 1/d (the higher-degree original vertex), as before.
    #   3. Tie: on equal weights, select the *neighbor* of the vertex being
    #      scanned, never the scanned vertex itself.
    #
    # Original vertices are scanned in ascending order of degree (largest
    # auxiliary weight first) via a bucket sort. Rule 3 then makes each
    # scanned, still-unselected vertex behave like a member of a maximal
    # independent set whose neighbors enter the cover -- the classical
    # min-degree MIS-complement greedy -- which on regular graphs returns,
    # e.g., n/2 vertices on an even cycle and exactly one side of K_{n,n}.
    #
    # Every pair is inspected at most twice (once from each endpoint) and
    # each selection is O(1), so this stays O(n + m).
    # ------------------------------------------------------------------
    # Every surviving live vertex is an auxiliary vertex (u, i); its
    # original vertex is u.
    origin: dict[Any, Any] = {aux: aux[0] for aux in aux_weight}

    # incident[u] lists (aux of u, aux of the neighbor) for each edge at u.
    incident: dict[Any, list[tuple[Any, Any]]] = {}
    for a, b in pairs:
        incident.setdefault(origin[a], []).append((a, b))
        incident.setdefault(origin[b], []).append((b, a))

    # Bucket sort of original vertices by degree, ascending: O(n + m).
    max_deg = max((len(v) for v in incident.values()), default=0)
    buckets: list[list[Any]] = [[] for _ in range(max_deg + 1)]
    for u in graph.nodes():
        if u in incident:
            buckets[len(incident[u])].append(u)
    scan_order = [u for bucket in buckets for u in bucket]

    def solve(weighted: bool) -> set[Any]:
        selected: set[Any] = set()
        for u in scan_order:
            if u in selected:
                continue  # Rule 1: every edge at u is already covered.
            for x, y in incident[u]:
                w = origin[y]
                if w in selected:
                    continue  # Rule 1: edge (u, w) already covered.
                wx = aux_weight[x] if weighted else 1.0
                wy = aux_weight[y] if weighted else 1.0
                if wx < wy:
                    # Rule 2: u is the higher-degree endpoint. Selecting it
                    # covers all of u's remaining edges, so stop scanning u.
                    selected.add(u)
                    break
                # Rule 2 (wy < wx) or Rule 3 (tie): select the neighbor.
                selected.add(w)
        return selected

    unweighted_sol = solve(False)
    weighted_sol = solve(True)

    return weighted_sol if len(weighted_sol) <= len(unweighted_sol) else unweighted_sol


# ============================================================
# Linear-time redundant-vertex pruning -- Hvala strategy 4 (used by every candidate)
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
# Main ensemble (linear time)
# ============================================================

#: Human-readable names of the strategies, in evaluation order.
CANDIDATE_NAMES: dict[str, str] = {
    "c1": "component amplification (indices 0/1) + NetworkX min maximal matching",
    "c2": "maximal matching",
    "c3": "max-degree greedy",
    "c4": "Hallelujah reduction",
    "c5": "Min-to-Min",
    "c6": "primal-dual",
    "c7": "MIDS Baker reduction",
    "c8": "bipartite planar reduction",
    "c9": "bipartite layers (bounded Hopcroft-Karp + Koenig per layer)",
    "c10": "labelled weighted EDS reduction + treewidth DP, bags <= 6 (bipartite)",
    "c11": "weighted subdivision reduction (bipartite) + reducing-peeling",
    "c12": "bounded-treewidth exact DP",
    "c13": "union re-prune of c2..c8",
    "c14": "(1,2)-swap local search from the best of c1..c13",
    "c15": "reduce-and-search: (1,2)-swap ILS + exact reductions/folds, from best of c1..c14",
    "c16": "consensus forcing: vertices in all/most of c1..c15 forced, rest re-solved",
    "c17": "swap-maximize refinement of c1..c16",
}

#: Base strategies c1..c12 (c10, c11 and c12 complete or guide with the best earlier cover).
BASE_NAMES: tuple[str, ...] = tuple(f"c{i}" for i in range(1, 13))

#: The improvers, computed after the base strategies.
IMPROVER_NAMES: tuple[str, ...] = ("c13", "c14", "c15", "c16", "c17")

#: The candidates whose union c13 re-prunes (Hvala's union pass).
C13_INPUT_NAMES: tuple[str, ...] = ("c2", "c3", "c4", "c5", "c6", "c7", "c8")

#: The candidates whose best cover is c10's reference (core and non-bipartite parts).
C10_INPUT_NAMES: tuple[str, ...] = tuple(f"c{i}" for i in range(1, 10))

#: The strategy whose refined candidates give the final cover.
FINAL_NAME: str = "c17"

#: Decoding rules of c1, see component_split.DEFAULT_VARIANTS.
C1_RULES: tuple[str, ...] = tuple(component_split.DEFAULT_VARIANTS)

#: Default limits of c9 (Hopcroft-Karp phases per bipartite solve, layer orders per component).
DEFAULT_LAYER_PHASES: int = bipartite_peel.DEFAULT_PHASES
DEFAULT_LAYER_ORDERS: int = bipartite_peel.DEFAULT_MAX_ORDERS

#: Default work budget of the c17 refinement, in units of (n + m) per cover.
DEFAULT_REFINE_BUDGET: float | None = 100

#: Default number of consensus-forcing rounds of c16.
DEFAULT_CONSENSUS_ROUNDS: int = consensus_force.DEFAULT_ROUNDS

#: Default limits of the c15 reduce-and-search improver (rounds, search budget per (n + m) per round).
DEFAULT_SEARCH_ROUNDS: int = reduce_search.DEFAULT_ROUNDS
DEFAULT_SEARCH_BUDGET: float = reduce_search.DEFAULT_BUDGET

#: Default limits of the c10 EDS dynamic program (width cap, table budget per (n + m)).
DEFAULT_EDS_MAX_WIDTH: int = eds_treewidth.DEFAULT_MAX_WIDTH
DEFAULT_EDS_BUDGET: float = eds_treewidth.DEFAULT_DP_BUDGET


def _clean(graph: nx.Graph) -> nx.Graph:
    """Copy of ``graph`` without self-loops and isolated vertices."""
    G = graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def _matching_cover(adj: dict[Any, set[Any]]) -> set[Any]:
    """Both endpoints of the NetworkX minimum maximal matching approximation (as c2), in O(n + m)."""
    G = nx.Graph()
    G.add_edges_from((u, v) for u, nb in adj.items() for v in nb if v != u)
    return {x for e in nx.approximation.min_maximal_matching(G) for x in e}


def _consensus_inner(adj: dict[Any, set[Any]]) -> set[Any]:
    """Residual solver of c16: best of pruned NetworkX min maximal matching, Min-to-Min and c11 (max+fold), all O(n + m)."""
    if not adj:
        return set()
    covers = [prune_redundant_vertices(adj, _matching_cover(adj)),
              prune_redundant_vertices(adj, min_to_min_vertex_cover_linear(adj)),
              prune_redundant_vertices(adj, subdivision_reduction.subdivision_vertex_cover(adj, None, "max", True))]
    return min(covers, key=len)


def c1_cover(G: nx.Graph, adj: dict[Any, set[Any]], c1_rule: str | None = None,
             info: dict | None = None) -> set[Any]:
    """Strategy c1 on a clean graph ``G``: component amplification with copy indices 0 and 1.

    Every rule of :data:`C1_RULES` is decoded, completed greedily and pruned; the
    smallest cover is kept (or the one of ``c1_rule``). ``info`` receives ``rule``,
    ``<rule>_size``, the split statistics (``target`` = ``|M|``, ``r``, ``r_prime``,
    ``splits``, ``exact``, ``h_nodes``, ``h_edges``) and ``<rule>_<key>`` for the
    decoding statistics (``matching_h``, ``ties``, ``raw_size``). Linear time.
    """
    split_info: dict = {}
    variants = {rule: prune_redundant_vertices(adj, cover)
                for rule, cover in component_split.component_split_covers(G, split_info).items()}
    key = c1_rule if c1_rule in variants else min(C1_RULES, key=lambda k: len(variants[k]))
    if info is not None:
        info.update({"rule": key, **{f"{k}_size": len(v) for k, v in variants.items()},
                     **{k: split_info[k] for k in ("target", "r", "r_prime", "splits", "exact", "h_nodes", "h_edges")},
                     **{f"{k}_{s}": split_info[k][s] for k in variants for s in ("matching_h", "ties", "raw_size")}})
    return variants[key]


def c9_cover(adj: dict[Any, set[Any]], phases: int = DEFAULT_LAYER_PHASES, max_orders: int = DEFAULT_LAYER_ORDERS,
             info: dict | None = None) -> set[Any]:
    """Strategy c9: bipartite layers (see :mod:`salvador.bipartite_peel`), pruned. Linear time for fixed
    ``phases`` and ``max_orders``. ``info`` receives ``bipartite_components``, ``nonbipartite_components``,
    ``max_layers``, ``layers``, ``orders`` and ``exact``."""
    return prune_redundant_vertices(adj, bipartite_peel.bipartite_layers_cover(adj, phases, max_orders, info))


def _best(cands: dict[str, set[Any]], names) -> set[Any]:
    return min((cands[k] for k in names), key=len)


def ensemble_candidates(
    graph: nx.Graph,
    epsilon: float = 1,
    local_search_budget: float = local_search.DEFAULT_BUDGET,
    tw_max_width: int = treewidth_dp.DEFAULT_MAX_WIDTH,
    tw_budget: float = treewidth_dp.DEFAULT_DP_BUDGET,
    info: dict | None = None,
    eds_max_width: int = DEFAULT_EDS_MAX_WIDTH,
    eds_budget: float = DEFAULT_EDS_BUDGET,
    search_rounds: int = DEFAULT_SEARCH_ROUNDS,
    search_budget: float = DEFAULT_SEARCH_BUDGET,
    consensus_rounds: int = DEFAULT_CONSENSUS_ROUNDS,
    c1_rule: str | None = None,
    layer_phases: int = DEFAULT_LAYER_PHASES,
    layer_orders: int = DEFAULT_LAYER_ORDERS,
) -> dict[str, set[Any]]:
    """Return every candidate ``c1``...``c16`` by name, each a valid vertex cover of ``graph``.

    The strategies are computed in their numbering order (see the module
    docstring): the base strategies c1..c12, then the improvers c13 (union
    re-prune of c2..c8), c14 ((1,2)-swap local search from the best of c1..c13,
    ``local_search_budget * (n + m)`` work; ``0`` disables it), c15
    (reduce-and-search from the best of c1..c14, ``search_rounds`` rounds of
    ``search_budget * (n + m)``), and c16 (consensus forcing over c1..c15, up to
    ``consensus_rounds`` rounds). c10 and c12 are kept only when they are not larger
    than their reference, c14 and c15 only when they are not larger than their
    start, so ``|c14| <= min(|c1|, ..., |c13|)`` and ``|c15| <= |c14|``.

    ``info`` (optional dict) receives the diagnostics, prefixed by the strategy:
    ``c1_*`` (:func:`c1_cover`), ``c9_*`` (:func:`c9_cover`), ``c10_exact``,
    ``c10_raw_size``, ``c10_core_size``, ``c10_width``, ``c10_max_bag``,
    ``c10_bipartite_fraction``; the c11 counts ``c11_deg0``, ``c11_deg1``,
    ``c11_triangle``, ``c11_folds``, ``c11_peels``, ``c11_ref_peels``,
    ``c11_exact``, ``c11_fold_budget_hit``, ``c11_undominated`` (always 0),
    ``c11_nodes``, ``c11_edges`` and ``c11_variant``; ``c12_core_size``,
    ``c12_width``, ``c12_exact`` (treewidth DP); ``c15_start``,
    ``c15_search_gain``, ``c15_reduce_gain``, ``c15_rounds_improved``
    (reduce-and-search); and ``c16_lower_bound``, ``c16_rounds_run``,
    ``c16_forced``, ``c16_certified``, ``c16_gain``, ``c16_certified_ratio``
    (consensus forcing).
    """
    G = _clean(graph)

    if G.number_of_edges() == 0:
        return {name: set() for name in CANDIDATE_NAMES if name != FINAL_NAME}

    adj = {v: set(G[v]) for v in G}
    cands: dict[str, set[Any]] = {}

    # ---- Base strategies ----
    # 1: component amplification of the split graph (indices 0/1) + NetworkX min maximal matching.
    c1_info: dict = {}
    cands["c1"] = c1_cover(G, adj, c1_rule, c1_info)
    # 2-7: linear-time heuristics (Hvala strategies and Salvador's own), each pruned.
    cands["c2"] = prune_redundant_vertices(adj, maximal_matching_vertex_cover(G))
    cands["c3"] = prune_redundant_vertices(adj, bucket_degree_greedy(adj))
    cands["c4"] = prune_redundant_vertices(adj, covering_via_reduction_max_degree_1(G))
    cands["c5"] = prune_redundant_vertices(adj, min_to_min_vertex_cover_linear(adj))
    cands["c6"] = prune_redundant_vertices(adj, linear_min_weighted_vertex_cover(G))
    cands["c7"] = prune_redundant_vertices(adj, vc_reduction.solve_vc(G, epsilon))
    # 8: bipartite planar oriented-incidence reduction, cycle DP; the best pruned variant.
    cands["c8"] = min((prune_redundant_vertices(adj, c) for c in bipartite_reduction.bipartite_planar_covers(adj)),
                      key=len)
    # 9: bipartite layers, each solved by matching + bounded Hopcroft-Karp + Koenig.
    c9_info: dict = {}
    cands["c9"] = c9_cover(adj, layer_phases, layer_orders, c9_info)
    # 10: EDS gadget + treewidth DP on the bipartite components; reference: best of c1..c9.
    best = _best(cands, C10_INPUT_NAMES)
    eds_info: dict = {}
    c10 = prune_redundant_vertices(adj, eds_treewidth.eds_treewidth_cover(adj, best, eds_max_width, eds_budget, eds_info))
    cands["c10"] = c10 if len(c10) <= len(best) else set(best)
    # 11: weighted subdivision reduction; the guided variant peels along the best of c1..c10.
    variant_info: dict = {}
    c11_variants = {key: prune_redundant_vertices(adj, cover)
                    for key, cover in subdivision_reduction.subdivision_covers(
                        adj, info=variant_info, reference=_best(cands, cands)).items()}
    c11_key = min(c11_variants, key=lambda k: len(c11_variants[k]))
    cands["c11"] = c11_variants[c11_key]
    sub_info = variant_info[c11_key]
    # 12: bounded-treewidth exact DP, core fixed from the best of c1..c11.
    ref = _best(cands, cands)
    tw_info: dict = {}
    c12 = prune_redundant_vertices(adj, treewidth_dp.treewidth_vertex_cover(adj, ref, tw_max_width, tw_budget, tw_info))
    cands["c12"] = c12 if len(c12) <= len(ref) else set(ref)

    # ---- Improvers ----
    # 13: union of c2..c8, re-pruned.
    cands["c13"] = prune_redundant_vertices(adj, set().union(*(cands[k] for k in C13_INPUT_NAMES)))
    # 14: budgeted (1,2)-swap iterated local search from the best of c1..c13.
    best = _best(cands, cands)
    c14 = local_search.improve_vertex_cover(adj, best, budget=local_search_budget)
    cands["c14"] = c14 if len(c14) <= len(best) else set(best)
    # 15: reduce-and-search from the best of c1..c14.
    best = _best(cands, cands)
    search_info: dict = {}
    c15 = prune_redundant_vertices(
        adj, reduce_search.reduce_search_cover(adj, best, search_rounds, search_budget, info=search_info))
    cands["c15"] = c15 if len(c15) <= len(best) else set(best)
    # 16: consensus forcing over c1..c15: vertices that all (then most) of them contain are forced
    # into the cover -- deleted from the graph and added to the cover -- and the rest is solved again.
    cons_info: dict = {}
    cands["c16"] = consensus_force.consensus_force_cover(
        adj, [cands[f"c{i}"] for i in range(1, 16)], _consensus_inner, prune_redundant_vertices,
        rounds=consensus_rounds, info=cons_info)

    if info is not None:
        info.update({f"c1_{k}": v for k, v in c1_info.items()})
        info.update({f"c9_{k}": v for k, v in c9_info.items()})
        info.update({"c10_exact": eds_info["exact"], "c10_raw_size": len(c10),
                     "c10_core_size": eds_info["core_size"], "c10_width": eds_info["width"],
                     "c10_max_bag": eds_info["max_bag"], "c10_bipartite_fraction": eds_info["bipartite_fraction"],
                     **{f"c11_{k}": sub_info[k] for k in
                        ("deg0", "deg1", "triangle", "folds", "peels", "ref_peels", "exact", "fold_budget_hit",
                         "undominated", "nodes", "edges")},
                     "c11_variant": f"{c11_key[0]}{'+fold' if c11_key[1] else ''}",
                     "c12_core_size": tw_info["core_size"], "c12_width": tw_info["width"],
                     "c12_exact": tw_info["exact"], "c12_raw_size": len(c12),
                     **{f"c15_{k}": search_info[k] for k in ("start", "search_gain", "reduce_gain", "rounds_improved")},
                     **{f"c16_{k}": cons_info[k] for k in
                        ("lower_bound", "rounds_run", "forced", "certified", "gain", "certified_ratio")}})
    return cands


def maximize_solution(G: nx.Graph, S: set):
    """
    Repair a candidate set into an independent set and greedily maximize it,
    in linear time O(n + m).

    Phase 1 (repair): Nodes of S in conflict (adjacent to other members of S)
    are removed. Each conflicting edge inside S is resolved by discarding the
    endpoint with the higher current conflict count, so a single removal fixes
    as many conflicts as possible. Counts are maintained incrementally, giving
    O(n + m) total work.

    Phase 2 (grow): All nodes outside the repaired set (including those removed
    in Phase 1) are scanned in ascending-degree order (counting sort, O(n)) and
    added whenever they have no neighbor already in the set. This yields a
    maximal independent set containing the repaired kernel, at O(n + m) cost.

    Args:
        G (nx.Graph): An undirected NetworkX graph.
        S (set): Candidate node set (may contain conflicts).

    Returns:
        set: A maximal independent set of G.
    """
    independent = set(S)

    # --- Phase 1: remove conflict nodes ---
    # conflicts[u] = number of neighbors of u inside the current set.
    # Total adjacency scans are bounded by 2m -> O(n + m).
    conflicts = {u: sum(1 for v in G.adj[u] if v in independent) for u in independent}

    for u, v in G.edges():
        if u != v and u in independent and v in independent:
            # Discard the endpoint involved in more remaining conflicts.
            loser = u if conflicts[u] >= conflicts[v] else v
            independent.discard(loser)
            # Keep counts accurate; each node is removed at most once,
            # so these updates cost O(m) overall.
            for w in G.adj[loser]:
                if w in conflicts:
                    conflicts[w] -= 1
            del conflicts[loser]

    # --- Phase 2: add non-conflicting nodes to enlarge the set ---
    # Counting sort by degree (O(n)); low-degree nodes first tends to
    # block fewer future additions.
    buckets = {}
    max_degree = 0
    for u in G.nodes():
        if u not in independent:
            d = G.degree(u)
            buckets.setdefault(d, []).append(u)
            if d > max_degree:
                max_degree = d

    for d in range(max_degree + 1):
        for u in buckets.get(d, ()):
            if all(w not in independent for w in G.adj[u]):
                independent.add(u)

    return independent

def _is_minimal_cover(adj: dict[Any, set[Any]], cover: set[Any]) -> bool:
    """True when ``cover`` is a vertex cover from which no vertex can be dropped."""
    for v in adj:
        if v in cover:
            if all(w in cover for w in adj[v]):
                return False
        elif any(w not in cover for w in adj[v]):
            return False
    return True


def refine_cover(
    G: nx.Graph,
    cover: set[Any],
    v_max: Any = None,
    budget: float | None = DEFAULT_REFINE_BUDGET,
) -> set[Any]:
    """Swap-maximize refinement of one cover (strategy c17).

    For every ``u`` in ``cover ∪ {v_max}`` it forms the cover
    ``(cover - {u}) ∪ N(u)``, i.e. it moves ``u`` into the independent set
    ``I = V - cover`` and evicts the neighbours of ``u`` from it, then grows the
    result into a maximal independent set with :func:`maximize_solution`
    (vertices in ascending degree order) and keeps the complement when it is
    strictly smaller than the best found so far. Every ``u`` is evaluated
    from the original ``cover``. The returned set is a valid cover no larger
    than ``cover``.

    When ``cover`` is a minimal cover (every ensemble candidate is), only
    vertices adjacent to an evicted neighbour can enter the independent set,
    so each move is evaluated locally and gives exactly the set that the
    global :func:`maximize_solution` pass would give. Otherwise every move
    uses the global pass.

    Linear-time guarantee: every adjacency scan, and the sorting of each
    local candidate list, is charged to a work counter, and moves stop as soon
    as the next one would exceed ``budget * (n + m)``. With a fixed
    ``budget`` the routine is worst-case ``O(n + m)``; when the budget is not
    exhausted (the usual case) the result is identical to the unbounded pass.
    ``budget=None`` removes the cap (worst case ``O(|cover| (n + m))``).
    """
    from math import log2

    nodes = set(G.nodes())
    if v_max is None:
        v_max = max(nodes, key=G.degree)
    adj = {v: set(G[v]) for v in G}
    n_plus_m = G.number_of_nodes() + G.number_of_edges()
    limit = float("inf") if budget is None else budget * n_plus_m
    work = 0
    best_solution = cover
    evaluation_pool = cover.union({v_max})

    if not _is_minimal_cover(adj, cover):
        for u in evaluation_pool:
            if work + n_plus_m > limit:
                break
            work += n_plus_m
            candidate = (cover - {u}) | set(G.neighbors(u))
            solution = nodes - maximize_solution(G, nodes - candidate)
            if len(solution) < len(best_solution):
                best_solution = solution
        return best_solution

    pos = {v: i for i, v in enumerate(G.nodes())}
    deg = dict(G.degree())
    best_size = len(cover)
    for u in evaluation_pool:
        if u not in cover:
            continue  # cover is minimal: (cover ∪ N(u)) - {u} = cover, nothing changes
        if work > limit:
            break
        work += deg[u] + 1
        evicted = [w for w in adj[u] if w not in cover]
        # Vertices that may become free: neighbours of evicted vertices that are
        # in the cover and not adjacent to u (u now blocks N(u)).
        cand = set()
        for e in evicted:
            work += deg[e]
            for w in adj[e]:
                if w in cover and w != u and w not in adj[u]:
                    cand.add(w)
        k = len(cand)
        move_cost = sum(deg[w] for w in cand) + int(k * log2(k + 1))
        if work + move_cost > limit:
            break
        work += move_cost
        added: list[Any] = []
        if cand:
            evicted_set = set(evicted)
            added_set: set[Any] = set()
            for w in sorted(cand, key=lambda x: (deg[x], pos[x])):
                ok = True
                for y in adj[w]:
                    if y == u or y in added_set or (y not in cover and y not in evicted_set):
                        ok = False
                        break
                if ok:
                    added.append(w)
                    added_set.add(w)
        # new independent set = I - evicted + {u} + added
        new_size = len(cover) - 1 - len(added) + len(evicted)
        if new_size < best_size:
            best_size = new_size
            best_solution = (cover - {u} - set(added)) | set(evicted)
    return best_solution


def final_candidates(
    graph: nx.Graph,
    epsilon: float = 1,
    local_search_budget: float = local_search.DEFAULT_BUDGET,
    refine_budget: float | None = DEFAULT_REFINE_BUDGET,
    return_refined: bool = False,
    info: dict | None = None,
    tw_max_width: int = treewidth_dp.DEFAULT_MAX_WIDTH,
    tw_budget: float = treewidth_dp.DEFAULT_DP_BUDGET,
    eds_max_width: int = DEFAULT_EDS_MAX_WIDTH,
    eds_budget: float = DEFAULT_EDS_BUDGET,
    search_rounds: int = DEFAULT_SEARCH_ROUNDS,
    search_budget: float = DEFAULT_SEARCH_BUDGET,
    consensus_rounds: int = DEFAULT_CONSENSUS_ROUNDS,
    c1_rule: str | None = None,
    layer_phases: int = DEFAULT_LAYER_PHASES,
    layer_orders: int = DEFAULT_LAYER_ORDERS,
):
    """Every strategy of Salvador 0.1.4 (``c1``...``c17``) by name.

    ``c1``...``c16`` are the candidates of :func:`ensemble_candidates`. ``c17``
    is the final swap-maximize refinement: :func:`refine_cover` is applied to
    every candidate ``c1``...``c16`` (with the same maximum-degree vertex
    ``v_max`` for all of them, and a work budget of ``refine_budget * (n + m)``
    per distinct cover) and the smallest refined cover is kept. A refined
    cover is never larger than its input, so
    ``|c17| <= min(|c1|, ..., |c16|)``. Identical candidate covers are refined
    only once.

    With ``return_refined=True`` the result is ``(candidates, refined)``, where
    ``refined[name]`` is the refinement of candidate ``name``. ``info``
    (optional dict) receives the diagnostics of :func:`ensemble_candidates`.

    Running time: ``O(n + m)`` (``O((n + m) alpha(n))`` for c1) for the default parameters.
    """
    G = _clean(graph)
    cands = ensemble_candidates(G, epsilon, local_search_budget, tw_max_width, tw_budget, info=info,
                                eds_max_width=eds_max_width, eds_budget=eds_budget,
                                search_rounds=search_rounds, search_budget=search_budget,
                                consensus_rounds=consensus_rounds, c1_rule=c1_rule,
                                layer_phases=layer_phases, layer_orders=layer_orders)
    if G.number_of_edges() == 0:
        cands[FINAL_NAME] = set()
        return (cands, {k: set() for k in cands}) if return_refined else cands
    v_max = max(set(G.nodes()), key=G.degree)
    seen: dict[frozenset, set[Any]] = {}
    refined: dict[str, set[Any]] = {}
    for name, cover in cands.items():
        key = frozenset(cover)
        if key not in seen:
            seen[key] = refine_cover(G, cover, v_max, refine_budget)
        refined[name] = seen[key]
    cands[FINAL_NAME] = min(refined.values(), key=len)
    return (cands, refined) if return_refined else cands


def find_vertex_cover(
    graph: nx.Graph,
    epsilon: float = 1,
    refine_budget: float | None = DEFAULT_REFINE_BUDGET,
    eds_max_width: int = DEFAULT_EDS_MAX_WIDTH,
    eds_budget: float = DEFAULT_EDS_BUDGET,
) -> set[Any]:
    """Return the final cover of Salvador 0.1.4 (``c17``, the smallest refined candidate), in linear time.
    """
    return final_candidates(graph, epsilon, refine_budget=refine_budget,
                            eds_max_width=eds_max_width, eds_budget=eds_budget)[FINAL_NAME]


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
