"""Shared helpers for the car/ reproducibility scripts.

* ``car_ratio.py``      -- approximation-ratio study (exact small graphs and
                           large adversarial graphs with certified bounds).
* ``car_scaling.py``    -- doubling-size wall-clock study of the O(n + m) claim.
* ``car_strategies.py`` -- which ensemble strategy wins on which graphs.

This module holds the graph generators, the exact solvers, the certified
lower bounds on tau(G) and small numeric helpers they share. It has no
side effects on import.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path
from typing import Callable, Iterator

import networkx as nx

try:
    import salvador  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - convenience for direct runs
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salvador.version import __version__  # noqa: E402

SEED = 20260627
DEFAULT_EPSILON = 1  # Baker layering width k = ceil(1/epsilon) = 1: keeps the
# whole find_vertex_cover ensemble strictly O(n + m). epsilon < 1 is
# intentionally NOT used: it defeats the linear-time guarantee.
SUB2_TARGET = 7.0 / 4.0

# The explicit seven-vertex bipartite obstruction that was the worst-known
# witness for the pre-ensemble, epsilon=0.1 default (kept for continuity).
OBSTRUCTION_EDGES = [
    (0, 1), (0, 3), (2, 1), (4, 1), (4, 3),
    (5, 0), (5, 2), (5, 4), (5, 6), (6, 1), (6, 3),
]

OUT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Exact optima (no MILP).
# ----------------------------------------------------------------------------
def _bb_vertex_cover_size(G: nx.Graph) -> int:
    """Exact tau(G) = |V| - alpha(G) by branch-and-bound maximum independent set."""
    nodes = list(G.nodes())
    n = len(nodes)
    if G.number_of_edges() == 0:
        return 0
    idx = {v: i for i, v in enumerate(nodes)}
    adj = [0] * n
    for u, v in G.edges():
        adj[idx[u]] |= 1 << idx[v]
        adj[idx[v]] |= 1 << idx[u]
    best = 0

    def search(cand: int, chosen: int) -> None:
        nonlocal best
        if cand == 0:
            best = max(best, chosen)
            return
        if chosen + cand.bit_count() <= best:
            return
        tmp = cand
        branch = (tmp & -tmp).bit_length() - 1
        max_deg = -1
        while tmp:
            bit = tmp & -tmp
            i = bit.bit_length() - 1
            deg = (adj[i] & cand).bit_count()
            if deg > max_deg:
                max_deg, branch = deg, i
            tmp ^= bit
        vbit = 1 << branch
        search(cand & ~vbit & ~adj[branch], chosen + 1)
        search(cand & ~vbit, chosen)

    search((1 << n) - 1, 0)
    return n - best


def one_side(G: nx.Graph) -> set:
    side = {}
    for cc in nx.connected_components(G):
        side.update(nx.bipartite.color(G.subgraph(cc)))
    return {v for v, c in side.items() if c == 0}


def exact_vertex_cover_size(G: nx.Graph) -> tuple[int, str]:
    """Return (tau(G), method). Koenig on bipartite graphs, else B&B MIS."""
    if G.number_of_edges() == 0:
        return 0, "trivial"
    if nx.is_bipartite(G):
        matching = nx.bipartite.maximum_matching(G, top_nodes=one_side(G))
        return len(matching) // 2, "bipartite-matching"
    return _bb_vertex_cover_size(G), "exact-branch"


# ----------------------------------------------------------------------------
# Certified lower bounds on tau(G) for large graphs.
# ----------------------------------------------------------------------------
def greedy_maximal_matching_bound(G: nx.Graph) -> int:
    """|M| for NetworkX's greedy maximal matching (the legacy car/ bound)."""
    return len(nx.algorithms.matching.maximal_matching(G))


def karp_sipser_matching_bound(G: nx.Graph) -> int:
    """|M| for a Karp-Sipser matching, in O(n + m).

    Repeatedly match a vertex of residual degree 1 to its only neighbour
    (always part of some maximum matching of the residual graph); when none
    exists, match an arbitrary remaining edge. Any matching is a lower bound
    on tau(G), and on sparse graphs Karp-Sipser is usually (near-)maximum.
    """
    adj = {v: set(G[v]) - {v} for v in G}
    deg = {v: len(a) for v, a in adj.items()}
    ones = [v for v, d in deg.items() if d == 1]
    matched = set()
    size = 0

    def drop(x):
        for y in adj[x]:
            adj[y].discard(x)
            deg[y] -= 1
            if deg[y] == 1:
                ones.append(y)
        adj[x] = set()
        deg[x] = 0

    order = iter(list(adj))
    while True:
        u = None
        while ones:
            c = ones.pop()
            if c not in matched and deg[c] == 1:
                u = c
                break
        if u is None:
            for c in order:
                if c not in matched and deg[c] > 0:
                    u = c
                    break
        if u is None:
            return size
        v = next(iter(adj[u]))
        matched.add(u)
        matched.add(v)
        size += 1
        drop(u)
        drop(v)


def spectral_bound(G: nx.Graph) -> int | None:
    """Laplacian Hoffman-type bound tau(G) >= n - floor(n (1 - delta/mu)).

    For an independent set S (|S| = s) put x = 1_S - (s/n) 1. Then
    x^T L x = e(S, V \\ S) = sum_{v in S} deg(v) >= s * delta and
    x^T x = s (n - s) / n, so the Rayleigh quotient gives
    s * delta <= mu * s (n - s) / n, i.e. alpha(G) <= n (1 - delta / mu),
    where mu is the largest Laplacian eigenvalue and delta the minimum
    degree (isolated vertices removed first). For d-regular graphs this is
    Hoffman's ratio bound.

    mu is computed by Lanczos (scipy ``eigsh``) and inflated by the residual
    norm plus a relative margin, then capped by the rigorous Anderson-Morley
    bound mu <= max_{uv in E} (deg u + deg v). Returns None when scipy is
    unavailable or the bound is vacuous.
    """
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
    except ModuleNotFoundError:  # pragma: no cover
        return None
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    H.remove_nodes_from(list(nx.isolates(H)))
    n = H.number_of_nodes()
    if n < 3 or H.number_of_edges() == 0:
        return None
    nodes = list(H)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for u, v in H.edges():
        rows += [idx[u], idx[v]]
        cols += [idx[v], idx[u]]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    degs = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(degs) - A
    delta = float(degs.min())
    am = max(degs[idx[u]] + degs[idx[v]] for u, v in H.edges())
    try:
        vals, vecs = eigsh(L, k=1, which="LA", tol=1e-10, maxiter=20000)
        theta = float(vals[0])
        x = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
        resid = float(np.linalg.norm(L @ x - theta * x))
        mu = min(theta + resid + 1e-9 * abs(theta) + 1e-12, float(am))
    except Exception:  # pragma: no cover - fall back to the rigorous cap
        mu = float(am)
    alpha_ub = math.floor(n * (1.0 - delta / mu) + 1e-9)
    bound = n - alpha_ub
    return bound if bound > 0 else None


def clique_packing_bound(G: nx.Graph) -> int:
    """Greedy vertex-disjoint clique packing: tau(G) >= sum (|K_i| - 1).

    A vertex cover must contain all but at most one vertex of every clique,
    and disjoint cliques need disjoint cover vertices, so for any family of
    vertex-disjoint cliques K_1, ..., K_t we have tau(G) >= sum_i (|K_i| - 1).
    A matching is the special case |K_i| = 2; on locally dense graphs (e.g.
    Watts-Strogatz ring lattices, whose consecutive vertices form K4's) the
    bound is far stronger than any matching, which is capped at n/2.

    Cliques are grown greedily from unused vertices in ascending-degree
    order: the candidate set starts as the unused neighbourhood and is
    intersected with the neighbourhood of every vertex added; when it is
    small the next vertex is the candidate with most neighbours inside it.
    """
    adj = {v: set(G[v]) - {v} for v in G}
    order = sorted(adj, key=lambda v: len(adj[v]))
    used: set = set()
    total = 0
    for v in order:
        if v in used:
            continue
        cand = {u for u in adj[v] if u not in used}
        if not cand:
            continue
        clique = [v]
        while cand:
            if len(cand) <= 64:
                u = max(cand, key=lambda x: (len(adj[x] & cand), -len(adj[x])))
            else:
                u = next(iter(cand))
            clique.append(u)
            cand = (cand & adj[u]) - {u}
        used.update(clique)
        total += len(clique) - 1
    return total


def best_lower_bound(G: nx.Graph, *, spectral: bool = True) -> tuple[int, str, dict]:
    """Best certified lower bound on tau(G) and the method that achieved it.

    Bipartite graphs get an exact Hopcroft-Karp maximum matching (exact tau by
    Koenig). Otherwise the maximum of the greedy maximal matching, the
    Karp-Sipser matching, the greedy clique packing and (optionally) the
    spectral bound is used. The
    returned dict lists every bound that was computed.
    """
    if G.number_of_edges() == 0:
        return 0, "trivial", {}
    if nx.is_bipartite(G):
        m = nx.bipartite.hopcroft_karp_matching(G, top_nodes=one_side(G))
        val = len(m) // 2
        return val, "bipartite-maximum-matching (exact tau)", {"hopcroft_karp": val}
    bounds = {
        "greedy_maximal_matching": greedy_maximal_matching_bound(G),
        "karp_sipser_matching": karp_sipser_matching_bound(G),
        "clique_packing": clique_packing_bound(G),
    }
    if spectral:
        sb = spectral_bound(G)
        if sb is not None:
            bounds["spectral_laplacian_hoffman"] = sb
    method = max(bounds, key=bounds.get)
    return bounds[method], method, bounds


# ----------------------------------------------------------------------------
# Part A instance generators (small graphs, exact optima).
# ----------------------------------------------------------------------------
Instance = tuple[str, str, nx.Graph]  # (name, group, graph)


def obstruction_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(7))
    G.add_edges_from(OBSTRUCTION_EDGES)
    return G


def small_instances(rng: random.Random, big: bool) -> Iterator[Instance]:
    """All non-adaptive Part A instances (the hill climb lives in car_ratio.py)."""
    yield from small_instances_before_hill(rng, big)
    yield from small_instances_after_hill(rng, big)


def small_instances_before_hill(rng: random.Random, big: bool) -> Iterator[Instance]:
    """Part A instances drawn before the bipartite hill climb (RNG order kept)."""
    yield "bipartite_obstruction", "Bipartite obstruction (exact)", obstruction_graph()
    for t in range(56 if big else 12):
        perm = list(range(7))
        rng.shuffle(perm)
        G = nx.Graph()
        G.add_nodes_from(range(7))
        G.add_edges_from((perm[u], perm[v]) for u, v in OBSTRUCTION_EDGES)
        yield f"relabel_{t}", "Relabel / order stress (exact)", G
    for _ in range(100 if big else 25):
        a, b = rng.randint(2, 7), rng.randint(2, 7)
        p = rng.choice([0.3, 0.45, 0.6, 0.75])
        G = nx.bipartite.random_graph(a, b, p, seed=rng.randrange(2**32))
        if G.number_of_edges():
            yield f"randbip_{a}_{b}", "Random bipartite (exact)", G


def small_instances_after_hill(rng: random.Random, big: bool) -> Iterator[Instance]:
    """Part A instances drawn after the bipartite hill climb (RNG order kept)."""
    hi = 10 if big else 6
    for r in range(2, hi):
        for c in range(2, hi):
            yield f"grid_{r}x{c}", "Grids, bipartite (exact)", nx.grid_2d_graph(r, c)
    from networkx.generators.atlas import graph_atlas_g
    max_n = 7 if big else 6
    for i, G in enumerate(graph_atlas_g()):
        if 2 <= G.number_of_nodes() <= max_n and G.number_of_edges() > 0:
            yield f"atlas_{i}", "Graph atlas n<=7", G
    for _ in range(173 if big else 30):
        n = rng.randint(7, 12)
        p = rng.choice([0.2, 0.3, 0.45, 0.6])
        G = nx.gnp_random_graph(n, p, seed=rng.randrange(2**32))
        if G.number_of_edges():
            yield f"randgen_{n}", "Random general (exact, n<=12)", G


# ----------------------------------------------------------------------------
# Part B generators (large adversarial graphs).
# ----------------------------------------------------------------------------
def crown_graph(half: int) -> nx.Graph:
    """K_{half,half} minus a perfect matching: dense, near-regular, bipartite."""
    G = nx.Graph()
    for i in range(half):
        for j in range(half):
            if i != j:
                G.add_edge(("L", i), ("R", j))
    return G


def double_star_bridge(a: int, b: int) -> nx.Graph:
    """Two stars joined by a bridge between their centres."""
    G = nx.Graph()
    for i in range(a):
        G.add_edge("cL", ("La", i))
    for i in range(b):
        G.add_edge("cR", ("Rb", i))
    G.add_edge("cL", "cR")
    return G


def hierarchical_star_trap_size(levels: int, branching: int) -> int:
    """Node count of ``hierarchical_star_trap`` without building it."""
    return sum(branching ** i for i in range(levels + 1)) + branching ** levels


def hierarchical_star_trap(levels: int, branching: int) -> nx.Graph:
    """Tree of stars engineered to stress degree-based greedy tie-breaking."""
    G = nx.Graph()
    root = (0, 0)
    G.add_node(root)
    frontier = [root]
    node_id = 1
    for level in range(1, levels + 1):
        nxt = []
        for parent in frontier:
            for _ in range(branching):
                child = (level, node_id)
                node_id += 1
                G.add_edge(parent, child)
                nxt.append(child)
        frontier = nxt
    for leaf in frontier:
        G.add_edge(leaf, (levels + 1, node_id))
        node_id += 1
    return G


def random_regular(n: int, d: int, seed: int) -> nx.Graph:
    if (n * d) % 2:
        n += 1
    return nx.random_regular_graph(d, n, seed=seed)


def barabasi_albert(n: int, m_edges: int, seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m_edges, seed=seed)


def sparse_gnp(n: int, avg_degree: float, seed: int) -> nx.Graph:
    return nx.gnp_random_graph(n, min(1.0, avg_degree / max(1, n - 1)), seed=seed)


def watts_strogatz(n: int, k: int, beta: float, seed: int) -> nx.Graph:
    k = max(2, k - (k % 2))
    return nx.watts_strogatz_graph(n, k, beta, seed=seed)


LargeInstance = tuple[str, str, Callable[[], nx.Graph]]


def large_instances(max_n: int) -> Iterator[LargeInstance]:
    """Part B adversarial families, built lazily (same seeds as before)."""
    seed = SEED
    for half in (200, 400, 800):
        if half * half > max_n * 40:
            continue
        yield f"crown_{half}", "Crown graph K_{h,h} minus perfect matching (dense, near-regular)", (lambda h=half: crown_graph(h))
        yield f"complete_bipartite_{half}", "Complete bipartite (dense)", (lambda h=half: nx.complete_bipartite_graph(h, h))
    for size in (1000, 10000, 100000):
        if size <= max_n:
            yield f"double_star_{size}", "Double star + bridge (matching-heuristic stress)", (lambda s=size: double_star_bridge(s, s))
    for levels, branching in ((6, 4), (8, 4), (10, 4)):
        if hierarchical_star_trap_size(levels, branching) <= max_n:
            yield f"hier_trap_L{levels}_B{branching}", "Hierarchical star trap (degree-greedy tie-break stress)", (lambda l=levels, b=branching: hierarchical_star_trap(l, b))
    for n in (5000, 50000, 200000):
        if n > max_n:
            continue
        for d in (3, 6, 12):
            yield f"regular_n{n}_d{d}", f"Random {d}-regular (bounded-degree adversarial)", (lambda n=n, d=d, s=seed: random_regular(n, d, s))
            seed += 1
    for n in (5000, 50000, 200000):
        if n <= max_n:
            yield f"barabasi_albert_n{n}", "Barabasi-Albert scale-free hub graph", (lambda n=n, s=seed: barabasi_albert(n, 3, s))
            seed += 1
    for n in (5000, 50000, 200000):
        if n <= max_n:
            yield f"gnp_n{n}", "Sparse Erdos-Renyi, avg degree 4", (lambda n=n, s=seed: sparse_gnp(n, 4.0, s))
            seed += 1
    for n in (5000, 50000):
        if n <= max_n:
            yield f"ws_n{n}", "Watts-Strogatz small-world", (lambda n=n, s=seed: watts_strogatz(n, 6, 0.1, s))
            seed += 1


def load_edge_list(path: Path) -> nx.Graph:
    """Load a DIMACS-like edge list: every line whose last two tokens are
    integers is an edge (``e u v``, ``p u v`` or ``u v``); other lines are
    headers or comments."""
    G = nx.Graph()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2 or parts[0] in ("c", "%", "#"):
                continue
            try:
                u, v = int(parts[-2]), int(parts[-1])
            except ValueError:
                continue
            if len(parts) >= 3 and parts[1].lower() == "edge":
                continue
            if u != v:
                G.add_edge(u, v)
    return G


# ----------------------------------------------------------------------------
# Numeric helpers.
# ----------------------------------------------------------------------------
def linreg(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares y = slope*x + intercept, plus R^2."""
    n = len(xs)
    if n < 2:
        return {"slope": None, "intercept": None, "r2": None, "n_points": n}
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"slope": None, "intercept": None, "r2": None, "n_points": n}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return {"slope": slope, "intercept": intercept,
            "r2": 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot, "n_points": n}


def environment() -> dict:
    import platform
    return {"python": platform.python_version(), "platform": platform.platform(),
            "networkx": nx.__version__, "salvador": __version__}
