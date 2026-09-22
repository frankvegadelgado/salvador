from __future__ import annotations

"""car/ -- reproducibility suite for Salvador (default call, epsilon=1).

This suite has two parts.

PART A -- small-graph exact-ratio suite (unchanged in spirit from earlier
versions, now run under the linear-time default ``epsilon=1`` instead of the
previous ``epsilon=0.1``). It tests the public default call
``find_vertex_cover(G, epsilon=1)`` against exact optima. It does NOT use
MILP: optima are computed by maximum matching and Koenig's theorem on
bipartite instances, and by a deterministic branch-and-bound
maximum-independent-set solver otherwise (``tau(G) = |V| - alpha(G)``).

PART B -- large adversarial-graph suite (new). Exact optima are generally
NP-hard to certify at this scale, so every reported ratio in Part B is
instead a *provable upper bound* on the true approximation ratio: any
matching M of G satisfies ``tau(G) >= |M|`` (each matched edge is disjoint,
so a vertex cover needs a distinct endpoint per matched edge), hence
``|cover| / |M| >= |cover| / tau(G)``. For bipartite adversarial families we
use a maximum matching via Hopcroft-Karp (so Koenig's theorem makes the
bound *exact*, i.e. ``|M| = tau(G)``); for non-bipartite families we use a
fast linear-time greedy maximal matching, which is only a lower bound on
tau(G) (so the reported ratio over-states the true ratio, conservatively).

Part B also runs a doubling-size scaling study on two adversarial families
to empirically verify the O(n + m) linear-time guarantee proved for the
default call (epsilon=1) in the accompanying paper: it fits wall-clock time
against (n + m) by least squares and reports the slope, intercept, and R^2
of the fit, plus the raw time/(n+m) ratio at every scale.

Run from the repository root with:

    python car/car_experiment.py                 # Part A + Part B, default sizes
    python car/car_experiment.py --quick          # smaller/faster Part A
    python car/car_experiment.py --skip-large     # Part A only
    python car/car_experiment.py --skip-small     # Part B only
    python car/car_experiment.py --max-n 500000   # push Part B further
    python car/car_experiment.py --exact-matching-limit 20000  # widen exact
                                                    # (Hopcroft-Karp / blossom)
                                                    # matching bounds

Outputs (repository-relative):
    car/car_experiment.json   -- full machine-readable results
    car/car_summary.csv       -- Part A family summary (unchanged format)
    car/car_large_summary.csv -- Part B family summary
    car/car_scaling.csv       -- Part B doubling-size timing/regression rows
"""

import argparse
import gc
import itertools
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Iterable

import networkx as nx

try:
    from salvador.algorithm import find_vertex_cover
    from salvador.version import __version__
except ModuleNotFoundError:  # pragma: no cover - convenience for direct runs
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from salvador.algorithm import find_vertex_cover
    from salvador.version import __version__

SEED = 20260627
DEFAULT_EPSILON = 1  # Baker layering width k = ceil(1/epsilon) = 1: the
# tree-decomposition PTAS pass is skipped entirely (see
# salvador.baker_ptas.baker_ptas_ids_weighted), which is what keeps the
# whole find_vertex_cover ensemble strictly O(n + m). epsilon=0.1 (k=10) is
# intentionally NOT used here: it defeats the linear-time guarantee that
# this suite exists to demonstrate.
SUB2_TARGET = 7.0 / 4.0

# The explicit seven-vertex bipartite obstruction that was the worst-known
# witness for the pre-ensemble, epsilon=0.1 default (kept for continuity;
# see Proposition on the 7/4 lower bound in earlier paper revisions).
OBSTRUCTION_EDGES = [
    (0, 1), (0, 3), (2, 1), (4, 1), (4, 3),
    (5, 0), (5, 2), (5, 4), (5, 6), (6, 1), (6, 3),
]


# ----------------------------------------------------------------------------
# PART A: exact optima (no MILP).
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


def exact_vertex_cover_size(G: nx.Graph) -> tuple[int, str]:
    """Return (tau(G), method).  Uses Koenig on bipartite graphs, else B&B MIS."""
    if G.number_of_edges() == 0:
        return 0, "trivial"
    if nx.is_bipartite(G):
        matching = nx.bipartite.maximum_matching(G, top_nodes=_one_side(G))
        return len(matching) // 2, "bipartite-matching"
    return _bb_vertex_cover_size(G), "exact-branch"


def _one_side(G: nx.Graph):
    side = {}
    for cc in nx.connected_components(G):
        sub = G.subgraph(cc)
        color = nx.bipartite.color(sub)
        side.update(color)
    return {v for v, c in side.items() if c == 0}


def evaluate(name: str, G: nx.Graph, group: str) -> dict:
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    cover = set(find_vertex_cover(G, epsilon=DEFAULT_EPSILON))
    opt, method = exact_vertex_cover_size(G)
    ratio = 1.0 if opt == 0 else len(cover) / opt
    valid = all(u in cover or v in cover for u, v in G.edges())
    return {
        "group": group, "name": name,
        "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "cover": len(cover), "opt": opt, "ratio": ratio,
        "above_7_4": bool(ratio > SUB2_TARGET + 1e-12),
        "valid": bool(valid), "method": method,
        "edges": [[int(u), int(v)] for u, v in G.edges()] if G.number_of_nodes() <= 16 else None,
    }


def obstruction_rows() -> list[dict]:
    G = nx.Graph(); G.add_nodes_from(range(7)); G.add_edges_from(OBSTRUCTION_EDGES)
    return [evaluate("bipartite_obstruction", G, "Bipartite obstruction (exact)")]


def relabel_stress(rng: random.Random, trials: int) -> list[dict]:
    rows = []
    base = list(range(7))
    for t in range(trials):
        perm = base[:]
        rng.shuffle(perm)
        mp = {i: perm[i] for i in range(7)}
        G = nx.Graph(); G.add_nodes_from(range(7))
        G.add_edges_from((mp[u], mp[v]) for u, v in OBSTRUCTION_EDGES)
        rows.append(evaluate(f"relabel_{t}", G, "Relabel / order stress (exact)"))
    return rows


def random_bipartite(rng: random.Random, count: int) -> list[dict]:
    rows = []
    for _ in range(count):
        a = rng.randint(2, 7); b = rng.randint(2, 7)
        p = rng.choice([0.3, 0.45, 0.6, 0.75])
        G = nx.bipartite.random_graph(a, b, p, seed=rng.randrange(2**32))
        if G.number_of_edges() == 0:
            continue
        rows.append(evaluate(f"randbip_{a}_{b}", G, "Random bipartite (exact)"))
    return rows


def bipartite_hill_climb(rng: random.Random, restarts: int, steps: int) -> list[dict]:
    rows = []
    for r in range(restarts):
        a, b = rng.randint(3, 6), rng.randint(3, 6)
        G = nx.bipartite.random_graph(a, b, 0.5, seed=rng.randrange(2**32))
        if G.number_of_edges() == 0:
            G.add_edge(0, a)
        best = evaluate(f"hill_{r}_0", G, "Bipartite hill-climb (exact)")
        left = list(range(a)); right = list(range(a, a + b))
        for s in range(1, steps + 1):
            H = G.copy()
            u, v = rng.choice(left), rng.choice(right)
            if H.has_edge(u, v):
                H.remove_edge(u, v)
                if H.number_of_edges() == 0:
                    H.add_edge(u, v)
            else:
                H.add_edge(u, v)
            trial = evaluate(f"hill_{r}_{s}", H, "Bipartite hill-climb (exact)")
            if trial["ratio"] >= best["ratio"]:
                G, best = H, trial
        rows.append(best)
    return rows


def bipartite_grids(big: bool) -> list[dict]:
    rows = []
    hi = 10 if big else 6
    for r in range(2, hi):
        for c in range(2, hi):
            rows.append(evaluate(f"grid_{r}x{c}", nx.grid_2d_graph(r, c), "Grids, bipartite (exact)"))
    return rows


def atlas_rows(max_n: int) -> list[dict]:
    from networkx.generators.atlas import graph_atlas_g
    rows = []
    for i, G in enumerate(graph_atlas_g()):
        if 2 <= G.number_of_nodes() <= max_n and G.number_of_edges() > 0:
            rows.append(evaluate(f"atlas_{i}", G, "Graph atlas n<=7"))
    return rows


def random_general(rng: random.Random, count: int) -> list[dict]:
    rows = []
    for _ in range(count):
        n = rng.randint(7, 12)
        p = rng.choice([0.2, 0.3, 0.45, 0.6])
        G = nx.gnp_random_graph(n, p, seed=rng.randrange(2**32))
        if G.number_of_edges() == 0:
            continue
        rows.append(evaluate(f"randgen_{n}", G, "Random general (exact, n<=12)"))
    return rows


def summarise(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"instances": 0}
    worst = max(rows, key=lambda r: r["ratio"])
    return {
        "instances": len(rows),
        "mean_ratio": statistics.fmean(r["ratio"] for r in rows),
        "max_ratio": max(r["ratio"] for r in rows),
        "count_above_7_4": sum(1 for r in rows if r["above_7_4"]),
        "all_valid": all(r["valid"] for r in rows),
        "worst_instance": worst,
    }


def run_part_a(big: bool) -> dict:
    rng = random.Random(SEED)
    started = time.time()
    rows = []
    rows += obstruction_rows()
    rows += relabel_stress(rng, 56 if big else 12)
    rows += random_bipartite(rng, 100 if big else 25)
    rows += bipartite_hill_climb(rng, 80 if big else 12, 40 if big else 12)
    rows += bipartite_grids(big)
    rows += atlas_rows(7 if big else 6)
    rows += random_general(rng, 173 if big else 30)

    by_group = {}
    for g in sorted({r["group"] for r in rows}):
        by_group[g] = summarise([r for r in rows if r["group"] == g])
    overall = summarise(rows)

    return {
        "experiment": f"car/ Part A: default-call (epsilon={DEFAULT_EPSILON}) exact-ratio test",
        "salvador_version": __version__,
        "seed": SEED,
        "default_epsilon": DEFAULT_EPSILON,
        "threshold": {"sub2_target": SUB2_TARGET, "statement": "7/4 = 2 - 1/4 (legacy epsilon=0.1 ceiling, kept as a sanity backstop)"},
        "method": {
            "candidate_solver": f"salvador.algorithm.find_vertex_cover (default epsilon={DEFAULT_EPSILON})",
            "optimum_solver": "Koenig/maximum-matching on bipartite graphs, else branch-and-bound MIS; no MILP",
            "ratio": "|C| / tau(G)  (exact)",
        },
        "conclusion": {
            "max_ratio": overall["max_ratio"],
            "all_valid": overall["all_valid"],
            "count_above_7_4": overall["count_above_7_4"],
            "supports_7_4": overall["count_above_7_4"] == 0,
        },
        "overall_summary": overall,
        "summary_by_group": by_group,
        "raw_rows": rows,
        "elapsed_seconds": time.time() - started,
    }


# ----------------------------------------------------------------------------
# PART B: large adversarial graphs.
# ----------------------------------------------------------------------------
def complete_bipartite_adversarial(a: int, b: int) -> nx.Graph:
    """Dense complete bipartite graph: stresses every O(m)-per-edge routine."""
    return nx.complete_bipartite_graph(a, b)


def crown_graph(half: int) -> nx.Graph:
    """K_{half,half} minus a perfect matching: dense, near-regular, bipartite;
    classic adversarial instance for degree-based greedy tie-breaking since
    every vertex has identical degree half-1."""
    G = nx.Graph()
    left = [("L", i) for i in range(half)]
    right = [("R", i) for i in range(half)]
    G.add_nodes_from(left); G.add_nodes_from(right)
    for i in range(half):
        for j in range(half):
            if i != j:
                G.add_edge(("L", i), ("R", j))
    return G


def double_star_bridge(a: int, b: int) -> nx.Graph:
    """Two stars of size a and b joined by a bridge between their centers:
    adversarial for the maximal-matching 2-approximation, whose worst-case
    tightness is realized on star-like graphs joined by bridges."""
    G = nx.Graph()
    G.add_node("cL"); G.add_node("cR")
    for i in range(a):
        G.add_edge("cL", ("La", i))
    for i in range(b):
        G.add_edge("cR", ("Rb", i))
    G.add_edge("cL", "cR")
    return G


def hierarchical_star_trap_size(levels: int, branching: int) -> int:
    """Exact node count of ``hierarchical_star_trap(levels, branching)``
    without building it: root + all internal levels + one pendant leaf per
    final-level node. Used to skip the (potentially huge) construction
    entirely when it would exceed ``--max-n``, instead of building the full
    graph first and discarding it -- for ``levels=10, branching=4`` that
    construction alone is on the order of 1.4M nodes and can dominate the
    wall-clock time of a run whose actual instances are far smaller."""
    nodes_through_last_level = sum(branching ** i for i in range(levels + 1))
    pendant_leaves = branching ** levels
    return nodes_through_last_level + pendant_leaves


def hierarchical_star_trap(levels: int, branching: int) -> nx.Graph:
    """Engineered stress construction (not a literature-verified worst case)
    for degree-based greedy tie-breaking: a tree of stars where each
    internal 'hub' has the same local degree as many leaf-adjacent hubs
    one level down, so a max-degree-first greedy repeatedly has to choose
    among many equally-attractive but structurally different hubs."""
    G = nx.Graph()
    root = (0, 0)
    G.add_node(root)
    frontier = [root]
    node_id = 1
    for level in range(1, levels + 1):
        next_frontier = []
        for parent in frontier:
            for _ in range(branching):
                child = (level, node_id)
                node_id += 1
                G.add_edge(parent, child)
                next_frontier.append(child)
        frontier = next_frontier
    # Attach a pendant leaf to every leaf of the final level so every
    # bottom-level hub has degree branching+1, matching the internal hubs.
    for leaf in frontier:
        G.add_edge(leaf, (levels + 1, node_id))
        node_id += 1
    return G


def random_regular_adversarial(n: int, d: int, seed: int) -> nx.Graph:
    """Regular sparse graph: every vertex ties for minimum degree at every
    step, which is exactly the case that used to defeat the pre-fix
    Min-to-Min heuristic (see the note in salvador.algorithm)."""
    if (n * d) % 2 != 0:
        n += 1
    return nx.random_regular_graph(d, n, seed=seed)


def barabasi_albert_adversarial(n: int, m_edges: int, seed: int) -> nx.Graph:
    """Scale-free hub graph: highly degree-heterogeneous, stresses the
    max-degree greedy and the maximal-matching heuristics differently."""
    return nx.barabasi_albert_graph(n, m_edges, seed=seed)


def sparse_gnp_adversarial(n: int, avg_degree: float, seed: int) -> nx.Graph:
    """Erdos-Renyi graph with constant expected average degree: the
    canonical large, sparse, unstructured scaling instance."""
    p = min(1.0, avg_degree / max(1, n - 1))
    return nx.gnp_random_graph(n, p, seed=seed)


def watts_strogatz_adversarial(n: int, k: int, beta: float, seed: int) -> nx.Graph:
    """Small-world graph: locally dense (adversarial for pruning), globally
    sparse (keeps m = O(n) for the scaling study)."""
    k = max(2, k - (k % 2))
    return nx.watts_strogatz_graph(n, k, beta, seed=seed)


def maximal_matching_lower_bound(G: nx.Graph) -> tuple[int, str]:
    """A provable lower bound on tau(G): |M| for any matching M.

    Uses an exact maximum matching (hence an exact tau(G) by Koenig) on
    bipartite graphs via Hopcroft-Karp; otherwise falls back to a fast
    greedy *maximal* (not maximum) matching, which is only a lower bound.
    """
    if G.number_of_edges() == 0:
        return 0, "trivial"
    if nx.is_bipartite(G):
        top = _one_side(G)
        matching = nx.bipartite.hopcroft_karp_matching(G, top_nodes=top)
        return len(matching) // 2, "bipartite-maximum-matching (exact tau)"
    matching = nx.algorithms.matching.maximal_matching(G)
    return len(matching), "greedy-maximal-matching (lower bound only)"


def evaluate_large(name: str, G: nx.Graph, group: str, *, verbose: bool = True) -> dict:
    # Relabel to plain integers first: some generators below mix node-label
    # types (strings, tuples), and salvador.algorithm.covering_via_reduction_max_degree_1
    # breaks ties with a direct '<' comparison between node labels, which
    # raises TypeError across heterogeneous, non-comparable label types.
    # Integer relabeling sidesteps that without touching library code, and
    # matches what Part A's evaluate() already does.
    G = nx.convert_node_labels_to_integers(G, ordering="default")
    n0, m0 = G.number_of_nodes(), G.number_of_edges()
    if verbose:
        print(f"  [part B] {name}: n={n0:,} m={m0:,} ...", flush=True)
    t0 = time.perf_counter()
    cover = set(find_vertex_cover(G, epsilon=DEFAULT_EPSILON))
    elapsed = time.perf_counter() - t0
    valid = all(u in cover or v in cover for u, v in G.edges())
    lb, lb_method = maximal_matching_lower_bound(G)
    ratio_upper_bound = None if lb == 0 else len(cover) / lb
    cover_size = len(cover)
    if verbose:
        print(f"  [part B] {name}: done in {elapsed:.2f}s, cover={cover_size:,}, valid={valid}", flush=True)
    result = {
        "group": group, "name": name,
        "n": n0, "m": m0,
        "cover": cover_size,
        "lower_bound": lb, "lower_bound_method": lb_method,
        "ratio_upper_bound": ratio_upper_bound,
        "exact": lb_method.startswith("bipartite"),
        "valid": bool(valid),
        "elapsed_seconds": elapsed,
        "us_per_np1m": None if (n0 + m0) == 0 else elapsed * 1e6 / (n0 + m0),
    }
    # G (and the ensemble's internal candidate covers, gadgets, etc.) can be
    # tens to hundreds of MB for the largest instances; drop the references
    # and collect explicitly so peak memory doesn't ratchet up across a long
    # sequential run through many large instances instead of being freed
    # between them.
    del G, cover
    gc.collect()
    return result


def run_part_b_families(max_n: int) -> list[dict]:
    rows = []
    seed = SEED

    # Dense adversarial families (kept moderate: m grows quadratically).
    for half in (200, 400, 800):
        if half * half > max_n * 40:
            continue
        rows.append(evaluate_large(f"crown_{half}", crown_graph(half), "Crown graph K_{h,h} minus perfect matching (dense, near-regular)"))
        rows.append(evaluate_large(f"complete_bipartite_{half}", complete_bipartite_adversarial(half, half), "Complete bipartite (dense)"))

    # Bridge / star adversarial family (should stay ratio <= 2 for c1 alone,
    # and the ensemble should do at least as well).
    for size in (1000, 10000, 100000):
        if size > max_n:
            continue
        rows.append(evaluate_large(f"double_star_{size}", double_star_bridge(size, size), "Double star + bridge (matching-heuristic stress)"))

    # Hierarchical greedy-tie-break trap. Size is checked analytically
    # (hierarchical_star_trap_size) BEFORE building the graph, so an
    # oversized configuration (levels=10, branching=4 is ~1.4M nodes) is
    # skipped without ever allocating it.
    for levels, branching in ((6, 4), (8, 4), (10, 4)):
        if hierarchical_star_trap_size(levels, branching) > max_n:
            continue
        G = hierarchical_star_trap(levels, branching)
        rows.append(evaluate_large(f"hier_trap_L{levels}_B{branching}", G, "Hierarchical star trap (degree-greedy tie-break stress)"))

    # Large sparse regular graphs (the family that exposed the O(n^2)
    # Min-to-Min bug before the linear-time fix).
    for n in (5000, 50000, 200000):
        if n > max_n:
            continue
        for d in (3, 6, 12):
            rows.append(evaluate_large(f"regular_n{n}_d{d}", random_regular_adversarial(n, d, seed), f"Random {d}-regular (bounded-degree adversarial)"))
            seed += 1

    # Scale-free hubs.
    for n in (5000, 50000, 200000):
        if n > max_n:
            continue
        rows.append(evaluate_large(f"barabasi_albert_n{n}", barabasi_albert_adversarial(n, 3, seed), "Barabasi-Albert scale-free hub graph"))
        seed += 1

    # Large sparse unstructured Erdos-Renyi.
    for n in (5000, 50000, 200000):
        if n > max_n:
            continue
        rows.append(evaluate_large(f"gnp_n{n}", sparse_gnp_adversarial(n, 4.0, seed), "Sparse Erdos-Renyi, avg degree 4"))
        seed += 1

    # Small-world.
    for n in (5000, 50000):
        if n > max_n:
            continue
        rows.append(evaluate_large(f"ws_n{n}", watts_strogatz_adversarial(n, 6, 0.1, seed), "Watts-Strogatz small-world"))
        seed += 1

    return rows


def run_part_b_scaling(max_n: int) -> list[dict]:
    """Doubling-size scaling study for the O(n + m) linear-time claim."""
    rows = []
    size = 1000
    seed = SEED + 10_000
    while size <= max_n:
        # Family 1: bounded-degree regular graph (m = O(n)).
        G1 = random_regular_adversarial(size, 4, seed)
        r1 = evaluate_large(f"scaling_regular4_n{size}", G1, "Scaling: 4-regular")
        r1["family"] = "regular4"
        rows.append(r1)

        # Family 2: scale-free hub graph (m = O(n), but degree-heterogeneous).
        G2 = barabasi_albert_adversarial(size, 3, seed + 1)
        r2 = evaluate_large(f"scaling_ba3_n{size}", G2, "Scaling: Barabasi-Albert m=3")
        r2["family"] = "barabasi_albert3"
        rows.append(r2)

        seed += 2
        size *= 2

    return rows


def _linreg(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares y = slope*x + intercept, plus R^2. No numpy
    dependency beyond what the package already requires elsewhere; pure
    Python is fine here since the number of scaling points is small."""
    n = len(xs)
    if n < 2:
        return {"slope": None, "intercept": None, "r2": None, "n_points": n}
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return {"slope": None, "intercept": None, "r2": None, "n_points": n}
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"slope": slope, "intercept": intercept, "r2": r2, "n_points": n}


def summarise_scaling(rows: list[dict]) -> dict:
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    out = {}
    for fam, frows in by_family.items():
        frows = sorted(frows, key=lambda r: r["n"] + r["m"])
        xs = [float(r["n"] + r["m"]) for r in frows]
        ys = [float(r["elapsed_seconds"]) for r in frows]
        fit = _linreg(xs, ys)
        out[fam] = {
            "fit_time_vs_(n+m)": fit,
            "points": [
                {"n": r["n"], "m": r["m"], "n_plus_m": r["n"] + r["m"],
                 "elapsed_seconds": r["elapsed_seconds"],
                 "us_per_np1m": r["us_per_np1m"]}
                for r in frows
            ],
        }
    return out


def run_part_b(max_n: int) -> dict:
    started = time.time()
    print(f"[part B] starting family sweep (--max-n={max_n:,}) ...", flush=True)
    family_rows = run_part_b_families(max_n)
    print(f"[part B] family sweep done in {time.time() - started:.1f}s; starting scaling study ...", flush=True)
    scaling_started = time.time()
    scaling_rows = run_part_b_scaling(max_n)
    print(f"[part B] scaling study done in {time.time() - scaling_started:.1f}s", flush=True)

    by_group = {}
    for g in sorted({r["group"] for r in family_rows}):
        grows = [r for r in family_rows if r["group"] == g]
        ratios = [r["ratio_upper_bound"] for r in grows if r["ratio_upper_bound"] is not None]
        by_group[g] = {
            "instances": len(grows),
            "max_ratio_upper_bound": max(ratios) if ratios else None,
            "mean_ratio_upper_bound": statistics.fmean(ratios) if ratios else None,
            "all_valid": all(r["valid"] for r in grows),
            "max_n": max(r["n"] for r in grows),
            "max_elapsed_seconds": max(r["elapsed_seconds"] for r in grows),
        }

    all_ratios = [r["ratio_upper_bound"] for r in family_rows if r["ratio_upper_bound"] is not None]
    scaling_summary = summarise_scaling(scaling_rows)

    return {
        "experiment": f"car/ Part B: large adversarial graphs (epsilon={DEFAULT_EPSILON})",
        "salvador_version": __version__,
        "default_epsilon": DEFAULT_EPSILON,
        "max_n_requested": max_n,
        "note": (
            "ratio_upper_bound = |cover| / (a provable lower bound on tau(G)). "
            "It is EXACT (equals |cover|/tau(G)) when lower_bound_method starts "
            "with 'bipartite'; otherwise it is a conservative upper bound on "
            "the true ratio, because a greedy maximal matching can under-count "
            "tau(G) by up to a factor of 2."
        ),
        "family_rows": family_rows,
        "summary_by_group": by_group,
        "overall_max_ratio_upper_bound": max(all_ratios) if all_ratios else None,
        "overall_all_valid": all(r["valid"] for r in family_rows) if family_rows else True,
        "scaling_rows": scaling_rows,
        "scaling_summary": scaling_summary,
        "elapsed_seconds": time.time() - started,
    }


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller/faster Part A sweep")
    ap.add_argument("--skip-large", action="store_true", help="run Part A only")
    ap.add_argument("--skip-small", action="store_true", help="run Part B only")
    ap.add_argument("--max-n", type=int, default=200_000, help="largest instance size for Part B (default 200000)")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent
    result = {
        "salvador_version": __version__,
        "default_epsilon": DEFAULT_EPSILON,
        "seed": SEED,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
    }

    if not args.skip_small:
        part_a = run_part_a(big=not args.quick)
        result["part_a"] = part_a
        by_group = part_a["summary_by_group"]
        with (out / "car_summary.csv").open("w", encoding="utf-8") as fh:
            fh.write("group,instances,mean_ratio,max_ratio,count_above_7_4,all_valid\n")
            for g, s in by_group.items():
                fh.write(f"{g},{s['instances']},{s['mean_ratio']:.6f},{s['max_ratio']:.6f},"
                         f"{s['count_above_7_4']},{s['all_valid']}\n")
            overall = part_a["overall_summary"]
            fh.write(f"OVERALL,{overall['instances']},{overall['mean_ratio']:.6f},"
                     f"{overall['max_ratio']:.6f},{overall['count_above_7_4']},{overall['all_valid']}\n")
        print(json.dumps({
            "part": "A (small, exact ratio)",
            "default_epsilon": DEFAULT_EPSILON,
            "instances": overall["instances"],
            "max_ratio": overall["max_ratio"],
            "all_valid": overall["all_valid"],
            "worst_instance": overall["worst_instance"]["name"],
        }, indent=2))

    if not args.skip_large:
        part_b = run_part_b(args.max_n)
        result["part_b"] = part_b
        with (out / "car_large_summary.csv").open("w", encoding="utf-8") as fh:
            fh.write("group,instances,max_ratio_upper_bound,mean_ratio_upper_bound,all_valid,max_n,max_elapsed_seconds\n")
            for g, s in part_b["summary_by_group"].items():
                fh.write(f"{g},{s['instances']},{s['max_ratio_upper_bound']},{s['mean_ratio_upper_bound']},"
                         f"{s['all_valid']},{s['max_n']},{s['max_elapsed_seconds']:.6f}\n")
        with (out / "car_scaling.csv").open("w", encoding="utf-8") as fh:
            fh.write("family,n,m,n_plus_m,elapsed_seconds,us_per_np1m\n")
            for fam, s in part_b["scaling_summary"].items():
                for p in s["points"]:
                    fh.write(f"{fam},{p['n']},{p['m']},{p['n_plus_m']},{p['elapsed_seconds']:.6f},{p['us_per_np1m']:.4f}\n")
            fh.write("\nfamily,slope_seconds_per_np1m,intercept_seconds,r2,n_points\n")
            for fam, s in part_b["scaling_summary"].items():
                fit = s["fit_time_vs_(n+m)"]
                fh.write(f"{fam},{fit['slope']},{fit['intercept']},{fit['r2']},{fit['n_points']}\n")
        print(json.dumps({
            "part": "B (large adversarial graphs)",
            "default_epsilon": DEFAULT_EPSILON,
            "max_n_requested": args.max_n,
            "instances": len(part_b["family_rows"]),
            "overall_max_ratio_upper_bound": part_b["overall_max_ratio_upper_bound"],
            "overall_all_valid": part_b["overall_all_valid"],
            "scaling_fits": {fam: s["fit_time_vs_(n+m)"] for fam, s in part_b["scaling_summary"].items()},
        }, indent=2))

    (out / "car_experiment.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
