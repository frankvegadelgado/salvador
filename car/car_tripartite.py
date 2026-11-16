"""car/ tripartite study: how far is each strategy from the optimum on tripartite graphs?

A tripartite graph is an undirected graph whose vertex set splits into three
independent sets (a 3-colourable graph). Unlike the bipartite case, the minimum
vertex cover tau(G) is NP-hard here (planar graphs of maximum degree 3 are
tripartite), so no polynomial-time reference exists. The reference is therefore
computed *for reference only* (it is never given to Salvador) by an integer
linear program (HiGHS through ``scipy.optimize.milp``) with a time limit
(``--ilp-time``), together with certified lower bounds (NetworkX minimum maximal
matching, Karp-Sipser matching, vertex-disjoint clique packing, spectral bound)
from ``car_common.best_lower_bound``. Per instance:

* ``tau_lb``   -- best certified lower bound on tau(G) (ILP dual bound or the bounds above);
* ``tau_best`` -- smallest vertex cover known (ILP incumbent or any strategy);
* ``proven``   -- ``tau_lb == tau_best``, i.e. tau(G) is known exactly;
* ``ratio_lb`` = |c17| / tau_best and ``ratio_ub`` = |c17| / tau_lb: the true
  ratio |c17| / tau(G) lies in [ratio_lb, ratio_ub]; the two coincide when proven.

The cover evaluated is ``c17``, the final cover of
``salvador.algorithm.final_candidates`` (the same cover ``find_vertex_cover``
returns). Per instance the report also gives every strategy size c1..c17, which
strategies reach tau_best, c10's decoded cover (``c10_raw``), whether c9 / c10 /
c12 were exact and their core sizes, whether the graph happens to be bipartite,
the ILP status and the times. Per family it gives how many instances c17
matches tau_best (and how many of those are proven optimal), the worst ratio
interval, the total excess, and how many instances each strategy solves.

Every instance carries its own explicit 3-colouring (a node attribute used only
to assert that the graph really is tripartite; it is removed before Salvador
sees the graph) and the labels are shuffled.

Families (all tripartite; chosen to be hard to solve exactly and to approximate):

* random tripartite graphs with average degree 3, 4.5 and 7 (the range where
  independent sets are hardest to find), balanced and unbalanced parts;
* random d-regular-like tripartite graphs (k random perfect matchings between
  every pair of parts, degree about 2k);
* hidden triangle factor (a_i, b_i, c_i) plus random cross edges, where every
  cover needs two vertices per triangle and greedy rules are misled;
* Latin-square triangle systems: edge-disjoint triangles (a_i, b_j, c_{i+j}),
  t triangles per vertex, so every edge lies in exactly one triangle;
* sparse blow-ups of odd cycles C_k (k = 5, 11, 25): consecutive blobs joined
  by random perfect matchings; the all-1/2 solution of the LP relaxation is
  optimal for the relaxation while the integral optimum is larger;
* planted independent set inside a random tripartite graph;
* power-law tripartite graphs (Chung-Lu weights, cross-part edges only);
* triangular lattices (3-coloured by (i + j) mod 3), complete and with random
  edge deletions;
* complete tripartite K_{a,b,c} and K_{a,a,a} minus a triangle factor (dense).

Run from the repository root:

    python car/car_tripartite.py                  # default sizes (up to --max-n)
    python car/car_tripartite.py --quick          # small sizes only
    python car/car_tripartite.py --max-n 1000
    python car/car_tripartite.py --ilp-time 120   # more time for the reference ILP
    python car/car_tripartite.py --timeout 600    # skip instances taking more than 600 s

Every instance is built and evaluated in a worker process with a time limit
(``--timeout``, default 300 s, 0 = no limit): an instance whose construction or
evaluation takes longer is skipped and listed in car_tripartite.json
(``skipped_instances``). Each instance has its own random generator, derived
from SEED and its name, so the worker can rebuild it independently. The script
uses the repository's own salvador package when run from a clone, and the
installed package otherwise (see car_common.py).

Outputs (in car/): car_tripartite.json, car_tripartite.csv (one row per
instance) and car_tripartite_summary.csv (one row per family).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time

import networkx as nx

import car_common as cc
from salvador.algorithm import CANDIDATE_NAMES, DEFAULT_EDS_BUDGET, DEFAULT_EDS_MAX_WIDTH, FINAL_NAME, final_candidates

SEED = 20261114
NAMES = list(CANDIDATE_NAMES)  # c1..c17


# ----------------------------------------------------------------------------
# Generators (every graph returned carries a proper 3-colouring in node attribute "part")
# ----------------------------------------------------------------------------
def parts_graph(sizes: list[int]) -> tuple[nx.Graph, list[list[int]]]:
    """Empty graph on sum(sizes) vertices, consecutive blocks being the parts 0, 1, 2, ..."""
    G = nx.Graph()
    groups, start = [], 0
    for p, s in enumerate(sizes):
        ids = list(range(start, start + s))
        G.add_nodes_from(ids, part=p)
        groups.append(ids)
        start += s
    return G, groups


def add_random_cross_edges(G: nx.Graph, groups: list[list[int]], m: int, rng: random.Random) -> None:
    """Add up to m new random edges between different parts (uniform over the cross pairs)."""
    pairs = [(a, b) for a in range(len(groups)) for b in range(a + 1, len(groups))]
    weights = [len(groups[a]) * len(groups[b]) for a, b in pairs]
    m = min(m, sum(weights) // 2)
    target = G.number_of_edges() + m
    while G.number_of_edges() < target:
        a, b = rng.choices(pairs, weights)[0]
        G.add_edge(rng.choice(groups[a]), rng.choice(groups[b]))


def three_sizes(n: int, shares: tuple[float, float, float]) -> list[int]:
    a, b = int(n * shares[0]), int(n * shares[1])
    return [a, b, n - a - b]


def random_tripartite(sizes: list[int], avg_deg: float, rng: random.Random) -> nx.Graph:
    G, groups = parts_graph(sizes)
    add_random_cross_edges(G, groups, int(sum(sizes) * avg_deg / 2), rng)
    return G


def regular_tripartite(third: int, k: int, rng: random.Random) -> nx.Graph:
    """k random perfect matchings between every pair of three parts of size ``third``."""
    G, groups = parts_graph([third] * 3)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        for _ in range(k):
            perm = list(range(third))
            rng.shuffle(perm)
            G.add_edges_from((groups[a][i], groups[b][perm[i]]) for i in range(third))
    return G


def triangle_factor_noise(third: int, extra_deg: float, rng: random.Random) -> nx.Graph:
    """Triangles (a_i, b_i, c_i) plus random cross edges (about extra_deg per vertex)."""
    G, groups = parts_graph([third] * 3)
    for i in range(third):
        a, b, c = groups[0][i], groups[1][i], groups[2][i]
        G.add_edges_from(((a, b), (b, c), (a, c)))
    add_random_cross_edges(G, groups, int(3 * third * extra_deg / 2), rng)
    return G


def latin_triangles(a: int, t: int, rng: random.Random) -> nx.Graph:
    """Edge-disjoint triangles (A_i, B_j, C_{i+j}) for the cells j = i + s, s in a random t-set.

    ``a`` is odd, so i -> 2i + s is a bijection and every vertex lies in exactly t triangles."""
    a -= 1 - a % 2
    G, groups = parts_graph([a] * 3)
    for s in rng.sample(range(a), min(t, a)):
        for i in range(a):
            j = (i + s) % a
            k = (i + j) % a
            A, B, C = groups[0][i], groups[1][j], groups[2][k]
            G.add_edges_from(((A, B), (B, C), (A, C)))
    return G


def odd_cycle_blowup(k: int, t: int, r: int, rng: random.Random) -> nx.Graph:
    """Blobs of t independent vertices around an odd cycle C_k; consecutive blobs joined by r random perfect matchings."""
    G = nx.Graph()
    groups = []
    for i in range(k):
        colour = i % 2 if i < k - 1 else 2
        ids = list(range(i * t, (i + 1) * t))
        G.add_nodes_from(ids, part=colour)
        groups.append(ids)
    for i in range(k):
        nxt = groups[(i + 1) % k]
        for _ in range(r):
            perm = list(range(t))
            rng.shuffle(perm)
            G.add_edges_from((groups[i][x], nxt[perm[x]]) for x in range(t))
    return G


def planted_independent(n: int, avg_deg: float, frac: float, rng: random.Random) -> nx.Graph:
    """Random tripartite graph with all edges inside a random vertex set I (|I| = frac * n) removed."""
    G = random_tripartite(three_sizes(n, (1 / 3, 1 / 3, 1 / 3)), avg_deg / (1 - frac * frac), rng)
    planted = set(rng.sample(range(n), int(frac * n)))
    G.remove_edges_from([(u, v) for u, v in G.edges() if u in planted and v in planted])
    return G


def power_law_tripartite(n: int, exponent: float, avg_deg: float, rng: random.Random) -> nx.Graph:
    """Chung-Lu style: endpoint probability proportional to a Pareto weight, cross-part pairs only."""
    G = nx.Graph()
    for v in range(n):
        G.add_node(v, part=rng.randrange(3))
    cum = list(itertools.accumulate(min(rng.paretovariate(exponent - 1), n ** 0.5) for _ in range(n)))
    target, attempts = int(n * avg_deg / 2), 0
    while G.number_of_edges() < target and attempts < 60 * target:
        attempts += 1
        u, v = rng.choices(range(n), cum_weights=cum, k=2)
        if G.nodes[u]["part"] != G.nodes[v]["part"]:
            G.add_edge(u, v)
    return G


def triangular_lattice(rows: int, cols: int, keep: float, rng: random.Random) -> nx.Graph:
    """Triangular lattice with edges (i,j)-(i,j+1), (i+1,j), (i+1,j+1); colour (i + j) mod 3."""
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node(i * cols + j, part=(i + j) % 3)
    for i in range(rows):
        for j in range(cols):
            for di, dj in ((0, 1), (1, 0), (1, 1)):
                if i + di < rows and j + dj < cols and rng.random() < keep:
                    G.add_edge(i * cols + j, (i + di) * cols + j + dj)
    return G


def complete_tripartite(a: int, b: int, c: int) -> nx.Graph:
    G, groups = parts_graph([a, b, c])
    for x, y in ((0, 1), (1, 2), (0, 2)):
        G.add_edges_from((u, v) for u in groups[x] for v in groups[y])
    return G


def complete_minus_triangles(a: int) -> nx.Graph:
    """K_{a,a,a} minus the triangles (A_i, B_i, C_i): dense, every part is a maximum independent set."""
    G = complete_tripartite(a, a, a)
    for i in range(a):
        G.remove_edges_from(((i, a + i), (a + i, 2 * a + i), (i, 2 * a + i)))
    return G


def instances(max_n: int, quick: bool, rng: random.Random):
    sizes = [n for n in ((60, 150) if quick else (60, 150, 400, 1000, 3000)) if n <= max_n]
    for n in sizes:
        reps = 3 if n <= 400 else 1
        for r in range(reps):
            for deg in (3, 4.5, 7):
                yield f"gnp_tri_bal_n{n}_d{deg}_r{r}", "random tripartite (balanced)", \
                    lambda n=n, deg=deg: random_tripartite(three_sizes(n, (1 / 3, 1 / 3, 1 / 3)), deg, rng)
                yield f"gnp_tri_unbal_n{n}_d{deg}_r{r}", "random tripartite (1:2:3)", \
                    lambda n=n, deg=deg: random_tripartite(three_sizes(n, (1 / 6, 1 / 3, 1 / 2)), deg, rng)
            for k in (2, 3, 4):
                yield f"reg_tri_n{n}_k{k}_r{r}", "random regular-like tripartite (k matchings per pair)", \
                    lambda n=n, k=k: regular_tripartite(n // 3, k, rng)
            for ed in (1.0, 3.0):
                yield f"hidden_triangles_n{n}_e{ed}_r{r}", "hidden triangle factor + random edges", \
                    lambda n=n, ed=ed: triangle_factor_noise(n // 3, ed, rng)
            for t in (2, 3, 4):
                yield f"latin_triangles_n{n}_t{t}_r{r}", "Latin-square edge-disjoint triangles", \
                    lambda n=n, t=t: latin_triangles(n // 3, t, rng)
            for k in (5, 11, 25):
                if n // k >= 4:
                    yield f"oddcycle_blowup_n{n}_k{k}_r{r}", "sparse odd-cycle blow-up", \
                        lambda n=n, k=k: odd_cycle_blowup(k, n // k, 3, rng)
            yield f"planted_is_n{n}_r{r}", "planted independent set", lambda n=n: planted_independent(n, 6.0, 0.4, rng)
            yield f"powerlaw_tri_n{n}_r{r}", "power-law tripartite", lambda n=n: power_law_tripartite(n, 2.5, 4.0, rng)
            side = int(n ** 0.5)
            yield f"tri_lattice_thinned_{side}x{side}_r{r}", "triangular lattice, 80% edges kept", \
                lambda side=side: triangular_lattice(side, side, 0.8, rng)
        side = int(n ** 0.5)
        yield f"tri_lattice_{side}x{side}", "triangular lattice", lambda side=side: triangular_lattice(side, side, 1.0, rng)
        if n <= 400:
            yield f"K_{n // 6}_{n // 3}_{n - n // 6 - n // 3}", "complete tripartite", \
                lambda n=n: complete_tripartite(n // 6, n // 3, n - n // 6 - n // 3)
            yield f"K3_minus_triangles_{n // 3}", "K_{a,a,a} minus triangle factor", \
                lambda n=n: complete_minus_triangles(n // 3)


# ----------------------------------------------------------------------------
# Reference solution (never given to Salvador)
# ----------------------------------------------------------------------------
def shuffle_labels(G: nx.Graph, rng: random.Random) -> nx.Graph:
    nodes = list(G.nodes())
    perm = list(range(len(nodes)))
    rng.shuffle(perm)
    return nx.relabel_nodes(G, {v: perm[i] for i, v in enumerate(nodes)})


def ilp_reference(G: nx.Graph, time_limit: float) -> dict:
    """Minimum vertex cover by integer programming (HiGHS): bounds, status and the incumbent cover size."""
    t0 = time.perf_counter()
    out = {"lb": None, "ub": None, "status": "scipy-unavailable", "seconds": 0.0}
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ModuleNotFoundError:  # pragma: no cover
        return out
    edges = list(G.edges())
    if not edges:
        return {"lb": 0, "ub": 0, "status": "optimal", "seconds": 0.0}
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    m, n = len(edges), len(nodes)
    cols = np.fromiter((idx[x] for e in edges for x in e), dtype=np.int64, count=2 * m)
    A = sp.csr_matrix((np.ones(2 * m), (np.repeat(np.arange(m), 2), cols)), shape=(m, n))
    res = milp(c=np.ones(n), constraints=LinearConstraint(A, lb=np.ones(m), ub=np.inf),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True})
    if res.x is not None:
        cover = {nodes[i] for i in range(n) if res.x[i] > 0.5}
        if all(u in cover or v in cover for u, v in edges):
            out["ub"] = len(cover)
    dual = getattr(res, "mip_dual_bound", None)
    if res.status == 0 and out["ub"] is not None:
        out["lb"] = out["ub"]
    elif dual is not None and math.isfinite(dual):
        out["lb"] = math.ceil(dual - 1e-6)
    out["status"] = {0: "optimal", 1: "time-limit"}.get(res.status, f"status-{res.status}")
    out["seconds"] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_instance(name: str, group: str, G: nx.Graph, rng: random.Random, args) -> dict:
    G = shuffle_labels(nx.convert_node_labels_to_integers(G), rng)
    G.remove_edges_from(nx.selfloop_edges(G))
    part = nx.get_node_attributes(G, "part")
    assert len(part) == G.number_of_nodes() and set(part.values()) <= {0, 1, 2}, name
    assert all(part[u] != part[v] for u, v in G.edges()), f"{name}: not tripartite"
    H = nx.Graph()  # the graph Salvador sees: no colouring, no attributes
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    G = H

    ref = ilp_reference(G, args.ilp_time)
    lb_bound, lb_method, _ = cc.best_lower_bound(G)
    info: dict = {}
    t0 = time.perf_counter()
    cands = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info,
                             eds_max_width=args.eds_max_width, eds_budget=args.eds_budget)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in NAMES}

    tau_lb = max(lb_bound, ref["lb"] or 0)
    for k in NAMES:
        C = cands[k]
        if not all(u in C or v in C for u, v in G.edges()):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
        if len(C) < tau_lb:
            raise AssertionError(f"{name}: strategy {k} is below a certified lower bound -- impossible")
    tau_best = min([s for s in sizes.values()] + ([ref["ub"]] if ref["ub"] is not None else []))
    proven = tau_lb >= tau_best
    final = sizes[FINAL_NAME]
    bound_method = "ilp-dual-bound" if (ref["lb"] or 0) >= lb_bound else lb_method
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "bipartite": nx.is_bipartite(G),
        "tau_lb": tau_lb, "tau_lb_method": bound_method, "tau_best": tau_best, "proven": proven,
        "ilp_status": ref["status"], "ilp_seconds": ref["seconds"],
        "sizes": sizes, "final": final,
        "ratio_lb": final / tau_best if tau_best else 1.0,
        "ratio_ub": final / tau_lb if tau_lb else 1.0,
        "excess_lb": final - tau_best, "excess_ub": final - tau_lb,
        "matches_best": final == tau_best,
        "optimal": proven and final == tau_best,
        "optimal_strategies": [k for k in NAMES if sizes[k] == tau_best],
        "c10_exact": bool(info.get("c10_exact", True)),
        "c10_raw": info.get("c10_raw_size", sizes["c10"]),
        "c10_core_size": info.get("c10_core_size", 0),
        "c10_width": info.get("c10_width", 0),
        "c12_exact": bool(info.get("c12_exact", True)),
        "c12_core_size": info.get("c12_core_size", 0),
        "c9_exact": bool(info.get("c9_exact", False)),
        "elapsed_seconds": elapsed,
    }
    print(f"  [tripartite] {group[:34]:34} {name:36} n={row['n']:>6,} m={row['m']:>8,}"
          f" tau in [{tau_lb:>6,}, {tau_best:>6,}]{' proven' if proven else ''}"
          f" c17={final:>6,} ratio in [{row['ratio_lb']:.4f}, {row['ratio_ub']:.4f}]"
          f" c10={'exact' if row['c10_exact'] else 'core=' + str(row['c10_core_size'])}"
          f" c12={'exact' if row['c12_exact'] else 'core=' + str(row['c12_core_size'])}"
          f" ilp={ref['status']} {ref['seconds']:.1f}s  {elapsed:6.2f}s", flush=True)
    return row


def _instance_job(name: str, args) -> dict:
    """Build instance ``name`` (with its own generator) and evaluate it; runs in the worker process."""
    rng = random.Random(SEED)
    for nm, group, build in instances(args.max_n, args.quick, rng):
        if nm == name:
            rng.seed(f"{SEED}:{name}")  # the builders draw from this generator
            return run_instance(name, group, build(), rng, args)
    raise KeyError(name)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(grows)
        out[g] = {
            "instances": k,
            "proven": sum(r["proven"] for r in grows),
            "matches_best": sum(r["matches_best"] for r in grows),
            "optimal": sum(r["optimal"] for r in grows),
            "matches_best_rate": sum(r["matches_best"] for r in grows) / k if k else 0.0,
            "max_ratio_lb": max((r["ratio_lb"] for r in grows), default=1.0),
            "max_ratio_ub": max((r["ratio_ub"] for r in grows), default=1.0),
            "mean_ratio_lb": sum(r["ratio_lb"] for r in grows) / k if k else 1.0,
            "mean_ratio_ub": sum(r["ratio_ub"] for r in grows) / k if k else 1.0,
            "total_excess_lb": sum(r["excess_lb"] for r in grows),
            "total_excess_ub": sum(r["excess_ub"] for r in grows),
            "total_tau_best": sum(r["tau_best"] for r in grows),
            "bipartite_instances": sum(r["bipartite"] for r in grows),
            "c10_exact": sum(r["c10_exact"] for r in grows),
            "c10_raw_matches_best": sum(r["c10_raw"] == r["tau_best"] for r in grows),
            "c12_exact": sum(r["c12_exact"] for r in grows),
            "c9_exact": sum(r["c9_exact"] for r in grows),
            "best_by_strategy": {s: sum(s in r["optimal_strategies"] for r in grows) for s in NAMES},
            "ilp_seconds": sum(r["ilp_seconds"] for r in grows),
            "seconds": sum(r["elapsed_seconds"] for r in grows),
        }
    return out


def print_table(summary: dict) -> None:
    header = (f"{'family':52} {'inst':>5} {'prov':>5} {'=best':>5} {'opt':>5} {'ratio lb':>9} {'ratio ub':>9}"
              f" {'exc lb':>7} {'exc ub':>7} {'c10ex':>5} {'c12ex':>5}  best by strategy (" + " ".join(NAMES) + ")")
    print("\nTripartite graphs: final cover c17 against tau (ILP + certified bounds); true ratio in [ratio lb, ratio ub]")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        by = " ".join(f"{s['best_by_strategy'][k]:>4}" for k in NAMES)
        print(f"{g[:52]:52} {s['instances']:>5} {s['proven']:>5} {s['matches_best']:>5} {s['optimal']:>5}"
              f" {s['max_ratio_lb']:9.5f} {s['max_ratio_ub']:9.5f} {s['total_excess_lb']:>7} {s['total_excess_ub']:>7}"
              f" {s['c10_exact']:>5} {s['c12_exact']:>5}  {by}")
    print("prov = tau known exactly (lower bound = best cover known); =best = c17 equals the best cover known; "
          "opt = proven and equal; exc lb/ub = |c17| - best known / |c17| - certified lower bound")
    print("\nLegend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="sizes 60 and 150 only")
    ap.add_argument("--max-n", type=int, default=3000, help="largest instance size (default 3000)")
    ap.add_argument("--ilp-time", type=float, default=30.0,
                    help="time limit (seconds) of the reference ILP per instance (default 30)")
    ap.add_argument("--eds-max-width", type=int, default=DEFAULT_EDS_MAX_WIDTH,
                    help=f"c10: largest elimination bag minus one (default {DEFAULT_EDS_MAX_WIDTH})")
    ap.add_argument("--eds-budget", type=float, default=DEFAULT_EDS_BUDGET,
                    help=f"c10: DP table budget in units of (n + m) (default {DEFAULT_EDS_BUDGET})")
    cc.add_timeout_argument(ap, default=300.0)
    args = ap.parse_args()

    started = time.time()
    rows = []
    skipped = []
    names = [(name, group) for name, group, _ in instances(args.max_n, args.quick, random.Random(SEED))]
    with cc.InstanceRunner(args.timeout) as runner:
        for name, group in names:
            row = runner.run(_instance_job, name, args)
            if row is None:
                print(f"  [tripartite] {group[:34]:34} {name:36} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    out = cc.OUT_DIR
    with (out / "car_tripartite.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,n,m,bipartite,tau_lb,tau_lb_method,tau_best,proven," + ",".join(NAMES)
                 + ",final,ratio_lb,ratio_ub,excess_lb,excess_ub,matches_best,optimal,optimal_strategies,"
                   "c10_exact,c10_raw,c10_core_size,c12_exact,c12_core_size,c9_exact,ilp_status,ilp_seconds,elapsed_seconds\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},{r['bipartite']},{r['tau_lb']},{r['tau_lb_method']},"
                     f"{r['tau_best']},{r['proven']},"
                     + ",".join(str(r["sizes"][k]) for k in NAMES)
                     + f",{r['final']},{r['ratio_lb']:.6f},{r['ratio_ub']:.6f},{r['excess_lb']},{r['excess_ub']},"
                     + f"{r['matches_best']},{r['optimal']},{'+'.join(r['optimal_strategies'])},"
                     + f"{r['c10_exact']},{r['c10_raw']},{r['c10_core_size']},{r['c12_exact']},{r['c12_core_size']},"
                     + f"{r['c9_exact']},{r['ilp_status']},{r['ilp_seconds']:.3f},{r['elapsed_seconds']:.3f}\n")
    with (out / "car_tripartite_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,proven,matches_best,optimal,matches_best_rate,max_ratio_lb,max_ratio_ub,"
                 "mean_ratio_lb,mean_ratio_ub,total_excess_lb,total_excess_ub,total_tau_best,bipartite_instances,"
                 "c10_exact,c10_raw_matches_best,c12_exact,c9_exact,"
                 + ",".join(f"best_{k}" for k in NAMES) + ",ilp_seconds,seconds\n")
        for g, s in summary.items():
            fh.write(f"\"{g}\",{s['instances']},{s['proven']},{s['matches_best']},{s['optimal']},"
                     f"{s['matches_best_rate']:.4f},{s['max_ratio_lb']:.6f},{s['max_ratio_ub']:.6f},"
                     f"{s['mean_ratio_lb']:.6f},{s['mean_ratio_ub']:.6f},{s['total_excess_lb']},{s['total_excess_ub']},"
                     f"{s['total_tau_best']},{s['bipartite_instances']},{s['c10_exact']},{s['c10_raw_matches_best']},"
                     f"{s['c12_exact']},{s['c9_exact']},"
                     + ",".join(str(s["best_by_strategy"][k]) for k in NAMES)
                     + f",{s['ilp_seconds']:.2f},{s['seconds']:.2f}\n")
    (out / "car_tripartite.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": SEED,
         "candidates": CANDIDATE_NAMES, "eds_max_width": args.eds_max_width, "eds_budget": args.eds_budget,
         "ilp_time": args.ilp_time, "timeout": args.timeout, "skipped_instances": skipped,
         "summary": summary, "rows": rows, "elapsed_seconds": time.time() - started},
        indent=2, default=str), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
