from __future__ import annotations

"""car/ -- default-call (epsilon=0.1) approximation-ratio experiment for Salvador.

This suite tests the public default call ``find_vertex_cover(G, epsilon=0.1)``
against exact optima.  It does NOT use MILP: optima are computed by maximum
matching and Koenig's theorem on bipartite instances, and by a deterministic
branch-and-bound maximum-independent-set solver otherwise
(``tau(G) = |V| - alpha(G)``).

The headline question is whether the default call stays at or below the
conservative threshold ``7/4 = 2 - 1/4``.  The known worst case is an explicit
seven-vertex bipartite obstruction on which the default call returns a cover of
size 5 against optimum 3 (ratio 5/3); the same obstruction is solved optimally at
``epsilon in {0.25, 0.5}``, so the 5/3 behaviour is specific to the default
layering rather than intrinsic to the gadget.

Run from the repository root with:

    python car/car_experiment.py            # full feasible suite
    python car/car_experiment.py --quick    # smaller, faster sweep

Outputs:
    car/car_experiment.json
    car/car_summary.csv
"""

import argparse
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
DEFAULT_EPSILON = 0.1
SUB2_TARGET = 7.0 / 4.0

# The explicit seven-vertex bipartite obstruction (ratio 5/3 at epsilon=0.1).
OBSTRUCTION_EDGES = [
    (0, 1), (0, 3), (2, 1), (4, 1), (4, 3),
    (5, 0), (5, 2), (5, 4), (5, 6), (6, 1), (6, 3),
]


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


def exact_vertex_cover_size(G: nx.Graph) -> tuple[int, str]:
    """Return (tau(G), method).  Uses Koenig on bipartite graphs, else B&B MIS."""
    if G.number_of_edges() == 0:
        return 0, "trivial"
    if nx.is_bipartite(G):
        # Koenig: minimum vertex cover size == maximum matching size.
        matching = nx.bipartite.maximum_matching(G, top_nodes=_one_side(G))
        # maximum_matching returns each matched pair twice.
        return len(matching) // 2, "bipartite-matching"
    return _bb_vertex_cover_size(G), "exact-branch"


def _one_side(G: nx.Graph):
    side = {}
    for cc in nx.connected_components(G):
        sub = G.subgraph(cc)
        color = nx.bipartite.color(sub)
        side.update(color)
    return {v for v, c in side.items() if c == 0}


# ----------------------------------------------------------------------------
# Evaluation under the default call.
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Test classes.
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    big = not args.quick
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

    result = {
        "experiment": "car/ default-call (epsilon=0.1) ratio test for Salvador v0.0.6",
        "salvador_version": __version__,
        "seed": SEED,
        "default_epsilon": DEFAULT_EPSILON,
        "threshold": {"sub2_target": SUB2_TARGET, "statement": "7/4 = 2 - 1/4"},
        "method": {
            "candidate_solver": "salvador.algorithm.find_vertex_cover (default epsilon=0.1)",
            "optimum_solver": "Koenig/maximum-matching on bipartite graphs, else branch-and-bound MIS; no MILP",
            "ratio": "|C| / tau(G)",
        },
        "conclusion": {
            "max_ratio": overall["max_ratio"],
            "all_valid": overall["all_valid"],
            "count_above_7_4": overall["count_above_7_4"],
            "supports_7_4": overall["count_above_7_4"] == 0,
            "note": "5/3 obstruction is solved optimally at epsilon in {0.25, 0.5}.",
        },
        "overall_summary": overall,
        "summary_by_group": by_group,
        "raw_rows": rows,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
    }

    out = Path(__file__).resolve().parent
    (out / "car_experiment.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "car_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,mean_ratio,max_ratio,count_above_7_4,all_valid\n")
        for g, s in by_group.items():
            fh.write(f"{g},{s['instances']},{s['mean_ratio']:.6f},{s['max_ratio']:.6f},"
                     f"{s['count_above_7_4']},{s['all_valid']}\n")
        fh.write(f"OVERALL,{overall['instances']},{overall['mean_ratio']:.6f},"
                 f"{overall['max_ratio']:.6f},{overall['count_above_7_4']},{overall['all_valid']}\n")

    print(json.dumps({
        "salvador_version": __version__,
        "default_epsilon": DEFAULT_EPSILON,
        "instances": overall["instances"],
        "max_ratio": overall["max_ratio"],
        "all_valid": overall["all_valid"],
        "count_above_7_4": overall["count_above_7_4"],
        "supports_7_4": result["conclusion"]["supports_7_4"],
        "worst_instance": overall["worst_instance"]["name"],
    }, indent=2))


if __name__ == "__main__":
    main()
