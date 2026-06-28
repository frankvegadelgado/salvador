from __future__ import annotations

"""car/ -- epsilon-aware sqrt(2) experiment for Salvador v0.0.5.

The accuracy-controlled pipeline solves the weighted independent-dominating-set
gadget with a Baker-style PTAS whose layering width is ``k = ceil(1/epsilon)``.
Because the forest-core gadget is itself a forest, the PTAS solves it
near-optimally, and the question is whether the resulting cover stays at or below
``sqrt(2) ~= 1.41421`` times the optimum.

The experiment does NOT use MILP.  Exact optima are computed feasibly through the
identity ``tau(G) = |V| - alpha(G)`` with a deterministic branch-and-bound maximum
independent set solver, so every reported ratio is measured against a true OPT.

The script sweeps several values of ``epsilon`` per graph and records, for each
graph, the best (smallest) ratio attained as ``epsilon`` decreases.  The headline
question is: across all tested families, does the activated pipeline keep the best
ratio at or below ``sqrt(2)``?

This is empirical evidence only, not a proof.  Run from the repository root with:

    python car/car_sqrt2_experiment.py            # full feasible suite
    python car/car_sqrt2_experiment.py --quick    # smaller, faster sweep

Outputs:
    car/car_sqrt2_experiment.json
    car/car_sqrt2_summary.csv
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
SQRT2 = math.sqrt(2.0)

# epsilon = 1.0 reproduces the v0.0.4 greedy baseline (k = 1); the smaller values
# activate progressively more thorough PTAS layering (k = 2, 4, 10, 20).
EPSILONS = [1.0, 0.5, 0.25, 0.1, 0.05]


# ----------------------------------------------------------------------------
# Exact optimum via branch-and-bound maximum independent set (no MILP).
# ----------------------------------------------------------------------------
def exact_vertex_cover_size(G: nx.Graph) -> int:
    """Exact minimum vertex-cover size through tau(G) = |V| - alpha(G)."""
    nodes = list(G.nodes())
    n = len(nodes)
    if G.number_of_edges() == 0:
        return 0

    idx = {v: i for i, v in enumerate(nodes)}
    adj = [0] * n
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        adj[iu] |= 1 << iv
        adj[iv] |= 1 << iu

    best = 0

    def search(candidates: int, chosen: int) -> None:
        nonlocal best
        if candidates == 0:
            best = max(best, chosen)
            return
        if chosen + candidates.bit_count() <= best:
            return
        tmp = candidates
        branch_vertex = (tmp & -tmp).bit_length() - 1
        max_deg = -1
        while tmp:
            bit = tmp & -tmp
            i = bit.bit_length() - 1
            deg = (adj[i] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_vertex = i
            tmp ^= bit
        vbit = 1 << branch_vertex
        search(candidates & ~vbit & ~adj[branch_vertex], chosen + 1)
        search(candidates & ~vbit, chosen)

    search((1 << n) - 1, 0)
    return n - best


def exact_vertex_cover(G: nx.Graph) -> list[int]:
    """Return one lexicographically first exact minimum vertex cover."""
    nodes = list(G.nodes())
    opt = exact_vertex_cover_size(G)
    for combo in itertools.combinations(nodes, opt):
        candidate = set(combo)
        if all(u in candidate or v in candidate for u, v in G.edges()):
            return [int(x) for x in combo]
    raise RuntimeError("no vertex cover found")


def canonical_edges(G: nx.Graph) -> list[list[int]]:
    mapping = {v: i for i, v in enumerate(sorted(G.nodes()))}
    return [[mapping[u], mapping[v]]
            for u, v in sorted(G.edges(), key=lambda e: (mapping[e[0]], mapping[e[1]]))]


# ----------------------------------------------------------------------------
# Per-graph evaluation across the epsilon sweep.
# ----------------------------------------------------------------------------
def evaluate_graph(name: str, G: nx.Graph, family: str,
                   epsilons: Iterable[float]) -> dict:
    """Evaluate one graph at every epsilon; record the per-epsilon ratios and best."""
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    opt = exact_vertex_cover_size(G)

    per_epsilon = {}
    best_ratio = math.inf
    best_epsilon = None
    all_valid = True
    for eps in epsilons:
        cover = set(find_vertex_cover(G, epsilon=eps))
        valid = all(u in cover or v in cover for u, v in G.edges())
        all_valid = all_valid and valid
        ratio = 1.0 if opt == 0 else len(cover) / opt
        per_epsilon[f"{eps:g}"] = {
            "epsilon": eps,
            "salvador_size": len(cover),
            "ratio": ratio,
            "valid": bool(valid),
        }
        if ratio < best_ratio:
            best_ratio = ratio
            best_epsilon = eps

    return {
        "name": name,
        "family": family,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "opt": opt,
        "per_epsilon": per_epsilon,
        "best_ratio": best_ratio,
        "best_epsilon": best_epsilon,
        "best_above_sqrt2": bool(best_ratio > SQRT2 + 1e-12),
        "all_valid": bool(all_valid),
        "edges": canonical_edges(G) if G.number_of_nodes() <= 16 else None,
    }


# ----------------------------------------------------------------------------
# Graph families (all kept feasible for exact optima).
# ----------------------------------------------------------------------------
def atlas_graphs(max_n: int, epsilons) -> list[dict]:
    """All nonempty graphs in the NetworkX atlas up to ``max_n`` vertices."""
    from networkx.generators.atlas import graph_atlas_g
    rows = []
    for i, G in enumerate(graph_atlas_g()):
        if G.number_of_nodes() < 2 or G.number_of_nodes() > max_n:
            continue
        if G.number_of_edges() == 0:
            continue
        rows.append(evaluate_graph(f"atlas_{i}", G, "atlas", epsilons))
    return rows


def structured_families(epsilons, big: bool) -> list[dict]:
    rows: list[dict] = []
    path_max = 33 if big else 17
    for n in range(2, path_max):
        rows.append(evaluate_graph(f"path_{n}", nx.path_graph(n), "paths", epsilons))
    for n in range(3, path_max):
        rows.append(evaluate_graph(f"cycle_{n}", nx.cycle_graph(n), "cycles", epsilons))
    comp_max = 23 if big else 13
    for n in range(2, comp_max):
        rows.append(evaluate_graph(f"complete_{n}", nx.complete_graph(n), "complete", epsilons))
    star_max = 41 if big else 21
    for leaves in range(2, star_max):
        rows.append(evaluate_graph(f"star_{leaves}", nx.star_graph(leaves), "stars", epsilons))
    bip_max = 13 if big else 8
    for a in range(1, bip_max):
        for b in range(1, bip_max):
            rows.append(evaluate_graph(f"complete_bipartite_{a}_{b}",
                                       nx.complete_bipartite_graph(a, b),
                                       "complete_bipartite", epsilons))
    wheel_max = 25 if big else 13
    for n in range(4, wheel_max):
        rows.append(evaluate_graph(f"wheel_{n}", nx.wheel_graph(n), "wheels", epsilons))
    grid_r = 6 if big else 5
    grid_c = 7 if big else 5
    for r in range(2, grid_r):
        for c in range(2, grid_c):
            rows.append(evaluate_graph(f"grid_{r}x{c}", nx.grid_2d_graph(r, c), "grids", epsilons))
    barbell_max = 9 if big else 6
    for k in range(2, barbell_max):
        rows.append(evaluate_graph(f"barbell_{k}_1", nx.barbell_graph(k, 1), "barbells", epsilons))
    return rows


def random_gnp(epsilons, big: bool) -> list[dict]:
    rows: list[dict] = []
    rng = random.Random(SEED)
    probabilities = [0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80]
    n_hi = 17 if big else 13
    trials = 10 if big else 4
    for n in range(7, n_hi):
        for p in probabilities:
            for _ in range(trials):
                seed = rng.randrange(2**32)
                G = nx.gnp_random_graph(n, p, seed=seed)
                if G.number_of_edges() == 0:
                    continue
                rows.append(evaluate_graph(f"gnp_n{n}_p{p:.2f}_seed{seed}", G,
                                           "erdos_renyi", epsilons))
    return rows


def regular_and_hard(epsilons) -> list[dict]:
    """Near-regular / expander-like graphs -- the hardest class for the heuristic."""
    rows: list[dict] = []
    rows.append(evaluate_graph("petersen", nx.petersen_graph(), "regular_hard", epsilons))
    rows.append(evaluate_graph("dodecahedral", nx.dodecahedral_graph(), "regular_hard", epsilons))
    rows.append(evaluate_graph("desargues", nx.desargues_graph(), "regular_hard", epsilons))
    rng = random.Random(SEED + 7)
    for n, d in [(8, 3), (10, 3), (12, 3), (12, 4), (14, 4), (16, 3)]:
        try:
            G = nx.random_regular_graph(d, n, seed=rng.randrange(2**32))
        except nx.NetworkXError:
            continue
        rows.append(evaluate_graph(f"regular_d{d}_n{n}", G, "regular_hard", epsilons))
    # An explicit balanced 8-vertex stress instance for the activated pipeline.
    W = nx.Graph()
    W.add_nodes_from(range(8))
    W.add_edges_from([(0, 2), (0, 3), (2, 4), (2, 5), (2, 6),
                      (3, 4), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)])
    rows.append(evaluate_graph("balanced_witness_8", W, "regular_hard", epsilons))
    return rows


# ----------------------------------------------------------------------------
# Summaries.
# ----------------------------------------------------------------------------
def summarise(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"instances": 0}
    best = max(rows, key=lambda r: r["best_ratio"])
    best_ratios = [r["best_ratio"] for r in rows]
    return {
        "instances": len(rows),
        "min_best_ratio": min(best_ratios),
        "mean_best_ratio": statistics.fmean(best_ratios),
        "median_best_ratio": statistics.median(best_ratios),
        "max_best_ratio": max(best_ratios),
        "count_best_above_sqrt2": sum(1 for r in rows if r["best_above_sqrt2"]),
        "worst_instance": best,
    }


def per_epsilon_summary(rows: list[dict], epsilons) -> dict:
    """Max ratio and sqrt(2) violations at each fixed epsilon."""
    out = {}
    for eps in epsilons:
        key = f"{eps:g}"
        ratios = [r["per_epsilon"][key]["ratio"] for r in rows]
        out[key] = {
            "epsilon": eps,
            "max_ratio": max(ratios) if ratios else None,
            "mean_ratio": statistics.fmean(ratios) if ratios else None,
            "count_above_sqrt2": sum(1 for x in ratios if x > SQRT2 + 1e-12),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="smaller/faster sweep for a quick check")
    args = ap.parse_args()
    big = not args.quick
    epsilons = EPSILONS if big else [1.0, 0.25, 0.1]
    atlas_max_n = 7 if big else 6

    started = time.time()
    atlas = atlas_graphs(atlas_max_n, epsilons)
    structured = structured_families(epsilons, big)
    randoms = random_gnp(epsilons, big)
    hard = regular_and_hard(epsilons)
    all_rows = atlas + structured + randoms + hard

    overall = summarise(all_rows)
    by_family = {fam: summarise([r for r in all_rows if r["family"] == fam])
                 for fam in sorted({r["family"] for r in all_rows})}
    by_epsilon = per_epsilon_summary(all_rows, epsilons)

    # The worst instance, with an exact optimum cover attached for the record.
    worst = overall["worst_instance"]
    if worst.get("edges"):
        Wg = nx.Graph()
        Wg.add_nodes_from(range(worst["n"]))
        Wg.add_edges_from(tuple(e) for e in worst["edges"])
        worst = dict(worst)
        worst["exact_cover"] = exact_vertex_cover(Wg)

    result = {
        "experiment": "car/ epsilon-aware sqrt(2) test for Salvador v0.0.5",
        "salvador_version": __version__,
        "seed": SEED,
        "thresholds": {"sqrt2": SQRT2},
        "epsilons": epsilons,
        "method": {
            "candidate_solver": "salvador.algorithm.find_vertex_cover (epsilon active in v0.0.5)",
            "optimum_solver": "exact branch-and-bound maximum independent set; no MILP",
            "ratio": "|C_salvador(epsilon)| / tau(G)",
            "best_ratio": "minimum ratio over the epsilon sweep for each graph",
            "purpose": "test whether the activated pipeline keeps the best ratio <= sqrt(2); not a proof",
        },
        "conclusion": {
            "max_best_ratio": overall["max_best_ratio"],
            "supports_sqrt2_for_activated_pipeline":
                overall["max_best_ratio"] <= SQRT2 + 1e-12,
            "count_best_above_sqrt2": overall["count_best_above_sqrt2"],
            "note": "best_ratio is taken at the smallest epsilon that helped; "
                    "compare per_epsilon_summary to see epsilon's effect.",
        },
        "overall_summary": overall,
        "per_epsilon_summary": by_epsilon,
        "summary_by_family": by_family,
        "worst_instance": worst,
        "raw_rows": all_rows,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
    }

    out_dir = Path(__file__).resolve().parent
    (out_dir / "car_sqrt2_experiment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    with (out_dir / "car_sqrt2_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family,instances,min_best_ratio,mean_best_ratio,median_best_ratio,"
                 "max_best_ratio,count_best_above_sqrt2,worst_name,worst_n,worst_m,worst_opt\n")
        for fam, s in by_family.items():
            if s["instances"] == 0:
                continue
            w = s["worst_instance"]
            fh.write(f"{fam},{s['instances']},{s['min_best_ratio']:.12g},"
                     f"{s['mean_best_ratio']:.12g},{s['median_best_ratio']:.12g},"
                     f"{s['max_best_ratio']:.12g},{s['count_best_above_sqrt2']},"
                     f"{w['name']},{w['n']},{w['m']},{w['opt']}\n")
        w = overall["worst_instance"]
        fh.write(f"OVERALL,{overall['instances']},{overall['min_best_ratio']:.12g},"
                 f"{overall['mean_best_ratio']:.12g},{overall['median_best_ratio']:.12g},"
                 f"{overall['max_best_ratio']:.12g},{overall['count_best_above_sqrt2']},"
                 f"{w['name']},{w['n']},{w['m']},{w['opt']}\n")

    print(json.dumps({
        "salvador_version": __version__,
        "epsilons": epsilons,
        "instances": overall["instances"],
        "max_best_ratio": overall["max_best_ratio"],
        "count_best_above_sqrt2": overall["count_best_above_sqrt2"],
        "supports_sqrt2_for_activated_pipeline":
            result["conclusion"]["supports_sqrt2_for_activated_pipeline"],
        "per_epsilon_max_ratio": {k: v["max_ratio"] for k, v in by_epsilon.items()},
        "worst_instance": overall["worst_instance"]["name"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
