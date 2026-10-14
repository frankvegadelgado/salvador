"""car/ regular graphs with progressively higher degree (Salvador 0.1.2, default call).

For every degree d of a progressive sequence (default 3, 4, 5, 6, 8, 10, 12,
16, 20, 24, 32, 48, 64) and every size n, Salvador's ensemble is run on

* **general** random d-regular graphs (``networkx.random_regular_graph``,
  almost surely non-bipartite: c10 does not apply, c13 has a large core);
* **bipartite** random d-regular graphs (union of d edge-disjoint random
  perfect matchings): tau(G) = n/2 exactly (Koenig + Hall), c10 applies.

Optimum / lower bound per instance:

* bipartite: tau(G) by Hopcroft-Karp (exact, = n/2);
* general with n <= --exact-n: tau(G) by the exact branch and bound of
  car_common (exact ratio);
* general, larger n: the best certified lower bound of car_common (matchings,
  clique packing, Laplacian Hoffman bound); the ratio is then an *upper bound*
  on the true ratio. Also reported, for the trend only and NOT certified, the
  ratio against the Frieze-Luczak estimate tau ~ n (1 - 2 ln d / d) of random
  d-regular graphs (meaningful for large d).

Run from the repository root:

    python car/car_regular.py                          # default degrees, n = 200, 1000, 5000
    python car/car_regular.py --quick                  # n = 200 only, 2 repetitions
    python car/car_regular.py --degrees 3 6 12 24 48 96 --sizes 1000 10000
    python car/car_regular.py --family general         # or bipartite (default: both)
    python car/car_regular.py --timeout 60             # skip instances taking more than 60 s

Every instance (construction and evaluation) runs in a worker process with a
time limit (``--timeout``, default 120 s, 0 = no limit); an instance that takes
longer is skipped and listed in car_regular.json (``skipped_instances``). Each
instance has its own random generator derived from the seed and its name.

Outputs (in car/): car_regular.json, car_regular.csv (one row per instance) and
car_regular_summary.csv (one row per family, n and degree).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter

import networkx as nx

import car_common as cc
from salvador.algorithm import CANDIDATE_NAMES, final_candidates

SEED = cc.SEED
DEFAULT_DEGREES = (3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
DEFAULT_SIZES = (200, 1000, 5000)
FAMILIES = ("general", "bipartite")
PRE_FINAL = [k for k in CANDIDATE_NAMES if k != "c15"]


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def general_regular(n: int, d: int, rng) -> nx.Graph:
    if (n * d) % 2:
        n += 1
    return nx.random_regular_graph(d, n, seed=rng.randrange(2**31))


def bipartite_regular(n: int, d: int, rng) -> nx.Graph:
    """Simple d-regular bipartite graph on 2 * (n // 2) vertices: d edge-disjoint random perfect matchings.

    Each matching is a random permutation; an edge already present is removed by
    swapping its target with a random position whose two new edges are both new
    (a swap changes only those two positions, so it never creates a repeated edge).
    """
    half = n // 2
    if d > half:
        raise ValueError(f"d = {d} > n/2 = {half}")
    if 2 * d > half:
        # dense case: the bipartite complement of a random (half - d)-regular bipartite graph
        H = bipartite_regular(n, half - d, rng)
        G = nx.Graph()
        G.add_nodes_from(range(2 * half))
        G.add_edges_from((i, half + j) for i in range(half) for j in range(half) if not H.has_edge(i, half + j))
        return G
    nbrs = [set() for _ in range(half)]
    G = nx.Graph()
    G.add_nodes_from(range(2 * half))
    for _ in range(d):
        perm = list(range(half))
        rng.shuffle(perm)
        for i in range(half):
            tries = 0
            while perm[i] in nbrs[i]:
                j = rng.randrange(half)
                if perm[j] not in nbrs[i] and perm[i] not in nbrs[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                tries += 1
                if tries > 50 * half:
                    raise RuntimeError("could not complete the matching")
        for i in range(half):
            nbrs[i].add(perm[i])
            G.add_edge(i, half + perm[i])
    return G


def instance_list(families, sizes, degrees, reps, exact_n):
    """(name, family, n, d, rep) of every instance, small exact ones first."""
    out = []
    for family in families:
        size_list = ([exact_n] if family == "general" and exact_n > 0 else []) + list(sizes)
        for n in size_list:
            for d in degrees:
                if d >= n or (family == "bipartite" and d > n // 2):
                    continue
                for r in range(reps):
                    out.append((f"{family}_reg_n{n}_d{d}_r{r}", family, n, d, r))
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def frieze_luczak_tau(n: int, d: int) -> float | None:
    """Estimate n (1 - 2 ln d / d) of tau for random d-regular graphs (asymptotic in d; NOT certified)."""
    if d < 3:
        return None
    est = n * (1.0 - 2.0 * math.log(d) / d)
    return est if est > 0 else None


def run_instance(name: str, family: str, n: int, d: int, exact_n: int) -> dict:
    rng = cc.instance_rng(SEED, name)
    t0 = time.perf_counter()
    G = general_regular(n, d, rng) if family == "general" else bipartite_regular(n, d, rng)
    build = time.perf_counter() - t0
    n_actual, m = G.number_of_nodes(), G.number_of_edges()
    degs = {deg for _, deg in G.degree()}
    assert degs == {d}, f"{name}: not {d}-regular ({sorted(degs)})"

    t0 = time.perf_counter()
    if family == "bipartite":
        lb, method = cc.exact_vertex_cover_size(G)
        exact = True
    elif n_actual <= exact_n:
        lb, method = cc.exact_vertex_cover_size(G)
        exact = True
    else:
        lb, method, _ = cc.best_lower_bound(G)
        exact = method.startswith("bipartite")
    bound_seconds = time.perf_counter() - t0

    info: dict = {}
    t0 = time.perf_counter()
    cands = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in CANDIDATE_NAMES}
    for k in CANDIDATE_NAMES:
        C = cands[k]
        if not all(u in C or v in C for u, v in G.edges()):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
        if exact and len(C) < lb:
            raise AssertionError(f"{name}: strategy {k} is below tau -- impossible")
    final = sizes["c15"]
    best_pre = min(sizes[k] for k in PRE_FINAL)
    est = frieze_luczak_tau(n_actual, d) if family == "general" else None
    row = {
        "instance": name, "family": family, "n": n_actual, "d": d, "m": m,
        "lower_bound": lb, "lower_bound_method": method, "exact": exact,
        "cover": final, "cover_fraction": final / n_actual,
        "ratio": final / lb if lb else 1.0,
        "optimal": exact and final == lb,
        "ratio_vs_fl_estimate": None if est is None else final / est,
        "best_before_c15": best_pre,
        "winners": [k for k in PRE_FINAL if sizes[k] == best_pre],
        "c15_gain": best_pre - final,
        **{f"size_{k}": sizes[k] for k in CANDIDATE_NAMES},
        "c10_exact": bool(info.get("c10_exact", False)),
        "c10_core_size": info.get("c10_core_size", 0),
        "c10_bipartite_fraction": info.get("c10_bipartite_fraction", 0.0),
        "c13_exact": bool(info.get("c13_exact", False)),
        "c13_core_size": info.get("c13_core_size", 0),
        "build_seconds": round(build, 3),
        "bound_seconds": round(bound_seconds, 3),
        "seconds": round(elapsed, 3),
    }
    kind = "tau" if exact else "LB "
    print(f"  [regular] {name:30} n={n_actual:>6,} d={d:>3} m={m:>8,} {kind}={lb:>6,} cover={final:>6,} "
          f"ratio{'=' if exact else '<='}{row['ratio']:.4f} winners={','.join(row['winners'])} "
          f"[{elapsed:.2f}s]", flush=True)
    return row


def _instance_job(name: str, family: str, n: int, d: int, exact_n: int) -> dict:
    return run_instance(name, family, n, d, exact_n)


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def summarise(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["family"], r["n"], r["d"]) for r in rows}, key=lambda k: (FAMILIES.index(k[0]), k[1], k[2]))
    for family, n, d in keys:
        g = [r for r in rows if (r["family"], r["n"], r["d"]) == (family, n, d)]
        wins = Counter(w for r in g for w in r["winners"])
        fl = [r["ratio_vs_fl_estimate"] for r in g if r["ratio_vs_fl_estimate"] is not None]
        out.append({
            "family": family, "n": n, "d": d, "instances": len(g),
            "exact_bound": all(r["exact"] for r in g),
            "mean_ratio": round(statistics.fmean(r["ratio"] for r in g), 6),
            "max_ratio": round(max(r["ratio"] for r in g), 6),
            "optimal": sum(r["optimal"] for r in g),
            "mean_cover_fraction": round(statistics.fmean(r["cover_fraction"] for r in g), 6),
            "mean_ratio_vs_fl_estimate": round(statistics.fmean(fl), 6) if fl else None,
            "top_winner": wins.most_common(1)[0][0] if wins else None,
            "winner_counts": " ".join(f"{k}:{v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])),
            "mean_c15_gain": round(statistics.fmean(r["c15_gain"] for r in g), 3),
            "c10_exact": sum(r["c10_exact"] for r in g),
            "mean_c10_core": round(statistics.fmean(r["c10_core_size"] for r in g), 1),
            "mean_c13_core": round(statistics.fmean(r["c13_core_size"] for r in g), 1),
            "mean_seconds": round(statistics.fmean(r["seconds"] for r in g), 3),
            "max_seconds": round(max(r["seconds"] for r in g), 3),
        })
    return out


def print_summary(summary: list[dict]) -> None:
    print("\nSummary per family, n and degree (ratio is exact when 'exact', else |C| / certified LB)")
    print(f"{'family':10} {'n':>6} {'d':>4} {'N':>3} {'exact':>5} {'mean ratio':>10} {'max ratio':>9} {'opt':>4} "
          f"{'|C|/n':>7} {'vs FL':>7} {'winner':>7} {'c15 gain':>8} {'c10 ex':>6} {'mean s':>7}")
    for s in summary:
        fl = "" if s["mean_ratio_vs_fl_estimate"] is None else f"{s['mean_ratio_vs_fl_estimate']:.4f}"
        print(f"{s['family']:10} {s['n']:>6} {s['d']:>4} {s['instances']:>3} {str(s['exact_bound']):>5} "
              f"{s['mean_ratio']:>10.4f} {s['max_ratio']:>9.4f} {s['optimal']:>4} {s['mean_cover_fraction']:>7.4f} "
              f"{fl:>7} {str(s['top_winner']):>7} {s['mean_c15_gain']:>8} {s['c10_exact']:>6} {s['mean_seconds']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES),
                    help=f"degrees, in increasing order (default: {' '.join(map(str, DEFAULT_DEGREES))})")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                    help=f"numbers of vertices (default: {' '.join(map(str, DEFAULT_SIZES))})")
    ap.add_argument("--reps", type=int, default=3, help="instances per (family, n, d) (default: 3)")
    ap.add_argument("--family", choices=FAMILIES, action="append",
                    help="general and/or bipartite (repeatable; default: both)")
    ap.add_argument("--exact-n", type=int, default=40,
                    help="extra size of general graphs solved exactly by branch and bound (default: 40; 0 = none)")
    ap.add_argument("--quick", action="store_true", help="n = 200 only, 2 repetitions, no exact size")
    cc.add_timeout_argument(ap)
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.reps, args.exact_n = [200], 2, 0
    families = tuple(args.family) if args.family else FAMILIES
    degrees = sorted(set(args.degrees))

    jobs = instance_list(families, sorted(set(args.sizes)), degrees, args.reps, args.exact_n)
    print(f"Regular graphs: {len(jobs)} instances, degrees {degrees}, families {list(families)}, "
          f"time limit {args.timeout:g} s per instance")
    started = time.time()
    rows, skipped = [], []
    with cc.InstanceRunner(args.timeout) as runner:
        for name, family, n, d, _ in jobs:
            row = runner.run(_instance_job, name, family, n, d, args.exact_n)
            if row is None:
                print(f"  [regular] {name}: skipped (more than {args.timeout:g} s to build and evaluate)", flush=True)
                skipped.append(name)
                continue
            rows.append(row)
    summary = summarise(rows)
    print_summary(summary)
    print(f"\n{len(rows)} instances in {time.time() - started:.1f} s; skipped (time limit): {len(skipped)}"
          + (f" -- {', '.join(skipped)}" if skipped else ""))

    out = cc.OUT_DIR
    with open(out / "car_regular.json", "w") as fh:
        json.dump({"environment": cc.environment(), "seed": SEED, "degrees": degrees, "sizes": sorted(set(args.sizes)),
                   "reps": args.reps, "exact_n": args.exact_n, "families": list(families), "timeout": args.timeout,
                   "skipped_instances": skipped, "rows": rows, "summary": summary}, fh, indent=1)
    flat = [{**r, "winners": " ".join(r["winners"])} for r in rows]
    for path, data in ((out / "car_regular.csv", flat), (out / "car_regular_summary.csv", summary)):
        with open(path, "w", newline="") as fh:
            if data:
                writer = csv.DictWriter(fh, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)


if __name__ == "__main__":
    main()
