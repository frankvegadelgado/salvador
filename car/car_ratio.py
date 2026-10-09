"""car/ approximation-ratio study for Salvador (default call, epsilon=1).

PART A -- small graphs with exact optima (no MILP): Koenig / maximum
matching on bipartite instances, branch-and-bound maximum independent set
otherwise (tau(G) = |V| - alpha(G)). Reports the exact ratio |C| / tau(G).

PART B -- large adversarial graphs. Exact optima are out of reach, so each
reported ratio is a *certified upper bound* |C| / LB on the true ratio,
where LB <= tau(G) is the best of:

* an exact Hopcroft-Karp maximum matching on bipartite graphs (then
  LB = tau(G) by Koenig and the ratio is exact);
* the greedy maximal matching used by earlier versions of car/ (kept and
  reported separately as ``ratio_upper_bound_legacy`` for comparison);
* a Karp-Sipser matching (linear time, near-maximum on sparse graphs);
* the Laplacian Hoffman bound tau(G) >= n - floor(n (1 - delta/mu_max)),
  which is far stronger than any matching on regular graphs, where every
  matching is at most n/2 while tau(G) is much larger.

Run from the repository root:

    python car/car_ratio.py                 # Part A + Part B
    python car/car_ratio.py --quick         # smaller Part A
    python car/car_ratio.py --skip-large    # Part A only
    python car/car_ratio.py --skip-small    # Part B only
    python car/car_ratio.py --max-n 50000   # smaller Part B
    python car/car_ratio.py --skip "watts"  # also skip instances matching a pattern

The sparse Erdos-Renyi G(n, p) instance with n = 200,000 is skipped by default
(``--keep-gnp-200k`` restores it): building it takes too long. It is skipped
before its builder is called, so the graph is never constructed.

Outputs (in car/): car_ratio.json, car_summary.csv, car_large_summary.csv.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time

import networkx as nx

import car_common as cc
from salvador.algorithm import find_vertex_cover


# ----------------------------------------------------------------------------
# PART A
# ----------------------------------------------------------------------------
def evaluate(name: str, G: nx.Graph, group: str) -> dict:
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    cover = set(find_vertex_cover(G, epsilon=cc.DEFAULT_EPSILON))
    opt, method = cc.exact_vertex_cover_size(G)
    ratio = 1.0 if opt == 0 else len(cover) / opt
    return {
        "group": group, "name": name,
        "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "cover": len(cover), "opt": opt, "ratio": ratio,
        "above_7_4": bool(ratio > cc.SUB2_TARGET + 1e-12),
        "valid": all(u in cover or v in cover for u, v in G.edges()),
        "method": method,
        "edges": [[int(u), int(v)] for u, v in G.edges()] if G.number_of_nodes() <= 16 else None,
    }


def bipartite_hill_climb(rng: random.Random, restarts: int, steps: int) -> list[dict]:
    """Adaptive search for bad bipartite instances (keeps the worst ratio)."""
    rows = []
    for r in range(restarts):
        a, b = rng.randint(3, 6), rng.randint(3, 6)
        G = nx.bipartite.random_graph(a, b, 0.5, seed=rng.randrange(2**32))
        if G.number_of_edges() == 0:
            G.add_edge(0, a)
        best = evaluate(f"hill_{r}_0", G, "Bipartite hill-climb (exact)")
        left, right = list(range(a)), list(range(a, a + b))
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


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"instances": 0}
    return {
        "instances": len(rows),
        "mean_ratio": statistics.fmean(r["ratio"] for r in rows),
        "max_ratio": max(r["ratio"] for r in rows),
        "count_above_7_4": sum(1 for r in rows if r["above_7_4"]),
        "all_valid": all(r["valid"] for r in rows),
        "worst_instance": max(rows, key=lambda r: r["ratio"]),
    }


def run_part_a(big: bool) -> dict:
    # Same RNG consumption order as the original single-script suite:
    # obstruction, relabel, random bipartite, hill climb, grids, atlas, general.
    rng = random.Random(cc.SEED)
    started = time.time()
    rows: list[dict] = []
    for name, group, G in cc.small_instances_before_hill(rng, big):
        rows.append(evaluate(name, G, group))
    rows += bipartite_hill_climb(rng, 80 if big else 12, 40 if big else 12)
    for name, group, G in cc.small_instances_after_hill(rng, big):
        rows.append(evaluate(name, G, group))
    by_group = {g: summarise([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})}
    overall = summarise(rows)
    return {
        "experiment": f"car/ Part A: default-call (epsilon={cc.DEFAULT_EPSILON}) exact-ratio test",
        "salvador_version": cc.__version__,
        "seed": cc.SEED,
        "conclusion": {
            "max_ratio": overall["max_ratio"],
            "all_valid": overall["all_valid"],
            "count_above_7_4": overall["count_above_7_4"],
        },
        "overall_summary": overall,
        "summary_by_group": by_group,
        "raw_rows": rows,
        "elapsed_seconds": time.time() - started,
    }


# ----------------------------------------------------------------------------
# PART B
# ----------------------------------------------------------------------------
def evaluate_large(name: str, G: nx.Graph, group: str) -> dict:
    G = nx.convert_node_labels_to_integers(G, ordering="default")
    n0, m0 = G.number_of_nodes(), G.number_of_edges()
    print(f"  [ratio] {name}: n={n0:,} m={m0:,} ...", flush=True)
    t0 = time.perf_counter()
    cover = set(find_vertex_cover(G, epsilon=cc.DEFAULT_EPSILON))
    elapsed = time.perf_counter() - t0
    valid = all(u in cover or v in cover for u, v in G.edges())
    lb, method, bounds = cc.best_lower_bound(G)
    legacy = bounds.get("greedy_maximal_matching", lb)
    row = {
        "group": group, "name": name, "n": n0, "m": m0,
        "cover": len(cover),
        "lower_bound": lb, "lower_bound_method": method, "all_lower_bounds": bounds,
        "ratio_upper_bound": None if lb == 0 else len(cover) / lb,
        "ratio_upper_bound_legacy": None if legacy == 0 else len(cover) / legacy,
        "exact": method.startswith("bipartite"),
        "valid": bool(valid),
        "elapsed_seconds": elapsed,
    }
    print(f"  [ratio] {name}: cover={len(cover):,} LB={lb:,} ({method}) "
          f"ratio<={row['ratio_upper_bound']:.4f} [{elapsed:.1f}s]", flush=True)
    del G, cover
    gc.collect()
    return row


_GNP_MARKERS = ("erdos", "erdős", "renyi", "rényi", "gnp", "g(n,p)", "g(n, p)")
_200K_MARKERS = ("200000", "200_000", "200,000", "200k", "2e5")


def is_gnp_200k(name: str, group: str) -> bool:
    """True for the sparse Erdos-Renyi G(n, p) Part B instance with n = 200,000."""
    text = f"{name} {group}".lower()
    return any(m in text for m in _GNP_MARKERS) and any(m in text for m in _200K_MARKERS)


def skip_instance(name: str, group: str, patterns: list[str], keep_gnp_200k: bool) -> bool:
    if not keep_gnp_200k and is_gnp_200k(name, group):
        return True
    text = f"{name} {group}".lower()
    return any(p.lower() in text for p in patterns)


def run_part_b(max_n: int, patterns: list[str] | None = None, keep_gnp_200k: bool = False) -> dict:
    started = time.time()
    rows = []
    skipped = []
    for name, group, build in cc.large_instances(max_n):
        if skip_instance(name, group, patterns or [], keep_gnp_200k):
            print(f"  [ratio] {name}: skipped (not built)", flush=True)
            skipped.append(name)
            continue
        rows.append(evaluate_large(name, build(), group))
    by_group = {}
    for g in sorted({r["group"] for r in rows}):
        grows = [r for r in rows if r["group"] == g]
        ratios = [r["ratio_upper_bound"] for r in grows if r["ratio_upper_bound"] is not None]
        legacy = [r["ratio_upper_bound_legacy"] for r in grows if r["ratio_upper_bound_legacy"] is not None]
        by_group[g] = {
            "instances": len(grows),
            "max_ratio_upper_bound": max(ratios) if ratios else None,
            "mean_ratio_upper_bound": statistics.fmean(ratios) if ratios else None,
            "max_ratio_upper_bound_legacy": max(legacy) if legacy else None,
            "all_valid": all(r["valid"] for r in grows),
            "max_n": max(r["n"] for r in grows),
            "max_elapsed_seconds": max(r["elapsed_seconds"] for r in grows),
        }
    ratios = [r["ratio_upper_bound"] for r in rows if r["ratio_upper_bound"] is not None]
    legacy = [r["ratio_upper_bound_legacy"] for r in rows if r["ratio_upper_bound_legacy"] is not None]
    worst = max(rows, key=lambda r: r["ratio_upper_bound"] or 0) if rows else None
    return {
        "experiment": f"car/ Part B: large adversarial graphs (epsilon={cc.DEFAULT_EPSILON})",
        "salvador_version": cc.__version__,
        "max_n_requested": max_n,
        "note": ("ratio_upper_bound = |cover| / LB with LB the best certified lower bound "
                 "on tau(G); exact when lower_bound_method starts with 'bipartite'. "
                 "ratio_upper_bound_legacy uses the greedy maximal matching of earlier "
                 "car/ versions and is reported only for comparison."),
        "family_rows": rows,
        "skipped_instances": skipped,
        "summary_by_group": by_group,
        "overall_max_ratio_upper_bound": max(ratios) if ratios else None,
        "overall_worst_instance": None if worst is None else worst["name"],
        "overall_max_ratio_upper_bound_legacy": max(legacy) if legacy else None,
        "overall_all_valid": all(r["valid"] for r in rows),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller/faster Part A sweep")
    ap.add_argument("--skip-large", action="store_true", help="run Part A only")
    ap.add_argument("--skip-small", action="store_true", help="run Part B only")
    ap.add_argument("--max-n", type=int, default=200_000, help="largest Part B instance (default 200000)")
    ap.add_argument("--skip", action="append", default=[], metavar="PATTERN",
                    help="skip Part B instances whose name or group contains PATTERN (repeatable)")
    ap.add_argument("--keep-gnp-200k", action="store_true",
                    help="do not skip the sparse Erdos-Renyi G(n, p) instance with n = 200,000")
    args = ap.parse_args()

    out = cc.OUT_DIR
    result = {"salvador_version": cc.__version__, "default_epsilon": cc.DEFAULT_EPSILON,
              "seed": cc.SEED, "environment": cc.environment()}

    if not args.skip_small:
        part_a = run_part_a(big=not args.quick)
        result["part_a"] = part_a
        with (out / "car_summary.csv").open("w", encoding="utf-8") as fh:
            fh.write("group,instances,mean_ratio,max_ratio,count_above_7_4,all_valid\n")
            for g, s in part_a["summary_by_group"].items():
                fh.write(f"{g},{s['instances']},{s['mean_ratio']:.6f},{s['max_ratio']:.6f},"
                         f"{s['count_above_7_4']},{s['all_valid']}\n")
            o = part_a["overall_summary"]
            fh.write(f"OVERALL,{o['instances']},{o['mean_ratio']:.6f},{o['max_ratio']:.6f},"
                     f"{o['count_above_7_4']},{o['all_valid']}\n")
        print(json.dumps({"part": "A (small, exact ratio)", "instances": o["instances"],
                          "max_ratio": o["max_ratio"], "all_valid": o["all_valid"],
                          "worst_instance": o["worst_instance"]["name"]}, indent=2))

    if not args.skip_large:
        part_b = run_part_b(args.max_n, args.skip, args.keep_gnp_200k)
        result["part_b"] = part_b
        with (out / "car_large_summary.csv").open("w", encoding="utf-8") as fh:
            fh.write("group,instances,max_ratio_upper_bound,mean_ratio_upper_bound,"
                     "max_ratio_upper_bound_legacy,all_valid,max_n,max_elapsed_seconds\n")
            for g, s in part_b["summary_by_group"].items():
                fh.write(f"{g},{s['instances']},{s['max_ratio_upper_bound']},{s['mean_ratio_upper_bound']},"
                         f"{s['max_ratio_upper_bound_legacy']},{s['all_valid']},{s['max_n']},"
                         f"{s['max_elapsed_seconds']:.6f}\n")
        print(json.dumps({"part": "B (large adversarial graphs)", "max_n_requested": args.max_n,
                          "instances": len(part_b["family_rows"]),
                          "skipped_instances": part_b["skipped_instances"],
                          "overall_max_ratio_upper_bound": part_b["overall_max_ratio_upper_bound"],
                          "overall_worst_instance": part_b["overall_worst_instance"],
                          "overall_max_ratio_upper_bound_legacy": part_b["overall_max_ratio_upper_bound_legacy"],
                          "overall_all_valid": part_b["overall_all_valid"]}, indent=2))

    (out / "car_ratio.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
