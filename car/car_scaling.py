"""car/ scaling study for Salvador: empirical check of the O(n + m) claim.

Doubles the instance size from 1,000 up to ``--max-n`` on two families,

* a random 4-regular graph (bounded degree, m = 2n), and
* a Barabasi-Albert graph with m = 3 (degree-heterogeneous hubs),

times the default call ``find_vertex_cover(G, epsilon=1)`` on each, fits
wall-clock time against (n + m) by least squares (slope, intercept, R^2)
and reports time / (n + m) at every size. Each cover is also checked for
validity. A flat time / (n + m) column and R^2 close to 1 are the
empirical signature of linear time.

Run from the repository root:

    python car/car_scaling.py                # up to n = 200,000
    python car/car_scaling.py --max-n 50000

Outputs (in car/): car_scaling.json and car_scaling.csv.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import networkx as nx

import car_common as cc
from salvador.algorithm import find_vertex_cover

FAMILIES = {
    "regular4": ("Scaling: 4-regular", lambda n, s: cc.random_regular(n, 4, s)),
    "barabasi_albert3": ("Scaling: Barabasi-Albert m=3", lambda n, s: cc.barabasi_albert(n, 3, s)),
}


def time_one(name: str, G: nx.Graph, family: str) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    gc.collect()
    t0 = time.perf_counter()
    cover = find_vertex_cover(G, epsilon=cc.DEFAULT_EPSILON)
    elapsed = time.perf_counter() - t0
    valid = all(u in cover or v in cover for u, v in G.edges())
    print(f"  [scaling] {name}: n={n:,} m={m:,} {elapsed:.2f}s "
          f"({elapsed * 1e6 / (n + m):.2f} us per n+m) valid={valid}", flush=True)
    return {"family": family, "name": name, "n": n, "m": m, "n_plus_m": n + m,
            "cover": len(cover), "valid": valid, "elapsed_seconds": elapsed,
            "us_per_np1m": elapsed * 1e6 / (n + m)}


def run(max_n: int) -> dict:
    started = time.time()
    rows = []
    size, seed = 1000, cc.SEED + 10_000
    while size <= max_n:
        for offset, (fam, (_, build)) in enumerate(FAMILIES.items()):
            rows.append(time_one(f"scaling_{fam}_n{size}", build(size, seed + offset), fam))
        seed += 2
        size *= 2
    summary = {}
    for fam in FAMILIES:
        frows = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["n_plus_m"])
        summary[fam] = {
            "fit_time_vs_(n+m)": cc.linreg([float(r["n_plus_m"]) for r in frows],
                                           [r["elapsed_seconds"] for r in frows]),
            "points": frows,
        }
    return {"experiment": f"car/ scaling study (epsilon={cc.DEFAULT_EPSILON})",
            "salvador_version": cc.__version__, "max_n_requested": max_n,
            "environment": cc.environment(), "all_valid": all(r["valid"] for r in rows),
            "scaling_summary": summary, "elapsed_seconds": time.time() - started}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-n", type=int, default=200_000, help="largest instance (default 200000)")
    args = ap.parse_args()
    result = run(args.max_n)
    out = cc.OUT_DIR
    with (out / "car_scaling.csv").open("w", encoding="utf-8") as fh:
        fh.write("family,n,m,n_plus_m,elapsed_seconds,us_per_np1m\n")
        for fam, s in result["scaling_summary"].items():
            for p in s["points"]:
                fh.write(f"{fam},{p['n']},{p['m']},{p['n_plus_m']},{p['elapsed_seconds']:.6f},{p['us_per_np1m']:.4f}\n")
        fh.write("\nfamily,slope_seconds_per_np1m,intercept_seconds,r2,n_points\n")
        for fam, s in result["scaling_summary"].items():
            f = s["fit_time_vs_(n+m)"]
            fh.write(f"{fam},{f['slope']},{f['intercept']},{f['r2']},{f['n_points']}\n")
    (out / "car_scaling.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"all_valid": result["all_valid"],
                      "fits": {k: v["fit_time_vs_(n+m)"] for k, v in result["scaling_summary"].items()}}, indent=2))


if __name__ == "__main__":
    main()
