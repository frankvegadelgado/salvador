"""car/ scaling study: does the running time of Salvador grow linearly in n + m?

Every strategy c1..c17 runs in worst-case linear time
with the default budgets (Theorem "runtime"). This script measures it. For
every graph family it builds graphs of doubling size, n = n0, 2 n0, 4 n0, ...,
times the default call of Salvador (final_candidates, the call behind
find_vertex_cover, epsilon = 1), and fits

    log(seconds) = a + b * log(n + m)

by least squares. Linear time means slope b = 1 (up to noise and cache
effects); quadratic time would give b = 2. The script reports, per family:

* the slope b over all sizes, and over the larger half of the sizes (the
  asymptotic regime), with R^2;
* the time per element, microseconds per (n + m), at every size, which stays
  roughly constant for a linear algorithm;
* the doubling ratios t(2s) / t(s), which stay near 2 for a linear algorithm
  (4 would mean quadratic);
* the same slope for every strategy separately (c1, c2..c6 together, c7, c8,
  c9, c10, c11, c12, c14, c15, c16, c17, pruning -- which includes the union
  re-prune c13 -- and the rest), so a super-linear piece would be visible even
  if it is small.

Measuring the strategies separately
-----------------------------------
The breakdown wraps the public functions that the ensemble calls (results are
unchanged; only time is recorded). Time is charged to the *outermost* wrapped
call: the swap searches and reduction passes that c15 runs internally are
charged to c15, not to c14 and c11, and the pruning inside c1 and c9 to them. Everything not wrapped (cleaning the
graph, building adjacency maps, comparisons) is "rest". The breakdown run is
a second call on the same graph; the headline total is the first, unwrapped
call. Use --no-breakdown to skip it.

The graph generators are themselves linear (fast G(n,p), configuration-style
random regular graphs, Barabasi-Albert, Watts-Strogatz, grids, random trees,
random bipartite graphs with uniform random edges, the random triangle-free
process); their time is reported separately and is not part of the fit.
Vertex labels are shuffled.

Every instance (construction and timing) runs in a worker process with a time
limit (--timeout, default 600 s). When an instance of a family exceeds it,
the larger instances of that family are skipped (they would exceed it too),
and the family's fit uses the sizes that finished.

Outputs (in car/): car_scaling.json (every row and fit), car_scaling.csv (one
row per instance, with the per-strategy seconds), car_scaling_summary.csv (one
row per family: slopes, R^2, time per element, doubling ratios) and
car_scaling_table.tex (a LaTeX table of the summary for the paper).

Run from the repository root:

    python car/car_scaling.py --quick           # sizes 1,000 .. 16,000 (minutes)
    python car/car_scaling.py                   # sizes 1,000 .. 128,000
    python car/car_scaling.py --max-n 256000 --reps 3
    python car/car_scaling.py --families reg3 er4 ba3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import time

import networkx as nx

import car_common as cc

SEED = 20261014

#: family key -> (description, average degree hint used only for the label)
FAMILIES: dict[str, str] = {
    "reg3": "Random 3-regular",
    "reg6": "Random 6-regular",
    "reg12": "Random 12-regular",
    "er4": "Sparse Erdos-Renyi, avg degree 4",
    "er16": "Erdos-Renyi, avg degree 16",
    "ba3": "Barabasi-Albert, m = 3",
    "ws6": "Watts-Strogatz, k = 6, p = 0.1",
    "grid": "2D grid",
    "tree": "Random tree",
    "bip6": "Random bipartite, avg degree 6",
    "tf4": "Random triangle-free process, avg degree 4",
}

PIECES = ("c1", "c2-c6", "c7", "c8", "c9", "c10", "c11", "c12", "c14", "c15", "c16", "c17", "prune", "rest")


# ----------------------------------------------------------------------------
# Linear-time generators
# ----------------------------------------------------------------------------
def _relabel(G: nx.Graph, rng: random.Random) -> nx.Graph:
    nodes = list(G)
    perm = nodes[:]
    rng.shuffle(perm)
    H = nx.Graph()
    H.add_nodes_from(perm)
    mp = dict(zip(nodes, perm))
    H.add_edges_from((mp[u], mp[v]) for u, v in G.edges())
    return H


def build(family: str, n: int, rng: random.Random) -> nx.Graph:
    s = rng.randrange(2**31)
    if family.startswith("reg"):
        d = int(family[3:])
        G = nx.random_regular_graph(d, n - (n * d) % 2, seed=s)
    elif family.startswith("er"):
        d = int(family[2:])
        G = nx.fast_gnp_random_graph(n, d / (n - 1), seed=s)
    elif family == "ba3":
        G = nx.barabasi_albert_graph(n, 3, seed=s)
    elif family == "ws6":
        G = nx.watts_strogatz_graph(n, 6, 0.1, seed=s)
    elif family == "grid":
        k = max(2, int(round(math.sqrt(n))))
        G = nx.grid_2d_graph(k, k)
    elif family == "tree":
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from((i, rng.randrange(i)) for i in range(1, n))
    elif family == "bip6":
        half = n // 2
        G = nx.Graph()
        G.add_nodes_from(range(n))
        target = 3 * n  # average degree 6
        while G.number_of_edges() < target:
            G.add_edge(rng.randrange(half), half + rng.randrange(n - half))
    elif family == "tf4":
        G = nx.empty_graph(n)
        target, tries = 2 * n, 0
        while G.number_of_edges() < target and tries < 40 * target:
            tries += 1
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v and not G.has_edge(u, v) and set(G[u]).isdisjoint(G[v]):
                G.add_edge(u, v)
    else:
        raise KeyError(family)
    G.remove_nodes_from([v for v in list(G) if G.degree(v) == 0])
    return _relabel(G, rng)


# ----------------------------------------------------------------------------
# Per-strategy timing (installed in the worker process)
# ----------------------------------------------------------------------------
_acc: dict[str, float] = {}
_stack: list[str] = []
_installed = False


def _timed(piece: str, fn):
    def wrapper(*a, **k):
        if _stack:  # nested call: charged to the outermost piece
            return fn(*a, **k)
        _stack.append(piece)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _acc[piece] = _acc.get(piece, 0.0) + time.perf_counter() - t0
            _stack.pop()
    wrapper.__wrapped__ = fn
    return wrapper


def install() -> None:
    global _installed
    if _installed:
        return
    from salvador import (algorithm, bipartite_reduction, consensus_force, eds_treewidth, local_search, reduce_search,
                          subdivision_reduction, treewidth_dp, vc_reduction)
    for name in ("maximal_matching_vertex_cover", "bucket_degree_greedy", "covering_via_reduction_max_degree_1",
                 "min_to_min_vertex_cover_linear", "linear_min_weighted_vertex_cover"):
        setattr(algorithm, name, _timed("c2-c6", getattr(algorithm, name)))
    algorithm.prune_redundant_vertices = _timed("prune", algorithm.prune_redundant_vertices)
    algorithm.c1_cover = _timed("c1", algorithm.c1_cover)  # its pruning is charged to c1 (outermost)
    vc_reduction.solve_vc = _timed("c7", vc_reduction.solve_vc)
    bipartite_reduction.bipartite_planar_covers = _timed("c8", bipartite_reduction.bipartite_planar_covers)
    algorithm.c9_cover = _timed("c9", algorithm.c9_cover)
    eds_treewidth.eds_treewidth_cover = _timed("c10", eds_treewidth.eds_treewidth_cover)
    subdivision_reduction.subdivision_covers = _timed("c11", subdivision_reduction.subdivision_covers)
    treewidth_dp.treewidth_vertex_cover = _timed("c12", treewidth_dp.treewidth_vertex_cover)
    local_search.improve_vertex_cover = _timed("c14", local_search.improve_vertex_cover)
    reduce_search.reduce_search_cover = _timed("c15", reduce_search.reduce_search_cover)
    consensus_force.consensus_force_cover = _timed("c16", consensus_force.consensus_force_cover)
    algorithm.refine_cover = _timed("c17", algorithm.refine_cover)
    _installed = True


# ----------------------------------------------------------------------------
# One instance (runs in the worker process)
# ----------------------------------------------------------------------------
def _job(family: str, n_target: int, reps: int, breakdown: bool) -> dict:
    from salvador.algorithm import FINAL_NAME, final_candidates

    rng = cc.instance_rng(SEED, f"{family}_{n_target}")
    t0 = time.perf_counter()
    G = build(family, n_target, rng)
    build_seconds = time.perf_counter() - t0
    n, m = G.number_of_nodes(), G.number_of_edges()

    times = []
    cover = None
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        cands = final_candidates(G, epsilon=cc.DEFAULT_EPSILON)
        times.append(time.perf_counter() - t0)
        cover = cands[FINAL_NAME]
    if not all(u in cover or v in cover for u, v in G.edges()):
        raise AssertionError(f"{family} n={n}: invalid cover")

    pieces: dict[str, float] = {}
    if breakdown:
        install()
        _acc.clear()
        gc.collect()
        t0 = time.perf_counter()
        info: dict = {}
        final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info)
        total_b = time.perf_counter() - t0
        pieces = {p: _acc.get(p, 0.0) for p in PIECES if p != "rest"}
        pieces["rest"] = max(0.0, total_b - sum(pieces.values()))
    seconds = statistics.median(times)
    return {
        "family": family, "group": FAMILIES[family], "n_target": n_target, "n": n, "m": m, "size": n + m,
        "build_seconds": build_seconds, "seconds": seconds, "seconds_all_reps": times,
        "us_per_element": 1e6 * seconds / (n + m), "cover": len(cover), "pieces": pieces,
    }


# ----------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------
def loglog_fit(xs: list[float], ys: list[float]) -> dict | None:
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": b, "intercept": a, "r2": r2, "points": len(pts)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for fam in FAMILIES:
        F = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["size"])
        if not F:
            continue
        xs = [r["size"] for r in F]
        ys = [r["seconds"] for r in F]
        half = F[len(F) // 2:] if len(F) >= 4 else F
        piece_slopes = {}
        for p in PIECES:
            pys = [r["pieces"].get(p, 0.0) for r in F]
            # a piece that takes under 1 ms everywhere is timer noise: no slope
            if max(pys, default=0.0) >= 1e-3:
                fit = loglog_fit(xs, pys)
                piece_slopes[p] = fit["slope"] if fit else None
        out[fam] = {
            "group": FAMILIES[fam],
            "sizes": [r["n"] for r in F],
            "fit_all": loglog_fit(xs, ys),
            "fit_large_half": loglog_fit([r["size"] for r in half], [r["seconds"] for r in half]),
            "us_per_element": [r["us_per_element"] for r in F],
            "doubling_ratios": [F[i + 1]["seconds"] / F[i]["seconds"] for i in range(len(F) - 1)
                                if F[i]["seconds"] > 0],
            "max_n": F[-1]["n"], "max_m": F[-1]["m"], "max_seconds": F[-1]["seconds"],
            "piece_slopes": piece_slopes,
        }
    return out


def print_summary(summary: dict) -> None:
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    print("\nScaling of the default call (seconds vs n + m, log-log least squares)")
    head = (f"{'family':44} {'sizes':>5} {'max n':>8} {'slope':>6} {'R2':>6} {'slope*':>6} "
            f"{'us/elem first':>13} {'us/elem last':>12} {'doubling ratios'}")
    print(head)
    print("-" * len(head))
    for fam, s in summary.items():
        fa, fh = s["fit_all"], s["fit_large_half"]
        print(f"{s['group'][:44]:44} {len(s['sizes']):>5} {s['max_n']:>8,} {f(fa and fa['slope'])} "
              f"{f(fa and fa['r2'])} {f(fh and fh['slope'])} {s['us_per_element'][0]:>13.2f} "
              f"{s['us_per_element'][-1]:>12.2f} " + " ".join(f"{x:.2f}" for x in s["doubling_ratios"]))
    print("slope  = exponent b in seconds ~ (n + m)^b over all sizes (1 = linear, 2 = quadratic);")
    print("slope* = the same over the larger half of the sizes; us/elem = microseconds per (n + m).")
    print("\nPer-strategy slopes (pieces taking at least 1 ms at some size):")
    print(f"{'family':44} " + " ".join(f"{p:>6}" for p in PIECES))
    for fam, s in summary.items():
        print(f"{s['group'][:44]:44} " + " ".join(f"{f(s['piece_slopes'].get(p), 2):>6}" for p in PIECES))


def write_outputs(rows: list[dict], summary: dict, skipped: list[str], args, started: float) -> None:
    out = cc.OUT_DIR
    with (out / "car_scaling.csv").open("w", encoding="utf-8") as fh:
        fh.write("family,group,n,m,size,build_seconds,seconds,us_per_element,cover,"
                 + ",".join(f"sec_{p}" for p in PIECES) + "\n")
        for r in rows:
            fh.write(",".join([r["family"], f"\"{r['group']}\"", str(r["n"]), str(r["m"]), str(r["size"]),
                               f"{r['build_seconds']:.4f}", f"{r['seconds']:.4f}", f"{r['us_per_element']:.4f}",
                               str(r["cover"])] + [f"{r['pieces'].get(p, 0.0):.4f}" for p in PIECES]) + "\n")
    with (out / "car_scaling_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family,group,instances,max_n,max_m,slope_all,r2_all,slope_large_half,us_per_element_first,"
                 "us_per_element_last,max_doubling_ratio,max_seconds,"
                 + ",".join(f"slope_{p}" for p in PIECES) + "\n")
        for fam, s in summary.items():
            fa, fh_ = s["fit_all"] or {}, s["fit_large_half"] or {}
            cell = lambda x: "" if x is None else f"{x:.4f}"
            fh.write(",".join([fam, f"\"{s['group']}\"", str(len(s["sizes"])), str(s["max_n"]), str(s["max_m"]),
                               cell(fa.get("slope")), cell(fa.get("r2")), cell(fh_.get("slope")),
                               cell(s["us_per_element"][0]), cell(s["us_per_element"][-1]),
                               cell(max(s["doubling_ratios"], default=None)), cell(s["max_seconds"])]
                              + [cell(s["piece_slopes"].get(p)) for p in PIECES]) + "\n")
    lines = [
        "% Generated by car/car_scaling.py -- running time of the default call against n + m.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Family & Sizes & Max $n$ & Slope & $R^2$ & Slope (large half) & $\\mu$s per $(n+m)$, first/last \\\\",
        "\\midrule",
    ]
    for fam, s in summary.items():
        fa, fh_ = s["fit_all"] or {}, s["fit_large_half"] or {}
        g = s["group"].replace("Erdos-Renyi", "Erd\\H{o}s--R\\'enyi").replace("Barabasi-Albert", "Barab\\'asi--Albert")
        g = g.replace("Watts-Strogatz", "Watts--Strogatz")
        max_n = f"{s['max_n']:,}".replace(",", "{,}")
        lines.append(f"{g} & {len(s['sizes'])} & {max_n} & {fa.get('slope', float('nan')):.3f} & "
                     f"{fa.get('r2', float('nan')):.3f} & {fh_.get('slope', float('nan')):.3f} & "
                     f"{s['us_per_element'][0]:.1f} / {s['us_per_element'][-1]:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out / "car_scaling_table.tex").write_text("\n".join(lines), encoding="utf-8")
    (out / "car_scaling.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": SEED, "args": vars(args),
         "pieces": PIECES, "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="sizes 1,000 .. 16,000")
    ap.add_argument("--min-n", type=int, default=1000, help="smallest size (default 1000)")
    ap.add_argument("--max-n", type=int, default=None, help="largest size (default 128000; 16000 with --quick)")
    ap.add_argument("--reps", type=int, default=1, help="timed repetitions per instance; the median is used")
    ap.add_argument("--families", nargs="*", default=None, choices=list(FAMILIES),
                    help="families to run (default: all)")
    ap.add_argument("--no-breakdown", action="store_true", help="skip the per-strategy timing run")
    cc.add_timeout_argument(ap, default=600.0)
    args = ap.parse_args()
    max_n = args.max_n or (16_000 if args.quick else 128_000)
    families = args.families or list(FAMILIES)
    sizes = []
    n = args.min_n
    while n <= max_n:
        sizes.append(n)
        n *= 2

    started = time.time()
    rows: list[dict] = []
    skipped: list[str] = []
    with cc.InstanceRunner(args.timeout) as runner:
        for fam in families:
            failed = False
            for n in sizes:
                name = f"{fam}_n{n}"
                if failed:
                    print(f"  [scaling] {FAMILIES[fam][:40]:40} n={n:>9,} skipped (a smaller size timed out)",
                          flush=True)
                    skipped.append(name)
                    continue
                row = runner.run(_job, fam, n, args.reps, not args.no_breakdown)
                if row is None:
                    print(f"  [scaling] {FAMILIES[fam][:40]:40} n={n:>9,} skipped (more than {args.timeout:g} s)",
                          flush=True)
                    skipped.append(name)
                    failed = True
                    continue
                rows.append(row)
                top = sorted(row["pieces"].items(), key=lambda kv: -kv[1])[:3]
                print(f"  [scaling] {row['group'][:40]:40} n={row['n']:>9,} m={row['m']:>10,} "
                      f"t={row['seconds']:8.2f}s  {row['us_per_element']:6.2f} us/elem  build={row['build_seconds']:.2f}s"
                      + ("  top: " + ", ".join(f"{p} {t:.2f}s" for p, t in top) if top else ""), flush=True)
                gc.collect()
            write_outputs(rows, summarise(rows), skipped, args, started)  # partial results after each family

    summary = summarise(rows)
    write_outputs(rows, summary, skipped, args, started)
    print_summary(summary)
    if skipped:
        print(f"\nSkipped: {', '.join(skipped)}")
    print("\nNote: a slope near 1 on every family is what O(n + m) predicts; timings include Python overhead and"
          " cache effects, which raise the time per element slowly with size. The fit measures the instances run,"
          " it does not prove the worst-case bound, which is Theorem runtime of the paper.")


if __name__ == "__main__":
    main()
