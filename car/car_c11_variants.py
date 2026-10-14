"""car/ c11 variant study: which (peel, fold) setting of c11 is worth keeping?

c11 (salvador/subdivision_reduction.py) solves vertex cover on the weighted
subdivision of G by exact reductions (degree 0, degree 1, triangle and, with
``fold``, the degree-2 fold) and a heuristic peel when none applies:

    max        peel = select the largest residual degree      (the first c11)
    max+fold   the same, with degree-2 folding
    min        peel = leave out the smallest residual degree (min-degree MIS rule)
    min+fold   the same, with degree-2 folding
    ref        peel = select the largest residual degree among the vertices of
               the reference cover (best of c1..c10), falling back to max
    ref+fold   the same, with degree-2 folding

The ensemble runs ``subdivision_reduction.VARIANTS`` (currently max+fold and
ref+fold) and keeps the smaller cover; ``ensemble_c11`` is that cover's size.

Per instance the script runs all four variants (each pruned exactly as in the
ensemble) and, as the bar to beat, the ensemble candidates c1..c10 (``best_c1_c10``
is the smallest of c1..c10, c9 included; ``c10`` is reported on its own). It
records every variant's size, its gap to best_c1_c10 (negative = c11 would
improve the ensemble before c12 and c13), its reductions, folds and peels, whether it
needed no peel (then it is a minimum cover), whether the fold budget was hit,
and its running time. When an exact optimum is cheap (Part A) it is reported too.

Per family it reports, for every variant: instances where it strictly beats
best_c1_c10, ties it, loses; total and mean gap; how often it is exact; and
total seconds.

Instances: the small Part A graphs (exact tau) and the Part B adversarial
families of car_common, plus optionally a directory of edge-list files.

Run from the repository root:

    python car/car_c11_variants.py --quick --max-n 5000
    python car/car_c11_variants.py                       # full: Part A + Part B up to 50000
    python car/car_c11_variants.py --skip-small --max-n 200000
    python car/car_c11_variants.py --dimacs experiment --skip-small --skip-large
    python car/car_c11_variants.py --fold-budget 16      # larger fold work budget

Every Part B / edge-list instance runs in a worker process with a time limit
(``--timeout``, default 120 s).

Outputs (in car/): car_c11_variants.json, car_c11_variants.csv (one row per
instance) and car_c11_variants_summary.csv (one row per family x variant).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import networkx as nx

import car_common as cc
from salvador import subdivision_reduction as sr
from salvador.algorithm import ensemble_candidates, prune_redundant_vertices

ALL_VARIANTS: tuple[tuple[str, bool], ...] = (("max", False), ("max", True), ("min", False), ("min", True),
                                              ("ref", False), ("ref", True))
BEFORE_C11 = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10")


def vname(key: tuple[str, bool]) -> str:
    return key[0] + ("+fold" if key[1] else "")


NAMES = [vname(k) for k in ALL_VARIANTS]


def run_instance(name: str, group: str, G: nx.Graph, fold_budget: float, exact: bool) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    adj = {v: set(G[v]) for v in G}
    edges = list(G.edges())

    t0 = time.perf_counter()
    cands = ensemble_candidates(G)
    ensemble_seconds = time.perf_counter() - t0
    ref_cover = min((cands[k] for k in BEFORE_C11), key=len)
    best = len(ref_cover)

    sizes, gaps, details = {}, {}, {}
    for key in ALL_VARIANTS:
        vi: dict = {}
        t0 = time.perf_counter()
        cover = prune_redundant_vertices(adj, sr.subdivision_vertex_cover(adj, vi, key[0], key[1], fold_budget, ref_cover))
        secs = time.perf_counter() - t0
        if not all(u in cover or v in cover for u, v in edges):
            raise AssertionError(f"{name}: c11 variant {vname(key)} is not a vertex cover")
        nm = vname(key)
        sizes[nm] = len(cover)
        gaps[nm] = len(cover) - best
        details[nm] = {"exact": vi["exact"], "deg0": vi["deg0"], "deg1": vi["deg1"],
                       "triangle": vi["triangle"], "folds": vi["folds"], "peels": vi["peels"],
                       "ref_peels": vi["ref_peels"],
                       "fold_budget_hit": vi["fold_budget_hit"], "seconds": secs}

    tau = None
    if exact and G.number_of_nodes() <= 40:
        tau, _ = cc.exact_vertex_cover_size(G)
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "best_c1_c10": best, "c10": len(cands["c10"]), "tau": tau,
        "ensemble_c11": len(cands["c11"]), "c12": len(cands["c12"]), "c13": len(cands["c13"]),
        "sizes": sizes, "gaps": gaps, "details": details,
        "ensemble_seconds": ensemble_seconds,
    }
    print(f"  [c11] {group[:30]:30} {name:26} best_c1_c10={best:>7,} "
          + " ".join(f"{nm}={sizes[nm]:>7,}({gaps[nm]:+d})" for nm in NAMES)
          + f" c11={len(cands['c11']):,} c12={len(cands['c12']):,} c13={len(cands['c13']):,}"
          + (f" tau={tau}" if tau is not None else ""), flush=True)
    return row


def _large_job(name: str, max_n: int, fold_budget: float) -> dict:
    for nm, group, build in cc.large_instances(max_n):
        if nm == name:
            return run_instance(name, group, build(), fold_budget, exact=False)
    raise KeyError(name)


def _dimacs_job(path: str, fold_budget: float) -> dict:
    G = cc.load_edge_list(Path(path))
    if not G.number_of_edges():
        return {}
    return run_instance(Path(path).name.split(".clq")[0], "DIMACS / edge-list files", G, fold_budget, exact=False)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(grows)
        per = {}
        for nm in NAMES:
            gap = [r["gaps"][nm] for r in grows]
            per[nm] = {
                "beats_best_c1_c10": sum(x < 0 for x in gap),
                "ties": sum(x == 0 for x in gap),
                "loses": sum(x > 0 for x in gap),
                "total_gap": sum(gap),
                "mean_gap": sum(gap) / k if k else 0.0,
                "best_variant": sum(r["sizes"][nm] == min(r["sizes"].values()) for r in grows),
                "exact": sum(r["details"][nm]["exact"] for r in grows),
                "optimal": sum(r["tau"] is not None and r["sizes"][nm] == r["tau"] for r in grows),
                "fold_budget_hit": sum(r["details"][nm]["fold_budget_hit"] for r in grows),
                "seconds": sum(r["details"][nm]["seconds"] for r in grows),
            }
        out[g] = {"instances": k, "with_tau": sum(r["tau"] is not None for r in grows),
                  "ensemble_seconds": sum(r["ensemble_seconds"] for r in grows), "per_variant": per}
    return out


def print_table(summary: dict) -> None:
    print("\nc11 variants against best_c1_c10 (smallest of c1..c10): beats / ties / loses, total gap"
          " (negative = c11 improves the ensemble), exact = no peel needed")
    head = f"{'family':40} {'inst':>5} " + " ".join(f"{nm:>30}" for nm in NAMES)
    print(head)
    print("-" * len(head))
    for g, s in summary.items():
        cells = []
        for nm in NAMES:
            p = s["per_variant"][nm]
            cells.append(f"{p['beats_best_c1_c10']:>4}/{p['ties']:>4}/{p['loses']:>4} {p['total_gap']:>+7} ex{p['exact']:>4}")
        print(f"{g[:40]:40} {s['instances']:>5} " + " ".join(f"{c:>30}" for c in cells))
    print("\nSeconds per variant (all instances): "
          + ", ".join(f"{nm} {summary['ALL']['per_variant'][nm]['seconds']:.2f}" for nm in NAMES)
          + f"; ensemble c1..c13 {summary['ALL']['ensemble_seconds']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller small-graph sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip the small Part A graphs")
    ap.add_argument("--skip-large", action="store_true", help="skip the Part B adversarial families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest Part B instance (default 50000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    ap.add_argument("--fold-budget", type=float, default=sr.DEFAULT_FOLD_BUDGET,
                    help=f"fold work budget in units of (n + m) (default {sr.DEFAULT_FOLD_BUDGET})")
    cc.add_timeout_argument(ap)
    args = ap.parse_args()

    started = time.time()
    rows: list[dict] = []
    if not args.skip_small:
        rng = random.Random(cc.SEED)
        for name, group, G in cc.small_instances(rng, big=not args.quick):
            rows.append(run_instance(name, group, G, args.fold_budget, exact=True))
    skipped: list[str] = []
    with cc.InstanceRunner(args.timeout) as runner:
        if not args.skip_large:
            for name, group, _ in cc.large_instances(args.max_n):
                row = runner.run(_large_job, name, args.max_n, args.fold_budget)
                if row is None:
                    print(f"  [c11] {group[:30]:30} {name:26} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    rows.append(row)
                gc.collect()
        if args.dimacs:
            for path in sorted(Path(args.dimacs).iterdir()):
                if path.is_file():
                    row = runner.run(_dimacs_job, str(path), args.fold_budget)
                    if row is None:
                        print(f"  [c11] {path.name}: skipped (more than {args.timeout:g} s)", flush=True)
                        skipped.append(path.name)
                    elif row:
                        rows.append(row)

    summary = summarise(rows)
    out = cc.OUT_DIR
    with (out / "car_c11_variants.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,n,m,best_c1_c10,c10,ensemble_c11,c12,c13,tau,"
                 + ",".join(f"{nm},{nm}_gap,{nm}_exact,{nm}_folds,{nm}_peels,{nm}_seconds" for nm in NAMES)
                 + ",ensemble_seconds\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},{r['best_c1_c10']},{r['c10']},"
                     f"{r['ensemble_c11']},{r['c12']},{r['c13']},"
                     f"{'' if r['tau'] is None else r['tau']},"
                     + ",".join(f"{r['sizes'][nm]},{r['gaps'][nm]},{r['details'][nm]['exact']},"
                                f"{r['details'][nm]['folds']},{r['details'][nm]['peels']},"
                                f"{r['details'][nm]['seconds']:.4f}" for nm in NAMES)
                     + f",{r['ensemble_seconds']:.3f}\n")
    with (out / "car_c11_variants_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,variant,beats_best_c1_c10,ties,loses,total_gap,mean_gap,best_variant,"
                 "exact,optimal,with_tau,fold_budget_hit,seconds\n")
        for g, s in summary.items():
            for nm in NAMES:
                p = s["per_variant"][nm]
                fh.write(f"\"{g}\",{s['instances']},{nm},{p['beats_best_c1_c10']},{p['ties']},{p['loses']},"
                         f"{p['total_gap']},{p['mean_gap']:.4f},{p['best_variant']},{p['exact']},"
                         f"{p['optimal']},{s['with_tau']},{p['fold_budget_hit']},{p['seconds']:.3f}\n")
    (out / "car_c11_variants.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(),
         "variants": NAMES, "ensemble_variants": [vname(k) for k in sr.VARIANTS],
         "fold_budget": args.fold_budget, "timeout": args.timeout, "skipped_instances": skipped,
         "summary": summary, "rows": rows, "elapsed_seconds": time.time() - started},
        indent=2, default=str), encoding="utf-8")
    print_table(summary)
    if skipped:
        print(f"\nSkipped (more than {args.timeout:g} s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
