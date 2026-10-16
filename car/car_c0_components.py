"""car/ c0 study: does amplifying the connected components before the matching help?

Strategy c0 (salvador/component_split.py) relabels every vertex u as (u, 0),
splits bridges (and, when no bridge is left, edges of the NetworkX minimum
maximal matching) of the split graph H off as isolated edges -- the other
edges of an endpoint (u, i) move to a new copy (u, i') -- until H has
r' >= |M| components (M = NetworkX minimum maximal matching of G, the matching
of c1, so |M| = |C1| / 2), and decodes the NetworkX minimum maximal matching
of H: an edge ((a, i), (b, j)) costs nothing when a or b is already selected,
otherwise selects the smaller-index endpoint (rule "low") or the larger-index
one (rule "high") when i != j, and goes to a residual graph when i == j
(covered optimally by Koenig when the residual graph is bipartite, otherwise
by number of occurrences). The decoded set is repaired and pruned.

Per instance this script records:

* |c0| under both rules ("low": i < j selects (a, i); "high": selects (b, j)),
  which rule is smaller, and |c1| (pruned NetworkX min maximal matching), the
  baseline with the same matching subroutine;
* the split statistics: components r of G and r' of H, the target |M|,
  bridge splits, matching splits, rounds, the largest copy index, and the
  decoding statistics: |M_H|, index ties i = j, open residual edges, whether
  the residual cover was exact (bipartite residual), reused selections and
  the raw decoded size before repair/pruning;
* the quality against tau(G) (exact on the small graphs, Koenig on bipartite
  graphs) or a certified lower bound L <= tau (large graphs), and whether the
  empirical bound |c0| <= 2 tau - r' holds (exact tau only; the a priori
  bound proved for c1 is |C1| <= 2 tau - r);
* with the ensemble (default), whether c0 is strictly smaller than the best of
  c1..c14 and whether its refinement is a source of the final cover c15.

Run from the repository root:

    python car/car_c0_components.py --quick --max-n 5000
    python car/car_c0_components.py                       # full sweep, Part B up to 50000
    python car/car_c0_components.py --no-ensemble         # c0 and c1 only (much faster)
    python car/car_c0_components.py --dimacs experiment   # also the DIMACS complement graphs
    python car/car_c0_components.py --timeout 60          # skip instances taking more than 60 s

Outputs (in car/): car_c0_components.json, car_c0_components.csv (one row per
instance) and car_c0_components_summary.csv (one row per family).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path

import networkx as nx

import car_common as cc
from salvador import component_split
from salvador.algorithm import (CANDIDATE_NAMES, final_candidates, maximal_matching_vertex_cover,
                                prune_redundant_vertices)

RULES = list(component_split.RULES)
PRE_C0 = [k for k in CANDIDATE_NAMES if k not in ("c0", "c15")]  # c1..c14


def is_cover(G: nx.Graph, cover: set) -> bool:
    return all(u in cover or v in cover for u, v in G.edges() if u != v)


def run_instance(name: str, group: str, G: nx.Graph, exact: bool, ensemble: bool, spectral: bool) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    adj = {v: set(G[v]) for v in G}

    t0 = time.perf_counter()
    info: dict = {}
    raw = component_split.component_split_covers(G, info)
    c0 = {rule: prune_redundant_vertices(adj, raw[rule]) for rule in RULES}
    t_c0 = time.perf_counter() - t0

    t0 = time.perf_counter()
    c1 = prune_redundant_vertices(adj, maximal_matching_vertex_cover(G))
    t_c1 = time.perf_counter() - t0

    for rule in RULES:
        if not is_cover(G, c0[rule]):
            raise AssertionError(f"{name}: c0 ({rule}) is not a vertex cover")

    if exact:
        tau, method = cc.exact_vertex_cover_size(G)
        lb, lb_method = tau, method
    else:
        lb, lb_method, _ = cc.best_lower_bound(G, spectral=spectral)
        tau = lb if lb_method.startswith("bipartite") else None
    sizes = {rule: len(c0[rule]) for rule in RULES}
    best_rule = min(RULES, key=lambda k: sizes[k])
    c0_size = sizes[best_rule]
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        **{f"c0_{rule}": sizes[rule] for rule in RULES},
        "c0": c0_size, "c0_rule": best_rule if sizes["low"] != sizes["high"] else "tie",
        "c1": len(c1), "c0_minus_c1": c0_size - len(c1),
        "tau": tau, "lower_bound": lb, "lower_bound_method": lb_method,
        **{f"ratio_{rule}": (sizes[rule] / lb) if lb else 1.0 for rule in RULES},
        "ratio_c0": (c0_size / lb) if lb else 1.0, "ratio_c1": (len(c1) / lb) if lb else 1.0,
        "r": info["r"], "r_prime": info["r_prime"], "target": info["target"],
        "bridge_splits": info["bridge_splits"], "matching_splits": info["matching_splits"],
        "rounds": info["rounds"], "max_index": info["max_index"],
        **{f"{rule}_{k}": info[rule][k] for rule in RULES
           for k in ("matching_h", "ties", "residual_open", "residual_exact", "reused", "raw_size")},
        "bound_2tau_minus_rprime": (2 * tau - info["r_prime"]) if (tau is not None and exact) else None,
        "bound_holds": (c0_size <= 2 * tau - info["r_prime"]) if (tau is not None and exact) else None,
        "seconds_c0": round(t_c0, 4), "seconds_c1": round(t_c1, 4),
    }
    if ensemble:
        t0 = time.perf_counter()
        cands, refined = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, return_refined=True)
        row["seconds_ensemble"] = round(time.perf_counter() - t0, 3)
        best_pre = min(len(cands[k]) for k in PRE_C0)
        row.update({"best_c1_c14": best_pre, "c0_minus_best": len(cands["c0"]) - best_pre,
                    "c0_strictly_best": len(cands["c0"]) < best_pre, "c15": len(cands["c15"]),
                    "c15_from_c0": len(refined["c0"]) == len(cands["c15"]),
                    "c15_only_from_c0": len(refined["c0"]) == len(cands["c15"])
                    and all(len(refined[k]) > len(cands["c15"]) for k in PRE_C0)})
    ref = f"tau={tau}" if tau is not None else f"L={lb}"
    print(f"  [c0] {group[:30]:30} {name:28} n={row['n']:>7,} m={row['m']:>8,} {ref:>10}"
          f"  low={sizes['low']:>7,} high={sizes['high']:>7,} c1={len(c1):>7,} ({row['c0_minus_c1']:+d})"
          f"  r={row['r']} r'={row['r_prime']} (target {row['target']}, bridges {row['bridge_splits']},"
          f" matching {row['matching_splits']}, max idx {row['max_index']})"
          + (f"  bound {'ok' if row['bound_holds'] else 'VIOLATED'}" if row["bound_holds"] is not None else "")
          + (f"  best c1..c14={row['best_c1_c14']:,} ({row['c0_minus_best']:+d}) c15={row['c15']:,}"
             f"{' <-c0' if row['c15_from_c0'] else ''}" if ensemble else "")
          + f"  {t_c0:.2f}s", flush=True)
    return row


def _large_job(name: str, max_n: int, ensemble: bool, spectral: bool) -> dict:
    for nm, group, build in cc.large_instances(max_n):
        if nm == name:
            return run_instance(name, group, build(), exact=False, ensemble=ensemble, spectral=spectral)
    raise KeyError(name)


def _dimacs_job(path: str, ensemble: bool, spectral: bool) -> dict:
    G = cc.load_edge_list(Path(path))
    if not G.number_of_edges():
        return {}
    return run_instance(Path(path).name.split(".clq")[0], "DIMACS / edge-list files", G,
                        exact=False, ensemble=ensemble, spectral=spectral)


def summarise(rows: list[dict], ensemble: bool) -> list[dict]:
    out = []
    groups = sorted({r["group"] for r in rows})
    for g in groups + ["ALL"]:
        rs = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(rs)
        if not k:
            continue
        checked = [r for r in rs if r["bound_holds"] is not None]
        s = {
            "family": g, "instances": k,
            "low_smaller": sum(r["c0_low"] < r["c0_high"] for r in rs),
            "high_smaller": sum(r["c0_high"] < r["c0_low"] for r in rs),
            "low_high_tie": sum(r["c0_low"] == r["c0_high"] for r in rs),
            "c0_lt_c1": sum(r["c0_minus_c1"] < 0 for r in rs),
            "c0_eq_c1": sum(r["c0_minus_c1"] == 0 for r in rs),
            "c0_gt_c1": sum(r["c0_minus_c1"] > 0 for r in rs),
            "mean_c0_minus_c1": sum(r["c0_minus_c1"] for r in rs) / k,
            **{f"mean_ratio_{x}": sum(r[f"ratio_{x}"] for r in rs) / k for x in RULES + ["c0", "c1"]},
            **{f"max_ratio_{x}": max(r[f"ratio_{x}"] for r in rs) for x in RULES + ["c0", "c1"]},
            "mean_rprime_over_target": sum(r["r_prime"] / max(r["target"], 1) for r in rs) / k,
            "mean_bridge_share": sum(r["bridge_splits"] / max(r["bridge_splits"] + r["matching_splits"], 1)
                                     for r in rs) / k,
            "residual_exact_low": sum(bool(r["low_residual_exact"]) for r in rs),
            "bound_checked": len(checked), "bound_violations": sum(not r["bound_holds"] for r in checked),
            "seconds_c0": round(sum(r["seconds_c0"] for r in rs), 3),
            "seconds_c1": round(sum(r["seconds_c1"] for r in rs), 3),
        }
        if ensemble:
            s.update({"c0_strictly_best": sum(r["c0_strictly_best"] for r in rs),
                      "c15_from_c0": sum(r["c15_from_c0"] for r in rs),
                      "c15_only_from_c0": sum(r["c15_only_from_c0"] for r in rs),
                      "mean_c0_minus_best": sum(r["c0_minus_best"] for r in rs) / k})
        out.append(s)
    return out


def print_table(summary: list[dict], ensemble: bool) -> None:
    head = (f"{'family':44} {'N':>4} {'low<':>5} {'high<':>5} {'tie':>5} {'c0<c1':>6} {'=':>5} {'c0>c1':>6}"
            f" {'d(c1)':>7} {'ratio low':>9} {'high':>7} {'c1':>7} {'max c0':>7} {'max c1':>7} {'r/tgt':>6}"
            f" {'bridge%':>7} {'bnd viol':>8}" + (f" {'c0 best':>7} {'c15<-c0':>7} {'only c0':>7}" if ensemble else ""))
    print("\nc0 (component amplification) per family: rules low/high, against c1 and tau (or L)")
    print(head)
    print("-" * len(head))
    for s in summary:
        print(f"{s['family'][:44]:44} {s['instances']:>4} {s['low_smaller']:>5} {s['high_smaller']:>5}"
              f" {s['low_high_tie']:>5} {s['c0_lt_c1']:>6} {s['c0_eq_c1']:>5} {s['c0_gt_c1']:>6}"
              f" {s['mean_c0_minus_c1']:>7.2f} {s['mean_ratio_low']:>9.4f} {s['mean_ratio_high']:>7.4f}"
              f" {s['mean_ratio_c1']:>7.4f} {s['max_ratio_c0']:>7.4f} {s['max_ratio_c1']:>7.4f}"
              f" {s['mean_rprime_over_target']:>6.2f} {100 * s['mean_bridge_share']:>6.1f}%"
              f" {str(s['bound_violations']) + '/' + str(s['bound_checked']):>8}"
              + (f" {s['c0_strictly_best']:>7} {s['c15_from_c0']:>7} {s['c15_only_from_c0']:>7}" if ensemble else ""))
    print("\nratio = |cover| / tau on exact/bipartite instances, |cover| / L (certified lower bound) otherwise;"
          " bound = |c0| <= 2 tau - r' (exact tau only); r/tgt = r' / |M|.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller small-graph sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip the small exact graphs")
    ap.add_argument("--skip-large", action="store_true", help="skip the large adversarial families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest large instance (default 50000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    ap.add_argument("--no-ensemble", action="store_true", help="only c0 and c1 (skip c1..c15)")
    ap.add_argument("--no-spectral", action="store_true", help="skip the spectral lower bound on large graphs")
    cc.add_timeout_argument(ap)
    args = ap.parse_args()
    ensemble = not args.no_ensemble
    spectral = not args.no_spectral

    started = time.time()
    rows: list[dict] = []
    if not args.skip_small:
        rng = random.Random(cc.SEED)
        for name, group, G in cc.small_instances(rng, big=not args.quick):
            rows.append(run_instance(name, group, G, exact=True, ensemble=ensemble, spectral=spectral))
    skipped: list[str] = []
    with cc.InstanceRunner(args.timeout) as runner:
        if not args.skip_large:
            for name, group, _ in cc.large_instances(args.max_n):
                row = runner.run(_large_job, name, args.max_n, ensemble, spectral)
                if row is None:
                    print(f"  [c0] {group[:30]:30} {name:28} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    rows.append(row)
                gc.collect()
        if args.dimacs:
            for path in sorted(Path(args.dimacs).iterdir()):
                if path.is_file():
                    row = runner.run(_dimacs_job, str(path), ensemble, spectral)
                    if row is None:
                        print(f"  [c0] {path.name}: skipped (more than {args.timeout:g} s)", flush=True)
                        skipped.append(path.name)
                    elif row:
                        rows.append(row)

    summary = summarise(rows, ensemble)
    print_table(summary, ensemble)
    if skipped:
        print(f"\nSkipped (more than {args.timeout:g} s): {', '.join(skipped)}")

    out = cc.OUT_DIR
    stem = "car_c0_components"
    (out / f"{stem}.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "rules": RULES,
         "ensemble": ensemble, "timeout": args.timeout, "skipped_instances": skipped,
         "summary": summary, "rows": rows, "elapsed_seconds": time.time() - started}, indent=1, default=str),
        encoding="utf-8")
    for path, data in ((out / f"{stem}.csv", rows), (out / f"{stem}_summary.csv", summary)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            if data:
                fields = list(dict.fromkeys(k for d in data for k in d))
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)


if __name__ == "__main__":
    main()
