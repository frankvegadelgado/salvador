"""car/ c1 study: does amplifying the connected components before the matching help?

Strategy c1 (salvador/component_split.py) gives every vertex u the copies (u, 0)
and (u, 1). Edges of the NetworkX minimum maximal matching M of G (the matching
of c2) with both endpoints at index 0 are split off as isolated edges -- the
other edges of an endpoint u move to (u, 1) -- until the split graph H has r'
components with an edge, the largest value not above |M|. The NetworkX minimum
maximal matching of H is decoded: an edge ((a, i), (b, j)) costs nothing when a
or b is already selected; otherwise, for i != j, the rule picks index 0 (low),
index 1 (high) or the endpoint of larger degree in G (deg_low / deg_high, ties
by index), and for i == j the endpoint of larger degree in G. The decoded set is
completed greedily and pruned; c1 keeps the smallest of deg_high, high, deg_low.

Per instance this script records:

* the pruned cover of every rule (low, high, deg_low, deg_high), the cover of c1,
  and |c2| (pruned NetworkX min maximal matching), the baseline with the same
  matching subroutine;
* the split statistics: components r of G and r' of H, the target |M|, the
  number of edges split, whether r' = |M|, and the decoding statistics: |M_H|,
  matched edges with equal indices, and the raw decoded size before completion;
* the quality against tau(G) (exact on the small graphs, Koenig on bipartite
  graphs) or a certified lower bound L <= tau (large graphs), and whether the
  empirical bound |c1| <= 2 tau - r' holds (exact tau only; the bound proved for
  c2 is |c2| <= 2 tau - r);
* with the ensemble (default), whether c1 is strictly smaller than every other
  candidate c2..c16 and whether its refinement is a source of the final cover c17.

Run from the repository root:

    python car/car_c1_components.py --quick --max-n 5000
    python car/car_c1_components.py                       # full sweep, large families up to 50000
    python car/car_c1_components.py --no-ensemble         # c1 and c2 only (much faster)
    python car/car_c1_components.py --dimacs experiment   # also the DIMACS complement graphs
    python car/car_c1_components.py --timeout 60          # skip instances taking more than 60 s

Outputs (in car/): car_c1_components.json, car_c1_components.csv (one row per
instance) and car_c1_components_summary.csv (one row per family).
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
from salvador.algorithm import (CANDIDATE_NAMES, FINAL_NAME, c1_cover, final_candidates,
                                maximal_matching_vertex_cover, prune_redundant_vertices)

RULES = list(component_split.RULES)
OTHERS = [k for k in CANDIDATE_NAMES if k not in ("c1", FINAL_NAME)]  # c2..c16


def is_cover(G: nx.Graph, cover: set) -> bool:
    return all(u in cover or v in cover for u, v in G.edges() if u != v)


def run_instance(name: str, group: str, G: nx.Graph, exact: bool, ensemble: bool, spectral: bool) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    adj = {v: set(G[v]) for v in G}

    t0 = time.perf_counter()
    info: dict = {}
    raw = component_split.component_split_covers(G, info, variants=component_split.RULES)
    rules = {rule: prune_redundant_vertices(adj, raw[rule]) for rule in RULES}
    c1_info: dict = {}
    c1 = c1_cover(G, adj, info=c1_info)
    t_c1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    c2 = prune_redundant_vertices(adj, maximal_matching_vertex_cover(G))
    t_c2 = time.perf_counter() - t0

    for rule, cover in [*rules.items(), ("c1", c1)]:
        if not is_cover(G, cover):
            raise AssertionError(f"{name}: c1 ({rule}) is not a vertex cover")

    if exact or nx.is_bipartite(G):
        tau, method = cc.exact_vertex_cover_size(G)
        lb, lb_method = tau, method
    else:
        lb, lb_method, _ = cc.best_lower_bound(G, spectral=spectral)
        tau = None
    sizes = {rule: len(rules[rule]) for rule in RULES}
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        **{f"c1_{rule}": sizes[rule] for rule in RULES},
        "c1": len(c1), "c1_rule": c1_info["rule"],
        "c2": len(c2), "c1_minus_c2": len(c1) - len(c2),
        "tau": tau, "lower_bound": lb, "lower_bound_method": lb_method,
        **{f"ratio_{rule}": (sizes[rule] / lb) if lb else 1.0 for rule in RULES},
        "ratio_c1": (len(c1) / lb) if lb else 1.0, "ratio_c2": (len(c2) / lb) if lb else 1.0,
        "r": info["r"], "r_prime": info["r_prime"], "target": info["target"], "splits": info["splits"],
        "r_prime_exact": info["exact"],
        **{f"{rule}_{k}": info[rule][k] for rule in RULES for k in ("matching_h", "ties", "raw_size")},
        "bound_2tau_minus_rprime": (2 * tau - info["r_prime"]) if tau is not None else None,
        "bound_holds": (len(c1) <= 2 * tau - info["r_prime"]) if tau is not None else None,
        "seconds_c1": round(t_c1, 4), "seconds_c2": round(t_c2, 4),
    }
    if ensemble:
        t0 = time.perf_counter()
        cands, refined = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, return_refined=True)
        row["seconds_ensemble"] = round(time.perf_counter() - t0, 3)
        best_others = min(len(cands[k]) for k in OTHERS)
        final = len(cands[FINAL_NAME])
        row.update({"best_c2_c16": best_others, "c1_minus_best": len(cands["c1"]) - best_others,
                    "c1_strictly_best": len(cands["c1"]) < best_others, FINAL_NAME: final,
                    "final_from_c1": len(refined["c1"]) == final,
                    "final_only_from_c1": len(refined["c1"]) == final
                    and all(len(refined[k]) > final for k in OTHERS)})
    ref = f"tau={tau}" if tau is not None else f"L={lb}"
    print(f"  [c1] {group[:30]:30} {name:28} n={row['n']:>7,} m={row['m']:>8,} {ref:>10}"
          f"  c1={len(c1):>7,} ({row['c1_rule']}) " + " ".join(f"{r}={sizes[r]:,}" for r in RULES)
          + f" c2={len(c2):>7,} ({row['c1_minus_c2']:+d})  r={row['r']} r'={row['r_prime']} (target {row['target']},"
          f" splits {row['splits']})"
          + (f"  bound {'ok' if row['bound_holds'] else 'VIOLATED'}" if row["bound_holds"] is not None else "")
          + (f"  best c2..c16={row['best_c2_c16']:,} ({row['c1_minus_best']:+d}) c17={row[FINAL_NAME]:,}"
             f"{' <-c1' if row['final_from_c1'] else ''}" if ensemble else "")
          + f"  {t_c1:.2f}s", flush=True)
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
            **{f"best_rule_{rule}": sum(r[f"c1_{rule}"] == r["c1"] for r in rs) for rule in RULES},
            "c1_lt_c2": sum(r["c1_minus_c2"] < 0 for r in rs),
            "c1_eq_c2": sum(r["c1_minus_c2"] == 0 for r in rs),
            "c1_gt_c2": sum(r["c1_minus_c2"] > 0 for r in rs),
            "mean_c1_minus_c2": sum(r["c1_minus_c2"] for r in rs) / k,
            **{f"mean_ratio_{x}": sum(r[f"ratio_{x}"] for r in rs) / k for x in RULES + ["c1", "c2"]},
            **{f"max_ratio_{x}": max(r[f"ratio_{x}"] for r in rs) for x in RULES + ["c1", "c2"]},
            "mean_rprime_over_target": sum(r["r_prime"] / max(r["target"], 1) for r in rs) / k,
            "rprime_exact": sum(r["r_prime_exact"] for r in rs),
            "bound_checked": len(checked), "bound_violations": sum(not r["bound_holds"] for r in checked),
            "seconds_c1": round(sum(r["seconds_c1"] for r in rs), 3),
            "seconds_c2": round(sum(r["seconds_c2"] for r in rs), 3),
        }
        if ensemble:
            s.update({"c1_strictly_best": sum(r["c1_strictly_best"] for r in rs),
                      "final_from_c1": sum(r["final_from_c1"] for r in rs),
                      "final_only_from_c1": sum(r["final_only_from_c1"] for r in rs),
                      "mean_c1_minus_best": sum(r["c1_minus_best"] for r in rs) / k})
        out.append(s)
    return out


def print_table(summary: list[dict], ensemble: bool) -> None:
    head = (f"{'family':44} {'N':>4} {'c1<c2':>6} {'=':>5} {'c1>c2':>6} {'d(c2)':>7} {'ratio c1':>9} {'c2':>7}"
            f" {'max c1':>7} {'max c2':>7} {'r/tgt':>6} {'r=|M|':>6} {'bnd viol':>8}"
            + (f" {'c1 best':>7} {'c17<-c1':>7} {'only c1':>7}" if ensemble else ""))
    print("\nc1 (component amplification, indices 0/1) per family: against c2 and tau (or L)")
    print(head)
    print("-" * len(head))
    for s in summary:
        print(f"{s['family'][:44]:44} {s['instances']:>4} {s['c1_lt_c2']:>6} {s['c1_eq_c2']:>5} {s['c1_gt_c2']:>6}"
              f" {s['mean_c1_minus_c2']:>7.2f} {s['mean_ratio_c1']:>9.4f} {s['mean_ratio_c2']:>7.4f}"
              f" {s['max_ratio_c1']:>7.4f} {s['max_ratio_c2']:>7.4f} {s['mean_rprime_over_target']:>6.2f}"
              f" {s['rprime_exact']:>6} {str(s['bound_violations']) + '/' + str(s['bound_checked']):>8}"
              + (f" {s['c1_strictly_best']:>7} {s['final_from_c1']:>7} {s['final_only_from_c1']:>7}" if ensemble else ""))
    print("\nratio = |cover| / tau on exact/bipartite instances, |cover| / L (certified lower bound) otherwise;"
          " bound = |c1| <= 2 tau - r' (exact tau only); r/tgt = r' / |M|.")
    print("Rule kept by c1 (ties credit every rule): "
          + ", ".join(f"{rule} {summary[-1][f'best_rule_{rule}']}" for rule in RULES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller small-graph sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip the small exact graphs")
    ap.add_argument("--skip-large", action="store_true", help="skip the large adversarial families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest large instance (default 50000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    ap.add_argument("--no-ensemble", action="store_true", help="only c1 and c2 (skip the ensemble)")
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
                    print(f"  [c1] {group[:30]:30} {name:28} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    rows.append(row)
                gc.collect()
        if args.dimacs:
            for path in sorted(Path(args.dimacs).iterdir()):
                if path.is_file():
                    row = runner.run(_dimacs_job, str(path), ensemble, spectral)
                    if row is None:
                        print(f"  [c1] {path.name}: skipped (more than {args.timeout:g} s)", flush=True)
                        skipped.append(path.name)
                    elif row:
                        rows.append(row)

    summary = summarise(rows, ensemble)
    print_table(summary, ensemble)
    if skipped:
        print(f"\nSkipped (more than {args.timeout:g} s): {', '.join(skipped)}")

    out = cc.OUT_DIR
    stem = "car_c1_components"
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
