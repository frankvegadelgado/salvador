"""car/ strategy-winner report: which strategy wins, and how often (Salvador 0.1.4).

Salvador's ensemble computes seventeen strategies in their numbering order (each
one a valid cover); ``salvador.algorithm.find_vertex_cover`` returns ``c17``:

    base strategies                                 improvers
    c1  component amplification (indices 0/1)       c13 union re-prune of c2..c8
    c2  maximal matching                            c14 (1,2)-swap local search, from the best of c1..c13
    c3  max-degree greedy                           c15 reduce-and-search, from the best of c1..c14
    c4  Hallelujah reduction                        c16 consensus forcing (votes of c1..c15)
    c5  Min-to-Min                                  c17 swap-maximize refinement of c1..c16 (final)
    c6  primal-dual
    c7  MIDS Baker reduction
    c8  bipartite planar reduction
    c9  bipartite layers (bounded Hopcroft-Karp + Koenig)
    c10 EDS reduction + treewidth DP (bipartite), reference: best of c1..c9
    c11 weighted subdivision reduction, guided by the best of c1..c10
    c12 bounded-treewidth exact DP, core from the best of c1..c11

The report gives, per instance:

* the **winners among the base strategies** c1..c12: every strategy attaining
  the minimum (ties credit all of them) and, when the minimum is attained once,
  the *sole* winner;
* for each of c10, c12, c13, c14, c15, c16 and c17, its **gain** over its
  reference (the best of the strategies before it, see ``REFERENCE``): positive
  when it strictly improved, 0 when it tied, negative when it is larger (only
  c11 and c13 can be larger than the best before them; c10 and c12 keep their
  reference when they are larger, c14 and c15 their start);
* **from which strategies** the final cover was obtained (every strategy whose
  refined cover attains ``|c17|``);
* the diagnostics of c1 (``r'``, the target ``|M|``, the rule kept), c9 (layers,
  orders, exact), c10 (exact, raw size, core, width, bipartite share), c11
  (exact, variant, reductions, folds and peels, gap to the best base strategy),
  c12 (exact, core, width), c15 (search and reduction gains) and c16 (rounds,
  forced vertices, lower bound, certified ratio |C|/L).

It aggregates these per graph family: wins, sole wins and win rate per base
strategy, how often (and by how much in total) each improver improved, how often
c9, c10, c11 and c12 were exact, the mean c11 peel share, and how often each
strategy was a source of the final cover.

Instances: the non-adaptive small Part A graphs, the Part B adversarial
families up to ``--max-n`` and, optionally, every edge-list file in a
directory (e.g. ``--dimacs experiment``).

Run from the repository root:

    python car/car_strategies.py
    python car/car_strategies.py --quick --max-n 5000
    python car/car_strategies.py --dimacs experiment
    python car/car_strategies.py --refine-budget -1    # unbounded c17 refinement
    python car/car_strategies.py --tw-max-width 12 --tw-budget 1024   # larger c12 limits
    python car/car_strategies.py --eds-max-width 5 --eds-budget 1024  # larger c10 budget (bags <= 6)
    python car/car_strategies.py --search-rounds 5 --search-budget 200  # longer c15 search
    python car/car_strategies.py --timeout 60    # skip instances taking more than 60 s

The Part B families and the edge-list files are built and evaluated in a worker
process with a time limit (``--timeout``, default 120 s, 0 = no limit): an
instance whose construction or evaluation takes longer is skipped and listed
in car_strategies.json (``skipped_instances``).

Outputs (in car/): car_strategies.json, car_strategies.csv (one row per
instance) and car_strategies_summary.csv (one row per family x strategy).
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
from salvador.algorithm import (BASE_NAMES, CANDIDATE_NAMES, DEFAULT_EDS_BUDGET, DEFAULT_EDS_MAX_WIDTH,
                                DEFAULT_REFINE_BUDGET, DEFAULT_SEARCH_BUDGET, DEFAULT_SEARCH_ROUNDS, FINAL_NAME,
                                final_candidates)
from salvador.treewidth_dp import DEFAULT_DP_BUDGET, DEFAULT_MAX_WIDTH

BASE = list(BASE_NAMES)                                      # c1..c12
PRE_FINAL = [k for k in CANDIDATE_NAMES if k != FINAL_NAME]  # c1..c16 (input of c17)
#: Every strategy that uses or improves the earlier covers, with the strategies it is compared with.
REFERENCE: dict[str, list[str]] = {k: [f"c{i}" for i in range(1, int(k[1:]))]
                                   for k in ("c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17")}
IMPROVERS = list(REFERENCE)

# Set from the command line in main().
REFINE_BUDGET: float | None = DEFAULT_REFINE_BUDGET
TW_MAX_WIDTH: int = DEFAULT_MAX_WIDTH
TW_BUDGET: float = DEFAULT_DP_BUDGET
EDS_MAX_WIDTH: int = DEFAULT_EDS_MAX_WIDTH
EDS_BUDGET: float = DEFAULT_EDS_BUDGET
SEARCH_ROUNDS: int = DEFAULT_SEARCH_ROUNDS
SEARCH_BUDGET: float = DEFAULT_SEARCH_BUDGET


def _set_limits(limits: dict) -> None:
    """Apply the command-line limits (the worker process starts with the module defaults)."""
    global REFINE_BUDGET, TW_MAX_WIDTH, TW_BUDGET, EDS_MAX_WIDTH, EDS_BUDGET, SEARCH_ROUNDS, SEARCH_BUDGET
    REFINE_BUDGET = limits["refine_budget"]
    SEARCH_ROUNDS = limits["search_rounds"]
    SEARCH_BUDGET = limits["search_budget"]
    TW_MAX_WIDTH = limits["tw_max_width"]
    TW_BUDGET = limits["tw_budget"]
    EDS_MAX_WIDTH = limits["eds_max_width"]
    EDS_BUDGET = limits["eds_budget"]


def _large_job(name: str, max_n: int, limits: dict) -> dict:
    """Build Part B instance ``name`` and evaluate it; runs in the worker process."""
    _set_limits(limits)
    for nm, group, build in cc.large_instances(max_n):
        if nm == name:
            return run_instance(name, group, build())
    raise KeyError(name)


def _dimacs_job(path: str, limits: dict) -> dict:
    """Load an edge-list file and evaluate it (an empty dict for a file without edges); runs in the
    worker process."""
    _set_limits(limits)
    G = cc.load_edge_list(Path(path))
    if not G.number_of_edges():
        return {}
    return run_instance(Path(path).name.split(".clq")[0], "DIMACS / edge-list files", G)


def run_instance(name: str, group: str, G: nx.Graph) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    t0 = time.perf_counter()
    info: dict = {}
    cands, refined = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, refine_budget=REFINE_BUDGET,
                                      return_refined=True, info=info,
                                      tw_max_width=TW_MAX_WIDTH, tw_budget=TW_BUDGET,
                                      eds_max_width=EDS_MAX_WIDTH, eds_budget=EDS_BUDGET,
                                      search_rounds=SEARCH_ROUNDS, search_budget=SEARCH_BUDGET)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in CANDIDATE_NAMES}
    for k in CANDIDATE_NAMES:
        v = cands[k]
        if not all(a in v or b in v for a, b in G.edges() if a != b):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
    if info.get("c11_undominated", 0):
        raise AssertionError(f"{name}: c11 left {info['c11_undominated']} edge-nodes undominated")
    best = min(sizes[k] for k in BASE)
    winners = [k for k in BASE if sizes[k] == best]
    refined_sizes = {k: len(refined[k]) for k in PRE_FINAL}
    final = sizes[FINAL_NAME]
    gains = {k: min(sizes[x] for x in REFERENCE[k]) - sizes[k] for k in IMPROVERS}
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "sizes": sizes, "refined_sizes": refined_sizes,
        "best_base": best, "winners": winners,
        "sole_winner": winners[0] if len(winners) == 1 else None,
        "final": final, "gains": gains,
        "final_from": [k for k in PRE_FINAL if refined_sizes[k] == final],
        "c1_rule": info.get("c1_rule", ""), "c1_r_prime": info.get("c1_r_prime", 0),
        "c1_target": info.get("c1_target", 0), "c1_splits": info.get("c1_splits", 0),
        "c9_layers": info.get("c9_max_layers", 0), "c9_orders": info.get("c9_orders", 0),
        "c9_exact": bool(info.get("c9_exact", False)),
        "c9_bipartite": info.get("c9_nonbipartite_components", 1) == 0,
        "c10_exact": bool(info.get("c10_exact", True)),
        "c10_raw": info.get("c10_raw_size", sizes["c10"]),
        "c10_core_size": info.get("c10_core_size", 0),
        "c10_width": info.get("c10_width", 0),
        "c10_bipartite_fraction": info.get("c10_bipartite_fraction", 1.0),
        "c11_exact": bool(info.get("c11_exact", False)),
        "c11_deg0": info.get("c11_deg0", 0),
        "c11_deg1": info.get("c11_deg1", 0),
        "c11_triangle": info.get("c11_triangle", 0),
        "c11_folds": info.get("c11_folds", 0),
        "c11_variant": info.get("c11_variant", ""),
        "c11_peels": info.get("c11_peels", 0),
        "c11_peel_fraction": info.get("c11_peels", 0) / max(G.number_of_nodes(), 1),
        "c11_gap": sizes["c11"] - best,
        "c12_exact": bool(info.get("c12_exact", True)),
        "c12_core_size": info.get("c12_core_size", 0),
        "c12_width": info.get("c12_width", 0),
        "c15_search_gain": info.get("c15_search_gain", 0),
        "c15_reduce_gain": info.get("c15_reduce_gain", 0),
        "c15_rounds_improved": info.get("c15_rounds_improved", 0),
        "c16_rounds_run": info.get("c16_rounds_run", 0),
        "c16_forced": info.get("c16_forced", 0),
        "c16_lower_bound": info.get("c16_lower_bound", 0),
        "c16_certified_ratio": info.get("c16_certified_ratio", 0.0),
        "elapsed_seconds": elapsed,
    }
    print(f"  [strategies] {group[:28]:28} {name:28} best base={best:>7,} by {','.join(winners):24}"
          f" final={final:>7,} from {','.join(row['final_from'])}"
          f" gains " + " ".join(f"{k}={gains[k]:+d}" for k in IMPROVERS)
          + f" c1: r'={row['c1_r_prime']}/{row['c1_target']} c9: layers={row['c9_layers']}"
          f"{' exact' if row['c9_exact'] else ''}"
          f" c10={'exact' if row['c10_exact'] else 'core=' + str(row['c10_core_size'])}"
          f" c11={'exact' if row['c11_exact'] else 'peels=' + str(row['c11_peels'])}"
          f" c12={'exact' if row['c12_exact'] else 'core=' + str(row['c12_core_size'])}"
          f" c16 |C|/L={row['c16_certified_ratio']:.4f}", flush=True)
    return row


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(grows)
        per = {}
        for s in BASE:
            wins = sum(s in r["winners"] for r in grows)
            sole = sum(r["sole_winner"] == s for r in grows)
            per[s] = {"wins": wins, "sole_wins": sole, "win_rate": wins / k if k else 0.0}
        improvers = {}
        for s in IMPROVERS:
            imp = [r["gains"][s] for r in grows if r["gains"][s] > 0]
            improvers[s] = {"improved": len(imp), "improved_rate": len(imp) / k if k else 0.0,
                            "ties": sum(r["gains"][s] == 0 for r in grows),
                            "worse": sum(r["gains"][s] < 0 for r in grows),
                            "total_gain": sum(imp), "mean_gain_when_improved": sum(imp) / len(imp) if imp else 0.0}
        out[g] = {
            "instances": k,
            "per_strategy": per,
            "improvers": improvers,
            "final_source_counts": {s: sum(s in r["final_from"] for r in grows) for s in PRE_FINAL},
            "c1_r_prime_equals_target": sum(r["c1_r_prime"] == r["c1_target"] for r in grows),
            "c9_exact": sum(r["c9_exact"] for r in grows),
            "c9_mean_layers": (sum(r["c9_layers"] for r in grows) / k) if k else 0.0,
            "c9_max_layers": max((r["c9_layers"] for r in grows), default=0),
            "c10_exact": sum(r["c10_exact"] for r in grows),
            "c10_raw_best": sum(r["c10_raw"] <= r["best_base"] for r in grows),
            "c10_bipartite_instances": sum(r["c10_bipartite_fraction"] == 1.0 for r in grows),
            "c10_max_width": max((r["c10_width"] for r in grows), default=0),
            "c11_exact": sum(r["c11_exact"] for r in grows),
            "c11_mean_peel_fraction": (sum(r["c11_peel_fraction"] for r in grows) / k) if k else 0.0,
            "c11_mean_gap": (sum(r["c11_gap"] for r in grows) / k) if k else 0.0,
            "c12_exact": sum(r["c12_exact"] for r in grows),
            "c12_max_width": max((r["c12_width"] for r in grows), default=0),
            "c15_search_gain": sum(r["c15_search_gain"] for r in grows),
            "c15_reduce_gain": sum(r["c15_reduce_gain"] for r in grows),
            "c16_mean_rounds": (sum(r["c16_rounds_run"] for r in grows) / k) if k else 0.0,
            "max_certified_ratio": max((r["c16_certified_ratio"] for r in grows), default=0.0),
        }
    return out


def print_table(summary: dict) -> None:
    header = f"{'family':44} {'inst':>5} " + " ".join(f"{s:>5}" for s in BASE)
    print("\nWins among the base strategies c1..c12 (ties credit every tied strategy; [sole wins] in the second line)")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        per = s["per_strategy"]
        print(f"{g[:44]:44} {s['instances']:>5} " + " ".join(f"{per[k]['wins']:>5}" for k in BASE))
        print(f"{'':44} {'':>5} " + " ".join(f"{'[' + str(per[k]['sole_wins']) + ']':>5}" for k in BASE))
    print("\nStrategies after c9: instances on which each one is strictly smaller than the best of the strategies"
          " before it [total vertices saved]")
    header = f"{'family':44} {'inst':>5} " + " ".join(f"{s:>12}" for s in IMPROVERS) + \
        "  c9ex c10ex c11ex c12ex  c9 layers  max|C|/L"
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        imp = s["improvers"]
        print(f"{g[:44]:44} {s['instances']:>5} "
              + " ".join(f"{imp[k]['improved']:>5} [{imp[k]['total_gain']:>4}]" for k in IMPROVERS)
              + f"  {s['c9_exact']:>4} {s['c10_exact']:>5} {s['c11_exact']:>5} {s['c12_exact']:>5}"
              + f"  {s['c9_mean_layers']:>9.2f} {s['max_certified_ratio']:>9.4f}")
    print("\nSource of the final cover c17 (number of instances on which the refinement of"
          " each strategy attains |c17|; ties credit every source)")
    header2 = f"{'family':44} {'inst':>5} " + " ".join(f"{s:>5}" for s in PRE_FINAL)
    print(header2)
    print("-" * len(header2))
    for g, s in summary.items():
        src = s["final_source_counts"]
        print(f"{g[:44]:44} {s['instances']:>5} " + " ".join(f"{src[k]:>5}" for k in PRE_FINAL))
    print("\nLegend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def main() -> None:
    global REFINE_BUDGET, TW_MAX_WIDTH, TW_BUDGET, EDS_MAX_WIDTH, EDS_BUDGET, SEARCH_ROUNDS, SEARCH_BUDGET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller small-graph sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip the small Part A graphs")
    ap.add_argument("--skip-large", action="store_true", help="skip the Part B adversarial families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest Part B instance (default 50000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    ap.add_argument("--refine-budget", type=float, default=DEFAULT_REFINE_BUDGET,
                    help="work budget of the c17 refinement in units of (n + m) per cover "
                         f"(default {DEFAULT_REFINE_BUDGET}; -1 = unbounded)")
    ap.add_argument("--tw-max-width", type=int, default=DEFAULT_MAX_WIDTH,
                    help=f"c12: largest elimination bag minus one (default {DEFAULT_MAX_WIDTH})")
    ap.add_argument("--tw-budget", type=float, default=DEFAULT_DP_BUDGET,
                    help=f"c12: DP table budget in units of (n + m) (default {DEFAULT_DP_BUDGET})")
    ap.add_argument("--eds-max-width", type=int, default=DEFAULT_EDS_MAX_WIDTH,
                    help=f"c10: largest elimination bag minus one (default {DEFAULT_EDS_MAX_WIDTH})")
    ap.add_argument("--search-rounds", type=int, default=DEFAULT_SEARCH_ROUNDS,
                    help=f"c15: search + reduction rounds (default {DEFAULT_SEARCH_ROUNDS})")
    ap.add_argument("--search-budget", type=float, default=DEFAULT_SEARCH_BUDGET,
                    help=f"c15: local-search work per round in units of (n + m) (default {DEFAULT_SEARCH_BUDGET})")
    ap.add_argument("--eds-budget", type=float, default=DEFAULT_EDS_BUDGET,
                    help=f"c10: DP table budget in units of (n + m) (default {DEFAULT_EDS_BUDGET})")
    cc.add_timeout_argument(ap)
    args = ap.parse_args()
    REFINE_BUDGET = None if args.refine_budget < 0 else args.refine_budget
    TW_MAX_WIDTH = args.tw_max_width
    TW_BUDGET = args.tw_budget
    EDS_MAX_WIDTH = args.eds_max_width
    EDS_BUDGET = args.eds_budget
    SEARCH_ROUNDS = args.search_rounds
    SEARCH_BUDGET = args.search_budget

    started = time.time()
    rows: list[dict] = []
    if not args.skip_small:
        rng = random.Random(cc.SEED)
        for name, group, G in cc.small_instances(rng, big=not args.quick):
            rows.append(run_instance(name, group, G))
    limits = {"refine_budget": REFINE_BUDGET, "tw_max_width": TW_MAX_WIDTH, "tw_budget": TW_BUDGET,
              "eds_max_width": EDS_MAX_WIDTH, "eds_budget": EDS_BUDGET,
              "search_rounds": SEARCH_ROUNDS, "search_budget": SEARCH_BUDGET}
    skipped: list[str] = []
    with cc.InstanceRunner(args.timeout) as runner:
        if not args.skip_large:
            for name, group, _ in cc.large_instances(args.max_n):
                row = runner.run(_large_job, name, args.max_n, limits)
                if row is None:
                    print(f"  [strategies] {group[:28]:28} {name:28} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    rows.append(row)
                gc.collect()
        if args.dimacs:
            for path in sorted(Path(args.dimacs).iterdir()):
                if path.is_file():
                    row = runner.run(_dimacs_job, str(path), limits)
                    if row is None:
                        print(f"  [strategies] {path.name}: skipped (more than {args.timeout:g} s)", flush=True)
                        skipped.append(path.name)
                    elif row:
                        rows.append(row)

    summary = summarise(rows)
    out = cc.OUT_DIR
    diag = ["c1_rule", "c1_r_prime", "c1_target", "c1_splits", "c9_layers", "c9_orders", "c9_exact", "c10_exact",
            "c10_raw", "c10_core_size", "c10_width", "c10_bipartite_fraction", "c11_exact", "c11_variant", "c11_deg0",
            "c11_deg1", "c11_triangle", "c11_folds", "c11_peels", "c11_peel_fraction", "c11_gap", "c12_exact",
            "c12_core_size", "c12_width", "c15_search_gain", "c15_reduce_gain", "c15_rounds_improved",
            "c16_rounds_run", "c16_forced", "c16_lower_bound", "c16_certified_ratio", "elapsed_seconds"]
    with (out / "car_strategies.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,n,m," + ",".join(CANDIDATE_NAMES) + ",best_base,winners,sole_winner,final_from,"
                 + ",".join(f"{k}_gain" for k in IMPROVERS) + "," + ",".join(diag) + "\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},"
                     + ",".join(str(r["sizes"][k]) for k in CANDIDATE_NAMES)
                     + f",{r['best_base']},{'+'.join(r['winners'])},{r['sole_winner'] or ''},{'+'.join(r['final_from'])},"
                     + ",".join(str(r["gains"][k]) for k in IMPROVERS) + ","
                     + ",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k]) for k in diag) + "\n")
    with (out / "car_strategies_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,strategy,name,wins_or_improved,sole_wins,rate,total_gain,final_source\n")
        for g, s in summary.items():
            src = s["final_source_counts"]
            for k in BASE:
                p = s["per_strategy"][k]
                fh.write(f"\"{g}\",{s['instances']},{k},\"{CANDIDATE_NAMES[k]}\","
                         f"{p['wins']},{p['sole_wins']},{p['win_rate']:.4f},,{src[k]}\n")
            for k in IMPROVERS:
                p = s["improvers"][k]
                fh.write(f"\"{g}\",{s['instances']},{k},\"{CANDIDATE_NAMES[k]} (strict improvements over "
                         f"c1..c{int(k[1:]) - 1})\",{p['improved']},,{p['improved_rate']:.4f},{p['total_gain']},"
                         f"{src.get(k, '')}\n")
    (out / "car_strategies.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(),
         "candidates": CANDIDATE_NAMES, "base_strategies": BASE,
         "refine_budget": REFINE_BUDGET, "tw_max_width": TW_MAX_WIDTH, "tw_budget": TW_BUDGET,
         "eds_max_width": EDS_MAX_WIDTH, "eds_budget": EDS_BUDGET,
         "search_rounds": SEARCH_ROUNDS, "search_budget": SEARCH_BUDGET,
         "timeout": args.timeout, "skipped_instances": skipped,
         "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")
    print_table(summary)
    if skipped:
        print(f"\nSkipped (more than {args.timeout:g} s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
