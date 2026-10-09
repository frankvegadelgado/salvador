"""car/ strategy-winner report: which strategy wins, and how often (Salvador 0.1.0).

``salvador.algorithm.final_candidates`` exposes all twelve strategies (every
one a valid cover); ``salvador.algorithm.find_vertex_cover`` returns ``c12``:

    c1  maximal matching            c7  union re-prune
    c2  max-degree greedy           c8  bipartite planar reduction
    c3  Hallelujah reduction        c9  (1,2)-swap local search
    c4  Min-to-Min                  c10 EDS reduction + treewidth DP (bipartite)
    c5  primal-dual                 c11 bounded-treewidth exact DP
    c6  MIDS Baker reduction        c12 swap-maximize refinement of c1..c11

Two of them are *improvers* rather than independent strategies:

* ``c9`` starts from the best of ``c1``...``c8`` and can only shrink it;
* ``c12`` applies the budgeted swap-maximize refinement to every one of
  ``c1``...``c11`` and keeps the smallest result, so it can only shrink the
  best of them.

``c10`` builds the edge-dominating-set gadget of every bipartite component (one
node per edge; for each vertex a path through its edge-nodes), computes a
minimum EDS with a treewidth DP and decodes each chosen edge ``((u,k),(u,x))``
to ``u``; the DP core and the non-bipartite components are taken from the best
of ``c1``...``c9``, and the reference is kept when the decoded cover is larger.
"c10 exact" means the minimum EDS was computed with an empty core (it does not
mean the decoded cover is minimum); ``c10_raw`` is the decoded cover before the
comparison with the reference. ``c11`` is exact (a minimum vertex cover) when its elimination core is
empty; otherwise it completes the best of ``c1``...``c10`` exactly outside the
core. The report records, per instance, whether c10 and c11 were exact, their
core sizes and widths, and the share of vertices in bipartite components.

So the report gives, per instance:

* the **winners among the base strategies** ``c1``...``c8``, ``c10`` and ``c11``:
  every strategy attaining the minimum (ties credit all of them) and, when
  the minimum is attained once, the *sole* winner;
* whether **c9 strictly improved** on the best of ``c1``...``c8``, and by how much;
* whether **c10 strictly improved** on its reference, the best of ``c1``...``c9``;
* whether **c12 strictly improved** on the best of ``c1``...``c11``, by how
  much, and **from which strategies** the final cover was obtained (every
  strategy whose refined cover attains ``|c12|``).

It aggregates these per graph family: wins, sole wins and win rate per base
strategy, how often (and by how much on average) c9, c10 and c12 improved,
how often c10 and c11 were exact, and how
often each strategy was a source of the final cover.

Instances: the non-adaptive small Part A graphs, the Part B adversarial
families up to ``--max-n`` and, optionally, every edge-list file in a
directory (e.g. ``--dimacs experiment/clique_complement``).

Run from the repository root:

    python car/car_strategies.py
    python car/car_strategies.py --quick --max-n 5000
    python car/car_strategies.py --dimacs experiment/clique_complement
    python car/car_strategies.py --refine-budget -1    # unbounded c12 refinement
    python car/car_strategies.py --tw-max-width 12 --tw-budget 1024   # larger c11 limits
    python car/car_strategies.py --eds-max-width 8 --eds-budget 1024  # larger c10 limits

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
from salvador.algorithm import (BASE_NAMES, C9_INPUT_NAMES, C10_INPUT_NAMES, CANDIDATE_NAMES,
                                DEFAULT_EDS_BUDGET, DEFAULT_EDS_MAX_WIDTH, DEFAULT_REFINE_BUDGET,
                                final_candidates)
from salvador.treewidth_dp import DEFAULT_DP_BUDGET, DEFAULT_MAX_WIDTH

BASE = list(BASE_NAMES)                                      # c1..c8, c11
C9_INPUT = list(C9_INPUT_NAMES)                              # c1..c8 (input of c9)
C10_INPUT = list(C10_INPUT_NAMES)                            # c1..c9 (reference of c10)
PRE_FINAL = [k for k in CANDIDATE_NAMES if k != "c12"]       # c1..c11 (input of c12)

# Set from the command line in main().
REFINE_BUDGET: float | None = DEFAULT_REFINE_BUDGET
TW_MAX_WIDTH: int = DEFAULT_MAX_WIDTH
TW_BUDGET: float = DEFAULT_DP_BUDGET
EDS_MAX_WIDTH: int = DEFAULT_EDS_MAX_WIDTH
EDS_BUDGET: float = DEFAULT_EDS_BUDGET


def run_instance(name: str, group: str, G: nx.Graph) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    t0 = time.perf_counter()
    info: dict = {}
    cands, refined = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, refine_budget=REFINE_BUDGET,
                                      return_refined=True, info=info,
                                      tw_max_width=TW_MAX_WIDTH, tw_budget=TW_BUDGET,
                                      eds_max_width=EDS_MAX_WIDTH, eds_budget=EDS_BUDGET)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in CANDIDATE_NAMES}
    for k in CANDIDATE_NAMES:
        v = cands[k]
        if not all(a in v or b in v for a, b in G.edges() if a != b):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
    best = min(sizes[k] for k in BASE)
    winners = [k for k in BASE if sizes[k] == best]
    best_legacy = min(sizes[k] for k in C9_INPUT)
    best_pre_final = min(sizes[k] for k in PRE_FINAL)
    refined_sizes = {k: len(refined[k]) for k in PRE_FINAL}
    c12_from = [k for k in PRE_FINAL if refined_sizes[k] == sizes["c12"]]
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "sizes": sizes, "refined_sizes": refined_sizes,
        "best_base": best, "winners": winners,
        "sole_winner": winners[0] if len(winners) == 1 else None,
        "best_c1_c8": best_legacy, "c9_gain": best_legacy - sizes["c9"],
        "final": sizes["c12"],
        "best_c1_c11": best_pre_final, "c12_gain": best_pre_final - sizes["c12"], "c12_from": c12_from,
        "best_c1_c9": min(sizes[k] for k in C10_INPUT),
        "c10_gain": min(sizes[k] for k in C10_INPUT) - sizes["c10"],
        "c10_exact": bool(info.get("c10_exact", True)),
        "c10_raw": info.get("c10_raw_size", sizes["c10"]),
        "c10_core_size": info.get("c10_core_size", 0),
        "c10_width": info.get("c10_width", 0),
        "c10_bipartite_fraction": info.get("c10_bipartite_fraction", 1.0),
        "c11_exact": bool(info.get("c11_exact", True)),
        "c11_core_size": info.get("c11_core_size", 0),
        "c11_width": info.get("c11_width", 0),
        "elapsed_seconds": elapsed,
    }
    print(f"  [strategies] {group[:28]:28} {name:28} best={best:>7,} by {','.join(winners):28}"
          f" c9=-{row['c9_gain']:<5} c10=-{row['c10_gain']:<5} c10raw={row['c10_raw']:>7,} final={sizes['c12']:>7,} (-{row['c12_gain']}"
          f" from {','.join(c12_from)})"
          f" c10={'exact' if row['c10_exact'] else ('core=' + str(row['c10_core_size']) if row['c10_bipartite_fraction'] == 1.0 else 'bip=' + format(row['c10_bipartite_fraction'], '.2f'))}"
          f" c11={'exact' if row['c11_exact'] else 'core=' + str(row['c11_core_size'])}"
          f" w={row['c11_width']}", flush=True)
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
        final_source = {s: sum(s in r["c12_from"] for r in grows) for s in PRE_FINAL}
        imp9 = [r for r in grows if r["c9_gain"] > 0]
        imp12 = [r for r in grows if r["c12_gain"] > 0]
        imp10 = [r for r in grows if r["c10_gain"] > 0]
        out[g] = {
            "instances": k,
            "per_strategy": per,
            "c9_improved": len(imp9),
            "c9_improved_rate": len(imp9) / k if k else 0.0,
            "c9_mean_gain_when_improved": (sum(r["c9_gain"] for r in imp9) / len(imp9)) if imp9 else 0.0,
            "c9_total_gain": sum(r["c9_gain"] for r in grows),
            "c12_improved": len(imp12),
            "c12_improved_rate": len(imp12) / k if k else 0.0,
            "c12_mean_gain_when_improved": (sum(r["c12_gain"] for r in imp12) / len(imp12)) if imp12 else 0.0,
            "c12_total_gain": sum(r["c12_gain"] for r in grows),
            "c12_source_counts": final_source,
            "c10_improved": len(imp10),
            "c10_improved_rate": len(imp10) / k if k else 0.0,
            "c10_mean_gain_when_improved": (sum(r["c10_gain"] for r in imp10) / len(imp10)) if imp10 else 0.0,
            "c10_total_gain": sum(r["c10_gain"] for r in grows),
            "c10_exact": sum(r["c10_exact"] for r in grows),
            "c10_raw_best": sum(r["c10_raw"] <= r["best_base"] for r in grows),
            "c10_raw_mean_excess": (sum(r["c10_raw"] - r["best_base"] for r in grows) / k) if k else 0.0,
            "c10_exact_rate": (sum(r["c10_exact"] for r in grows) / k) if k else 0.0,
            "c10_bipartite_instances": sum(r["c10_bipartite_fraction"] == 1.0 for r in grows),
            "c10_mean_core_fraction": (sum(r["c10_core_size"] / r["n"] for r in grows if r["n"]) / k) if k else 0.0,
            "c10_max_width": max((r["c10_width"] for r in grows), default=0),
            "c11_exact": sum(r["c11_exact"] for r in grows),
            "c11_exact_rate": (sum(r["c11_exact"] for r in grows) / k) if k else 0.0,
            "c11_mean_core_fraction": (sum(r["c11_core_size"] / r["n"] for r in grows if r["n"]) / k) if k else 0.0,
            "c11_max_width": max((r["c11_width"] for r in grows), default=0),
        }
    return out


def print_table(summary: dict) -> None:
    header = (f"{'family':44} {'inst':>5} " + " ".join(f"{s:>5}" for s in BASE)
              + "  c9-impr     c10-impr    c12-impr    c10-exact   c10raw=best c11-exact")
    print("\nWins among the base strategies c1..c8, c10, c11 "
          "(ties credit every tied strategy; [sole wins] in the second line)")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        per = s["per_strategy"]
        print(f"{g[:44]:44} {s['instances']:>5} " + " ".join(f"{per[k]['wins']:>5}" for k in BASE)
              + f"  {s['c9_improved']:>4} ({100 * s['c9_improved_rate']:3.0f}%)"
              + f"  {s['c10_improved']:>4} ({100 * s['c10_improved_rate']:3.0f}%)"
              + f"  {s['c12_improved']:>4} ({100 * s['c12_improved_rate']:3.0f}%)"
              + f"  {s['c10_exact']:>4} ({100 * s['c10_exact_rate']:3.0f}%)"
              + f"  {s['c10_raw_best']:>4} ({100 * s['c10_raw_best'] / max(s['instances'], 1):3.0f}%)"
              + f"  {s['c11_exact']:>4} ({100 * s['c11_exact_rate']:3.0f}%)")
        print(f"{'':44} {'':>5} " + " ".join(f"{'[' + str(per[k]['sole_wins']) + ']':>5}" for k in BASE))
    print("\nSource of the final cover c12 (number of instances on which the refinement of"
          " each strategy attains |c12|; ties credit every source)")
    header2 = f"{'family':44} {'inst':>5} " + " ".join(f"{s:>5}" for s in PRE_FINAL)
    print(header2)
    print("-" * len(header2))
    for g, s in summary.items():
        src = s["c12_source_counts"]
        print(f"{g[:44]:44} {s['instances']:>5} " + " ".join(f"{src[k]:>5}" for k in PRE_FINAL))
    print("\nLegend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def main() -> None:
    global REFINE_BUDGET, TW_MAX_WIDTH, TW_BUDGET, EDS_MAX_WIDTH, EDS_BUDGET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller small-graph sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip the small Part A graphs")
    ap.add_argument("--skip-large", action="store_true", help="skip the Part B adversarial families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest Part B instance (default 50000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    ap.add_argument("--refine-budget", type=float, default=DEFAULT_REFINE_BUDGET,
                    help="work budget of the c12 refinement in units of (n + m) per cover "
                         f"(default {DEFAULT_REFINE_BUDGET}; -1 = unbounded)")
    ap.add_argument("--tw-max-width", type=int, default=DEFAULT_MAX_WIDTH,
                    help=f"c11: largest elimination bag minus one (default {DEFAULT_MAX_WIDTH})")
    ap.add_argument("--tw-budget", type=float, default=DEFAULT_DP_BUDGET,
                    help=f"c11: DP table budget in units of (n + m) (default {DEFAULT_DP_BUDGET})")
    ap.add_argument("--eds-max-width", type=int, default=DEFAULT_EDS_MAX_WIDTH,
                    help=f"c10: largest elimination bag minus one (default {DEFAULT_EDS_MAX_WIDTH})")
    ap.add_argument("--eds-budget", type=float, default=DEFAULT_EDS_BUDGET,
                    help=f"c10: DP table budget in units of (n + m) (default {DEFAULT_EDS_BUDGET})")
    args = ap.parse_args()
    REFINE_BUDGET = None if args.refine_budget < 0 else args.refine_budget
    TW_MAX_WIDTH = args.tw_max_width
    TW_BUDGET = args.tw_budget
    EDS_MAX_WIDTH = args.eds_max_width
    EDS_BUDGET = args.eds_budget

    started = time.time()
    rows: list[dict] = []
    if not args.skip_small:
        rng = random.Random(cc.SEED)
        for name, group, G in cc.small_instances(rng, big=not args.quick):
            rows.append(run_instance(name, group, G))
    if not args.skip_large:
        for name, group, build in cc.large_instances(args.max_n):
            rows.append(run_instance(name, group, build()))
            gc.collect()
    if args.dimacs:
        for path in sorted(Path(args.dimacs).iterdir()):
            if path.is_file():
                G = cc.load_edge_list(path)
                if G.number_of_edges():
                    rows.append(run_instance(path.name.split(".clq")[0], "DIMACS / edge-list files", G))

    summary = summarise(rows)
    out = cc.OUT_DIR
    with (out / "car_strategies.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,n,m," + ",".join(CANDIDATE_NAMES)
                 + ",best_base,winners,sole_winner,c9_gain,c10_gain,c12_gain,c12_from,"
                 + "c10_exact,c10_raw,c10_core_size,c10_width,c10_bipartite_fraction,"
                 + "c11_exact,c11_core_size,c11_width,elapsed_seconds\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},"
                     + ",".join(str(r["sizes"][k]) for k in CANDIDATE_NAMES)
                     + f",{r['best_base']},{'+'.join(r['winners'])},{r['sole_winner'] or ''},"
                     + f"{r['c9_gain']},{r['c10_gain']},{r['c12_gain']},{'+'.join(r['c12_from'])},"
                     + f"{r['c10_exact']},{r['c10_raw']},{r['c10_core_size']},{r['c10_width']},{r['c10_bipartite_fraction']:.4f},"
                     + f"{r['c11_exact']},{r['c11_core_size']},{r['c11_width']},{r['elapsed_seconds']:.3f}\n")
    with (out / "car_strategies_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,strategy,name,wins,sole_wins,win_rate,final_source\n")
        for g, s in summary.items():
            src = s["c12_source_counts"]
            for k in BASE:
                p = s["per_strategy"][k]
                fh.write(f"\"{g}\",{s['instances']},{k},\"{CANDIDATE_NAMES[k]}\","
                         f"{p['wins']},{p['sole_wins']},{p['win_rate']:.4f},{src[k]}\n")
            fh.write(f"\"{g}\",{s['instances']},c9,\"{CANDIDATE_NAMES['c9']} (strict improvements)\","
                     f"{s['c9_improved']},,{s['c9_improved_rate']:.4f},{src['c9']}\n")
            fh.write(f"\"{g}\",{s['instances']},c12,\"{CANDIDATE_NAMES['c12']} (strict improvements)\","
                     f"{s['c12_improved']},,{s['c12_improved_rate']:.4f},\n")
    (out / "car_strategies.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(),
         "candidates": CANDIDATE_NAMES, "base_strategies": BASE,
         "refine_budget": REFINE_BUDGET, "tw_max_width": TW_MAX_WIDTH, "tw_budget": TW_BUDGET,
         "eds_max_width": EDS_MAX_WIDTH, "eds_budget": EDS_BUDGET,
         "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
