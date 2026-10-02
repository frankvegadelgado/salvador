"""car/ strategy-winner report: which ensemble candidate wins, and how often.

``salvador.algorithm.find_vertex_cover`` returns the smallest of nine
candidate covers (all pruned, all valid):

    c1 maximal matching           c6 MIDS Baker reduction
    c2 max-degree greedy          c7 union re-prune
    c3 Hallelujah reduction       c8 bipartite planar reduction
    c4 Min-to-Min                 c9 (1,2)-swap local search
    c5 primal-dual

``c9`` starts from the best of ``c1``...``c8`` and can only shrink it, so
it would trivially "win" every instance. This script therefore reports two
things per instance:

* the **winners among c1..c8** -- every candidate attaining the minimum
  (ties credit all of them) and, when the minimum is attained once, the
  *sole* winner;
* whether **c9 strictly improved** on that minimum, and by how much.

It aggregates these per graph family: wins, sole wins and win rate per
strategy, plus how often (and by how much on average) c9 improved.

Instances: the non-adaptive small Part A graphs, the Part B adversarial
families up to ``--max-n`` and, optionally, every edge-list file in a
directory (e.g. ``--dimacs experiment/clique_complement``).

Run from the repository root:

    python car/car_strategies.py
    python car/car_strategies.py --quick --max-n 5000
    python car/car_strategies.py --dimacs experiment/clique_complement

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
from salvador.algorithm import CANDIDATE_NAMES, ensemble_candidates

BASE = [k for k in CANDIDATE_NAMES if k != "c9"]


def run_instance(name: str, group: str, G: nx.Graph) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    t0 = time.perf_counter()
    cands = ensemble_candidates(G, epsilon=cc.DEFAULT_EPSILON)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(v) for k, v in cands.items()}
    for k, v in cands.items():
        if not all(a in v or b in v for a, b in G.edges()):
            raise AssertionError(f"{name}: candidate {k} is not a vertex cover")
    best = min(sizes[k] for k in BASE)
    winners = [k for k in BASE if sizes[k] == best]
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "sizes": sizes, "best_c1_c8": best, "winners": winners,
        "sole_winner": winners[0] if len(winners) == 1 else None,
        "final": sizes["c9"], "c9_gain": best - sizes["c9"], "elapsed_seconds": elapsed,
    }
    print(f"  [strategies] {group[:28]:28} {name:28} best={best:>7,} by {','.join(winners):24}"
          f" c9={sizes['c9']:>7,} (-{row['c9_gain']})", flush=True)
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
        improved = [r for r in grows if r["c9_gain"] > 0]
        out[g] = {
            "instances": k,
            "per_strategy": per,
            "c9_improved": len(improved),
            "c9_improved_rate": len(improved) / k if k else 0.0,
            "c9_mean_gain_when_improved": (sum(r["c9_gain"] for r in improved) / len(improved)) if improved else 0.0,
            "c9_total_gain": sum(r["c9_gain"] for r in grows),
        }
    return out


def print_table(summary: dict) -> None:
    header = f"{'family':52} {'inst':>5} " + " ".join(f"{s:>5}" for s in BASE) + "  c9-impr"
    print("\nWins among c1..c8 (ties credit every tied strategy; [sole wins] in the second line)")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        per = s["per_strategy"]
        print(f"{g[:52]:52} {s['instances']:>5} " + " ".join(f"{per[k]['wins']:>5}" for k in BASE)
              + f"  {s['c9_improved']:>4} ({100 * s['c9_improved_rate']:.0f}%)")
        print(f"{'':52} {'':>5} " + " ".join(f"{'[' + str(per[k]['sole_wins']) + ']':>5}" for k in BASE))
    print("\nLegend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller small-graph sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip the small Part A graphs")
    ap.add_argument("--skip-large", action="store_true", help="skip the Part B adversarial families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest Part B instance (default 50000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    args = ap.parse_args()

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
        fh.write("group,name,n,m," + ",".join(CANDIDATE_NAMES) + ",best_c1_c8,winners,sole_winner,c9_gain\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},"
                     + ",".join(str(r["sizes"][k]) for k in CANDIDATE_NAMES)
                     + f",{r['best_c1_c8']},{'+'.join(r['winners'])},{r['sole_winner'] or ''},{r['c9_gain']}\n")
    with (out / "car_strategies_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,strategy,name,wins,sole_wins,win_rate\n")
        for g, s in summary.items():
            for k in BASE:
                p = s["per_strategy"][k]
                fh.write(f"\"{g}\",{s['instances']},{k},\"{CANDIDATE_NAMES[k]}\",{p['wins']},{p['sole_wins']},{p['win_rate']:.4f}\n")
            fh.write(f"\"{g}\",{s['instances']},c9,\"{CANDIDATE_NAMES['c9']} (strict improvements)\","
                     f"{s['c9_improved']},,{s['c9_improved_rate']:.4f}\n")
    (out / "car_strategies.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(),
         "candidates": CANDIDATE_NAMES, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
