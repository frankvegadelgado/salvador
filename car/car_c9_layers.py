"""car/ c9 study: bipartite layers solved one after another (Salvador 0.1.4).

Strategy c9 (salvador/bipartite_peel.py) treats every connected component on
its own. A bipartite component is solved with a linear-time near-optimal
bipartite strategy: the NetworkX minimum maximal matching, at most ``phases``
Hopcroft-Karp phases and Koenig's construction (a minimum cover when the phases
reach a maximum matching). A non-bipartite component is split into bipartite
edge layers: a breadth-first majority 2-colouring puts at least half of the
edges between the two colours -- layer 0, copy index 0 --, the remaining edges
are coloured the same way -- layer 1, copy index 1 -- and so on, so a component
with m_c edges has at most floor(log2 m_c) + 1 layers. The layers are solved in
several orders (all of them when there are at most ``max_orders``); in each
order a vertex selected in an earlier layer is discarded from the later ones,
and the smallest cover is kept.

This script runs the whole ensemble (final_candidates) once per instance and
records, for c9:

* the size, the ratio to tau(G) (exact) or to a certified lower bound L <= tau;
* whether c9 is optimal, above 1.5 tau, above 2 tau - r (r = components with an
  edge), and whether its bipartite solves were exact (``exact``);
* the number of layers (the script checks the bound floor(log2 m) + 1 and stops
  with an error if it fails), the number of orders tried and the components;
* c9 against c1, c2, the best of the other candidates c1..c8, c10..c16, and the
  final cover c17: strictly smaller, equal or larger; whether c17 came from c9.

Instances:

* car_common's small exact graphs (graph atlas n <= 7, random bipartite and
  general graphs, grids), tau exact;
* hand-made non-bipartite graphs: odd cycles, wheels, complete graphs,
  Petersen, circulants, Mycielski graphs, triangular and king's grids,
  odd-cycle chains, bipartite graphs plus a few odd edges, and mixtures;
* (full mode) --trials random small graphs with exact tau;
* (full mode) car_common's large families with certified lower bounds, and
  car_bipartite.py's bipartite families up to --bip-max-n (tau exact by Koenig);
* optionally the edge-list files of --dimacs (e.g. ``--dimacs experiment``).

Run from the repository root:

    python car/car_c9_layers.py --quick                 # small exact graphs and the hand-made graphs
    python car/car_c9_layers.py                         # + random small graphs, large and bipartite families
    python car/car_c9_layers.py --dimacs experiment     # + the DIMACS clique complements
    python car/car_c9_layers.py --phases 32 --max-orders 24   # larger c9 limits

Every large instance is built and evaluated in a worker process with a time
limit (``--timeout``, default 120 s, 0 = no limit) and is skipped if it exceeds it.

Outputs (in car/): car_c9_layers.json (summary and rows), car_c9_layers.csv (one
row per instance) and car_c9_layers_summary.csv (one row per group).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from pathlib import Path

import networkx as nx

import car_common as cc
from salvador.algorithm import DEFAULT_LAYER_ORDERS, DEFAULT_LAYER_PHASES, FINAL_NAME, final_candidates

SEED = 20261016
OTHERS = [f"c{i}" for i in range(1, 17) if i != 9]  # c1..c8, c10..c16, the candidates c9 is compared with
LIMITS = {"phases": DEFAULT_LAYER_PHASES, "max_orders": DEFAULT_LAYER_ORDERS}


# ----------------------------------------------------------------------------
# Instances.
# ----------------------------------------------------------------------------
def stress_instances():
    """Non-bipartite graphs (and mixtures) on which the peeling decides the cover."""
    grp = "Hand-made non-bipartite (exact)"
    for k in (3, 5, 7, 9, 15):
        yield f"cycle_{k}", grp, nx.cycle_graph(k)
    for k in (4, 5, 6, 8, 12):
        yield f"wheel_{k}", grp, nx.wheel_graph(k + 1)
    for k in (3, 4, 5, 6, 8):
        yield f"complete_{k}", grp, nx.complete_graph(k)
    yield "petersen", grp, nx.petersen_graph()
    yield "dodecahedral", grp, nx.dodecahedral_graph()
    for n, jumps in ((11, (1, 2)), (13, (1, 3)), (15, (1, 4)), (16, (1, 3, 5)), (17, (2, 3))):
        yield f"circulant_{n}_{'_'.join(map(str, jumps))}", grp, nx.circulant_graph(n, jumps)
    M = nx.cycle_graph(5)
    for step in range(2):
        M = nx.mycielskian(M)
        yield f"mycielski_C5_{step + 1}", grp, M
    for r, c in ((3, 3), (3, 4), (4, 4)):
        yield f"triangular_grid_{r}x{c}", grp, nx.triangular_lattice_graph(r, c)
        G = nx.grid_2d_graph(r, c)
        G.add_edges_from(((i, j), (i + 1, j + 1)) for i in range(r - 1) for j in range(c - 1))
        G.add_edges_from(((i + 1, j), (i, j + 1)) for i in range(r - 1) for j in range(c - 1))
        yield f"kings_grid_{r}x{c}", grp, G
    for links, k in ((3, 3), (4, 5), (5, 3)):
        G = nx.Graph()
        prev = None
        for t in range(links):
            cyc = [(t, i) for i in range(k)]
            G.add_edges_from(zip(cyc, cyc[1:] + cyc[:1]))
            if prev is not None:
                G.add_edge(prev, cyc[0])
            prev = cyc[k // 2]
        yield f"odd_cycle_chain_{links}x{k}", grp, G
    rng = random.Random(SEED)
    for a, b, odd in ((5, 5, 1), (6, 6, 2), (7, 5, 3), (8, 8, 4)):
        G = nx.bipartite.random_graph(a, b, 0.4, seed=rng.randrange(2**31))
        left = list(range(a))
        for _ in range(odd):
            u, v = rng.sample(left, 2)
            G.add_edge(u, v)  # an edge inside one side: an odd cycle when u, v are connected
        yield f"bipartite_plus_{odd}_odd_{a}x{b}", grp, G
    for k in (5, 7):
        G = nx.disjoint_union(nx.grid_2d_graph(3, 4), nx.cycle_graph(k))
        G = nx.disjoint_union(G, nx.complete_graph(4))
        yield f"mixed_grid_C{k}_K4", "Hand-made mixtures (exact)", G


def random_instances(trials: int, rng: random.Random):
    for t in range(trials):
        kind = t % 3
        n = rng.randint(5, 12)
        if kind == 0:
            G = nx.gnp_random_graph(n, rng.choice([0.2, 0.3, 0.45, 0.6, 0.75]), seed=rng.randrange(2**31))
            yield f"rgnp_{t}", "Random G(n,p) (exact, n<=12)", G
        elif kind == 1:
            G = nx.random_labeled_tree(n, seed=rng.randrange(2**31)) if hasattr(nx, "random_labeled_tree") \
                else nx.random_tree(n, seed=rng.randrange(2**31))
            u, v = rng.sample(range(n), 2)
            G.add_edge(u, v)  # one chord: an odd or an even cycle
            yield f"rtree_chord_{t}", "Random trees + one chord (exact)", G
        else:
            a, b = rng.randint(2, 6), rng.randint(2, 6)
            G = nx.bipartite.random_graph(a, b, rng.choice([0.3, 0.5, 0.7]), seed=rng.randrange(2**31))
            for _ in range(rng.randint(1, 3)):
                side = list(range(a)) if rng.random() < 0.5 or b < 2 else list(range(a, a + b))
                if len(side) >= 2:
                    G.add_edge(*rng.sample(side, 2))
            yield f"rbip_odd_{t}", "Random bipartite + odd edges (exact)", G


# ----------------------------------------------------------------------------
# Evaluation.
# ----------------------------------------------------------------------------
def clean(G: nx.Graph) -> nx.Graph:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def evaluate(name: str, group: str, G: nx.Graph, exact: bool, spectral: bool = True,
             limits: dict | None = None) -> dict | None:
    limits = limits or LIMITS
    G = clean(G)
    if G.number_of_edges() == 0:
        return None
    info: dict = {}
    t0 = time.perf_counter()
    cands, refined = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info, return_refined=True,
                                      layer_phases=limits["phases"], layer_orders=limits["max_orders"])
    seconds = time.perf_counter() - t0
    for k, C in cands.items():
        if not all(a in C or b in C for a, b in G.edges()):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
    m = G.number_of_edges()
    layers = info.get("c9_max_layers", 0)
    if layers > math.floor(math.log2(m)) + 1:
        raise AssertionError(f"{name}: c9 used {layers} layers, more than floor(log2 m) + 1 = "
                             f"{math.floor(math.log2(m)) + 1}")
    bipartite = nx.is_bipartite(G)
    if exact or bipartite:
        tau, method = cc.exact_vertex_cover_size(G)
        lb = tau
    else:
        lb, method, _ = cc.best_lower_bound(G, spectral=spectral)
        tau = None
    sizes = {k: len(v) for k, v in cands.items()}
    c9 = sizes["c9"]
    best_others = min(sizes[k] for k in OTHERS)
    r = nx.number_connected_components(G)
    final = sizes[FINAL_NAME]
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": m, "r": r,
        "bipartite": bipartite, "tau": tau, "lower_bound": lb, "lower_bound_method": method,
        "exact_flag": tau is not None,
        "c9": c9, "c1": sizes["c1"], "c2": sizes["c2"], "best_others": best_others, "c17": final,
        "c9_vs_best": c9 - best_others,  # < 0: c9 strictly smaller than every other candidate
        "c17_from_c9": len(refined["c9"]) == final,
        "c9_refined": len(refined["c9"]),
        "layers": layers, "layers_total": info.get("c9_layers", 0), "orders": info.get("c9_orders", 0),
        "c9_exact": bool(info.get("c9_exact", False)),
        "bipartite_components": info.get("c9_bipartite_components", 0),
        "nonbipartite_components": info.get("c9_nonbipartite_components", 0),
        "seconds": round(seconds, 4),
        "edges": [list(e) for e in G.edges()] if G.number_of_nodes() <= 14 else None,
    }
    if tau is not None:
        row.update({"c9_ratio": c9 / tau, "c9_optimal": c9 == tau, "c9_above_1_5": c9 > 1.5 * tau,
                    "c9_above_2tau_minus_r": c9 > 2 * tau - r, "c17_ratio": final / tau})
    else:
        row.update({"c9_ratio": c9 / lb if lb else 1.0, "c9_optimal": None, "c9_above_1_5": None,
                    "c9_above_2tau_minus_r": None, "c17_ratio": final / lb if lb else 1.0})
    print(f"  [c9] {group[:32]:32} {name:28} n={row['n']:>6} m={row['m']:>7} "
          f"{('tau=' + str(tau)) if tau is not None else ('L=' + str(lb)):>10} c9={c9:<6}"
          f" (layers {layers}, orders {row['orders']}{', exact' if row['c9_exact'] else ''})"
          f" c1={sizes['c1']} c2={sizes['c2']} best others={best_others} c17={final}"
          f" ratio={row['c9_ratio']:.4f}", flush=True)
    return row


def _large_job(name: str, max_n: int, spectral: bool, limits: dict) -> dict | None:
    for nm, group, build in cc.large_instances(max_n):
        if nm == name:
            return evaluate(name, group, build(), exact=False, spectral=spectral, limits=limits)
    raise KeyError(name)


def _bipartite_job(name: str, max_n: int, seed: int, limits: dict) -> dict | None:
    from car_bipartite import instances as bip_instances, shuffle_labels

    rng = random.Random(seed)
    for nm, group, make in bip_instances(max_n, False, rng):
        if nm == name:
            rng.seed(f"{seed}:{name}")
            G = shuffle_labels(nx.convert_node_labels_to_integers(make()), rng)
            return evaluate(name, f"Bipartite: {group}", G, exact=False, limits=limits)
    raise KeyError(name)


def _dimacs_job(path: str, spectral: bool, limits: dict) -> dict | None:
    G = cc.load_edge_list(Path(path))
    if not G.number_of_edges():
        return None
    return evaluate(Path(path).name.split(".clq")[0], "DIMACS / edge-list files", G, exact=False,
                    spectral=spectral, limits=limits)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        ex = [r for r in grows if r["exact_flag"]]
        k = len(grows)
        out[g] = {
            "instances": k, "exact_instances": len(ex),
            "bipartite_instances": sum(r["bipartite"] for r in grows),
            "c9_optimal": sum(r["c9_optimal"] for r in ex),
            "c9_exact": sum(r["c9_exact"] for r in grows),
            "c17_optimal": sum(r["c17"] == r["tau"] for r in ex),
            "c9_above_1_5": sum(r["c9_above_1_5"] for r in ex),
            "c9_above_2tau_minus_r": sum(r["c9_above_2tau_minus_r"] for r in ex),
            "c9_max_ratio": max((r["c9_ratio"] for r in grows), default=1.0),
            "c9_mean_ratio": (sum(r["c9_ratio"] for r in grows) / k) if k else 1.0,
            "c9_worst_instance": max(grows, key=lambda r: r["c9_ratio"])["name"] if grows else None,
            "c9_strictly_best": sum(r["c9_vs_best"] < 0 for r in grows),
            "c9_ties_best": sum(r["c9_vs_best"] == 0 for r in grows),
            "c9_worse_than_best": sum(r["c9_vs_best"] > 0 for r in grows),
            "c9_total_gain": sum(-r["c9_vs_best"] for r in grows if r["c9_vs_best"] < 0),
            "c9_smaller_than_c1": sum(r["c9"] < r["c1"] for r in grows),
            "c9_smaller_than_c2": sum(r["c9"] < r["c2"] for r in grows),
            "c9_larger_than_c2": sum(r["c9"] > r["c2"] for r in grows),
            "c17_from_c9": sum(r["c17_from_c9"] for r in grows),
            "mean_layers": (sum(r["layers"] for r in grows) / k) if k else 0.0,
            "max_layers": max((r["layers"] for r in grows), default=0),
            "max_layers_over_log2m": max((r["layers"] / (math.floor(math.log2(r["m"])) + 1) for r in grows),
                                         default=0.0),
            "seconds": sum(r["seconds"] for r in grows),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small exact graphs and the hand-made graphs only")
    ap.add_argument("--trials", type=int, default=2000, help="random small graphs with exact tau (default 2000)")
    ap.add_argument("--skip-large", action="store_true", help="skip car_common's large families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest large instance (default 50000)")
    ap.add_argument("--skip-bipartite", action="store_true", help="skip car_bipartite.py's families")
    ap.add_argument("--bip-max-n", type=int, default=10_000, help="largest bipartite instance (default 10000)")
    ap.add_argument("--dimacs", type=str, default=None, help="directory of edge-list files to include")
    ap.add_argument("--no-spectral", action="store_true", help="skip the spectral lower bound on large graphs")
    ap.add_argument("--phases", type=int, default=DEFAULT_LAYER_PHASES,
                    help=f"c9: Hopcroft-Karp phases per bipartite solve (default {DEFAULT_LAYER_PHASES})")
    ap.add_argument("--max-orders", type=int, default=DEFAULT_LAYER_ORDERS,
                    help=f"c9: largest number of layer orders per component (default {DEFAULT_LAYER_ORDERS})")
    ap.add_argument("--seed", type=int, default=SEED)
    cc.add_timeout_argument(ap)
    args = ap.parse_args()
    spectral = not args.no_spectral
    limits = {"phases": args.phases, "max_orders": args.max_orders}
    LIMITS.update(limits)

    started = time.time()
    rows: list[dict] = []
    skipped: list[str] = []

    def add(row):
        if row is not None:
            rows.append(row)

    for name, group, G in stress_instances():
        add(evaluate(name, group, G, exact=True))
    for name, group, G in cc.small_instances(random.Random(cc.SEED), big=not args.quick):
        add(evaluate(name, group, G, exact=True))
    if not args.quick:
        for name, group, G in random_instances(args.trials, random.Random(args.seed)):
            add(evaluate(name, group, G, exact=True))
    with cc.InstanceRunner(args.timeout) as runner:
        jobs = []
        if not args.quick and not args.skip_large:
            jobs += [(name, _large_job, (name, args.max_n, spectral, limits))
                     for name, _, _ in cc.large_instances(args.max_n)]
        if not args.quick and not args.skip_bipartite:
            from car_bipartite import instances as bip_instances
            jobs += [(name, _bipartite_job, (name, args.bip_max_n, args.seed, limits))
                     for name, _, _ in bip_instances(args.bip_max_n, False, random.Random(args.seed))]
        if args.dimacs:
            jobs += [(p.name, _dimacs_job, (str(p), spectral, limits))
                     for p in sorted(Path(args.dimacs).iterdir()) if p.is_file()]
        for name, job, job_args in jobs:
            row = runner.run(job, *job_args)
            if row is None and job is not _dimacs_job:
                print(f"  [c9] {name}: skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            add(row)
            gc.collect()

    summary = summarise(rows)
    print("\nc9 (bipartite layers): optimum where tau is known, layers, and comparison with the other candidates")
    head = (f"{'group':40} {'inst':>5} {'exact':>5} {'bip':>4} {'c9 opt':>6} {'c9 ex':>5} {'c17 opt':>7}"
            f" {'>1.5tau':>7} {'>2tau-r':>7} {'max ratio':>9} {'mean':>7} {'<best':>5} {'=best':>5} {'>best':>5}"
            f" {'gain':>5} {'<c1':>4} {'<c2':>4} {'>c2':>4} {'c17<-9':>6} {'layers':>6} {'max':>4}")
    print(head)
    print("-" * len(head))
    for g, s in summary.items():
        print(f"{g[:40]:40} {s['instances']:>5} {s['exact_instances']:>5} {s['bipartite_instances']:>4}"
              f" {s['c9_optimal']:>6} {s['c9_exact']:>5} {s['c17_optimal']:>7} {s['c9_above_1_5']:>7}"
              f" {s['c9_above_2tau_minus_r']:>7} {s['c9_max_ratio']:>9.4f} {s['c9_mean_ratio']:>7.4f}"
              f" {s['c9_strictly_best']:>5} {s['c9_ties_best']:>5} {s['c9_worse_than_best']:>5} {s['c9_total_gain']:>5}"
              f" {s['c9_smaller_than_c1']:>4} {s['c9_smaller_than_c2']:>4} {s['c9_larger_than_c2']:>4}"
              f" {s['c17_from_c9']:>6} {s['mean_layers']:>6.2f} {s['max_layers']:>4}")
    alls = summary["ALL"]
    print("\n<best / =best / >best: c9 strictly smaller than / equal to / larger than the best of the other"
          " candidates c1..c8, c10..c16; gain: vertices saved where smaller; c17<-9: the refinement of c9 attains |c17|;"
          " c9 ex: every bipartite solve reached a maximum matching.")
    print(f"Largest layers / (floor(log2 m) + 1): {alls['max_layers_over_log2m']:.3f} (at most 1 by construction)")
    worst = [r for r in rows if r["exact_flag"] and not r["c9_optimal"]]
    worst.sort(key=lambda r: (-r["c9_ratio"], r["n"]))
    for r in worst[:10]:
        print(f"  not optimal: {r['name']} n={r['n']} tau={r['tau']} c9={r['c9']} ratio={r['c9_ratio']:.4f}"
              + (f" edges={r['edges']}" if r["edges"] else ""))
    if skipped:
        print(f"Skipped (more than {args.timeout:g} s): {', '.join(skipped)}")

    out = cc.OUT_DIR
    stem = "car_c9_layers"
    (out / f"{stem}.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": args.seed,
         "quick": args.quick, "trials": args.trials, "limits": limits,
         "skipped_instances": skipped, "summary": summary,
         "rows": [{k: v for k, v in r.items() if k != "edges"} for r in rows],
         "elapsed_seconds": time.time() - started}, indent=1, default=str), encoding="utf-8")
    cols = ["group", "name", "n", "m", "r", "bipartite", "tau", "lower_bound", "lower_bound_method", "c9", "c1", "c2",
            "best_others", "c17", "c9_vs_best", "c17_from_c9", "c9_ratio", "c9_optimal", "c9_exact", "c9_above_1_5",
            "c9_above_2tau_minus_r", "layers", "layers_total", "orders", "bipartite_components",
            "nonbipartite_components", "seconds"]
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    with open(out / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["group", "instances", "exact_instances", "bipartite_instances", "c9_optimal", "c9_exact",
                  "c17_optimal", "c9_above_1_5", "c9_above_2tau_minus_r", "c9_max_ratio", "c9_mean_ratio",
                  "c9_worst_instance", "c9_strictly_best", "c9_ties_best", "c9_worse_than_best", "c9_total_gain",
                  "c9_smaller_than_c1", "c9_smaller_than_c2", "c9_larger_than_c2", "c17_from_c9", "mean_layers",
                  "max_layers", "max_layers_over_log2m", "seconds"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for g, s in summary.items():
            w.writerow({"group": g, **s})


if __name__ == "__main__":
    main()
