"""car/ c1 bound study: does c1 satisfy |c1| <= 2 tau - r' ?

c1 (salvador/component_split.py) splits edges of the NetworkX minimum maximal
matching M of G, both endpoints at copy index 0, into isolated edges (the other
edges of an endpoint u move to its only other copy (u, 1)) until the split graph
H has r' components, the largest value not above |M| (so r' <= |M| <= tau and
r' >= |M| - 1); it then decodes the NetworkX minimum maximal matching of H by
copy index and degree (no residual graph: with the indices 0 and 1 only, an
edge with equal indices takes its endpoint of larger degree in G), completes
the decoded set and prunes it. A cover with |C| <= 2 tau - r' is within ratio
2 - r'/tau of the optimum: at most 3/2 when r' = |M| (since |M| >= tau/2), and at
most 3/2 + 1/tau when r' = |M| - 1. The bound is not proved and is measured here.

Variants (every decoding rule x every completion), then the pipelines:

* select     (matched edge ((a, i), (b, j)), i != j, a and b both unselected):
               low       -- index 0
               high      -- index 1
               deg_low   -- larger degree in G, ties by index 0
               deg_high  -- larger degree in G, ties by index 1
* completion (the decoded set is not a cover of G in general):
               greedy    -- bucket-queue greedy on the uncovered edges, then prune (as c1)
               prune     -- salvador's prune_redundant_vertices only (one endpoint per uncovered edge)
* pipelines:   c1  -- salvador.algorithm.c1_cover, the smallest of deg_high, high, deg_low (greedy)
               c2  -- pruned NetworkX minimum maximal matching

For every instance with a known optimum tau the script records whether
|cover| > 2 tau - r' (bound violation), whether |cover| > 2 tau - r (r =
components of G with an edge; the bound proved for c2), whether
|cover| > 1.5 tau, the ratio, and whether r' = |M| (``r_prime_exact``). The
smallest counterexample of every variant is stored with its edge list, so a
violation can be checked by hand.

Large graphs. Bipartite graphs (car_bipartite.py's families up to --bip-max-n)
have tau exact by Koenig and are checked like the small graphs. Without exact
tau, the bound is decided from a certified lower bound L <= tau
(car_common.best_lower_bound) and an upper bound U >= tau (the smallest cover
found by any variant):

  certified   |cover| <= 2L - r'   proves |cover| <= 2 tau - r'
  refuted     |cover| >  2U - r'   proves |cover| >  2 tau - r'
  undecided   otherwise (L too weak to decide)

Instances: hand-made stress graphs (paths, stars, double stars, spiders,
caterpillars, K_{2,k}, brooms), car_common's small exact graphs, --trials random
small graphs with exact tau, and the large families up to --max-n.

Run from the repository root:

    python car/car_c1_fix.py --quick                 # small exact graphs + stress graphs only
    python car/car_c1_fix.py                         # also 2000 random small graphs and the large families
    python car/car_c1_fix.py --trials 20000 --skip-large
    python car/car_c1_fix.py --trials 0 --bip-max-n 50000

Outputs (in car/): car_c1_fix.json (summary, smallest counterexample per
variant), car_c1_fix.csv (one row per instance x variant) and
car_c1_fix_summary.csv (one row per variant).
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import random
import time

import networkx as nx

import car_common as cc
from salvador import component_split
from salvador.algorithm import c1_cover, maximal_matching_vertex_cover, prune_redundant_vertices

SELECT = component_split.RULES
COMPLETION = ("greedy", "prune")
VARIANTS = [f"{s}/{c}" for s, c in itertools.product(SELECT, COMPLETION)]
PIPELINES = ("c1", "c2")
VARIANTS = VARIANTS + list(PIPELINES)
SEED = 20261016


def all_variants(G: nx.Graph) -> tuple[dict[str, set], dict]:
    """Covers of every variant on G (no self-loops, no isolated vertices) and the split statistics."""
    info: dict = {}
    H = component_split.build_split_graph(G, info)
    MH = list(nx.approximation.min_maximal_matching(H))
    adj = {v: set(G[v]) for v in G}
    covers = {}
    for rule in SELECT:
        raw = component_split.decode(G, H, rule, matching=MH)
        covers[f"{rule}/greedy"] = prune_redundant_vertices(adj, component_split.complete_cover(G, raw))
        covers[f"{rule}/prune"] = prune_redundant_vertices(adj, raw)
    c1_info: dict = {}
    covers["c1"] = c1_cover(G, adj, info=c1_info)
    covers["c2"] = prune_redundant_vertices(adj, maximal_matching_vertex_cover(G))
    info["c1_rule"] = c1_info["rule"]
    return covers, info


# ----------------------------------------------------------------------------
# Instances.
# ----------------------------------------------------------------------------
def stress_instances():
    """Hand-made graphs on which a leaf-for-hub choice is expensive."""
    yield "path_3", "Stress: paths/stars/spiders", nx.path_graph(3)
    for k in (3, 5, 10, 50):
        yield f"star_{k}", "Stress: paths/stars/spiders", nx.star_graph(k)
    for k in (2, 3, 10, 100):
        yield f"double_star_{k}", "Stress: paths/stars/spiders", cc.double_star_bridge(k, k)
    for legs, length in ((3, 2), (5, 2), (4, 3), (10, 2)):
        G = nx.Graph()
        for leg in range(legs):
            prev = "c"
            for step in range(length):
                G.add_edge(prev, (leg, step))
                prev = (leg, step)
        yield f"spider_{legs}x{length}", "Stress: paths/stars/spiders", G
    for spine, hair in ((4, 2), (6, 3), (10, 1), (10, 4)):
        G = nx.path_graph(spine)
        for v in range(spine):
            for h in range(hair):
                G.add_edge(v, (v, h))
        yield f"caterpillar_{spine}x{hair}", "Stress: caterpillars/brooms/K2k", G
    for k in (3, 6, 12):
        yield f"K_2_{k}", "Stress: caterpillars/brooms/K2k", nx.complete_bipartite_graph(2, k)
    for handle, bristles in ((3, 4), (6, 8)):
        G = nx.path_graph(handle)
        for b in range(bristles):
            G.add_edge(handle - 1, ("b", b))
        yield f"broom_{handle}_{bristles}", "Stress: caterpillars/brooms/K2k", G
    for p in (3, 4, 6):
        yield f"path_{p}", "Stress: paths/stars/spiders", nx.path_graph(p)


def random_instances(trials: int, rng: random.Random):
    for t in range(trials):
        kind = t % 3
        n = rng.randint(4, 12)
        if kind == 0:
            G = nx.random_labeled_tree(n, seed=rng.randrange(2**31)) if hasattr(nx, "random_labeled_tree") \
                else nx.random_tree(n, seed=rng.randrange(2**31))
            yield f"rtree_{t}", "Random trees (exact, n<=12)", G
        elif kind == 1:
            G = nx.gnp_random_graph(n, rng.choice([0.15, 0.25, 0.4, 0.6]), seed=rng.randrange(2**31))
            yield f"rgnp_{t}", "Random G(n,p) (exact, n<=12)", G
        else:
            spine = rng.randint(2, 5)
            G = nx.path_graph(spine)
            for v in range(spine):
                for h in range(rng.randint(0, 3)):
                    G.add_edge(v, (v, h))
            yield f"rcat_{t}", "Random caterpillars (exact)", G


# ----------------------------------------------------------------------------
# Evaluation.
# ----------------------------------------------------------------------------
def clean(G: nx.Graph) -> nx.Graph:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def verdict(size: int, lower: int, upper: int, r_prime: int) -> str:
    """Bound |cover| <= 2 tau - r' without tau, from L <= tau <= U."""
    if size <= 2 * lower - r_prime:
        return "certified"
    if size > 2 * upper - r_prime:
        return "refuted"
    return "undecided"


def evaluate(name: str, group: str, G: nx.Graph, exact: bool, spectral: bool = True) -> dict | None:
    G = clean(G)
    if G.number_of_edges() == 0:
        return None
    t0 = time.perf_counter()
    covers, info = all_variants(G)
    seconds = time.perf_counter() - t0
    for v, c in covers.items():
        if not all(a in c or b in c for a, b in G.edges()):
            raise AssertionError(f"{name}: variant {v} is not a vertex cover")
    if exact or nx.is_bipartite(G):
        tau, method = cc.exact_vertex_cover_size(G)
        lb = tau
    else:
        lb, method, _ = cc.best_lower_bound(G, spectral=spectral)
        tau = None
    sizes = {v: len(c) for v, c in covers.items()}
    upper = min(sizes.values())
    row = {"group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
           "tau": tau, "lower_bound": lb, "upper_bound": upper, "exact_flag": tau is not None,
           "lower_bound_method": method, "r_prime": info["r_prime"], "target": info["target"],
           "r_prime_exact": info["exact"], "splits": info["splits"], "r": info["r"], "c1_rule": info["c1_rule"],
           "sizes": sizes, "seconds": round(seconds, 4),
           "edges": [list(e) for e in G.edges()] if G.number_of_nodes() <= 14 else None}
    print(f"  [c1fix] {group[:30]:30} {name:24} n={row['n']:>6} m={row['m']:>7} "
          f"{('tau=' + str(tau)) if tau is not None else ('L=' + str(lb)):>9} r'={info['r_prime']}/{info['target']}"
          f" c1={sizes['c1']} ({info['c1_rule']}) c2={sizes['c2']} best={upper}"
          + (f"  2tau-r'={2 * tau - info['r_prime']}" if tau is not None else
             f"  2L-r'={2 * lb - info['r_prime']} c1 " + verdict(sizes["c1"], lb, upper, info["r_prime"])), flush=True)
    return row


def _bipartite_job(name: str, max_n: int, seed: int, spectral: bool) -> dict | None:
    from car_bipartite import instances as bip_instances, shuffle_labels

    rng = random.Random(seed)
    for nm, group, make in bip_instances(max_n, False, rng):
        if nm != name:
            continue
        rng.seed(f"{seed}:{name}")
        G = shuffle_labels(nx.convert_node_labels_to_integers(make()), rng)
        return evaluate(name, f"Bipartite: {group}", G, exact=False, spectral=spectral)
    raise KeyError(name)


def _large_job(name: str, max_n: int, spectral: bool) -> dict | None:
    for nm, group, build in cc.large_instances(max_n):
        if nm == name:
            return evaluate(name, group, build(), exact=False, spectral=spectral)
    raise KeyError(name)


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    exact_rows = [r for r in rows if r["exact_flag"]]
    inexact = [r for r in rows if not r["exact_flag"]]
    for v in VARIANTS:
        ratios = [(r["sizes"][v] / r["lower_bound"]) if r["lower_bound"] else 1.0 for r in rows]
        viol = [r for r in exact_rows if r["sizes"][v] > 2 * r["tau"] - r["r_prime"]]
        proven = [r for r in exact_rows if r["sizes"][v] > 2 * r["tau"] - r["r"]]
        above = [r for r in exact_rows if r["sizes"][v] > 1.5 * r["tau"]]
        smallest = min(viol, key=lambda r: (r["n"], r["m"]), default=None)
        smallest15 = min(above, key=lambda r: (r["n"], r["m"]), default=None)
        worst = max(rows, key=lambda r: (r["sizes"][v] / r["lower_bound"]) if r["lower_bound"] else 1.0, default=None)
        verdicts = [verdict(r["sizes"][v], r["lower_bound"], r["upper_bound"], r["r_prime"]) for r in inexact]
        out.append({
            "variant": v, "is_pipeline": v in PIPELINES, "instances": len(rows), "exact_instances": len(exact_rows),
            "bound_violations": len(viol), "above_1_5": len(above), "proven_bound_violations": len(proven),
            "inexact_instances": len(inexact), "certified": verdicts.count("certified"),
            "refuted": verdicts.count("refuted"), "undecided": verdicts.count("undecided"),
            "refuted_instances": [r["name"] for r, x in zip(inexact, verdicts) if x == "refuted"],
            "undecided_instances": [r["name"] for r, x in zip(inexact, verdicts) if x == "undecided"],
            "optimal": sum(r["sizes"][v] == r["tau"] for r in exact_rows),
            "max_ratio": max(ratios, default=1.0), "mean_ratio": sum(ratios) / len(ratios) if ratios else 1.0,
            "worst_instance": worst["name"] if worst else None,
            "smallest_violation": None if smallest is None else
            {"name": smallest["name"], "n": smallest["n"], "tau": smallest["tau"], "r_prime": smallest["r_prime"],
             "size": smallest["sizes"][v], "edges": smallest["edges"]},
            "smallest_above_1_5": None if smallest15 is None else
            {"name": smallest15["name"], "n": smallest15["n"], "tau": smallest15["tau"],
             "size": smallest15["sizes"][v], "edges": smallest15["edges"]},
        })
    out.sort(key=lambda s: (s["bound_violations"], s["above_1_5"], s["max_ratio"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small exact graphs and stress graphs only")
    ap.add_argument("--trials", type=int, default=2000, help="random small graphs with exact tau (default 2000)")
    ap.add_argument("--skip-large", action="store_true", help="skip the large families")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest large instance (default 50000)")
    ap.add_argument("--no-spectral", action="store_true", help="skip the spectral lower bound on large graphs")
    ap.add_argument("--skip-bipartite", action="store_true", help="skip car_bipartite.py's bipartite families")
    ap.add_argument("--bip-max-n", type=int, default=10_000, help="largest bipartite instance (default 10000)")
    ap.add_argument("--seed", type=int, default=SEED)
    cc.add_timeout_argument(ap)
    args = ap.parse_args()
    spectral = not args.no_spectral

    started = time.time()
    rows: list[dict] = []

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
    skipped: list[str] = []
    with cc.InstanceRunner(args.timeout) as runner:
        jobs = []
        if not args.quick and not args.skip_large:
            jobs += [(name, _large_job, (name, args.max_n, spectral)) for name, _, _ in cc.large_instances(args.max_n)]
        if not args.quick and not args.skip_bipartite:
            from car_bipartite import instances as bip_instances
            jobs += [(name, _bipartite_job, (name, args.bip_max_n, args.seed, spectral))
                     for name, _, _ in bip_instances(args.bip_max_n, False, random.Random(args.seed))]
        for name, job, job_args in jobs:
            row = runner.run(job, *job_args)
            if row is None:
                print(f"  [c1fix] {name}: skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            add(row)
            gc.collect()

    summary = summarise(rows)
    print("\nVariants (select/completion on the split graph with indices 0 and 1) and pipelines, best first;"
          " * = c1 (salvador's c1) and c2 (pruned matching)")
    head = (f"{'variant':20} {'exact':>6} {'>2tau-r_':>9} {'>2tau-r':>8} {'>1.5tau':>8} {'optimal':>8}"
            f" {'max ratio':>10} {'mean':>7} {'no tau':>6} {'cert':>5} {'refut':>5} {'undec':>5}  worst")
    print(head)
    print("-" * len(head))
    for s in summary:
        print(f"{s['variant'] + (' *' if s['is_pipeline'] else ''):20} {s['exact_instances']:>6} {s['bound_violations']:>9}"
              f" {s['proven_bound_violations']:>8} {s['above_1_5']:>8} {s['optimal']:>8} {s['max_ratio']:>10.4f}"
              f" {s['mean_ratio']:>7.4f} {s['inexact_instances']:>6} {s['certified']:>5} {s['refuted']:>5}"
              f" {s['undecided']:>5}  {s['worst_instance']}")
    exact_r = sum(r["r_prime_exact"] for r in rows)
    print(f"\nr' = |M| on {exact_r} of {len(rows)} instances (otherwise r' = |M| - 1: the next split of an edge of M"
          " would add two components).")
    for name in PIPELINES:
        s_ = next(x for x in summary if x["variant"] == name)
        print(f"{name}: violations of 2 tau - r' {s_['bound_violations']}, of 2 tau - r {s_['proven_bound_violations']}"
              f", above 1.5 tau {s_['above_1_5']}, max ratio {s_['max_ratio']:.4f}; without exact tau:"
              f" {s_['inexact_instances']} -> certified {s_['certified']}, refuted {s_['refuted']},"
              f" undecided {s_['undecided']}")
        for key in ("refuted_instances", "undecided_instances", "smallest_violation", "smallest_above_1_5"):
            if s_[key]:
                print(f"  {key}: {s_[key]}")
    if skipped:
        print(f"Skipped (more than {args.timeout:g} s): {', '.join(skipped)}")

    out = cc.OUT_DIR
    stem = "car_c1_fix"
    (out / f"{stem}.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": args.seed,
         "variants": VARIANTS, "pipelines": list(PIPELINES), "quick": args.quick, "trials": args.trials,
         "skipped_instances": skipped, "summary": summary,
         "rows": [{k: v for k, v in r.items() if k != "edges"} for r in rows],
         "elapsed_seconds": time.time() - started}, indent=1, default=str), encoding="utf-8")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "name", "n", "m", "tau", "lower_bound", "exact", "r_prime", "target", "variant", "size",
                    "ratio", "bound_2tau_minus_rprime", "bound_holds", "above_1_5", "upper_bound", "verdict_without_tau"])
        for r in rows:
            for v in VARIANTS:
                size = r["sizes"][v]
                ex = r["exact_flag"]
                b = 2 * r["tau"] - r["r_prime"] if ex else None
                w.writerow([r["group"], r["name"], r["n"], r["m"], r["tau"], r["lower_bound"], ex, r["r_prime"],
                            r["target"], v, size, round(size / r["lower_bound"], 6) if r["lower_bound"] else 1.0,
                            b, (size <= b) if ex else "", (size > 1.5 * r["tau"]) if ex else "", r["upper_bound"],
                            "" if ex else verdict(size, r["lower_bound"], r["upper_bound"], r["r_prime"])])
    with open(out / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["variant", "is_pipeline", "instances", "exact_instances", "bound_violations",
                  "proven_bound_violations", "above_1_5", "optimal", "max_ratio", "mean_ratio", "inexact_instances",
                  "certified", "refuted", "undecided", "worst_instance"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
