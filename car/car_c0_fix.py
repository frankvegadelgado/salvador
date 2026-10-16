"""car/ c0 repair study: which decoding rule (if any) makes c0 satisfy |c0| <= 2 tau - r' ?

car_c0_components.py found that c0 (salvador/component_split.py) violates the
bound |c0| <= 2 tau - r' (r' = components of the split graph H) on 47 of 286
exact instances, e.g. the path a-b-c (tau = 1, r' = 1, c0 = {a, c}), the star
K_{1,3} (tau = 1, c0 = 3 leaves) and the double star (tau = 2, c0 = 1001). The
failures come from the decoding: the matched edge of H is resolved by copy index
(or sent to the residual graph when the indices tie) without looking at G, so a
leaf can be selected instead of its hub, and the repair step then adds the hub's
other neighbours one by one.

Since 0.1.3 the split graph H stops exactly at r' = |M| (no overshoot, so
r' <= tau and 2 tau - r' >= tau is always feasible). This script decodes the
NetworkX minimum maximal matching M_H of that H with every variant, the product of

* select   (matched edge ((a, i), (b, j)), i != j, a and b both unselected):
             low       -- the smaller index (c0 rule "low")
             high      -- the larger index (c0 rule "high")
             deg_low   -- the endpoint of larger degree in G, ties by smaller index
             deg_high  -- the endpoint of larger degree in G, ties by larger index
* residual (matched edges with i == j still uncovered):
             konig     -- Koenig when the residual graph is bipartite, else by occurrences (c0)
             occ       -- greedy by number of occurrences in the residual graph
             gdeg      -- each residual edge takes its endpoint of larger degree in G
* repair   (the decoded set is not a cover of G in general):
             prune     -- salvador's prune_redundant_vertices (repair + prune, as c0)
             greedy    -- greedy max-uncovered-degree completion, then prune

and also runs the complete pipelines:

  c0_old          the first c0 (overshooting splits, low/high, Koenig, prune; r' of its own H)
  c0_new/decoded  salvador's c0 decoding: deg_high|high|deg_low with gdeg, greedy completion, prune
  c0_new/ls       + budgeted (1,2)-swap local search
  c0_new          + c1 when c1 is smaller: salvador.algorithm.c0_cover, the c0 of the ensemble
  c1              pruned NetworkX minimum maximal matching

and records, for every variant and instance with a known optimum tau, whether
|cover| > 2 tau - r' (bound violation), whether |cover| > 2 tau - r (r = components
of G; the proven bound, expected 0 for c0_new and c1), whether |cover| > 1.5 tau,
and the ratio.
The smallest counterexample of every variant is stored with its edge list, so a
violation can be checked by hand.

Large graphs. Bipartite graphs (the crown and complete bipartite families and
car_bipartite.py's families up to --bip-max-n) have tau exact by Koenig and are
checked like the small graphs. Without exact tau, the bound is decided from a
certified lower bound L <= tau (car_common.best_lower_bound) and an upper bound
U >= tau (the smallest cover found by any variant or pipeline):

  certified   |cover| <= 2L - r'   proves |cover| <= 2 tau - r'
  refuted     |cover| >  2U - r'   proves |cover| >  2 tau - r'
  undecided   otherwise (L too weak to decide)

The ratio against L is reported as before.

Instances: car_common's small exact graphs (graph atlas, random bipartite and
general graphs, grids), hand-made stress graphs (paths, stars, double stars,
spiders, caterpillars, K_{2,k}, brooms), --trials random small graphs (trees,
G(n, p), caterpillars; tau exact), and the large families up to --max-n.

Run from the repository root:

    python car/car_c0_fix.py --quick                 # small exact graphs + stress graphs only
    python car/car_c0_fix.py                         # also 2000 random small graphs and large families
    python car/car_c0_fix.py --trials 20000 --skip-large
    python car/car_c0_fix.py --trials 0 --bip-max-n 50000   # large graphs only (plus the small exact sets)

Outputs (in car/): car_c0_fix.json (summary, worst/smallest counterexample per
variant), car_c0_fix.csv (one row per instance x variant) and
car_c0_fix_summary.csv (one row per variant).
"""

from __future__ import annotations

import argparse
import csv
import gc
import heapq
import itertools
import json
import random
import time

import networkx as nx

import car_common as cc
from salvador import component_split
from salvador.algorithm import c0_cover, maximal_matching_vertex_cover, prune_redundant_vertices

SELECT = ("low", "high", "deg_low", "deg_high")
RESIDUAL = ("konig", "occ", "gdeg")
REPAIR = ("prune", "greedy")
VARIANTS = [f"{s}/{r}/{p}" for s, r, p in itertools.product(SELECT, RESIDUAL, REPAIR)]
#: The pipelines compared with the decoding grid: the first c0 (overshooting splits, index rules,
#: Koenig residual, prune), the repaired c0 of salvador.algorithm.c0_cover by stage (decoded and
#: pruned; + (1,2)-swap local search; + fallback to c1 = final c0), and c1 itself.
PIPELINES = ("c0_old", "c0_new/decoded", "c0_new/ls", "c0_new", "c1")
C0_VARIANTS = ("c0_old", "c0_new")
VARIANTS = VARIANTS + list(PIPELINES)
SEED = 20261016


# ----------------------------------------------------------------------------
# Decoding variants.
# ----------------------------------------------------------------------------
def _greedy_cover(edges, pos) -> set:
    """Greedy vertex cover of an edge list by largest current degree (lazy heap), ties by position."""
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    heap = [(-len(nb), pos[v], v) for v, nb in adj.items()]
    heapq.heapify(heap)
    cover: set = set()
    while heap:
        d, _, v = heapq.heappop(heap)
        cur = len(adj[v])
        if v in cover or cur == 0:
            continue
        if -d != cur:  # stale entry: re-queue with the current degree
            heapq.heappush(heap, (-cur, pos[v], v))
            continue
        cover.add(v)
        for w in adj[v]:
            adj[w].discard(v)
            heapq.heappush(heap, (-len(adj[w]), pos[w], w))
        adj[v] = set()
    return cover


def _konig_or_occ(edges, pos) -> set:
    if not edges:
        return set()
    R = nx.Graph()
    R.add_edges_from(edges)
    if not nx.is_bipartite(R):
        return _greedy_cover(edges, pos)
    cover: set = set()
    for comp in nx.connected_components(R):
        S = R.subgraph(comp)
        top = {v for v, c in nx.bipartite.color(S).items() if c == 0}
        M = nx.bipartite.hopcroft_karp_matching(S, top_nodes=top)
        cover |= nx.bipartite.to_vertex_cover(S, M, top_nodes=top)
    return cover


def decode(G: nx.Graph, matched, select: str, residual: str, pos: dict) -> set:
    """Decode the matched edges of H (already normalised to i <= j and sorted) to vertices of G."""
    deg = G.degree
    selected: set = set()
    ties = []
    for (a, i), (b, j) in matched:
        if i == j:
            ties.append((a, b))
            continue
        if a in selected or b in selected:
            continue
        if select == "low":
            pick = a
        elif select == "high":
            pick = b
        else:
            da, db = deg[a], deg[b]
            if da != db:
                pick = a if da > db else b
            else:
                pick = a if select == "deg_low" else b
        selected.add(pick)
    open_ties = [(a, b) for a, b in ties if a not in selected and b not in selected]
    if residual == "konig":
        selected |= _konig_or_occ(open_ties, pos)
    elif residual == "occ":
        selected |= _greedy_cover(open_ties, pos)
    else:  # gdeg
        for a, b in open_ties:
            if a in selected or b in selected:
                continue
            da, db = deg[a], deg[b]
            selected.add(a if (da, -pos[a]) >= (db, -pos[b]) else b)
    return selected


def repair(G: nx.Graph, adj: dict, raw: set, how: str, pos: dict) -> set:
    if how == "greedy":
        uncovered = [(u, v) for u, v in G.edges() if u not in raw and v not in raw]
        raw = raw | _greedy_cover(uncovered, pos)
    return prune_redundant_vertices(adj, raw)


def all_variants(G: nx.Graph) -> tuple[dict[str, set], dict]:
    """Covers of every variant on G (no self-loops, no isolated vertices) and the split statistics."""
    info: dict = {}
    H = component_split.build_split_graph(G, info)
    pos = {u: k for k, u in enumerate(G)}
    matched = []
    for x, y in nx.approximation.min_maximal_matching(H):
        (a, i), (b, j) = x, y
        if (j, pos[b]) < (i, pos[a]):
            (a, i), (b, j) = (b, j), (a, i)
        matched.append(((a, i), (b, j)))
    matched.sort(key=lambda e: (e[0][1], e[1][1], pos[e[0][0]], pos[e[1][0]]))
    adj = {v: set(G[v]) for v in G}
    covers = {}
    raw_cache: dict = {}
    for s, r, p in itertools.product(SELECT, RESIDUAL, REPAIR):
        if (s, r) not in raw_cache:
            raw_cache[(s, r)] = decode(G, matched, s, r, pos)
        covers[f"{s}/{r}/{p}"] = repair(G, adj, raw_cache[(s, r)], p, pos)
    # Pipelines.
    legacy_info: dict = {}
    legacy = component_split.component_split_covers(G, legacy_info, variants=component_split.LEGACY_VARIANTS,
                                                    exact_stop=False, complete=False)
    covers["c0_old"] = min((prune_redundant_vertices(adj, c) for c in legacy.values()), key=len)
    info["r_prime_legacy"] = legacy_info["r_prime"]
    covers["c0_new/decoded"] = min((prune_redundant_vertices(adj, c)
                                    for c in component_split.component_split_covers(G).values()), key=len)
    covers["c0_new/ls"] = c0_cover(G, adj, set(G))  # no fallback: c1 is never smaller than V
    c1 = prune_redundant_vertices(adj, maximal_matching_vertex_cover(G))
    new_info: dict = {}
    covers["c0_new"] = c0_cover(G, adj, c1, info=new_info)
    covers["c1"] = c1
    info["c0_new_fallback_c1"] = new_info["fallback_c1"]
    info["c0_new_ls_gain"] = new_info["ls_gain"]
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
    if exact:
        tau, _ = cc.exact_vertex_cover_size(G)
        lb = tau
    else:
        lb, method, _ = cc.best_lower_bound(G, spectral=spectral)
        tau = lb if method.startswith("bipartite") else None
    sizes = {v: len(c) for v, c in covers.items()}
    upper = min(sizes.values())  # a vertex cover of G, so tau <= upper
    row = {"group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
           "tau": tau, "lower_bound": lb, "upper_bound": upper, "exact_flag": tau is not None,
           "lower_bound_method": "exact" if exact else method,
           "r_prime": info["r_prime"], "target": info["target"],
           "r_prime_legacy": info["r_prime_legacy"], "r": nx.number_connected_components(G),
           "c0_new_fallback_c1": info["c0_new_fallback_c1"], "c0_new_ls_gain": info["c0_new_ls_gain"],
           "sizes": sizes, "seconds": round(seconds, 4),
           "edges": [list(e) for e in G.edges()] if G.number_of_nodes() <= 14 else None}
    best = min(sizes.values())
    print(f"  [c0fix] {group[:30]:30} {name:24} n={row['n']:>6} m={row['m']:>7} "
          f"{('tau=' + str(tau)) if tau is not None else ('L=' + str(lb)):>9} r'={info['r_prime']:<6} "
          f"c0 old={sizes['c0_old']} new={sizes['c0_new']} (decoded {sizes['c0_new/decoded']}, ls {sizes['c0_new/ls']})"
          f" c1={sizes['c1']}  best={best}"
          + (f"  2tau-r'={2 * tau - info['r_prime']}" if tau is not None else
             f"  2L-r'={2 * lb - info['r_prime']} 2U-r'={2 * upper - info['r_prime']} c0_new "
             + verdict(sizes["c0_new"], lb, upper, info["r_prime"])), flush=True)
    return row


def verdict(size: int, lower: int, upper: int, r_prime: int) -> str:
    """Bound |cover| <= 2 tau - r' without tau, from L <= tau <= U.

    certified  size <= 2L - r'  (then size <= 2 tau - r': the bound holds, proven)
    refuted    size >  2U - r'  (then size >  2 tau - r': the bound fails, proven)
    undecided  otherwise
    """
    if size <= 2 * lower - r_prime:
        return "certified"
    if size > 2 * upper - r_prime:
        return "refuted"
    return "undecided"


def _bipartite_job(name: str, max_n: int, quick: bool, seed: int, spectral: bool) -> dict | None:
    """car_bipartite.py's family ``name`` (same construction as car_c10_bags.py); tau exact by Koenig."""
    from car_bipartite import instances as bip_instances, shuffle_labels

    rng = random.Random(seed)
    for nm, group, make in bip_instances(max_n, quick, rng):
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


def rp(row: dict, variant: str) -> int:
    """r' of the split graph the variant was decoded from (the first c0 used overshooting splits)."""
    return row["r_prime_legacy"] if variant == "c0_old" else row["r_prime"]


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for v in VARIANTS:
        exact_rows = [r for r in rows if r.get("exact_flag")]
        ratios = [(r["sizes"][v] / r["lower_bound"]) if r["lower_bound"] else 1.0 for r in rows]
        viol = [r for r in exact_rows if r["sizes"][v] > 2 * r["tau"] - rp(r, v)]
        proven = [r for r in exact_rows if r["sizes"][v] > 2 * r["tau"] - r["r"]]
        above = [r for r in exact_rows if r["sizes"][v] > 1.5 * r["tau"]]
        smallest = min(viol, key=lambda r: (r["n"], r["m"]), default=None)
        smallest15 = min(above, key=lambda r: (r["n"], r["m"]), default=None)
        worst = max(rows, key=lambda r: (r["sizes"][v] / r["lower_bound"]) if r["lower_bound"] else 1.0, default=None)
        inexact = [r for r in rows if not r.get("exact_flag")]
        verdicts = [verdict(r["sizes"][v], r["lower_bound"], r["upper_bound"], rp(r, v)) for r in inexact]
        large_exact = [r for r in exact_rows if r["n"] > 14]
        out.append({
            "variant": v, "is_c0": v in C0_VARIANTS, "instances": len(rows), "exact_instances": len(exact_rows),
            "bound_violations": len(viol), "above_1_5": len(above), "proven_bound_violations": len(proven),
            "fallback_c1": sum(r["c0_new_fallback_c1"] for r in rows) if v == "c0_new" else None,
            "large_exact_instances": len(large_exact),
            "large_exact_violations": sum(r["sizes"][v] > 2 * r["tau"] - rp(r, v) for r in large_exact),
            "inexact_instances": len(inexact), "certified": verdicts.count("certified"),
            "refuted": verdicts.count("refuted"), "undecided": verdicts.count("undecided"),
            "refuted_instances": [r["name"] for r, x in zip(inexact, verdicts) if x == "refuted"],
            "undecided_instances": [r["name"] for r, x in zip(inexact, verdicts) if x == "undecided"],
            "optimal": sum(r["sizes"][v] == r["tau"] for r in exact_rows),
            "max_ratio": max(ratios, default=1.0), "mean_ratio": sum(ratios) / len(ratios) if ratios else 1.0,
            "worst_instance": worst["name"] if worst else None,
            "smallest_violation": None if smallest is None else
            {"name": smallest["name"], "n": smallest["n"], "tau": smallest["tau"], "r_prime": rp(smallest, v),
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
    ap.add_argument("--skip-bipartite", action="store_true",
                    help="skip car_bipartite.py's large bipartite families (tau exact by Koenig)")
    ap.add_argument("--bip-max-n", type=int, default=10_000,
                    help="largest bipartite instance (default 10000; car_bipartite sizes 100, 1000, 10000, 50000)")
    ap.add_argument("--seed", type=int, default=SEED)
    cc.add_timeout_argument(ap)
    args = ap.parse_args()
    spectral = not args.no_spectral

    started = time.time()
    rows: list[dict] = []

    def add(row, exact):  # noqa: ARG001  (row["exact_flag"] is set from tau in evaluate)
        if row is not None:
            rows.append(row)

    for name, group, G in stress_instances():
        add(evaluate(name, group, G, exact=True), True)
    for name, group, G in cc.small_instances(random.Random(cc.SEED), big=not args.quick):
        add(evaluate(name, group, G, exact=True), True)
    if not args.quick:
        for name, group, G in random_instances(args.trials, random.Random(args.seed)):
            add(evaluate(name, group, G, exact=True), True)
    skipped: list[str] = []
    if not args.quick and not args.skip_large:
        with cc.InstanceRunner(args.timeout) as runner:
            for name, group, _ in cc.large_instances(args.max_n):
                row = runner.run(_large_job, name, args.max_n, spectral)
                if row is None:
                    print(f"  [c0fix] {name}: skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    add(row, False)
                gc.collect()
    if not args.quick and not args.skip_bipartite:
        from car_bipartite import instances as bip_instances

        with cc.InstanceRunner(args.timeout) as runner:
            for name, _, _ in bip_instances(args.bip_max_n, False, random.Random(args.seed)):
                row = runner.run(_bipartite_job, name, args.bip_max_n, False, args.seed, spectral)
                if row is None:
                    print(f"  [c0fix] {name}: skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    add(row, False)
                gc.collect()

    summary = summarise(rows)
    print("\nVariants (select/residual/repair on the exact-stop split graph) and pipelines, best first;"
          " * = c0_old (first c0) and c0_new (salvador's c0)")
    head = (f"{'variant':26} {'exact':>6} {'>2tau-r_':>9} {'>2tau-r':>8} {'>1.5tau':>8} {'optimal':>8}"
            f" {'max ratio':>10} {'mean':>7} {'lg exact':>8} {'lg viol':>7} {'no tau':>6} {'cert':>5} {'refut':>5}"
            f" {'undec':>5}  worst")
    print(head)
    print("-" * len(head))
    for s in summary:
        print(f"{s['variant'] + (' *' if s['is_c0'] else ''):26} {s['exact_instances']:>6} {s['bound_violations']:>9}"
              f" {s['proven_bound_violations']:>8} {s['above_1_5']:>8} {s['optimal']:>8} {s['max_ratio']:>10.4f} {s['mean_ratio']:>7.4f}"
              f" {s['large_exact_instances']:>8} {s['large_exact_violations']:>7} {s['inexact_instances']:>6}"
              f" {s['certified']:>5} {s['refuted']:>5} {s['undecided']:>5}  {s['worst_instance']}")
    for name in ("c0_old", "c0_new"):
        s_ = next(x for x in summary if x["variant"] == name)
        print(f"{name}: violations of 2 tau - r' {s_['bound_violations']}, of 2 tau - r {s_['proven_bound_violations']}"
              f", above 1.5 tau {s_['above_1_5']}, max ratio {s_['max_ratio']:.4f}"
              + (f", fallback to c1 on {s_['fallback_c1']} instances" if name == "c0_new" else ""))
        print(f"  large graphs with exact tau (n > 14): {s_['large_exact_instances']}, violations "
              f"{s_['large_exact_violations']}; without exact tau: {s_['inexact_instances']} -> certified "
              f"{s_['certified']} (|c| <= 2L - r'), refuted {s_['refuted']} (|c| > 2U - r'), undecided {s_['undecided']}")
        for key in ("refuted_instances", "undecided_instances"):
            if s_[key]:
                print(f"  {key}: {', '.join(s_[key])}")
        for key in ("smallest_violation", "smallest_above_1_5"):
            if s_[key]:
                print(f"  {key}: {s_[key]}")
    best = summary[0]
    print(f"\nBest variant: {best['variant']}  (bound violations {best['bound_violations']}, above 1.5 tau "
          f"{best['above_1_5']})")
    for key in ("smallest_violation", "smallest_above_1_5"):
        if best[key]:
            print(f"  {key}: {best[key]}")
    if skipped:
        print(f"Skipped (more than {args.timeout:g} s): {', '.join(skipped)}")

    out = cc.OUT_DIR
    stem = "car_c0_fix"
    (out / f"{stem}.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": args.seed,
         "variants": VARIANTS, "c0_variants": list(C0_VARIANTS), "quick": args.quick, "trials": args.trials,
         "skipped_instances": skipped, "summary": summary,
         "rows": [{k: v for k, v in r.items() if k != "edges"} for r in rows],
         "elapsed_seconds": time.time() - started}, indent=1, default=str), encoding="utf-8")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "name", "n", "m", "tau", "lower_bound", "exact", "r_prime", "variant", "size", "ratio",
                    "bound_2tau_minus_rprime", "bound_holds", "above_1_5", "upper_bound", "verdict_without_tau"])
        for r in rows:
            for v in VARIANTS:
                size = r["sizes"][v]
                ex = r["exact_flag"]
                b = 2 * r["tau"] - rp(r, v) if ex else None
                w.writerow([r["group"], r["name"], r["n"], r["m"], r["tau"], r["lower_bound"], ex, r["r_prime"], v,
                            size, round(size / r["lower_bound"], 6) if r["lower_bound"] else 1.0,
                            b, (size <= b) if ex else "", (size > 1.5 * r["tau"]) if ex else "",
                            r["upper_bound"],
                            "" if ex else verdict(size, r["lower_bound"], r["upper_bound"], rp(r, v))])
    with open(out / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["variant", "is_c0", "instances", "exact_instances", "bound_violations", "proven_bound_violations",
                  "above_1_5", "fallback_c1", "optimal",
                  "max_ratio", "mean_ratio", "large_exact_instances", "large_exact_violations",
                  "inexact_instances", "certified", "refuted", "undecided", "worst_instance"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
