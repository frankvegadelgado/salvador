"""car/ c10 bag study: is every bag of the c10 dynamic program at most 6 nodes?

c10 (salvador/eds_treewidth.py) solves the labelled, weighted edge-dominating-set
gadget of every bipartite component by bucket elimination. Its bags are capped
at MAX_BAG = 6 nodes (fill degree <= 5); the program raises AssertionError if a
joined table ever has more than MAX_BAG nodes. This experiment checks, and
measures:

Part A  (bags) every instance of car_bipartite.py's families: the largest bag
        built (expected <= 6; an AssertionError would stop the run), the largest
        fill degree, the core size, whether the component DPs were exact, the
        cover size against tau(G) (Hopcroft-Karp / Koenig, reference only), the
        validity of the cover and the time.
Part B  (exactness and weights) small random bipartite graphs: when c10 is exact
        its cover must be a minimum vertex cover (|C| = tau) and, among the
        minimum covers, have the fewest doubly covered edges (brute force).
        Expected: 0 non-minimum covers, 0 non-minimum overlaps.
Part C  (necessary bag size) a certified lower bound (minor-min-width) on the
        treewidth of the gadget of the instances of part A with at most
        --lb-max-n vertices: an exact decomposition needs a bag of at least
        LB + 1 nodes, so LB + 1 > 6 means the cap forces a core there.

Run from the repository root:

    python car/car_c10_bags.py --quick           # sizes 100 and 1000
    python car/car_c10_bags.py --max-n 10000
    python car/car_c10_bags.py --timeout 60      # skip instances taking more than 60 s

Every instance of part A (construction, c10, tau and the lower bound) runs in a
worker process with a time limit (``--timeout``, default 120 s, 0 = no limit);
an instance that takes longer is skipped and listed in car_c10_bags.json
(``skipped_instances``). Each instance has its own random generator derived from
the seed and its name.

Outputs (in car/): car_c10_bags.json, car_c10_bags.csv (one row per instance)
and car_c10_bags_summary.csv (one row per family). The gadget is eliminated in
min-degree order (eds_treewidth.DEFAULT_ORDER).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time

import networkx as nx

import car_common as cc  # noqa: F401  (puts the repository root on sys.path)
from car_bipartite import exact_tau, instances, shuffle_labels
from salvador import eds_treewidth

SEED = 20261010


def matching_cover(adj) -> set:
    """Both endpoints of a greedy maximal matching: a valid reference cover."""
    matched: set = set()
    for u in adj:
        if u in matched:
            continue
        for w in adj[u]:
            if w != u and w not in matched:
                matched.add(u)
                matched.add(w)
                break
    return matched


def is_cover(adj, cover) -> bool:
    return all(u in cover or w in cover for u in adj for w in adj[u] if w != u)


def overlap(adj, cover) -> int:
    return sum(1 for u in adj for w in adj[u] if u < w and u in cover and w in cover)


def gadget_graph(G: nx.Graph) -> nx.Graph:
    adj = {u: set(G[u]) for u in G}
    H = nx.Graph()
    offset = 0
    for comp in nx.connected_components(G):
        g = eds_treewidth.build_gadget(list(comp), adj)
        H.add_edges_from((a + offset, b + offset) for a, b in g["gedges"])
        offset += g["n_nodes"]
    return H


def minor_min_width(G: nx.Graph) -> int:
    """Treewidth lower bound (minor-min-width): repeatedly contract a minimum-degree vertex into its
    minimum-degree neighbour; the largest minimum degree seen bounds the treewidth from below.

    In-place contraction on adjacency sets with a lazy heap, about O(m log n) overall (contracting
    with networkx's ``contracted_nodes`` copies the graph at every step, which is quadratic)."""
    import heapq  # noqa: PLC0415

    adj = {v: set(G[v]) - {v} for v in G}
    heap = [(len(nb), v) for v, nb in adj.items()]
    heapq.heapify(heap)
    lb = 0
    while len(adj) > 1:
        d, v = heapq.heappop(heap)
        if v not in adj or d != len(adj[v]):
            continue  # stale entry
        lb = max(lb, d)
        nbrs = adj.pop(v)
        if not nbrs:
            continue
        u = min(nbrs, key=lambda w: len(adj[w]))
        for w in nbrs:
            adj[w].discard(v)
            if w != u and u not in adj[w]:
                adj[w].add(u)
                adj[u].add(w)
        for w in nbrs:
            heapq.heappush(heap, (len(adj[w]), w))
    return lb


def run_c10(G: nx.Graph) -> dict:
    adj = {u: set(G[u]) for u in G}
    ref = matching_cover(adj)
    info: dict = {}
    started = time.time()
    cover = eds_treewidth.eds_treewidth_cover(adj, ref, info=info)
    seconds = time.time() - started
    return {"cover": cover, "reference": len(ref), "seconds": seconds, "valid": is_cover(adj, cover),
            "overlap": overlap(adj, cover), **info}


def _instance_job(name: str, args) -> dict:
    """Build instance ``name`` of car_bipartite.py's families (own generator) and measure it; runs in
    the worker process."""
    rng = random.Random(args.seed)
    for nm, group, make in instances(args.max_n, args.quick, rng):
        if nm != name:
            continue
        rng.seed(f"{args.seed}:{name}")
        G = shuffle_labels(nx.convert_node_labels_to_integers(make()), rng)
        G.remove_edges_from(nx.selfloop_edges(G))
        r = run_c10(G)
        tau = exact_tau(G)
        lb = minor_min_width(gadget_graph(G)) if G.number_of_nodes() <= args.lb_max_n else None
        return {"instance": name, "family": group, "n": G.number_of_nodes(), "m": G.number_of_edges(),
                "max_bag": r["max_bag"], "width": r["width"], "core_size": r["core_size"], "exact": r["exact"],
                "exact_components": r["exact_components"], "bipartite_components": r["bipartite_components"],
                "cover": len(r["cover"]), "tau": tau, "reference": r["reference"], "valid": r["valid"],
                "overlap": r["overlap"], "gadget_tw_lb": lb, "seconds": round(r["seconds"], 3)}
    raise KeyError(name)


def part_a(args) -> tuple[list[dict], list[str]]:
    print(f"Part A: largest bag of c10 (MAX_BAG = {eds_treewidth.MAX_BAG}, {eds_treewidth.DEFAULT_ORDER} order), "
          f"time limit {args.timeout:g} s per instance")
    rows = []
    skipped = []
    names = [name for name, _, _ in instances(args.max_n, args.quick, random.Random(args.seed))]
    with cc.InstanceRunner(args.timeout) as runner:
        for name in names:
            row = runner.run(_instance_job, name, args)
            if row is None:
                print(f"  {name:40} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
                continue
            rows.append(row)
            lb = row["gadget_tw_lb"]
            print(f"  {name:40} bag {row['max_bag']}  core {row['core_size']:>7}  exact {str(row['exact']):5}  "
                  f"|C| {row['cover']:>6}  tau {row['tau']:>6}  ref {row['reference']:>6}  valid {row['valid']}  "
                  f"tw>= {lb if lb is not None else '-':>3}  {row['seconds']:.2f}s", flush=True)
    return rows, skipped


def part_b(args, rng) -> dict:
    print("Part B: exactness and overlap on small bipartite graphs")
    tested = exact = not_min = not_min_overlap = 0
    for _ in range(args.trials):
        n1, n2 = rng.randint(1, 7), rng.randint(1, 7)
        G = nx.bipartite.random_graph(n1, n2, rng.uniform(0.2, 0.8), seed=rng.randrange(2**31))
        G.remove_nodes_from([v for v in list(G) if G.degree(v) == 0])
        if G.number_of_edges() == 0:
            continue
        adj = {u: set(G[u]) for u in G}
        r = run_c10(G)
        tested += 1
        assert r["valid"] and r["max_bag"] <= eds_treewidth.MAX_BAG
        if not r["exact"]:
            continue
        exact += 1
        tau = exact_tau(G)
        if len(r["cover"]) != tau:
            not_min += 1
            continue
        best = min(overlap(adj, set(c)) for c in itertools.combinations(sorted(G), tau) if is_cover(adj, set(c)))
        not_min_overlap += r["overlap"] != best
    result = {"tested": tested, "exact": exact, "not_minimum_cover": not_min, "not_minimum_overlap": not_min_overlap}
    print(f"  tested {tested}, exact {exact}, non-minimum covers {not_min} (expected 0), "
          f"non-minimum overlaps {not_min_overlap} (expected 0)")
    return result


def summarise(rows: list[dict]) -> list[dict]:
    fam: dict[str, list[dict]] = {}
    for row in rows:
        fam.setdefault(row["family"], []).append(row)
    out = []
    for family, rs in fam.items():
        lbs = [r["gadget_tw_lb"] for r in rs if r["gadget_tw_lb"] is not None]
        out.append({"family": family, "instances": len(rs), "max_bag": max(r["max_bag"] for r in rs),
                    "all_bags_le_6": all(r["max_bag"] <= eds_treewidth.MAX_BAG for r in rs),
                    "exact": sum(r["exact"] for r in rs), "cover_eq_tau": sum(r["cover"] == r["tau"] for r in rs),
                    "invalid": sum(not r["valid"] for r in rs),
                    "max_gadget_tw_lb": max(lbs) if lbs else None,
                    "needs_bag_gt_6": sum(1 for x in lbs if x + 1 > eds_treewidth.MAX_BAG),
                    "seconds": round(sum(r["seconds"] for r in rs), 2)})
    return out


def main() -> None:
    helper = argparse.ArgumentParser(description="c10 bag study (bags <= MAX_BAG = 6).")
    helper.add_argument("--max-n", type=int, default=1000, help="largest instance size (default: 1000)")
    helper.add_argument("--quick", action="store_true", help="small sizes only")
    helper.add_argument("--trials", type=int, default=300, help="random graphs of part B (default: 300)")
    helper.add_argument("--lb-max-n", type=int, default=1000,
                        help="largest instance for the treewidth lower bound of part C (default: 1000)")
    helper.add_argument("--seed", type=int, default=SEED)
    cc.add_timeout_argument(helper)
    args = helper.parse_args()
    rng = random.Random(args.seed)

    rows, skipped = part_a(args)
    exactness = part_b(args, rng)
    summary = summarise(rows)
    print("\nSummary per family")
    print(f"{'family':45} {'N':>3} {'max bag':>7} {'<=6':>4} {'exact':>5} {'=tau':>4} {'invalid':>7} "
          f"{'tw LB':>5} {'need>6':>6}")
    for s in summary:
        print(f"{s['family']:45} {s['instances']:>3} {s['max_bag']:>7} {str(s['all_bags_le_6']):>4} {s['exact']:>5} "
              f"{s['cover_eq_tau']:>4} {s['invalid']:>7} {str(s['max_gadget_tw_lb']):>5} {s['needs_bag_gt_6']:>6}")
    overall = max((r["max_bag"] for r in rows), default=0)
    print(f"\nLargest bag over all instances: {overall} (MAX_BAG = {eds_treewidth.MAX_BAG}); "
          f"skipped (time limit): {len(skipped)}" + (f" -- {', '.join(skipped)}" if skipped else ""))

    out = cc.OUT_DIR
    stem = "car_c10_bags"
    with open(out / f"{stem}.json", "w") as fh:
        json.dump({"environment": cc.environment(), "max_bag": eds_treewidth.MAX_BAG, "order": eds_treewidth.DEFAULT_ORDER,
                   "timeout": args.timeout,
                   "skipped_instances": skipped, "rows": rows,
                   "exactness": exactness, "summary": summary}, fh, indent=1)
    for path, data in ((out / f"{stem}.csv", rows), (out / f"{stem}_summary.csv", summary)):
        with open(path, "w", newline="") as fh:
            if data:
                writer = csv.DictWriter(fh, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)


if __name__ == "__main__":
    main()
