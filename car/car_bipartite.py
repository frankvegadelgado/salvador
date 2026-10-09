"""car/ bipartite study: does Salvador need an exact bipartite strategy?

Salvador 0.1.0 no longer has a bipartite-exact strategy (the former c10,
Hopcroft-Karp + Koenig, which is O(m sqrt(n)) and therefore not linear). Its
c10 is now the edge-dominating-set gadget solved by a linear-time treewidth
DP on the bipartite components (a minimum EDS when the DP core is empty,
decoded to a cover). This experiment measures, on bipartite graphs only, how
far the linear-time ensemble is from the optimum, how far c10's own decoded
cover (``c10_raw``) is from it, and how often c10 and c11 are exact.

For every instance the minimum vertex cover size tau(G) is computed *for
reference only* (it is never given to Salvador) by a Hopcroft-Karp maximum
matching: on a bipartite graph tau(G) = nu(G) by Koenig's theorem, so the
reported ratio |C| / tau(G) is exact.

The cover evaluated is ``c12``, the final cover of
``salvador.algorithm.final_candidates`` (the same cover
``find_vertex_cover`` returns).

Per instance the report gives every strategy size c1..c12, tau(G), the c12
ratio and excess |c12| - tau(G), which strategies already reach tau(G),
c10's decoded cover before the comparison with its reference (``c10_raw``),
whether c10 (minimum EDS) and c11 (minimum cover) were exact and their core
sizes, and the time.
Per family it gives how many instances c12 solved exactly, the worst ratio,
the total excess, and how many instances each strategy solved exactly.

Families (all bipartite, labels shuffled):

* random bipartite G(n1, n2, p), balanced and unbalanced, average degree 3 to 20;
* random d-regular bipartite graphs (union of d random perfect matchings);
* bipartite graphs with power-law degrees (bipartite configuration model);
* bipartite double covers G x K2 of random 3- and 5-regular graphs;
* 2D/3D grids, grids with random edge deletions, hypercubes Q_k and random
  subgraphs of hypercubes;
* random trees, even cycles, complete bipartite K_{a,b}, crown graphs;
* "hidden matching" graphs: a perfect matching plus random edges from one
  side to the other, the shape on which greedy rules are easily misled.

Run from the repository root:

    python car/car_bipartite.py                  # default sizes (up to --max-n)
    python car/car_bipartite.py --quick          # small sizes only
    python car/car_bipartite.py --max-n 5000

Outputs (in car/): car_bipartite.json, car_bipartite.csv (one row per
instance) and car_bipartite_summary.csv (one row per family).
"""

from __future__ import annotations

import argparse
import json
import random
import time

import networkx as nx

import car_common as cc
from salvador.algorithm import CANDIDATE_NAMES, DEFAULT_EDS_BUDGET, DEFAULT_EDS_MAX_WIDTH, final_candidates

SEED = 20261009
NAMES = list(CANDIDATE_NAMES)  # c1..c12


# ----------------------------------------------------------------------------
# Generators (every graph returned is bipartite)
# ----------------------------------------------------------------------------
def shuffle_labels(G: nx.Graph, rng: random.Random) -> nx.Graph:
    nodes = list(G.nodes())
    perm = list(range(len(nodes)))
    rng.shuffle(perm)
    return nx.relabel_nodes(G, {v: perm[i] for i, v in enumerate(nodes)})


def random_bipartite(n1: int, n2: int, avg_deg: float, rng: random.Random) -> nx.Graph:
    p = min(1.0, avg_deg * (n1 + n2) / (2.0 * n1 * n2))
    return nx.bipartite.random_graph(n1, n2, p, seed=rng.randrange(2**31))


def regular_bipartite(half: int, d: int, rng: random.Random) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(2 * half))
    for _ in range(d):
        perm = list(range(half))
        rng.shuffle(perm)
        G.add_edges_from((i, half + perm[i]) for i in range(half))
    return G


def power_law_bipartite(n: int, exponent: float, rng: random.Random) -> nx.Graph:
    half = n // 2
    def seq() -> list[int]:
        return [max(1, min(half - 1, int(rng.paretovariate(exponent - 1)))) for _ in range(half)]
    a, b = seq(), seq()
    diff = sum(a) - sum(b)
    i = 0
    while diff != 0:  # balance the two degree sums
        if diff > 0:
            b[i % half] += 1
            diff -= 1
        else:
            a[i % half] += 1
            diff += 1
        i += 1
    M = nx.bipartite.configuration_model(a, b, seed=rng.randrange(2**31))
    return nx.Graph(M)


def double_cover(n: int, d: int, rng: random.Random) -> nx.Graph:
    H = nx.random_regular_graph(d, n - (n * d) % 2, seed=rng.randrange(2**31))
    return nx.tensor_product(H, nx.complete_graph(2))


def thinned_grid(rows: int, cols: int, keep: float, rng: random.Random) -> nx.Graph:
    G = nx.grid_2d_graph(rows, cols)
    G.remove_edges_from([e for e in list(G.edges()) if rng.random() > keep])
    return G


def hypercube_subgraph(k: int, keep: float, rng: random.Random) -> nx.Graph:
    G = nx.hypercube_graph(k)
    G.remove_edges_from([e for e in list(G.edges()) if rng.random() > keep])
    return G


def crown(k: int) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from((("a", i), ("b", j)) for i in range(k) for j in range(k) if i != j)
    return G


def hidden_matching(half: int, extra_deg: float, rng: random.Random) -> nx.Graph:
    """Perfect matching a_i - b_i plus random a -> b edges; tau = half."""
    G = nx.Graph()
    G.add_edges_from((i, half + i) for i in range(half))
    for i in range(half):
        k = int(extra_deg) + (1 if rng.random() < extra_deg - int(extra_deg) else 0)
        for j in rng.sample(range(half), min(k, half)):
            G.add_edge(i, half + j)
    return G


def instances(max_n: int, quick: bool, rng: random.Random):
    sizes = [n for n in ((100, 1000) if quick else (100, 1000, 10_000, 50_000)) if n <= max_n]
    for n in sizes:
        reps = 3 if n <= 1000 else 1
        for r in range(reps):
            for deg in (3, 6, 20):
                if deg * n <= 2_000_000:
                    yield f"gnp_bip_bal_n{n}_d{deg}_r{r}", "random bipartite (balanced)", \
                        lambda n=n, deg=deg: random_bipartite(n // 2, n - n // 2, deg, rng)
                    yield f"gnp_bip_unbal_n{n}_d{deg}_r{r}", "random bipartite (1:2)", \
                        lambda n=n, deg=deg: random_bipartite(n // 3, n - n // 3, deg, rng)
            for d in (3, 4, 8):
                yield f"reg_bip_n{n}_d{d}_r{r}", "random d-regular bipartite", \
                    lambda n=n, d=d: regular_bipartite(n // 2, d, rng)
            yield f"powerlaw_bip_n{n}_r{r}", "power-law bipartite", lambda n=n: power_law_bipartite(n, 2.5, rng)
            for d in (3, 5):
                yield f"double_cover_n{n}_d{d}_r{r}", "double cover G x K2 of random regular", \
                    lambda n=n, d=d: double_cover(n // 2, d, rng)
            for ed in (1.5, 4.0):
                yield f"hidden_matching_n{n}_e{ed}_r{r}", "hidden perfect matching + random edges", \
                    lambda n=n, ed=ed: hidden_matching(n // 2, ed, rng)
            yield f"tree_n{n}_r{r}", "random tree", lambda n=n: nx.random_labeled_tree(n, seed=rng.randrange(2**31))
            side = int(n ** 0.5)
            yield f"thinned_grid_{side}x{side}_r{r}", "2D grid, 70% edges kept", \
                lambda side=side: thinned_grid(side, side, 0.7, rng)
        side = int(n ** 0.5)
        yield f"grid_{side}x{side}", "2D grid", lambda side=side: nx.grid_2d_graph(side, side)
        c = max(2, round(n ** (1 / 3)))
        yield f"grid3d_{c}^3", "3D grid", lambda c=c: nx.grid_graph(dim=[c, c, c])
        yield f"even_cycle_{n}", "even cycle", lambda n=n: nx.cycle_graph(n - n % 2)
        if n <= 1000:
            yield f"K_{n // 4}_{n - n // 4}", "complete bipartite", \
                lambda n=n: nx.complete_bipartite_graph(n // 4, n - n // 4)
            yield f"crown_{n // 2}", "crown graph", lambda n=n: crown(n // 2)
    for k in (8, 10, 12, 14):
        if 2 ** k <= max_n and not (quick and k > 10):
            yield f"hypercube_Q{k}", "hypercube", lambda k=k: nx.hypercube_graph(k)
            yield f"hypercube_Q{k}_half", "hypercube, 50% edges kept", lambda k=k: hypercube_subgraph(k, 0.5, rng)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def exact_tau(G: nx.Graph) -> int:
    """tau(G) = nu(G) for bipartite G (Koenig); reference only."""
    total = 0
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        if H.number_of_edges() == 0:
            continue
        top = nx.bipartite.sets(H)[0]
        total += len(nx.bipartite.hopcroft_karp_matching(H, top_nodes=top)) // 2
    return total


def run_instance(name: str, group: str, G: nx.Graph, rng: random.Random, args) -> dict:
    G = shuffle_labels(nx.convert_node_labels_to_integers(G), rng)
    G.remove_edges_from(nx.selfloop_edges(G))
    assert nx.is_bipartite(G), name
    tau = exact_tau(G)
    info: dict = {}
    t0 = time.perf_counter()
    cands = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info,
                             eds_max_width=args.eds_max_width, eds_budget=args.eds_budget)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in NAMES}
    for k in NAMES:
        C = cands[k]
        if not all(u in C or v in C for u, v in G.edges()):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
        if len(C) < tau:
            raise AssertionError(f"{name}: strategy {k} is below tau -- impossible")
    final = sizes["c12"]
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "tau": tau, "sizes": sizes, "final": final,
        "ratio": final / tau if tau else 1.0, "excess": final - tau,
        "optimal": final == tau,
        "optimal_strategies": [k for k in NAMES if sizes[k] == tau],
        "c10_exact": bool(info.get("c10_exact", True)),
        "c10_raw": info.get("c10_raw_size", sizes["c10"]),
        "c10_core_size": info.get("c10_core_size", 0),
        "c10_width": info.get("c10_width", 0),
        "c11_exact": bool(info.get("c11_exact", True)),
        "c11_core_size": info.get("c11_core_size", 0),
        "elapsed_seconds": elapsed,
    }
    print(f"  [bipartite] {group[:34]:34} {name:34} n={row['n']:>7,} tau={tau:>7,}"
          f" c12={sizes['c12']:>7,} ratio={row['ratio']:.6f} excess={row['excess']:<4}"
          f" c10={'exact' if row['c10_exact'] else 'core=' + str(row['c10_core_size'])}"
          f" c11={'exact' if row['c11_exact'] else 'core=' + str(row['c11_core_size'])} {elapsed:6.2f}s", flush=True)
    return row


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(grows)
        out[g] = {
            "instances": k,
            "optimal": sum(r["optimal"] for r in grows),
            "optimal_rate": sum(r["optimal"] for r in grows) / k if k else 0.0,
            "max_ratio": max((r["ratio"] for r in grows), default=1.0),
            "mean_ratio": sum(r["ratio"] for r in grows) / k if k else 1.0,
            "total_excess": sum(r["excess"] for r in grows),
            "total_tau": sum(r["tau"] for r in grows),
            "c10_exact": sum(r["c10_exact"] for r in grows),
            "c10_raw_optimal": sum(r["c10_raw"] == r["tau"] for r in grows),
            "c10_raw_excess": sum(r["c10_raw"] - r["tau"] for r in grows),
            "c10_mean_core_fraction": sum(r["c10_core_size"] / r["n"] for r in grows) / k if k else 0.0,
            "c11_exact": sum(r["c11_exact"] for r in grows),
            "optimal_by_strategy": {s: sum(s in r["optimal_strategies"] for r in grows) for s in NAMES},
            "seconds": sum(r["elapsed_seconds"] for r in grows),
        }
    return out


def print_table(summary: dict) -> None:
    header = (f"{'family':42} {'inst':>5} {'opt':>5} {'opt%':>6} {'max ratio':>10} {'excess':>7}"
              f" {'c10ex':>5} {'raw10':>5} {'c11ex':>5}  optimal by strategy (c1..c12)")
    print("\nBipartite graphs: final cover c12 against the exact optimum tau = nu (Koenig)")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        by = " ".join(f"{s['optimal_by_strategy'][k]:>4}" for k in NAMES)
        print(f"{g[:42]:42} {s['instances']:>5} {s['optimal']:>5} {100 * s['optimal_rate']:5.1f}%"
              f" {s['max_ratio']:10.6f} {s['total_excess']:>7} {s['c10_exact']:>5} {s['c10_raw_optimal']:>5} {s['c11_exact']:>5}  {by}")
    print("\nLegend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="sizes 100 and 1000 only")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest instance size (default 50000)")
    ap.add_argument("--eds-max-width", type=int, default=DEFAULT_EDS_MAX_WIDTH,
                    help=f"c10: largest elimination bag minus one (default {DEFAULT_EDS_MAX_WIDTH})")
    ap.add_argument("--eds-budget", type=float, default=DEFAULT_EDS_BUDGET,
                    help=f"c10: DP table budget in units of (n + m) (default {DEFAULT_EDS_BUDGET})")
    args = ap.parse_args()

    started = time.time()
    rng = random.Random(SEED)
    rows = [run_instance(name, group, build(), rng, args)
            for name, group, build in instances(args.max_n, args.quick, rng)]
    summary = summarise(rows)

    out = cc.OUT_DIR
    with (out / "car_bipartite.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,n,m,tau," + ",".join(NAMES)
                 + ",final,ratio,excess,optimal,optimal_strategies,c10_exact,c10_raw,c10_core_size,c11_exact,c11_core_size,elapsed_seconds\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},{r['tau']},"
                     + ",".join(str(r["sizes"][k]) for k in NAMES)
                     + f",{r['final']},{r['ratio']:.6f},{r['excess']},{r['optimal']},"
                     + f"{'+'.join(r['optimal_strategies'])},{r['c10_exact']},{r['c10_raw']},{r['c10_core_size']},"
                     + f"{r['c11_exact']},{r['c11_core_size']},"
                     + f"{r['elapsed_seconds']:.3f}\n")
    with (out / "car_bipartite_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,optimal,optimal_rate,max_ratio,mean_ratio,total_excess,total_tau,"
                 "c10_exact,c10_raw_optimal,c10_raw_excess,c11_exact," + ",".join(f"optimal_{k}" for k in NAMES) + ",seconds\n")
        for g, s in summary.items():
            fh.write(f"\"{g}\",{s['instances']},{s['optimal']},{s['optimal_rate']:.4f},{s['max_ratio']:.6f},"
                     f"{s['mean_ratio']:.6f},{s['total_excess']},{s['total_tau']},{s['c10_exact']},{s['c10_raw_optimal']},{s['c10_raw_excess']},"
                     f"{s['c11_exact']}," + ",".join(str(s["optimal_by_strategy"][k]) for k in NAMES)
                     + f",{s['seconds']:.2f}\n")
    (out / "car_bipartite.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": SEED,
         "candidates": CANDIDATE_NAMES, "eds_max_width": args.eds_max_width, "eds_budget": args.eds_budget,
         "summary": summary, "rows": rows, "elapsed_seconds": time.time() - started},
        indent=2, default=str), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
