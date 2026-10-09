"""car/ YES-type instances of the Khot-Regev reduction (Salvador 0.1.0).

Khot and Regev reduce Unique Games to Vertex Cover. A Unique Games instance
is a graph H with a label set [k] and, on every edge (v, w), a permutation
pi_vw of [k]; a labeling L satisfies the edge when L(v) = pi_vw(L(w)). The
reduction replaces every vertex v of H by a long-code block: one vertex
(v, x) for every subset x of [k], weighted by the p-biased measure
mu_p(x) = p^|x| (1 - p)^(k - |x|) with p = 1/2 - delta. For every edge
(v, w) of H, (v, x) and (w, y) are joined when x and pi_vw(y) are
**disjoint**.

YES case. If L satisfies a 1 - eta fraction of the edges, the "dictator" set

    I = {(v, x) : L(v) in x}

has weight p (almost 1/2) and spans edges only on the eta fraction of
violated constraints: for a satisfied edge, L(v) = pi_vw(L(w)) lies in both
x and pi_vw(y). So the YES graph has a vertex cover of weight about
1 - p = 1/2 + delta, while the NO case forces every cover to weigh almost 1.
These are exactly the instances on which the factor-2 barrier is decided.

Unweighted instances
--------------------
Salvador is unweighted, so every block keeps ``s`` subsets drawn from mu_p
(a *sampled* long code; duplicates allowed, each sample is its own vertex).
The planted dictator set then contains about ``p * s`` vertices per block.
Vertices of I that are joined by a violated constraint are removed
greedily, which leaves an independent set I'. Hence

    tau(G) <= number of non-isolated vertices outside I'   ("planted cover", a certified upper bound).

For small instances (``--exact-max-n``, default 600) tau(G) is also
computed by an integer program (scipy ``milp``, time limit ``--exact-time``);
when it finishes, the reported ratio |C| / tau is exact.

The Unique Games instance is generated with a planted labeling:

* constraint graph: a random d-regular graph on N vertices;
* constraints: a random permutation per edge, fixed so that
  pi_vw(L(w)) = L(v) ("permutation" UG), or a shift i -> i + c mod k
  ("linear" UG, Max-2-Lin(Z_k));
* noise: a fraction ``eta`` of the edges get a fresh random constraint (the
  YES case of UG only requires 1 - eta completeness).

Families vary the label size k, the bias p, the noise eta, the degree d and
the number of samples per block s. Every vertex label is shuffled, so the
block structure is not visible from the labels.

Per instance the report gives n, m, every strategy size c1..c12, the planted
cover n - |I'|, tau when exact, the final cover c12 (what
``find_vertex_cover`` returns), its ratio to the planted cover (a lower
bound on the true ratio) and to tau when known, and whether c12 found the
planted cover or better. It also gives c10's own decoded cover before the
comparison with its reference (``c10_raw``).

Run from the repository root:

    python car/car_yes_instances.py              # default families
    python car/car_yes_instances.py --quick      # small instances only
    python car/car_yes_instances.py --exact-max-n 0   # skip the integer program

Outputs (in car/): car_yes_instances.json, car_yes_instances.csv (one row per
instance) and car_yes_instances_summary.csv (one row per family).
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
# Unique Games with a planted labeling
# ----------------------------------------------------------------------------
def unique_games(N: int, d: int, k: int, eta: float, linear: bool, rng: random.Random):
    """Return (constraint edges [(v, w, pi)], labeling L, violated edge set)."""
    H = nx.random_regular_graph(d, N - (N * d) % 2, seed=rng.randrange(2**31))
    L = {v: rng.randrange(k) for v in H}
    cons = []
    violated = set()
    for v, w in H.edges():
        if rng.random() < eta:
            # Fresh random constraint (noise); may or may not be satisfied by L.
            if linear:
                c = rng.randrange(k)
                pi = tuple((i + c) % k for i in range(k))
            else:
                pi = list(range(k))
                rng.shuffle(pi)
                pi = tuple(pi)
        elif linear:
            c = (L[v] - L[w]) % k
            pi = tuple((i + c) % k for i in range(k))
        else:
            pi = list(range(k))
            rng.shuffle(pi)
            j = pi.index(L[v])
            pi[j], pi[L[w]] = pi[L[w]], pi[j]  # now pi[L[w]] = L[v]
            pi = tuple(pi)
        if pi[L[w]] != L[v]:
            violated.add((v, w))
        cons.append((v, w, pi))
    return H, cons, L, violated


def sample_subset(k: int, p: float, rng: random.Random) -> int:
    x = 0
    for i in range(k):
        if rng.random() < p:
            x |= 1 << i
    return x


def apply_perm(y: int, pi: tuple[int, ...]) -> int:
    out = 0
    i = 0
    while y:
        if y & 1:
            out |= 1 << pi[i]
        y >>= 1
        i += 1
    return out


def khot_regev_yes(N: int, d: int, k: int, p: float, eta: float, s: int, linear: bool,
                   rng: random.Random):
    """Sampled Khot-Regev graph of a planted UG instance.

    Returns (G, planted independent set I').
    """
    H, cons, L, violated = unique_games(N, d, k, eta, linear, rng)
    blocks = {v: [sample_subset(k, p, rng) for _ in range(s)] for v in H}
    vid = {}
    for v in H:
        for t in range(s):
            vid[(v, t)] = len(vid)
    G = nx.Graph()
    G.add_nodes_from(range(len(vid)))
    for v, w, pi in cons:
        bx = blocks[v]
        by = [apply_perm(y, pi) for y in blocks[w]]
        for a, x in enumerate(bx):
            for b, y in enumerate(by):
                if x & y == 0:
                    G.add_edge(vid[(v, a)], vid[(w, b)])
    # Planted dictator set, made independent greedily on violated constraints.
    I = {vid[(v, t)] for v in H for t in range(s) if blocks[v][t] >> L[v] & 1}
    for u in sorted(I):
        if u in I and any(w in I for w in G[u]):
            I.discard(u)
    return G, I


def shuffle_labels(G: nx.Graph, I: set, rng: random.Random):
    perm = list(range(G.number_of_nodes()))
    rng.shuffle(perm)
    mapping = {v: perm[i] for i, v in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping), {mapping[v] for v in I}


def instances(quick: bool, max_n: int):
    """(name, group, params) for every instance."""
    blocks = (40, 80) if quick else (40, 80, 160, 320)
    reps = 2 if quick else 3
    for N in blocks:
        for r in range(reps):
            # Base family: permutation UG, k = 4, p = 0.45, no noise, d = 4, s = 16.
            base = dict(N=N, d=4, k=4, p=0.45, eta=0.0, s=16, linear=False)
            yield f"kr_base_N{N}_r{r}", "KR permutation UG (k=4, p=0.45, eta=0)", base
            for k in (3, 6):
                yield f"kr_k{k}_N{N}_r{r}", f"KR label size k={k}", {**base, "k": k}
            for p in (0.35, 0.49):
                yield f"kr_p{p}_N{N}_r{r}", f"KR bias p={p}", {**base, "p": p}
            for eta in (0.02, 0.1):
                yield f"kr_eta{eta}_N{N}_r{r}", f"KR noise eta={eta}", {**base, "eta": eta}
            yield f"kr_d8_N{N}_r{r}", "KR constraint degree d=8", {**base, "d": 8}
            yield f"kr_s32_N{N}_r{r}", "KR samples per block s=32", {**base, "s": 32}
            yield f"kr_linear_N{N}_r{r}", "KR linear UG Max-2-Lin(Z_k)", {**base, "linear": True}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def exact_tau(G: nx.Graph, time_limit: float):
    """tau(G) by an integer program; (value, proved_optimal) or (None, False)."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return None, False
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = list(G.edges())
    if not E:
        return 0, True
    A = lil_matrix((len(E), len(nodes)))
    for r, (u, v) in enumerate(E):
        A[r, idx[u]] = 1
        A[r, idx[v]] = 1
    res = milp(c=np.ones(len(nodes)), constraints=LinearConstraint(A.tocsr(), lb=1, ub=np.inf),
               integrality=np.ones(len(nodes)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.x is None:
        return None, False
    return int(round(res.fun)), res.status == 0


def run_instance(name: str, group: str, params: dict, rng: random.Random, args) -> dict:
    G, I = khot_regev_yes(rng=rng, **params)
    G, I = shuffle_labels(G, I, rng)
    assert not any(u in I and v in I for u, v in G.edges()), name
    n, m = G.number_of_nodes(), G.number_of_edges()
    # Planted cover: every non-isolated vertex outside I' (isolated vertices need no cover).
    planted = sum(1 for v in G if v not in I and G.degree(v) > 0)
    info: dict = {}
    t0 = time.perf_counter()
    cands = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info,
                             eds_max_width=args.eds_max_width, eds_budget=args.eds_budget)
    elapsed = time.perf_counter() - t0
    # Isolated vertices are never needed in a cover; final_candidates drops them.
    sizes = {k: len(cands[k]) for k in NAMES}
    for k in NAMES:
        C = cands[k]
        if not all(u in C or v in C for u, v in G.edges()):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
    tau, tau_exact = (None, False)
    if n <= args.exact_max_n:
        tau, tau_exact = exact_tau(G, args.exact_time)
    final = sizes["c12"]
    row = {
        "group": group, "name": name, "params": params, "n": n, "m": m,
        "planted_independent": len(I), "planted_cover": planted,
        "tau": tau if tau_exact else None,
        "tau_upper": min(planted, tau) if tau is not None else planted,
        "sizes": sizes, "final": final,
        "ratio_vs_planted": final / planted if planted else 1.0,
        "ratio_vs_tau": (final / tau) if tau_exact and tau else None,
        "beats_or_matches_planted": final <= planted,
        "optimal": bool(tau_exact and final == tau),
        "c10_raw": info.get("c10_raw_size", sizes["c10"]),
        "c11_exact": bool(info.get("c11_exact", False)),
        "elapsed_seconds": elapsed,
    }
    tau_txt = f"tau={tau:>6}" if tau_exact else "tau=     ?"
    print(f"  [yes] {group[:38]:38} {name:24} n={n:>6,} m={m:>8,} planted={planted:>6,} {tau_txt}"
          f" c12={final:>6,} ratio/planted={row['ratio_vs_planted']:.4f}"
          + (f" ratio/tau={row['ratio_vs_tau']:.4f}" if row["ratio_vs_tau"] else "")
          + f" {elapsed:6.2f}s", flush=True)
    return row


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(grows)
        exact_rows = [r for r in grows if r["tau"] is not None]
        out[g] = {
            "instances": k,
            "matches_or_beats_planted": sum(r["beats_or_matches_planted"] for r in grows),
            "max_ratio_vs_planted": max((r["ratio_vs_planted"] for r in grows), default=1.0),
            "mean_ratio_vs_planted": sum(r["ratio_vs_planted"] for r in grows) / k if k else 1.0,
            "tau_known": len(exact_rows),
            "optimal": sum(r["optimal"] for r in exact_rows),
            "max_ratio_vs_tau": max((r["ratio_vs_tau"] for r in exact_rows), default=None),
            "best_by_strategy": {s: sum(r["sizes"][s] == r["final"] for r in grows) for s in NAMES},
            "seconds": sum(r["elapsed_seconds"] for r in grows),
        }
    return out


def print_table(summary: dict) -> None:
    header = (f"{'family':42} {'inst':>5} {'<=plant':>7} {'max/plant':>9} {'mean/plant':>10}"
              f" {'tau':>4} {'opt':>4} {'max/tau':>8}  attains |c12| (c1..c12)")
    print("\nKhot-Regev YES instances: final cover c12 against the planted cover and tau")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        by = " ".join(f"{s['best_by_strategy'][k]:>4}" for k in NAMES)
        mt = f"{s['max_ratio_vs_tau']:.4f}" if s["max_ratio_vs_tau"] is not None else "-"
        print(f"{g[:42]:42} {s['instances']:>5} {s['matches_or_beats_planted']:>7}"
              f" {s['max_ratio_vs_planted']:9.4f} {s['mean_ratio_vs_planted']:10.4f}"
              f" {s['tau_known']:>4} {s['optimal']:>4} {mt:>8}  {by}")
    print("\nratio/planted = |c12| / |planted cover| is a lower bound on the true ratio |c12| / tau;"
          " it is exact when the planted cover is optimal.")
    print("Legend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="UG sizes N = 40, 80 only")
    ap.add_argument("--max-n", type=int, default=10_000, help="skip instances with more vertices")
    ap.add_argument("--exact-max-n", type=int, default=600,
                    help="compute tau by integer programming up to this n (0 = never)")
    ap.add_argument("--exact-time", type=float, default=60.0, help="time limit per integer program (s)")
    ap.add_argument("--eds-max-width", type=int, default=DEFAULT_EDS_MAX_WIDTH,
                    help=f"c10: largest elimination bag minus one (default {DEFAULT_EDS_MAX_WIDTH})")
    ap.add_argument("--eds-budget", type=float, default=DEFAULT_EDS_BUDGET,
                    help=f"c10: DP table budget in units of (n + m) (default {DEFAULT_EDS_BUDGET})")
    args = ap.parse_args()

    started = time.time()
    rng = random.Random(SEED)
    rows = []
    for name, group, params in instances(args.quick, args.max_n):
        if params["N"] * params["s"] > args.max_n:
            continue
        rows.append(run_instance(name, group, params, rng, args))
    summary = summarise(rows)

    out = cc.OUT_DIR
    with (out / "car_yes_instances.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,N,d,k,p,eta,s,linear,n,m,planted_cover,tau," + ",".join(NAMES)
                 + ",final,ratio_vs_planted,ratio_vs_tau,beats_or_matches_planted,optimal,c10_raw,"
                 + "c11_exact,elapsed_seconds\n")
        for r in rows:
            q = r["params"]
            fh.write(f"\"{r['group']}\",{r['name']},{q['N']},{q['d']},{q['k']},{q['p']},{q['eta']},{q['s']},"
                     f"{q['linear']},{r['n']},{r['m']},{r['planted_cover']},{r['tau'] if r['tau'] is not None else ''},"
                     + ",".join(str(r["sizes"][k]) for k in NAMES)
                     + f",{r['final']},{r['ratio_vs_planted']:.6f},"
                     + (f"{r['ratio_vs_tau']:.6f}" if r["ratio_vs_tau"] else "")
                     + f",{r['beats_or_matches_planted']},{r['optimal']},{r['c10_raw']},"
                     + f"{r['c11_exact']},{r['elapsed_seconds']:.3f}\n")
    with (out / "car_yes_instances_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,matches_or_beats_planted,max_ratio_vs_planted,mean_ratio_vs_planted,"
                 "tau_known,optimal,max_ratio_vs_tau," + ",".join(f"attains_{k}" for k in NAMES) + ",seconds\n")
        for g, s in summary.items():
            mt = f"{s['max_ratio_vs_tau']:.6f}" if s["max_ratio_vs_tau"] is not None else ""
            fh.write(f"\"{g}\",{s['instances']},{s['matches_or_beats_planted']},{s['max_ratio_vs_planted']:.6f},"
                     f"{s['mean_ratio_vs_planted']:.6f},{s['tau_known']},{s['optimal']},{mt},"
                     + ",".join(str(s["best_by_strategy"][k]) for k in NAMES) + f",{s['seconds']:.2f}\n")
    (out / "car_yes_instances.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": SEED,
         "candidates": CANDIDATE_NAMES, "eds_max_width": args.eds_max_width, "eds_budget": args.eds_budget,
         "exact_max_n": args.exact_max_n, "exact_time": args.exact_time,
         "summary": summary, "rows": rows, "elapsed_seconds": time.time() - started},
        indent=2, default=str), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
