"""car/ mechanism study: WHY does Salvador behave well on the residue R?

The paper (Section "Why the per-class guarantees do not give a uniform
constant below 2") leaves open the residue R: nearly Koenig-Egervary graphs
with an independent set of almost half the vertices -- the YES-type instances
of the hardness reductions -- plus the bipartite graphs that no theorem
covers. car_bipartite.py and car_yes_instances.py show that Salvador does
well there. This script measures, instance by instance, the quantities that a
*proof* of that behaviour would have to use, and tests, with control families
built to break each one, which of them actually carries the result.

Measured on every instance
--------------------------
Size and structure
    n, m, minimum / maximum / mean degree, coefficient of variation of the
    degrees, number of components, bipartite or not.

Optimum and certified bounds
    * tau: an integer program (scipy ``milp`` / HiGHS) up to ``--exact-max-n``
      vertices with a time limit; when it does not finish, its dual bound;
    * the LP relaxation, computed exactly as half the maximum matching of the
      bipartite double cover, with its Nemhauser-Trotter decomposition
      V = V0 + V1 + V_half (persistency: some minimum cover contains V1 and
      avoids V0); the *kernel fraction* |V_half| / n;
    * the matching number nu (exact up to ``--matching-max-m`` edges) and the
      Koenig-Egervary gap tau / nu;
    * L = the best certified lower bound on tau.

Which proved certificate of the paper already covers the instance
    For R0 = 1.999999999 the script evaluates, with tau or L:
    * C1 matching bound 2 nu / tau                       (Theorem matching);
    * C2 Caro-Wei / min-degree-greedy bound (n - sum 1/(d+1)) / tau -- the
      potential argument of the regular-graph theorem, which holds for every
      graph (the min-degree greedy independent set has at least
      sum_v 1/(deg v + 1) vertices);
    * C3 Hallelujah bound sqrt(2 m W(G)) / tau            (Theorem c3);
    * C4 classes solved exactly by Min-to-Min (forest, cycle, K_n, K_ab
      components);
    * C5 max degree <= 3 -> 11/6                          (Theorem c2);
    * C6 regular with d < 2e9 -> 2 - 2/(d+1)              (Theorem regular).
    The instance is *in the residue* when no certificate gives <= R0. The best
    certified value is the smallest of them: it is a *proved* per-instance
    ratio bound, not a measurement.

Mechanism probes (the candidate explanations)
    H1 degree signature: AUC of "lower degree" as a predictor of membership
       in a reference maximum independent set (the integer-program optimum
       when available, otherwise the planted set, otherwise the complement of
       the returned cover). AUC = 0.5 means degrees carry no information;
       min-degree heuristics (c4, c8, maximize_solution) exploit AUC > 0.5.
    H2 LP / Nemhauser-Trotter structure: kernel fraction, and how well the
       returned cover agrees with persistency (share of V1 in the cover, share
       of V0 outside it). If the kernel is small, any cover that respects
       persistency has ratio at most (|V1| + |V_half|) / (|V1| + |V_half|/2).
    H3 local optimality: the returned cover is checked for (1,2)-swap
       improvements; the excess |C| - tau is split into what the greedy
       candidates left (best of c1..c8 minus tau) and what local search
       removed (c9, c12).
    H4 scaling: every family is generated at several sizes, so the excess
       ratio can be fitted against n (does it stay bounded, grow like
       log n, or tend to 0?).

Families
--------
    * kr_sampled      Khot-Regev YES instances with p-biased sampled long
                      codes (as in car_yes_instances.py);
    * kr_fixedweight  CONTROL for H1: the same reduction, but every subset in a
                      block has exactly t = round(p k) elements, so every
                      vertex of a block has the same expected degree and the
                      |x| -> degree signature disappears; with k = 4, t = 2 the
                      dictator set is exactly half of the vertices;
    * planted_regular CONTROL for H1: a planted independent set of fraction
                      alpha in a graph whose degrees are made equal inside and
                      outside the set (no degree signature);
    * random_regular  NO-like control: random d-regular graphs (tau/nu > 1);
    * thinned_grid    the bipartite family on which Part D found errors;
    * scaling series  every family above at several sizes (``--sizes``).

Outputs (in car/): car_mechanism.json, car_mechanism.csv (one row per
instance), car_mechanism_summary.csv (one row per family) and a printed
report: per-family means, the share of instances covered by a proved
certificate, the correlation of the excess with each probe, and the
hypothesis tests (H1: sampled vs fixed-weight vs planted-regular; H2: excess
vs kernel fraction; H3: greedy excess vs local-search repair; H4: excess
ratio vs n).

What this can and cannot show
-----------------------------
The probes identify which structural property the good behaviour depends
on. If, say, the excess vanishes exactly when AUC > 0.5 and grows on the
fixed-weight and planted-regular controls, the proof should go through a
degree-signature lemma, and the controls are the instances on which such a
lemma fails. A measurement on finitely many graphs is never a proof: the
output is a map of where to look and of the counterexample candidates.

Run from the repository root:

    python car/car_mechanism.py                 # default sizes
    python car/car_mechanism.py --quick         # small sizes only
    python car/car_mechanism.py --exact-max-n 0 # no integer programs

"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import deque

import networkx as nx

import car_common as cc
import car_yes_instances as kr
from car_bipartite import thinned_grid
from salvador import algorithm as A

SEED = 20261010
R0 = 1.999999999
NAMES = list(A.CANDIDATE_NAMES)


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def kr_graph(N, d, k, s, eta, linear, sampler, rng):
    """Khot-Regev YES graph with block subsets drawn by ``sampler(rng)``."""
    H, cons, L, _ = kr.unique_games(N, d, k, eta, linear, rng)
    blocks = {v: [sampler(rng) for _ in range(s)] for v in H}
    vid = {}
    for v in H:
        for t in range(s):
            vid[(v, t)] = len(vid)
    G = nx.Graph()
    G.add_nodes_from(range(len(vid)))
    for v, w, pi in cons:
        by = [kr.apply_perm(y, pi) for y in blocks[w]]
        for a, x in enumerate(blocks[v]):
            for b, y in enumerate(by):
                if x & y == 0:
                    G.add_edge(vid[(v, a)], vid[(w, b)])
    I = {vid[(v, t)] for v in H for t in range(s) if blocks[v][t] >> L[v] & 1}
    for u in sorted(I):
        if u in I and any(w in I for w in G[u]):
            I.discard(u)
    return G, I


def fixed_weight_sampler(k, t):
    def sample(rng):
        x = 0
        for i in rng.sample(range(k), t):
            x |= 1 << i
        return x
    return sample


def planted_regular(n, alpha, d, rng):
    """Planted independent set of size alpha*n; all degrees close to d."""
    k = int(alpha * n)
    I, C = list(range(k)), list(range(k, n))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in I:
        for w in rng.sample(C, d):
            G.add_edge(u, w)
    pool = [c for c in C for _ in range(max(0, d - G.degree(c)))]
    rng.shuffle(pool)
    for i in range(0, len(pool) - 1, 2):
        if pool[i] != pool[i + 1]:
            G.add_edge(pool[i], pool[i + 1])
    return G, set(I)


def instances(sizes, quick):
    """(name, family, N, builder) -- builder(rng) -> (G, planted set or None)."""
    for N in sizes:
        for name, family, build in _instances_at(N, quick):
            yield name, family, N, build


def _instances_at(N, quick):
    reps = 2 if quick else 3
    if True:
        for r in range(reps):
            yield (f"kr_sampled_N{N}_r{r}", "kr_sampled (k=4, p=0.45, s=16)",
                   lambda rng, N=N: kr_graph(N, 4, 4, 16, 0.0, False, lambda g: kr.sample_subset(4, 0.45, g), rng))
            yield (f"kr_sampled_k6_N{N}_r{r}", "kr_sampled (k=6, p=0.45, s=16)",
                   lambda rng, N=N: kr_graph(N, 4, 6, 16, 0.0, False, lambda g: kr.sample_subset(6, 0.45, g), rng))
            yield (f"kr_fixed_k4t2_N{N}_r{r}", "kr_fixedweight (k=4, |x|=2, s=16)",
                   lambda rng, N=N: kr_graph(N, 4, 4, 16, 0.0, False, fixed_weight_sampler(4, 2), rng))
            yield (f"kr_fixed_k6t3_N{N}_r{r}", "kr_fixedweight (k=6, |x|=3, s=16)",
                   lambda rng, N=N: kr_graph(N, 4, 6, 16, 0.0, False, fixed_weight_sampler(6, 3), rng))
            yield (f"kr_fixed_k5t2_N{N}_r{r}", "kr_fixedweight (k=5, |x|=2, s=16)",
                   lambda rng, N=N: kr_graph(N, 4, 5, 16, 0.0, False, fixed_weight_sampler(5, 2), rng))
            n = 16 * N
            yield (f"planted_reg_a045_d6_n{n}_r{r}", "planted_regular (alpha=0.45, d=6)",
                   lambda rng, n=n: planted_regular(n, 0.45, 6, rng))
            yield (f"planted_reg_a048_d10_n{n}_r{r}", "planted_regular (alpha=0.48, d=10)",
                   lambda rng, n=n: planted_regular(n, 0.48, 10, rng))
            yield (f"random_reg_d3_n{n}_r{r}", "random_regular (d=3)",
                   lambda rng, n=n: (nx.random_regular_graph(3, n, seed=rng.randrange(2**31)), None))
            yield (f"random_reg_d6_n{n}_r{r}", "random_regular (d=6)",
                   lambda rng, n=n: (nx.random_regular_graph(6, n, seed=rng.randrange(2**31)), None))
            side = int(math.sqrt(n))
            yield (f"thinned_grid_{side}_r{r}", "thinned_grid (70% edges)",
                   lambda rng, side=side: (nx.convert_node_labels_to_integers(thinned_grid(side, side, 0.7, rng)), None))


# ----------------------------------------------------------------------------
# Bounds and certificates
# ----------------------------------------------------------------------------
def nt_decomposition(G):
    """Half-integral LP optimum via Koenig on the bipartite double cover.

    Returns (lp_value, V0, V1, V_half)."""
    nodes = list(G)
    top = [(v, 0) for v in nodes]
    H = nx.Graph()
    H.add_nodes_from(top)
    H.add_nodes_from((v, 1) for v in nodes)
    for u, v in G.edges():
        H.add_edge((u, 0), (v, 1))
        H.add_edge((v, 0), (u, 1))
    mate = nx.bipartite.hopcroft_karp_matching(H, top_nodes=top)
    topset = set(top)
    Z = set()
    q = deque()
    for x in top:
        if x not in mate:
            Z.add(x)
            q.append(x)
    while q:
        x = q.popleft()
        if x in topset:
            for y in H[x]:
                if y not in Z and mate.get(x) != y:
                    Z.add(y)
                    q.append(y)
        else:
            y = mate.get(x)
            if y is not None and y not in Z:
                Z.add(y)
                q.append(y)
    cover = {x for x in top if x not in Z} | {x for x in Z if x not in topset}
    V0, V1, Vh = set(), set(), set()
    for v in nodes:
        c = ((v, 0) in cover) + ((v, 1) in cover)
        (V0 if c == 0 else V1 if c == 2 else Vh).add(v)
    return len(cover) / 2, V0, V1, Vh


def milp_tau(G, time_limit):
    """(tau or None, proved, dual bound, optimal independent set or None)."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return None, False, None, None
    nodes = list(G)
    idx = {v: i for i, v in enumerate(nodes)}
    E = list(G.edges())
    if not E:
        return 0, True, 0.0, set(nodes)
    M = lil_matrix((len(E), len(nodes)))
    for r, (u, v) in enumerate(E):
        M[r, idx[u]] = 1
        M[r, idx[v]] = 1
    res = milp(c=np.ones(len(nodes)), constraints=LinearConstraint(M.tocsr(), lb=1, ub=np.inf),
               integrality=np.ones(len(nodes)), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    dual = getattr(res, "mip_dual_bound", None)
    if res.x is None:
        return None, False, dual, None
    indep = {nodes[i] for i in range(len(nodes)) if res.x[i] < 0.5}
    return int(round(res.fun)), res.status == 0, dual, indep


def matching_number(G, max_m):
    if G.number_of_edges() > max_m:
        return None
    return len(nx.max_weight_matching(G, maxcardinality=True))


def c4_class(G):
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        k, e = H.number_of_nodes(), H.number_of_edges()
        if k <= 1 or e == k - 1:
            continue  # tree
        degs = [d for _, d in H.degree()]
        if e == k and max(degs) == 2:
            continue  # cycle
        if e == k * (k - 1) // 2:
            continue  # complete
        if nx.is_bipartite(H):
            a, b = nx.bipartite.sets(H)
            if e == len(a) * len(b):
                continue  # complete bipartite
        return False
    return True


def certificates(G, nu, ref_tau):
    """Proved ratio bounds of the paper evaluated with tau (or a lower bound)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = dict(G.degree())
    out = {}
    if nu is not None:
        out["C1_matching"] = 2 * nu / ref_tau
    cw = sum(1.0 / (d + 1) for d in deg.values())
    out["C2_caro_wei"] = (n - math.ceil(cw - 1e-9)) / ref_tau
    W = sum(1.0 / max(deg[u], deg[v]) for u, v in G.edges())
    out["C3_hallelujah"] = math.sqrt(2 * m * W) / ref_tau
    out["C4_mtm_class"] = 1.0 if c4_class(G) else None
    out["C5_subcubic"] = 11 / 6 if max(deg.values()) <= 3 else None
    dset = set(deg.values())
    out["C6_regular"] = (2 - 2 / (next(iter(dset)) + 1)) if len(dset) == 1 else None
    vals = {k: v for k, v in out.items() if v is not None}
    best_name = min(vals, key=vals.get)
    return out, best_name, vals[best_name]


# ----------------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------------
def degree_auc(G, members):
    """P(deg(member) < deg(non-member)) + 0.5 P(tie)."""
    deg = dict(G.degree())
    a = [deg[v] for v in G if v in members]
    b = [deg[v] for v in G if v not in members]
    if not a or not b:
        return None
    ranked = sorted([(d, 1) for d in a] + [(d, 0) for d in b])
    # Mann-Whitney U with ties, for "member has smaller degree".
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # 1-based average rank
        rank_sum += avg * sum(1 for t in range(i, j) if ranked[t][1] == 1)
        i = j
    u_small = rank_sum - len(a) * (len(a) + 1) / 2  # pairs where member ranks higher
    return 1 - u_small / (len(a) * len(b))


def has_12_swap(G, cover):
    """True when the independent set V - cover admits a (1,2)-swap improvement."""
    I = set(G) - cover
    tight = {v: sum(1 for w in G[v] if w in I) for v in cover}
    for x in I:
        one = [v for v in G[x] if v in cover and tight[v] == 1]
        for i, v in enumerate(one):
            nv = G[v]
            for w in one[i + 1:]:
                if w not in nv:
                    return True
    return False


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return None


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run_instance(name, family, size, G, planted, args):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from([v for v in list(G) if G.degree(v) == 0])
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = [d for _, d in G.degree()]
    t0 = time.perf_counter()
    cands = A.final_candidates(G, epsilon=cc.DEFAULT_EPSILON)
    elapsed = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in NAMES}
    C = cands["c12"]
    assert all(u in C or v in C for u, v in G.edges()), name

    lp, V0, V1, Vh = nt_decomposition(G)
    tau, proved, dual, opt_indep = (None, False, None, None)
    if n <= args.exact_max_n:
        tau, proved, dual, opt_indep = milp_tau(G, args.exact_time)
    lower = math.ceil(lp - 1e-9)
    if dual is not None:
        lower = max(lower, math.ceil(dual - 1e-6))
    if proved:
        lower = tau
    ref_tau = tau if proved else lower
    nu = matching_number(G, args.matching_max_m)
    certs, best_cert, best_val = certificates(G, nu, ref_tau)

    if proved and opt_indep is not None:
        ref_set, ref_kind = opt_indep, "optimum"
    elif planted:
        ref_set, ref_kind = {v for v in planted if v in G}, "planted"
    else:
        ref_set, ref_kind = set(G) - C, "returned"
    best_greedy = min(sizes[k] for k in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"))
    row = {
        "family": family, "name": name, "size_N": size, "n": n, "m": m,
        "deg_min": min(degs), "deg_max": max(degs), "deg_mean": 2 * m / n,
        "deg_cv": statistics.pstdev(degs) / (2 * m / n),
        "bipartite": nx.is_bipartite(G), "components": nx.number_connected_components(G),
        "sizes": sizes, "final": len(C),
        "tau": tau if proved else None, "lower_bound": lower, "lp": lp, "nu": nu,
        "ke_gap": (tau / nu) if (proved and nu) else None,
        "planted_cover": (n - len(ref_set)) if ref_kind == "planted" else None,
        "ratio": len(C) / tau if proved else None,
        "ratio_upper_bound": len(C) / lower if lower else 1.0,
        "excess": (len(C) - tau) if proved else None,
        "greedy_excess": (best_greedy - tau) if proved else None,
        "ls_repair": best_greedy - len(C),
        "certificates": certs, "best_certificate": best_cert, "best_certified_ratio": best_val,
        "in_residue": best_val > R0,
        "kernel_fraction": len(Vh) / n, "nt_V1": len(V1), "nt_V0": len(V0),
        "V1_in_cover": (len(V1 & C) / len(V1)) if V1 else None,
        "V0_outside_cover": (len(V0 - C) / len(V0)) if V0 else None,
        "persistency_respected": V1 <= C and not (V0 & C),
        "degree_auc": degree_auc(G, ref_set), "auc_reference": ref_kind,
        "has_12_swap": has_12_swap(G, C),
        "elapsed_seconds": elapsed,
    }
    print(f"  [mech] {family[:34]:34} {name:28} n={n:>6} tau={'%6d' % tau if proved else '     ?'}"
          f" C={len(C):>6} ub={row['ratio_upper_bound']:.4f} cert={best_cert}:{best_val:.3f}"
          f" auc={row['degree_auc'] if row['degree_auc'] is None else round(row['degree_auc'], 3)}"
          f" kernel={row['kernel_fraction']:.3f} 12swap={row['has_12_swap']} {elapsed:.2f}s", flush=True)
    return row


def summarise(rows):
    out = {}
    for fam in sorted({r["family"] for r in rows}) + ["ALL"]:
        F = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        ex = [r for r in F if r["tau"] is not None]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        out[fam] = {
            "instances": len(F),
            "tau_known": len(ex),
            "optimal": sum(r["excess"] == 0 for r in ex),
            "mean_ratio": mean([r["ratio"] for r in ex]),
            "max_ratio": max((r["ratio"] for r in ex), default=None),
            "max_ratio_upper_bound": max(r["ratio_upper_bound"] for r in F),
            "in_residue": sum(r["in_residue"] for r in F),
            "max_best_certified": max(r["best_certified_ratio"] for r in F),
            "mean_degree_auc": mean([r["degree_auc"] for r in F if r["degree_auc"] is not None]),
            "mean_kernel_fraction": mean([r["kernel_fraction"] for r in F]),
            "persistency_respected": sum(r["persistency_respected"] for r in F),
            "mean_ke_gap": mean([r["ke_gap"] for r in F if r["ke_gap"] is not None]),
            "with_12_swap": sum(r["has_12_swap"] for r in F),
            "mean_greedy_excess_rel": mean([r["greedy_excess"] / r["tau"] for r in ex if r["tau"]]),
            "mean_excess_rel": mean([r["excess"] / r["tau"] for r in ex if r["tau"]]),
            "corr_excess_auc": pearson([r["degree_auc"] for r in ex], [r["excess"] / r["tau"] for r in ex if r["tau"]]),
            "best_certificate_counts": {c: sum(r["best_certificate"] == c for r in F)
                                        for c in sorted({r["best_certificate"] for r in F})},
            "corr_excess_kernel": pearson([r["kernel_fraction"] for r in ex], [r["excess"] / r["tau"] for r in ex if r["tau"]]),
            "seconds": sum(r["elapsed_seconds"] for r in F),
        }
    return out


def scaling(rows):
    """Mean ratio (exact, else certified upper bound) by family and UG size N (n ~ 16 N)."""
    table = {}
    for r in rows:
        key = r["family"]
        val = r["ratio"] if r["ratio"] is not None else r["ratio_upper_bound"]
        table.setdefault(key, {}).setdefault(r["size_N"], []).append(val)
    return {f: {n: sum(v) / len(v) for n, v in sorted(d.items())} for f, d in table.items()}


def print_report(summary, scal):
    print("\nPer family (exact ratios where an integer program proved tau)")
    hdr = (f"{'family':38} {'inst':>4} {'tau':>4} {'opt':>4} {'max ratio':>9} {'max ub':>7} {'resid':>5}"
           f" {'auc':>6} {'kernel':>6} {'NTok':>4} {'12sw':>4} {'greedy ex':>9} {'final ex':>8}")
    print(hdr)
    print("-" * len(hdr))
    f = lambda x, p=4: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    for fam, s in summary.items():
        print(f"{fam[:38]:38} {s['instances']:>4} {s['tau_known']:>4} {s['optimal']:>4} {f(s['max_ratio']):>9}"
              f" {f(s['max_ratio_upper_bound'], 3):>7} {s['in_residue']:>5} {f(s['mean_degree_auc'], 3):>6}"
              f" {f(s['mean_kernel_fraction'], 3):>6} {s['persistency_respected']:>4} {s['with_12_swap']:>4}"
              f" {f(s['mean_greedy_excess_rel']):>9} {f(s['mean_excess_rel']):>8}")
    print("\nHypothesis tests")
    sa = {k: v for k, v in summary.items() if k.startswith("kr_sampled")}
    fw = {k: v for k, v in summary.items() if k.startswith("kr_fixedweight")}
    pr = {k: v for k, v in summary.items() if k.startswith("planted_regular")}
    for label, grp in (("H1 sampled long code", sa), ("H1 fixed-weight control", fw), ("H1 planted-regular control", pr)):
        for k, v in grp.items():
            print(f"  {label:28} {k[:40]:40} auc={f(v['mean_degree_auc'], 3)} mean excess={f(v['mean_excess_rel'])}"
                  f" max ratio={f(v['max_ratio'])}")
    alls = summary["ALL"]
    print(f"  H2 corr(excess, kernel fraction) over all exact instances: {f(alls['corr_excess_kernel'], 3)};"
          f" persistency respected on {alls['persistency_respected']}/{alls['instances']}")
    print(f"  H1 corr(excess, degree AUC) over all exact instances:      {f(alls['corr_excess_auc'], 3)}")
    print(f"  H3 mean relative excess after greedy (best c1..c8): {f(alls['mean_greedy_excess_rel'])};"
          f" after local search (c12): {f(alls['mean_excess_rel'])}; (1,2)-swap left: {alls['with_12_swap']}")
    print(f"  Residue: {alls['in_residue']}/{alls['instances']} instances have no proved certificate <= R0;"
          f" best certificate counts: {alls['best_certificate_counts']}")
    print("\n  H4 scaling (mean exact ratio, or certified upper bound when tau is unknown)")
    for fam, d in scal.items():
        print(f"    {fam[:38]:38} " + "  ".join(f"n~{16 * N}: {v:.4f}" for N, v in d.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="sizes N = 10, 20 only, 2 seeds")
    ap.add_argument("--sizes", type=str, default="10,20,40,80",
                    help="UG sizes N (graph sizes n = 16 N); default 10,20,40,80")
    ap.add_argument("--exact-max-n", type=int, default=1300, help="integer program up to this n (0 = never)")
    ap.add_argument("--exact-time", type=float, default=90.0, help="time limit per integer program (s)")
    ap.add_argument("--matching-max-m", type=int, default=40_000,
                    help="compute the matching number exactly up to this many edges")
    args = ap.parse_args()
    sizes = [10, 20] if args.quick else [int(x) for x in args.sizes.split(",")]

    started = time.time()
    rng = random.Random(SEED)
    rows = []
    for name, family, size, build in instances(sizes, args.quick):
        G, planted = build(rng)
        rows.append(run_instance(name, family, size, G, planted, args))
    summary = summarise(rows)
    scal = scaling(rows)

    out = cc.OUT_DIR
    cols = ["family", "name", "size_N", "n", "m", "deg_min", "deg_max", "deg_mean", "deg_cv", "bipartite", "components",
            "final", "tau", "lower_bound", "lp", "nu", "ke_gap", "planted_cover", "ratio", "ratio_upper_bound",
            "excess", "greedy_excess", "ls_repair", "best_certificate", "best_certified_ratio", "in_residue",
            "kernel_fraction", "nt_V1", "nt_V0", "V1_in_cover", "V0_outside_cover", "persistency_respected",
            "degree_auc", "auc_reference", "has_12_swap", "elapsed_seconds"]
    with (out / "car_mechanism.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "," + ",".join(NAMES) + "\n")
        for r in rows:
            fh.write(",".join("" if r[c] is None else f"\"{r[c]}\"" if isinstance(r[c], str) else str(r[c])
                              for c in cols) + "," + ",".join(str(r["sizes"][k]) for k in NAMES) + "\n")
    with (out / "car_mechanism_summary.csv").open("w", encoding="utf-8") as fh:
        keys = list(next(iter(summary.values())).keys())
        fh.write("family," + ",".join(keys) + "\n")
        for fam, s in summary.items():
            fh.write(f"\"{fam}\"," + ",".join("" if s[k] is None else str(s[k]) for k in keys) + "\n")
    (out / "car_mechanism.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": SEED, "R0": R0,
         "sizes": sizes, "exact_max_n": args.exact_max_n, "exact_time": args.exact_time,
         "summary": summary, "scaling": scal, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")
    print_report(summary, scal)


if __name__ == "__main__":
    main()
