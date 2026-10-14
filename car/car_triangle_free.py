"""car/ triangle-free graphs: how far below 2 is Salvador's ratio, instance by instance?

Context
-------
Assuming the Unique Games Conjecture, approximating Vertex Cover within
``2 - epsilon`` is hard already on triangle-free graphs. That statement is
about the *worst case over all* triangle-free graphs: under UGC no
polynomial-time algorithm achieves ratio ``<= 2 - epsilon`` on *every* one of
them (unless P = NP).

What this experiment can and cannot show
-----------------------------------------
* It **certifies, per instance**, an interval for the ratio
  ``|C| / tau(G)`` of Salvador's final cover ``C = c14``:

      ratio_lower = |C| / tau_upper  <=  |C| / tau(G)  <=  |C| / tau_lower = ratio_upper,

  where ``tau_lower`` is a certified lower bound (exact tau when proved, an
  integer-program dual bound, the LP relaxation, matchings, the Laplacian
  spectral bound) and ``tau_upper`` a certified upper bound (a planted cover,
  or ``|C|`` itself). ``eps_certified = 2 - ratio_upper`` is the margin below 2
  that is *proved* for that instance.
* It searches adversarially (a hill climb over small triangle-free graphs with
  exact tau) for the worst ratio it can find.
* It **cannot** show that the ``2 - epsilon`` barrier is broken. A ratio below
  ``2 - epsilon`` on every instance tried -- however many -- is evidence about
  those instances only: UGC-hardness concerns worst-case families that a
  finite sample need not contain, and most graphs are easy for any good
  heuristic. Breaking the barrier would need a proof that the algorithm
  achieves ``2 - epsilon`` on *all* triangle-free graphs, and under UGC that
  would imply P = NP. The summary prints this caveat with the numbers.

Instance families (every graph is checked to be triangle-free)
--------------------------------------------------------------
Part A -- exact tau (scipy ``milp``/HiGHS, or the branch and bound of car_common):

* named graphs: Petersen, Heawood, Moebius-Kantor, Pappus, Desargues,
  dodecahedron, Chvatal, Clebsch (folded 5-cube), Tutte 8-cage,
  Hoffman-Singleton, odd cycles, hypercubes (bipartite control);
* Mycielski graphs M3..M7 (triangle-free, chromatic number k);
* Andrasfai graphs And(k) (circulant, ``3k - 1`` vertices, connection set
  ``= 1 mod 3``: dense triangle-free);
* triangle-free Kneser graphs K(n, k) with ``n < 3k``;
* generalized Petersen graphs GP(n, k) and middle-third circulants;
* random triangle-free graphs (random triangle-free process) with n <= 40.

Part H -- adversarial hill climb: small triangle-free graphs mutated (add an
edge that creates no triangle / remove / rewire) to maximise ``|C| / tau``,
exact tau by branch and bound.

Part B -- large graphs, certified bounds (exact when the integer program
finishes within ``--exact-time`` for ``n <= --exact-max-n``):

* **Khot-Regev YES instances on a triangle-free constraint graph**: the
  sampled long-code reduction of car_yes_instances.py, built on a random
  d-regular constraint graph whose triangles are removed by degree-preserving
  switches. Every edge of the Khot-Regev graph joins two different blocks
  that are adjacent in the constraint graph, so a triangle-free constraint
  graph gives a triangle-free instance, with its planted cover intact. These
  are the instances of the reduction behind the hardness result (YES side);
* random triangle-free process graphs (average degree 4..32);
* G(n, p) with every triangle destroyed (an edge removed per triangle);
* random d-regular graphs made triangle-free by switches (d = 3..12);
* Frankl-Roedl graphs FR(q, t): {0,1}^q, x ~ y iff Hamming distance t, t even
  and t > 2q/3 (triangle-free; classic LP/SDP integrality-gap instances);
* large Mycielski, Andrasfai and middle-third circulant graphs;
* random bipartite graphs (control: tau exact by Koenig).

Outputs (in car/): car_triangle_free.json, car_triangle_free.csv (one row per
instance: every strategy size, bounds and their methods, ratio interval,
eps_certified, LP gap, timings) and car_triangle_free_summary.csv (one row per
family: worst exact ratio, worst certified upper bound, smallest certified
margin, worst planted lower bound, counts below 2 - eps for several eps).

Run from the repository root:

    python car/car_triangle_free.py --quick               # minutes
    python car/car_triangle_free.py                       # full
    python car/car_triangle_free.py --skip-small --max-n 100000
    python car/car_triangle_free.py --hill-restarts 40 --hill-steps 400
    python car/car_triangle_free.py --exact-max-n 3000 --exact-time 300
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import time
from pathlib import Path

import networkx as nx

import car_common as cc
import car_yes_instances as kr
from salvador.algorithm import CANDIDATE_NAMES, final_candidates

SEED = 20261013
NAMES = list(CANDIDATE_NAMES)  # c1..c14
FINAL = "c14"
EPS_LEVELS = (0.5, 0.25, 0.1, 0.05, 0.01)


# ----------------------------------------------------------------------------
# Triangle-free tools
# ----------------------------------------------------------------------------
def triangle_count(G: nx.Graph) -> int:
    return sum(nx.triangles(G).values()) // 3


def destroy_triangles(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Remove one edge from every triangle (greedily, random order) until none is left."""
    G = G.copy()
    edges = list(G.edges())
    rng.shuffle(edges)
    for u, v in edges:
        if G.has_edge(u, v) and not set(G[u]).isdisjoint(G[v]):
            G.remove_edge(u, v)
    return G


def switch_out_triangles(G: nx.Graph, rng: random.Random, max_rounds: int = 200) -> nx.Graph:
    """Degree-preserving double-edge switches that remove triangles.

    An edge {a, b} in a triangle and a random edge {c, d} (all four distinct)
    are replaced by {a, c}, {b, d} when that creates neither a multi-edge nor a
    triangle. Edges that cannot be switched are deleted at the end, so the
    result is always triangle-free (and near-regular).
    """
    G = G.copy()
    for _ in range(max_rounds):
        bad = [(u, v) for u, v in G.edges() if not set(G[u]).isdisjoint(G[v])]
        if not bad:
            return G
        edges = list(G.edges())
        for a, b in bad:
            if not G.has_edge(a, b) or set(G[a]).isdisjoint(G[b]):
                continue
            for _ in range(20):
                c, d = edges[rng.randrange(len(edges))]
                if rng.random() < 0.5:
                    c, d = d, c
                if len({a, b, c, d}) < 4 or not G.has_edge(c, d):
                    continue
                if G.has_edge(a, c) or G.has_edge(b, d):
                    continue
                G.remove_edge(a, b)
                G.remove_edge(c, d)
                if set(G[a]).isdisjoint(G[c]) and set(G[b]).isdisjoint(G[d]):
                    G.add_edge(a, c)
                    G.add_edge(b, d)
                    break
                G.add_edge(a, b)
                G.add_edge(c, d)
    return destroy_triangles(G, rng)


def triangle_free_process(n: int, target_edges: int, rng: random.Random, max_tries: int | None = None) -> nx.Graph:
    """Random triangle-free process: random pairs, added when they close no triangle."""
    G = nx.empty_graph(n)
    tries = 0
    max_tries = max_tries or 20 * target_edges + 1000
    while G.number_of_edges() < target_edges and tries < max_tries:
        tries += 1
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v or G.has_edge(u, v):
            continue
        if set(G[u]).isdisjoint(G[v]):
            G.add_edge(u, v)
    return G


# ----------------------------------------------------------------------------
# Named and structured triangle-free graphs
# ----------------------------------------------------------------------------
def clebsch() -> nx.Graph:
    """Folded 5-cube: {0,1}^4, x ~ y iff Hamming distance 1 or 4 (16 vertices, 5-regular)."""
    G = nx.Graph()
    for x in range(16):
        for y in range(x + 1, 16):
            if bin(x ^ y).count("1") in (1, 4):
                G.add_edge(x, y)
    return G


def andrasfai(k: int) -> nx.Graph:
    """Andrasfai graph And(k): Z_{3k-1}, i ~ j iff (i - j) mod (3k-1) = 1 mod 3."""
    n = 3 * k - 1
    return nx.circulant_graph(n, [s for s in range(1, n) if s % 3 == 1])


def middle_third_circulant(n: int) -> nx.Graph:
    """Cayley graph of Z_n with S = {s : n/3 < s < 2n/3}: sum-free, hence triangle-free."""
    S = [s for s in range(1, n // 2 + 1) if 3 * s > n and 3 * s < 2 * n]
    return nx.circulant_graph(n, S)


def kneser(n: int, k: int) -> nx.Graph:
    subsets = [frozenset(c) for c in itertools.combinations(range(n), k)]
    G = nx.Graph()
    G.add_nodes_from(range(len(subsets)))
    for i, a in enumerate(subsets):
        for j in range(i + 1, len(subsets)):
            if a.isdisjoint(subsets[j]):
                G.add_edge(i, j)
    return G


def frankl_rodl(q: int, t: int) -> nx.Graph:
    """{0,1}^q, x ~ y iff Hamming distance exactly t."""
    G = nx.Graph()
    G.add_nodes_from(range(1 << q))
    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(q), t)]
    for x in range(1 << q):
        for mk in masks:
            y = x ^ mk
            if x < y:
                G.add_edge(x, y)
    return G


def generalized_petersen(n: int, k: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(n):
        G.add_edge(("o", i), ("o", (i + 1) % n))
        G.add_edge(("o", i), ("i", i))
        G.add_edge(("i", i), ("i", (i + k) % n))
    return G


def named_graphs():
    yield "petersen", nx.petersen_graph()
    yield "heawood", nx.heawood_graph()
    yield "moebius_kantor", nx.moebius_kantor_graph()
    yield "pappus", nx.pappus_graph()
    yield "desargues", nx.desargues_graph()
    yield "dodecahedral", nx.dodecahedral_graph()
    yield "chvatal", nx.chvatal_graph()
    yield "clebsch", clebsch()
    yield "tutte_8_cage", nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5)
    yield "hoffman_singleton", nx.hoffman_singleton_graph()
    for k in (2, 3, 5, 8, 12, 20, 25):
        yield f"odd_cycle_C{2 * k + 1}", nx.cycle_graph(2 * k + 1)
    for d in (3, 4, 5, 6):
        yield f"hypercube_Q{d}", nx.hypercube_graph(d)


# ----------------------------------------------------------------------------
# Khot-Regev YES instances on a triangle-free constraint graph
# ----------------------------------------------------------------------------
def triangle_free_unique_games(N: int, d: int, k: int, eta: float, linear: bool, rng: random.Random):
    """As car_yes_instances.unique_games, on a triangle-free d-regular-ish constraint graph."""
    H = nx.random_regular_graph(d, N - (N * d) % 2, seed=rng.randrange(2**31))
    H = switch_out_triangles(H, rng)
    L = {v: rng.randrange(k) for v in H}
    cons = []
    for v, w in H.edges():
        if rng.random() < eta:
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
            pi[j], pi[L[w]] = pi[L[w]], pi[j]
            pi = tuple(pi)
        cons.append((v, w, pi))
    return H, cons, L


def khot_regev_triangle_free(N, d, k, p, eta, s, linear, rng):
    """Sampled Khot-Regev YES graph on a triangle-free constraint graph; (G, planted independent set)."""
    H, cons, L = triangle_free_unique_games(N, d, k, eta, linear, rng)
    blocks = {v: [kr.sample_subset(k, p, rng) for _ in range(s)] for v in H}
    vid = {(v, t): i for i, (v, t) in enumerate((v, t) for v in H for t in range(s))}
    G = nx.Graph()
    G.add_nodes_from(range(len(vid)))
    for v, w, pi in cons:
        bx = blocks[v]
        by = [kr.apply_perm(y, pi) for y in blocks[w]]
        for a, x in enumerate(bx):
            for b, y in enumerate(by):
                if x & y == 0:
                    G.add_edge(vid[(v, a)], vid[(w, b)])
    I = {vid[(v, t)] for v in H for t in range(s) if blocks[v][t] >> L[v] & 1}
    for u in sorted(I):
        if u in I and any(w in I for w in G[u]):
            I.discard(u)
    return kr.shuffle_labels(G, I, rng)


# ----------------------------------------------------------------------------
# Instance lists: (name, group, kind, params)
# ----------------------------------------------------------------------------
def part_a_instances(quick: bool):
    for name, _ in named_graphs():
        yield name, "A named triangle-free graphs", "named", {"name": name}
    for k in range(3, 7 if quick else 8):
        yield f"mycielski_M{k}", "A Mycielski graphs", "mycielski", {"k": k}
    for k in range(2, 12 if quick else 21):
        yield f"andrasfai_{k}", "A Andrasfai graphs", "andrasfai", {"k": k}
    for n, k in ((5, 2), (7, 3), (8, 3)) if quick else ((5, 2), (7, 3), (8, 3), (9, 4), (10, 4), (11, 4)):
        yield f"kneser_{n}_{k}", "A triangle-free Kneser graphs", "kneser", {"n": n, "k": k}
    for n in range(7, 20 if quick else 31, 2 if quick else 1):
        for k in range(2, n // 2 + 1):
            yield f"gp_{n}_{k}", "A generalized Petersen GP(n,k)", "gp", {"n": n, "k": k}
    for n in range(10, 40 if quick else 61, 3):
        yield f"midthird_{n}", "A middle-third circulants", "midthird", {"n": n}
    reps = 4 if quick else 12
    for n in (10, 14, 18, 24, 30, 40):
        for dens in (1.5, 2.5, 4.0):
            for r in range(reps):
                yield (f"tfp_n{n}_d{dens}_r{r}", "A random triangle-free (n<=40)", "tfp",
                       {"n": n, "avg_degree": dens})


def part_b_instances(quick: bool, max_n: int, max_m: int):
    reps = 1 if quick else 2
    blocks = (40, 80) if quick else (40, 80, 160, 320)
    for N in blocks:
        for r in range(reps):
            base = dict(N=N, d=4, k=4, p=0.45, eta=0.0, s=16, linear=False)
            fams = [("base", "B KR-TF permutation UG (k=4, p=0.45)", base),
                    ("k3", "B KR-TF label size k=3", {**base, "k": 3}),
                    ("k6", "B KR-TF label size k=6", {**base, "k": 6}),
                    ("p049", "B KR-TF bias p=0.49 (near 1/2)", {**base, "p": 0.49}),
                    ("p035", "B KR-TF bias p=0.35", {**base, "p": 0.35}),
                    ("eta01", "B KR-TF noise eta=0.1", {**base, "eta": 0.1}),
                    ("d3", "B KR-TF constraint degree d=3", {**base, "d": 3}),
                    ("d8", "B KR-TF constraint degree d=8", {**base, "d": 8}),
                    ("s32", "B KR-TF samples per block s=32", {**base, "s": 32}),
                    ("linear", "B KR-TF linear UG Max-2-Lin(Z_k)", {**base, "linear": True})]
            for tag, group, params in fams:
                if params["N"] * params["s"] <= max_n:
                    yield f"krtf_{tag}_N{N}_r{r}", group, "kr", params
    sizes = (2000, 10000) if quick else (2000, 10000, 50000, 100000)
    for n in sizes:
        if n > max_n:
            continue
        for avg in (4, 8, 16, 32):
            if n * avg / 2 <= max_m:
                for r in range(reps):
                    yield f"tfp_n{n}_d{avg}_r{r}", f"B random triangle-free process, avg degree {avg}", "tfp", {"n": n, "avg_degree": avg}
        for avg in (4, 8, 16):
            if n * avg / 2 <= max_m:
                for r in range(reps):
                    yield f"gnp_tf_n{n}_d{avg}_r{r}", f"B G(n,p) triangles destroyed, avg degree {avg}", "gnp_tf", {"n": n, "avg_degree": avg}
        for d in (3, 4, 6, 8, 12):
            if n * d / 2 <= max_m:
                for r in range(reps):
                    yield f"reg_tf_n{n}_d{d}_r{r}", f"B random {d}-regular, triangles switched out", "reg_tf", {"n": n, "d": d}
        for r in range(reps):
            yield f"bip_n{n}_r{r}", "B random bipartite (control, tau exact)", "bip", {"n": n, "avg_degree": 6}
    for q, t in ((8, 6), (10, 8), (11, 8), (12, 10), (13, 10), (14, 10)):
        n = 1 << q
        m = n * math.comb(q, t) // 2
        if n <= max_n and m <= max_m:
            yield f"frankl_rodl_q{q}_t{t}", "B Frankl-Roedl graphs FR(q,t)", "fr", {"q": q, "t": t}
    for k in (8, 9, 10):
        n = 3 * 2 ** (k - 2) - 1
        if n <= max_n:
            yield f"mycielski_M{k}", "B large Mycielski graphs", "mycielski", {"k": k}
    for k in (50, 100, 200, 400, 800):
        if 3 * k - 1 <= max_n and (3 * k - 1) * k / 2 <= max_m:
            yield f"andrasfai_{k}", "B large Andrasfai graphs", "andrasfai", {"k": k}
    for n in (200, 500, 1000, 2000):
        if n <= max_n and n * n / 6 <= max_m:
            yield f"midthird_{n}", "B large middle-third circulants", "midthird", {"n": n}


def build(kind: str, params: dict, rng: random.Random):
    """(G, planted independent set or None)."""
    if kind == "named":
        return dict(named_graphs())[params["name"]], None
    if kind == "mycielski":
        return nx.mycielski_graph(params["k"]), None
    if kind == "andrasfai":
        return andrasfai(params["k"]), None
    if kind == "kneser":
        return kneser(params["n"], params["k"]), None
    if kind == "gp":
        return generalized_petersen(params["n"], params["k"]), None
    if kind == "midthird":
        return middle_third_circulant(params["n"]), None
    if kind == "tfp":
        n = params["n"]
        return triangle_free_process(n, int(n * params["avg_degree"] / 2), rng), None
    if kind == "gnp_tf":
        n = params["n"]
        G = nx.fast_gnp_random_graph(n, params["avg_degree"] / (n - 1), seed=rng.randrange(2**31))
        return destroy_triangles(G, rng), None
    if kind == "reg_tf":
        n, d = params["n"], params["d"]
        G = nx.random_regular_graph(d, n - (n * d) % 2, seed=rng.randrange(2**31))
        return switch_out_triangles(G, rng), None
    if kind == "bip":
        n = params["n"]
        G = nx.bipartite.random_graph(n // 2, n - n // 2, params["avg_degree"] / (n / 2), seed=rng.randrange(2**31))
        return G, None
    if kind == "fr":
        return frankl_rodl(params["q"], params["t"]), None
    if kind == "kr":
        return khot_regev_triangle_free(rng=rng, **params)
    raise KeyError(kind)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(name: str, group: str, G: nx.Graph, planted_I, args: dict, exact_small: bool) -> dict:
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from([v for v in list(G) if G.degree(v) == 0])
    if planted_I is not None:
        planted_I = {v for v in planted_I if v in G}
    G = nx.convert_node_labels_to_integers(G, label_attribute="orig")
    if planted_I is not None:
        back = {d["orig"]: v for v, d in G.nodes(data=True)}
        planted_I = {back[v] for v in planted_I}
    n, m = G.number_of_nodes(), G.number_of_edges()
    tri = triangle_count(G)
    if tri:
        raise AssertionError(f"{name}: {tri} triangles -- the generator is not triangle-free")
    degs = [d for _, d in G.degree()]

    info: dict = {}
    t0 = time.perf_counter()
    cands = final_candidates(G, epsilon=cc.DEFAULT_EPSILON, info=info)
    salvador_seconds = time.perf_counter() - t0
    sizes = {k: len(cands[k]) for k in NAMES}
    edges = list(G.edges())
    for k in NAMES:
        C = cands[k]
        if not all(u in C or v in C for u, v in edges):
            raise AssertionError(f"{name}: strategy {k} is not a vertex cover")
    final = sizes[FINAL]

    # Certified bounds on tau.
    t0 = time.perf_counter()
    lower: dict[str, float] = {}
    upper: dict[str, int] = {"salvador_c14": final}
    tau = None
    method = None
    if nx.is_bipartite(G):
        tau, method = cc.exact_vertex_cover_size(G)
    elif exact_small and n <= args["bb_max_n"]:
        tau, method = cc.exact_vertex_cover_size(G)
    if tau is None and n <= args["exact_max_n"]:
        val, proved, dual = kr.exact_tau(G, args["exact_time"])
        if proved:
            tau, method = val, "integer program (proved)"
        else:
            if val is not None:
                upper["integer_program_incumbent"] = val
            if dual is not None:
                lower["integer_program_dual"] = float(dual)
    lp = None
    if m <= args["lp_max_m"]:
        lp = kr.lp_lower_bound(G)
        lower["lp_relaxation"] = lp
    if tau is None:
        lb, lb_method, all_lb = cc.best_lower_bound(G, spectral=not args["no_spectral"])
        for key, val in all_lb.items():
            lower[key] = float(val)
    if planted_I is not None:
        planted_cover = sum(1 for v in G if v not in planted_I)
        upper["planted"] = planted_cover
    bound_seconds = time.perf_counter() - t0

    if tau is not None:
        tau_lower = tau_upper = tau
        lower_method = upper_method = method
    else:
        lower_method = max(lower, key=lower.get) if lower else "none"
        tau_lower = math.ceil(lower[lower_method] - 1e-9) if lower else 1
        upper_method = min(upper, key=upper.get)
        tau_upper = upper[upper_method]
    tau_lower = max(tau_lower, 1)
    ratio_upper = final / tau_lower
    ratio_lower = final / tau_upper
    row = {
        "group": group, "name": name, "n": n, "m": m,
        "min_degree": min(degs), "max_degree": max(degs), "avg_degree": 2 * m / n,
        "bipartite": nx.is_bipartite(G), "triangles": tri,
        "sizes": sizes, "final": final, "final_fraction": final / n,
        "tau": tau, "tau_method": method, "exact": tau is not None,
        "tau_lower": tau_lower, "tau_lower_method": lower_method,
        "tau_upper": tau_upper, "tau_upper_method": upper_method,
        "lower_bounds": lower, "upper_bounds": upper,
        "ratio": (final / tau) if tau else None,
        "ratio_lower": ratio_lower, "ratio_upper": ratio_upper,
        "eps_certified": 2 - ratio_upper,
        "lp": lp, "ratio_vs_lp": (final / lp) if lp else None,
        "integrality_gap": (tau / lp) if (tau and lp) else None,
        "planted_cover": upper.get("planted"),
        "ratio_vs_planted": (final / upper["planted"]) if upper.get("planted") else None,
        "c11_exact": bool(info.get("c11_exact", False)), "c13_exact": bool(info.get("c13_exact", False)),
        "salvador_seconds": salvador_seconds, "bound_seconds": bound_seconds,
    }
    rat = f"ratio={row['ratio']:.4f}" if row["ratio"] else f"ratio in [{ratio_lower:.4f}, {ratio_upper:.4f}]"
    print(f"  [tf] {group[:34]:34} {name:26} n={n:>7,} m={m:>9,} |C|={final:>7,} "
          f"tau{'=' if tau else '>='}{tau_lower:>7,} ({lower_method[:18]}) {rat} eps_cert={row['eps_certified']:+.4f}"
          f" [{salvador_seconds:.1f}s+{bound_seconds:.1f}s]", flush=True)
    return row


def _job(name: str, group: str, kind: str, params: dict, args: dict, exact_small: bool) -> dict:
    """Build and evaluate one instance (module level: runs in the worker process)."""
    rng = cc.instance_rng(SEED, name)
    G, I = build(kind, params, rng)
    return evaluate(name, group, G, I, args, exact_small)


# ----------------------------------------------------------------------------
# Adversarial hill climb (small n, exact tau)
# ----------------------------------------------------------------------------
def _ratio_small(G: nx.Graph) -> tuple[float, int, int]:
    H = nx.Graph(G)
    H.remove_nodes_from([v for v in list(H) if H.degree(v) == 0])
    if H.number_of_edges() == 0:
        return 1.0, 0, 0
    C = final_candidates(H, epsilon=cc.DEFAULT_EPSILON)[FINAL]
    tau, _ = cc.exact_vertex_cover_size(H)
    return len(C) / tau, len(C), tau


def hill_climb(rng: random.Random, restarts: int, steps: int, n_range=(8, 14)) -> list[dict]:
    out = []
    for r in range(restarts):
        n = rng.randint(*n_range)
        G = triangle_free_process(n, rng.randint(n, 2 * n), rng)
        best, c, tau = _ratio_small(G)
        best_G = G.copy()
        for _ in range(steps):
            H = best_G.copy()
            move = rng.random()
            nodes = list(H)
            if move < 0.45:
                u, v = rng.sample(nodes, 2)
                if not H.has_edge(u, v) and set(H[u]).isdisjoint(H[v]):
                    H.add_edge(u, v)
            elif move < 0.75 and H.number_of_edges() > 1:
                u, v = rng.choice(list(H.edges()))
                H.remove_edge(u, v)
            elif H.number_of_edges() > 1:
                u, v = rng.choice(list(H.edges()))
                w = rng.choice(nodes)
                if w not in (u, v) and not H.has_edge(u, w):
                    H.remove_edge(u, v)
                    if set(H[u]).isdisjoint(H[w]):
                        H.add_edge(u, w)
                    else:
                        H.add_edge(u, v)
            if triangle_count(H):
                continue
            val, c2, t2 = _ratio_small(H)
            if val >= best:
                best, best_G, c, tau = val, H, c2, t2
        row = evaluate(f"hill_r{r}", "H adversarial hill climb (exact)", best_G, None,
                       {"bb_max_n": 64, "exact_max_n": 0, "exact_time": 1, "lp_max_m": 10**9, "no_spectral": True},
                       exact_small=True)
        row["edges_list"] = [list(e) for e in best_G.edges()]
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        F = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        ex = [r for r in F if r["exact"]]
        planted = [r for r in F if r["ratio_vs_planted"] is not None]
        gaps = [r["integrality_gap"] for r in F if r["integrality_gap"] is not None]
        worst = max(F, key=lambda r: r["ratio_upper"])
        out[g] = {
            "instances": len(F),
            "exact": len(ex),
            "optimal": sum(r["final"] == r["tau"] for r in ex),
            "max_exact_ratio": max((r["ratio"] for r in ex), default=None),
            "mean_exact_ratio": (sum(r["ratio"] for r in ex) / len(ex)) if ex else None,
            "max_ratio_upper": worst["ratio_upper"],
            "worst_instance": worst["name"],
            "min_eps_certified": min(r["eps_certified"] for r in F),
            "mean_ratio_upper": sum(r["ratio_upper"] for r in F) / len(F),
            "max_ratio_lower": max(r["ratio_lower"] for r in F),
            "max_ratio_vs_planted": max((r["ratio_vs_planted"] for r in planted), default=None),
            "max_integrality_gap": max(gaps, default=None),
            "certified_below": {str(e): sum(r["ratio_upper"] <= 2 - e for r in F) for e in EPS_LEVELS},
            "max_n": max(r["n"] for r in F),
            "seconds": sum(r["salvador_seconds"] for r in F),
        }
    return out


def print_summary(summary: dict) -> None:
    f = lambda x: "-" if x is None else f"{x:.4f}"
    print("\nTriangle-free graphs: ratio of Salvador's final cover c14 (exact where tau is proved, else the"
          " certified interval)")
    head = (f"{'family':48} {'N':>4} {'exact':>5} {'opt':>4} {'max exact':>9} {'max ub':>8} {'min eps':>8}"
            f" {'max lb':>7} {'vs plant':>8} {'LP gap':>7} " + " ".join(f"<=2-{e:<4}" for e in EPS_LEVELS))
    print(head)
    print("-" * len(head))
    for g, s in summary.items():
        print(f"{g[:48]:48} {s['instances']:>4} {s['exact']:>5} {s['optimal']:>4} {f(s['max_exact_ratio']):>9}"
              f" {s['max_ratio_upper']:>8.4f} {s['min_eps_certified']:>+8.4f} {s['max_ratio_lower']:>7.4f}"
              f" {f(s['max_ratio_vs_planted']):>8} {f(s['max_integrality_gap']):>7} "
              + " ".join(f"{s['certified_below'][str(e)]:>8}" for e in EPS_LEVELS))
    a = summary["ALL"]
    print(f"\nWorst certified upper bound: {a['max_ratio_upper']:.4f} on {a['worst_instance']}"
          f" (certified margin eps >= {a['min_eps_certified']:.4f} on every instance run).")
    print("max ub   = largest |C| / tau_lower (certified upper bound on the ratio);"
          " min eps = 2 - max ub, the smallest certified margin below 2;")
    print("max lb   = largest |C| / tau_upper (the true ratio is at least this);"
          " vs plant = |C| / planted cover (Khot-Regev YES instances);")
    print("LP gap   = tau / LP optimum where tau is exact (how far these instances are from LP-easy).")
    print("\nCaveat: these numbers certify the ratio on the instances above only. UGC-hardness of 2 - eps on"
          " triangle-free graphs is a worst-case statement over all such graphs; a finite experiment cannot"
          " show the barrier is broken, which under UGC would imply P = NP.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller sweep")
    ap.add_argument("--skip-small", action="store_true", help="skip Part A (exact small graphs)")
    ap.add_argument("--skip-hill", action="store_true", help="skip the adversarial hill climb")
    ap.add_argument("--skip-large", action="store_true", help="skip Part B (large graphs)")
    ap.add_argument("--max-n", type=int, default=50_000, help="largest Part B instance (default 50000)")
    ap.add_argument("--max-m", type=int, default=2_000_000, help="largest Part B edge count (default 2e6)")
    ap.add_argument("--hill-restarts", type=int, default=None, help="hill-climb restarts (default 8 quick / 24)")
    ap.add_argument("--hill-steps", type=int, default=None, help="hill-climb steps per restart (default 80 / 250)")
    ap.add_argument("--bb-max-n", type=int, default=40,
                    help="exact tau by branch and bound up to this n in Part A (default 40)")
    ap.add_argument("--exact-max-n", type=int, default=1300,
                    help="exact tau by integer program up to this n (default 1300; needs scipy)")
    ap.add_argument("--exact-time", type=float, default=60.0, help="integer-program time limit, s (default 60)")
    ap.add_argument("--lp-max-m", type=int, default=600_000, help="LP lower bound up to this many edges")
    ap.add_argument("--no-spectral", action="store_true", help="skip the Laplacian spectral bound")
    cc.add_timeout_argument(ap, default=600.0)
    args = ap.parse_args()
    bounds_args = {"bb_max_n": args.bb_max_n, "exact_max_n": args.exact_max_n, "exact_time": args.exact_time,
                   "lp_max_m": args.lp_max_m, "no_spectral": args.no_spectral}

    started = time.time()
    rows: list[dict] = []
    skipped: list[str] = []
    with cc.InstanceRunner(args.timeout) as runner:
        if not args.skip_small:
            for name, group, kind, params in part_a_instances(args.quick):
                G, _ = build(kind, params, cc.instance_rng(SEED, name))
                if triangle_count(G):
                    continue  # e.g. GP(n, k) with a triangle: not in scope
                row = runner.run(_job, name, group, kind, params, bounds_args, True)
                if row is None:
                    skipped.append(name)
                else:
                    rows.append(row)
        if not args.skip_hill:
            restarts = args.hill_restarts or (8 if args.quick else 24)
            steps = args.hill_steps or (80 if args.quick else 250)
            rows.extend(hill_climb(random.Random(SEED), restarts, steps))
        if not args.skip_large:
            for name, group, kind, params in part_b_instances(args.quick, args.max_n, args.max_m):
                row = runner.run(_job, name, group, kind, params, bounds_args, False)
                if row is None:
                    print(f"  [tf] {group[:34]:34} {name:26} skipped (more than {args.timeout:g} s)", flush=True)
                    skipped.append(name)
                else:
                    rows.append(row)
                gc.collect()

    summary = summarise(rows)
    out = cc.OUT_DIR
    cols = ["group", "name", "n", "m", "min_degree", "max_degree", "avg_degree", "bipartite"]
    tail = ["final", "final_fraction", "tau", "tau_method", "exact", "tau_lower", "tau_lower_method", "tau_upper",
            "tau_upper_method", "ratio", "ratio_lower", "ratio_upper", "eps_certified", "lp", "ratio_vs_lp",
            "integrality_gap", "planted_cover", "ratio_vs_planted", "c11_exact", "c13_exact",
            "salvador_seconds", "bound_seconds"]

    def cell(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        if isinstance(v, str) and ("," in v or " " in v):
            return f"\"{v}\""
        return str(v)

    with (out / "car_triangle_free.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols + NAMES + tail) + "\n")
        for r in rows:
            fh.write(",".join([cell(r[c]) for c in cols] + [str(r["sizes"][k]) for k in NAMES]
                              + [cell(r[c]) for c in tail]) + "\n")
    with (out / "car_triangle_free_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,exact,optimal,max_exact_ratio,mean_exact_ratio,max_ratio_upper,worst_instance,"
                 "min_eps_certified,mean_ratio_upper,max_ratio_lower,max_ratio_vs_planted,max_integrality_gap,max_n,"
                 + ",".join(f"certified_below_2_minus_{e}" for e in EPS_LEVELS) + ",seconds\n")
        for g, s in summary.items():
            fh.write(",".join([cell(g) if "," not in g else f"\"{g}\"", str(s["instances"]), str(s["exact"]),
                               str(s["optimal"]), cell(s["max_exact_ratio"]), cell(s["mean_exact_ratio"]),
                               cell(s["max_ratio_upper"]), s["worst_instance"], cell(s["min_eps_certified"]),
                               cell(s["mean_ratio_upper"]), cell(s["max_ratio_lower"]),
                               cell(s["max_ratio_vs_planted"]), cell(s["max_integrality_gap"]), str(s["max_n"])]
                              + [str(s["certified_below"][str(e)]) for e in EPS_LEVELS]
                              + [f"{s['seconds']:.3f}"]) + "\n")
    (out / "car_triangle_free.json").write_text(json.dumps(
        {"salvador_version": cc.__version__, "environment": cc.environment(), "seed": SEED,
         "args": vars(args), "candidates": CANDIDATE_NAMES, "eps_levels": EPS_LEVELS,
         "caveat": ("Per-instance certified ratios only; a finite experiment cannot show that the UGC "
                    "2 - eps barrier for triangle-free graphs is broken (a worst-case statement)."),
         "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")
    print_summary(summary)
    if skipped:
        print(f"\nSkipped (more than {args.timeout:g} s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
