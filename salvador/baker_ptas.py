"""Weighted IDS pass and Baker-style PTAS utilities.

Since Salvador v0.0.5 the production path calls :func:`baker_ptas_ids_weighted`
with an *active* ``epsilon``. The function runs a Baker-style PTAS: it layers the
graph by BFS distance, removes each residue class modulo ``k = ceil(1/epsilon)``
in turn, solves the remaining bounded-treewidth components exactly with the
tree-decomposition dynamic program :func:`solve_component_ids_weighted`, repairs
the removed vertices greedily, and keeps the best independent dominating set over
the ``k`` shifts. The greedy maximal independent set produced by
:func:`greedy_maximal_is_weighted` is retained as the ``k = 1`` baseline and as a
safe fallback, so validity is preserved while ``epsilon`` controls solution
quality.
"""

import random
from collections import deque
from functools import lru_cache
from itertools import product
from math import ceil
import time

IN    = 0
DOM   = 1
UNDOM = 2
INF   = float('inf')


# ══════════════════════════════════════════════════════════════════════════════
# Graph utilities
# ══════════════════════════════════════════════════════════════════════════════

def bfs_layers(adj, vertices):
    layer = {}
    for src in vertices:
        if src in layer: continue
        layer[src] = 0; q = deque([src])
        while q:
            v = q.popleft()
            for u in adj.get(v, ()):
                if u in vertices and u not in layer:
                    layer[u] = layer[v] + 1; q.append(u)
    return layer


def connected_components(adj, active):
    visited = set(); comps = []
    for start in active:
        if start in visited: continue
        comp = []; stack = [start]
        while stack:
            v = stack.pop()
            if v in visited: continue
            visited.add(v); comp.append(v)
            for u in adj.get(v, ()):
                if u in active and u not in visited: stack.append(u)
        comps.append(frozenset(comp))
    return comps


def greedy_maximal_is_weighted(adj, vertices, weights):
    """Greedy MIS with linear-time weight buckets for the Salvador gadget."""
    zeros, ones, heavy = [], [], []
    for v in vertices:
        w = weights.get(v, 1)
        if w == 0:
            zeros.append(v)
        elif w == 1:
            ones.append(v)
        else:
            heavy.append(v)
    vlist = zeros + ones + heavy
    selected = set(); excluded = set()
    for v in vlist:
        if v not in excluded:
            selected.add(v); excluded.add(v)
            excluded.update(adj.get(v, ()))
    return frozenset(selected)


def verify_ids(adj, vertices, S):
    S = frozenset(S); V = set(vertices)
    for v in S:
        for u in adj.get(v, ()):
            if u in S: return False, f"independence violated: ({v},{u})"
    dominated = set(S)
    for v in S: dominated.update(adj.get(v, ()))
    missing = V - dominated
    return (False, f"undominated: {missing}") if missing else (True, "valid IDS ✓")


# ══════════════════════════════════════════════════════════════════════════════
# Tree decomposition
# ══════════════════════════════════════════════════════════════════════════════

def min_degree_elimination(sub_adj, vertices):
    vset = set(vertices)
    work = {v: set(sub_adj.get(v, ())) & vset for v in vset}
    remaining = set(vset); order = []; bag_nbrs = {}
    while remaining:
        v = min(remaining, key=lambda x: len(work[x] & remaining))
        nbrs = list(work[v] & remaining)
        bag_nbrs[v] = set(nbrs); order.append(v); remaining.remove(v)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                u, w = nbrs[i], nbrs[j]; work[u].add(w); work[w].add(u)
    return order, bag_nbrs


def build_tree_decomposition(elim_order, bag_nbrs):
    n = len(elim_order); elim_idx = {v: i for i, v in enumerate(elim_order)}
    bags = []; parent = []
    for i, v in enumerate(elim_order):
        later = frozenset(u for u in bag_nbrs[v] if elim_idx[u] > i)
        bags.append(frozenset({v}) | later)
        parent.append(elim_idx[min(later, key=lambda u: elim_idx[u])] if later else -1)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0: children[p].append(i)
    return bags, parent, children, [frozenset({elim_order[i]}) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# DP  —  state generation (memoised + bitmask)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4096)
def _valid_initial_states(m: int, adj_bits: tuple) -> tuple:
    """
    Return all valid initial DP state tuples for a bag of size m.

    adj_bits[i] is an integer bitmask: bit j set ⟺ vertex i and j are adjacent.

    Memoised by (m, adj_bits) — identical bag structures share the result.    ← NEW
    Bitmask adjacency replaces the inner  ba[i][j]  loops.                    ← NEW

    Validity rules:
      1.  IN  → every adjacent vertex is DOM  (not IN, not UNDOM)
      2.  DOM → at least one adjacent vertex is IN
    """
    valid = []
    in_mask_for  = tuple(1 << j for j in range(m))  # mask with only bit j set

    for state in product(range(3), repeat=m):
        in_mask   = sum(in_mask_for[j] for j in range(m) if state[j] == IN)
        ok = True
        for i in range(m):
            nbr = adj_bits[i]
            if state[i] == IN:
                # All neighbours must be DOM — none can be IN or UNDOM
                non_dom_nbrs = nbr & ~sum(in_mask_for[j]
                                          for j in range(m) if state[j] == DOM)
                if non_dom_nbrs:                          # some neighbour not DOM
                    ok = False; break
            elif state[i] == DOM:
                if not (in_mask & nbr):                   # no IN neighbour
                    ok = False; break
        if ok:
            valid.append(state)
    return tuple(valid)


def _make_adj_bits(m, bv, sub_adj):
    """Build bitmask adjacency for a sorted bag vertex list."""
    vidx = {v: i for i, v in enumerate(bv)}
    bits = []
    for i, u in enumerate(bv):
        mask = 0
        for w in sub_adj.get(u, ()):
            if w in vidx:
                mask |= 1 << vidx[w]
        bits.append(mask)
    return tuple(bits)


# ══════════════════════════════════════════════════════════════════════════════
# Weighted TD-DP
# ══════════════════════════════════════════════════════════════════════════════

def solve_component_ids_weighted(sub_adj, component, weights):
    """Minimum weighted IDS on a connected component via tree-decomposition DP."""
    n = len(component)
    if n == 0: return 0, frozenset()
    if n == 1:
        v = next(iter(component))
        return weights.get(v, 1), frozenset({v})
    # ── n = 2 base case  ─────────────────────────────────────────────────── NEW
    if n == 2:
        u, v = tuple(component)
        wu, wv = weights.get(u, 1), weights.get(v, 1)
        if sub_adj.get(u, set()) & {v}:          # edge exists → pick cheaper
            return (wu, frozenset({u})) if wu <= wv else (wv, frozenset({v}))
        else:                                     # no edge → both needed
            return wu + wv, frozenset({u, v})

    elim_order, bag_nbrs = min_degree_elimination(sub_adj, component)
    bags, par_arr, chd_arr, own_arr = build_tree_decomposition(elim_order, bag_nbrs)
    nb   = len(bags)
    root = next(i for i in range(nb) if par_arr[i] == -1)

    post_order = []
    stk = [(root, False)]
    while stk:
        node, done = stk.pop()
        if done: post_order.append(node)
        else:
            stk.append((node, True))
            for c in chd_arr[node]: stk.append((c, False))

    dp = [None] * nb

    for t in post_order:
        bv   = sorted(bags[t]); m = len(bv)
        vidx = {v: i for i, v in enumerate(bv)}
        adj_bits = _make_adj_bits(m, bv, sub_adj)        # ← bitmask adjacency

        own_idx = [vidx[v] for v in own_arr[t] if v in vidx]
        own_w   = [weights.get(bv[i], 1) for i in own_idx]

        valid_states = _valid_initial_states(m, adj_bits)  # ← memoised
        curr = {
            s: (sum(own_w[k] for k, i in enumerate(own_idx) if s[i] == IN), [])
            for s in valid_states
        }

        for c in chd_arr[t]:
            cbv   = sorted(bags[c]); cvidx = {v: i for i, v in enumerate(cbv)}
            shared = [(v, vidx[v], cvidx[v]) for v in bv if v in cvidx]
            c_only = [j for j, v in enumerate(cbv) if v not in vidx]
            new_curr = {}

            for c_state, (c_cost, _) in dp[c].items():
                if c_cost == INF: continue
                if any(c_state[j] == UNDOM for j in c_only): continue
                for p_state, (p_cost, p_back) in curr.items():
                    if not all((p_state[pi] == IN) == (c_state[ci] == IN)
                               for _, pi, ci in shared): continue
                    ns = list(p_state)
                    for _, pi, ci in shared:
                        if p_state[pi] != IN:
                            if p_state[pi] == DOM or c_state[ci] == DOM:
                                ns[pi] = DOM
                    ns = tuple(ns); total = p_cost + c_cost
                    if ns not in new_curr or total < new_curr[ns][0]:
                        new_curr[ns] = (total, p_back + [(c, c_state)])
            curr = new_curr

        dp[t] = curr

    best_cost = INF; best_state = None
    for state, (cost, _) in dp[root].items():
        if cost < best_cost and all(s != UNDOM for s in state):
            best_cost = cost; best_state = state

    if best_state is None:
        gs = greedy_maximal_is_weighted(sub_adj, component, weights)
        return sum(weights.get(v, 1) for v in gs), gs

    solution = set()
    def backtrack(node, state):
        bv_loc = sorted(bags[node]); vloc = {v: i for i, v in enumerate(bv_loc)}
        for v in own_arr[node]:
            if v in vloc and state[vloc[v]] == IN: solution.add(v)
        _, bt = dp[node][state]
        for c_idx, c_st in bt: backtrack(c_idx, c_st)

    backtrack(root, best_state)
    return best_cost, frozenset(solution)


# ══════════════════════════════════════════════════════════════════════════════
# Baker's PTAS — weighted main loop
# ══════════════════════════════════════════════════════════════════════════════

def baker_ptas_ids_weighted(adj, weights=None, epsilon=0.5):
    """Return a weighted independent dominating set via an active Baker PTAS.

    Since v0.0.5 the ``epsilon`` parameter is *functional*. The layering width
    is ``k = ceil(1/epsilon)``: the graph is split by BFS-layer residues modulo
    ``k``, each residue class is removed in turn, the remaining bounded-treewidth
    components are solved exactly by tree-decomposition dynamic programming
    (:func:`solve_component_ids_weighted`), and the removed vertices are repaired
    greedily.  The best solution over the ``k`` shifts is returned, never worse
    than the greedy maximal independent set baseline.  Smaller ``epsilon`` means
    larger ``k`` and a more thorough (and weakly better) solve.

    Setting ``epsilon >= 1`` gives ``k = 1`` and recovers the cheap greedy
    maximal-independent-set baseline as a special case.
    """
    vertices = set(adj.keys())
    if not vertices: return frozenset()
    if len(vertices) == 1: return frozenset(vertices)
    if weights is None: weights = {}

    k      = max(1, ceil(1.0 / epsilon))
    # k == 1 degenerates to the greedy baseline; skip the layering for speed.
    if k == 1:
        return greedy_maximal_is_weighted(adj, vertices, weights)
    layers = bfs_layers(adj, vertices)
    best   = greedy_maximal_is_weighted(adj, vertices, weights)
    best_w = sum(weights.get(v, 1) for v in best)

    for i in range(k):
        removed = frozenset(v for v in vertices if layers[v] % k == i)
        active  = vertices - removed
        comps   = connected_components(adj, frozenset(active))
        solution = set(); feasible = True

        for comp in comps:
            sub = {v: frozenset(adj.get(v, ())) & comp for v in comp}
            cost, s = solve_component_ids_weighted(sub, comp, weights)
            if cost == INF or s is None: feasible = False; break
            solution |= s
        if not feasible: continue

        dominated = set(solution)
        for v in solution: dominated.update(adj.get(v, ()))

        for v in sorted(removed, key=lambda v: weights.get(v, 1)):
            if v not in dominated:
                solution.add(v); dominated.add(v)
                dominated.update(adj.get(v, ()))

        sol = frozenset(solution)
        ok, _ = verify_ids(adj, vertices, sol)
        w = sum(weights.get(v, 1) for v in sol)
        if ok and w < best_w: best = sol; best_w = w

    return best


def baker_ptas_ids(adj, epsilon=0.5):
    """Unweighted Baker PTAS — backward-compatible wrapper."""
    return baker_ptas_ids_weighted(adj, weights=None, epsilon=epsilon)


# ══════════════════════════════════════════════════════════════════════════════
# Graph generators
# ══════════════════════════════════════════════════════════════════════════════

def grid_graph(rows, cols):
    adj = {}
    for r in range(rows):
        for c in range(cols):
            v = (r, c); nbrs = set()
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                u = (r+dr, c+dc)
                if 0 <= u[0] < rows and 0 <= u[1] < cols: nbrs.add(u)
            adj[v] = nbrs
    return adj

def cycle_graph(n):
    adj = {i: set() for i in range(n)}
    for i in range(n): adj[i].add((i+1)%n); adj[(i+1)%n].add(i)
    return adj

def path_graph(n):
    adj = {i: set() for i in range(n)}
    for i in range(n-1): adj[i].add(i+1); adj[i+1].add(i)
    return adj

def triangulated_grid(rows, cols):
    adj = grid_graph(rows, cols)
    for r in range(rows-1):
        for c in range(cols-1):
            v, u = (r,c), (r+1,c+1); adj[v].add(u); adj[u].add(v)
    return adj

def outerplanar_fan(n):
    adj = path_graph(n)
    for i in range(1, n): adj[0].add(i); adj[i].add(0)
    return adj

def random_planar_grid(n, seed=42):
    random.seed(seed)
    cols = max(1, int(n**0.5)); rows = (n+cols-1)//cols
    full = grid_graph(rows, cols); verts = list(full.keys())[:n]; vset = set(verts)
    return {v: full[v] & vset for v in verts}


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    import networkx as nx

    def exact_brute(adj, vertices, weights):
        vl = list(vertices); best = None; best_w = INF
        for mask in range(1 << len(vl)):
            S = frozenset(vl[i] for i in range(len(vl)) if mask >> i & 1)
            ok, _ = verify_ids(adj, vl, S)
            w = sum(weights.get(v,1) for v in S)
            if ok and w < best_w: best = S; best_w = w
        return best, best_w

    RULE = "═"*66
    print(RULE)
    print("  Baker's PTAS — Minimum WEIGHTED IDS  (optimised)")
    print(RULE)

    # Correctness: weighted vs brute-force
    print("\n── Weighted PTAS vs exact ──────────────────────────────────────\n")
    cases = [
        ("Path P₅",  path_graph(5),   {0:5,1:1,2:5,3:1,4:5}),
        ("Cycle C₆", cycle_graph(6),  {i:(1 if i%2==0 else 10) for i in range(6)}),
        ("Fan n=6",  outerplanar_fan(6),{0:1,**{i:10 for i in range(1,6)}}),
    ]
    print(f"  {'Graph':<12}  {'OPT_w':>6}  {'PTAS_w':>6}  {'ratio':>6}  valid")
    for name, adj, wts in cases:
        V = list(adj.keys())
        _, opt_w = exact_brute(adj, V, wts)
        ptas     = baker_ptas_ids_weighted(adj, wts, epsilon=0.5)
        ptas_w   = sum(wts.get(v,1) for v in ptas)
        ok, _    = verify_ids(adj, V, ptas)
        print(f"  {name:<12}  {opt_w:>6.1f}  {ptas_w:>6.1f}  {ptas_w/opt_w:>6.3f}  {'✓' if ok else '✗'}")

    # Backward compatibility
    print("\n── Unweighted (backward-compatible) ───────────────────────────\n")
    for name, adj, eps in [("Path P₆",path_graph(6),0.5),
                            ("Cycle C₈",cycle_graph(8),0.5),
                            ("3×3 Grid",grid_graph(3,3),0.5),
                            ("Fan n=8",outerplanar_fan(8),0.5)]:
        V = list(adj.keys()); sol = baker_ptas_ids(adj, epsilon=eps)
        ok, _ = verify_ids(adj, V, sol)
        print(f"  {name:<18}  |V|={len(V):>3}  IDS={len(sol):>3}  {'✓' if ok else '✗'}")

    # Speed benchmark
    print("\n── Speed benchmark ─────────────────────────────────────────────\n")
    print(f"  {'Graph':<28}  {'nodes':>6}  {'IDS':>5}  {'ms':>7}  valid")
    benchmarks = [
        ("Grid 8×8",             grid_graph(8,8),         0.5),
        ("Grid 15×15",           grid_graph(15,15),        0.5),
        ("Grid 20×20",           grid_graph(20,20),        0.5),
        ("Triangulated 10×10",   triangulated_grid(10,10), 0.5),
        ("Triangulated 15×15",   triangulated_grid(15,15), 0.5),
        ("Fan n=200",            outerplanar_fan(200),     0.5),
        ("Random planar n=500",  random_planar_grid(500),  0.5),
    ]
    for name, adj, eps in benchmarks:
        V = set(adj.keys())
        t0 = time.perf_counter()
        sol = baker_ptas_ids(adj, epsilon=eps)
        ms  = 1000*(time.perf_counter()-t0)
        ok, _ = verify_ids(adj, V, sol)
        print(f"  {name:<28}  {len(V):>6}  {len(sol):>5}  {ms:>7.1f}  {'✓' if ok else '✗'}")
    print()


if __name__ == "__main__":
    run_demo()
