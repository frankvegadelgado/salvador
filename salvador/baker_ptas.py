"""
Baker's PTAS for Minimum Weighted Independent Dominating Set in Planar Graphs
==============================================================================
Extends the unweighted Baker PTAS with non-negative vertex weights.
The single algorithmic change is marked  ← WEIGHTED  below.

USAGE
-----
    adj     = {0:{1,2}, 1:{0,3}, 2:{0}, 3:{1}}
    weights = {0:5, 1:1, 2:1, 3:5}
    S = baker_ptas_ids_weighted(adj, weights, epsilon=0.5)

    S = baker_ptas_ids(adj, epsilon=0.5)          # unweighted (backward-compat)
"""

import time, random
from collections import deque
from itertools import product
from math import ceil

IN    = 0   # selected into IDS
DOM   = 1   # not selected, dominated
UNDOM = 2   # not selected, not yet dominated
INF   = float('inf')


# ── graph utilities ───────────────────────────────────────────────────────────

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
    """Greedy MIS — processes vertices by increasing weight."""       # ← WEIGHTED
    vlist    = sorted(vertices, key=lambda v: (weights.get(v, 1),
                                               len(adj.get(v, ()))))
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


# ── tree decomposition ────────────────────────────────────────────────────────

def min_degree_elimination(sub_adj, vertices):
    vset = set(vertices)
    work = {v: set(sub_adj.get(v, ())) & vset for v in vset}
    remaining = set(vset); order = []; bag_nbrs = {}
    while remaining:
        v = min(remaining, key=lambda x: len(work[x] & remaining))
        nbrs = list(work[v] & remaining)
        bag_nbrs[v] = set(nbrs); order.append(v); remaining.remove(v)
        for i in range(len(nbrs)):
            for j in range(i+1, len(nbrs)):
                u, w = nbrs[i], nbrs[j]
                work[u].add(w); work[w].add(u)
    return order, bag_nbrs


def build_tree_decomposition(elim_order, bag_nbrs):
    n = len(elim_order); elim_idx = {v: i for i, v in enumerate(elim_order)}
    bags = []; parent = []
    for i, v in enumerate(elim_order):
        later = frozenset(u for u in bag_nbrs[v] if elim_idx[u] > i)
        bags.append(frozenset({v}) | later)
        if later:
            parent.append(elim_idx[min(later, key=lambda u: elim_idx[u])])
        else:
            parent.append(-1)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0: children[p].append(i)
    owned = [frozenset({elim_order[i]}) for i in range(n)]
    return bags, parent, children, owned


# ── DP ────────────────────────────────────────────────────────────────────────

def _valid_initial_states(m, ba):
    valid = []
    for state in product(range(3), repeat=m):
        ok = True
        for i in range(m):
            if state[i] == IN:
                for j in range(m):
                    if i != j and ba[i][j] and state[j] != DOM:
                        ok = False; break
            elif state[i] == DOM:
                if not any(j != i and ba[i][j] and state[j] == IN for j in range(m)):
                    ok = False
            if not ok: break
        if ok: valid.append(state)
    return valid


def solve_component_ids_weighted(sub_adj, component, weights):
    """
    Exact minimum WEIGHTED IDS on one connected component via TD-DP.

    The only difference from the unweighted version:

        cost += weights.get(v, 1)   for each owned vertex v in state IN
                                                                  ← WEIGHTED
    """
    n = len(component)
    if n == 0: return 0, frozenset()
    if n == 1:
        v = next(iter(component))
        return weights.get(v, 1), frozenset({v})

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

        ba = [[False]*m for _ in range(m)]
        for i, u in enumerate(bv):
            for w in sub_adj.get(u, ()):
                if w in vidx: ba[i][vidx[w]] = True

        own_idx = [vidx[v] for v in own_arr[t] if v in vidx]
        own_w   = [weights.get(bv[i], 1) for i in own_idx]   # ← WEIGHTED

        valid_states = _valid_initial_states(m, ba)
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


# ── main PTAS ─────────────────────────────────────────────────────────────────

def baker_ptas_ids_weighted(adj, weights=None, epsilon=0.5):
    """
    Baker's (1+ε)-PTAS for Minimum WEIGHTED IDS on planar graphs.

    Parameters
    ----------
    adj     : dict[vertex, set[vertex]]   undirected adjacency
    weights : dict[vertex, float]         vertex weights (default 1)
    epsilon : float                       approximation ratio parameter

    Returns
    -------
    frozenset  —  IDS S with  Σ w(v) ≤ (1+ε)·OPT_weighted
    """
    vertices = set(adj.keys())
    if not vertices: return frozenset()
    if len(vertices) == 1: return frozenset(vertices)
    if weights is None: weights = {}

    k      = max(1, ceil(1.0 / epsilon))
    layers = bfs_layers(adj, vertices)

    best   = greedy_maximal_is_weighted(adj, vertices, weights)  # ← WEIGHTED
    best_w = sum(weights.get(v, 1) for v in best)               # ← WEIGHTED

    for i in range(k):
        removed = frozenset(v for v in vertices if layers[v] % k == i)
        active  = vertices - removed
        comps   = connected_components(adj, frozenset(active))
        solution = set(); feasible = True

        for comp in comps:
            sub = {v: frozenset(adj.get(v, ())) & comp for v in comp}
            cost, s = solve_component_ids_weighted(sub, comp, weights)  # ← WEIGHTED
            if cost == INF or s is None: feasible = False; break
            solution |= s
        if not feasible: continue

        dominated = set(solution)
        for v in solution: dominated.update(adj.get(v, ()))

        # repair: cheapest undominated removed vertex first          ← WEIGHTED
        for v in sorted(removed, key=lambda v: weights.get(v, 1)):
            if v not in dominated:
                solution.add(v); dominated.add(v)
                dominated.update(adj.get(v, ()))

        sol = frozenset(solution)
        ok, _ = verify_ids(adj, vertices, sol)
        w = sum(weights.get(v, 1) for v in sol)                    # ← WEIGHTED
        if ok and w < best_w:                                       # ← WEIGHTED
            best = sol; best_w = w

    return best


def baker_ptas_ids(adj, epsilon=0.5):
    """Unweighted Baker PTAS — backward-compatible wrapper."""
    return baker_ptas_ids_weighted(adj, weights=None, epsilon=epsilon)


# ── graph generators (unchanged from original) ────────────────────────────────

def grid_graph(rows, cols):
    adj = {}
    for r in range(rows):
        for c in range(cols):
            v = (r,c); nbrs = set()
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                u=(r+dr,c+dc)
                if 0<=u[0]<rows and 0<=u[1]<cols: nbrs.add(u)
            adj[v] = nbrs
    return adj

def cycle_graph(n):
    adj={i:set() for i in range(n)}
    for i in range(n): adj[i].add((i+1)%n); adj[(i+1)%n].add(i)
    return adj

def path_graph(n):
    adj={i:set() for i in range(n)}
    for i in range(n-1): adj[i].add(i+1); adj[i+1].add(i)
    return adj

def triangulated_grid(rows,cols):
    adj=grid_graph(rows,cols)
    for r in range(rows-1):
        for c in range(cols-1):
            v,u=(r,c),(r+1,c+1); adj[v].add(u); adj[u].add(v)
    return adj

def outerplanar_fan(n):
    adj=path_graph(n)
    for i in range(1,n): adj[0].add(i); adj[i].add(0)
    return adj

def random_planar_grid(n,seed=42):
    random.seed(seed)
    cols=max(1,int(n**0.5)); rows=(n+cols-1)//cols
    full=grid_graph(rows,cols); verts=list(full.keys())[:n]; vset=set(verts)
    return {v:full[v]&vset for v in verts}


# ── demo ─────────────────────────────────────────────────────────────────────

def exact_ids_weighted_brute(adj, vertices, weights):
    vlist=list(vertices); n=len(vlist); best=None; best_w=INF
    for mask in range(1<<n):
        S=frozenset(vlist[i] for i in range(n) if mask>>i&1)
        ok,_=verify_ids(adj,vlist,S)
        w=sum(weights.get(v,1) for v in S)
        if ok and w<best_w: best=S; best_w=w
    return best, best_w


def run_demo():
    RULE="═"*68
    print(RULE)
    print("  Baker's PTAS — Minimum WEIGHTED Independent Dominating Set")
    print(RULE)

    # unweighted backward-compatibility
    print("\n── Unweighted (backward-compatible) ───────────────────────────────\n")
    for name,adj,eps in [("Path P₆",path_graph(6),0.5),
                         ("Cycle C₈",cycle_graph(8),0.5),
                         ("3×3 Grid",grid_graph(3,3),0.5),
                         ("Fan n=8",outerplanar_fan(8),0.5)]:
        V=list(adj.keys()); sol=baker_ptas_ids(adj,epsilon=eps)
        ok,_=verify_ids(adj,V,sol)
        print(f"  {name:<18} |V|={len(V):>3}  IDS={len(sol):>3}  {'✓' if ok else '✗'}")

    # weighted vs brute-force on small graphs
    print("\n── Weighted PTAS vs exact (small graphs) ───────────────────────────\n")
    cases = [
        ("Path P₅",  path_graph(5),   {0:5,1:1,2:5,3:1,4:5}),
        ("Cycle C₆", cycle_graph(6),  {i:(1 if i%2==0 else 10) for i in range(6)}),
        ("Fan n=6",  outerplanar_fan(6),{0:1,1:10,2:10,3:10,4:10,5:10}),
    ]
    print(f"  {'Graph':<14} {'OPT_w':>7} {'PTAS_w':>7} {'ratio':>7}  Valid")
    for name,adj,wts in cases:
        V=list(adj.keys())
        opt,opt_w = exact_ids_weighted_brute(adj,V,wts)
        ptas      = baker_ptas_ids_weighted(adj,wts,epsilon=0.5)
        ptas_w    = sum(wts.get(v,1) for v in ptas)
        ok,_      = verify_ids(adj,V,ptas)
        ratio     = ptas_w/opt_w if opt_w else float('nan')
        print(f"  {name:<14} {opt_w:>7.1f} {ptas_w:>7.1f} {ratio:>7.3f}  {'✓' if ok else '✗'}")

    # larger benchmark (unit weights)
    print("\n── Benchmark (ε=0.5, unit weights) ────────────────────────────────\n")
    print(f"  {'Graph':<22} {'n':>5} {'m':>6} {'IDS':>5} {'ms':>7}  Valid")
    for name,adj,eps in [
        ("Path P₁₂",            path_graph(12),          0.5),
        ("Cycle C₁₅",           cycle_graph(15),          0.5),
        ("4×4 Grid",            grid_graph(4,4),          0.5),
        ("5×5 Grid",            grid_graph(5,5),          0.5),
        ("Triangulated 4×4",    triangulated_grid(4,4),   0.5),
        ("Fan n=20",            outerplanar_fan(20),      0.5),
        ("8×8 Grid",            grid_graph(8,8),          0.5),
        ("Random planar n=100", random_planar_grid(100),  0.5),
    ]:
        V=set(adj.keys()); m=sum(len(adj[v]) for v in V)//2
        t0=time.perf_counter()
        sol=baker_ptas_ids(adj,epsilon=eps)
        ms=1000*(time.perf_counter()-t0)
        ok,_=verify_ids(adj,V,sol)
        print(f"  {name:<22} {len(V):>5} {m:>6} {len(sol):>5} {ms:>7.1f}  {'✓' if ok else '✗'}")
    print()


if __name__ == "__main__":
    run_demo()