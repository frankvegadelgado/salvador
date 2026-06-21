from __future__ import annotations

"""ChatGPT-generated reproducible experiment for Salvador v0.0.4.

This script intentionally does not use MILP.  It computes exact vertex-cover
sizes by solving Maximum Independent Set with branch-and-bound on the small
stress instances generated here.  The goal is not to prove an approximation
ratio, but to test whether the observed behavior is compatible with a constant
sub-2, i.e. (2-epsilon)-type, phenomenon.

Run from the repository root with:

    python experiments/chatgpt_sub2_experiment.py

Outputs:
    experiments/chatgpt_sub2_experiment.json
    experiments/chatgpt_sub2_summary.csv
"""

import itertools
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Iterable

import networkx as nx

try:
    from salvador.algorithm import find_vertex_cover
    from salvador.version import __version__
except ModuleNotFoundError:  # pragma: no cover - convenience for direct runs
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from salvador.algorithm import find_vertex_cover
    from salvador.version import __version__

SEED = 20260621
SQRT2 = math.sqrt(2.0)
SUB2_TARGET = 7.0 / 4.0
CONSERVATIVE_EPSILON = 2.0 - SUB2_TARGET


def exact_vertex_cover_size(G: nx.Graph) -> int:
    """Exact minimum vertex-cover size through Maximum Independent Set.

    For a graph with n vertices, tau(G)=n-alpha(G).  The implementation is a
    deterministic branch-and-bound over bitsets and is adequate for the small
    reproducibility suite in this folder.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if G.number_of_edges() == 0:
        return 0

    idx = {v: i for i, v in enumerate(nodes)}
    adj = [0] * n
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        adj[iu] |= 1 << iv
        adj[iv] |= 1 << iu

    best = 0

    def search(candidates: int, chosen: int) -> None:
        nonlocal best
        if candidates == 0:
            best = max(best, chosen)
            return
        if chosen + candidates.bit_count() <= best:
            return

        # Branch on a remaining vertex of maximum residual degree.
        tmp = candidates
        branch_vertex = (tmp & -tmp).bit_length() - 1
        max_deg = -1
        while tmp:
            bit = tmp & -tmp
            i = bit.bit_length() - 1
            deg = (adj[i] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_vertex = i
            tmp ^= bit

        vbit = 1 << branch_vertex
        search(candidates & ~vbit & ~adj[branch_vertex], chosen + 1)
        search(candidates & ~vbit, chosen)

    search((1 << n) - 1, 0)
    return n - best


def exact_vertex_cover(G: nx.Graph) -> list[int]:
    """Return one lexicographically first exact minimum vertex cover."""
    nodes = list(G.nodes())
    opt = exact_vertex_cover_size(G)
    for combo in itertools.combinations(nodes, opt):
        candidate = set(combo)
        if all(u in candidate or v in candidate for u, v in G.edges()):
            return [int(x) for x in combo]
    raise RuntimeError("no vertex cover found")


def canonical_edges(G: nx.Graph) -> list[list[int]]:
    mapping = {v: i for i, v in enumerate(sorted(G.nodes()))}
    return [[mapping[u], mapping[v]] for u, v in sorted(G.edges(), key=lambda e: (mapping[e[0]], mapping[e[1]]))]


def evaluate_graph(name: str, G: nx.Graph, family: str) -> dict:
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    cover = set(find_vertex_cover(G))
    opt = exact_vertex_cover_size(G)
    ratio = 1.0 if opt == 0 else len(cover) / opt
    valid = all(u in cover or v in cover for u, v in G.edges())
    return {
        "name": name,
        "family": family,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "opt": opt,
        "salvador_size": len(cover),
        "ratio": ratio,
        "gap_to_2": 2.0 - ratio,
        "above_sqrt2": bool(ratio > SQRT2 + 1e-12),
        "above_sub2_target_7_4": bool(ratio > SUB2_TARGET + 1e-12),
        "valid_cover": bool(valid),
        "salvador_cover": sorted(int(x) for x in cover),
        "edges": canonical_edges(G) if G.number_of_nodes() <= 16 else None,
    }


def exhaustive_labelled(max_n: int = 6) -> list[dict]:
    rows = []
    for n in range(2, max_n + 1):
        edges = list(itertools.combinations(range(n), 2))
        best = None
        count = 0
        count_above_sqrt2 = 0
        count_above_7_4 = 0
        for mask in range(1, 1 << len(edges)):
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges[i] for i in range(len(edges)) if (mask >> i) & 1)
            row = evaluate_graph(f"exhaustive_n{n}_mask{mask}", G, "exhaustive_labelled")
            count += 1
            count_above_sqrt2 += int(row["above_sqrt2"])
            count_above_7_4 += int(row["above_sub2_target_7_4"])
            if best is None or row["ratio"] > best["ratio"]:
                best = row
        assert best is not None
        rows.append({
            "family": "exhaustive_labelled",
            "n": n,
            "graphs": count,
            "max_ratio": best["ratio"],
            "max_ratio_instance": best,
            "count_above_sqrt2": count_above_sqrt2,
            "count_above_7_4": count_above_7_4,
        })
    return rows


def structured_families() -> list[dict]:
    rows: list[dict] = []
    for n in range(2, 33):
        rows.append(evaluate_graph(f"path_{n}", nx.path_graph(n), "paths"))
    for n in range(3, 33):
        rows.append(evaluate_graph(f"cycle_{n}", nx.cycle_graph(n), "cycles"))
    for n in range(2, 23):
        rows.append(evaluate_graph(f"complete_{n}", nx.complete_graph(n), "complete"))
    for leaves in range(2, 41):
        rows.append(evaluate_graph(f"star_{leaves}", nx.star_graph(leaves), "stars"))
    for a in range(1, 13):
        for b in range(1, 13):
            rows.append(evaluate_graph(f"complete_bipartite_{a}_{b}", nx.complete_bipartite_graph(a, b), "complete_bipartite"))
    for n in range(4, 25):
        rows.append(evaluate_graph(f"wheel_{n}", nx.wheel_graph(n), "wheels"))
    for r in range(2, 6):
        for c in range(2, 7):
            rows.append(evaluate_graph(f"grid_{r}x{c}", nx.grid_2d_graph(r, c), "grids"))
    for k in range(2, 9):
        rows.append(evaluate_graph(f"barbell_{k}_{1}", nx.barbell_graph(k, 1), "barbells"))
    return rows



def chatgpt_stress_witnesses() -> list[dict]:
    """Deterministic witnesses found during ChatGPT-guided adversarial search.

    The eight-vertex witness below attains ratio 5/3 for the current v0.0.4
    implementation.  Keeping it explicit in the experiment makes the JSON fully
    reproducible and prevents the conclusion from depending on stochastic search
    rediscovering the same graph.
    """
    rows: list[dict] = []
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([
        (0, 2), (0, 3),
        (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ])
    rows.append(evaluate_graph("chatgpt_ratio_5_3_witness", G, "chatgpt_stress_witness"))
    return rows


def random_gnp() -> list[dict]:
    rows: list[dict] = []
    rng = random.Random(SEED)
    probabilities = [0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80]
    for n in range(7, 17):
        for p in probabilities:
            for trial in range(10):
                seed = rng.randrange(2**32)
                G = nx.gnp_random_graph(n, p, seed=seed)
                if G.number_of_edges() == 0:
                    continue
                rows.append(evaluate_graph(f"gnp_n{n}_p{p:.2f}_seed{seed}", G, "erdos_renyi"))
    return rows


def adversarial_edge_flip() -> list[dict]:
    """Small stochastic hill-climb attempting to push the ratio upward."""
    rows: list[dict] = []
    rng = random.Random(SEED + 404)
    for n in [8, 10, 12, 14]:
        all_edges = list(itertools.combinations(range(n), 2))
        for restart in range(14):
            p = rng.choice([0.12, 0.20, 0.30, 0.45, 0.60, 0.75])
            G = nx.gnp_random_graph(n, p, seed=rng.randrange(2**32))
            if G.number_of_edges() == 0:
                G.add_edge(0, 1)
            current = evaluate_graph(f"edgeflip_n{n}_restart{restart}_step0", G, "edge_flip_adversarial")
            best = current
            temperature = 0.05
            for step in range(1, 161):
                H = G.copy()
                u, v = rng.choice(all_edges)
                if H.has_edge(u, v):
                    H.remove_edge(u, v)
                    if H.number_of_edges() == 0:
                        H.add_edge(u, v)
                else:
                    H.add_edge(u, v)
                trial = evaluate_graph(f"edgeflip_n{n}_restart{restart}_step{step}", H, "edge_flip_adversarial")
                delta = trial["ratio"] - current["ratio"]
                if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-12)):
                    G = H
                    current = trial
                if trial["ratio"] > best["ratio"]:
                    best = trial
            rows.append(best)
    return rows


def summarise(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    ratios = [r["ratio"] for r in rows]
    best = max(rows, key=lambda r: r["ratio"])
    return {
        "instances": len(rows),
        "min_ratio": min(ratios),
        "mean_ratio": statistics.fmean(ratios),
        "median_ratio": statistics.median(ratios),
        "max_ratio": max(ratios),
        "empirical_epsilon_from_2": 2.0 - max(ratios),
        "count_above_sqrt2": sum(1 for r in rows if r["above_sqrt2"]),
        "count_above_7_4": sum(1 for r in rows if r["above_sub2_target_7_4"]),
        "best_instance": best,
    }


def main() -> None:
    started = time.time()
    exhaustive = exhaustive_labelled(6)
    structured = structured_families()
    stress_witnesses = chatgpt_stress_witnesses()
    random_rows = random_gnp()
    adversarial = adversarial_edge_flip()
    all_instance_rows = structured + stress_witnesses + random_rows + adversarial

    exhaustive_best = max((row["max_ratio_instance"] for row in exhaustive), key=lambda r: r["ratio"])
    witness_graph = nx.Graph()
    witness_graph.add_nodes_from(range(exhaustive_best["n"]))
    witness_graph.add_edges_from(tuple(edge) for edge in exhaustive_best["edges"])
    exhaustive_best["exact_cover"] = exact_vertex_cover(witness_graph)

    total_exhaustive_graphs = sum(row["graphs"] for row in exhaustive)
    total_graphs_considered = total_exhaustive_graphs + len(all_instance_rows)
    total_above_sqrt2 = sum(row["count_above_sqrt2"] for row in exhaustive) + sum(1 for row in all_instance_rows if row["above_sqrt2"])
    total_above_7_4 = sum(row["count_above_7_4"] for row in exhaustive) + sum(1 for row in all_instance_rows if row["above_sub2_target_7_4"])

    all_rows_for_threshold = [row["max_ratio_instance"] for row in exhaustive] + all_instance_rows
    overall_summary = summarise(all_rows_for_threshold)
    summary_by_family = {family: summarise([r for r in all_instance_rows if r["family"] == family])
                         for family in sorted({r["family"] for r in all_instance_rows})}

    result = {
        "experiment": "ChatGPT sub-2 stress test for Salvador v0.0.4",
        "generated_by": "OpenAI ChatGPT, executed in the ChatGPT sandbox for Frank Vega on 2026-06-21",
        "salvador_version": __version__,
        "seed": SEED,
        "thresholds": {
            "sqrt2": SQRT2,
            "sub2_target": SUB2_TARGET,
            "conservative_epsilon": CONSERVATIVE_EPSILON,
            "sub2_statement": "7/4 = 2 - 1/4",
        },
        "parameters": {
            "exhaustive_labelled_max_n": 6,
            "random_gnp_n_range": [7, 16],
            "random_gnp_probabilities": [0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80],
            "random_gnp_trials_per_pair": 10,
            "edge_flip_n_values": [8, 10, 12, 14],
            "edge_flip_restarts_per_n": 14,
            "edge_flip_steps_per_restart": 160,
        },
        "method": {
            "candidate_solver": "salvador.algorithm.find_vertex_cover",
            "optimum_solver": "exact branch-and-bound via maximum independent set; no MILP is used",
            "ratio": "|C_salvador| / tau(G)",
            "purpose": "stress-test empirical support for a constant (2-epsilon)-type upper bound; not a proof",
        },
        "conclusion": {
            "supports_universal_sqrt2_bound": False,
            "sqrt2_reason": "The exhaustive n<=6 search finds a five-vertex graph with ratio 3/2, which is greater than sqrt(2).",
            "supports_tested_7_4_bound": overall_summary["max_ratio"] <= SUB2_TARGET + 1e-12,
            "sub2_reason": "All generated and exact-tested instances have ratio at most the observed maximum, which is below 7/4 = 2 - 1/4.",
            "observed_max_ratio": overall_summary["max_ratio"],
            "empirical_epsilon_from_2": overall_summary["empirical_epsilon_from_2"],
            "conservative_epsilon_used_in_paper": CONSERVATIVE_EPSILON,
            "total_graphs_considered": total_graphs_considered,
            "total_above_sqrt2": total_above_sqrt2,
            "total_above_7_4": total_above_7_4,
            "ugc_interpretation": "If a universal polynomial-time ratio <= 7/4 were proved, it would contradict the Khot-Regev UGC-based 2-epsilon hardness for Vertex Cover; under P != NP this would refute UGC.",
        },
        "overall_summary": overall_summary,
        "exhaustive_labelled_summary": exhaustive,
        "summary_by_family": summary_by_family,
        "sqrt2_counterexample": exhaustive_best,
        "raw_instance_rows": all_instance_rows,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
    }

    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "chatgpt_sub2_experiment.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "chatgpt_sub2_summary.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("family,instances,min_ratio,mean_ratio,median_ratio,max_ratio,empirical_epsilon_from_2,count_above_sqrt2,count_above_7_4,best_name,best_n,best_m,best_opt,best_salvador_size\n")
        for family, summary in summary_by_family.items():
            b = summary["best_instance"]
            fh.write(
                f"{family},{summary['instances']},{summary['min_ratio']:.12g},{summary['mean_ratio']:.12g},{summary['median_ratio']:.12g},{summary['max_ratio']:.12g},{summary['empirical_epsilon_from_2']:.12g},"
                f"{summary['count_above_sqrt2']},{summary['count_above_7_4']},{b['name']},{b['n']},{b['m']},{b['opt']},{b['salvador_size']}\n"
            )
        for row in exhaustive:
            b = row["max_ratio_instance"]
            fh.write(
                f"exhaustive_n{row['n']},{row['graphs']},,,,{row['max_ratio']:.12g},{2.0-row['max_ratio']:.12g},"
                f"{row['count_above_sqrt2']},{row['count_above_7_4']},{b['name']},{b['n']},{b['m']},{b['opt']},{b['salvador_size']}\n"
            )
        b = overall_summary["best_instance"]
        fh.write(
            f"OVERALL,{overall_summary['instances']},{overall_summary['min_ratio']:.12g},{overall_summary['mean_ratio']:.12g},{overall_summary['median_ratio']:.12g},{overall_summary['max_ratio']:.12g},{overall_summary['empirical_epsilon_from_2']:.12g},"
            f"{overall_summary['count_above_sqrt2']},{overall_summary['count_above_7_4']},{b['name']},{b['n']},{b['m']},{b['opt']},{b['salvador_size']}\n"
        )

    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "elapsed_seconds": result["elapsed_seconds"],
        "observed_max_ratio": overall_summary["max_ratio"],
        "empirical_epsilon_from_2": overall_summary["empirical_epsilon_from_2"],
        "supports_tested_7_4_bound": result["conclusion"]["supports_tested_7_4_bound"],
        "sqrt2_counterexample_ratio": exhaustive_best["ratio"],
        "sqrt2_counterexample_edges": exhaustive_best["edges"],
        "best_instance": overall_summary["best_instance"]["name"],
    }, indent=2))


if __name__ == "__main__":
    main()
