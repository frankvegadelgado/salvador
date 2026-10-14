"""car/ timing study of car_ratio.py Part A: where does the time go?

Runs exactly the Part A sweep of car_ratio.py (same instances, same seed, same
hill climb) and times every piece of each find_vertex_cover call without
changing its result:

* c6   -- vc_reduction.solve_vc (Baker PTAS reduction)
* c8   -- bipartite_reduction.bipartite_planar_covers
* c9   -- local_search.improve_vertex_cover
* c10  -- eds_treewidth.eds_treewidth_cover (EDS gadget, min-degree order, bags <= 6)
* c11  -- subdivision_reduction.subdivision_vertex_cover (weighted subdivision)
* c12  -- reduce_search.reduce_search_cover (ILS + exact reductions)
* c13  -- treewidth_dp.treewidth_vertex_cover
* c14  -- refine_cover (all calls of one find_vertex_cover)
* rest -- everything else inside find_vertex_cover (c1..c5, c7, pruning)
* exact -- car_common.exact_vertex_cover_size (the exact optimum, not Salvador)

Run from the repository root:

    python car/car_ratio_timing.py            # full Part A (as python car/car_ratio.py)
    python car/car_ratio_timing.py --quick    # as python car/car_ratio.py --quick

Prints a progress line every --every evaluations, then the total seconds per
piece and the slowest instances. Outputs (in car/): car_ratio_timing.json and
car_ratio_timing.csv (one row per evaluation).
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import car_common as cc
import car_ratio
from salvador import (algorithm, bipartite_reduction, eds_treewidth, local_search, reduce_search,
                      subdivision_reduction,
                      treewidth_dp, vc_reduction)

PIECES = ("c6", "c8", "c9", "c10", "c11", "c12", "c13", "c14")
_current: dict[str, float] = {}


def _timed(piece: str, fn):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _current[piece] = _current.get(piece, 0.0) + time.perf_counter() - t0
    return wrapper


def install() -> None:
    """Wrap the pieces in place (results are unchanged; only time is recorded)."""
    vc_reduction.solve_vc = _timed("c6", vc_reduction.solve_vc)
    bipartite_reduction.bipartite_planar_covers = _timed("c8", bipartite_reduction.bipartite_planar_covers)
    local_search.improve_vertex_cover = _timed("c9", local_search.improve_vertex_cover)
    eds_treewidth.eds_treewidth_cover = _timed("c10", eds_treewidth.eds_treewidth_cover)
    subdivision_reduction.subdivision_vertex_cover = _timed(
        "c11", subdivision_reduction.subdivision_vertex_cover)
    reduce_search.reduce_search_cover = _timed("c12", reduce_search.reduce_search_cover)
    treewidth_dp.treewidth_vertex_cover = _timed("c13", treewidth_dp.treewidth_vertex_cover)
    algorithm.refine_cover = _timed("c14", algorithm.refine_cover)


def main() -> None:
    ap = argparse.ArgumentParser(description="Timing of car_ratio.py Part A per strategy.")
    ap.add_argument("--quick", action="store_true", help="smaller Part A (as car_ratio.py --quick)")
    ap.add_argument("--every", type=int, default=100, help="progress line every N evaluations (default: 100)")
    args = ap.parse_args()
    install()

    rows: list[dict] = []
    started = time.perf_counter()
    original_find = car_ratio.find_vertex_cover
    original_exact = cc.exact_vertex_cover_size
    pending: dict = {}

    def find(G, *a, **k):
        _current.clear()
        t0 = time.perf_counter()
        cover = original_find(G, *a, **k)
        total = time.perf_counter() - t0
        pending.clear()
        pending.update({"n": G.number_of_nodes(), "m": G.number_of_edges(), "salvador": total,
                        **{p: _current.get(p, 0.0) for p in PIECES}})
        pending["rest"] = total - sum(_current.get(p, 0.0) for p in PIECES)
        return cover

    def exact(G):
        t0 = time.perf_counter()
        res = original_exact(G)
        row = dict(pending)
        row["exact"] = time.perf_counter() - t0
        row["index"] = len(rows)
        rows.append(row)
        if len(rows) % args.every == 0:
            print(f"  {len(rows):5d} evaluations, {time.perf_counter() - started:7.1f} s", flush=True)
        return res

    car_ratio.find_vertex_cover = find
    cc.exact_vertex_cover_size = exact
    result = car_ratio.run_part_a(big=not args.quick)
    elapsed = time.perf_counter() - started

    del result
    keys = ("salvador", *PIECES, "rest", "exact")
    totals = {k: round(sum(r[k] for r in rows), 3) for k in keys}
    print(f"\nPart A: {len(rows)} evaluations in {elapsed:.1f} s "
          f"({'quick' if args.quick else 'full'}, c10 order {eds_treewidth.DEFAULT_ORDER})")
    print(f"{'piece':10} {'seconds':>10} {'share':>7}")
    for k in keys:
        share = totals[k] / elapsed if elapsed else 0.0
        print(f"{k:10} {totals[k]:>10.2f} {share:>6.1%}")
    slow = sorted(rows, key=lambda r: r["salvador"], reverse=True)[:15]
    print("\nSlowest find_vertex_cover calls")
    print(f"{'#':>5} {'n':>4} {'m':>5} {'total':>8} {'c10':>8} {'c11':>8} {'c12':>8} {'c13':>8} {'c9':>8} {'c14':>8}")
    for r in slow:
        print(f"{r['index']:>5} {r['n']:>4} {r['m']:>5} {r['salvador']:>8.3f} {r['c10']:>8.3f} {r['c11']:>8.3f} {r['c12']:>8.3f} "
              f"{r['c13']:>8.3f} {r['c9']:>8.3f} {r['c14']:>8.3f}")

    out = cc.OUT_DIR
    with open(out / "car_ratio_timing.json", "w") as fh:
        json.dump({"environment": cc.environment(), "quick": args.quick, "evaluations": len(rows),
                   "elapsed_seconds": elapsed, "totals": totals, "c10_order": eds_treewidth.DEFAULT_ORDER,
                   "slowest": slow}, fh, indent=1)
    with open(out / "car_ratio_timing.csv", "w", newline="") as fh:
        fields = ["index", "n", "m", *keys]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
