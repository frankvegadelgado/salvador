"""Backward-compatible entry point for the car/ suite.

The suite is now split into separate scripts (shared code in car_common.py):

    python car/car_ratio.py       # approximation ratio (Part A exact + Part B bounds)
    python car/car_scaling.py     # doubling-size O(n + m) timing study
    python car/car_strategies.py  # which ensemble strategy wins, and how often

Running this file runs car_ratio.py and then car_scaling.py with the same
``--max-n``; the old ``--quick``/``--skip-small``/``--skip-large`` flags are
forwarded to car_ratio.py (``--skip-large`` also skips the scaling study).
"""

from __future__ import annotations

import argparse
import sys

import car_ratio
import car_scaling


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-large", action="store_true")
    ap.add_argument("--skip-small", action="store_true")
    ap.add_argument("--max-n", type=int, default=200_000)
    args = ap.parse_args()
    ratio_argv = [sys.argv[0], "--max-n", str(args.max_n)]
    ratio_argv += ["--quick"] * args.quick + ["--skip-large"] * args.skip_large + ["--skip-small"] * args.skip_small
    sys.argv = ratio_argv
    car_ratio.main()
    if not args.skip_large:
        sys.argv = [sys.argv[0], "--max-n", str(args.max_n)]
        car_scaling.main()


if __name__ == "__main__":
    main()
