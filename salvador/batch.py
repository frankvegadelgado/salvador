"""Batch command-line entry point for directories of DIMACS instances."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import app, utils
from .version import __version__


def approximate_solutions(
    inputDirectory: str,
    verbose: bool = False,
    log: bool = False,
    count: bool = False,
    bruteForce: bool = False,
    approximation: bool = False,
) -> None:
    """Run Salvador on every regular file directly contained in a directory."""
    file_names = utils.get_file_names(inputDirectory)
    for file_name in file_names:
        input_file = str(Path(inputDirectory) / file_name)
        print(f"Test: {input_file}")
        app.approximate_solution(input_file, verbose, log, count, bruteForce, approximation)


def main() -> None:
    helper = argparse.ArgumentParser(
        prog="batch_vega",
        description="Compute approximate vertex covers for all DIMACS graphs in a directory.",
    )
    helper.add_argument("-i", "--inputDirectory", type=str, help="Input directory path", required=True)
    helper.add_argument(
        "-a",
        "--approximation",
        action="store_true",
        help="enable comparison with a polynomial-time approximation approach within a factor of at most 2",
    )
    helper.add_argument(
        "-b",
        "--bruteForce",
        action="store_true",
        help="enable comparison with the exponential-time brute-force approach",
    )
    helper.add_argument("-c", "--count", action="store_true", help="calculate the size of the vertex cover")
    helper.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    helper.add_argument("-l", "--log", action="store_true", help="enable file logging")
    helper.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = helper.parse_args()
    approximate_solutions(
        args.inputDirectory,
        verbose=args.verbose,
        log=args.log,
        count=args.count,
        bruteForce=args.bruteForce,
        approximation=args.approximation,
    )


if __name__ == "__main__":
    main()
