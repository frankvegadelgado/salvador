"""Command-line entry point for solving one DIMACS instance."""

from __future__ import annotations

import argparse
import time

from . import algorithm, applogger, parser, utils
from .version import __version__


def approximate_solution(
    inputFile: str,
    verbose: bool = False,
    log: bool = False,
    count: bool = False,
    bruteForce: bool = False,
    approximation: bool = False,
) -> None:
    """Find and print Salvador's approximate vertex cover for one input file."""
    logger = applogger.Logger(applogger.FileLogger() if log else applogger.ConsoleLogger(verbose))

    logger.info("Parsing the Input File started")
    started = time.time()
    graph = parser.read(inputFile)
    filename = utils.get_file_name(inputFile)
    logger.info(f"Parsing the Input File done in: {(time.time() - started) * 1000.0} milliseconds")

    approximate_result = None
    brute_force_result = None

    if approximation:
        logger.info("An approximate Solution with an approximation ratio of at most 2 started")
        started = time.time()
        approximate_result = algorithm.find_vertex_cover_approximation(graph)
        logger.info(
            "An approximate Solution with an approximation ratio of at most 2 done in: "
            f"{(time.time() - started) * 1000.0} milliseconds"
        )
        answer = utils.string_result_format(approximate_result, count)
        utils.println(f"{filename}: (approximation) {answer}", logger, log)

    if bruteForce:
        logger.info("A solution with an exponential-time complexity started")
        started = time.time()
        brute_force_result = algorithm.find_vertex_cover_brute_force(graph)
        logger.info(
            "A solution with an exponential-time complexity done in: "
            f"{(time.time() - started) * 1000.0} milliseconds"
        )
        answer = utils.string_result_format(brute_force_result, count)
        utils.println(f"{filename}: (Brute Force) {answer}", logger, log)

    logger.info("Our Algorithm with an approximate solution started")
    started = time.time()
    novel_result = algorithm.find_vertex_cover(graph)
    logger.info(
        "Our Algorithm with an approximate solution done in: "
        f"{(time.time() - started) * 1000.0} milliseconds"
    )

    answer = utils.string_result_format(novel_result, count)
    utils.println(f"{filename}: {answer}", logger, log)

    if novel_result and bruteForce and brute_force_result:
        utils.println(
            f"Exact Ratio (Salvador/Optimal): {len(novel_result) / len(brute_force_result)}",
            logger,
            log,
        )
    elif novel_result and approximation and approximate_result:
        utils.println(
            f"Upper Bound for Ratio (Salvador/Optimal): {2 * len(novel_result) / len(approximate_result)}",
            logger,
            log,
        )


def main() -> None:
    helper = argparse.ArgumentParser(
        prog="vega",
        description="Compute the approximate vertex cover for an undirected graph encoded in DIMACS format.",
    )
    helper.add_argument("-i", "--inputFile", type=str, help="input file path", required=True)
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
    approximate_solution(
        args.inputFile,
        verbose=args.verbose,
        log=args.log,
        count=args.count,
        bruteForce=args.bruteForce,
        approximation=args.approximation,
    )


if __name__ == "__main__":
    main()
