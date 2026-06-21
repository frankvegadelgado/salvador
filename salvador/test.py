"""Random sparse-matrix testing command for Salvador."""

from __future__ import annotations

import argparse
import math
import time

from . import algorithm, applogger, parser, utils
from .version import __version__


def restricted_float(value: str) -> float:
    """Parse a float constrained to the closed interval [0, 1]."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a floating-point literal") from exc

    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(f"{value!r} not in range [0.0, 1.0]")
    return parsed


def main() -> None:
    helper = argparse.ArgumentParser(
        prog="test_vega",
        description="The Salvador testing application using randomly generated, large sparse matrices.",
    )
    helper.add_argument("-d", "--dimension", type=int, help="dimension of the square matrices", required=True)
    helper.add_argument("-n", "--num_tests", type=int, default=5, help="number of tests to run")
    helper.add_argument(
        "-s",
        "--sparsity",
        type=restricted_float,
        default=0.95,
        help="sparsity of the matrices (0.0 for dense, close to 1.0 for very sparse)",
    )
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
    helper.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="write each generated random matrix to a DIMACS file in the current directory",
    )
    helper.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    helper.add_argument("-l", "--log", action="store_true", help="enable file logging")
    helper.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = helper.parse_args()
    logger = applogger.Logger(applogger.FileLogger() if args.log else applogger.ConsoleLogger(args.verbose))
    hash_string = utils.generate_short_hash(6 + math.ceil(math.log2(args.num_tests))) if args.write else None

    for i in range(args.num_tests):
        logger.info(f"Creating Matrix {i + 1}")
        sparse_matrix = utils.random_matrix_tests((args.dimension, args.dimension), args.sparsity)
        graph = utils.sparse_matrix_to_graph(sparse_matrix)

        logger.info(f"Matrix shape: {sparse_matrix.shape}")
        logger.info(f"Number of non-zero elements: {sparse_matrix.nnz}")
        logger.info(f"Sparsity: {1 - (sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]))}")

        approximate_result = None
        brute_force_result = None

        if args.approximation:
            logger.info("An approximate Solution with an approximation ratio of at most 2 started")
            started = time.time()
            approximate_result = algorithm.find_vertex_cover_approximation(graph)
            logger.info(
                "An approximate Solution with an approximation ratio of at most 2 done in: "
                f"{(time.time() - started) * 1000.0} milliseconds"
            )
            answer = utils.string_result_format(approximate_result, args.count)
            utils.println(f"{i + 1}-approximation Test: {answer}", logger, args.log)

        if args.bruteForce:
            logger.info("A solution with an exponential-time complexity started")
            started = time.time()
            brute_force_result = algorithm.find_vertex_cover_brute_force(graph)
            logger.info(
                "A solution with an exponential-time complexity done in: "
                f"{(time.time() - started) * 1000.0} milliseconds"
            )
            answer = utils.string_result_format(brute_force_result, args.count)
            utils.println(f"{i + 1}-Brute Force Test: {answer}", logger, args.log)

        logger.info("Our Algorithm with an approximate solution started")
        started = time.time()
        novel_result = algorithm.find_vertex_cover(graph)
        logger.info(
            "Our Algorithm with an approximate solution done in: "
            f"{(time.time() - started) * 1000.0} milliseconds"
        )

        answer = utils.string_result_format(novel_result, args.count)
        utils.println(f"{i + 1}-Salvador Test: {answer}", logger, args.log)

        if novel_result and args.bruteForce and brute_force_result:
            utils.println(
                f"Exact Ratio (Salvador/Optimal): {len(novel_result) / len(brute_force_result)}",
                logger,
                args.log,
            )
        elif novel_result and args.approximation and approximate_result:
            utils.println(
                f"Upper Bound for Ratio (Salvador/Optimal): {2 * len(novel_result) / len(approximate_result)}",
                logger,
                args.log,
            )

        if args.write:
            utils.println(f"Saving Matrix Test {i + 1}", logger, args.log)
            filename = f"sparse_matrix_{i + 1}_{hash_string}"
            parser.save_sparse_matrix_to_file(sparse_matrix, filename)
            utils.println(f"Matrix Test {i + 1} written to file {filename}.", logger, args.log)


if __name__ == "__main__":
    main()
