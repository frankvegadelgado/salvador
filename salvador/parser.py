"""DIMACS graph parser and writer utilities."""

from __future__ import annotations

import bz2
import lzma
from pathlib import Path
from typing import Iterable, TextIO

import networkx as nx

from . import utils

_COMPRESSED_OPENERS = {
    "xz": lzma.open,
    "lzma": lzma.open,
    "bz2": bz2.open,
    "bzip2": bz2.open,
}


def create_sparse_matrix_from_file(file: Iterable[str]) -> nx.Graph:
    """Create a NetworkX graph from a DIMACS edge-list stream.

    Only edge descriptor lines of the form ``e u v`` are used. Comment lines,
    problem-header lines, empty lines, and unrecognised metadata are skipped.
    Vertex labels are converted from DIMACS' one-based convention to the
    package's zero-based internal representation.
    """
    graph = nx.Graph()

    for line_number, raw_line in enumerate(file, start=1):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue

        parts = line.split()
        if len(parts) < 3 or parts[0].lower() != "e":
            continue

        try:
            u, v = int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid DIMACS edge at line {line_number}: {raw_line.rstrip()}"
            ) from exc

        if u <= 0 or v <= 0:
            raise ValueError(
                f"DIMACS vertices must be positive at line {line_number}: {raw_line.rstrip()}"
            )

        graph.add_edge(u - 1, v - 1)

    return graph


def save_sparse_matrix_to_file(matrix, filename: str) -> None:
    """Write a SciPy sparse adjacency matrix in DIMACS edge format."""
    rows, cols = matrix.nonzero()
    edges = [(int(i), int(j)) for i, j in zip(rows, cols) if i < j]

    with Path(filename).open("w", encoding="utf-8") as file:
        file.write(f"p edge {matrix.shape[0]} {len(edges)}\n")
        for i, j in edges:
            file.write(f"e {i + 1} {j + 1}\n")


def _open_text(filepath: str) -> TextIO:
    extension = utils.get_extension_without_dot(filepath)
    opener = _COMPRESSED_OPENERS.get(extension or "")
    if opener is not None:
        return opener(filepath, "rt")
    return Path(filepath).open("r", encoding="utf-8")


def read(filepath: str) -> nx.Graph:
    """Read a DIMACS graph, including supported compressed text formats."""
    try:
        with _open_text(filepath) as file:
            return create_sparse_matrix_from_file(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {filepath}") from exc
