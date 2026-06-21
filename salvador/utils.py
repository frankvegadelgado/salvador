"""General utility functions for Salvador."""

from __future__ import annotations

import os
import random
import string
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sparse


def get_file_names(directory: str) -> list[str]:
    """Return file names directly contained in ``directory``."""
    try:
        return sorted(
            f for f in os.listdir(directory) if not os.path.isdir(os.path.join(directory, f))
        )
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []


def get_file_name(filepath: str) -> str:
    """Return the final path component of ``filepath``."""
    return os.path.basename(filepath)


def get_extension_without_dot(filepath: str) -> str | None:
    """Return the file extension without the leading dot, if present."""
    return Path(get_file_name(filepath)).suffix[1:] or None


def has_one_on_diagonal(adjacency_matrix) -> bool:
    """Return ``True`` when a SciPy sparse matrix has a diagonal one."""
    diagonal = adjacency_matrix.diagonal()
    return bool(np.any(diagonal == 1))


def generate_short_hash(length: int = 6) -> str | None:
    """Generate a short random alphanumeric string."""
    if not isinstance(length, int) or length <= 0:
        print("Error: Length must be a positive integer.")
        return None

    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def make_symmetric(matrix):
    """Return a symmetric SciPy sparse matrix from a square sparse matrix."""
    if not sparse.issparse(matrix):
        raise TypeError("Input must be a SciPy sparse matrix.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix must be square to be made symmetric.")

    coo = matrix.tocoo()
    row_sym = np.concatenate([coo.row, coo.col])
    col_sym = np.concatenate([coo.col, coo.row])
    data_sym = np.concatenate([coo.data, coo.data])

    symmetric_matrix = sparse.csc_matrix((data_sym, (row_sym, col_sym)), shape=(rows, cols))
    symmetric_matrix.sum_duplicates()
    return symmetric_matrix


def random_matrix_tests(matrix_shape: tuple[int, int], sparsity: float = 0.9):
    """Generate one random symmetric sparse adjacency matrix."""
    rows, cols = matrix_shape
    size = rows * cols
    num_elements = int(size * (1 - sparsity))
    row_indices = np.random.randint(0, rows, size=num_elements, dtype=np.int32)
    col_indices = np.random.randint(0, cols, size=num_elements, dtype=np.int32)
    data = np.ones(num_elements, dtype=np.int8)

    sparse_matrix = sparse.csc_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
    symmetric_matrix = make_symmetric(sparse_matrix)
    symmetric_matrix.setdiag(0)
    symmetric_matrix.eliminate_zeros()
    return symmetric_matrix


def string_result_format(result: Iterable[Any] | None, count_result: bool = False) -> str:
    """Format a vertex-cover result for CLI output."""
    if result:
        result_set = set(result)
        if count_result:
            return f"Vertex Cover Size {len(result_set)}"
        formatted_string = ", ".join(str(x + 1) for x in sorted(result_set))
        return f"Vertex Cover Found {formatted_string}"
    return "Empty Graph"


def println(output: str, logger, file_logging: bool = False) -> None:
    """Print final output and optionally mirror it to the logger."""
    if file_logging:
        logger.info(output)
    print(output)


def sparse_matrix_to_graph(adj_matrix, is_directed: bool = False) -> nx.Graph:
    """Convert a SciPy sparse adjacency matrix to a NetworkX graph."""
    rows, cols = adj_matrix.nonzero()
    graph = nx.DiGraph() if is_directed else nx.Graph()

    for i, j in zip(rows, cols):
        if is_directed:
            if not graph.has_edge(i, j):
                graph.add_edge(i, j)
        elif i < j:
            graph.add_edge(i, j)

    return graph


def is_vertex_redundant(graph: nx.Graph, vertex: Any, vertex_set: set[Any]) -> bool:
    """Return whether ``vertex`` covers no edge not already covered by ``vertex_set``."""
    edges_covered_by_set = set()
    for v in vertex_set:
        edges_covered_by_set.update(graph.edges(v))

    edges_covered_by_vertex = set(graph.edges(vertex))
    return edges_covered_by_vertex.issubset(edges_covered_by_set)


def is_vertex_cover(graph: nx.Graph, vertex_cover: set[Any]) -> bool:
    """Return whether ``vertex_cover`` covers every edge of ``graph``."""
    return all(u in vertex_cover or v in vertex_cover for u, v in graph.edges())


def is_independent_set(graph: nx.Graph, subset: Iterable[Any]) -> bool:
    """Return whether ``subset`` is an independent set in ``graph``."""
    nodes = list(subset)
    return all(not graph.has_edge(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :])


def compute_weight(graph: nx.Graph, nodes: Iterable[Any]) -> float:
    """Compute the total ``weight`` attribute over ``nodes``."""
    return sum(graph.nodes[node]["weight"] for node in nodes)
