# Salvador: Approximate Vertex Cover Solver

**Current version:** v0.1.4

![In Loving Memory of Salvador Vega (The Grandfather I Never Knew)](docs/salvador.jpg)

This work builds upon [The Salvador Algorithm](https://www.preprints.org/manuscript/202605.2000).

---

# The Minimum Vertex Cover Problem

The **Minimum Vertex Cover (MVC)** problem is a classic optimization problem in computer science and graph theory. It involves finding the smallest set of vertices in a graph that **covers** all edges, meaning at least one endpoint of every edge is included in the set.

## Formal Definition

Given an undirected graph $G = (V, E)$, a **vertex cover** is a subset $V' \subseteq V$ such that for every edge $(u, v) \in E$, at least one of $u$ or $v$ belongs to $V'$. The MVC problem seeks the vertex cover with the smallest cardinality.

## Importance and Applications

- **Theoretical Significance:** MVC is a well-known NP-hard problem, central to complexity theory.
- **Practical Applications:**
  - **Network Security:** Identifying critical nodes to disrupt connections.
  - **Bioinformatics:** Analyzing gene regulatory networks.
  - **Wireless Sensor Networks:** Optimizing sensor coverage.

## Related Problems

- **Maximum Independent Set:** The complement of a vertex cover.
- **Set Cover Problem:** A generalization of MVC.

---

## Problem Statement

Input: A Boolean Adjacency Matrix $M$.

Answer: Find a Minimum Vertex Cover.

### Example Instance: 5 x 5 matrix

|        | c1  | c2  | c3  | c4  | c5  |
| ------ | --- | --- | --- | --- | --- |
| **r1** | 0   | 0   | 1   | 0   | 1   |
| **r2** | 0   | 0   | 0   | 1   | 0   |
| **r3** | 1   | 0   | 0   | 0   | 1   |
| **r4** | 0   | 1   | 0   | 0   | 0   |
| **r5** | 1   | 0   | 1   | 0   | 0   |

The input for undirected graph is typically provided in [DIMACS](http://dimacs.rutgers.edu/Challenges) format. In this way, the previous adjacency matrix is represented in a text file using the following string representation:

```
p edge 5 4
e 1 3
e 1 5
e 2 4
e 3 5
```

This represents a 5x5 matrix in DIMACS format such that each edge $(v,w)$ appears exactly once in the input file and is not repeated as $(w,v)$. In this format, every edge appears in the form of

```
e W V
```

where the fields W and V specify the endpoints of the edge while the lower-case character `e` signifies that this is an edge descriptor line.

_Example Solution:_

Vertex Cover Found `1, 3, 4`: Nodes `1`, `3`, and `4` constitute an optimal solution.

---

# Compile and Environment

## Prerequisites

- Python ≥ 3.12

## Installation

```bash
pip install salvador==0.1.4
```

## Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/frankvegadelgado/salvador.git
   cd salvador
   ```

2. Run the script:

   ```bash
   vega -i ./benchmarks/testMatrix1
   ```

   utilizing the `vega` command provided by Salvador to execute the Boolean adjacency matrix `salvador\benchmarks\testMatrix1`. The file `testMatrix1` represents the example described herein. We also support `.xz`, `.lzma`, `.bz2`, and `.bzip2` compressed text files.

   **Example Output:**

   ```
   testMatrix1: Vertex Cover Found 1, 3, 4
   ```

   This indicates nodes `1, 3, 4` form a vertex cover.

---

## Vertex Cover Size

Use the `-c` flag to count the nodes in the vertex cover:

```bash
vega -i ./benchmarks/testMatrix2 -c
```

**Output:**

```
testMatrix2: Vertex Cover Size 5
```

---

# Command Options

Display help and options:

```bash
vega -h
```

**Output:**

```bash
usage: vega [-h] -i INPUTFILE [-a] [-b] [-c] [-v] [-l] [--version]

Compute the Approximate Vertex Cover for undirected graph encoded in DIMACS format.

options:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputFile INPUTFILE
                        input file path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a factor of at most 2
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the vertex cover
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Batch Execution

Batch execution allows you to solve multiple graphs within a directory consecutively.

To view available command-line options for the `batch_vega` command, use the following in your terminal or command prompt:

```bash
batch_vega -h
```

This will display the following help information:

```bash
usage: batch_vega [-h] -i INPUTDIRECTORY [-a] [-b] [-c] [-v] [-l] [--version]

Compute the Approximate Vertex Cover for all undirected graphs encoded in DIMACS format and stored in a directory.

options:
  -h, --help            show this help message and exit
  -i INPUTDIRECTORY, --inputDirectory INPUTDIRECTORY
                        Input directory path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a factor of at most 2
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the vertex cover
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Testing Application

A command-line utility named `test_vega` is provided for evaluating the Algorithm using randomly generated, large sparse matrices. It supports the following options:

```bash
usage: test_vega [-h] -d DIMENSION [-n NUM_TESTS] [-s SPARSITY] [-a] [-b] [-c] [-w] [-v] [-l] [--version]

The Salvador Testing Application using randomly generated, large sparse matrices.

options:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        an integer specifying the dimensions of the square matrices
  -n NUM_TESTS, --num_tests NUM_TESTS
                        an integer specifying the number of tests to run
  -s SPARSITY, --sparsity SPARSITY
                        sparsity of the matrices (0.0 for dense, close to 1.0 for very sparse)
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a factor of at most 2
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the vertex cover
  -w, --write           write the generated random matrix to a file in the current directory
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Reproducibility Experiments (`car/`)

The `car/` folder contains the experiment scripts. Each script writes its results next to itself as `.json` and `.csv` files.

## How to run

**Recommended:** run every experiment quickly, on small graphs, with one command from the `car/` folder:

```bash
cd car
python run_all.py --quick
```

Install the package and its dependencies, then run a script from the repository root:

```bash
pip install salvador==0.1.4
python car/car_strategies.py
```

The scripts also run from a clone of the repository without installing the package. Every script accepts `-h` to list its options, `--quick` for a shorter run and `--timeout` to limit the time spent on each graph.

To run all the experiments one after another, use `car/run_all.py`:

```bash
python car/run_all.py --quick                          # every experiment on small graphs (much shorter)
python car/run_all.py --quick --only strategies scaling # only some experiments
python car/run_all.py --list                           # list the experiments and their quick options
python car/run_all.py                                  # every experiment in full (hours)
python car/run_all.py --skip triangle_free mechanism   # full runs except some
```

Each experiment writes its own result files, and `car/run_all_log.txt` records the time and status of each one.

## Strategies

Salvador 0.1.4 runs seventeen strategies in linear time, numbered in the order in which they are computed: the base strategies first, the improvers last.

| Strategy | What it does |
| -------- | ------------ |
| `c1` | Component amplification: copies (u, 0) and (u, 1) only; edges of the NetworkX minimum maximal matching with both endpoints at index 0 are split off until the split graph has as many components as possible without exceeding the matching size; the NetworkX minimum maximal matching of the split graph is decoded by copy index and degree. |
| `c2` | Maximal matching (NetworkX minimum maximal matching). |
| `c3` | Maximum-degree greedy (bucket queue). |
| `c4` | Degree-1 weighted reduction (Hallelujah). |
| `c5` | Min-to-Min. |
| `c6` | Primal-dual 2-approximation. |
| `c7` | Planar forest-core reduction to a weighted MIDS gadget (Baker-style PTAS, epsilon = 1). |
| `c8` | Bipartite planar oriented-incidence reduction (cycle DP). |
| `c9` | Bipartite layers: matching + bounded Hopcroft–Karp + Koenig on bipartite components; non-bipartite components split into bipartite layers solved in several orders. |
| `c10` | Edge-dominating-set gadget + treewidth DP on the bipartite components. |
| `c11` | Weighted subdivision reduction + reducing-peeling. |
| `c12` | Bounded-treewidth exact DP. |
| `c13` | Union of `c2`...`c8`, re-pruned. |
| `c14` | (1,2)-swap local search from the best of `c1`...`c13`. |
| `c15` | Reduce-and-search from the best of `c1`...`c14`. |
| `c16` | Consensus forcing over `c1`...`c15`. |
| `c17` | Swap-maximize refinement of every candidate; the smallest refined cover is returned. |

## What each script does

| Script | What it does |
| ------ | ------------ |
| `car_ratio.py` | Measures the approximation ratio on small graphs with a known optimum and on large adversarial graphs. |
| `car_ratio_timing.py` | Shows where the running time of `car_ratio.py` goes, strategy by strategy. |
| `car_scaling.py` | Measures how the running time grows with the size of the graph. |
| `car_strategies.py` | Reports which of the strategies `c1`...`c17` finds the smallest cover, and how often: wins among the base strategies `c1`...`c12`, and how often each later strategy (`c10`...`c17`) improves on the best of the strategies before it. |
| `car_c11_variants.py` | Compares the variants of strategy `c11`. |
| `car_c10_bags.py` | Checks the bag sizes of the dynamic program of strategy `c10`. |
| `car_c1_components.py` | Measures strategy `c1` (component amplification with the copy indices 0 and 1 before the NetworkX minimum maximal matching): every decoding rule against `c2`, tau (or a certified lower bound) and the rest of the ensemble. |
| `car_c1_fix.py` | Tests the bound size(c1) <= 2 tau - r' (r' = number of components of the split graph) for every decoding rule and completion of `c1` and for `c1` itself. On graphs with a known optimum it checks the bound exactly; on the others it marks each graph certified, refuted or undecided from a certified lower bound. |
| `car_c9_layers.py` | Studies strategy `c9`: bipartite components are solved with the NetworkX minimum maximal matching, a few Hopcroft–Karp phases and Koenig's theorem; non-bipartite components are split into bipartite layers (at most log2 m + 1), solved in several orders, a vertex selected in one layer being discarded from the later ones. It checks the bound on the number of layers and compares `c9` with the optimum (or a certified lower bound), `c1`, `c2`, the other candidates and the final cover `c17`. |
| `car_regular.py` | Runs regular graphs of increasing degree. |
| `car_bipartite.py` | Runs bipartite graphs, where the optimum is known exactly. |
| `car_yes_instances.py` | Runs YES instances of the Khot–Regev reduction from Unique Games. |
| `car_mechanism.py` | Studies which proved bounds apply to each graph and why the covers are good. |
| `car_triangle_free.py` | Runs triangle-free graphs. |
| `run_all.py` | Runs all the scripts above, quickly (`--quick`) or in full. |

Examples:

```bash
python car/car_ratio.py
python car/car_bipartite.py
python car/car_scaling.py --quick
python car/car_triangle_free.py --quick
python car/car_c1_fix.py                                # full c1 bound study
python car/car_c9_layers.py                             # full c9 study
python car/car_c9_layers.py --dimacs experiment         # also on the DIMACS clique complements
```

---

# Code

- Python implementation by **Frank Vega**.

---

# License

- MIT License.

