"""
Minimum Weighted Vertex Cover in Bipartite Graphs using Linear Programming
Based on George Dantzig's Linear Programming approach and Total Unimodularity

The key insight is that for bipartite graphs, the constraint matrix of the 
vertex cover LP relaxation is totally unimodular, which means the LP relaxation
gives integer solutions, making it equivalent to the integer program.

Author: Implementation based on Dantzig's Linear Programming theory
"""

from networkx.algorithms import bipartite
import io
import contextlib
# Suppress OR-Tools DLL loading output
with contextlib.redirect_stdout(io.StringIO()):
    from ortools.linear_solver import pywraplp

        
def find_vertex_cover(G, weight='weight'):
    """
    Solve minimum weighted vertex cover using Dantzig's LP approach.
    
    Mathematical Formulation:
    minimize: Σ w_i * x_i for all vertices i
    subject to: x_u + x_v >= 1 for all edges (u,v)
                0 <= x_i <= 1 for all vertices i
    
    For bipartite graphs, the constraint matrix is totally unimodular,
    so the LP relaxation gives integer solutions.
    
    Parameters:
    -----------
    G : networkx.Graph
        Bipartite graph with node weights
    weight : str
        Node attribute name for weights
    verbose : bool
        Print detailed solution information
        
    Returns:
    --------
    tuple: (vertex_cover_set, total_weight, solution_info)
    """
    
    # Verify bipartite property
    if not bipartite.is_bipartite(G):
        raise ValueError("Graph must be bipartite for this algorithm")
    
    # Initialize solver using Dantzig's simplex method (GLOP)
    solver = pywraplp.Solver.CreateSolver('GLOP')
            
    if not solver:
        raise RuntimeError("Could not create solver")
    
    # Create binary variables for each node
    x = {}
    for node in G.nodes():
        x[node] = solver.NumVar(0, 1, f'x_{node}')

    # Objective function: minimize the sum of x_i
    solver.Minimize(solver.Sum([G.nodes[node].get(weight, 1) * x[node] for node in G.nodes()]))

    # Constraints: each node must be dominated by at least one node in the set
    for u, v in G.edges():
        solver.Add(x[u] + x[v] >= 1)
            
    # Solve the problem
    status = solver.Solve()

    # Extract the solution
    if status == pywraplp.Solver.OPTIMAL:
        vertex_cover = [node for node in G.nodes() if x[node].solution_value() > 0.5]
        return vertex_cover
    else:
        raise Exception("No optimal solution found.")