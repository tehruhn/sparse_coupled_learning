# Code for computing learning in sparse flow networks
import numpy as np
from scipy.sparse import csc_matrix

def create_square_grid_graph(a, b, periodic=False):
    """
    Construct a square grid graph.

    Parameters:
    a (int): The width of the grid.
    b (int): The height of the grid.
    periodic (bool): If True, create a periodic grid (toroidal).

    Returns:
    NN (int): The number of nodes in the grid.
    NE (int): The number of edges in the grid.
    EI (list): The list of start nodes for each edge.
    EJ (list): The list of end nodes for each edge.
    """
    NN = a * b
    EI, EJ = [], []

    # Add horizontal and vertical edges
    for i in range(b):
        for j in range(a):
            if j < a - 1:  # Horizontal edges
                EI.append(i * a + j)
                EJ.append(i * a + j + 1)
            if i < b - 1:  # Vertical edges
                EI.append(i * a + j)
                EJ.append((i + 1) * a + j)

    # Add periodic edges if needed
    if periodic:
        for j in range(a):  # Connect last row to first row
            EI.append(j)
            EJ.append((b - 1) * a + j)     
        for i in range(b):  # Connect last column to first column
            EI.append(i * a)
            EJ.append(i * a + a - 1)

    EI = np.array(EI)
    EJ = np.array(EJ)
    NE = len(EI)

    return NN, NE, EI, EJ


def create_sparse_incidence_constraint_matrix(SourceNodes, SourceEdges, TargetNodes, TargetEdges, GroundNodes, NN, EI, EJ):
    """
    Construct sparse incidence and constraint matrices for a graph.

    Parameters:
    SourceNodes (list): Nodes that are sources.
    SourceEdges (list): Edges that are sources.
    TargetNodes (list): Nodes that are targets.
    TargetEdges (list): Edges that are targets.
    GroundNodes (list): Nodes that are grounded.
    NN (int): Total number of nodes.
    EI (list): Starting nodes for each edge.
    EJ (list): Ending nodes for each edge.

    Returns:
    tuple: A tuple containing the following matrices:
           - sDMF (csc_matrix): Incidence matrix for the free state.
           - sDMC (csc_matrix): Incidence matrix for the clamped state.
           - sBLF (csc_matrix): Constraint border Laplacian matrix for the free state.
           - sBLC (csc_matrix): Constraint border Laplacian matrix for the clamped state.
           - sDot (csc_matrix): Matrix for cost computation.
    """
    NE = len(EI)
    dataF, xF, yF = [], [], []
    dataC, xC, yC = [], [], []
    ncF, ncC = NN, NN

    # Consolidate constraints for ground, source, and target nodes and edges
    for nodes, data, x, y, nc in [(GroundNodes, dataF, xF, yF, ncF), 
                                 (GroundNodes, dataC, xC, yC, ncC), 
                                 (SourceNodes, dataF, xF, yF, ncF), 
                                 (SourceNodes, dataC, xC, yC, ncC), 
                                 (SourceEdges, dataF, xF, yF, ncF), 
                                 (SourceEdges, dataC, xC, yC, ncC), 
                                 (TargetNodes, dataC, xC, yC, ncC), 
                                 (TargetEdges, dataC, xC, yC, ncC)]:
        for node in nodes:
            # Node constraints
            if isinstance(node, int):
                data.extend([1., 1.])
                x.extend([node, nc])
                y.extend([nc, node])
                nc += 1
            # Edge constraints
            else:
                data.extend([1., 1., -1., -1.])
                x.extend([EI[node], nc, EJ[node], nc])
                y.extend([nc, EI[node], nc, EJ[node]])
                nc += 1

    # Construct incidence and constraint matrices
    sDMF = csc_matrix((np.r_[np.ones(NE), -np.ones(NE)], (np.r_[np.arange(NE), np.arange(NE)], np.r_[EI, EJ])), shape=(NE, ncF))
    sDMC = csc_matrix((np.r_[np.ones(NE), -np.ones(NE)], (np.r_[np.arange(NE), np.arange(NE)], np.r_[EI, EJ])), shape=(NE, ncC))
    sBLF = csc_matrix((dataF, (xF, yF)), shape=(ncF, ncF))
    sBLC = csc_matrix((dataC, (xC, yC)), shape=(ncC, ncC))

    return sDMF, sDMC, sBLF, sBLC, sBLC[ncC:,:ncC]