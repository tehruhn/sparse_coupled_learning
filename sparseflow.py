# Code for computing learning in sparse flow networks
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu
from numpy.random import choice, randn

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

def SparseIncidenceConstraintMatrix(SourceNodes, SourceEdges, TargetNodes, TargetEdges, GroundNodes, NN, EI, EJ):
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
    dF, xF, yF, dC, xC, yC = [], [], [], [], [], []
    nc = NN
    nc2 = NN

    node_groups = [(GroundNodes, True), (SourceNodes, True), (TargetNodes, False)]
    edge_groups = [(SourceEdges, True), (TargetEdges, False)]

    for nodes, include_in_f in node_groups:
        for node in nodes:
            dF.extend([1., 1.]) if include_in_f else None
            xF.extend([node, nc]) if include_in_f else None
            yF.extend([nc, node]) if include_in_f else None
            dC.extend([1., 1.])
            xC.extend([node, nc2])
            yC.extend([nc2, node])
            nc += 1 if include_in_f else 0
            nc2 += 1

    for edges, include_in_f in edge_groups:
        for edge in edges:
            d_vals = [1., 1., -1., -1.]
            x_vals = [EI[edge], nc, EJ[edge], nc]
            y_vals = [nc, EI[edge], nc, EJ[edge]]

            dF.extend(d_vals) if include_in_f else None
            xF.extend(x_vals) if include_in_f else None
            yF.extend(y_vals) if include_in_f else None
            dC.extend(d_vals)
            xC.extend(x_vals)
            yC.extend(y_vals)
            nc += 1 if include_in_f else 0
            nc2 += 1

    # Construct matrices
    sDMF = csc_matrix((np.r_[np.ones(NE),-np.ones(NE)], (np.r_[np.arange(NE),np.arange(NE)], np.r_[EI,EJ])), shape=(NE, nc))
    sDMC = csc_matrix((np.r_[np.ones(NE),-np.ones(NE)], (np.r_[np.arange(NE),np.arange(NE)], np.r_[EI,EJ])), shape=(NE, nc2))
    sBLF = csc_matrix((dF, (xF, yF)), shape=(nc, nc))
    sBLC = csc_matrix((dC, (xC, yC)), shape=(nc2, nc2))

    # Matrix for cost computation
    sDot = sBLC[nc2:, :nc2]

    return sDMF, sDMC, sBLF, sBLC, sDot



def learning_process(a, b, periodic, trials, nTaskTypes, nTasksPerType, eta, lr, Steps, sources, targets, sourceedges, targetedges):
    # Create square grid graph
    NN, NE, EI, EJ = create_square_grid_graph(a, b, periodic)
    GroundNodes = np.array([NN - 1])

    # Initialize variables for learning process
    costs = np.zeros(trials)
    conductances = np.ones(NE)  # Initial conductance values

    for trial in range(trials):
        # Prepare tasks and constraints
        TaskTypes = []
        for _ in range(nTaskTypes):
            # Random selection of source and target nodes and edges
            NodeList = choice(range(NN - 1), size=sources + targets, replace=False)
            EdgeList = choice(range(NE), size=sourceedges + targetedges, replace=False)
            
            # Create sparse incidence and constraint matrices
            sDMF, sDMC, sBLF, sBLC, sDot = SparseIncidenceConstraintMatrix(
                NodeList[:sources], EdgeList[:sourceedges],
                NodeList[sources:], EdgeList[sourceedges:],
                GroundNodes, NN, EI, EJ
            )
            TaskTypes.append([sDMF, sDMC, sBLF, sBLC, sDot])

        # Learning iterations
        for step in range(Steps):
            # Perform learning step
            cost, conductances = learning_step(
                TaskTypes, nTaskTypes, nTasksPerType, conductances, eta, lr
            )

            # Record cost for this trial
            if step == Steps - 1:
                costs[trial] = cost

    return costs, conductances

def learning_step(TaskTypes, nTaskTypes, nTasksPerType, conductances, eta, lr):
    # Randomly select a task type and task number
    task_type = choice(range(nTaskTypes))
    task_num = choice(range(nTasksPerType))

    # Retrieve the matrices for the selected task
    sDMF, sDMC, sBLF, sBLC, sDot = TaskTypes[task_type]

    # Solve for free and clamped states
    sK = diags(conductances)
    AFinv = splu(sBLF + sDMF.T @ sK @ sDMF)
    ACinv = splu(sBLC + sDMC.T @ sK @ sDMC)

    # Compute free and clamped states
    PF = AFinv.solve(sDMF)  # Free state
    PC = ACinv.solve(sDMC)  # Clamped state

    # Update conductances based on learning rule
    delta_conductance = compute_delta_conductance(PF, PC, eta, lr)
    conductances = np.clip(conductances - delta_conductance, 1e-6, 1e4)

    # Compute cost
    cost = compute_cost(sDot, PF, PC)

    return cost, conductances

def compute_delta_conductance(PF, PC, eta, lr):
    # Compute delta conductance based on learning rule
    delta_PF = PF[1] - PF[0]
    delta_PC = PC[1] - PC[0]
    return lr * (delta_PC**2 - delta_PF**2) / eta

def compute_cost(sDot, PF, PC):
    # Compute cost as the difference between free and clamped states
    return np.sum((sDot @ PF - sDot @ PC)**2)
