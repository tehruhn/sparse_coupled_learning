# Code for computing learning in sparse flow networks
import numpy as np

from numpy.random import choice, randn, randint
from scipy.sparse import csc_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, splu, minres
from numpy.linalg import norm


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
    dF, xF, yF, dC, xC, yC = [], [], [], [], [], []
    nc = NN
    nc2 = NN

    groups = [(GroundNodes, True, "n"), (SourceNodes, True, "n"), (SourceEdges, True, "e"), 
              (TargetNodes, False, "n") , (TargetEdges, False, "e")]

    for groupname, include_in_f, grouptype in groups:
        if grouptype == "n":
            for node in groupname:
                dF.extend([1., 1.]) if include_in_f else None
                xF.extend([node, nc]) if include_in_f else None
                yF.extend([nc, node]) if include_in_f else None
                dC.extend([1., 1.])
                xC.extend([node, nc2])
                yC.extend([nc2, node])
                nc += 1 if include_in_f else 0
                nc2 += 1
        else:
            for edge in groupname:
                d_vals = [1., 1., -1., -1.]
                x_vals = [EI[edge], nc, EJ[edge], nc]
                y_vals = [nc, EI[edge], nc, EJ[edge]]
                x2_vals = [EI[edge], nc2, EJ[edge], nc2]
                y2_vals = [nc2, EI[edge], nc2, EJ[edge]]

                dF.extend(d_vals) if include_in_f else None
                xF.extend(x_vals) if include_in_f else None
                yF.extend(y_vals) if include_in_f else None
                dC.extend(d_vals)
                xC.extend(x2_vals)
                yC.extend(y2_vals)
                nc += 1 if include_in_f else 0
                nc2 += 1

    # Construct matrices
    sDMF = csc_matrix((np.r_[np.ones(NE),-np.ones(NE)], (np.r_[np.arange(NE),np.arange(NE)], np.r_[EI,EJ])), shape=(NE, nc))
    sDMC = csc_matrix((np.r_[np.ones(NE),-np.ones(NE)], (np.r_[np.arange(NE),np.arange(NE)], np.r_[EI,EJ])), shape=(NE, nc2))
    sBLF = csc_matrix((dF, (xF, yF)), shape=(nc, nc))
    sBLC = csc_matrix((dC, (xC, yC)), shape=(nc2, nc2))

    # Matrix for cost computation
    sDot = sBLC[nc:, :nc]
    return sDMF, sDMC, sBLF, sBLC, sDot


def initialize_task_types(nTaskTypes, nTasksPerType, sources, targets, sourceedges, targetedges, NN, NE, EI, EJ):
    """
    Initialize different task types for the learning process.

    Parameters:
    nTaskTypes (int): Number of different task types.
    nTasksPerType (int): Number of tasks per type.
    sources (int): Number of source nodes.
    targets (int): Number of target nodes.
    sourceedges (int): Number of source edges.
    targetedges (int): Number of target edges.
    NN (int): Total number of nodes in the graph.
    NE (int): Total number of edges in the graph.
    EI (list): List of start nodes for each edge.
    EJ (list): List of end nodes for each edge.

    Returns:
    tuple: A tuple containing task types and shapes of the matrices.
    """
    task_types = []
    for _ in range(nTaskTypes):
        # Randomly choose nodes and edges for each task
        node_list = choice(range(NN-1), size=sources+targets, replace=False)
        edge_list = choice(range(NE), size=sourceedges+targetedges, replace=False)

        # Create sparse incidence and constraint matrices
        matrices = create_sparse_incidence_constraint_matrix(node_list[:sources], 
                                                   edge_list[:sourceedges], 
                                                   node_list[sources:], 
                                                   edge_list[sourceedges:], 
                                                   [NN-1], NN, EI, EJ)
        
        # Create constraints for each task type
        constraints = create_constraints(nTasksPerType, matrices[0].shape[1], 
                                         matrices[1].shape[1], 
                                         sources, targets, sourceedges, targetedges, NN)
        task_types.append(matrices + constraints)

    return task_types, matrices[0].shape[1], matrices[1].shape[1]


def create_constraints(nTasksPerType, fshape, cshape, sources, targets, sourceedges, targetedges, NN):
    """
    Create constraints for the tasks.

    Parameters:
    nTasksPerType (int): Number of tasks per type.
    fshape (int): Shape of the free state matrix.
    cshape (int): Shape of the clamped state matrix.
    sources (int): Number of source nodes.
    targets (int): Number of target nodes.
    sourceedges (int): Number of source edges.
    targetedges (int): Number of target edges.
    NN (int): Total number of nodes.

    Returns:
    tuple: A tuple containing constraints for free state, clamped state, and desired outcomes.
    """
    ff = np.zeros([nTasksPerType, fshape])
    fc = np.zeros([nTasksPerType, cshape])
    desired = np.zeros([nTasksPerType, cshape-fshape])

    for i in range(nTasksPerType):
        # Generate random data for node and edge constraints
        node_data = randn(sources)
        out_node_data = randn(targets) * 0.3
        edge_data = randn(sourceedges)
        out_edge_data = randn(targetedges) * 0.3

        # Set constraints for free and clamped states
        ff[i, NN:] = np.r_[0., node_data, edge_data] 
        fc[i, NN:] = np.r_[0., node_data, edge_data, out_node_data, out_edge_data]
        desired[i] = np.r_[out_node_data, out_edge_data]

    return (ff, fc, desired)


def cost_computation(task_type, K, sK, eta, lr):
    """
    Compute the cost for a single task type.

    Parameters:
    task_type (tuple): Contains matrices and constraints for a specific task type.
    K (array): Array of conductance values.
    sK (csc_matrix): Sparse diagonal matrix of conductance values.
    eta (float): Small parameter for nudging in the learning rule.
    lr (float): Learning rate.

    Returns:
    tuple: A tuple containing the free state (PF), clamped state (PC), and the cost.
    """
    # Unpack the task components
    sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, desired = task_type

    # Compute inverse matrix and solve for the free state
    AFinv = splu(sBLF + sDMF.T * sK * sDMF)
    PF = AFinv.solve(ff.T)

    # Calculate the sum value for the cost
    sumval = np.sum((sDot.dot(PF) - desired.T)**2)

    # Compute inverse matrix and solve for the clamped state
    ACinv = splu(sBLC + sDMC.T * sK * sDMC)
    PC = ACinv.solve(fc.T)

    return PF, PC, sumval


def state_solve(task_type, K, sK, eta, lr, tt, tn, x0, threshold, state_type="f"):
    """
    Solves for the state (either free or clamped) of a sparse flow network.

    Args:
    task_type (tuple): Contains matrices and vectors defining a task.
    K (np.array): Array of conductances for each edge.
    sK (scipy.sparse): Diagonal sparse matrix of conductances.
    eta (float): Learning rate parameter for adjustments.
    lr (float): Learning rate for conductance updates.
    tt (int): Index of the task type.
    tn (int): Index of the task number.
    x0 (np.array): Initial guess for iterative solving.
    threshold (float): Threshold to decide between direct and iterative solving.
    state_type (str): Specifies the type of state to solve for ('f' for free, 'c' for clamped).

    Returns:
    tuple: Tuple containing the solved pressures (PF) for free state or (PC) for clamped state.
    """
    sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, desired = task_type
    PF = PC = None

    if threshold > 1.e-50:
        # Direct solving approach
        if state_type == "f":
            AFinv = splu(sBLF + sDMF.T * sK * sDMF)
            PF = AFinv.solve(ff[tn])
        elif state_type == "c":
            ACinv = splu(sBLC + sDMC.T * sK * sDMC)
            PC = ACinv.solve(fc[tn])
    else:
        # Iterative solving approach
        if state_type == "f":
            PF = minres(sBLF + sDMF.T * sK * sDMF, ff[tn], tol=1.e-10, x0=x0)[0]
        elif state_type == "c":
            PC = minres(sBLC + sDMC.T * sK * sDMC, fc[tn], tol=1.e-10, x0=x0)[0]

    return PF, PC

def compute_cost_for_task(nTaskTypes, nTasksPerType, fshape, cshape,task_types, K, sK, eta, lr):
    """
    Compute the initial costs, free state and clamped state for each task type.

    Parameters:
    nTaskTypes (int): The number of task types.
    nTasksPerType (int): The number of tasks per type.
    fshape (int): The shape of the free state matrix.
    cshape (int): The shape of the clamped state matrix.
    task_types (list): A list of task types, where each type is a tuple of task-specific matrices and constraints.
    K (np.array): An array of conductance values for the edges.
    sK (csc_matrix): A sparse diagonal matrix with conductance values on the diagonal.
    eta (float): A small positive value used in the nudging process for the clamped state.
    lr (float): Learning rate for updating conductances.

    Returns:
    tuple: A tuple containing PFs (free states), PCs (clamped states), and CEq0 (initial total cost).
    """
    CEq0 = 0.
    PFs = np.zeros([nTaskTypes, fshape, nTasksPerType])
    PCs = np.zeros([nTaskTypes, cshape, nTasksPerType])
    # Iterate over each task type to compute free and clamped states, and accumulate costs
    for i in range(nTaskTypes):
        # Compute cost and states for each task type
        PF, PC, sumval = cost_computation(task_types[i], K, sK, eta, lr)
        
        # Store the computed states
        PFs[i] = PF
        PCs[i] = PC

        # Accumulate the total cost
        CEq0 += sumval

    # Average the total cost over all task types and tasks per type
    CEq0 /= nTaskTypes * nTasksPerType

    return PFs, PCs, CEq0

def update_conductances(K, lr, eta, sDMF, sDMC, PF, PC):
    """
    Update the conductances in the network.

    Parameters:
    K (np.array): Current conductances.
    lr (float): Learning rate.
    eta (float): Nudging parameter.
    sDMF, sDMC (csc_matrix): Sparse matrices for free and clamped states.
    PF, PC (np.array): Pressures in the free and clamped states.

    Returns:
    K, DK (np.array): Updated conductances.
    """
    DPF = sDMF * PF
    PPF = DPF**2
    DPC = sDMC * PC
    PPC = DPC**2
    DKL = 0.5 * (PPC - PPF) / eta
    K2 = K - lr * DKL
    K2 = K2.clip(1.e-6, 1.e4)
    return K2, K2-K


def perform_trials(nTaskTypes=2, nTasksPerType=1, eta=1.e-3, lr=3.0, Steps=40001, 
                   a=100, b=100, sources=5, targets=3, sourceedges=1, targetedges=1, trials=1):
    """
    Perform a series of trials to learn in a flow network.

    Parameters:
    nTaskTypes (int): Number of different task types.
    nTasksPerType (int): Number of tasks per task type.
    eta (float): Small positive value used in the nudging process for the clamped state.
    lr (float): Learning rate for updating conductances.
    Steps (int): Number of learning steps per trial.
    a, b (int): Dimensions of the square grid.
    sources, targets, sourceedges, targetedges (int): Number of sources, targets, source edges, and target edges.
    trials (int): Number of trials to perform.

    This function initializes a square grid graph and performs learning trials.
    In each trial, it computes the initial and ongoing costs of tasks and updates the conductances.
    """
    NN, NE, EI, EJ = create_square_grid_graph(a, b, False)
    CompSteps = np.unique(np.around(np.r_[0, np.logspace(0, np.log10(Steps-1), 430)])).astype(int)

    for t in range(trials):
        task_types, fshape, cshape = initialize_task_types(nTaskTypes, nTasksPerType, sources, targets, 
                                               sourceedges, targetedges, NN, NE, EI, EJ)
        K = np.ones(NE)
        sK = spdiags(K, 0, NE, NE, format='csc')
        PFs, PCs, CEq0 = compute_cost_for_task(nTaskTypes, nTasksPerType, fshape, cshape,
                                               task_types, K, sK, eta, lr)

        CEq = CEq0
        print(f"Trial {t}: Step 0, Initial Cost {CEq0:.4f}")

        # Iterate over training steps
        for steps in range(1, Steps + 1):
            # Choose a specific task
            tt = randint(0, nTaskTypes)
            tn = randint(0, nTasksPerType)
            sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, desired = task_types[tt]

            # Solve for free and clamped states

            # Free state
            PF, _ = state_solve(task_types[tt], K, sK, eta, lr, tt, tn, 
                                PFs[tt,:,tn], CEq/CEq0, state_type="f")
            
            # Clamped state
            FST = sDot.dot(PF)
            Nudge = FST + eta * (desired[tn] - FST)
            fc[tn,sDMF.shape[1]:] = Nudge
            _, PC = state_solve(task_types[tt], K, sK, eta, lr, tt, tn, 
                                PCs[tt,:,tn], CEq/CEq0, state_type="c")

            PFs[tt,:,tn], PCs[tt,:,tn] = PF, PC

            # Update conductances
            K, DK = update_conductances(K, lr, eta, sDMF, sDMC, PF, PC)
            sK = spdiags(K, 0, NE, NE, format='csc')

            # Print cost and norm of conductance change every 100 steps
            if steps % 100 == 0:
                _, _, CEq = compute_cost_for_task(nTaskTypes, nTasksPerType, fshape, cshape,
                                                  task_types, K, sK, eta, lr)
                print(f"Trial {t}: Step {steps}, Relative Cost {CEq/CEq0:.8f}, Norm of Conductance Change {norm(DK):.8f}")

            # Record costs at specific computation steps
            if steps in CompSteps:
                _, _, CEq = compute_cost_for_task(nTaskTypes, nTasksPerType, fshape, cshape,
                                                  task_types, K, sK, eta, lr)

    return K

