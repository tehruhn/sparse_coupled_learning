import numpy as np
from typing import List
from numpy import array, ndarray
from numpy.random import choice, randn, randint
from scipy.sparse import csc_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, splu, minres
from numpy.linalg import norm
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import networkx

from task_utils import *

from LinearNetwork import LinearNetwork, LinearNetworkError

class LinearNetworkSolver:
    """
    A class for taking linear networks and solving them for a particular task.
    """

    def __init__(self, network: LinearNetwork) -> None:
        """
        Initializes the LinearNetworkSolver with a given linear network.

        Args:
            network (LinearNetwork): The linear network to solve.
        """
        self._network = network

        self._source_nodes = np.array([], dtype=int)
        self._source_edges = np.array([], dtype=int)
        self._target_nodes = np.array([], dtype=int)
        self._target_edges = np.array([], dtype=int)
        self._ground_nodes = np.array([], dtype=int)

        self.sDMF = None 
        self.sDMC = None
        self.sBLF = None 
        self.sBLC = None 
        self.sDot = None

        self.ntasks = 0

    @property
    def source_nodes(self) -> np.ndarray:
        """Gets the source nodes."""
        return self._source_nodes

    @source_nodes.setter
    def source_nodes(self, value: np.ndarray) -> None:
        """Sets the source nodes."""
        self._source_nodes = value

    @property
    def source_edges(self) -> np.ndarray:
        """Gets the source edges."""
        return self._source_edges

    @source_edges.setter
    def source_edges(self, value: np.ndarray) -> None:
        """Sets the source edges."""
        self._source_edges = value

    @property
    def target_nodes(self) -> np.ndarray:
        """Gets the target nodes."""
        return self._target_nodes

    @target_nodes.setter
    def target_nodes(self, value: np.ndarray) -> None:
        """Sets the target nodes."""
        self._target_nodes = value

    @property
    def target_edges(self) -> np.ndarray:
        """Gets the target edges."""
        return self._target_edges

    @target_edges.setter
    def target_edges(self, value: np.ndarray) -> None:
        """Sets the target edges."""
        self._target_edges = value

    @property
    def ground_nodes(self) -> np.ndarray:
        """Gets the ground nodes."""
        return self._ground_nodes

    @ground_nodes.setter
    def ground_nodes(self, value: np.ndarray) -> None:
        """Sets the ground nodes."""
        self._ground_nodes = value

    def _create_sparse_incidence_constraint_matrices(self) -> None:
        """
        Constructs and assigns sparse incidence and constraint matrices for the linear network
        to class attributes. This method processes the network's nodes and edges to generate matrices
        essential for solving linear network problems, focusing on flow or connectivity.
        
        Raises:
            ValueError: If the network has not been properly initialized.
            TypeError: If node or edge arrays are not of expected numpy ndarray type.
        """
        if not self._network.is_initialized:
            raise ValueError("The LinearNetwork instance has not been initialized.")

        dF, xF, yF, dC, xC, yC = [], [], [], [], [], []
        nc = self._network.NN
        nc2 = self._network.NN

        groups = [
            (self._ground_nodes, True, "n"), 
            (self._source_nodes, True, "n"), 
            (self._source_edges, True, "e"), 
            (self._target_nodes, False, "n") , 
            (self._target_edges, False, "e")
        ]

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
                    x_vals = [self._network.EI[edge], nc, self._network.EJ[edge], nc]
                    y_vals = [nc, self._network.EI[edge], nc, self._network.EJ[edge]]
                    x2_vals = [self._network.EI[edge], nc2, self._network.EJ[edge], nc2]
                    y2_vals = [nc2, self._network.EI[edge], nc2, self._network.EJ[edge]]

                    dF.extend(d_vals) if include_in_f else None
                    xF.extend(x_vals) if include_in_f else None
                    yF.extend(y_vals) if include_in_f else None
                    dC.extend(d_vals)
                    xC.extend(x2_vals)
                    yC.extend(y2_vals)
                    nc += 1 if include_in_f else 0
                    nc2 += 1

        # Construct matrices
        sDMF = csc_matrix(
            (np.r_[np.ones(self._network.NE),-np.ones(self._network.NE)], 
            (np.r_[np.arange(self._network.NE),np.arange(self._network.NE)], 
            np.r_[self._network.EI, self._network.EJ])), 
            shape=(self._network.NE, nc)
        )

        sDMC = csc_matrix(
            (np.r_[np.ones(self._network.NE),-np.ones(self._network.NE)], 
            (np.r_[np.arange(self._network.NE),np.arange(self._network.NE)], 
            np.r_[self._network.EI, self._network.EJ])), 
            shape=(self._network.NE, nc2)
        )
        
        sBLF = csc_matrix((dF, (xF, yF)), shape=(nc, nc))
        sBLC = csc_matrix((dC, (xC, yC)), shape=(nc2, nc2))

        # Matrix for cost computation
        sDot = sBLC[nc:, :nc]

        # Assign matrices to local attributes
        self.sDMF = sDMF 
        self.sDMC = sDMC
        self.sBLF = sBLF 
        self.sBLC = sBLC 
        self.sDot = sDot


    def _create_constraints(
        self, 
        in_node_data: Optional[np.ndarray] = None, 
        out_node_data: Optional[np.ndarray] = None, 
        in_edge_data: Optional[np.ndarray] = None, 
        out_edge_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Constructs and assigns constraints matrices for the network tasks, including constraints for 
        free and clamped states, and desired outcomes for optimization tasks.

        Parameters:
        - in_node_data (np.ndarray, optional): Input data for nodes, for each task.
        - out_node_data (np.ndarray, optional): Output (desired) data for nodes, for each task.
        - in_edge_data (np.ndarray, optional): Input data for edges, for each task.
        - out_edge_data (np.ndarray, optional): Output (desired) data for edges, for each task.
        
        Raises:
        - ValueError: If the input data arrays are not numpy arrays or have incompatible shapes.
        """
        # Default to zeros if None
        in_node_data = np.zeros((0,)) if in_node_data is None else in_node_data
        out_node_data = np.zeros((0,)) if out_node_data is None else out_node_data
        in_edge_data = np.zeros((0,)) if in_edge_data is None else in_edge_data
        out_edge_data = np.zeros((0,)) if out_edge_data is None else out_edge_data

        self.ntasks = in_node_data.shape[0] if in_node_data.ndim > 0 else 0
        fshape, cshape = self.sDMF.shape[1], self.sDMC.shape[1]
        ff = np.zeros([self.ntasks, fshape])
        fc = np.zeros([self.ntasks, cshape])
        desired = np.zeros([self.ntasks, cshape - fshape])

        for i in range(self.ntasks):
            # Data for node and edge constraints
            i_n_data = in_node_data[i] if i < len(in_node_data) else np.zeros(0)
            o_n_data = out_node_data[i] if i < len(out_node_data) else np.zeros(0)
            i_e_data = in_edge_data[i] if i < len(in_edge_data) else np.zeros(0)
            o_e_data = out_edge_data[i] if i < len(out_edge_data) else np.zeros(0)

            # Update constraints for free and clamped states
            ff[i, self._network.NN:] = np.r_[0., i_n_data, i_e_data]
            fc[i, self._network.NN:] = np.r_[0., i_n_data, i_e_data, o_n_data, o_e_data]
            desired[i] = np.r_[o_n_data, o_e_data]

        # Assign the constructed constraints to class attributes
        self.ff, self.fc, self.desired = ff, fc, desired

    
   
    def set_up_training_task(
        self, 
        source_nodes: Optional[List[int]] = None, 
        target_nodes: Optional[List[int]] = None, 
        source_edges: Optional[List[int]] = None, 
        target_edges: Optional[List[int]] = None, 
        ground_nodes: Optional[List[int]] = None, 
        in_node: Optional[np.ndarray] = None, 
        out_node: Optional[np.ndarray] = None, 
        in_edge: Optional[np.ndarray] = None, 
        out_edge: Optional[np.ndarray] = None
    ) -> None:
        """
        Sets up a training task for the network solver by configuring source, target, and ground nodes,
        along with the input and output data for nodes and edges.
        
        Args:
        - source_nodes, target_nodes, ground_nodes: Optional lists of node indices for sources, targets, and grounds.
        - source_edges, target_edges: Optional lists of edge indices for source and target edges.
        - in_node, out_node: Optional numpy arrays for input and output node data.
        - in_edge, out_edge: Optional numpy arrays for input and output edge data.

        Raises:
        - ValueError: If the network has not been initialized or if input data types are incorrect.
        """
        if self._network is None:
            raise ValueError("Initialize network first")

        # Default to empty lists if None
        self._source_nodes = np.array(source_nodes if source_nodes is not None else [], dtype=int)
        self._target_nodes = np.array(target_nodes if target_nodes is not None else [], dtype=int)
        self._source_edges = np.array(source_edges if source_edges is not None else [], dtype=int)
        self._target_edges = np.array(target_edges if target_edges is not None else [], dtype=int)
        self._ground_nodes = np.array(ground_nodes if ground_nodes is not None else [], dtype=int)

        # Set up constraint matrix
        self._create_sparse_incidence_constraint_matrices()

        # Include training data
        self._create_constraints(
            in_node_data=in_node, 
            out_node_data=out_node, 
            in_edge_data=in_edge, 
            out_edge_data=out_edge
        )


    
    def cost_computation(self, sK: csc_matrix) -> tuple:
        """
        Computes the cost of the current configuration based on the difference between
        the desired state and the state achieved by applying the current conductance matrix.

        Parameters:
        - sK (csc_matrix): Sparse diagonal matrix of conductance values.

        Returns:
        - tuple: A tuple containing the pressures in the free state (PF), the pressures in the 
                 clamped state (PC), and the sum of squared differences (cost) between the 
                 desired and actual states.

        Raises:
        - ValueError: If the input matrix sK is not of the correct type or shape.
        - Exception: Handles exceptions related to matrix operations and inversion.
        """
        if not isinstance(sK, csc_matrix):
            raise ValueError("sK must be a scipy.sparse.csc_matrix.")
        
        try:
            # Compute inverse matrix and solve for the free state
            AFinv = splu(self.sBLF + self.sDMF.T.dot(sK).dot(self.sDMF))
            PF = AFinv.solve(self.ff.T)

            # Calculate the sum value for the cost
            sumval = np.sum((self.sDot.dot(PF) - self.desired.T)**2)

            # Compute inverse matrix and solve for the clamped state
            ACinv = splu(self.sBLC + self.sDMC.T.dot(sK).dot(self.sDMC))
            PC = ACinv.solve(self.fc.T)

            return PF, PC, sumval
        except Exception as e:
            # Log error or raise custom exception as needed
            raise Exception(f"Failed to compute cost due to: {e}")

    
    def state_solve(
        self, 
        sK: csc_matrix, 
        tn: int, 
        x0: np.ndarray, 
        threshold: float, 
        state_type: str = "f"
    ) -> tuple:
        """
        Solves for the state (either free or clamped) of the network based on the provided conductance matrix.

        Parameters:
        - sK (csc_matrix): The sparse diagonal matrix of conductance values.
        - tn (int): The task number to solve for.
        - x0 (np.ndarray): The initial guess for iterative solving.
        - threshold (float): The threshold to decide between direct and iterative solving methods.
        - state_type (str): The type of state to solve for ('f' for free, 'c' for clamped).

        Returns:
        - tuple: The solved pressures (PF) for free state or (PC) for clamped state.

        Raises:
        - ValueError: If the state_type is invalid or if input types are incorrect.
        """
        PF = PC = None
        if not isinstance(sK, csc_matrix):
            raise ValueError("sK must be a scipy.sparse.csc_matrix.")
        if not isinstance(x0, np.ndarray):
            raise ValueError("x0 must be a numpy.ndarray.")
        if state_type not in ["f", "c"]:
            raise ValueError("state_type must be 'f' for free or 'c' for clamped.")
        
        try:
            if threshold > 1.e-50:
                # Direct solving approach
                if state_type == "f":
                    AFinv = splu(self.sBLF + self.sDMF.T.dot(sK).dot(self.sDMF))
                    PF = AFinv.solve(self.ff[tn])
                elif state_type == "c":
                    ACinv = splu(self.sBLC + self.sDMC.T.dot(sK).dot(self.sDMC))
                    PC = ACinv.solve(self.fc[tn])
            else:
                # Iterative solving approach
                if state_type == "f":
                    PF, _ = minres(self.sBLF + self.sDMF.T.dot(sK).dot(self.sDMF), self.ff[tn], tol=1.e-10, x0=x0)
                elif state_type == "c":
                    PC, _ = minres(self.sBLC + self.sDMC.T.dot(sK).dot(self.sDMC), self.fc[tn], tol=1.e-10, x0=x0)
            
            return PF, PC
        except Exception as e:
            raise Exception(f"Failed to solve state due to: {e}")


    def update_conductances(
        self, 
        K: ndarray, 
        lr: float, 
        eta: float, 
        PF: ndarray, 
        PC: ndarray
    ) -> tuple:
        """
        Updates the conductances in the network based on the pressure differences
        between the free state (PF) and the clamped state (PC).

        Parameters:
        - K (ndarray): The current conductance values.
        - lr (float): Learning rate for the update step.
        - eta (float): Small positive parameter to avoid division by zero and stabilize the update.
        - PF (ndarray): Pressures in the free state.
        - PC (ndarray): Pressures in the clamped state.

        Returns:
        - tuple: A tuple containing the updated conductances (K2) and the difference (DKL) between the new and old conductances.

        Raises:
        - ValueError: If eta is non-positive or if any input array does not match expected dimensions.
        """
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if not all(isinstance(arr, np.ndarray) for arr in [K, PF, PC]):
            raise ValueError("K, PF, and PC must be numpy arrays.")
        
        try:
            DPF = self.sDMF.dot(PF)
            PPF = DPF ** 2
            DPC = self.sDMC.dot(PC)
            PPC = DPC ** 2
            DKL = 0.5 * (PPC - PPF) / eta
            K2 = K - lr * DKL
            K2 = np.clip(K2, 1.e-6, 1.e4)  # Ensure conductances remain within specified bounds
            
            return K2, K2 - K
        except Exception as e:
            raise Exception(f"Failed to update conductances due to: {e}")
        
    
    def compute_cost_for_task(self, sK: csc_matrix) -> tuple:
        """
        Computes the cost and updates the states for the network based on a given
        sparse conductance matrix (sK).

        Parameters:
        - sK (csc_matrix): Sparse diagonal matrix of conductance values.

        Returns:
        - tuple: A tuple containing arrays for pressures in free and clamped states
                 for all tasks, along with the averaged total cost (CEq0).

        Raises:
        - ValueError: If sK is not a csc_matrix or if ntasks is zero, indicating
                      that no tasks have been set up.
        """
        if not isinstance(sK, csc_matrix):
            raise ValueError("sK must be a scipy.sparse.csc_matrix.")
        if self.ntasks == 0:
            raise ValueError("No tasks have been set up. Please set up tasks before computing cost.")

        try:
            fshape, cshape = self.sDMF.shape[1], self.sDMC.shape[1]
            ntasks = self.ntasks
            CEq0 = 0.
            PFs = np.zeros([1, fshape, ntasks])
            PCs = np.zeros([1, cshape, ntasks])

            # Compute cost and states for each task type
            PF, PC, sumval = self.cost_computation(sK)
            
            # Store the computed states
            PFs[0] = PF
            PCs[0] = PC

            # Accumulate the total cost
            CEq0 += sumval

            # Average the total cost over all task types and tasks per type
            CEq0 /= ntasks

            return PFs, PCs, CEq0
        except Exception as e:
            raise Exception(f"Failed to compute cost for task due to: {e}")

    def compute_power(self, K, PF) -> float:
        """
        Computes Power for the system
        """
        DP = self.sDMF@PF
        power = np.dot(K, np.square(DP))
        return power
    
    def perform_trial(
        self, 
        source_nodes: Optional[List[int]] = None, 
        target_nodes: Optional[List[int]] = None,
        source_edges: Optional[List[int]] = None, 
        target_edges: Optional[List[int]] = None, 
        ground_nodes: Optional[List[int]] = None, 
        in_node: Optional[np.ndarray] = None, 
        out_node: Optional[np.ndarray] = None, 
        in_edge: Optional[np.ndarray] = None, 
        out_edge: Optional[np.ndarray] = None,
        eta: float = 1.e-3, 
        lr: float = 3.0, 
        steps: int = 40001,
        every_nth: int = 100,
        init_strategy = "random",
        debug = False
    ) -> np.ndarray:
        """
        Executes a single trial of the optimization process, adjusting the network's conductances
        to minimize cost over a series of steps.

        Args:
            source_nodes, target_nodes, source_edges, target_edges, ground_nodes: Configuration of the network.
            in_node, out_node, in_edge, out_edge: Training data for the network.
            eta (float): Small positive parameter for the update equation.
            lr (float): Learning rate for conductance updates.
            steps (int): Number of optimization steps to perform.
            init_strategy (str) : random, ones, decimal

        Returns:
            np.ndarray: The final conductances after optimization.

        Raises:
            ValueError: If the network is not initialized.
        """
        if self._network is None:
            raise ValueError("Initialize network first")

        CompSteps = np.unique(np.around(np.r_[0, np.logspace(0, np.log10(steps - 1), 430)])).astype(int)

        self.set_up_training_task(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            source_edges=source_edges,
            target_edges=target_edges,
            ground_nodes=ground_nodes,
            in_node=in_node,
            out_node=out_node,
            in_edge=in_edge,
            out_edge=out_edge
        )
        if init_strategy == "random":
            K = np.random.uniform(0.5, 1.5, self._network.NE)
        elif init_strategy == "ones":
            K = np.ones(self._network.NE)
        else:
            K = np.ones(self._network.NE)/10.0
        sK = spdiags(K, 0, self._network.NE, self._network.NE, format='csc')

        PFs, PCs, CEq0 = self.compute_cost_for_task(sK)

        CEq = CEq0
        if debug :
            print(f"Step 0, Initial Cost {CEq0:.4f}")

        all_costs = [(0, CEq0)]

        for step in range(1, steps + 1):
            tn = randint(0, self.ntasks)

            PF, _ = self.state_solve(sK, tn, PFs[0,:,tn], CEq / CEq0, state_type="f")
            
            FST = self.sDot.dot(PF)
            Nudge = FST + eta * (self.desired[tn] - FST)
            self.fc[tn, self.sDMF.shape[1]:] = Nudge

            _, PC = self.state_solve(sK, tn, PCs[0,:,tn], CEq / CEq0, state_type="c")

            PFs[0,:,tn], PCs[0,:,tn] = PF, PC

            K, DK = self.update_conductances(K, lr, eta, PF, PC)
            sK = spdiags(K, 0, self._network.NE, self._network.NE, format='csc')

            if step % every_nth == 0:
                _, _, CEq = self.compute_cost_for_task(sK)
                if debug :
                    ratio = "{:e}".format(CEq / CEq0)
                    print(f"Step {step}, Relative Cost {ratio}, Norm of Conductance Change {norm(DK):.8f}, Power {self.compute_power(K, PF)}")
                all_costs.append((step, CEq))

        return K, all_costs



if __name__ == "__main__":

    # make the linear network
    linNet = LinearNetwork("./Net1.pkl")
    g = linNet.to_networkx_graph()
    print(g)
    solver = LinearNetworkSolver(linNet)
    
    # add source, target, ground nodes
    source_nodes = np.array([3, 8], dtype=int)
    target_nodes = np.array([4, 5], dtype=int)
    ground_nodes = np.array([2], dtype=int)

    # generate data
    tri, trt = encode_regression_data_in_correct_format()

    # pass data to the network and train it
    K, costs = solver.perform_trial(source_nodes=source_nodes, 
                            target_nodes=target_nodes,
                            ground_nodes=ground_nodes,
                            in_node=tri,
                            out_node=trt,
                            lr=1.e-1,
                            steps=2000,
                            debug=True
                            )
    print(costs)
    

