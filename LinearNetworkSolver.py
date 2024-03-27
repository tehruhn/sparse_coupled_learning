import numpy as np
from typing import List
from numpy import array
from numpy.random import choice, randn, randint
from scipy.sparse import csc_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, splu, minres
from numpy.linalg import norm
import matplotlib.pyplot as plt

from task_utils import *

from LinearNetwork import LinearNetwork

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

    def _create_sparse_incidence_constraint_matrices(self):
        """
        Constructs and assigns sparse incidence and constraint matrices for the linear network
        to class attributes. This method processes the network's nodes and edges to generate matrices
        that are essential for solving linear network problems, particularly those related to flow
        or connectivity.

        The method assigns the following matrices to class attributes:
        - sDMF: Sparse incidence matrix for free nodes and edges.
        - sDMC: Sparse incidence matrix for clamped nodes and edges.
        - sBLF: Sparse constraint border Laplacian matrix for free nodes and edges.
        - sBLC: Sparse constraint border Laplacian matrix for clamped nodes and edges.
        - sDot: Sparse matrix for cost computation in optimization tasks.

        These matrices are stored in the respective class attributes `sDMF`, `sDMC`, `sBLF`, `sBLC`, 
        and `sDot` for further use.
        """
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
        self.sBLC =sBLC 
        self.sDot = sDot


    def _create_constraints(self, in_node_data=[], out_node_data=[], in_edge_data=[], out_edge_data=[]):
        """
        Add constraints to the data
        """
        self.ntasks, fshape, cshape = in_node_data.shape[0], self.sDMF.shape[1], self.sDMC.shape[1]
        ff = np.zeros([self.ntasks, fshape])
        fc = np.zeros([self.ntasks, cshape])
        desired = np.zeros([self.ntasks, cshape-fshape])

        for i in range(self.ntasks):
            # Data for node and edge constraints
            i_n_data = in_node_data[i] if len(in_node_data) > 0 else np.zeros(0)
            o_n_data = out_node_data[i] if len(out_node_data) > 0 else np.zeros(0)
            i_e_data = in_edge_data[i] if len(in_edge_data) > 0 else np.zeros(0)
            o_e_data = out_edge_data[i] if len(out_edge_data) > 0 else np.zeros(0)

            # Set constraints for free and clamped states
            ff[i, self._network.NN:] = np.r_[0., i_n_data, i_e_data] 
            fc[i, self._network.NN:] = np.r_[0., i_n_data, i_e_data, o_n_data, o_e_data]
            desired[i] = np.r_[o_n_data, o_e_data]

        self.ff, self.fc, self.desired = ff, fc, desired

    
    def set_up_training_task(self, source_nodes=[], target_nodes=[], source_edges=[], target_edges=[], 
                             ground_nodes=[], in_node=[], out_node=[], in_edge=[], out_edge=[]):
        if self._network == None:
            print("Initialize network first")
            return
        self._source_nodes = source_nodes
        self._target_nodes = target_nodes
        self._source_edges = source_edges
        self._target_edges = target_edges
        self._ground_nodes = ground_nodes

        # set up constraint matrix
        self._create_sparse_incidence_constraint_matrices()

        # include training data
        self._create_constraints(in_node_data=in_node, out_node_data=out_node, in_edge_data=in_edge, out_edge_data=out_edge)

    def cost_computation(self, sK):
        # Compute inverse matrix and solve for the free state
        AFinv = splu(self.sBLF + self.sDMF.T * sK * self.sDMF)
        PF = AFinv.solve(self.ff.T)

        # Calculate the sum value for the cost
        sumval = np.sum((self.sDot.dot(PF) - self.desired.T)**2)

        # Compute inverse matrix and solve for the clamped state
        ACinv = splu(self.sBLC + self.sDMC.T * sK * self.sDMC)
        PC = ACinv.solve(self.fc.T)

        return PF, PC, sumval
    
    def state_solve(self, sK, tn, x0, threshold, state_type="f"):
        PF = PC = None
        if threshold > 1.e-50:
            # Direct solving approach
            if state_type == "f":
                AFinv = splu(self.sBLF + self.sDMF.T * sK * self.sDMF)
                PF = AFinv.solve(self.ff[tn])
            elif state_type == "c":
                ACinv = splu(self.sBLC + self.sDMC.T * sK * self.sDMC)
                PC = ACinv.solve(self.fc[tn])
        else:
            # Iterative solving approach
            if state_type == "f":
                PF = minres(self.sBLF + self.sDMF.T * sK * self.sDMF, self.ff[tn], tol=1.e-10, x0=x0)[0]
            elif state_type == "c":
                PC = minres(self.sBLC + self.sDMC.T * sK * self.sDMC, self.fc[tn], tol=1.e-10, x0=x0)[0]

        return PF, PC

    def update_conductances(self, K, lr, eta, PF, PC):
        DPF = self.sDMF * PF
        PPF = DPF**2
        DPC = self.sDMC * PC
        PPC = DPC**2
        DKL = 0.5 * (PPC - PPF) / eta
        K2 = K - lr * DKL
        K2 = K2.clip(1.e-6, 1.e4)
        return K2, K2-K
    
    def compute_cost_for_task(self, sK):
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
    
    def perform_trial(self, source_nodes=[], target_nodes=[], 
                       source_edges=[], target_edges=[], 
                       ground_nodes=[], in_node=[], 
                       out_node=[], in_edge=[], out_edge=[],
                       eta=1.e-3, 
                       lr=3.0, 
                       steps=40001):
        if self._network == None:
            print("Initialize network first")
            return
        CompSteps = np.unique(np.around(np.r_[0, np.logspace(0, np.log10(steps-1), 430)])).astype(int)

        # Run the trial
        self.set_up_training_task(source_nodes=source_nodes,
                                  target_nodes=target_nodes,
                                  source_edges=source_edges,
                                  target_edges=target_edges,
                                  ground_nodes=ground_nodes,
                                  in_node=in_node,
                                  out_node=out_node,
                                  in_edge=in_edge,
                                  out_edge=out_edge)
        K = np.ones(self._network.NE)
        sK = spdiags(K, 0,self._network. NE, self._network.NE, format='csc')
        PFs, PCs, CEq0 = self.compute_cost_for_task(sK)

        CEq = CEq0
        print(f"Step 0, Initial Cost {CEq0:.4f}")

        # Iterate over training steps
        for steps in range(1, steps + 1):
            # Choose a specific task
            tt = 0
            tn = randint(0, self.ntasks)

            # Solve for free and clamped states
            # Free state
            PF, _ = self.state_solve(sK, tn,
                                PFs[tt,:,tn], 
                                CEq/CEq0, 
                                state_type="f")
            
            # Clamped state
            FST = self.sDot.dot(PF)
            Nudge = FST + eta * (self.desired[tn] - FST)
            self.fc[tn,self.sDMF.shape[1]:] = Nudge
            _, PC = self.state_solve(sK, tn, 
                                PCs[tt,:,tn],
                                CEq/CEq0, 
                                state_type="c")

            PFs[tt,:,tn], PCs[tt,:,tn] = PF, PC

            # Update conductances
            K, DK = self.update_conductances(K, lr, eta, PF, PC)
            sK = spdiags(K, 0, self._network.NE, self._network.NE, format='csc')

            # Print cost and norm of conductance change every 100 steps
            if steps % 100 == 0:
                _, _, CEq = self.compute_cost_for_task(sK)
                print(f"Step {steps}, Relative Cost {CEq/CEq0:.8f}, Norm of Conductance Change {norm(DK):.8f}")

            # Record costs at specific computation steps
            if steps in CompSteps:
                _, _, CEq = self.compute_cost_for_task(sK)

        return K




if __name__ == "__main__":

    linNet = LinearNetwork("./Net1.pkl")
    g = linNet.to_networkx_graph()
    print(g)
    solver = LinearNetworkSolver(linNet)
    
    source_nodes = np.array([3, 8], dtype=int)
    target_nodes = np.array([4, 5], dtype=int)
    ground_nodes = np.array([2], dtype=int)

    tri, trt = encode_regression_data_in_correct_format()
    solver.perform_trial(source_nodes=source_nodes, 
                                    target_nodes=target_nodes,
                                    ground_nodes=ground_nodes,
                                    in_node=tri,
                                    out_node=trt)
    

