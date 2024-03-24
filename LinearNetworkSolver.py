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

    def create_sparse_incidence_constraint_matrices(self):
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

    # def create_constraints_for_problem()



if __name__ == "__main__":

    linNet = LinearNetwork("./Net1.pkl")
    g = linNet.to_networkx_graph()
    print(g)
    solver = LinearNetworkSolver(linNet)
    
    solver.source_nodes = np.array([3, 8], dtype=int)
    solver.target_nodes = np.array([4, 5], dtype=int)
    solver.ground_nodes = np.array([2], dtype=int)

    # print(solver.ground_nodes)
    solver.create_sparse_incidence_constraint_matrices()
    # print(solver.sDMC)
    tri, tro, ti, to = generate_regression_data()
    print(tri)        

